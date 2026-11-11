"""Tom de voz avaliado por um modelo de áudio (Azure OpenAI gpt-audio). OPCIONAL.

Liga com VOICE_TONE_ENABLED=true. O modelo OUVE a gravação e descreve esforço respiratório,
cansaço e tom emocional. É uma segunda opinião sobre a prosódia medida em `prosody.py`:
a resposta varia entre execuções, por isso nunca decide a severidade sozinha (ver alerts/engine.py).
Se a chamada falhar, a análise segue sem o tom de voz e registra um aviso.
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Protocol

import requests

from vitaltrace.audio.models import ProsodyMetrics, VoiceTone

LEVELS = ("nenhum", "leve", "moderado", "intenso")
MAX_AUDIO_BYTES = 20 * 1024 * 1024

SYSTEM_PROMPT = (
    "Você é um assistente de triagem que OUVE a gravação de um paciente numa consulta médica e descreve "
    "apenas COMO a pessoa fala (não o que ela diz). Avalie sinais audíveis: respiração ofegante ou "
    "inspirações audíveis entre as palavras, frases interrompidas para tomar fôlego, voz fraca ou que "
    "some no fim das frases, lentidão, tremor, choro, pressa ou tensão. Não faça diagnóstico. "
    "Responda SOMENTE com um objeto JSON, sem texto antes ou depois, neste formato:\n"
    '{"respiratory_effort": "nenhum|leve|moderado|intenso", '
    '"fatigue": "nenhum|leve|moderado|intenso", '
    '"emotional_tone": "uma ou duas palavras, ex.: calmo, ansioso, assustado, abatido", '
    '"summary": "uma frase curta em português descrevendo o que se ouve"}'
)


class VoiceToneError(RuntimeError):
    pass


class VoiceToneAnalyzer(Protocol):
    def analyze(
        self, wav_path: Path, transcript: str, prosody: ProsodyMetrics | None
    ) -> VoiceTone: ...


def _level(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in LEVELS else "nenhum"


def parse_voice_tone(content: str, source: str) -> VoiceTone:
    """Extrai o JSON da resposta (tolera cercas de código e texto em volta)."""
    match = re.search(r"\{.*\}", content or "", flags=re.DOTALL)
    if not match:
        raise VoiceToneError("o modelo não devolveu JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise VoiceToneError("o modelo devolveu um JSON inválido") from exc
    if not isinstance(data, dict):
        raise VoiceToneError("o modelo devolveu um JSON inesperado")
    return VoiceTone(
        respiratory_effort=_level(data.get("respiratory_effort")),
        fatigue=_level(data.get("fatigue")),
        emotional_tone=str(data.get("emotional_tone") or "").strip()[:60],
        summary=str(data.get("summary") or "").strip()[:400],
        source=source,
    )


class AzureOpenAIVoiceToneAnalyzer:
    """Chat completions com ENTRADA de áudio (gpt-audio, gpt-audio-mini, gpt-4o-audio-preview)."""

    def __init__(
        self, endpoint: str, key: str, deployment: str, api_version: str, timeout_s: float = 120.0
    ) -> None:
        self._url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        self._key = key
        self._deployment = deployment
        self._api_version = api_version
        self._timeout_s = timeout_s

    def analyze(
        self, wav_path: Path, transcript: str, prosody: ProsodyMetrics | None
    ) -> VoiceTone:
        data = wav_path.read_bytes()
        if len(data) > MAX_AUDIO_BYTES:
            raise VoiceToneError("áudio grande demais para a avaliação de tom de voz")
        body = {
            "model": self._deployment,
            "modalities": ["text"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Gravação do paciente:"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(data).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                },
            ],
        }
        resp = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self._url,
                    params={"api-version": self._api_version},
                    headers={"api-key": self._key},
                    json=body,
                    timeout=self._timeout_s,
                )
            except requests.RequestException as exc:
                raise VoiceToneError(f"falha de rede ao chamar o Azure OpenAI ({type(exc).__name__})") from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            break
        if resp is None or resp.status_code != 200:
            detail = resp.text[:200] if resp is not None else ""
            code = resp.status_code if resp is not None else "?"
            raise VoiceToneError(f"HTTP {code}: {detail}")
        try:
            content = resp.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise VoiceToneError("resposta inesperada do Azure OpenAI") from exc
        return parse_voice_tone(content, self._deployment)


class MockVoiceToneAnalyzer:
    """Sem Azure: deriva a impressão da prosódia medida (serve só para validar o fluxo)."""

    def analyze(
        self, wav_path: Path, transcript: str, prosody: ProsodyMetrics | None
    ) -> VoiceTone:
        pause = (prosody.pause_ratio if prosody and prosody.reliable else None) or 0.0
        if pause >= 0.5:
            effort, tone = "intenso", "abatido"
        elif pause >= 0.4:
            effort, tone = "moderado", "cansado"
        else:
            effort, tone = "nenhum", "calmo"
        return VoiceTone(
            respiratory_effort=effort,
            fatigue=effort,
            emotional_tone=tone,
            summary="Avaliação simulada a partir das pausas medidas (modo MOCK_AZURE).",
            source="mock",
        )
