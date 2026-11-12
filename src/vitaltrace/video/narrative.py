"""Descrição da cena por um modelo multimodal (Azure OpenAI com entrada de imagem). OPCIONAL.

Liga com VIDEO_NARRATIVE_ENABLED=true e AZURE_OPENAI_VISION_DEPLOYMENT=<implantação que aceita imagem>.
O modelo recebe alguns quadros do vídeo e descreve o que se vê: ambiente, equipamentos, quem
aparece, o que o paciente está fazendo. As MEDIDAS (ângulos, repetições, eventos) continuam vindo
só da pose; o texto do modelo entra no relatório como descrição e nunca gera alerta.
Se a chamada falhar, a análise segue sem a descrição e registra um aviso.
"""

import base64
import time
from typing import Protocol

import requests

MAX_FRAMES = 6
MAX_CHARS = 900

SYSTEM_PROMPT = (
    "Você descreve, em português do Brasil, quadros de um vídeo de uma sessão de fisioterapia para um "
    "médico que não assistiu à gravação. Diga só o que é visível: ambiente, equipamentos e apoios, quantas "
    "pessoas aparecem e o papel aparente de cada uma, a posição do paciente e o movimento que ele executa "
    "ao longo dos quadros (estão em ordem de tempo). Não identifique ninguém, não comente aparência "
    "física, não faça diagnóstico e não invente números. As medidas automáticas vêm no texto do usuário "
    "apenas como contexto. Responda em um único parágrafo de 3 a 5 frases."
)


class NarrativeError(RuntimeError):
    pass


class SceneNarrator(Protocol):
    def describe(self, frames_jpeg: list[bytes], facts: str) -> str: ...


class AzureOpenAISceneNarrator:
    def __init__(self, endpoint: str, key: str, deployment: str, api_version: str, timeout_s: float = 90.0) -> None:
        self._url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        self._key = key
        self._deployment = deployment
        self._api_version = api_version
        self._timeout_s = timeout_s

    def describe(self, frames_jpeg: list[bytes], facts: str) -> str:
        if not frames_jpeg:
            raise NarrativeError("sem quadros para descrever")
        content: list[dict] = [{"type": "text", "text": f"Medidas automáticas (contexto):\n{facts[:3000]}\n\nQuadros em ordem de tempo:"}]
        for jpeg in frames_jpeg[:MAX_FRAMES]:
            uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
        body = {
            "model": self._deployment,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        }
        resp = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self._url, params={"api-version": self._api_version}, headers={"api-key": self._key},
                    json=body, timeout=self._timeout_s,
                )  # fmt: skip
            except requests.RequestException as exc:
                raise NarrativeError(f"falha de rede ao chamar o Azure OpenAI ({type(exc).__name__})") from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            break
        if resp is None or resp.status_code != 200:
            raise NarrativeError(f"HTTP {resp.status_code if resp is not None else '?'}: {resp.text[:200] if resp is not None else ''}")
        try:
            text = resp.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise NarrativeError("resposta inesperada do Azure OpenAI") from exc
        text = " ".join(str(text).split())
        if not text:
            raise NarrativeError("o modelo não devolveu texto")
        return text[:MAX_CHARS]
