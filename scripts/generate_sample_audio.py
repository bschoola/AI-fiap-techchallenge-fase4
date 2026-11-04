"""Gera áudios de exemplo SINTÉTICOS (Azure TTS) em data/samples/ para a demonstração.

Não são gravações de pacientes reais. Uso: python scripts/generate_sample_audio.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import azure.cognitiveservices.speech as speechsdk  # noqa: E402

from medmonitor.config import Settings  # noqa: E402

VOICE = "pt-BR-FranciscaNeural"
OUT_DIR = ROOT / "data" / "samples"


def _ssml(body: str) -> str:
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
        f'<voice name="{VOICE}">{body}</voice></speak>'
    )


SAMPLES = {
    # Termos críticos: deve gerar alertas CRÍTICOS.
    "paciente_critico.wav": _ssml(
        "Doutor, estou com muita falta de ar e uma dor no peito desde ontem à noite. "
        "Também me sinto muito cansado e com tontura."
    ),
    # Termos negados e sentimento positivo: não deve gerar alertas.
    "paciente_estavel.wav": _ssml(
        "Bom dia, doutor. Hoje estou me sentindo bem, dormi tranquilo, "
        "não tenho dor e também não tenho falta de ar."
    ),
    # Fala lenta e com muitas pausas: simula fadiga (alertas de ritmo de fala).
    "paciente_fadiga.wav": _ssml(
        '<prosody rate="-35%">Doutor <break time="1800ms"/> eu estou muito <break time="1500ms"/> '
        'cansado e sem forças <break time="2000ms"/> desde ontem <break time="1800ms"/> '
        "não consigo fazer quase nada <break time=\"1500ms\"/> hoje.</prosody>"
    ),
}


def main() -> int:
    settings = Settings()
    missing = [m for m in settings.missing_azure_settings() if m.startswith("AZURE_SPEECH")]
    if missing:
        print(f"Variáveis ausentes no .env: {', '.join(missing)}")
        return 2

    config = speechsdk.SpeechConfig(
        subscription=settings.azure_speech_key, region=settings.azure_speech_region
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, ssml in SAMPLES.items():
        out = OUT_DIR / filename
        audio = speechsdk.audio.AudioOutputConfig(filename=str(out))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio)
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            print(f"FALHOU {filename}: {details.error_details}")
            return 1
        del synthesizer
        print(f"OK {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
