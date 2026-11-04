"""Valida as credenciais do Azure (Speech e Language) antes de rodar a aplicação.

Uso: python scripts/azure_smoke_test.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medmonitor.config import Settings  # noqa: E402


def check_language(settings: Settings) -> None:
    from medmonitor.audio.language_service import AzureLanguageAnalyzer

    analyzer = AzureLanguageAnalyzer(
        settings.azure_language_endpoint,
        settings.azure_language_key,
        settings.text_language,
    )
    result = analyzer.analyze("Estou muito cansado e com falta de ar desde ontem.")
    print(f"   sentimento={result.sentiment.label} frases-chave={result.key_phrases}")


def check_speech(settings: Settings) -> None:
    import azure.cognitiveservices.speech as speechsdk

    config = speechsdk.SpeechConfig(
        subscription=settings.azure_speech_key, region=settings.azure_speech_region
    )
    config.speech_synthesis_voice_name = "pt-BR-FranciscaNeural"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "teste.wav"
        audio = speechsdk.audio.AudioOutputConfig(filename=str(out))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio)
        result = synthesizer.speak_text_async("Teste de conexão com o Azure Speech.").get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            raise RuntimeError(f"{details.reason}: {details.error_details}")
        del synthesizer
        print(f"   áudio de teste sintetizado ({out.stat().st_size} bytes)")


def main() -> int:
    settings = Settings()
    missing = settings.missing_azure_settings()
    if missing:
        print(f"Variáveis ausentes no .env: {', '.join(missing)}")
        return 2

    failures = 0
    for name, check in (("Azure AI Language", check_language), ("Azure AI Speech", check_speech)):
        print(f"→ {name}")
        try:
            check(settings)
            print("   OK")
        except Exception as exc:  # noqa: BLE001 — queremos o motivo na tela
            failures += 1
            print(f"   FALHOU: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
