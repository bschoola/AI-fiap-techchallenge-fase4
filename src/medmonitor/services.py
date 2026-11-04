"""Montagem (composition root) dos serviços a partir da configuração."""

from medmonitor.audio.language_service import AzureLanguageAnalyzer
from medmonitor.audio.mock import MockTextAnalyzer, MockTranscriber
from medmonitor.audio.pipeline import AudioPipeline
from medmonitor.audio.speech_service import AzureSpeechTranscriber
from medmonitor.config import Settings


def build_audio_pipeline(settings: Settings) -> AudioPipeline:
    if settings.mock_azure:
        return AudioPipeline(
            MockTranscriber(settings.speech_language), MockTextAnalyzer(), settings
        )

    missing = settings.missing_azure_settings()
    if missing:
        raise RuntimeError(
            "Configuração do Azure incompleta (faltando: "
            f"{', '.join(missing)}). Preencha o arquivo .env "
            "(veja docs/azure-setup.md) ou use MOCK_AZURE=true para desenvolver sem Azure."
        )

    return AudioPipeline(
        AzureSpeechTranscriber(
            key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            language=settings.speech_language,
        ),
        AzureLanguageAnalyzer(
            endpoint=settings.azure_language_endpoint,
            key=settings.azure_language_key,
            language=settings.text_language,
        ),
        settings,
    )
