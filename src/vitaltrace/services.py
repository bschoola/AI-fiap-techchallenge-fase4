"""Montagem (composition root) dos serviços a partir da configuração."""

from dataclasses import dataclass

from vitaltrace.audio.language_service import AzureLanguageAnalyzer
from vitaltrace.audio.mock import MockTextAnalyzer, MockTranscriber
from vitaltrace.audio.pipeline import AudioPipeline
from vitaltrace.audio.speech_service import AzureSpeechTranscriber
from vitaltrace.audio.voice_tone import (
    AzureOpenAIVoiceToneAnalyzer,
    MockVoiceToneAnalyzer,
    VoiceToneAnalyzer,
)
from vitaltrace.config import Settings
from vitaltrace.db import Database
from vitaltrace.video.analyzer import VideoAnalyzer, VideoError, build_video_analyzer


def build_voice_tone(settings: Settings) -> tuple[VoiceToneAnalyzer | None, str | None]:
    """Opcional (VOICE_TONE_ENABLED). Sem configuração, o áudio segue funcionando e avisa o motivo."""
    if not settings.voice_tone_enabled:
        return None, None
    if settings.mock_azure:
        return MockVoiceToneAnalyzer(), None
    missing = settings.missing_voice_tone_settings()
    if missing:
        return None, (
            "Avaliação do tom de voz ligada (VOICE_TONE_ENABLED), mas falta configurar: "
            f"{', '.join(missing)}."
        )
    return (
        AzureOpenAIVoiceToneAnalyzer(
            endpoint=settings.azure_openai_endpoint,
            key=settings.azure_openai_key,
            deployment=settings.azure_openai_audio_deployment,
            api_version=settings.azure_openai_api_version,
        ),
        None,
    )


def build_audio_pipeline(settings: Settings) -> AudioPipeline:
    tone, tone_unavailable = build_voice_tone(settings)
    if settings.mock_azure:
        return AudioPipeline(
            MockTranscriber(settings.speech_language),
            MockTextAnalyzer(),
            settings,
            tone,
            tone_unavailable,
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
        tone,
        tone_unavailable,
    )


@dataclass
class Services:
    settings: Settings
    db: Database
    audio: AudioPipeline | None
    audio_error: str | None
    video: VideoAnalyzer | None
    video_error: str | None


def build_services(settings: Settings) -> Services:
    """Os módulos de áudio e vídeo são opcionais: se faltar configuração, o portal sobe e avisa."""
    db = Database(settings.database_url)
    db.create_all()

    audio, audio_error = None, None
    try:
        audio = build_audio_pipeline(settings)
    except RuntimeError as exc:
        audio_error = str(exc)

    video, video_error = None, None
    try:
        video = build_video_analyzer(settings)
    except (VideoError, ImportError) as exc:
        video_error = str(exc)

    return Services(settings, db, audio, audio_error, video, video_error)
