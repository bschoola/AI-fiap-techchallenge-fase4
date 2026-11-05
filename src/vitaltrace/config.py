"""Configuração da aplicação (variáveis de ambiente / arquivo .env)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Azure AI Speech
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"
    # Azure AI Language (antigo Text Analytics)
    azure_language_key: str = ""
    azure_language_endpoint: str = ""

    # True = usa serviços simulados, sem chamar o Azure (dev/testes)
    mock_azure: bool = False

    speech_language: str = "pt-BR"
    text_language: str = "pt"

    # Postgres no Docker (veja docker-compose.yml); SQLite local serve para dev/testes.
    database_url: str = "sqlite:///data/vitaltrace.db"
    secret_key: str = "troque-esta-chave-em-producao"
    seed_on_startup: bool = True
    hospital_name: str = "Hospital Modelo · Unidade Central"

    max_upload_mb: int = 25
    max_video_mb: int = 200

    # Vídeo: "auto" usa YOLOv8-pose se o ultralytics estiver instalado; senão, OpenCV.
    video_backend: str = "auto"  # auto | yolo | motion
    yolo_pose_weights: str = "yolov8n-pose.pt"
    video_sample_fps: float = 2.0
    video_max_seconds: int = 120

    # Limiares das heurísticas de áudio. São pontos de partida: ajuste com dados reais.
    negative_sentiment_threshold: float = 0.6
    slow_speech_wpm: float = 90.0
    high_pause_ratio: float = 0.5

    def missing_azure_settings(self) -> list[str]:
        required = {
            "AZURE_SPEECH_KEY": self.azure_speech_key,
            "AZURE_SPEECH_REGION": self.azure_speech_region,
            "AZURE_LANGUAGE_KEY": self.azure_language_key,
            "AZURE_LANGUAGE_ENDPOINT": self.azure_language_endpoint,
        }
        return [name for name, value in required.items() if not value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
