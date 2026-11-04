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

    database_path: str = "data/alerts.db"
    max_upload_mb: int = 25

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
