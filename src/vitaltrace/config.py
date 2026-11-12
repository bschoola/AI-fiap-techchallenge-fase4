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

    # Azure OpenAI (opcional): gerador de áudios de demonstração e avaliação do tom de voz
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_audio_deployment: str = "gpt-audio"  # chat completions com saída de áudio (recomendado)
    azure_openai_tts_deployment: str = "gpt-4o-mini-tts"  # /audio/speech (gpt-4o-mini-tts, tts-hd, tts)
    azure_openai_api_version: str = "2025-04-01-preview"

    # True = um modelo de áudio (AZURE_OPENAI_AUDIO_DEPLOYMENT) ouve a gravação e descreve o tom de voz
    voice_tone_enabled: bool = False

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

    # Vídeo: "auto" usa YOLO-pose se o ultralytics estiver instalado; senão, OpenCV.
    video_backend: str = "auto"  # auto | yolo | motion
    # Pesos da pose. Se não carregarem (ultralytics antigo, sem internet), tenta yolo11s e yolov8n.
    # Em máquina lenta, troque por yolo26n-pose.pt: mais rápido, menos preciso nas articulações.
    yolo_pose_weights: str = "yolo26s-pose.pt"
    video_sample_fps: float = 2.0
    video_max_seconds: int = 120
    video_detail_fps: float = 8.0  # amostragem do modo "Detalhada"
    video_analysis_width: int = 960  # largura (px) do quadro analisado; o original maior é reduzido a isso
    video_imgsz: int = 640  # resolução de entrada do modelo no quadro inteiro
    # 2º estágio: refaz a pose só no recorte do paciente, tirado do vídeo em resolução original
    video_refine_pose: bool = True
    video_refine_imgsz: int = 480
    # Guarda o vídeo com o esqueleto desenhado (reduzido, na taxa de amostragem) junto da análise
    video_annotated: bool = True
    video_annotated_dir: str = "data/video_annotated"
    # Descrição da cena por modelo multimodal do Azure OpenAI (opcional; exige implantação com imagem)
    video_narrative_enabled: bool = False
    azure_openai_vision_deployment: str = ""
    # Link do YouTube na análise de vídeo (precisa do yt-dlp e, para o YouTube, do runtime Deno)
    youtube_enabled: bool = True
    youtube_max_height: int = 720  # resolução máxima baixada do YouTube

    # Limiares das heurísticas de áudio. São pontos de partida: ajuste com dados reais.
    negative_sentiment_threshold: float = 0.6
    slow_speech_wpm: float = 90.0
    high_pause_ratio: float = 0.5
    # Fala entrecortada (padrão de quem está sem fôlego): TODOS os critérios precisam ser atendidos.
    # Calibrado nos áudios de data/samples: a consulta normal fica em 29% de pausa, mediana 0,5 s
    # e nenhuma pausa longa; a de falta de ar, em 54%, 1,1 s e 17 pausas longas por minuto.
    breathless_pause_ratio: float = 0.45
    breathless_median_pause_s: float = 0.8
    breathless_long_pauses_per_min: float = 6.0
    breathless_max_phrase_s: float = 2.0
    monotone_pitch_st: float = 3.0  # variação de tom abaixo disso = fala monótona (só informativo)

    def missing_voice_tone_settings(self) -> list[str]:
        required = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_KEY": self.azure_openai_key,
            "AZURE_OPENAI_AUDIO_DEPLOYMENT": self.azure_openai_audio_deployment,
        }
        return [name for name, value in required.items() if not value.strip()]

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
