"""Modelos da análise de vídeo."""

from pydantic import BaseModel, Field

from vitaltrace.models import Alert, Severity

Keypoint = tuple[float, float, float]  # x, y, confiança (pixels da imagem analisada)


class PersonObservation(BaseModel):
    t: float  # segundos desde o início do vídeo
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: list[Keypoint] | None = None  # 17 pontos COCO, quando há pose
    frame_size: tuple[int, int] = (0, 0)  # largura, altura


class VideoEvent(BaseModel):
    type: str  # fall_suspected | prolonged_immobility | knee_asymmetry | low_mobility | no_person
    severity: Severity
    t_start: float
    t_end: float
    description: str


class VideoMetrics(BaseModel):
    duration_s: float = 0.0
    frames_analyzed: int = 0
    person_detected_ratio: float = 0.0
    mobility_index: float | None = None  # 0 (parado) a 1 (muito ativo)
    immobility_s: float = 0.0
    posture_seconds: dict[str, float] = Field(default_factory=dict)
    median_knee_diff_deg: float | None = None


class VideoAnalysis(BaseModel):
    patient_id: str = ""
    backend: str = ""
    metrics: VideoMetrics = Field(default_factory=VideoMetrics)
    events: list[VideoEvent] = Field(default_factory=list)
    report: str = ""
    keyframes: list[str] = Field(default_factory=list)  # data URIs JPEG com a pose desenhada
    alerts: list[Alert] = Field(default_factory=list)

    def payload(self) -> dict:
        """O que fica salvo no banco (sem as imagens, que são pesadas)."""
        return {
            "backend": self.backend,
            "metrics": self.metrics.model_dump(),
            "events": [e.model_dump(mode="json") for e in self.events],
            "report": self.report,
        }
