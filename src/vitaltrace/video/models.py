"""Modelos da análise de vídeo."""

from pydantic import BaseModel, Field

from vitaltrace.models import Alert, Severity

Keypoint = tuple[float, float, float]  # x, y, confiança (pixels da imagem analisada)


class PersonObservation(BaseModel):
    t: float  # segundos desde o início do vídeo
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: list[Keypoint] | None = None  # 17 pontos COCO, quando há pose
    frame_size: tuple[int, int] = (0, 0)  # largura, altura
    people: int = 1  # pessoas detectadas no quadro (a mais evidente é a analisada)


class VideoEvent(BaseModel):
    type: str  # fall_suspected | prolonged_immobility | knee_asymmetry | low_mobility | no_person | trunk_lean | shoulder_tilt | hip_tilt | postural_sway | rom_asymmetry | rep_irregular | rep_fatigue
    severity: Severity
    t_start: float
    t_end: float
    description: str


class Keyframe(BaseModel):
    """Quadro do vídeo com a pessoa/pose desenhada; guardado para a linha do tempo do paciente."""

    t: float  # segundos desde o início do vídeo
    label: str
    image: str  # data URI JPEG
    event_index: int | None = None  # posição do evento relacionado em `events` (None = quadro de exemplo)


class VideoMetrics(BaseModel):
    duration_s: float = 0.0
    frames_analyzed: int = 0
    person_detected_ratio: float = 0.0
    mobility_index: float | None = None  # 0 (parado) a 1 (muito ativo)
    immobility_s: float = 0.0
    posture_seconds: dict[str, float] = Field(default_factory=dict)
    median_knee_diff_deg: float | None = None


class VideoQuality(BaseModel):
    """Quão confiável é a análise: o que o vídeo permitiu (ou não) avaliar."""

    depth: str = "fast"  # fast | detailed
    sample_fps: float = 2.0
    cuts: int = 0  # cortes de cena detectados (vídeo editado)
    cut_times: list[float] = Field(default_factory=list)
    keypoint_ratio: float | None = None  # fração de quadros com esqueleto utilizável
    mean_keypoint_conf: float | None = None
    multi_person_ratio: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class VideoAnalysis(BaseModel):
    patient_id: str = ""
    backend: str = ""
    metrics: VideoMetrics = Field(default_factory=VideoMetrics)
    events: list[VideoEvent] = Field(default_factory=list)
    report: str = ""
    keyframes: list[Keyframe] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    quality: VideoQuality = Field(default_factory=VideoQuality)
    joints: dict[str, float] = Field(default_factory=dict)  # amplitude de movimento (graus) por articulação
    curve: dict | None = None  # curva do ângulo da articulação mais ativa + repetições

    def payload(self) -> dict:
        """O que fica salvo no banco. Os quadros-chave (poucos e reduzidos) entram para a linha do tempo."""
        return {
            "backend": self.backend,
            "metrics": self.metrics.model_dump(),
            "events": [e.model_dump(mode="json") for e in self.events],
            "report": self.report,
            "keyframes": [k.model_dump() for k in self.keyframes],
            "quality": self.quality.model_dump(),
            "joints": self.joints,
            "curve": self.curve,
        }
