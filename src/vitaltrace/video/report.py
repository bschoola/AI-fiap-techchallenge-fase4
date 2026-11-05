"""Relatório automático da análise de vídeo (texto determinístico em pt-BR)."""

from vitaltrace.models import Severity
from vitaltrace.video.models import VideoEvent, VideoMetrics

_TYPE_LABELS = {
    "fall_suspected": "Queda suspeita",
    "prolonged_immobility": "Imobilidade prolongada",
    "knee_asymmetry": "Assimetria entre os joelhos",
    "low_mobility": "Mobilidade reduzida",
    "no_person": "Pessoa pouco visível",
}
_POSTURES = {"standing": "em pé", "sitting": "sentado", "lying": "deitado", "upright": "na vertical"}


def type_label(event_type: str) -> str:
    return _TYPE_LABELS.get(event_type, event_type)


def build_report(metrics: VideoMetrics, events: list[VideoEvent], backend: str) -> str:
    lines = [
        f"Vídeo de {metrics.duration_s:.0f} s analisado com {backend}: "
        f"{metrics.frames_analyzed} quadros amostrados, pessoa visível em "
        f"{metrics.person_detected_ratio:.0%} deles."
    ]
    if metrics.posture_seconds:
        parts = [f"{_POSTURES.get(k, k)} {v:.0f} s" for k, v in metrics.posture_seconds.items()]
        lines.append("Posturas: " + ", ".join(parts) + ".")
    if metrics.mobility_index is not None:
        lines.append(f"Índice de mobilidade: {metrics.mobility_index:.2f} (0 = parado, 1 = muito ativo).")
    if not events:
        lines.append("Nenhum desvio detectado nesta gravação.")
    else:
        serious = [e for e in events if e.severity.rank >= Severity.HIGH.rank]
        lines.append(
            f"{len(events)} evento(s) detectado(s)"
            + (f", {len(serious)} de severidade alta ou crítica." if serious else ".")
        )
        for e in events:
            lines.append(f"• {type_label(e.type)} ({e.t_start:.0f}–{e.t_end:.0f} s): {e.description}")
    lines.append("Apoio à decisão: revise o trecho do vídeo antes de qualquer conduta.")
    return "\n".join(lines)
