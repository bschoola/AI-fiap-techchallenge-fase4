"""Relatório automático da análise de vídeo (texto determinístico em pt-BR)."""

from vitaltrace.models import Severity
from vitaltrace.video.models import VideoEvent, VideoMetrics

_TYPE_LABELS = {
    "fall_suspected": "Queda suspeita",
    "prolonged_immobility": "Imobilidade prolongada",
    "knee_asymmetry": "Assimetria entre os joelhos",
    "low_mobility": "Mobilidade reduzida",
    "no_person": "Pessoa pouco visível",
    "trunk_lean": "Tronco inclinado",
    "shoulder_tilt": "Ombros desalinhados",
    "hip_tilt": "Quadril desalinhado",
    "postural_sway": "Oscilação postural",
    "rom_asymmetry": "Assimetria de amplitude",
    "rep_irregular": "Ritmo irregular das repetições",
    "rep_fatigue": "Queda de amplitude (fadiga)",
    "low_pose_quality": "Pose pouco confiável",
}
_POSTURES = {"standing": "em pé", "sitting": "sentado", "lying": "deitado", "upright": "na vertical"}


def type_label(event_type: str) -> str:
    return _TYPE_LABELS.get(event_type, event_type)


_CHOICE = {"manual": "indicada pelo médico", "auto": "escolhida automaticamente", "single": "única em cena"}


def people_line(people: dict | None) -> str | None:
    if not people or not people.get("max"):
        return None
    if people["max"] <= 1:
        return "Pessoas em cena: 1 (o paciente)."
    line = f"Pessoas em cena: até {people['max']} ao mesmo tempo"
    if people.get("analyzed") is not None:
        line += f"; analisada a pessoa nº {people['analyzed']} ({_CHOICE.get(people.get('choice'), 'escolhida automaticamente')})"
    return line + ". As demais foram ignoradas nas medidas."


def build_report(
    metrics: VideoMetrics,
    events: list[VideoEvent],
    backend: str,
    quality=None,
    joints=None,
    people: dict | None = None,
    scene: list[dict] | None = None,
    ranges: dict[str, dict] | None = None,
    narrative: str = "",
) -> str:
    lines = [
        f"Vídeo de {metrics.duration_s:.0f} s analisado com {backend}: "
        f"{metrics.frames_analyzed} quadros amostrados, paciente visível em "
        f"{metrics.person_detected_ratio:.0%} deles."
    ]
    who = people_line(people)
    if who:
        lines.append(who)
    if scene:
        lines.append("O que foi visto:")
        for s in scene:
            lines.append(f"• {s['t_start']:.0f}–{s['t_end']:.0f} s: {s['text']}")
    if narrative:
        lines.append(f"Descrição da cena (modelo multimodal, não usada nas medidas): {narrative}")
    if metrics.posture_seconds:
        parts = [f"{_POSTURES.get(k, k)} {v:.0f} s" for k, v in metrics.posture_seconds.items()]
        lines.append("Posturas: " + ", ".join(parts) + ".")
    if metrics.mobility_index is not None:
        lines.append(f"Índice de mobilidade: {metrics.mobility_index:.2f} (0 = parado, 1 = muito ativo).")
    if ranges:
        parts = [f"{k} {v['min']}°–{v['max']}° (amplitude {v['rom']}°)" for k, v in ranges.items()]
        lines.append("Ângulos medidos (2D): " + "; ".join(parts) + ".")
    elif joints:
        lines.append("Amplitude de movimento: " + ", ".join(f"{k} {v}°" for k, v in joints.items()) + ".")
    if quality is not None:
        if quality.depth == "detailed":
            lines.append(f"Análise detalhada ({quality.sample_fps:g} quadros/s; {quality.cuts} corte(s) de cena).")
        for w in quality.warnings:
            lines.append(f"Atenção: {w}")
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
