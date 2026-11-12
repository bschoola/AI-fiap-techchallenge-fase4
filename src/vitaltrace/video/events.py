"""Eventos a partir da linha do tempo de observações (queda, imobilidade, assimetria, mobilidade)."""

import numpy as np

from vitaltrace.models import Severity
from vitaltrace.video import pose
from vitaltrace.video.models import PersonObservation, VideoEvent, VideoMetrics

FALL_WINDOW_S = 2.0  # de pé → deitado em até 2 s
LYING_AFTER_FALL_S = 3.0  # e permanece deitado
CRITICAL_LYING_S = 5.0
LOW_MOTION = 0.03  # alturas-do-corpo por segundo
IMMOBILE_MIN_S = 10.0
MOBILITY_REFERENCE = 0.5  # movimento (alturas/s) que equivale a índice 1,0
KNEE_DIFF_LIMIT_DEG = 15.0
LOW_MOBILITY_INDEX = 0.2
MOTION_DT_S = 0.45  # velocidade medida em janelas de ≥ 0,45 s: com mais quadros/s o tremor da pose não vira "movimento"


def _detect_falls(
    obs: list[PersonObservation], postures: list[str], end_t: float, segs: list[int]
) -> list[VideoEvent]:
    events: list[VideoEvent] = []
    skip_until = -1.0
    for i in range(1, len(obs)):
        if obs[i].t <= skip_until or postures[i] != "lying":
            continue
        recent_upright = any(
            pose.is_upright(postures[j]) for j in range(i) if obs[i].t - obs[j].t <= FALL_WINDOW_S and segs[j] == segs[i]  # um corte de cena não é queda
        )
        if not recent_upright:
            continue
        after = [k for k in range(i, len(obs)) if obs[k].t - obs[i].t <= LYING_AFTER_FALL_S]
        lying_ratio = sum(postures[k] == "lying" for k in after) / len(after)
        if lying_ratio < 0.8:
            continue  # levantou logo em seguida (agachou, abaixou-se)
        lying_until = obs[i].t
        for k in range(i, len(obs)):
            if postures[k] != "lying":
                break
            lying_until = obs[k].t
        lying_s = lying_until - obs[i].t
        severity = Severity.CRITICAL if lying_s >= CRITICAL_LYING_S else Severity.HIGH
        events.append(
            VideoEvent(
                type="fall_suspected",
                severity=severity,
                t_start=round(obs[i].t, 1),
                t_end=round(lying_until, 1),
                description=(
                    f"Passou da posição vertical para deitado em menos de {FALL_WINDOW_S:.0f} s "
                    f"e permaneceu assim por cerca de {lying_s:.0f} s."
                ),
            )
        )
        skip_until = lying_until
    return events


def _detect_immobility(
    obs: list[PersonObservation], motions: list[float], falls: list[VideoEvent]
) -> tuple[list[VideoEvent], float]:
    events: list[VideoEvent] = []
    total = 0.0
    run_start: float | None = None
    run_end = 0.0

    def close() -> None:
        nonlocal total, run_start
        if run_start is None:
            return
        length = run_end - run_start
        total += length
        overlaps_fall = any(f.t_start - 1 <= run_end and run_start <= f.t_end + 1 for f in falls)
        if length >= IMMOBILE_MIN_S and not overlaps_fall:
            events.append(
                VideoEvent(
                    type="prolonged_immobility",
                    severity=Severity.HIGH if length >= 2 * IMMOBILE_MIN_S else Severity.MEDIUM,
                    t_start=round(run_start, 1),
                    t_end=round(run_end, 1),
                    description=f"Quase sem movimento por {length:.0f} s seguidos.",
                )
            )
        run_start = None

    for k in range(1, len(obs)):
        if motions[k - 1] < LOW_MOTION:
            if run_start is None:
                run_start = obs[k - 1].t
            run_end = obs[k].t
        else:
            close()
    close()
    return events, total


def _motions(detected: list[PersonObservation], segs: list[int]) -> list[float]:
    """Velocidade de cada observação (a partir da 2ª) contra uma anterior ≥ MOTION_DT_S antes; NaN entre cortes."""
    out, j = [], 0
    for k in range(1, len(detected)):
        while j + 1 < k and detected[k].t - detected[j + 1].t >= MOTION_DT_S:
            j += 1
        ref = j if detected[k].t - detected[j].t >= MOTION_DT_S and segs[j] == segs[k] else k - 1
        out.append(pose.motion_per_second(detected[ref], detected[k]) if segs[ref] == segs[k] else float("nan"))
    return out


def analyze_observations(
    observations: list[PersonObservation | None],
    times: list[float],
    duration_s: float,
    segments: list[int] | None = None,
) -> tuple[VideoMetrics, list[VideoEvent]]:
    """`observations` tem um item por quadro amostrado (None = ninguém detectado).

    `segments` (opcional) dá o trecho contínuo de cada quadro: cortes de cena separam trechos, e nada
    (velocidade, queda) é calculado de um trecho para outro."""
    segments = segments or [0] * len(observations)
    detected = [o for o in observations if o is not None]
    metrics = VideoMetrics(
        duration_s=round(duration_s, 1),
        frames_analyzed=len(observations),
        person_detected_ratio=round(len(detected) / len(observations), 2) if observations else 0.0,
    )
    events: list[VideoEvent] = []
    if len(detected) < 2:
        if observations:
            events.append(
                VideoEvent(
                    type="no_person",
                    severity=Severity.LOW,
                    t_start=0.0,
                    t_end=round(duration_s, 1),
                    description="Pessoa quase não aparece no vídeo; a análise não é conclusiva.",
                )
            )
        return metrics, events

    segs = [sg for o, sg in zip(observations, segments) if o is not None]
    postures = [pose.classify_posture(o) for o in detected]
    motions = _motions(detected, segs)

    seconds: dict[str, float] = {}
    for (a, b), posture, sa, sb in zip(zip(detected, detected[1:]), postures, segs, segs[1:]):
        if sa != sb:
            continue
        seconds[posture] = seconds.get(posture, 0.0) + (b.t - a.t)
    metrics.posture_seconds = {k: round(v, 1) for k, v in seconds.items()}
    metrics.mobility_index = round(float(min(1.0, np.nanmedian(motions) / MOBILITY_REFERENCE if np.isfinite(motions).any() else 0.0)), 2)

    falls = _detect_falls(detected, postures, detected[-1].t, segs)
    immobility, immobile_s = _detect_immobility(detected, motions, falls)
    metrics.immobility_s = round(immobile_s, 1)
    events += falls + immobility

    diffs = []
    for o, posture in zip(detected, postures):
        if posture == "lying" or pose.view(o) == "lateral":
            continue  # de perfil um joelho encobre o outro: comparar os lados não faz sentido
        angles = pose.knee_angles(o)
        if angles:
            diffs.append(abs(angles[0] - angles[1]))
    if len(diffs) >= 5:
        median_diff = float(np.median(diffs))
        metrics.median_knee_diff_deg = round(median_diff, 1)
        if median_diff > KNEE_DIFF_LIMIT_DEG:
            events.append(
                VideoEvent(
                    type="knee_asymmetry",
                    severity=Severity.MEDIUM,
                    t_start=round(detected[0].t, 1),
                    t_end=round(detected[-1].t, 1),
                    description=(
                        f"Diferença mediana de {median_diff:.0f}° entre os ângulos dos joelhos "
                        f"(limite {KNEE_DIFF_LIMIT_DEG:.0f}°): possível compensação de carga."
                    ),
                )
            )

    if metrics.mobility_index is not None and metrics.mobility_index < LOW_MOBILITY_INDEX and not falls:
        events.append(
            VideoEvent(
                type="low_mobility",
                severity=Severity.LOW,
                t_start=round(detected[0].t, 1),
                t_end=round(detected[-1].t, 1),
                description=f"Índice de mobilidade baixo ({metrics.mobility_index:.2f}).",
            )
        )
    events.sort(key=lambda e: e.t_start)
    return metrics, events
