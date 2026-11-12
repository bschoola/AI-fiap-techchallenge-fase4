"""Análise detalhada de um único vídeo: sem histórico do paciente, só o que aparece na gravação.

Procura, dentro do próprio vídeo, sinais de desvio: tronco inclinado por muito tempo, ombros/quadril
desalinhados, oscilação em pé, assimetria de amplitude entre os lados (joelho, quadril, ombro e cotovelo)
e, em exercícios repetitivos,
ritmo irregular e queda de amplitude (fadiga). Limiares conservadores, pensados para evitar alarme falso:
são ângulos 2D de uma câmera só, então os achados são apoio à revisão humana, não diagnóstico.
"""

from dataclasses import dataclass, field

import numpy as np

from vitaltrace.models import Severity
from vitaltrace.video import pose
from vitaltrace.video.models import PersonObservation, VideoEvent

LEAN_LIMIT_DEG = 20.0
LEAN_MEDIUM_DEG = 30.0
LEAN_MIN_S = 5.0
TILT_LIMIT_DEG = 8.0
TILT_MEDIUM_DEG = 14.0
TILT_MIN_FRAMES = 8
SWAY_MOTION = 0.15  # "parado em pé": abaixo disso (alturas do corpo por segundo)
SWAY_MIN_S = 6.0
SWAY_LIMIT = 0.10  # desvio-padrão lateral do quadril / comprimento do tronco
SWAY_MEDIUM = 0.20
ROM_MIN_DEG = 25.0  # amplitude mínima para considerar que há um movimento repetitivo
ROM_ASYM_RATIO = 0.65
ROM_ASYM_MEDIUM = 0.5
ROM_MIN_FRAMES = 12
MIN_REPS_RHYTHM = 4
MIN_REPS_FATIGUE = 6
RHYTHM_CV_LIMIT = 0.35
RHYTHM_CV_MEDIUM = 0.6
FATIGUE_DROP = 0.25
FATIGUE_DROP_MEDIUM = 0.30
MAX_CURVE_POINTS = 400

JOINTS = {
    "joelho esquerdo": (pose.L_HIP, pose.L_KNEE, pose.L_ANKLE),
    "joelho direito": (pose.R_HIP, pose.R_KNEE, pose.R_ANKLE),
    "quadril esquerdo": (pose.L_SHOULDER, pose.L_HIP, pose.L_KNEE),
    "quadril direito": (pose.R_SHOULDER, pose.R_HIP, pose.R_KNEE),
    "ombro esquerdo": (pose.L_ELBOW, pose.L_SHOULDER, pose.L_HIP),
    "ombro direito": (pose.R_ELBOW, pose.R_SHOULDER, pose.R_HIP),
    "cotovelo esquerdo": (pose.L_SHOULDER, pose.L_ELBOW, pose.L_WRIST),
    "cotovelo direito": (pose.R_SHOULDER, pose.R_ELBOW, pose.R_WRIST),
}
_PAIRS = [
    ("joelho esquerdo", "joelho direito"),
    ("quadril esquerdo", "quadril direito"),
    ("ombro esquerdo", "ombro direito"),
    ("cotovelo esquerdo", "cotovelo direito"),
]


@dataclass
class DetailResult:
    events: list[VideoEvent] = field(default_factory=list)
    joints: dict[str, float] = field(default_factory=dict)
    curve: dict | None = None
    ranges: dict[str, dict] = field(default_factory=dict)  # mínimo, máximo e amplitude por articulação
    series: dict[str, np.ndarray] = field(default_factory=dict)  # ângulo por quadro (NaN = não medido)
    postures: list[str] = field(default_factory=list)
    views: list[str | None] = field(default_factory=list)


def smooth_keypoints(
    observations: list[PersonObservation | None], segments: list[int], window: int = 3
) -> list[PersonObservation | None]:
    """Mediana móvel das coordenadas dentro de cada trecho: tira o tremor da pose sem borrar movimentos.

    Cada ponto só usa os vizinhos em que ele próprio é confiável (um ponto chutado não contamina
    os bons), e uma falha de um único quadro entre dois quadros bons é preenchida pela média."""
    out = list(observations)
    half = window // 2
    n = len(observations)

    def kp_at(j: int, k: int):
        if not (0 <= j < n):
            return None
        o = observations[j]
        if o is None or not o.keypoints or k >= len(o.keypoints) or o.keypoints[k][2] < pose.MIN_CONF:
            return None
        return o.keypoints[k]

    for i, o in enumerate(observations):
        if o is None or not o.keypoints:
            continue
        new, changed = list(o.keypoints), False
        for k, (x, y, c) in enumerate(o.keypoints):
            if c >= pose.MIN_CONF:
                near = [kp_at(j, k) for j in range(i - half, i + half + 1) if 0 <= j < n and segments[j] == segments[i]]
                near = [p for p in near if p is not None]
                if len(near) >= 2:
                    new[k] = (float(np.median([p[0] for p in near])), float(np.median([p[1] for p in near])), c)
                    changed = True
            else:
                before = kp_at(i - 1, k) if i > 0 and segments[i - 1] == segments[i] else None
                after = kp_at(i + 1, k) if i + 1 < n and segments[i + 1] == segments[i] else None
                if before is not None and after is not None:
                    new[k] = ((before[0] + after[0]) / 2, (before[1] + after[1]) / 2, 0.9 * min(before[2], after[2]))
                    changed = True
        if changed:
            out[i] = o.model_copy(update={"keypoints": new})
    return out


def _runs(values: list[float | None], segments: list[int], times: list[float], predicate, min_s: float):
    """Trechos consecutivos (mesmo segmento, sem falhas) em que `predicate(valor)` vale por ≥ min_s."""
    runs, start = [], None
    for i in range(len(values) + 1):
        ok = i < len(values) and values[i] is not None and predicate(values[i])
        if ok and start is not None and segments[i] != segments[i - 1]:
            ok_run = (start, i - 1)
            if times[i - 1] - times[start] >= min_s:
                runs.append(ok_run)
            start = i
            continue
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if times[i - 1] - times[start] >= min_s:
                runs.append((start, i - 1))
            start = None
    return runs


def _severity(value: float, medium: float) -> Severity:
    return Severity.MEDIUM if value >= medium else Severity.LOW


def _trunk_lean(obs, postures, segments, times) -> list[VideoEvent]:
    lean = [
        pose.trunk_lean_deg(o) if o is not None and postures[i] in ("standing", "sitting") else None
        for i, o in enumerate(obs)
    ]
    events = []
    for a, b in _runs(lean, segments, times, lambda v: v > LEAN_LIMIT_DEG, LEAN_MIN_S)[:2]:
        median = float(np.median([v for v in lean[a : b + 1]]))
        events.append(
            VideoEvent(
                type="trunk_lean",
                severity=_severity(median, LEAN_MEDIUM_DEG),
                t_start=round(times[a], 1),
                t_end=round(times[b], 1),
                description=(
                    f"Tronco inclinado cerca de {median:.0f}° (limite {LEAN_LIMIT_DEG:.0f}°) por "
                    f"{times[b] - times[a]:.0f} s seguidos. Se não faz parte do exercício, pode indicar "
                    "postura de proteção ou perda de controle do tronco."
                ),
            )
        )
    return events


def _tilts(obs, postures, times, views) -> list[VideoEvent]:
    events = []
    for etype, label, (left, right) in (
        ("shoulder_tilt", "ombros", (pose.L_SHOULDER, pose.R_SHOULDER)),
        ("hip_tilt", "quadril", (pose.L_HIP, pose.R_HIP)),
    ):
        values, stamps = [], []
        for i, o in enumerate(obs):
            if o is None or postures[i] != "standing" or views[i] == "lateral":
                continue  # de perfil a linha dos ombros/quadril não mede desnível
            tilt = pose.line_tilt_deg(o, left, right)
            if tilt is not None:
                values.append(tilt)
                stamps.append(times[i])
        if len(values) < TILT_MIN_FRAMES:
            continue
        median = float(np.median(values))
        if median > TILT_LIMIT_DEG:
            events.append(
                VideoEvent(
                    type=etype,
                    severity=_severity(median, TILT_MEDIUM_DEG),
                    t_start=round(stamps[0], 1),
                    t_end=round(stamps[-1], 1),
                    description=(
                        f"Linha dos {label} inclinada {median:.0f}° em relação à horizontal, em média "
                        f"(limite {TILT_LIMIT_DEG:.0f}°). Confirme que a câmera está nivelada."
                    ),
                )
            )
    return events


def _sway(obs, postures, segments, times) -> list[VideoEvent]:
    flags: list[float | None] = [None] * len(obs)
    for i in range(1, len(obs)):
        a, b = obs[i - 1], obs[i]
        if a and b and segments[i] == segments[i - 1] and postures[i] == "standing":
            flags[i] = pose.motion_per_second(a, b)
    best = None
    for a, b in _runs(flags, segments, times, lambda m: m < SWAY_MOTION, SWAY_MIN_S):
        xs, torsos = [], []
        for o in obs[a : b + 1]:
            c, t = pose.hip_center(o), pose.torso_length(o)
            if c is not None and t is not None:
                xs.append(c[0])
                torsos.append(t)
        if len(xs) < 8:
            continue
        sway = float(np.std(xs) / np.median(torsos))
        if best is None or sway > best[0]:
            best = (sway, a, b)
    if best and best[0] > SWAY_LIMIT:
        sway, a, b = best
        return [
            VideoEvent(
                type="postural_sway",
                severity=_severity(sway, SWAY_MEDIUM),
                t_start=round(times[a], 1),
                t_end=round(times[b], 1),
                description=(
                    f"Oscilação lateral do quadril de {sway:.2f} do comprimento do tronco enquanto "
                    f"parado em pé por {times[b] - times[a]:.0f} s: possível instabilidade de equilíbrio."
                ),
            )
        ]
    return []


def _joint_series(obs, postures) -> dict[str, np.ndarray]:
    series = {}
    for name, triple in JOINTS.items():
        arr = np.full(len(obs), np.nan)
        for i, o in enumerate(obs):
            if o is not None and o.keypoints and postures[i] != "lying":
                angle = pose.joint_angle(o, *triple)
                if angle is not None:
                    arr[i] = angle
        series[name] = arr
    return series


def _rom(values: np.ndarray) -> float | None:
    valid = values[~np.isnan(values)]
    if len(valid) < ROM_MIN_FRAMES:
        return None
    return float(np.percentile(valid, 95) - np.percentile(valid, 5))


def _extremes(t: np.ndarray, v: np.ndarray) -> list[tuple[float, float, str]] | None:
    """Vales e picos alternados do ângulo (zigue-zague: só vale a oscilação ≥ delta, contra ruído).

    O delta é pequeno em relação à amplitude total, para que repetições que vão ficando menores
    (fadiga) continuem sendo contadas em vez de sumirem."""
    if len(v) >= 3:
        v = np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    rng = float(np.percentile(v, 95) - np.percentile(v, 5))
    if rng < ROM_MIN_DEG:
        return None
    delta = max(8.0, 0.12 * rng)
    hi = lo = (float(t[0]), float(v[0]))
    direction = 0
    out: list[tuple[float, float, str]] = []
    for ti, vi in zip(t, v):
        point = (float(ti), float(vi))
        if point[1] > hi[1]:
            hi = point
        if point[1] < lo[1]:
            lo = point
        if direction <= 0 and point[1] - lo[1] >= delta:
            out.append((lo[0], lo[1], "min"))
            direction, hi = 1, point
        elif direction >= 0 and hi[1] - point[1] >= delta:
            out.append((hi[0], hi[1], "max"))
            direction, lo = -1, point
    if direction == 1:
        out.append((hi[0], hi[1], "max"))
    elif direction == -1:
        out.append((lo[0], lo[1], "min"))
    return out


def _repetitions(series: dict[str, np.ndarray], times: list[float], segments: list[int]):
    """Escolhe a articulação/trecho com mais movimento e analisa ritmo e amplitude das repetições."""
    seg_arr = np.array(segments)
    t_arr = np.array(times)
    best = None
    for name, values in series.items():
        for sid in sorted(set(segments)):
            mask = (seg_arr == sid) & ~np.isnan(values)
            if mask.sum() < 8:
                continue
            rng = float(np.percentile(values[mask], 95) - np.percentile(values[mask], 5))
            if best is None or rng > best[0]:
                best = (rng, name, t_arr[mask], values[mask])
    if best is None or best[0] < 15:
        return None, []
    _, name, t, v = best
    step = max(1, len(t) // MAX_CURVE_POINTS)
    curve = {"joint": name, "points": [[round(float(a), 1), round(float(b))] for a, b in zip(t[::step], v[::step])],
             "reps": [], "n_reps": 0}  # fmt: skip
    ext = _extremes(t, v)
    if not ext:
        return curve, []
    valleys = [e for e in ext if e[2] == "min"]
    curve["reps"] = [round(e[0], 1) for e in valleys]
    curve["n_reps"] = len(valleys)
    events: list[VideoEvent] = []
    durations = np.diff([e[0] for e in valleys])
    if len(valleys) >= MIN_REPS_RHYTHM and len(durations) >= 3 and durations.mean() > 0:
        cv = float(durations.std() / durations.mean())
        curve["cv"] = round(cv, 2)
        if cv > RHYTHM_CV_LIMIT:
            events.append(
                VideoEvent(
                    type="rep_irregular",
                    severity=_severity(cv, RHYTHM_CV_MEDIUM),
                    t_start=round(valleys[0][0], 1),
                    t_end=round(valleys[-1][0], 1),
                    description=(
                        f"Ritmo irregular nas {len(valleys)} repetições de {name}: duração de cada repetição "
                        f"varia {cv:.0%} (limite {RHYTHM_CV_LIMIT:.0%}); pode indicar dor, hesitação ou perda de controle."
                    ),
                )
            )
    amps = [abs(ext[i + 1][1] - ext[i][1]) for i in range(len(ext) - 1)]
    if len(valleys) >= MIN_REPS_FATIGUE and len(amps) >= MIN_REPS_FATIGUE:
        third = max(2, len(amps) // 3)
        first, last = float(np.mean(amps[:third])), float(np.mean(amps[-third:]))
        drop = (first - last) / first if first > 0 else 0.0
        curve["amp_drop"] = round(drop, 2)
        if drop >= FATIGUE_DROP:
            events.append(
                VideoEvent(
                    type="rep_fatigue",
                    severity=_severity(drop, FATIGUE_DROP_MEDIUM),
                    t_start=round(ext[len(ext) // 2][0], 1),
                    t_end=round(ext[-1][0], 1),
                    description=(
                        f"A amplitude de {name} caiu {drop:.0%} do início ({first:.0f}°) ao fim ({last:.0f}°) "
                        "das repetições: possível fadiga ou limitação progressiva."
                    ),
                )
            )
    return curve, events


def analyze_detail(
    observations: list[PersonObservation | None], times: list[float], segments: list[int]
) -> DetailResult:
    """`observations` já suavizadas; um item por quadro amostrado (None = ninguém detectado)."""
    result = DetailResult()
    if sum(o is not None for o in observations) < 5:
        return result
    postures = [pose.classify_posture(o) if o is not None else "" for o in observations]
    views = [pose.view(o) if o is not None else None for o in observations]
    result.postures, result.views = postures, views

    result.events += _trunk_lean(observations, postures, segments, times)
    result.events += _tilts(observations, postures, times, views)
    result.events += _sway(observations, postures, segments, times)

    series = _joint_series(observations, postures)
    result.series = series
    roms = {name: _rom(values) for name, values in series.items()}
    result.joints = {name: round(r) for name, r in roms.items() if r is not None}
    for name, values in series.items():
        valid = values[~np.isnan(values)]
        if roms[name] is not None:
            lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
            result.ranges[name] = {"min": round(lo), "max": round(hi), "rom": round(hi - lo), "frames": int(len(valid))}

    # Simetria entre os lados só vale quando os DOIS lados estão à vista: de perfil o lado de trás
    # fica encoberto e a "assimetria" seria só oclusão.
    both_sides = np.array([v in ("frontal", "oblique") for v in views])
    for left, right in _PAIRS:
        seen = both_sides & ~np.isnan(series[left]) & ~np.isnan(series[right])
        a, b = _rom(series[left][seen]), _rom(series[right][seen])
        if a is None or b is None or max(a, b) < ROM_MIN_DEG:
            continue
        ratio = min(a, b) / max(a, b)
        if ratio < ROM_ASYM_RATIO:
            weak, strong = (left, right) if a < b else (right, left)
            valid_t = [times[i] for i in np.flatnonzero(seen)]
            result.events.append(
                VideoEvent(
                    type="rom_asymmetry",
                    severity=Severity.MEDIUM if ratio < ROM_ASYM_MEDIUM else Severity.LOW,
                    t_start=round(min(valid_t), 1),
                    t_end=round(max(valid_t), 1),
                    description=(
                        f"Amplitude de {weak} ({min(a, b):.0f}°) é {1 - ratio:.0%} menor que a de {strong} "
                        f"({max(a, b):.0f}°): possível limitação ou proteção de um dos lados."
                    ),
                )
            )
    result.curve, rep_events = _repetitions(series, times, segments)
    result.events += rep_events
    return result
