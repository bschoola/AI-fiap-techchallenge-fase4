"""Descreve o que foi visto no vídeo, trecho a trecho, em linguagem de consultório.

Não é diagnóstico nem classificação treinada: são regras simples sobre a postura, a articulação
que mais se moveu e o deslocamento da pessoa. Serve para o médico conferir, antes de olhar os
achados, se o sistema "viu" o mesmo exercício que ele.
"""

import numpy as np

from vitaltrace.video import pose
from vitaltrace.video.detail import ROM_MIN_DEG, _extremes
from vitaltrace.video.models import PersonObservation

MIN_FRAMES = 5
MIN_SECONDS = 2.0
MAX_SEGMENTS = 10
WALK_TRAVEL = 1.2  # deslocamento lateral, em alturas do corpo
POSTURES = {"standing": "em pé", "sitting": "sentado", "lying": "deitado", "upright": "na vertical"}
VIEWS = {"frontal": "vista frontal", "lateral": "vista lateral", "oblique": "vista oblíqua"}


def _mode(values: list) -> str | None:
    values = [v for v in values if v]
    return max(set(values), key=values.count) if values else None


def _exercise(posture_share: dict[str, float], joint: str | None, rom: float, travel: float) -> str:
    standing, sitting, lying = (posture_share.get(k, 0.0) for k in ("standing", "sitting", "lying"))
    moving = joint is not None and rom >= ROM_MIN_DEG
    lower = moving and (joint.startswith("joelho") or joint.startswith("quadril"))
    if travel >= WALK_TRAVEL and standing + posture_share.get("upright", 0.0) >= 0.5:
        return "marcha ou deslocamento pela sala"
    if lying >= 0.6:
        return f"exercício deitado com movimento de {joint}" if moving else "deitado, sem movimento repetitivo"
    if lower and sitting >= 0.2 and standing >= 0.2:
        return "sentar e levantar"
    if lower and sitting >= 0.6:
        return "flexão e extensão de joelho sentado"
    if lower:
        return "agachamento ou flexão de joelho e quadril em pé"
    if moving and joint.startswith("ombro"):
        return "elevação de braço (ombro)"
    if moving and joint.startswith("cotovelo"):
        return "flexão e extensão de cotovelo"
    return "postura mantida, sem movimento repetitivo amplo"


def describe_scene(
    observations: list[PersonObservation | None],
    times: list[float],
    segments: list[int],
    series: dict[str, np.ndarray],
    postures: list[str],
    views: list[str | None],
    people_counts: list[int] | None = None,
) -> list[dict]:
    """Um resumo por trecho contínuo: quem aparece, como, e o que provavelmente está fazendo."""
    people_counts = people_counts or [1 if o is not None else 0 for o in observations]
    seg_arr, t_arr = np.array(segments), np.array(times)
    scene = []
    for sid in sorted(set(segments)):
        idx = np.flatnonzero(seg_arr == sid)
        seen = [i for i in idx if observations[i] is not None]
        if len(seen) < MIN_FRAMES or times[idx[-1]] - times[idx[0]] < MIN_SECONDS:
            continue
        share: dict[str, float] = {}
        for i in seen:
            if postures[i]:
                share[postures[i]] = share.get(postures[i], 0.0) + 1 / len(seen)
        joint, rom, lo, hi = None, 0.0, None, None
        for name, values in series.items():
            valid = values[idx][~np.isnan(values[idx])]
            if len(valid) < 8:
                continue
            p5, p95 = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
            if p95 - p5 > rom:
                joint, rom, lo, hi = name, p95 - p5, p5, p95
        reps = 0
        if joint is not None and rom >= ROM_MIN_DEG:
            mask = idx[~np.isnan(series[joint][idx])]
            ext = _extremes(t_arr[mask], series[joint][mask]) or []
            reps = sum(1 for e in ext if e[2] == "min")
        xs = [(observations[i].bbox[0] + observations[i].bbox[2]) / 2 for i in seen]
        height = float(np.median([pose.bbox_height(observations[i]) for i in seen]))
        travel = (max(xs) - min(xs)) / height
        posture = max(share, key=share.get) if share else None
        view = _mode([views[i] for i in seen])
        people = int(max(people_counts[i] for i in idx))
        exercise = _exercise(share, joint, rom, travel)
        item = {
            "t_start": round(float(times[idx[0]]), 1),
            "t_end": round(float(times[idx[-1]]), 1),
            "people": people,
            "posture": POSTURES.get(posture or "", None),
            "view": VIEWS.get(view or "", None),
            "exercise": exercise,
            "joint": joint if rom >= ROM_MIN_DEG else None,
            "min": round(lo) if lo is not None and rom >= ROM_MIN_DEG else None,
            "max": round(hi) if hi is not None and rom >= ROM_MIN_DEG else None,
            "reps": reps,
            "visible": round(len(seen) / len(idx), 2),
        }
        who = "1 pessoa em cena" if people <= 1 else f"{people} pessoas em cena"
        how = ", ".join(x for x in (item["posture"], item["view"]) if x)
        text = f"{who}; paciente {how}. " if how else f"{who}. "
        text += f"Provável {exercise}"
        if item["joint"]:
            text += f": {joint} entre {item['min']}° e {item['max']}°"
            if reps >= 2:
                text += f", {reps} repetições"
        item["text"] = text + "."
        scene.append(item)
    return scene[:MAX_SEGMENTS]
