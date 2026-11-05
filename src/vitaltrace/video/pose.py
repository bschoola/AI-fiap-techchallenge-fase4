"""Geometria de pose: postura, movimento e ângulos. Funções puras, fáceis de testar."""

import math

import numpy as np

from vitaltrace.video.models import PersonObservation

# índices COCO-17
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
MIN_CONF = 0.3
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]  # fmt: skip


def bbox_height(obs: PersonObservation) -> float:
    return max(1.0, obs.bbox[3] - obs.bbox[1])


def bbox_width(obs: PersonObservation) -> float:
    return max(1.0, obs.bbox[2] - obs.bbox[0])


def _pt(obs: PersonObservation, idx: int) -> np.ndarray | None:
    if not obs.keypoints or idx >= len(obs.keypoints):
        return None
    x, y, c = obs.keypoints[idx]
    return np.array([x, y]) if c >= MIN_CONF else None


def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


def classify_posture(obs: PersonObservation) -> str:
    """standing | sitting | lying | upright (só bbox: de pé ou sentado, sem distinguir)."""
    shoulders = _mid(_pt(obs, L_SHOULDER), _pt(obs, R_SHOULDER))
    hips = _mid(_pt(obs, L_HIP), _pt(obs, R_HIP))
    if shoulders is not None and hips is not None:
        torso = hips - shoulders
        length = float(np.linalg.norm(torso))
        if length > 1:
            # ângulo do tronco em relação à vertical da imagem (0° = em pé)
            angle = math.degrees(math.atan2(abs(torso[0]), abs(torso[1]) + 1e-6))
            if angle > 60:
                return "lying"
            knees = _mid(_pt(obs, L_KNEE), _pt(obs, R_KNEE))
            if knees is not None and (knees[1] - hips[1]) < 0.55 * length:
                return "sitting"
            return "standing"
    return "lying" if bbox_width(obs) / bbox_height(obs) > 1.2 else "upright"


def is_upright(posture: str) -> bool:
    return posture in ("standing", "sitting", "upright")


def motion_per_second(prev: PersonObservation, cur: PersonObservation) -> float:
    """Deslocamento entre duas observações, em alturas-do-corpo por segundo."""
    dt = max(cur.t - prev.t, 1e-3)
    height = (bbox_height(prev) + bbox_height(cur)) / 2
    if prev.keypoints and cur.keypoints:
        disp = []
        for a, b in zip(prev.keypoints, cur.keypoints):
            if a[2] >= MIN_CONF and b[2] >= MIN_CONF:
                disp.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        if len(disp) >= 4:
            return float(np.mean(disp)) / height / dt
    cx_p, cy_p = (prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2
    cx_c, cy_c = (cur.bbox[0] + cur.bbox[2]) / 2, (cur.bbox[1] + cur.bbox[3]) / 2
    shift = math.hypot(cx_c - cx_p, cy_c - cy_p)
    resize = abs(bbox_width(cur) - bbox_width(prev)) + abs(bbox_height(cur) - bbox_height(prev))
    return (shift + 0.5 * resize) / height / dt


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def knee_angles(obs: PersonObservation) -> tuple[float, float] | None:
    """Ângulos (quadril–joelho–tornozelo) esquerdo e direito, se os dois lados estão visíveis."""
    pts = {i: _pt(obs, i) for i in (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)}
    if any(p is None for p in pts.values()):
        return None
    left = _angle(pts[L_HIP], pts[L_KNEE], pts[L_ANKLE])
    right = _angle(pts[R_HIP], pts[R_KNEE], pts[R_ANKLE])
    return left, right
