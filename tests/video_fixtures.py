"""Gera vídeos sintéticos (retângulo escuro = "pessoa") para testar o pipeline sem arquivos externos."""

from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 320, 240, 10


def write_video(path: Path, scenes: list[tuple[str, float]]) -> Path:
    """`scenes`: lista de (cena, segundos). Cenas: empty, walk, still, fall, lying; dark_empty e dark_lying têm outro fundo (simulam um corte)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    rng = np.random.RandomState(0)
    x = 40
    for scene, seconds in scenes:
        for k in range(int(seconds * FPS)):
            dark = scene.startswith("dark_")
            frame = np.full((H, W, 3), 40 if dark else 225, np.uint8)
            frame += rng.randint(0, 3, frame.shape, dtype=np.uint8)
            if scene == "walk":
                x = (x + 6) % (W - 80) + 10
                cv2.rectangle(frame, (x, 80), (x + 30, 200), (40, 40, 40), -1)
            elif scene == "still":
                cv2.rectangle(frame, (x, 80), (x + 30, 200), (40, 40, 40), -1)
            elif scene in ("fall", "lying"):
                cv2.rectangle(frame, (x, 170), (x + 110, 205), (40, 40, 40), -1)
            elif scene == "dark_lying":  # outra cena (corte): pessoa deitada em fundo escuro
                cv2.rectangle(frame, (60, 170), (170, 205), (210, 210, 210), -1)
            writer.write(frame)
    writer.release()
    return path


def skeleton(t, cx=100.0, top=20.0, height=160.0, knee_off=0.0, arm_up=0.0, conf=0.9, frame=(320, 240), side=False):
    """Pessoa em pé com 17 pontos COCO. knee_off dobra os joelhos para fora; arm_up (0–1) levanta os braços;
    side=True põe os dois lados quase no mesmo x (vista lateral)."""
    from vitaltrace.video.models import PersonObservation

    def y(frac):
        return top + frac * height

    half = 0.01 * height if side else 0.09 * height
    hip = 0.01 * height if side else 0.06 * height
    kp = [(cx, y(0.05), conf)] * 17
    kp = list(kp)
    kp[5], kp[6] = (cx - half, y(0.25), conf), (cx + half, y(0.25), conf)
    arm = 0.17 * height
    ey, wy = y(0.25) + arm * (1 - 2 * arm_up), y(0.25) + 2 * arm * (1 - 2 * arm_up)
    kp[7], kp[8] = (cx - half - 0.02 * height, ey, conf), (cx + half + 0.02 * height, ey, conf)
    kp[9], kp[10] = (cx - half - 0.02 * height, wy, conf), (cx + half + 0.02 * height, wy, conf)
    kp[11], kp[12] = (cx - hip, y(0.5), conf), (cx + hip, y(0.5), conf)
    kp[13], kp[14] = (cx - hip - knee_off, y(0.75), conf), (cx + hip + knee_off, y(0.75), conf)
    kp[15], kp[16] = (cx - hip, y(1.0), conf), (cx + hip, y(1.0), conf)
    bbox = (cx - 0.2 * height, top, cx + 0.2 * height, top + height)
    return PersonObservation(t=t, bbox=bbox, keypoints=kp, frame_size=frame)
