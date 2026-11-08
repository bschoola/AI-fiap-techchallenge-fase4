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
