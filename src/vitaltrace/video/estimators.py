"""Estimadores de pessoa: YOLOv8-pose (preferido) e movimento via OpenCV (fallback leve)."""

from typing import Any, Protocol

import cv2
import numpy as np

from vitaltrace.video.models import PersonObservation


class PersonEstimator(Protocol):
    name: str

    def estimate(self, frame_bgr: np.ndarray, t: float) -> PersonObservation | None: ...


def _to_np(value: Any) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


class YoloPoseEstimator:
    """YOLOv8-pose em CPU. Os pesos `yolov8n-pose.pt` (~6 MB) são baixados na primeira execução."""

    def __init__(self, weights: str = "yolov8n-pose.pt", conf: float = 0.35, model: Any = None):
        if model is None:
            from ultralytics import YOLO  # import tardio: a dependência é opcional

            model = YOLO(weights)
        self._model = model
        self._conf = conf
        self.name = f"YOLOv8-pose ({weights})"

    def estimate(self, frame_bgr: np.ndarray, t: float) -> PersonObservation | None:
        results = self._model.predict(frame_bgr, verbose=False, device="cpu", conf=self._conf, imgsz=480)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        res = results[0]
        boxes = _to_np(res.boxes.xyxy)
        confs = _to_np(res.boxes.conf)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(confs * areas))  # a pessoa mais evidente do quadro
        keypoints = None
        if getattr(res, "keypoints", None) is not None:
            data = _to_np(res.keypoints.data)
            if len(data) > best:
                keypoints = [(float(x), float(y), float(c)) for x, y, c in data[best]]
        h, w = frame_bgr.shape[:2]
        return PersonObservation(
            t=t,
            bbox=tuple(float(v) for v in boxes[best]),
            keypoints=keypoints,
            frame_size=(w, h),
        )


class MotionBBoxEstimator:
    """Sem rede neural: subtração de fundo (MOG2) + maior região em movimento.

    Mais simples e menos preciso que a pose (não vê articulações), mas roda em qualquer
    máquina e serve de plano B. Câmera deve estar parada."""

    name = "OpenCV (subtração de fundo)"

    def __init__(self, min_area_ratio: float = 0.01):
        self._bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=32, detectShadows=False)
        self._min_area_ratio = min_area_ratio
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._seen = 0

    def estimate(self, frame_bgr: np.ndarray, t: float) -> PersonObservation | None:
        h, w = frame_bgr.shape[:2]
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self._seen += 1
        # aprende o fundo rápido nos primeiros quadros; depois devagar, para que quem
        # fica parado (ou caído) não seja absorvido pelo fundo cedo demais
        mask = self._bg.apply(blurred, learningRate=0.5 if self._seen <= 3 else 0.003)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area < self._min_area_ratio * w * h or area > 0.7 * w * h:
            return None  # ruído ou mudança global de iluminação
        x, y, bw, bh = cv2.boundingRect(biggest)
        return PersonObservation(t=t, bbox=(float(x), float(y), float(x + bw), float(y + bh)), frame_size=(w, h))
