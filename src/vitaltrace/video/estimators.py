"""Estimadores de pessoa: YOLO-pose (preferido) e movimento via OpenCV (fallback leve)."""

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
    """YOLO-pose em CPU. Os pesos são baixados na primeira execução.

    `estimate_all` devolve TODAS as pessoas do quadro (quem é o paciente é decidido depois, pelo
    rastreador). `refine` roda a pose de novo só no recorte de uma pessoa, em resolução maior:
    é o segundo estágio, que melhora o mapeamento das articulações de quem aparece pequeno."""

    def __init__(
        self,
        weights: str = "yolo26s-pose.pt",
        conf: float = 0.35,
        model: Any = None,
        imgsz: int = 640,
        refine_imgsz: int = 480,
    ):
        if model is None:
            from ultralytics import YOLO  # import tardio: a dependência é opcional

            model = YOLO(weights)
        self._model = model
        self._conf = conf
        self.imgsz = imgsz
        self.refine_imgsz = refine_imgsz
        self.name = f"YOLO-pose ({weights})"

    def reset(self) -> None:
        """Sem estado entre quadros: a identidade das pessoas fica no rastreador."""

    def _predict(self, image: np.ndarray, imgsz: int, conf: float):
        results = self._model.predict(image, verbose=False, device="cpu", conf=conf, imgsz=imgsz)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        res = results[0]
        boxes, confs = _to_np(res.boxes.xyxy), _to_np(res.boxes.conf)
        keypoints = _to_np(res.keypoints.data) if getattr(res, "keypoints", None) is not None else None
        return boxes, confs, keypoints

    def estimate_all(self, frame_bgr: np.ndarray, t: float) -> list[PersonObservation]:
        found = self._predict(frame_bgr, self.imgsz, self._conf)
        if found is None:
            return []
        boxes, confs, keypoints = found
        h, w = frame_bgr.shape[:2]
        people = []
        for i in range(len(boxes)):
            kp = None
            if keypoints is not None and len(keypoints) > i:
                kp = [(float(x), float(y), float(c)) for x, y, c in keypoints[i]]
            people.append(
                PersonObservation(
                    t=t,
                    bbox=tuple(float(v) for v in boxes[i]),
                    keypoints=kp,
                    frame_size=(w, h),
                    people=int(len(boxes)),
                    conf=float(confs[i]),
                )
            )
        return people

    def estimate(self, frame_bgr: np.ndarray, t: float) -> PersonObservation | None:
        """A pessoa mais evidente do quadro (maior e mais confiável). Usado sem rastreamento."""
        people = self.estimate_all(frame_bgr, t)
        if not people:
            return None
        return max(people, key=lambda p: p.conf * (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))

    def refine(self, crop_bgr: np.ndarray) -> tuple[list[tuple[float, float, float]], float] | None:
        """Pose da pessoa principal de um recorte: (17 pontos nas coordenadas do recorte, confiança)."""
        found = self._predict(crop_bgr, self.refine_imgsz, min(self._conf, 0.25))
        if found is None or found[2] is None:
            return None
        boxes, confs, keypoints = found
        h, w = crop_bgr.shape[:2]
        # o recorte foi centrado no paciente: vale a detecção grande e mais perto do centro
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        off = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
        best = int(np.argmax(confs * area * (1.0 - np.minimum(off, 0.9))))
        if len(keypoints) <= best:
            return None
        return [(float(x), float(y), float(c)) for x, y, c in keypoints[best]], float(confs[best])


class MotionBBoxEstimator:
    """Sem rede neural: subtração de fundo (MOG2) + maior região em movimento.

    Mais simples e menos preciso que a pose (não vê articulações), mas roda em qualquer
    máquina e serve de plano B. Câmera deve estar parada."""

    name = "OpenCV (subtração de fundo)"

    def __init__(self, min_area_ratio: float = 0.01):
        self._min_area_ratio = min_area_ratio
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=32, detectShadows=False)
        self._seen = 0
        self._last_t: float | None = None

    def estimate(self, frame_bgr: np.ndarray, t: float) -> PersonObservation | None:
        h, w = frame_bgr.shape[:2]
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self._seen += 1
        dt = 0.5 if self._last_t is None else max(t - self._last_t, 0.0)  # aprendizado por segundo, não por quadro
        self._last_t = t
        # aprende o fundo rápido nos primeiros quadros; depois devagar, para que quem
        # fica parado (ou caído) não seja absorvido pelo fundo cedo demais
        mask = self._bg.apply(blurred, learningRate=0.5 if self._seen <= 3 else min(0.02, 0.006 * dt))
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
