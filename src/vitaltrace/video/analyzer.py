"""Lê o vídeo, amostra quadros, estima a pessoa e produz métricas, eventos e relatório."""

import base64
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from vitaltrace.config import Settings
from vitaltrace.models import Alert, Severity
from vitaltrace.video import pose
from vitaltrace.video.estimators import MotionBBoxEstimator, PersonEstimator, YoloPoseEstimator
from vitaltrace.video.events import analyze_observations
from vitaltrace.video.models import Keyframe, PersonObservation, VideoAnalysis, VideoEvent
from vitaltrace.video.report import build_report, type_label

MAX_WIDTH = 640  # quadros são reduzidos para acelerar a inferência em CPU
MAX_KEYFRAMES = 3
KEYFRAME_WIDTH = 480  # quadros guardados no banco: menores que os usados na inferência


class VideoError(RuntimeError):
    pass


def _draw(frame: np.ndarray, obs: PersonObservation | None, label: str) -> np.ndarray:
    out = frame.copy()
    if obs is not None:
        x1, y1, x2, y2 = (int(v) for v in obs.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (245, 184, 61), 2)
        if obs.keypoints:
            for a, b in pose.SKELETON:
                if obs.keypoints[a][2] >= pose.MIN_CONF and obs.keypoints[b][2] >= pose.MIN_CONF:
                    cv2.line(out, tuple(int(v) for v in obs.keypoints[a][:2]),
                             tuple(int(v) for v in obs.keypoints[b][:2]), (110, 118, 15), 3)  # fmt: skip
            for x, y, c in obs.keypoints:
                if c >= pose.MIN_CONF:
                    cv2.circle(out, (int(x), int(y)), 4, (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (43, 42, 14), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _to_data_uri(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


class VideoAnalyzer:
    def __init__(
        self,
        estimator_factory: Callable[[], PersonEstimator],
        sample_fps: float = 2.0,
        max_seconds: int = 120,
    ):
        # Fábrica porque o estimador de movimento guarda o modelo de fundo: um novo por vídeo.
        self._factory = estimator_factory
        self._sample_fps = sample_fps
        self._max_seconds = max_seconds
        self._lock = threading.Lock()  # a inferência em CPU é serializada
        self.backend = estimator_factory().name

    def analyze(self, path: Path) -> VideoAnalysis:
        with self._lock:
            return self._analyze(path, self._factory())

    def _analyze(self, path: Path, estimator: PersonEstimator) -> VideoAnalysis:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoError("Não foi possível abrir o vídeo. Use MP4, AVI, MOV ou WebM.")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, round(fps / self._sample_fps))
            limit_frames = int(self._max_seconds * fps)

            observations: list[PersonObservation | None] = []
            times: list[float] = []
            index = 0
            while index < (limit_frames if total <= 0 else min(total, limit_frames)):
                if not cap.grab():
                    break
                if index % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        frame = self._resize(frame)
                        t = index / fps
                        observations.append(estimator.estimate(frame, t))
                        times.append(t)
                index += 1
            duration = (times[-1] - times[0]) if len(times) > 1 else 0.0
            if len(times) < 2:
                raise VideoError("O vídeo é curto demais para análise (menos de 2 quadros amostrados).")

            metrics, events = analyze_observations(observations, times, duration)
            keyframes = self._keyframes(cap, observations, times, events)
        finally:
            cap.release()
        return VideoAnalysis(
            backend=self.backend,
            metrics=metrics,
            events=events,
            report=build_report(metrics, events, self.backend),
            keyframes=keyframes,
        )

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= MAX_WIDTH:
            return frame
        return cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)

    def _keyframes(self, cap, observations, times, events: list[VideoEvent]) -> list[Keyframe]:
        """Quadros com a pose desenhada: um por evento mais grave; senão, o meio do vídeo."""
        ranked = sorted(enumerate(events), key=lambda item: item[1].severity.rank, reverse=True)
        wanted = [(e.t_start, type_label(e.type), idx) for idx, e in ranked[:MAX_KEYFRAMES]]
        if not wanted:
            wanted = [(times[len(times) // 2], "Quadro de exemplo", None)]
        images: list[Keyframe] = []
        for t_wanted, label, event_index in wanted:
            nearest = min(range(len(times)), key=lambda i: abs(times[i] - t_wanted))
            cap.set(cv2.CAP_PROP_POS_MSEC, times[nearest] * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = self._resize(frame)
            drawn = _draw(frame, observations[nearest], f"{label} · {times[nearest]:.0f} s")
            if drawn.shape[1] > KEYFRAME_WIDTH:
                drawn = cv2.resize(
                    drawn, (KEYFRAME_WIDTH, int(drawn.shape[0] * KEYFRAME_WIDTH / drawn.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
            images.append(
                Keyframe(t=round(times[nearest], 1), label=label, image=_to_data_uri(drawn), event_index=event_index)
            )
        return images


def build_video_alerts(patient_id: int | str, analysis: VideoAnalysis) -> list[Alert]:
    alerts = []
    for e in analysis.events:
        if e.severity.rank < Severity.MEDIUM.rank:
            continue
        alerts.append(
            Alert(
                patient_id=str(patient_id),
                source="video",
                severity=e.severity,
                title=f"{type_label(e.type)} ({e.t_start:.0f}–{e.t_end:.0f} s)",
                description=e.description,
                evidence={"event": e.model_dump(mode="json"), "backend": analysis.backend},
            )
        )
    return alerts


def build_video_analyzer(settings: Settings) -> VideoAnalyzer:
    backend = settings.video_backend.lower()
    factory: Callable[[], PersonEstimator] = MotionBBoxEstimator
    if backend != "motion":
        try:
            yolo = YoloPoseEstimator(settings.yolo_pose_weights)  # carrega o modelo uma única vez
            factory = lambda: yolo  # noqa: E731
        except Exception as exc:  # ultralytics ausente ou pesos indisponíveis
            if backend == "yolo":
                raise VideoError(f"YOLOv8-pose indisponível: {exc}") from exc
    return VideoAnalyzer(factory, settings.video_sample_fps, settings.video_max_seconds)
