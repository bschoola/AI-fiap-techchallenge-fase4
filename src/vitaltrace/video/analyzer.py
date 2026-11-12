"""Lê o vídeo, acompanha as pessoas, fixa o paciente e produz métricas, eventos e relatório.

Duas passadas pelo vídeo:

1. `scan`: amostra quadros, detecta TODAS as pessoas e dá um número estável a cada uma
   (tracking.py). Se houver mais de uma candidata a paciente, a tela pergunta ao médico quem é.
2. `finish`: com o paciente fixado, refaz a pose só no recorte dele em resolução original
   (modo detalhado), limpa o esqueleto (clean.py), mede, descreve a cena e gera relatório,
   quadros-chave e o vídeo com o esqueleto desenhado.

`analyze` faz as duas de uma vez, com escolha automática do paciente.
"""

import base64
import secrets
import shutil
import subprocess
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vitaltrace.config import Settings
from vitaltrace.models import Alert, Severity
from vitaltrace.video import clean, pose, tracking
from vitaltrace.video.describe import describe_scene
from vitaltrace.video.detail import ROM_MIN_DEG, analyze_detail, smooth_keypoints
from vitaltrace.video.estimators import MotionBBoxEstimator, PersonEstimator, YoloPoseEstimator
from vitaltrace.video.events import analyze_observations
from vitaltrace.video.models import Keyframe, PersonObservation, VideoAnalysis, VideoEvent, VideoQuality
from vitaltrace.video.report import build_report, type_label

DEFAULT_WIDTH = 960  # maior lado do quadro analisado (o original maior é reduzido a isso)
MAX_KEYFRAMES = 3
MAX_KEYFRAMES_DETAILED = 6
CUT_CORRELATION = 0.6  # histograma de cinza muito diferente do quadro anterior
CUT_DIFF = 45.0  # ou miniatura muito diferente (diferença média de pixel, 0–255)
DEPTHS = ("fast", "detailed")
KEYFRAME_WIDTH = 640  # quadros guardados no banco
ANNOTATED_WIDTH = 640  # vídeo com o esqueleto desenhado
REFINE_GAIN = 1.25  # só refaz a pose no recorte se o paciente ganhar ao menos 25% de pixels
MIN_POSE_CONF = 0.4  # confiança média do esqueleto abaixo disso: achados de pose não são emitidos
THUMB_HEIGHT = 220
# achados que dependem do esqueleto (os demais — queda, imobilidade — valem só com a caixa)
POSE_EVENTS = {
    "knee_asymmetry", "trunk_lean", "shoulder_tilt", "hip_tilt", "postural_sway",
    "rom_asymmetry", "rep_irregular", "rep_fatigue",
}  # fmt: skip


class VideoError(RuntimeError):
    pass


def _ascii(text: str) -> str:
    """As fontes do OpenCV não têm acentos: 'Oscilação' viraria 'Oscila??o'."""
    text = text.replace("·", "-").replace("°", " graus").replace("º", "o").replace("–", "-")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _draw(
    frame: np.ndarray,
    obs: PersonObservation | None,
    label: str,
    k: float = 1.0,
    others: list[PersonObservation] | None = None,
) -> np.ndarray:
    """Desenha o paciente (caixa e esqueleto) e, em cinza, as outras pessoas. `k` = escala das coordenadas."""
    out = frame.copy()
    for o in others or []:
        x1, y1, x2, y2 = (int(v * k) for v in o.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (170, 170, 170), 1)
        if o.track_id is not None:
            cv2.putText(out, f"n. {o.track_id}", (x1 + 3, max(40, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    if obs is not None:
        x1, y1, x2, y2 = (int(v * k) for v in obs.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (245, 184, 61), 2)
        if obs.keypoints:
            pts = [(int(x * k), int(y * k), c) for x, y, c in obs.keypoints]
            for a, b in pose.SKELETON:
                if pts[a][2] >= pose.MIN_CONF and pts[b][2] >= pose.MIN_CONF:
                    cv2.line(out, pts[a][:2], pts[b][:2], (110, 118, 15), 3)
            for x, y, c in pts:
                if c >= pose.MIN_CONF:
                    cv2.circle(out, (x, y), 4, (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (43, 42, 14), -1)
    cv2.putText(out, _ascii(label), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _signature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist, cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)


def _is_cut(prev: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray]) -> bool:
    corr = cv2.compareHist(prev[0], cur[0], cv2.HISTCMP_CORREL)
    return corr < CUT_CORRELATION or float(np.mean(np.abs(prev[1] - cur[1]))) > CUT_DIFF


def _to_data_uri(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _fit(frame: np.ndarray, side: int) -> np.ndarray:
    """Reduz para que o MAIOR lado tenha no máximo `side` (vídeo de celular em pé incluso)."""
    h, w = frame.shape[:2]
    if max(h, w) <= side:
        return frame
    scale = side / max(h, w)
    return cv2.resize(frame, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)


def _appearance(frame: np.ndarray, p: PersonObservation) -> np.ndarray | None:
    """Histograma de cor do tronco (matiz e saturação): reencontra a pessoa depois de um corte."""
    x1, y1, x2, y2 = p.bbox
    w, h = x2 - x1, y2 - y1
    box = (x1 + 0.25 * w, y1 + 0.2 * h, x2 - 0.25 * w, y1 + 0.6 * h)
    if p.keypoints and len(p.keypoints) >= 13:
        torso = [p.keypoints[i] for i in (pose.L_SHOULDER, pose.R_SHOULDER, pose.L_HIP, pose.R_HIP)]
        if all(k[2] >= pose.MIN_CONF for k in torso):
            xs, ys = [k[0] for k in torso], [k[1] for k in torso]
            if max(xs) - min(xs) >= 6 and max(ys) - min(ys) >= 6:
                box = (min(xs), min(ys), max(xs), max(ys))
    fh, fw = frame.shape[:2]
    xa, ya, xb, yb = int(max(0, box[0])), int(max(0, box[1])), int(min(fw, box[2])), int(min(fh, box[3]))
    if xb - xa < 4 or yb - ya < 4:
        return None
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    total = float(hist.sum())
    return hist / total if total > 0 else None


@dataclass
class VideoScan:
    """Resultado da 1ª passada: quem aparece em cada quadro amostrado, já com número estável."""

    depth: str
    sample_fps: float
    fps: float
    indices: list[int]  # índice, no vídeo, de cada quadro amostrado
    times: list[float]
    segments: list[int]
    cut_times: list[float]
    frames: list[list[PersonObservation]]
    stats: list[tracking.TrackStats]
    source_size: tuple[int, int]
    analysis_size: tuple[int, int]
    tracked: bool
    thumbs: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def candidates(self) -> list[tracking.TrackStats]:
        return tracking.candidates(self.stats) if self.tracked else []

    @property
    def needs_choice(self) -> bool:
        """Duas ou mais candidatas a paciente aparecem AO MESMO TEMPO: melhor o médico dizer quem é."""
        ids = {c.id for c in self.candidates()}
        if len(ids) < 2:
            return False
        together = sum(1 for people in self.frames if len(ids & {p.track_id for p in people}) >= 2)
        return together >= 5

    def candidate_cards(self) -> list[dict]:
        cards = []
        for rank, c in enumerate(self.candidates()):
            thumb = self.thumbs.get(c.id)
            cards.append(
                {
                    "id": c.id,
                    "image": _to_data_uri(thumb) if thumb is not None else "",
                    "share": round(c.share, 2),
                    "first_t": round(c.first_t, 1),
                    "last_t": round(c.last_t, 1),
                    "activity": round(c.activity, 2),
                    "suggested": rank == 0,
                }
            )
        return cards


@dataclass
class PendingVideo:
    path: Path
    scan: VideoScan
    patient_id: int
    source: dict
    folder: Path
    created: float = field(default_factory=time.monotonic)


class PendingVideos:
    """Vídeos já varridos, à espera de o médico indicar o paciente. Poucos, por pouco tempo, e apagados."""

    def __init__(self, ttl_s: float = 20 * 60, limit: int = 3):
        self._ttl_s, self._limit = ttl_s, limit
        self._items: dict[str, PendingVideo] = {}
        self._lock = threading.Lock()

    def _expire(self) -> None:
        now = time.monotonic()
        old = [k for k, v in self._items.items() if now - v.created > self._ttl_s]
        while len(self._items) - len(old) > self._limit:
            rest = [k for k in self._items if k not in old]
            old.append(min(rest, key=lambda k: self._items[k].created))
        for k in old:
            shutil.rmtree(self._items.pop(k).folder, ignore_errors=True)

    def put(self, item: PendingVideo) -> str:
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._items[token] = item
            self._expire()
        return token

    def pop(self, token: str, patient_id: int) -> PendingVideo | None:
        with self._lock:
            self._expire()
            item = self._items.get(token)
            if item is None or item.patient_id != patient_id:
                return None
            return self._items.pop(token)


class VideoAnalyzer:
    def __init__(
        self,
        estimator_factory: Callable[[], PersonEstimator],
        sample_fps: float = 2.0,
        max_seconds: int = 120,
        detail_fps: float = 8.0,
        width: int = DEFAULT_WIDTH,
        refine: bool = True,
        annotate: bool = True,
        narrator=None,
    ):
        # Fábrica porque o estimador de movimento guarda o modelo de fundo: um novo por vídeo.
        self._factory = estimator_factory
        self._sample_fps = sample_fps
        self._detail_fps = max(detail_fps, sample_fps)
        self._max_seconds = max_seconds
        self._width = max(160, width)
        self._refine = refine
        self._annotate = annotate
        self._narrator = narrator
        self._lock = threading.Lock()  # a inferência em CPU é serializada
        self.backend = estimator_factory().name
        self.detail_fps = self._detail_fps
        self.pending = PendingVideos()

    # ------------------------------------------------------------------ API
    def analyze(
        self, path: Path, depth: str = "detailed", person: int | None = None, annotated_to: Path | None = None
    ) -> VideoAnalysis:
        """`fast`: amostragem leve, só eventos básicos. `detailed`: mais quadros, pose refinada e anomalias no próprio vídeo."""
        return self.finish(path, self.scan(path, depth), person, annotated_to)

    def scan(self, path: Path, depth: str = "detailed") -> VideoScan:
        depth = depth if depth in DEPTHS else "detailed"
        with self._lock:
            return self._scan(path, self._factory(), depth)

    def finish(
        self, path: Path, scan: VideoScan, person: int | None = None, annotated_to: Path | None = None
    ) -> VideoAnalysis:
        with self._lock:
            return self._finish(path, scan, self._factory(), person, annotated_to)

    # ------------------------------------------------------------ 1ª passada
    def _scan(self, path: Path, estimator: PersonEstimator, depth: str) -> VideoScan:
        sample_fps = self._detail_fps if depth == "detailed" else self._sample_fps
        reset = getattr(estimator, "reset", lambda: None)
        reset()
        detect_all = getattr(estimator, "estimate_all", None)
        tracker = tracking.PoseTracker()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoError("Não foi possível abrir o vídeo. Use MP4, AVI, MOV ou WebM.")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if not (1.0 <= fps <= 240.0):
                fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, round(fps / sample_fps))
            limit_frames = int(self._max_seconds * fps)

            frames: list[list[PersonObservation]] = []
            times: list[float] = []
            segments: list[int] = []
            indices: list[int] = []
            cut_times: list[float] = []
            hists: dict[int, np.ndarray] = {}
            thumbs: dict[int, np.ndarray] = {}
            thumb_score: dict[int, float] = {}
            source_size, analysis_size = (0, 0), (0, 0)
            segment, last_sig = 0, None
            index = 0
            while index < (limit_frames if total <= 0 else min(total, limit_frames)):
                if not cap.grab():
                    break
                if index % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        source_size = (frame.shape[1], frame.shape[0])
                        frame = _fit(frame, self._width)
                        analysis_size = (frame.shape[1], frame.shape[0])
                        t = index / fps
                        sig = _signature(frame)
                        if last_sig is not None and _is_cut(last_sig, sig):
                            segment += 1
                            cut_times.append(round(t, 1))
                            reset()  # nova cena: quem estava sendo seguido pode não existir mais
                            tracker.reset()
                        last_sig = sig
                        if detect_all is not None:
                            people = tracker.update(detect_all(frame, t), t)
                            for p in people:
                                self._remember(frame, p, hists, thumbs, thumb_score)
                        else:  # estimador de uma pessoa só (plano B): não há quem confundir
                            one = estimator.estimate(frame, t)
                            people = [one.model_copy(update={"track_id": 0})] if one is not None else []
                        frames.append(people)
                        times.append(t)
                        segments.append(segment)
                        indices.append(index)
                index += 1
        finally:
            cap.release()
        if len(times) < 2:
            raise VideoError("O vídeo é curto demais para análise (menos de 2 quadros amostrados).")
        if detect_all is not None:
            frames = tracking.merge_fragments(frames, segments, hists)
            thumbs = {tid: img for tid, img in thumbs.items() if any(p.track_id == tid for people in frames for p in people)}
        return VideoScan(
            depth=depth,
            sample_fps=fps / step,
            fps=fps,
            indices=indices,
            times=times,
            segments=segments,
            cut_times=cut_times,
            frames=frames,
            stats=tracking.summarize(frames, segments, hists) if detect_all is not None else [],
            source_size=source_size,
            analysis_size=analysis_size,
            tracked=detect_all is not None,
            thumbs=thumbs,
        )

    @staticmethod
    def _remember(frame, p: PersonObservation, hists, thumbs, thumb_score) -> None:
        """Acumula a aparência de cada pessoa e guarda a melhor miniatura dela (para o médico escolher)."""
        hist = _appearance(frame, p)
        if hist is not None:
            hists[p.track_id] = hists.get(p.track_id, 0) + hist
        x1, y1, x2, y2 = p.bbox
        score = p.conf * (x2 - x1) * (y2 - y1)
        if score > 1.2 * thumb_score.get(p.track_id, 0.0):
            fh, fw = frame.shape[:2]
            px, py = 0.12 * (x2 - x1), 0.06 * (y2 - y1)
            crop = frame[int(max(0, y1 - py)) : int(min(fh, y2 + py)), int(max(0, x1 - px)) : int(min(fw, x2 + px))]
            if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 8:
                scale = THUMB_HEIGHT / crop.shape[0]
                thumbs[p.track_id] = cv2.resize(crop, (max(8, round(crop.shape[1] * scale)), THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
                thumb_score[p.track_id] = score

    # ------------------------------------------------------------ 2ª passada
    def _finish(self, path, scan: VideoScan, estimator, person: int | None, annotated_to: Path | None) -> VideoAnalysis:
        times, segments = scan.times, scan.segments
        detailed = scan.depth == "detailed"
        if scan.tracked:
            if person is not None and person not in {s.id for s in scan.stats}:
                person = None
            chosen = tracking.choose(scan.stats, person)
        else:
            person, chosen = None, {seg: [0] for seg in set(segments)}
        observations = tracking.pick(scan.frames, segments, chosen)
        counts = [len(people) for people in scan.frames]

        jpegs: list[bytes | None] | None = None
        refined = 0
        if detailed:
            refiner = estimator if self._refine and hasattr(estimator, "refine") else None
            observations, refined, jpegs = self._second_pass(path, scan, observations, refiner)
        swaps = dropped = 0
        if detailed:
            observations, swaps = clean.fix_swaps(observations, segments)
            observations, dropped = clean.drop_implausible(observations, segments)
            observations = smooth_keypoints(observations, segments)

        duration = times[-1] - times[0]
        metrics, events = analyze_observations(observations, times, duration, segments)
        joints, curve, ranges, scene = {}, None, {}, []
        if detailed:
            found = analyze_detail(observations, times, segments)
            events = sorted(self._drop_false_immobility(events, found.series, times) + found.events, key=lambda e: e.t_start)
            joints, curve, ranges = found.joints, found.curve, found.ranges
            if found.postures:
                scene = describe_scene(observations, times, segments, found.series, found.postures, found.views, counts)
        people = self._people(scan, chosen, person, counts)
        quality = self._quality(observations, scan, metrics.duration_s, curve, people)
        quality.fixed_swaps, quality.dropped_points = swaps, dropped
        with_pose = sum(1 for o in observations if o is not None and o.keypoints)
        if detailed and with_pose:
            quality.refined_ratio = round(refined / with_pose, 2)
        events = self._gate(events, quality)

        limit = MAX_KEYFRAMES_DETAILED if detailed else MAX_KEYFRAMES
        k = 1.0
        if jpegs is not None and any(j is not None for j in jpegs):
            k = min(1.0, ANNOTATED_WIDTH / max(scan.analysis_size))
            keyframes = self._keyframes_from(jpegs, observations, times, events, limit, k)
        else:
            keyframes = self._keyframes_seek(path, observations, times, events, limit)

        narrative = ""
        if detailed and self._narrator is not None and jpegs:
            facts = build_report(metrics, events, self.backend, quality, joints, people, scene, ranges)
            seen = [j for j, o in zip(jpegs, observations) if j is not None and o is not None] or [j for j in jpegs if j]
            picks = [seen[i] for i in np.linspace(0, len(seen) - 1, min(6, len(seen))).astype(int)] if seen else []
            try:
                narrative = self._narrator.describe(picks, facts)
            except Exception as exc:  # noqa: BLE001 — a descrição é opcional: nunca derruba a análise
                quality.warnings.append(f"Descrição da cena por IA indisponível ({exc}).")

        annotated = None
        if detailed and self._annotate and annotated_to is not None and jpegs:
            annotated = self._write_annotated(annotated_to, jpegs, observations, scan, chosen, scene, k)
            if annotated is None:
                quality.warnings.append("Não foi possível gerar o vídeo com o esqueleto (codificação falhou); os quadros-chave e as medidas não são afetados.")
        return VideoAnalysis(
            backend=self.backend,
            metrics=metrics,
            events=events,
            report=build_report(metrics, events, self.backend, quality, joints, people, scene, ranges, narrative),
            keyframes=keyframes,
            quality=quality,
            joints=joints,
            curve=curve,
            joint_ranges=ranges,
            people=people,
            scene=scene,
            narrative=narrative,
            annotated_path=str(annotated) if annotated else None,
        )

    def _second_pass(self, path, scan: VideoScan, observations, refiner):
        """Relê os quadros amostrados: refina a pose do paciente no recorte original e guarda miniaturas."""
        position = {idx: i for i, idx in enumerate(scan.indices)}
        jpegs: list[bytes | None] = [None] * len(scan.indices)
        out = list(observations)
        refined = 0
        cap = cv2.VideoCapture(str(path))
        try:
            for index in range(scan.indices[-1] + 1):
                if not cap.grab():
                    break
                i = position.get(index)
                if i is None:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                o = out[i]
                if refiner is not None and o is not None and o.keypoints:
                    sx = frame.shape[1] / max(1, scan.analysis_size[0])
                    if self._worth_refining(o, sx, scan.analysis_size, refiner):
                        better = self._refine_one(refiner, frame, o, sx)
                        if better is not None:
                            out[i] = better
                            refined += 1
                ok, buf = cv2.imencode(".jpg", _fit(frame, ANNOTATED_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    jpegs[i] = buf.tobytes()
        finally:
            cap.release()
        return out, refined, jpegs

    @staticmethod
    def _worth_refining(o: PersonObservation, sx: float, analysis_size, refiner) -> bool:
        """Vale quando o recorte dá ao modelo mais pixels do paciente do que o quadro inteiro deu."""
        height = pose.bbox_height(o)
        first = height * getattr(refiner, "imgsz", 640) / max(analysis_size)  # altura na 1ª passada
        second = min(height * sx, getattr(refiner, "refine_imgsz", 480) / 1.2)  # altura no recorte (com margem)
        return second >= REFINE_GAIN * first

    @staticmethod
    def _refine_one(refiner, frame: np.ndarray, o: PersonObservation, sx: float) -> PersonObservation | None:
        x1, y1, x2, y2 = o.bbox
        px, py = 0.15 * (x2 - x1), 0.10 * (y2 - y1)
        fh, fw = frame.shape[:2]
        cx1, cy1 = int(max(0, (x1 - px) * sx)), int(max(0, (y1 - py) * sx))
        cx2, cy2 = int(min(fw, (x2 + px) * sx)), int(min(fh, (y2 + py) * sx))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return None
        found = refiner.refine(frame[cy1:cy2, cx1:cx2])
        if found is None:
            return None
        keypoints = [((x + cx1) / sx, (y + cy1) / sx, c) for x, y, c in found[0]]
        if len(keypoints) != len(o.keypoints):
            return None
        if float(np.mean([k[2] for k in keypoints])) + 0.05 < float(np.mean([k[2] for k in o.keypoints])):
            return None  # o recorte saiu pior que o quadro inteiro: fica o original
        visible = [k for k in keypoints if k[2] >= pose.MIN_CONF]
        inside = [k for k in visible if x1 - px <= k[0] <= x2 + px and y1 - py <= k[1] <= y2 + py]
        if not visible or len(inside) < 0.7 * len(visible):
            return None  # pegou outra pessoa do recorte
        return o.model_copy(update={"keypoints": keypoints})

    # ------------------------------------------------------------- resumo
    @staticmethod
    def _people(scan: VideoScan, chosen: dict[int, list[int]], person: int | None, counts: list[int]) -> dict:
        most = max(counts) if counts else 0
        analyzed = None
        if scan.tracked and most > 1:
            main = tracking.main_segment(scan.stats)
            analyzed = (chosen.get(main) or [None])[0]
        choice = "single" if most <= 1 else ("manual" if person is not None else "auto")
        return {"max": most, "analyzed": analyzed, "choice": choice, "candidates": len(scan.candidates())}

    def _quality(self, observations, scan: VideoScan, duration, curve, people: dict) -> VideoQuality:
        detected = [o for o in observations if o is not None]
        with_pose = [o for o in detected if o.keypoints]
        usable = [o for o in with_pose if sum(k[2] >= pose.MIN_CONF for k in o.keypoints) >= 12]
        multi = sum(1 for people_in_frame in scan.frames if len(people_in_frame) > 1)
        q = VideoQuality(
            depth=scan.depth,
            sample_fps=round(scan.sample_fps, 1),
            cuts=len(scan.cut_times),
            cut_times=scan.cut_times[:20],
            multi_person_ratio=round(multi / len(scan.frames), 2) if scan.frames else 0.0,
            analysis_width=scan.analysis_size[0],
            source_size=scan.source_size,
        )
        if with_pose:
            q.keypoint_ratio = round(len(usable) / len(observations), 2)
            q.mean_keypoint_conf = round(float(np.mean([k[2] for o in with_pose for k in o.keypoints])), 2)
        w = q.warnings
        ratio = len(detected) / len(observations) if observations else 0.0
        if ratio < 0.5:
            w.append(f"O paciente aparece em só {ratio:.0%} dos quadros: os resultados são pouco confiáveis.")
        if not with_pose:
            w.append("Sem pose (esqueleto): ângulos, simetria e repetições não foram avaliados; só movimento e queda.")
        elif q.keypoint_ratio is not None and q.keypoint_ratio < 0.5:
            w.append("O corpo inteiro aparece em menos da metade dos quadros: ângulos e simetria menos confiáveis.")
        if q.cuts:
            w.append(f"Vídeo editado: {q.cuts} corte(s) de cena. Velocidade e quedas são avaliadas dentro de cada trecho, nunca entre cortes.")
        if q.multi_person_ratio > 0.3:
            who = f"da pessoa nº {people['analyzed']}" if people.get("analyzed") is not None else "do paciente"
            how = "indicada por você" if people.get("choice") == "manual" else "escolhida automaticamente"
            tail = " Depois de um corte ela é reencontrada pela cor da roupa, o que pode falhar." if q.cuts else ""
            w.append(f"Mais de uma pessoa aparece em boa parte do vídeo; as medidas são só {who} ({how}).{tail}")
        if duration < 10:
            w.append("Vídeo curto (menos de 10 s): ritmo e fadiga de repetições podem não ser avaliáveis.")
        if scan.depth == "detailed" and with_pose and not curve:
            w.append("Não foi encontrado movimento repetitivo com amplitude suficiente: ritmo e fadiga não avaliados.")
        if 0 < max(scan.source_size) < 480:
            w.append(f"Vídeo em baixa resolução ({scan.source_size[0]}×{scan.source_size[1]}): as articulações ficam menos precisas.")
        return q

    @staticmethod
    def _drop_false_immobility(events: list[VideoEvent], series: dict, times: list[float]) -> list[VideoEvent]:
        """Exercício de um membro só (extensão de joelho sentado, elevação de braço) mexe poucos pontos do
        corpo e parecia "imobilidade". Se uma articulação trabalhou no intervalo, o evento não vale."""
        t = np.array(times)
        kept = []
        for e in events:
            if e.type in ("prolonged_immobility", "low_mobility"):
                window = (t >= e.t_start) & (t <= e.t_end)
                active = False
                for values in series.values():
                    valid = values[window][~np.isnan(values[window])]
                    if len(valid) >= 8 and np.percentile(valid, 95) - np.percentile(valid, 5) >= ROM_MIN_DEG:
                        active = True
                        break
                if active:
                    continue
            kept.append(e)
        return kept

    @staticmethod
    def _gate(events: list[VideoEvent], quality: VideoQuality) -> list[VideoEvent]:
        """Esqueleto pouco confiável não vira achado clínico: os eventos de pose saem e fica o aviso."""
        conf = quality.mean_keypoint_conf
        if conf is None or conf >= MIN_POSE_CONF:
            return events
        kept = [e for e in events if e.type not in POSE_EVENTS]
        removed = len(events) - len(kept)
        quality.warnings.append(
            f"Esqueleto pouco confiável (confiança média {conf:.2f}): "
            + (f"{removed} achado(s) de postura/amplitude foram descartados. " if removed else "achados de postura/amplitude não são emitidos. ")
            + "Grave com o corpo inteiro em cena, boa luz e câmera parada."
        )
        return kept

    # ------------------------------------------------------- quadros-chave
    @staticmethod
    def _wanted(times, events: list[VideoEvent], limit: int):
        ranked = sorted(enumerate(events), key=lambda item: item[1].severity.rank, reverse=True)
        wanted = [(e.t_start, type_label(e.type), idx) for idx, e in ranked[:limit]]
        return wanted or [(times[len(times) // 2], "Quadro de exemplo", None)]

    def _keyframes_from(self, jpegs, observations, times, events, limit: int, k: float) -> list[Keyframe]:
        """Quadros com a pose desenhada: um por evento mais grave; senão, o meio do vídeo."""
        images: list[Keyframe] = []
        have = [i for i, j in enumerate(jpegs) if j is not None]
        for t_wanted, label, event_index in self._wanted(times, events, limit):
            nearest = min(have, key=lambda i: abs(times[i] - t_wanted))
            frame = cv2.imdecode(np.frombuffer(jpegs[nearest], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            drawn = _draw(frame, observations[nearest], f"{label} · {times[nearest]:.0f} s", k)
            images.append(Keyframe(t=round(times[nearest], 1), label=label, image=_to_data_uri(_fit(drawn, KEYFRAME_WIDTH)), event_index=event_index))
        return images

    def _keyframes_seek(self, path, observations, times, events, limit: int) -> list[Keyframe]:
        images: list[Keyframe] = []
        cap = cv2.VideoCapture(str(path))
        try:
            for t_wanted, label, event_index in self._wanted(times, events, limit):
                nearest = min(range(len(times)), key=lambda i: abs(times[i] - t_wanted))
                cap.set(cv2.CAP_PROP_POS_MSEC, times[nearest] * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                drawn = _draw(_fit(frame, self._width), observations[nearest], f"{label} · {times[nearest]:.0f} s")
                images.append(Keyframe(t=round(times[nearest], 1), label=label, image=_to_data_uri(_fit(drawn, KEYFRAME_WIDTH)), event_index=event_index))
        finally:
            cap.release()
        return images

    # ------------------------------------------------------ vídeo anotado
    def _write_annotated(self, dest: Path, jpegs, observations, scan: VideoScan, chosen, scene, k: float) -> Path | None:
        """Vídeo reduzido com o esqueleto do paciente e as outras pessoas em cinza. Falhou? Segue sem ele.

        Preferência: H.264 em MP4 pelo ffmpeg (toca em qualquer navegador). Sem ffmpeg, VP8 em WebM
        pelo OpenCV. O arquivo só é aceito depois de reaberto e de ter um quadro decodificado."""
        try:
            first = next(j for j in jpegs if j is not None)
            sample = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
            size = (sample.shape[1] - sample.shape[1] % 2, sample.shape[0] - sample.shape[0] % 2)
            fps = round(max(1.0, min(30.0, scan.sample_fps)), 2)
            patient_ids = {i for ids in chosen.values() for i in ids}

            def frames():
                for i, jpeg in enumerate(jpegs):
                    if jpeg is None:
                        continue
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    t = scan.times[i]
                    label = f"{t:5.1f} s"
                    o = observations[i]
                    if o is not None and o.track_id and len(scan.frames[i]) > 1:
                        label += f" | paciente n. {o.track_id}"
                    part = next((s for s in scene if s["t_start"] <= t <= s["t_end"]), None)
                    if part:
                        label += f" | {part['exercise']}"
                    others = [p for p in scan.frames[i] if p.track_id not in patient_ids]
                    drawn = _draw(frame, o, label[: max(20, size[0] // 9)], k, others)
                    yield drawn if (drawn.shape[1], drawn.shape[0]) == size else cv2.resize(drawn, size)

            for write in (_encode_h264, _encode_vp8):
                target = write(dest, frames(), fps, size)
                if target is not None and _playable(target):
                    return target
                if target is not None:
                    target.unlink(missing_ok=True)
            return None
        except Exception:  # noqa: BLE001 — o vídeo anotado é um extra
            return None


def _encode_h264(dest: Path, frames, fps: float, size: tuple[int, int]) -> Path | None:
    """Manda os quadros (JPEG) para o ffmpeg e grava H.264/yuv420p com o índice no começo do arquivo."""
    if shutil.which("ffmpeg") is None:
        return None
    target = dest.with_suffix(".mp4")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps}", "-c:v", "mjpeg", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart", str(target),
    ]  # fmt: skip
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        code = proc.wait(timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
        proc.wait()
        code = 1
    if code != 0:
        target.unlink(missing_ok=True)
        return None
    return target


def _encode_vp8(dest: Path, frames, fps: float, size: tuple[int, int]) -> Path | None:
    target = dest.with_suffix(".webm")
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        writer.release()
        target.unlink(missing_ok=True)
        return None
    for frame in frames:
        writer.write(frame)
    writer.release()
    return target


def _playable(path: Path) -> bool:
    """Reabre o arquivo e decodifica um quadro: cabeçalho sem vídeo dentro não vira player quebrado na tela."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        cap.release()


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


FALLBACK_WEIGHTS = ("yolo11s-pose.pt", "yolov8n-pose.pt")


def _load_yolo(settings: Settings) -> YoloPoseEstimator:
    """Tenta os pesos configurados e, se não carregarem (ultralytics antigo, sem download), os anteriores."""
    last: Exception | None = None
    for weights in dict.fromkeys((settings.yolo_pose_weights, *FALLBACK_WEIGHTS)):
        try:
            return YoloPoseEstimator(weights, imgsz=settings.video_imgsz, refine_imgsz=settings.video_refine_imgsz)
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 — pesos desconhecidos para a versão instalada ou sem rede
            last = exc
    raise last if last is not None else VideoError("Nenhum peso de pose disponível.")


def _build_narrator(settings: Settings):
    if not settings.video_narrative_enabled or settings.mock_azure:
        return None
    if not (settings.azure_openai_endpoint.strip() and settings.azure_openai_key.strip() and settings.azure_openai_vision_deployment.strip()):
        return None
    from vitaltrace.video.narrative import AzureOpenAISceneNarrator

    return AzureOpenAISceneNarrator(
        settings.azure_openai_endpoint, settings.azure_openai_key,
        settings.azure_openai_vision_deployment, settings.azure_openai_api_version,
    )  # fmt: skip


def build_video_analyzer(settings: Settings) -> VideoAnalyzer:
    backend = settings.video_backend.lower()
    factory: Callable[[], PersonEstimator] = MotionBBoxEstimator
    if backend != "motion":
        try:
            yolo = _load_yolo(settings)  # carrega o modelo uma única vez
            factory = lambda: yolo  # noqa: E731
        except Exception as exc:  # ultralytics ausente ou pesos indisponíveis
            if backend == "yolo":
                raise VideoError(f"YOLO-pose indisponível: {exc}") from exc
    return VideoAnalyzer(
        factory,
        settings.video_sample_fps,
        settings.video_max_seconds,
        settings.video_detail_fps,
        width=settings.video_analysis_width,
        refine=settings.video_refine_pose,
        annotate=settings.video_annotated,
        narrator=_build_narrator(settings),
    )
