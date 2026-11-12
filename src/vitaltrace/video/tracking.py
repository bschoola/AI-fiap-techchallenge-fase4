"""Quem é quem no vídeo: rastreia cada pessoa entre os quadros e escolhe o paciente.

Antes a pessoa analisada era decidida quadro a quadro ("a mais evidente"), e com o fisioterapeuta
em cena a análise trocava de corpo no meio do exercício. Agora:

1. `PoseTracker` dá a cada pessoa um número estável (associa as detecções de um quadro às do
   anterior por sobreposição da caixa e semelhança do esqueleto);
2. `summarize` resume cada pessoa rastreada (tempo em cena, tamanho, quanto se movimenta);
3. `choose` fixa UMA pessoa por trecho contínuo do vídeo: a indicada pelo médico ou, sem
   indicação, a que mais parece estar se exercitando. Depois de um corte de cena a identidade
   é reencontrada pela aparência (cores da roupa).

Sem dependência extra e determinístico: fácil de testar e de explicar.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vitaltrace.video import pose
from vitaltrace.video.models import PersonObservation

MAX_GAP_S = 1.5  # uma pessoa encoberta por até este tempo volta com o mesmo número
MAX_COST = 0.8
MIN_SHARE = 0.25  # fração do trecho em cena para ser candidata a paciente
MIN_FRAMES = 5
CHAIN_SIMILARITY = 0.6  # aparência mínima para dois pedaços serem a mesma pessoa
_ANGLES = [(11, 13, 15), (12, 14, 16), (5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12), (5, 11, 13), (6, 12, 14)]


def _iou(a: tuple, b: tuple) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(o: PersonObservation) -> tuple[float, float]:
    return (o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2


def _pose_distance(a: PersonObservation, b: PersonObservation, height: float) -> float | None:
    """Distância média entre os pontos visíveis nos dois esqueletos, em alturas do corpo."""
    if not a.keypoints or not b.keypoints:
        return None
    d = [
        np.hypot(p[0] - q[0], p[1] - q[1])
        for p, q in zip(a.keypoints, b.keypoints)
        if p[2] >= pose.MIN_CONF and q[2] >= pose.MIN_CONF
    ]
    return float(np.mean(d)) / height if len(d) >= 4 else None


def _cost(prev: PersonObservation, cur: PersonObservation, gap_s: float) -> float:
    height = (pose.bbox_height(prev) + pose.bbox_height(cur)) / 2
    iou = _iou(prev.bbox, cur.bbox)
    (px, py), (cx, cy) = _center(prev), _center(cur)
    shift = float(np.hypot(cx - px, cy - py)) / height
    if iou < 0.05 and shift > 0.6 * (1 + gap_s):
        return 9.0  # longe demais para ser a mesma pessoa
    size = abs(pose.bbox_height(cur) - pose.bbox_height(prev)) / height
    kd = _pose_distance(prev, cur, height)
    near = min(1.0, (kd if kd is not None else shift) / 0.5)
    return 0.45 * (1 - iou) + 0.45 * near + 0.10 * min(1.0, size)


@dataclass
class _Track:
    id: int
    last: PersonObservation
    hits: int = 1


class PoseTracker:
    """Associação gulosa ótima (húngaro) entre as pessoas de um quadro e as já rastreadas."""

    def __init__(self, max_gap_s: float = MAX_GAP_S):
        self._max_gap_s = max_gap_s
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Corte de cena: ninguém do trecho anterior continua. Os números não são reaproveitados."""
        self._tracks = []

    def update(self, people: list[PersonObservation], t: float) -> list[PersonObservation]:
        self._tracks = [tr for tr in self._tracks if t - tr.last.t <= self._max_gap_s]
        assigned: dict[int, int] = {}
        if self._tracks and people:
            cost = np.array([[_cost(tr.last, p, t - tr.last.t) for p in people] for tr in self._tracks])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= MAX_COST:
                    assigned[c] = r
        # Por eliminação: sobrou uma pessoa rastreada e uma detecção sem par. É o caso de uma QUEDA
        # (a caixa muda de forma de um quadro para o outro) e não pode virar "pessoa nova".
        free_tracks = [r for r in range(len(self._tracks)) if r not in assigned.values() and t - self._tracks[r].last.t <= 0.6]
        free_people = [c for c in range(len(people)) if c not in assigned]
        if len(free_tracks) == 1 and len(free_people) == 1:
            prev, cur = self._tracks[free_tracks[0]].last, people[free_people[0]]
            (px, py), (cx, cy) = _center(prev), _center(cur)
            if np.hypot(cx - px, cy - py) <= 1.5 * max(pose.bbox_height(prev), pose.bbox_height(cur)):
                assigned[free_people[0]] = free_tracks[0]
        out = []
        for i, p in enumerate(people):
            if i in assigned:
                track = self._tracks[assigned[i]]
                track.hits += 1
            else:
                track = _Track(self._next_id, p)
                self._next_id += 1
                self._tracks.append(track)
            p = p.model_copy(update={"track_id": track.id, "people": len(people)})
            track.last = p
            out.append(p)
        return out


MERGE_GAP_S = 10.0


def merge_fragments(
    frames: list[list[PersonObservation]], segments: list[int], hists: dict[int, np.ndarray] | None = None
) -> list[list[PersonObservation]]:
    """Junta pedaços da MESMA pessoa que o rastreador numerou duas vezes (ficou encoberta por mais tempo).

    Dois pedaços viram um quando estão no mesmo trecho, um começa depois de o outro acabar (até 10 s),
    reaparece perto de onde sumiu, com tamanho parecido e — havendo aparência — com as mesmas cores.
    Sem isso o paciente "fatiado" perdia para o terapeuta na escolha automática."""
    hists = hists if hists is not None else {}
    first: dict[int, PersonObservation] = {}
    last: dict[int, PersonObservation] = {}
    seg_of: dict[int, int] = {}
    for people, seg in zip(frames, segments):
        for p in people:
            if p.track_id is None:
                continue
            first.setdefault(p.track_id, p)
            last[p.track_id] = p
            seg_of[p.track_id] = seg
    alias: dict[int, int] = {}
    for b in sorted(first, key=lambda i: first[i].t):
        best, best_gap = None, MERGE_GAP_S
        for a in last:
            if a == b or a in alias or seg_of[a] != seg_of[b]:
                continue
            gap = first[b].t - last[a].t
            if not (0 < gap <= best_gap):
                continue
            ha, hb = pose.bbox_height(last[a]), pose.bbox_height(first[b])
            (ax, ay), (bx, by) = _center(last[a]), _center(first[b])
            if not (0.6 <= hb / ha <= 1.6) or np.hypot(bx - ax, by - ay) > max(ha, hb):
                continue
            sim = _similarity(hists.get(a), hists.get(b))
            if sim is not None and sim < CHAIN_SIMILARITY:
                continue
            best, best_gap = a, gap
        if best is not None:
            alias[b] = best
            last[best] = last[b]  # o pedaço absorvido estende o original
            if b in hists and best in hists:
                hists[best] = hists[best] + hists.pop(b)
    if not alias:
        return frames
    return [[p.model_copy(update={"track_id": alias.get(p.track_id, p.track_id)}) for p in people] for people in frames]


@dataclass
class TrackStats:
    id: int
    segment: int
    frames: int
    share: float  # fração dos quadros do trecho em que aparece
    height: float  # altura mediana da caixa / altura do quadro
    activity: float  # 0 (parada) a 1 (muito movimento articular ou deslocamento)
    centrality: float  # 1 = no centro do quadro
    first_t: float
    last_t: float
    score: float = 0.0
    similarity: float | None = None  # aparência em relação ao paciente do trecho principal
    hist: np.ndarray | None = field(default=None, repr=False)


def _activity(items: list[PersonObservation]) -> float:
    best = 0.0
    for a, b, c in _ANGLES:
        values = [v for v in (pose.joint_angle(o, a, b, c) for o in items) if v is not None]
        if len(values) >= MIN_FRAMES:
            best = max(best, float(np.percentile(values, 95) - np.percentile(values, 5)))
    xs = [_center(o)[0] for o in items]
    heights = [pose.bbox_height(o) for o in items]
    travel = (max(xs) - min(xs)) / float(np.median(heights)) if xs else 0.0
    return float(min(1.0, max(best / 90.0, 0.5 * travel)))


def summarize(
    frames: list[list[PersonObservation]], segments: list[int], hists: dict[int, np.ndarray] | None = None
) -> list[TrackStats]:
    """Um resumo por pessoa rastreada. `frames[i]` são as pessoas do quadro i (já com track_id)."""
    by_track: dict[int, list[PersonObservation]] = {}
    seg_of: dict[int, int] = {}
    seg_frames: dict[int, int] = {}
    for people, seg in zip(frames, segments):
        seg_frames[seg] = seg_frames.get(seg, 0) + 1
        for p in people:
            if p.track_id is None:
                continue
            by_track.setdefault(p.track_id, []).append(p)
            seg_of[p.track_id] = seg
    stats = []
    for tid, items in by_track.items():
        w, h = items[0].frame_size
        h, w = max(h, 1), max(w, 1)
        height = float(np.median([pose.bbox_height(o) for o in items])) / h
        cx = float(np.median([_center(o)[0] for o in items])) / w
        s = TrackStats(
            id=tid,
            segment=seg_of[tid],
            frames=len(items),
            share=len(items) / seg_frames[seg_of[tid]],
            height=min(1.0, height),
            activity=_activity(items),
            centrality=1.0 - min(1.0, abs(cx - 0.5) * 2) * 0.5,
            first_t=items[0].t,
            last_t=items[-1].t,
            hist=(hists or {}).get(tid),
        )
        # quem fica mais tempo em cena, maior no quadro, mais ao centro e se mexendo mais
        s.score = s.share * (0.4 + s.height) * (0.5 + s.activity) * s.centrality
        stats.append(s)
    return sorted(stats, key=lambda s: s.id)


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, float(a @ b) / denom) if denom > 0 else None


def main_segment(stats: list[TrackStats]) -> int | None:
    """Trecho com mais quadros de pessoa: é nele que o médico escolhe o paciente."""
    totals: dict[int, int] = {}
    for s in stats:
        totals[s.segment] = totals.get(s.segment, 0) + s.frames
    return max(totals, key=lambda k: (totals[k], -k)) if totals else None


def candidates(stats: list[TrackStats]) -> list[TrackStats]:
    """Pessoas do trecho principal que ficam tempo suficiente em cena, da mais provável para a menos."""
    seg = main_segment(stats)
    found = [s for s in stats if s.segment == seg and s.frames >= MIN_FRAMES and s.share >= MIN_SHARE]
    if found:
        tallest = max(s.height for s in found)
        found = [s for s in found if s.height >= 0.4 * tallest]  # gente pequena ao fundo não entra
    return sorted(found, key=lambda s: s.score, reverse=True)[:6]


def _chain(primary: TrackStats, others: list[TrackStats]) -> list[int]:
    """Outros números que são a MESMA pessoa no mesmo trecho (o rastreador a perdeu e reencontrou).

    Só entra quem não aparece ao mesmo tempo que os já escolhidos e, havendo aparência, se parece."""
    taken = [(primary.first_t, primary.last_t)]
    ids = [primary.id]
    for s in sorted(others, key=lambda s: s.score, reverse=True):
        if any(s.first_t <= b + 0.3 and a <= s.last_t + 0.3 for a, b in taken):
            continue
        sim = _similarity(primary.hist, s.hist)
        if sim is not None and sim < CHAIN_SIMILARITY:
            continue
        ids.append(s.id)
        taken.append((s.first_t, s.last_t))
    return ids


def choose(stats: list[TrackStats], prefer: int | None = None) -> dict[int, list[int]]:
    """Para cada trecho, os números que correspondem ao paciente. `prefer` = escolha do médico.

    O primeiro número de cada lista é o principal; os demais são a mesma pessoa reencontrada."""
    if not stats:
        return {}
    by_id = {s.id: s for s in stats}
    anchor = by_id.get(prefer) if prefer is not None else None
    if anchor is None:
        ranked = candidates(stats) or sorted(stats, key=lambda s: s.score, reverse=True)
        anchor = ranked[0]
    primaries = {anchor.segment: anchor}
    for seg in sorted({s.segment for s in stats} - {anchor.segment}):
        best, best_score = None, -1.0
        for s in (s for s in stats if s.segment == seg):
            s.similarity = _similarity(anchor.hist, s.hist)
            score = s.score * (0.3 + 0.7 * s.similarity if s.similarity is not None else 1.0)
            if score > best_score:
                best, best_score = s, score
        if best is not None:
            primaries[seg] = best
    return {
        seg: _chain(primary, [s for s in stats if s.segment == seg and s.id != primary.id])
        for seg, primary in primaries.items()
    }


def pick(
    frames: list[list[PersonObservation]], segments: list[int], chosen: dict[int, list[int]]
) -> list[PersonObservation | None]:
    """A linha do tempo de uma pessoa só: em cada quadro, o paciente daquele trecho (ou None)."""
    out: list[PersonObservation | None] = []
    for people, seg in zip(frames, segments):
        wanted = chosen.get(seg) or []
        out.append(next((p for p in people if p.track_id in wanted), None))
    return out
