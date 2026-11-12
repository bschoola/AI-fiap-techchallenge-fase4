"""Limpeza do esqueleto antes de medir ângulos.

Dois erros típicos da pose em vídeo real, que viravam "anomalia" falsa:

- troca de lado: de perfil ou de costas o modelo às vezes devolve o lado esquerdo no lugar do
  direito por alguns quadros, e a curva do "joelho esquerdo" passa a misturar as duas pernas;
- ponto que salta: com um membro encoberto o modelo chuta um ponto longe do corpo, e o osso
  "estica" de um quadro para o outro.

Nada aqui inventa posição: a troca é desfeita (os pontos já existiam) e o ponto implausível é
só marcado como não confiável (confiança 0), para ficar fora dos ângulos.
"""

import numpy as np

from vitaltrace.video import pose
from vitaltrace.video.models import PersonObservation

_FACE = [(1, 2), (3, 4)]
_GROUPS = {"arms": [(7, 8), (9, 10)], "legs": [(13, 14), (15, 16)]}
_BODY = [(5, 6), (11, 12)] + _GROUPS["arms"] + _GROUPS["legs"]
_BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
SWAP_LOOKBACK_S = 1.0
SWAP_RATIO = 0.5  # trocado precisa "encaixar" pelo menos 2x melhor que o original
SWAP_MIN_SEP = 0.08  # lados quase sobrepostos (perfil): não dá para saber, não mexe
SWAP_MIN_JUMP = 0.10  # e o original precisa ter saltado (alturas do corpo por par)
BONE_WINDOW_S = 2.0
BONE_STRETCH = 1.6


def _costs(prev, cur, pairs, height):
    straight = swapped = 0.0
    n = 0
    for l, r in pairs:
        pl, pr, cl, cr = prev[l], prev[r], cur[l], cur[r]
        if min(pl[2], pr[2], cl[2], cr[2]) < pose.MIN_CONF:
            continue
        if np.hypot(pl[0] - pr[0], pl[1] - pr[1]) < SWAP_MIN_SEP * height:
            continue
        straight += np.hypot(cl[0] - pl[0], cl[1] - pl[1]) + np.hypot(cr[0] - pr[0], cr[1] - pr[1])
        swapped += np.hypot(cl[0] - pr[0], cl[1] - pr[1]) + np.hypot(cr[0] - pl[0], cr[1] - pl[1])
        n += 1
    return straight, swapped, n


def _is_swapped(prev, cur, pairs, height, min_pairs) -> bool:
    straight, swapped, n = _costs(prev, cur, pairs, height)
    return n >= min_pairs and swapped < SWAP_RATIO * straight and straight / n > SWAP_MIN_JUMP * height


def _swap(kp: list, pairs) -> list:
    kp = list(kp)
    for l, r in pairs:
        kp[l], kp[r] = kp[r], kp[l]
    return kp


def fix_swaps(
    observations: list[PersonObservation | None], segments: list[int]
) -> tuple[list[PersonObservation | None], int]:
    """Desfaz trocas esquerda/direita comparando cada quadro com o anterior do mesmo trecho.

    Mantém a rotulagem do começo do trecho: o que importa para amplitude e simetria é que
    "esquerdo" seja sempre a mesma perna ao longo da gravação."""
    out = list(observations)
    fixed = 0
    prev_i: int | None = None
    for i, o in enumerate(observations):
        if o is None or not o.keypoints or len(o.keypoints) < 17:
            continue
        ref = out[prev_i] if prev_i is not None else None
        if ref is not None and segments[prev_i] == segments[i] and o.t - ref.t <= SWAP_LOOKBACK_S:
            height = (pose.bbox_height(ref) + pose.bbox_height(o)) / 2
            kp, changed = list(o.keypoints), False
            if _is_swapped(ref.keypoints, kp, _BODY, height, 3):
                kp, changed = _swap(kp, _BODY + _FACE), True
            else:  # só um par de membros trocado (pernas cruzando na marcha de perfil)
                for pairs in _GROUPS.values():
                    if _is_swapped(ref.keypoints, kp, pairs, height, 2):
                        kp, changed = _swap(kp, pairs), True
            if changed:
                out[i] = o.model_copy(update={"keypoints": kp})
                fixed += 1
        prev_i = i
    return out, fixed


def drop_implausible(
    observations: list[PersonObservation | None], segments: list[int]
) -> tuple[list[PersonObservation | None], int]:
    """Marca como não confiável a ponta de um osso que "esticou" de repente.

    Cada osso é comparado com a própria mediana nos 2 s vizinhos (mesmo trecho), então aproximar-se
    da câmera ou agachar não dispara nada. Só o comprimento MAIOR é suspeito: menor pode ser o
    membro apontando para a câmera."""
    n = len(observations)
    t = np.array([o.t if o is not None else np.nan for o in observations])
    seg = np.array(segments)
    conf = {i: [list(k) for k in o.keypoints] for i, o in enumerate(observations) if o is not None and o.keypoints and len(o.keypoints) >= 17}
    dropped = 0
    for a, b in _BONES:
        length = np.full(n, np.nan)
        for i, kp in conf.items():
            if kp[a][2] >= pose.MIN_CONF and kp[b][2] >= pose.MIN_CONF:
                length[i] = np.hypot(kp[a][0] - kp[b][0], kp[a][1] - kp[b][1])
        for i in np.flatnonzero(~np.isnan(length)):
            near = (seg == seg[i]) & (np.abs(t - t[i]) <= BONE_WINDOW_S) & ~np.isnan(length)
            near[i] = False
            if near.sum() < 4:
                continue
            median = float(np.median(length[near]))
            if median > 2 and length[i] > BONE_STRETCH * median:
                conf[i][b][2] = 0.0  # a ponta distal é a que costuma estar errada
                dropped += 1
    out = list(observations)
    if dropped:
        for i, kp in conf.items():
            out[i] = observations[i].model_copy(update={"keypoints": [tuple(k) for k in kp]})
    return out, dropped
