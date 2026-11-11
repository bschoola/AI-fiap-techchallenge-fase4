"""Prosódia: como a pessoa fala (pausas, fôlego, tom), medida direto na forma de onda.

A transcrição diz O QUE foi dito; aqui medimos COMO. Uma fala entrecortada, com trechos curtos
separados por pausas longas, é o padrão de quem está sem fôlego, mesmo que a pessoa nunca diga
"falta de ar". Só numpy, roda em CPU em bem menos de 1 s por minuto de áudio.

Heurística de triagem, não diagnóstico: os limiares ficam em `config.py`.
"""

import wave
from pathlib import Path

import numpy as np

from vitaltrace.audio.models import ProsodyMetrics

FRAME_S = 0.025
HOP_S = 0.010
SPEECH_BELOW_PEAK_DB = 30.0  # fala = até 30 dB abaixo dos trechos mais fortes
MIN_ABOVE_NOISE_DB = 6.0  # ...e pelo menos 6 dB acima do ruído de fundo
MERGE_GAP_S = 0.20  # silêncios menores que isso são articulação (oclusivas), não pausa
MIN_RUN_S = 0.10  # estalos mais curtos que isso não contam como fala
LONG_PAUSE_S = 1.0
MIN_SPEECH_S = 3.0
MIN_DURATION_S = 5.0
F0_MIN_HZ, F0_MAX_HZ = 70.0, 400.0
VOICING_THRESHOLD = 0.45  # autocorrelação normalizada mínima para considerar o quadro sonoro


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """WAV PCM 16 bits -> float32 em [-1, 1], mono."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"WAV com {8 * width} bits não suportado (esperado PCM 16 bits)")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        return np.empty((0, size), dtype=x.dtype)
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop]


def _speech_runs(db: np.ndarray) -> list[tuple[int, int]]:
    """Trechos de fala (em quadros), já sem micro-silêncios e sem estalos."""
    peak = float(np.percentile(db, 95))
    noise = float(np.percentile(db, 5))
    threshold = max(peak - SPEECH_BELOW_PEAK_DB, noise + MIN_ABOVE_NOISE_DB)
    if peak - noise < MIN_ABOVE_NOISE_DB * 2:  # sem contraste: tudo ruído ou tudo fala
        return []
    active = np.concatenate([[False], db > threshold, [False]])
    edges = np.flatnonzero(active[1:] != active[:-1])
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist()))

    merged: list[list[int]] = []
    for start, end in runs:
        if merged and (start - merged[-1][1]) * HOP_S < MERGE_GAP_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if (e - s) * HOP_S >= MIN_RUN_S]


def _pitch_variation(x: np.ndarray, rate: int, runs: list[tuple[int, int]]) -> tuple[float | None, float | None]:
    """F0 por autocorrelação nos trechos de fala -> (mediana em Hz, desvio em semitons)."""
    size = int(0.040 * rate)
    hop = int(0.020 * rate)
    lag_min, lag_max = int(rate / F0_MAX_HZ), int(rate / F0_MIN_HZ)
    if lag_max >= size:
        return None, None
    window = np.hanning(size)
    f0: list[float] = []
    for start, end in runs:
        seg = x[int(start * HOP_S * rate): int(end * HOP_S * rate)]
        frames = _frames(seg, size, hop)
        if not len(frames):
            continue
        frames = (frames - frames.mean(axis=1, keepdims=True)) * window
        spec = np.fft.rfft(frames, n=2 * size, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :size]
        energy = ac[:, 0]
        ok = energy > 1e-8
        ac = ac[ok] / energy[ok, None]
        if not len(ac):
            continue
        lags = np.argmax(ac[:, lag_min:lag_max], axis=1) + lag_min
        strength = ac[np.arange(len(ac)), lags]
        f0.extend((rate / lags[strength >= VOICING_THRESHOLD]).tolist())
    if len(f0) < 20:
        return None, None
    values = np.asarray(f0)
    median = float(np.median(values))
    semitones = 12.0 * np.log2(values / median)
    semitones = semitones[np.abs(semitones) < 12.0]  # descarta erros de oitava
    return round(median, 1), round(float(semitones.std()), 2)


def analyze_prosody(wav_path: Path) -> ProsodyMetrics:
    x, rate = read_wav_mono(wav_path)
    duration = len(x) / rate if rate else 0.0
    size, hop = int(FRAME_S * rate), int(HOP_S * rate)
    frames = _frames(x, size, hop)
    if not len(frames):
        return ProsodyMetrics(duration_s=round(duration, 2))

    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    db = 20.0 * np.log10(rms + 1e-10)
    runs = _speech_runs(db)
    if not runs:
        return ProsodyMetrics(duration_s=round(duration, 2))

    run_s = np.array([(e - s) * HOP_S for s, e in runs])
    pauses = np.array([(runs[i + 1][0] - runs[i][1]) * HOP_S for i in range(len(runs) - 1)])
    speech = float(run_s.sum())
    span = (runs[-1][1] - runs[0][0]) * HOP_S  # do início da 1ª fala ao fim da última
    long_pauses = int((pauses >= LONG_PAUSE_S).sum()) if len(pauses) else 0
    f0_median, f0_sd = _pitch_variation(x, rate, runs)

    return ProsodyMetrics(
        duration_s=round(duration, 2),
        speech_s=round(speech, 2),
        pause_ratio=round(min(1.0, max(0.0, 1.0 - speech / span)), 3) if span > 0 else None,
        pause_count=int(len(pauses)),
        long_pause_count=long_pauses,
        long_pauses_per_min=round(long_pauses / (span / 60.0), 1) if span > 0 else None,
        pauses_per_min=round(len(pauses) / (span / 60.0), 1) if span > 0 else None,
        median_pause_s=round(float(np.median(pauses)), 2) if len(pauses) else None,
        max_pause_s=round(float(pauses.max()), 2) if len(pauses) else None,
        mean_phrase_s=round(float(run_s.mean()), 2),
        pitch_median_hz=f0_median,
        pitch_variation_st=f0_sd,
        reliable=duration >= MIN_DURATION_S and speech >= MIN_SPEECH_S,
    )
