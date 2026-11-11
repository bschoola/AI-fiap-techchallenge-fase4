import wave

import numpy as np
import pytest

from vitaltrace.audio.prosody import analyze_prosody, read_wav_mono

RATE = 16000


def write_wav(path, samples, rate=RATE, width=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def synth(pattern, pitch=(180.0, 240.0)):
    """pattern = [(segundos de fala, segundos de pausa), ...] -> voz sintética com ruído de fundo."""
    rng = np.random.default_rng(0)
    parts = []
    for speech_s, pause_s in pattern:
        t = np.arange(int(speech_s * RATE)) / RATE
        f0 = np.linspace(pitch[0], pitch[1], len(t))
        phase = 2 * np.pi * np.cumsum(f0) / RATE
        voice = sum(np.sin(k * phase) / k for k in (1, 2, 3))
        parts.append(0.3 * voice)
        parts.append(np.zeros(int(pause_s * RATE)))
    x = np.concatenate(parts) + 0.001 * rng.standard_normal(sum(len(p) for p in parts))
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def test_breathless_pattern_is_measured(tmp_path):
    path = tmp_path / "ofegante.wav"
    write_wav(path, synth([(0.8, 1.2)] * 10))
    m = analyze_prosody(path)

    assert m.reliable is True
    assert m.pause_count == 9
    assert m.long_pause_count == 9
    assert m.pause_ratio == pytest.approx(0.57, abs=0.05)
    assert m.median_pause_s == pytest.approx(1.2, abs=0.1)
    assert m.mean_phrase_s == pytest.approx(0.8, abs=0.1)
    assert m.long_pauses_per_min > 20


def test_fluent_speech_has_few_pauses(tmp_path):
    path = tmp_path / "fluente.wav"
    write_wav(path, synth([(3.0, 0.4)] * 5))
    m = analyze_prosody(path)

    assert m.reliable is True
    assert m.long_pause_count == 0
    assert m.pause_ratio < 0.2
    assert m.mean_phrase_s == pytest.approx(3.0, abs=0.2)


def test_pitch_variation_separates_monotone_from_expressive(tmp_path):
    flat, lively = tmp_path / "monotono.wav", tmp_path / "expressivo.wav"
    write_wav(flat, synth([(2.0, 0.5)] * 4, pitch=(200.0, 204.0)))
    write_wav(lively, synth([(2.0, 0.5)] * 4, pitch=(150.0, 300.0)))
    a, b = analyze_prosody(flat), analyze_prosody(lively)

    assert a.pitch_median_hz == pytest.approx(202, abs=8)
    assert a.pitch_variation_st < 1.0 < b.pitch_variation_st


def test_silence_is_not_reliable(tmp_path):
    path = tmp_path / "silencio.wav"
    write_wav(path, np.zeros(RATE * 6, dtype=np.int16))
    m = analyze_prosody(path)

    assert m.reliable is False
    assert m.pause_ratio is None


def test_short_clip_is_not_reliable(tmp_path):
    path = tmp_path / "curto.wav"
    write_wav(path, synth([(1.0, 0.3)]))
    assert analyze_prosody(path).reliable is False


def test_non_16bit_wav_is_rejected(tmp_path):
    path = tmp_path / "8bits.wav"
    write_wav(path, np.zeros(RATE, dtype=np.uint8), width=1)
    with pytest.raises(ValueError):
        read_wav_mono(path)
