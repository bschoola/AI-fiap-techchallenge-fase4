from medmonitor.audio.metrics import compute_speech_metrics
from medmonitor.audio.models import SpeechSegment, Transcription


def make(segments, text):
    return Transcription(
        text=text,
        language="pt-BR",
        segments=[SpeechSegment(start=s, end=e, text="x") for s, e in segments],
    )


def test_wpm_and_pause_ratio():
    # 20 palavras em 10 s de intervalo, com 2 s de pausa => 120 wpm, 20% de pausa
    t = make([(0.0, 4.0), (6.0, 10.0)], " ".join(["palavra"] * 20))
    m = compute_speech_metrics(t)
    assert m.words_per_minute == 120.0
    assert m.pause_ratio == 0.2
    assert m.reliable is True


def test_short_clip_is_not_reliable():
    t = make([(0.0, 2.0)], "poucas palavras aqui")
    assert compute_speech_metrics(t).reliable is False


def test_no_segments_returns_only_word_count():
    m = compute_speech_metrics(Transcription(text="uma duas", language="pt-BR"))
    assert m.words == 2
    assert m.words_per_minute is None
