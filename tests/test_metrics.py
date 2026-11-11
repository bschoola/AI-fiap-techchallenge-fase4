from vitaltrace.audio.metrics import compute_speech_metrics
from vitaltrace.audio.models import SpeechSegment, Transcription, WordTiming
from vitaltrace.audio.speech_service import parse_word_timings


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


def test_word_timings_reveal_pauses_inside_a_segment():
    # Um único segmento de 20 s (o Azure devolve a frase inteira), mas a pessoa fala
    # 10 palavras de 0,4 s separadas por 1,6 s de silêncio: 80% de pausa.
    t = make([(0.0, 20.0)], " ".join(["palavra"] * 10))
    t.words = [WordTiming(word="palavra", start=i * 2.0, end=i * 2.0 + 0.4) for i in range(10)]
    m = compute_speech_metrics(t)

    assert m.pause_ratio == 0.8
    assert m.words_per_minute == 30.0
    assert m.articulation_wpm == 150.0  # devagar por causa das pausas, não da articulação


def test_short_gaps_between_words_are_not_pauses():
    t = make([(0.0, 5.0)], " ".join(["palavra"] * 10))
    t.words = [WordTiming(word="palavra", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(10)]
    m = compute_speech_metrics(t)

    assert m.pause_ratio < 0.05


def test_parse_word_timings_from_azure_json():
    raw = '{"NBest": [{"Words": [{"Word": "bom", "Offset": 5000000, "Duration": 3000000}, {"Word": "dia", "Offset": 9000000, "Duration": 2000000}, {"Word": "x"}]}]}'
    words = parse_word_timings(raw)

    assert [(w.word, w.start, w.end) for w in words] == [("bom", 0.5, 0.8), ("dia", 0.9, 1.1)]
    assert parse_word_timings("") == []
    assert parse_word_timings("não é json") == []
    assert parse_word_timings('{"NBest": []}') == []
