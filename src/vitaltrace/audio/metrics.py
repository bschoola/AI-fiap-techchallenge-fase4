"""Métricas simples de fala derivadas dos segmentos da transcrição."""

from vitaltrace.audio.models import SpeechMetrics, Transcription

MIN_WORDS = 10
MIN_DURATION_S = 5.0
WORD_GAP_S = 0.25  # silêncio entre palavras acima disso conta como pausa


def compute_speech_metrics(transcription: Transcription) -> SpeechMetrics:
    """Ritmo de fala (palavras/min) e proporção de pausas.

    Heurística para sinalizar possível fadiga/disartria; não é um diagnóstico.
    """
    words = len(transcription.text.split())
    segments = transcription.segments
    if not segments:
        return SpeechMetrics(words=words)

    start = min(s.start for s in segments)
    end = max(s.end for s in segments)
    span = end - start
    if span <= 0:
        return SpeechMetrics(words=words)

    # Os segmentos do Azure são frases inteiras e engolem as pausas internas. Havendo marcação
    # por palavra, o tempo de fala é a soma das palavras mais os intervalos curtos entre elas.
    timings = sorted(transcription.words, key=lambda w: w.start)
    articulation_wpm = None
    if len(timings) >= 2:
        speaking = sum(max(0.0, w.end - w.start) for w in timings)
        speaking += sum(
            gap
            for a, b in zip(timings, timings[1:])
            if 0 < (gap := b.start - a.end) < WORD_GAP_S
        )
        speaking = min(speaking, span)
        if speaking > 0:
            articulation_wpm = round(len(timings) / (speaking / 60.0), 1)
    else:
        speaking = sum(max(0.0, s.end - s.start) for s in segments)
    pause_ratio = min(1.0, max(0.0, 1.0 - speaking / span))
    wpm = words / (span / 60.0)
    return SpeechMetrics(
        words=words,
        duration_s=round(span, 2),
        words_per_minute=round(wpm, 1),
        pause_ratio=round(pause_ratio, 3),
        articulation_wpm=articulation_wpm,
        reliable=words >= MIN_WORDS and span >= MIN_DURATION_S,
    )
