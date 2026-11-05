"""Métricas simples de fala derivadas dos segmentos da transcrição."""

from vitaltrace.audio.models import SpeechMetrics, Transcription

MIN_WORDS = 10
MIN_DURATION_S = 5.0


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

    speaking = sum(max(0.0, s.end - s.start) for s in segments)
    pause_ratio = min(1.0, max(0.0, 1.0 - speaking / span))
    wpm = words / (span / 60.0)
    return SpeechMetrics(
        words=words,
        duration_s=round(span, 2),
        words_per_minute=round(wpm, 1),
        pause_ratio=round(pause_ratio, 3),
        reliable=words >= MIN_WORDS and span >= MIN_DURATION_S,
    )
