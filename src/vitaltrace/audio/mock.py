"""Serviços simulados para desenvolver/testar sem Azure (MOCK_AZURE=true).

O transcritor simulado IGNORA o conteúdo do áudio e devolve sempre um texto fixo.
Use apenas para validar o fluxo; para a demonstração final use o Azure.
"""

from pathlib import Path

from vitaltrace.audio.models import (
    SentimentResult,
    SpeechSegment,
    TextAnalysis,
    Transcription,
)

DEMO_SEGMENTS = (
    (0.5, 6.0, "Doutor, estou com muita falta de ar e uma dor no peito desde ontem."),
    (7.0, 12.0, "Também me sinto muito cansado e com tontura."),
)

_NEGATIVE_HINTS = ("falta", "dor", "cansad", "mal", "ruim", "tontur", "febre")


class MockTranscriber:
    def __init__(self, language: str = "pt-BR") -> None:
        self._language = language

    def transcribe(self, wav_path: Path) -> Transcription:
        segments = [SpeechSegment(start=s, end=e, text=t) for s, e, t in DEMO_SEGMENTS]
        return Transcription(
            text=" ".join(s.text for s in segments),
            language=self._language,
            segments=segments,
        )


class MockTextAnalyzer:
    def analyze(self, text: str) -> TextAnalysis:
        lowered = text.lower()
        hits = sum(hint in lowered for hint in _NEGATIVE_HINTS)
        negative = min(0.95, 0.15 + 0.2 * hits) if text.strip() else 0.0
        positive = 0.0 if hits else (0.6 if text.strip() else 0.0)
        neutral = max(0.0, 1.0 - negative - positive)
        label = max(
            (("negative", negative), ("positive", positive), ("neutral", neutral)),
            key=lambda pair: pair[1],
        )[0]
        phrases = list(dict.fromkeys(w.strip(".,;!?") for w in text.split() if len(w) > 6))
        return TextAnalysis(
            sentiment=SentimentResult(
                label=label, positive=positive, neutral=neutral, negative=negative
            ),
            key_phrases=phrases[:5],
        )
