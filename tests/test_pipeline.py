from pathlib import Path

from medmonitor.audio.models import SpeechSegment, TextAnalysis, Transcription
from medmonitor.audio.pipeline import AudioPipeline
from medmonitor.models import Severity


class FakeTranscriber:
    def transcribe(self, wav_path: Path) -> Transcription:
        return Transcription(
            text="Doutor estou com falta de ar",
            language="pt-BR",
            segments=[SpeechSegment(start=0, end=3, text="Doutor estou com falta de ar")],
        )


class FakeAnalyzer:
    def analyze(self, text: str) -> TextAnalysis:
        return TextAnalysis(key_phrases=["falta de ar"])


def test_pipeline_audio_produces_alerts(settings):
    pipeline = AudioPipeline(FakeTranscriber(), FakeAnalyzer(), settings)
    result = pipeline.analyze_audio("p1", Path("qualquer.wav"))

    assert result.transcription.text.endswith("falta de ar")
    assert result.alerts[0].severity == Severity.CRITICAL
    assert result.alerts[0].patient_id == "p1"


def test_pipeline_text_has_no_segments(settings):
    pipeline = AudioPipeline(FakeTranscriber(), FakeAnalyzer(), settings)
    result = pipeline.analyze_text("p1", "Estou bem, sem dor.")

    assert result.metrics.words_per_minute is None
    assert result.alerts == []
