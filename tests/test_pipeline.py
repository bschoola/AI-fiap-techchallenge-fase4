import wave
from pathlib import Path

import numpy as np

from vitaltrace.audio.models import SpeechSegment, TextAnalysis, Transcription, VoiceTone
from vitaltrace.audio.pipeline import AudioPipeline
from vitaltrace.audio.voice_tone import VoiceToneError
from vitaltrace.models import Severity


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


class NeutralTranscriber:
    """Paciente que não diz nenhum termo crítico: só a forma de falar pode gerar alerta."""

    def transcribe(self, wav_path: Path) -> Transcription:
        text = "Eu vim da recepção até aqui e parei duas vezes no corredor"
        return Transcription(text=text, language="pt-BR", segments=[SpeechSegment(start=0, end=20, text=text)])


def breathless_wav(path: Path) -> Path:
    rate = 16000
    t = np.arange(int(0.8 * rate)) / rate
    burst = 0.3 * np.sin(2 * np.pi * 200 * t)
    x = np.concatenate([np.concatenate([burst, np.zeros(int(1.2 * rate))]) for _ in range(10)])
    x = x + 0.001 * np.random.default_rng(0).standard_normal(len(x))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return path


def test_pipeline_flags_breathless_speech_without_critical_terms(settings, tmp_path):
    pipeline = AudioPipeline(NeutralTranscriber(), FakeAnalyzer(), settings)
    result = pipeline.analyze_audio("p1", breathless_wav(tmp_path / "ofegante.wav"))

    assert result.critical_terms == []
    assert result.prosody.long_pause_count == 9
    assert result.alerts[0].severity == Severity.HIGH
    assert "Fala entrecortada" in result.alerts[0].title


def test_pipeline_voice_tone_confirms_and_failure_only_warns(settings, tmp_path):
    wav = breathless_wav(tmp_path / "ofegante.wav")

    class Tone:
        def analyze(self, wav_path, transcript, prosody):
            return VoiceTone(respiratory_effort="intenso", source="fake")

    class BrokenTone:
        def analyze(self, wav_path, transcript, prosody):
            raise VoiceToneError("HTTP 500")

    ok = AudioPipeline(NeutralTranscriber(), FakeAnalyzer(), settings, Tone()).analyze_audio("p1", wav)
    assert ok.voice_tone.respiratory_effort == "intenso"
    assert ok.alerts[0].severity == Severity.CRITICAL

    broken = AudioPipeline(NeutralTranscriber(), FakeAnalyzer(), settings, BrokenTone()).analyze_audio("p1", wav)
    assert broken.voice_tone is None
    assert "HTTP 500" in broken.warnings[0]
    assert broken.alerts[0].severity == Severity.HIGH  # a análise segue sem o tom de voz


def test_pipeline_survives_unreadable_wav(settings, tmp_path):
    bad = tmp_path / "ruim.wav"
    bad.write_bytes(b"isto nao e um wav")
    result = AudioPipeline(FakeTranscriber(), FakeAnalyzer(), settings).analyze_audio("p1", bad)

    assert result.prosody is None
    assert result.warnings
    assert result.alerts[0].severity == Severity.CRITICAL  # o termo crítico continua valendo
