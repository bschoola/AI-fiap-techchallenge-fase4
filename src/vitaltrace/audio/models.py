"""Modelos do pipeline de áudio."""

from pydantic import BaseModel, Field

from vitaltrace.models import Alert, Severity


class SpeechSegment(BaseModel):
    start: float  # segundos
    end: float
    text: str


class Transcription(BaseModel):
    text: str
    language: str
    segments: list[SpeechSegment] = Field(default_factory=list)


class SentimentResult(BaseModel):
    label: str = "neutral"  # positive | neutral | negative
    positive: float = 0.0
    neutral: float = 1.0
    negative: float = 0.0


class TextAnalysis(BaseModel):
    sentiment: SentimentResult = Field(default_factory=SentimentResult)
    key_phrases: list[str] = Field(default_factory=list)


class CriticalTermMatch(BaseModel):
    term: str
    category: str
    severity: Severity
    negated: bool = False
    context: str = ""
    position: int = 0


class SpeechMetrics(BaseModel):
    words: int = 0
    duration_s: float | None = None
    words_per_minute: float | None = None
    pause_ratio: float | None = None
    # False quando o trecho é curto demais para tirar conclusões sobre o ritmo de fala
    reliable: bool = False


class AudioAnalysis(BaseModel):
    patient_id: str
    transcription: Transcription
    text_analysis: TextAnalysis
    critical_terms: list[CriticalTermMatch] = Field(default_factory=list)
    metrics: SpeechMetrics = Field(default_factory=SpeechMetrics)
    alerts: list[Alert] = Field(default_factory=list)
