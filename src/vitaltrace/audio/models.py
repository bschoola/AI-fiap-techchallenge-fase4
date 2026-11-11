"""Modelos do pipeline de áudio."""

from pydantic import BaseModel, Field

from vitaltrace.models import Alert, Severity


class SpeechSegment(BaseModel):
    start: float  # segundos
    end: float
    text: str


class WordTiming(BaseModel):
    word: str
    start: float  # segundos
    end: float


class Transcription(BaseModel):
    text: str
    language: str
    segments: list[SpeechSegment] = Field(default_factory=list)
    # Marcação por palavra (Azure Speech com request_word_level_timestamps); vazia se indisponível
    words: list[WordTiming] = Field(default_factory=list)


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
    # Velocidade só enquanto a pessoa fala (sem as pausas). Separa "fala devagar" de
    # "fala normal, mas para muito". Exige marcação por palavra.
    articulation_wpm: float | None = None
    # False quando o trecho é curto demais para tirar conclusões sobre o ritmo de fala
    reliable: bool = False


class ProsodyMetrics(BaseModel):
    """Como a pessoa fala, medido na forma de onda (ver audio/prosody.py)."""

    duration_s: float | None = None
    speech_s: float | None = None
    pause_ratio: float | None = None  # fração do tempo em silêncio, entre a 1ª e a última fala
    pause_count: int = 0
    long_pause_count: int = 0  # pausas de 1 s ou mais
    long_pauses_per_min: float | None = None
    pauses_per_min: float | None = None
    median_pause_s: float | None = None
    max_pause_s: float | None = None
    mean_phrase_s: float | None = None  # duração média de cada trecho falado sem parar
    pitch_median_hz: float | None = None
    pitch_variation_st: float | None = None  # desvio do tom em semitons (baixo = fala monótona)
    reliable: bool = False


class VoiceTone(BaseModel):
    """Impressão do tom de voz dada por um modelo de áudio (opcional; ver audio/voice_tone.py)."""

    respiratory_effort: str = "nenhum"  # nenhum | leve | moderado | intenso
    fatigue: str = "nenhum"  # nenhum | leve | moderado | intenso
    emotional_tone: str = ""  # ex.: calmo, ansioso, assustado, abatido
    summary: str = ""
    source: str = ""  # nome do deployment, ou "mock"


class AudioAnalysis(BaseModel):
    patient_id: str
    transcription: Transcription
    text_analysis: TextAnalysis
    critical_terms: list[CriticalTermMatch] = Field(default_factory=list)
    metrics: SpeechMetrics = Field(default_factory=SpeechMetrics)
    prosody: ProsodyMetrics | None = None  # None quando não há arquivo de áudio (texto colado)
    voice_tone: VoiceTone | None = None  # None quando VOICE_TONE_ENABLED=false ou a chamada falhou
    warnings: list[str] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
