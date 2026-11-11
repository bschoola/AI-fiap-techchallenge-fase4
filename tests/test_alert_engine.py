from vitaltrace.alerts.engine import build_audio_alerts
from vitaltrace.audio.critical_terms import find_critical_terms
from vitaltrace.audio.models import (
    ProsodyMetrics,
    SentimentResult,
    SpeechMetrics,
    TextAnalysis,
    VoiceTone,
)
from vitaltrace.models import Severity


def build(settings, text="", sentiment=None, metrics=None, prosody=None, voice_tone=None):
    return build_audio_alerts(
        "p1",
        find_critical_terms(text),
        TextAnalysis(sentiment=sentiment or SentimentResult()),
        metrics or SpeechMetrics(),
        settings,
        prosody,
        voice_tone,
    )


# Valores medidos em data/samples/consulta_falta_de_ar.wav e consulta_normal.wav
BREATHLESS = ProsodyMetrics(
    duration_s=83.4, speech_s=37.8, pause_ratio=0.542, pause_count=42, long_pause_count=24,
    long_pauses_per_min=17.4, median_pause_s=1.06, max_pause_s=2.25, mean_phrase_s=0.88,
    pitch_variation_st=2.3, reliable=True,
)  # fmt: skip
NORMAL = ProsodyMetrics(
    duration_s=59.2, speech_s=41.8, pause_ratio=0.285, pause_count=32, long_pause_count=0,
    long_pauses_per_min=0.0, median_pause_s=0.49, max_pause_s=0.96, mean_phrase_s=1.27,
    pitch_variation_st=4.3, reliable=True,
)  # fmt: skip
SLOW = SpeechMetrics(words=80, duration_s=83, words_per_minute=58, pause_ratio=0.2, reliable=True)


def test_critical_term_creates_critical_alert(settings):
    alerts = build(settings, "Estou com falta de ar e muito cansado")
    assert alerts[0].severity == Severity.CRITICAL
    assert "falta de ar" in alerts[0].title
    # cansaço é da categoria "geral", falta de ar é "respiratório": dois alertas
    assert len(alerts) == 2


def test_negated_term_creates_no_alert(settings):
    assert build(settings, "Não tenho falta de ar") == []


def test_terms_in_same_category_are_grouped(settings):
    alerts = build(settings, "Tenho tosse e chiado no peito")
    assert len(alerts) == 1
    assert alerts[0].severity == Severity.MEDIUM


def test_negative_sentiment_alert(settings):
    sentiment = SentimentResult(label="negative", positive=0, neutral=0.1, negative=0.9)
    alerts = build(settings, sentiment=sentiment)
    assert [a.severity for a in alerts] == [Severity.LOW]


def test_slow_speech_alert_only_when_reliable(settings):
    slow = SpeechMetrics(words=20, duration_s=30, words_per_minute=40, pause_ratio=0.1, reliable=True)
    assert [a.severity for a in build(settings, metrics=slow)] == [Severity.MEDIUM]

    unreliable = slow.model_copy(update={"reliable": False})
    assert build(settings, metrics=unreliable) == []


def test_alerts_sorted_by_severity(settings):
    metrics = SpeechMetrics(words=20, duration_s=30, words_per_minute=40, pause_ratio=0.8, reliable=True)
    alerts = build(settings, "dor no peito", metrics=metrics)
    ranks = [a.severity.rank for a in alerts]
    assert ranks == sorted(ranks, reverse=True)


def test_breathless_speech_alone_is_high_and_replaces_isolated_alerts(settings):
    # O paciente nunca diz "falta de ar": o alerta vem só de COMO ele fala
    alerts = build(settings, "Eu vim da recepção até aqui e parei duas vezes", metrics=SLOW, prosody=BREATHLESS)

    assert len(alerts) == 1
    assert alerts[0].severity == Severity.HIGH
    assert "Fala entrecortada" in alerts[0].title
    assert "monótono" in alerts[0].description
    assert alerts[0].evidence["corroboration"] == []


def test_breathless_speech_confirmed_by_reported_complaint_is_critical(settings):
    alerts = build(settings, "Estou muito cansada", metrics=SLOW, prosody=BREATHLESS)
    fused = next(a for a in alerts if "Fala entrecortada" in a.title)

    assert fused.severity == Severity.CRITICAL
    assert "cansaço relatado" in fused.description


def test_breathless_speech_confirmed_by_voice_tone_is_critical(settings):
    tone = VoiceTone(respiratory_effort="intenso", fatigue="moderado", source="gpt-audio")
    alerts = build(settings, metrics=SLOW, prosody=BREATHLESS, voice_tone=tone)

    assert [a.severity for a in alerts] == [Severity.CRITICAL]
    assert "tom de voz avaliado por IA" in alerts[0].description


def test_negated_complaint_does_not_confirm(settings):
    alerts = build(settings, "Não tenho falta de ar", metrics=SLOW, prosody=BREATHLESS)
    assert [a.severity for a in alerts] == [Severity.HIGH]


def test_normal_prosody_creates_no_alert(settings):
    fluent = SpeechMetrics(words=150, duration_s=59, words_per_minute=152, pause_ratio=0.1, reliable=True)
    assert build(settings, metrics=fluent, prosody=NORMAL) == []


def test_unreliable_prosody_is_ignored(settings):
    short = BREATHLESS.model_copy(update={"reliable": False})
    assert build(settings, prosody=short) == []


def test_acoustic_pause_ratio_overrides_transcript_estimate(settings):
    # Muita pausa, mas sem o padrão completo (pausas curtas): só o alerta isolado de pausas
    pausy = NORMAL.model_copy(update={"pause_ratio": 0.6})
    alerts = build(settings, metrics=SLOW.model_copy(update={"words_per_minute": 120}), prosody=pausy)

    assert [a.title for a in alerts] == ["Pausas frequentes na fala"]
    assert "60%" in alerts[0].description


def test_voice_tone_alone_never_exceeds_medium(settings):
    tone = VoiceTone(respiratory_effort="intenso", fatigue="intenso", summary="Ofegante.", source="gpt-audio")
    alerts = build(settings, prosody=NORMAL, voice_tone=tone)

    assert [a.severity for a in alerts] == [Severity.MEDIUM]
    assert "avaliação por IA" in alerts[0].title


def test_mild_voice_tone_creates_no_alert(settings):
    tone = VoiceTone(respiratory_effort="leve", fatigue="moderado", source="gpt-audio")
    assert build(settings, prosody=NORMAL, voice_tone=tone) == []
