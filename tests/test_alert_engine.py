from medmonitor.alerts.engine import build_audio_alerts
from medmonitor.audio.critical_terms import find_critical_terms
from medmonitor.audio.models import SentimentResult, SpeechMetrics, TextAnalysis
from medmonitor.models import Severity


def build(settings, text="", sentiment=None, metrics=None):
    return build_audio_alerts(
        "p1",
        find_critical_terms(text),
        TextAnalysis(sentiment=sentiment or SentimentResult()),
        metrics or SpeechMetrics(),
        settings,
    )


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
