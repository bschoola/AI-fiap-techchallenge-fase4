from datetime import date, timedelta
from types import SimpleNamespace as NS

from vitaltrace.anomaly.anamnesis import detect_anamnesis_anomalies
from vitaltrace.anomaly.models import Finding
from vitaltrace.anomaly.movement import detect_movement_anomalies
from vitaltrace.anomaly.prescriptions import detect_prescription_anomalies
from vitaltrace.fusion.risk import compute_risk, correlate
from vitaltrace.models import Severity

TODAY = date(2026, 11, 5)


def rx(drug, dose, start_days_ago, end_days_ago=None, freq="1x/dia"):
    return NS(
        drug=drug, dose_mg=dose, frequency=freq,
        start_date=TODAY - timedelta(days=start_days_ago),
        end_date=None if end_days_ago is None else TODAY - timedelta(days=end_days_ago),
    )  # fmt: skip


def test_doubled_dose_is_high():
    f = detect_prescription_anomalies([rx("Furosemida", 40, 60, 3), rx("Furosemida", 80, 3)], TODAY)
    assert f[0].severity == Severity.HIGH and f[0].evidence["ratio"] == 2.0


def test_repeated_increases_are_high_even_if_small():
    items = [rx("X", 10, 55, 40), rx("X", 12, 40, 10), rx("X", 14, 10)]
    f = detect_prescription_anomalies(items, TODAY)
    assert f[0].severity == Severity.HIGH and f[0].evidence["increases_60d"] == 2


def test_old_change_and_dose_reduction_are_ignored():
    assert detect_prescription_anomalies([rx("X", 20, 300, 200), rx("X", 40, 200)], TODAY) == []
    assert detect_prescription_anomalies([rx("X", 80, 20, 10), rx("X", 40, 10)], TODAY) == []


def test_new_drug_and_polypharmacy():
    f = detect_prescription_anomalies([rx("Nova", 5, 2)], TODAY)
    assert f[0].detector == "new_drug"
    many = [rx(f"d{i}", 10, 100) for i in range(5)]
    assert any(x.detector == "polypharmacy" for x in detect_prescription_anomalies(many, TODAY))


def video(mobility, falls=0):
    return {"metrics": {"mobility_index": mobility},
            "events": [{"type": "fall_suspected"}] * falls}  # fmt: skip


def test_mobility_drop_detected():
    f = detect_movement_anomalies([video(0.6), video(0.62), video(0.3)])
    assert f and f[0].severity == Severity.HIGH


def test_stable_mobility_and_short_history_are_quiet():
    assert detect_movement_anomalies([video(0.6), video(0.62), video(0.58)]) == []
    assert detect_movement_anomalies([video(0.6), video(0.1)]) == []


def test_fall_in_latest_video_is_critical():
    f = detect_movement_anomalies([video(0.6), video(0.5, falls=1)])
    assert f[0].severity == Severity.CRITICAL


def test_anamnesis_terms_and_negation():
    f = detect_anamnesis_anomalies(["Refere falta de ar e dor no peito."])
    assert {x.evidence["category"] for x in f} == {"respiratório", "cardiovascular"}
    assert detect_anamnesis_anomalies(["Nega falta de ar. Sem dor no peito."]) == []


def test_anamnesis_worsening_language():
    f = detect_anamnesis_anomalies(["Piora da dispneia.", "Evolução progressiva do edema.", "Estável."])
    assert any(x.detector == "worsening_language" for x in f)


def finding(source, severity, detector="rule", **evidence):
    return Finding(source=source, detector=detector, severity=severity, title="t", description="d", evidence=evidence)


def test_risk_levels():
    assert compute_risk([]).level == "stable"
    assert compute_risk([finding("vitals", Severity.LOW)]).level == "stable"
    assert compute_risk([finding("vitals", Severity.MEDIUM)]).level == "stable"
    assert compute_risk([finding("vitals", Severity.HIGH)]).level == "attention"
    r = compute_risk([finding("vitals", Severity.CRITICAL)])
    assert r.level == "critical" and r.score == 40


def test_risk_caps_each_source_and_non_critical_score():
    many = [finding("vitals", Severity.HIGH) for _ in range(10)]
    assert compute_risk(many).by_source["Sinais vitais"] == 50
    both = many + [finding("prescription", Severity.HIGH), finding("movement", Severity.HIGH)]
    r = compute_risk(both)
    assert r.level == "attention" and r.score <= 89


def test_correlation_treatment_without_response():
    findings = [
        finding("prescription", Severity.HIGH, "dose_change", drug="Furosemida", days_ago=2),
        finding("vitals", Severity.MEDIUM, "trend", signal="spo2"),
    ]
    extra = correlate(findings)
    assert [e.detector for e in extra] == ["treatment_response"]
    assert correlate(findings[:1]) == []
