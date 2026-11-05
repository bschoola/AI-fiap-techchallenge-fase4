import numpy as np
import pandas as pd

from vitaltrace.anomaly.vitals import detect_vital_anomalies, rule_findings, trend_findings
from vitaltrace.models import Severity


def make_frame(n=84, seed=0, **overrides) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    ts = pd.date_range(end="2026-11-05 09:00", periods=n, freq="4h")
    df = pd.DataFrame(
        {
            "ts": ts,
            "heart_rate": 74 + rng.normal(0, 3, n),
            "spo2": 97 + rng.normal(0, 0.5, n),
            "resp_rate": 16 + rng.normal(0, 0.8, n),
            "systolic": 120 + rng.normal(0, 4, n),
            "diastolic": 78 + rng.normal(0, 3, n),
            "temperature": 36.6 + rng.normal(0, 0.1, n),
        }
    )
    for column, values in overrides.items():
        df.loc[df.index[-len(values):], column] = values
    return df.round({c: 1 for c in df.columns if c != "ts"})


def test_stable_series_has_no_findings():
    assert detect_vital_anomalies(make_frame()) == []


def test_low_spo2_rule_is_critical_below_90():
    f = rule_findings(make_frame(spo2=[96, 93, 89, 88]))
    assert f and f[0].severity == Severity.CRITICAL
    assert f[0].evidence["signal"] == "spo2"


def test_spo2_between_90_and_92_is_high():
    f = rule_findings(make_frame(spo2=[96, 93, 91, 91]))
    assert f[0].severity == Severity.HIGH


def test_old_violations_do_not_trigger_rules():
    df = make_frame()
    df.loc[2, "spo2"] = 80  # há semanas
    assert rule_findings(df) == []


def test_progressive_decline_triggers_trend_and_zscore():
    n = 84
    df = make_frame(spo2=list(np.linspace(97, 92.5, 24)), heart_rate=list(np.linspace(74, 100, 24)))
    kinds = {(f.detector, f.evidence.get("signal")) for f in detect_vital_anomalies(df)}
    assert ("trend", "spo2") in kinds
    assert ("zscore", "heart_rate") in kinds
    assert len(df) == n


def test_trend_needs_enough_points():
    assert trend_findings(make_frame(n=4)) == []


def test_empty_frame():
    assert detect_vital_anomalies(pd.DataFrame(columns=["ts"])) == []


def test_findings_are_sorted_by_severity():
    f = detect_vital_anomalies(make_frame(spo2=list(np.linspace(97, 88, 24))))
    ranks = [x.severity.rank for x in f]
    assert ranks == sorted(ranks, reverse=True)
