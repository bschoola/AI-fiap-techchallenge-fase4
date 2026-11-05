"""Anomalias em séries temporais de sinais vitais.

Três camadas complementares, todas leves o bastante para rodar em CPU:
 1. regras clínicas (limiares) — o que o médico espera ver;
 2. desvio robusto (mediana/MAD) em relação à linha de base do próprio paciente;
 3. tendência (regressão linear) e Isolation Forest multivariado para padrões incomuns.
Os limiares são pontos de partida didáticos, não critérios diagnósticos.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from vitaltrace.anomaly.models import Finding
from vitaltrace.models import Severity

SIGNALS = ["heart_rate", "spo2", "resp_rate", "systolic", "diastolic", "temperature"]
LABELS = {
    "heart_rate": "Frequência cardíaca",
    "spo2": "SpO₂",
    "resp_rate": "Frequência respiratória",
    "systolic": "Pressão sistólica",
    "diastolic": "Pressão diastólica",
    "temperature": "Temperatura",
}
UNITS = {"heart_rate": "bpm", "spo2": "%", "resp_rate": "irpm", "systolic": "mmHg",
         "diastolic": "mmHg", "temperature": "°C"}  # fmt: skip
# menor variação plausível por sinal: evita z-scores enormes quando a série é muito estável
SIGMA_FLOOR = {"heart_rate": 3.0, "spo2": 0.8, "resp_rate": 1.0, "systolic": 4.0,
               "diastolic": 3.0, "temperature": 0.15}  # fmt: skip

# (sinal, comparação, limiar, severidade, descrição)
RULES = [
    ("spo2", "lt", 90, Severity.CRITICAL, "saturação abaixo de 90%"),
    ("spo2", "lt", 92, Severity.HIGH, "saturação abaixo de 92%"),
    ("heart_rate", "gt", 130, Severity.CRITICAL, "taquicardia importante (> 130 bpm)"),
    ("heart_rate", "gt", 110, Severity.HIGH, "taquicardia (> 110 bpm)"),
    ("heart_rate", "lt", 40, Severity.CRITICAL, "bradicardia importante (< 40 bpm)"),
    ("heart_rate", "lt", 50, Severity.HIGH, "bradicardia (< 50 bpm)"),
    ("resp_rate", "gt", 28, Severity.CRITICAL, "taquipneia importante (> 28 irpm)"),
    ("resp_rate", "gt", 24, Severity.HIGH, "taquipneia (> 24 irpm)"),
    ("systolic", "gt", 180, Severity.CRITICAL, "pressão sistólica > 180 mmHg"),
    ("systolic", "gt", 160, Severity.HIGH, "pressão sistólica > 160 mmHg"),
    ("systolic", "lt", 80, Severity.CRITICAL, "hipotensão importante (< 80 mmHg)"),
    ("systolic", "lt", 90, Severity.HIGH, "hipotensão (< 90 mmHg)"),
    ("temperature", "gt", 39.5, Severity.CRITICAL, "febre alta (> 39,5 °C)"),
    ("temperature", "gt", 38.5, Severity.HIGH, "febre (> 38,5 °C)"),
    ("temperature", "gt", 37.8, Severity.MEDIUM, "febrícula (> 37,8 °C)"),
]


def _fmt(signal: str, value: float) -> str:
    text = f"{value:.1f}".replace(".", ",") if signal == "temperature" else f"{value:.0f}"
    unit = UNITS[signal]
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def _recent_mask(df: pd.DataFrame, hours: float) -> pd.Series:
    return df["ts"] >= df["ts"].max() - pd.Timedelta(hours=hours)


def rule_findings(df: pd.DataFrame, recent_hours: float = 48) -> list[Finding]:
    """Violações de limiar. A pior regra de cada sinal vira um achado só."""
    findings: list[Finding] = []
    recent = df[_recent_mask(df, recent_hours)]
    for signal in SIGNALS:
        best = None  # (severity rank, rule, matching rows)
        for sig, op, limit, severity, text in RULES:
            if sig != signal:
                continue
            rows = recent[recent[signal] < limit] if op == "lt" else recent[recent[signal] > limit]
            if rows.empty:
                continue
            if best is None or severity.rank > best[0]:
                best = (severity.rank, (severity, text, op), rows)
        if best is None:
            continue
        _, (severity, text, op), rows = best
        worst = rows[signal].min() if op == "lt" else rows[signal].max()
        last = rows.iloc[-1]
        findings.append(
            Finding(
                source="vitals",
                detector="rule",
                severity=severity,
                title=f"{LABELS[signal]} em {_fmt(signal, float(last[signal]))}: {text}",
                description=(
                    f"{len(rows)} aferição(ões) nas últimas {recent_hours:.0f} h fora do limite; "
                    f"pior valor {_fmt(signal, float(worst))}."
                ),
                ts=last["ts"].to_pydatetime(),
                evidence={"signal": signal, "count": int(len(rows)), "worst": float(worst)},
            )
        )
    return findings


def zscore_findings(df: pd.DataFrame, window_hours: float = 24) -> list[Finding]:
    """Compara o último dia com a linha de base do próprio paciente (mediana/MAD)."""
    recent_mask = _recent_mask(df, window_hours)
    baseline, recent = df[~recent_mask], df[recent_mask]
    if len(baseline) < 12 or recent.empty:
        return []
    findings: list[Finding] = []
    for signal in SIGNALS:
        median = float(baseline[signal].median())
        mad = float((baseline[signal] - median).abs().median()) * 1.4826
        sigma = max(mad, SIGMA_FLOOR[signal])
        z = (recent[signal] - median) / sigma
        mean_z = float(z.mean())
        if abs(mean_z) < 3.0 or (z.abs() > 3.0).sum() < 2:
            continue
        direction = "acima" if mean_z > 0 else "abaixo"
        findings.append(
            Finding(
                source="vitals",
                detector="zscore",
                severity=Severity.HIGH if abs(mean_z) >= 6 else Severity.MEDIUM,
                title=f"{LABELS[signal]} {direction} da linha de base do paciente",
                description=(
                    f"Média das últimas {window_hours:.0f} h: {_fmt(signal, float(recent[signal].mean()))} "
                    f"contra mediana habitual de {_fmt(signal, median)} (z robusto {mean_z:+.1f})."
                ),
                ts=recent.iloc[-1]["ts"].to_pydatetime(),
                evidence={"signal": signal, "z": round(mean_z, 2), "baseline": median},
            )
        )
    return findings


# variação mínima, em 3 dias, que vale como "tendência" (e se o sentido é de piora)
_TREND_RULES = {
    "spo2": (-2.5, "queda progressiva"),
    "heart_rate": (12.0, "aumento progressivo"),
    "systolic": (15.0, "aumento progressivo"),
    "resp_rate": (3.0, "aumento progressivo"),
}


def trend_findings(df: pd.DataFrame, days: float = 3) -> list[Finding]:
    window = df[_recent_mask(df, days * 24)]
    if len(window) < 8:
        return []
    x = (window["ts"] - window["ts"].iloc[0]).dt.total_seconds().to_numpy() / 86400
    findings: list[Finding] = []
    for signal, (min_change, label) in _TREND_RULES.items():
        y = window[signal].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(((y - fitted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
        r2 = 1 - ss_res / ss_tot
        change = slope * days
        worsening = change <= min_change if min_change < 0 else change >= min_change
        if worsening and r2 >= 0.3:
            findings.append(
                Finding(
                    source="vitals",
                    detector="trend",
                    severity=Severity.MEDIUM,
                    title=f"{LABELS[signal]}: {label} em {days:.0f} dias",
                    description=(
                        f"Variação estimada de {change:+.1f} {UNITS[signal]} em {days:.0f} dias "
                        f"(R² = {r2:.2f})."
                    ),
                    ts=window.iloc[-1]["ts"].to_pydatetime(),
                    evidence={"signal": signal, "change": round(float(change), 2), "r2": round(r2, 2)},
                )
            )
    return findings


def isolation_forest_findings(df: pd.DataFrame, window_hours: float = 24) -> list[Finding]:
    """Novidade multivariada: treina na linha de base e pontua o último dia."""
    recent_mask = _recent_mask(df, window_hours)
    baseline, recent = df[~recent_mask], df[recent_mask]
    if len(baseline) < 30 or recent.empty:
        return []
    # Pré-filtro barato: sem nenhum sinal minimamente fora do habitual, não vale treinar a floresta.
    medians = baseline[SIGNALS].median()
    sigmas = pd.Series({s: max((baseline[s] - medians[s]).abs().median() * 1.4826, SIGMA_FLOOR[s]) for s in SIGNALS})
    if (((recent[SIGNALS] - medians) / sigmas).abs().max().max()) < 1.8:
        return []
    scaler = StandardScaler().fit(baseline[SIGNALS])
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=0)
    model.fit(scaler.transform(baseline[SIGNALS]))
    scores = model.decision_function(scaler.transform(recent[SIGNALS]))
    outliers = int((scores < -0.05).sum())
    if outliers < 3:
        return []
    return [
        Finding(
            source="vitals",
            detector="isolation_forest",
            severity=Severity.MEDIUM,
            title="Combinação incomum de sinais vitais",
            description=(
                f"{outliers} de {len(recent)} aferições das últimas {window_hours:.0f} h destoam "
                "do padrão multivariado habitual deste paciente (Isolation Forest)."
            ),
            ts=recent.iloc[-1]["ts"].to_pydatetime(),
            evidence={"outliers": outliers, "min_score": round(float(scores.min()), 3)},
        )
    ]


def detect_vital_anomalies(df: pd.DataFrame) -> list[Finding]:
    if df.empty:
        return []
    df = df.sort_values("ts").reset_index(drop=True)
    findings = [
        *rule_findings(df),
        *zscore_findings(df),
        *trend_findings(df),
        *isolation_forest_findings(df),
    ]
    findings.sort(key=lambda f: f.severity.rank, reverse=True)
    return findings
