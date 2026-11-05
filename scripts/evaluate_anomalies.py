"""Avalia os detectores de sinais vitais em séries sintéticas com e sem deriva clínica.

Uso: python scripts/evaluate_anomalies.py [N]   (N = séries por perfil; padrão 150)
Os rótulos vêm do próprio gerador (stable/attention/critical), então é uma checagem de
sanidade dos limiares, não uma validação clínica.
"""

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vitaltrace.anomaly.vitals import detect_vital_anomalies  # noqa: E402
from vitaltrace.db.seed import generate_vitals  # noqa: E402
from vitaltrace.fusion.risk import compute_risk  # noqa: E402

COLS = ["ts", "heart_rate", "spo2", "resp_rate", "systolic", "diastolic", "temperature"]
REASONS = ["Hipertensão arterial", "Diabetes tipo 2", "DPOC", "Controle de insuficiência cardíaca", "Lombalgia"]


def main(n: int) -> None:
    now = datetime(2026, 11, 5, 9, 0)
    print(f"{'perfil':<10} {'n':>4}  estável  atenção  crítico   (nível calculado, %)")
    results = {}
    for profile in ("stable", "attention", "critical"):
        levels = Counter()
        for i in range(n):
            rows = generate_vitals(10_000 + i, REASONS[i % len(REASONS)], profile, now)
            df = pd.DataFrame([[getattr(r, c) for c in COLS] for r in rows], columns=COLS)
            levels[compute_risk(detect_vital_anomalies(df)).level] += 1
        results[profile] = levels
        pct = {k: 100 * levels[k] / n for k in ("stable", "attention", "critical")}
        print(f"{profile:<10} {n:>4}  {pct['stable']:6.1f}  {pct['attention']:7.1f}  {pct['critical']:7.1f}")
    fp = 1 - results["stable"]["stable"] / n
    tp_att = 1 - results["attention"]["stable"] / n
    tp_crit = results["critical"]["critical"] / n
    print(f"\nfalso alarme (estável marcado como atenção/crítico): {fp:.1%}")
    print(f"sensibilidade perfil 'atenção' (>= atenção): {tp_att:.1%}")
    print(f"sensibilidade perfil 'crítico' (= crítico): {tp_crit:.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
