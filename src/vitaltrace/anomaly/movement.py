"""Anomalias de movimento a partir do histórico de análises de vídeo do paciente."""

from typing import Any

import numpy as np

from vitaltrace.anomaly.models import Finding
from vitaltrace.models import Severity


def _mobility(payload: dict[str, Any]) -> float | None:
    value = (payload.get("metrics") or {}).get("mobility_index")
    return float(value) if value is not None else None


def detect_movement_anomalies(history: list[dict[str, Any]]) -> list[Finding]:
    """`history` em ordem cronológica (mais antigo → mais recente), cada item um payload de vídeo."""
    findings: list[Finding] = []
    values = [m for m in (_mobility(p) for p in history) if m is not None]

    if len(values) >= 3:
        baseline = float(np.mean(values[:-1]))
        latest = values[-1]
        drop = (baseline - latest) / baseline if baseline > 0 else 0.0
        if drop >= 0.25:
            findings.append(
                Finding(
                    source="movement",
                    detector="mobility_drop",
                    severity=Severity.HIGH if drop >= 0.4 else Severity.MEDIUM,
                    title=f"Queda de {drop:.0%} no índice de mobilidade",
                    description=(
                        f"Índice atual {latest:.2f} contra média anterior de {baseline:.2f} "
                        f"({len(values) - 1} sessões)."
                    ),
                    evidence={"latest": latest, "baseline": round(baseline, 3), "drop": round(drop, 3)},
                )
            )

    if history:
        events = (history[-1].get("events") or [])
        falls = [e for e in events if e.get("type") == "fall_suspected"]
        if falls:
            findings.append(
                Finding(
                    source="movement",
                    detector="fall",
                    severity=Severity.CRITICAL,
                    title="Possível queda detectada no vídeo mais recente",
                    description=f"{len(falls)} evento(s) de queda suspeita na última análise.",
                    evidence={"events": falls[:3]},
                )
            )
    return findings
