"""Anomalias na evolução das prescrições (mudanças de dose, novos fármacos, polifarmácia)."""

from collections import defaultdict
from datetime import date
from typing import Protocol

from vitaltrace.anomaly.models import Finding
from vitaltrace.models import Severity


class PrescriptionLike(Protocol):
    drug: str
    dose_mg: float
    frequency: str
    start_date: date
    end_date: date | None


def _fmt_dose(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def detect_prescription_anomalies(
    items: list[PrescriptionLike], today: date | None = None
) -> list[Finding]:
    today = today or date.today()
    by_drug: dict[str, list[PrescriptionLike]] = defaultdict(list)
    for item in items:
        by_drug[item.drug].append(item)

    findings: list[Finding] = []
    for drug, rows in by_drug.items():
        rows.sort(key=lambda r: r.start_date)
        changes = [
            (prev, cur)
            for prev, cur in zip(rows, rows[1:])
            if cur.dose_mg != prev.dose_mg
        ]
        increases = [
            (prev, cur)
            for prev, cur in changes
            if cur.dose_mg > prev.dose_mg and (today - cur.start_date).days <= 60
        ]
        if increases:
            prev, cur = increases[-1]
            ratio = cur.dose_mg / prev.dose_mg
            days_ago = (today - cur.start_date).days
            repeated = len(increases) >= 2
            severity = (
                Severity.HIGH
                if ratio >= 2 or repeated
                else Severity.MEDIUM
                if ratio >= 1.5
                else Severity.LOW
            )
            if severity.rank >= Severity.LOW.rank:
                detail = (
                    f"{len(increases)} aumentos de dose em 60 dias"
                    if repeated
                    else "aumento de dose"
                )
                findings.append(
                    Finding(
                        source="prescription",
                        detector="dose_change",
                        severity=severity,
                        title=(
                            f"{drug}: dose de {_fmt_dose(prev.dose_mg)} mg para "
                            f"{_fmt_dose(cur.dose_mg)} mg ({ratio:.1f}×)"
                        ),
                        description=(
                            f"{detail.capitalize()}; o último foi há {days_ago} dia(s). "
                            "Mudanças rápidas de dose costumam indicar resposta clínica insuficiente."
                        ),
                        evidence={
                            "drug": drug,
                            "ratio": round(ratio, 2),
                            "increases_60d": len(increases),
                            "days_ago": days_ago,
                        },
                    )
                )
        if len(rows) == 1 and 0 <= (today - rows[0].start_date).days <= 7:
            r = rows[0]
            findings.append(
                Finding(
                    source="prescription",
                    detector="new_drug",
                    severity=Severity.LOW,
                    title=f"{drug} iniciado "
                    + ("hoje" if r.start_date == today else f"há {(today - r.start_date).days} dia(s)"),
                    description=f"Novo fármaco ({_fmt_dose(r.dose_mg)} mg, {r.frequency}). "
                    "Acompanhar resposta e eventos adversos.",
                    evidence={"drug": drug},
                )
            )

    active = {
        r.drug for r in items if r.end_date is None or r.end_date >= today
    }
    if len(active) >= 5:
        findings.append(
            Finding(
                source="prescription",
                detector="polypharmacy",
                severity=Severity.LOW,
                title=f"Polifarmácia: {len(active)} medicamentos ativos",
                description="Revisar interações e necessidade de cada item.",
                evidence={"active": sorted(active)},
            )
        )
    findings.sort(key=lambda f: f.severity.rank, reverse=True)
    return findings
