"""Sinais de alerta no texto das anamneses (sem depender do Azure)."""

import re

from vitaltrace.anomaly.models import Finding
from vitaltrace.audio.critical_terms import find_critical_terms, normalize
from vitaltrace.models import Severity

_WORSENING = re.compile(r"\b(piora|piorou|progressiv\w+|agravament\w+|descompensa\w+)\b")


def detect_anamnesis_anomalies(notes_newest_first: list[str]) -> list[Finding]:
    """Analisa a anamnese mais recente e a evolução do texto nas últimas consultas."""
    if not notes_newest_first:
        return []
    findings: list[Finding] = []
    latest = notes_newest_first[0]

    by_category: dict[str, list] = {}
    for match in find_critical_terms(latest):
        if not match.negated:
            by_category.setdefault(match.category, []).append(match)
    for category, items in by_category.items():
        severity = max((m.severity for m in items), key=lambda s: s.rank)
        terms = sorted({m.term.lower() for m in items})
        findings.append(
            Finding(
                source="anamnesis",
                detector="critical_terms",
                severity=severity,
                title=f"Anamnese menciona ({category}): {', '.join(terms)}",
                description="Termo encontrado na última consulta registrada.",
                evidence={"category": category, "terms": terms},
            )
        )

    worsening = sum(1 for n in notes_newest_first[:3] if _WORSENING.search(normalize(n)))
    if worsening >= 2:
        findings.append(
            Finding(
                source="anamnesis",
                detector="worsening_language",
                severity=Severity.MEDIUM,
                title="Relatos de piora em consultas seguidas",
                description=f"{worsening} das últimas {min(3, len(notes_newest_first))} anamneses "
                "descrevem piora ou evolução progressiva.",
                evidence={"consultations": worsening},
            )
        )
    return findings
