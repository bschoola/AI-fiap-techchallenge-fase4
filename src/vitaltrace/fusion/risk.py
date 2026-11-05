"""Score de risco (0–100) e nível, a partir dos achados de todas as modalidades."""

from collections import defaultdict
from dataclasses import dataclass, field

from vitaltrace.anomaly.models import SOURCE_LABELS, Finding
from vitaltrace.models import Severity

POINTS = {
    Severity.CRITICAL: 40.0,
    Severity.HIGH: 25.0,
    Severity.MEDIUM: 12.0,
    Severity.LOW: 5.0,
    Severity.INFO: 0.0,
}
SOURCE_CAP = 50.0  # nenhuma modalidade sozinha passa disso: o score exige convergência
ATTENTION_SCORE = 15.0


@dataclass
class RiskAssessment:
    score: float
    level: str  # stable | attention | critical
    by_source: dict[str, float] = field(default_factory=dict)
    drivers: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LEVEL_LABELS[self.level]


LEVEL_LABELS = {"stable": "ESTÁVEL", "attention": "ATENÇÃO", "critical": "CRÍTICO"}


def compute_risk(findings: list[Finding]) -> RiskAssessment:
    per_source: dict[str, float] = defaultdict(float)
    for f in findings:
        per_source[f.source] += POINTS[f.severity]
    capped = {src: min(pts, SOURCE_CAP) for src, pts in per_source.items() if pts > 0}
    score = min(100.0, sum(capped.values()))

    has_critical = any(f.severity == Severity.CRITICAL for f in findings)
    has_high = any(f.severity == Severity.HIGH for f in findings)
    if has_critical:
        level = "critical"
    elif has_high or score >= ATTENTION_SCORE:
        level = "attention"
    else:
        level = "stable"

    if level != "critical":
        score = min(score, 89.0)  # 90+ fica reservado a quem tem algum achado crítico

    ordered = sorted(findings, key=lambda f: f.severity.rank, reverse=True)
    drivers = [f"{SOURCE_LABELS.get(f.source, f.source)}: {f.title}" for f in ordered[:4]]
    return RiskAssessment(
        score=round(score, 1),
        level=level,
        by_source={SOURCE_LABELS.get(k, k): round(v, 1) for k, v in capped.items()},
        drivers=drivers,
    )


def correlate(findings: list[Finding]) -> list[Finding]:
    """Achados que só aparecem ao cruzar modalidades."""
    extra: list[Finding] = []
    escalated = [
        f for f in findings
        if f.detector == "dose_change" and f.evidence.get("days_ago", 999) <= 14
    ]  # fmt: skip
    worsening_vitals = [
        f for f in findings
        if f.source == "vitals" and f.detector in ("trend", "zscore", "rule")
        and f.severity.rank >= Severity.MEDIUM.rank
    ]  # fmt: skip
    if escalated and worsening_vitals:
        drugs = sorted({f.evidence.get("drug", "") for f in escalated})
        extra.append(
            Finding(
                source="fusion",
                detector="treatment_response",
                severity=Severity.HIGH,
                title="Piora clínica apesar do ajuste de tratamento",
                description=(
                    f"Aumento recente de dose ({', '.join(drugs)}) e, ao mesmo tempo, sinais vitais "
                    "em deterioração. Considerar reavaliação presencial e revisão da conduta."
                ),
                evidence={"drugs": drugs, "vitals": [f.title for f in worsening_vitals[:3]]},
            )
        )

    reported = [f for f in findings if f.source in ("audio", "anamnesis")
                and f.severity.rank >= Severity.HIGH.rank]  # fmt: skip
    objective = [f for f in findings if f.source in ("vitals", "movement")
                 and f.severity.rank >= Severity.MEDIUM.rank]  # fmt: skip
    if reported and objective:
        extra.append(
            Finding(
                source="fusion",
                detector="symptom_vitals",
                severity=Severity.HIGH,
                title="Sintomas relatados coincidem com alterações medidas",
                description=(
                    "A queixa do paciente (voz/anamnese) é compatível com os desvios objetivos "
                    "de sinais vitais ou movimento, o que reforça a prioridade do caso."
                ),
                evidence={"reported": [f.title for f in reported[:2]],
                          "measured": [f.title for f in objective[:2]]},
            )
        )
    return extra
