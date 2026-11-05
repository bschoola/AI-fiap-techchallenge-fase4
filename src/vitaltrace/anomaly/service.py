"""Orquestra os detectores para um paciente, calcula o risco e gera os alertas."""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from vitaltrace.anomaly.anamnesis import detect_anamnesis_anomalies
from vitaltrace.anomaly.models import Finding
from vitaltrace.anomaly.movement import detect_movement_anomalies
from vitaltrace.anomaly.prescriptions import detect_prescription_anomalies
from vitaltrace.anomaly.vitals import detect_vital_anomalies
from vitaltrace.db import repos
from vitaltrace.db.orm import Patient
from vitaltrace.fusion.risk import RiskAssessment, compute_risk, correlate
from vitaltrace.fusion.summary import DISCLAIMER, build_summary
from vitaltrace.models import Alert, Severity


@dataclass
class AnomalyReport:
    patient_id: int
    findings: list[Finding]
    risk: RiskAssessment
    summary: str
    disclaimer: str = DISCLAIMER
    new_alerts: int = 0
    generated_at: datetime = field(default_factory=datetime.now)


def _open_modal_findings(s: Session, patient_id: int) -> list[Finding]:
    """Alertas de áudio/vídeo ainda em aberto entram no risco como achados dessas modalidades."""
    findings = []
    for row in repos.list_alerts(s, patient_id=patient_id):
        if row.source in ("audio", "video"):
            findings.append(
                Finding(
                    source=row.source,
                    detector="alert",
                    severity=Severity(row.severity),
                    title=row.title,
                    description=row.description,
                    evidence=row.evidence or {},
                )
            )
    return findings


def collect_findings(s: Session, patient: Patient, now: datetime | None = None) -> list[Finding]:
    now = now or datetime.now()
    findings: list[Finding] = []
    findings += detect_vital_anomalies(repos.vitals_frame(s, patient.id))
    findings += detect_prescription_anomalies(repos.prescriptions(s, patient.id), today=now.date())
    history = [a.payload for a in reversed(repos.latest_analyses(s, patient.id, "video", limit=6))]
    findings += detect_movement_anomalies(history)
    notes = [c.anamnesis for c in repos.consultations(s, patient.id)]
    findings += detect_anamnesis_anomalies(notes)
    findings += _open_modal_findings(s, patient.id)
    findings += correlate(findings)
    findings.sort(key=lambda f: f.severity.rank, reverse=True)
    return findings


def alerts_from_findings(patient_id: int, findings: list[Finding]) -> list[Alert]:
    """Só achados de severidade alta ou crítica viram alertas; os demais ficam no relatório."""
    return [
        f.to_alert(patient_id)
        for f in findings
        if f.detector != "alert" and f.severity.rank >= Severity.HIGH.rank
    ]


def analyze_patient(
    s: Session, patient: Patient, *, persist: bool = True, now: datetime | None = None
) -> AnomalyReport:
    findings = collect_findings(s, patient, now)
    risk = compute_risk(findings)
    summary = build_summary(patient.name, risk, findings)
    report = AnomalyReport(patient.id, findings, risk, summary)
    if persist:
        created = repos.add_alerts(s, alerts_from_findings(patient.id, findings))
        report.new_alerts = len(created)
        repos.update_risk(s, patient, risk.level, risk.score)
        repos.save_analysis(
            s,
            patient.id,
            "anomaly",
            summary,
            {
                "risk": {"score": risk.score, "level": risk.level, "by_source": risk.by_source},
                "findings": [f.model_dump(mode="json") for f in findings],
            },
        )
    return report
