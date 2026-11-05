"""Consultas e gravações usadas pelo portal."""

from datetime import datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from vitaltrace.audio.critical_terms import normalize
from vitaltrace.db.orm import (
    AlertRow,
    Analysis,
    Consultation,
    Doctor,
    Patient,
    Prescription,
    VitalSign,
)
from vitaltrace.models import Alert, AlertStatus, utcnow


def _naive_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


# ------------------------------------------------------------------ médicos
def get_doctor_by_email(s: Session, email: str) -> Doctor | None:
    return s.scalar(select(Doctor).where(func.lower(Doctor.email) == email.strip().lower()))


def get_doctor(s: Session, doctor_id: int) -> Doctor | None:
    return s.get(Doctor, doctor_id)


# ---------------------------------------------------------------- pacientes
def count_patients(s: Session) -> int:
    return s.scalar(select(func.count()).select_from(Patient)) or 0


def list_patients(s: Session, *, query: str = "", limit: int | None = None) -> list[Patient]:
    stmt = select(Patient).order_by(Patient.last_consult_at.desc())
    q = normalize(query.strip())
    if q:
        stmt = stmt.where(Patient.name_search.contains(q))
    if limit:
        stmt = stmt.limit(limit)
    return list(s.scalars(stmt))


def get_patient(s: Session, patient_id: int) -> Patient | None:
    return s.get(Patient, patient_id)


# ----------------------------------------------------------------- alertas
def _to_row(alert: Alert) -> AlertRow:
    return AlertRow(
        id=alert.id,
        patient_id=int(alert.patient_id),
        source=alert.source,
        severity=alert.severity.value,
        title=alert.title,
        description=alert.description,
        evidence=alert.evidence,
        status=alert.status.value,
        created_at=_naive_utc(alert.created_at),
    )


def add_alerts(s: Session, alerts: list[Alert], *, dedupe: bool = True) -> list[AlertRow]:
    """Grava alertas. Com `dedupe`, ignora o que já está em aberto (mesmo paciente/origem/título)."""
    saved: list[AlertRow] = []
    for alert in alerts:
        if dedupe:
            exists = s.scalar(
                select(AlertRow.id).where(
                    AlertRow.patient_id == int(alert.patient_id),
                    AlertRow.source == alert.source,
                    AlertRow.title == alert.title,
                    AlertRow.status == AlertStatus.OPEN.value,
                )
            )
            if exists:
                continue
        row = _to_row(alert)
        s.add(row)
        saved.append(row)
    s.flush()
    return saved


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def list_alerts(
    s: Session,
    *,
    status: AlertStatus | None = AlertStatus.OPEN,
    patient_id: int | None = None,
    limit: int = 100,
) -> list[AlertRow]:
    stmt = select(AlertRow).options(joinedload(AlertRow.patient))
    if status is not None:
        stmt = stmt.where(AlertRow.status == status.value)
    if patient_id is not None:
        stmt = stmt.where(AlertRow.patient_id == patient_id)
    rows = list(s.scalars(stmt.order_by(AlertRow.created_at.desc()).limit(limit)))
    rows.sort(key=lambda r: _SEVERITY_ORDER.get(r.severity, 9))  # estável: mantém a data
    return rows


def acknowledge_alert(s: Session, alert_id: str, doctor_name: str) -> AlertRow | None:
    row = s.get(AlertRow, alert_id)
    if row is None:
        return None
    if row.status != AlertStatus.ACKNOWLEDGED.value:
        row.status = AlertStatus.ACKNOWLEDGED.value
        row.acknowledged_at = _naive_utc(utcnow())
        row.acknowledged_by = doctor_name
    return row


def open_alert_counts(s: Session) -> dict[int, int]:
    rows = s.execute(
        select(AlertRow.patient_id, func.count())
        .where(AlertRow.status == AlertStatus.OPEN.value)
        .group_by(AlertRow.patient_id)
    )
    return {pid: n for pid, n in rows}


# ------------------------------------------------------- dados clínicos
def vitals_frame(s: Session, patient_id: int) -> pd.DataFrame:
    rows = s.execute(
        select(
            VitalSign.ts,
            VitalSign.heart_rate,
            VitalSign.spo2,
            VitalSign.resp_rate,
            VitalSign.systolic,
            VitalSign.diastolic,
            VitalSign.temperature,
        )
        .where(VitalSign.patient_id == patient_id)
        .order_by(VitalSign.ts)
    ).all()
    return pd.DataFrame(
        rows,
        columns=["ts", "heart_rate", "spo2", "resp_rate", "systolic", "diastolic", "temperature"],
    )


def prescriptions(s: Session, patient_id: int) -> list[Prescription]:
    return list(
        s.scalars(
            select(Prescription)
            .where(Prescription.patient_id == patient_id)
            .order_by(Prescription.drug, Prescription.start_date)
        )
    )


def consultations(s: Session, patient_id: int) -> list[Consultation]:
    return list(
        s.scalars(
            select(Consultation)
            .options(joinedload(Consultation.doctor))
            .where(Consultation.patient_id == patient_id)
            .order_by(Consultation.occurred_at.desc())
        )
    )


def add_consultation(
    s: Session,
    *,
    patient: Patient,
    doctor_id: int | None,
    anamnesis: str,
    spo2: float | None = None,
    heart_rate: float | None = None,
    systolic: float | None = None,
    diastolic: float | None = None,
    occurred_at: datetime | None = None,
) -> Consultation:
    when = _naive_utc(occurred_at or utcnow())
    consult = Consultation(
        patient_id=patient.id,
        doctor_id=doctor_id,
        occurred_at=when,
        anamnesis=anamnesis.strip(),
        spo2=spo2,
        heart_rate=heart_rate,
        systolic=systolic,
        diastolic=diastolic,
    )
    s.add(consult)
    patient.last_consult_at = when
    s.flush()
    return consult


# --------------------------------------------------------------- análises
def save_analysis(
    s: Session, patient_id: int, kind: str, summary: str, payload: dict
) -> Analysis:
    row = Analysis(
        patient_id=patient_id,
        kind=kind,
        created_at=_naive_utc(utcnow()),
        summary=summary,
        payload=payload,
    )
    s.add(row)
    s.flush()
    return row


def latest_analyses(s: Session, patient_id: int, kind: str, limit: int = 5) -> list[Analysis]:
    return list(
        s.scalars(
            select(Analysis)
            .where(Analysis.patient_id == patient_id, Analysis.kind == kind)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
        )
    )


def update_risk(s: Session, patient: Patient, level: str, score: float) -> None:
    patient.risk_level = level
    patient.risk_score = round(score, 1)
