"""Tabelas do portal."""

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    crm: Mapped[str] = mapped_column(String(40))
    specialty: Mapped[str] = mapped_column(String(80), default="")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    name_search: Mapped[str] = mapped_column(String(160), index=True)  # sem acentos, minúsculo
    birth_date: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(1))
    record_number: Mapped[str] = mapped_column(String(20), unique=True)
    reason: Mapped[str] = mapped_column(String(200))  # motivo do acompanhamento
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    last_consult_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    # stable | attention | critical — recalculado pela fusão de risco
    risk_level: Mapped[str] = mapped_column(String(12), default="stable")
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)

    consultations: Mapped[list["Consultation"]] = relationship(
        back_populates="patient", order_by="Consultation.occurred_at.desc()"
    )


class Consultation(Base):
    """Uma consulta: texto da anamnese + sinais aferidos naquele momento."""

    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("doctors.id"), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    anamnesis: Mapped[str] = mapped_column(Text)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    systolic: Mapped[float | None] = mapped_column(Float, nullable=True)
    diastolic: Mapped[float | None] = mapped_column(Float, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="consultations")
    doctor: Mapped[Doctor | None] = relationship()


class VitalSign(Base):
    """Série temporal de sinais vitais (monitor/oxímetro/aferições)."""

    __tablename__ = "vital_signs"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    heart_rate: Mapped[float] = mapped_column(Float)
    spo2: Mapped[float] = mapped_column(Float)
    resp_rate: Mapped[float] = mapped_column(Float)
    systolic: Mapped[float] = mapped_column(Float)
    diastolic: Mapped[float] = mapped_column(Float)
    temperature: Mapped[float] = mapped_column(Float)


class Prescription(Base):
    """Cada linha é uma prescrição vigente por um período; mudar a dose cria outra linha."""

    __tablename__ = "prescriptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    drug: Mapped[str] = mapped_column(String(80))
    dose_mg: Mapped[float] = mapped_column(Float)
    frequency: Mapped[str] = mapped_column(String(40))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)


class Analysis(Base):
    """Resultado persistido de uma análise de áudio, vídeo ou de anomalias."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # audio | video | anomaly
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class AlertRow(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    source: Mapped[str] = mapped_column(String(20))
    severity: Mapped[str] = mapped_column(String(10))
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(14), index=True, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(120), nullable=True)

    patient: Mapped[Patient] = relationship()
