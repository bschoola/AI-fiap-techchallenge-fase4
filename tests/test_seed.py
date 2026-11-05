from sqlalchemy import func, select

from vitaltrace.db import repos
from vitaltrace.db.orm import Consultation, Patient, Prescription, VitalSign
from vitaltrace.db.seed import seed_database


def test_seed_creates_30_patients(db):
    with db.session() as s:
        assert repos.count_patients(s) == 30
        assert s.scalar(select(func.count()).select_from(VitalSign)) == 30 * 84
        assert s.scalar(select(func.count()).select_from(Consultation)) >= 30 * 3
        assert s.scalar(select(func.count()).select_from(Prescription)) >= 20


def test_seed_is_idempotent(db):
    assert seed_database(db) is False
    with db.session() as s:
        assert repos.count_patients(s) == 30


def test_seed_has_one_demo_doctor(db):
    with db.session() as s:
        doctor = repos.get_doctor_by_email(s, "RAFAEL.MENEZES@hospitalmodelo.demo")
        assert doctor is not None and doctor.name == "Dr. Rafael Menezes"
        assert doctor.password_hash.startswith("scrypt$")


def test_recent_patients_are_ordered_by_last_consultation(db):
    with db.session() as s:
        recent = repos.list_patients(s, limit=5)
        assert [p.name for p in recent[:2]] == ["Helena Martins Duarte", "Roberto Almeida Lima"]
        dates = [p.last_consult_at for p in repos.list_patients(s)]
        assert dates == sorted(dates, reverse=True)


def test_search_ignores_accents_and_case(db):
    with db.session() as s:
        assert [p.name for p in repos.list_patients(s, query="jose")] == ["José Antônio Ribeiro"]
        assert [p.name for p in repos.list_patients(s, query="  ANTONIO ")] == [
            "José Antônio Ribeiro",
            "Antônio Carlos Medeiros",
        ]
        assert repos.list_patients(s, query="zzz") == []


def test_risk_levels_come_from_detectors(db):
    with db.session() as s:
        levels = {p.name: p.risk_level for p in s.scalars(select(Patient))}
    assert levels["Helena Martins Duarte"] == "critical"
    assert levels["Roberto Almeida Lima"] == "attention"
    assert levels["Carlos Eduardo Pereira"] == "stable"
    # a maioria dos pacientes é estável: o detector não pode gerar falso positivo em massa
    assert sum(v == "stable" for v in levels.values()) >= 15
    assert sum(v == "critical" for v in levels.values()) == 1


def test_seed_opens_alerts_for_helena(db):
    with db.session() as s:
        alerts = repos.list_alerts(s)
        helena = [a for a in alerts if a.patient.name.startswith("Helena")]
        assert sum(a.severity == "critical" for a in helena) >= 2
        assert alerts[0].severity == "critical"  # mais graves primeiro
