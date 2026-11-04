from medmonitor.alerts.repository import AlertRepository
from medmonitor.models import Alert, AlertStatus, Severity


def make_alert(**kwargs) -> Alert:
    defaults = dict(
        patient_id="p1",
        source="audio",
        severity=Severity.HIGH,
        title="t",
        description="d",
        evidence={"k": "ç"},
    )
    return Alert(**{**defaults, **kwargs})


def test_add_list_and_roundtrip(tmp_path):
    repo = AlertRepository(tmp_path / "db" / "alerts.db")
    alert = make_alert()
    repo.add_many([alert])

    [stored] = repo.list()
    assert stored.id == alert.id
    assert stored.evidence == {"k": "ç"}
    assert stored.created_at == alert.created_at


def test_filter_by_status_and_patient(tmp_path):
    repo = AlertRepository(tmp_path / "alerts.db")
    a, b = make_alert(patient_id="p1"), make_alert(patient_id="p2")
    repo.add_many([a, b])
    repo.acknowledge(a.id)

    assert [x.id for x in repo.list(status=AlertStatus.OPEN)] == [b.id]
    assert [x.id for x in repo.list(patient_id="p1")] == [a.id]


def test_acknowledge(tmp_path):
    repo = AlertRepository(tmp_path / "alerts.db")
    alert = make_alert()
    repo.add_many([alert])

    updated = repo.acknowledge(alert.id)
    assert updated.status == AlertStatus.ACKNOWLEDGED
    assert updated.acknowledged_at is not None
    assert repo.acknowledge("inexistente") is None
