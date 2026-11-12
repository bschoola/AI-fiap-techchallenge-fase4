import shutil

import pytest
from fastapi.testclient import TestClient

from vitaltrace.config import Settings
from vitaltrace.db import Database
from vitaltrace.db.seed import DEMO_EMAIL, DEMO_PASSWORD, seed_database
from vitaltrace.web.app import create_app


@pytest.fixture(scope="session")
def seeded_file(tmp_path_factory):
    """Banco SQLite semeado uma única vez (a semeadura roda os detectores nos 30 pacientes)."""
    path = tmp_path_factory.mktemp("seed") / "template.db"
    database = Database(f"sqlite:///{path}")
    database.create_all()
    seed_database(database)
    database.engine.dispose()
    return path


@pytest.fixture(scope="session")
def db(seeded_file) -> Database:
    """Somente leitura: compartilhado entre testes."""
    return Database(f"sqlite:///{seeded_file}")


@pytest.fixture
def settings(tmp_path, seeded_file) -> Settings:
    """Cada teste recebe uma cópia própria do banco semeado."""
    copy = tmp_path / "vitaltrace.db"
    shutil.copy(seeded_file, copy)
    return Settings(
        _env_file=None,
        mock_azure=True,
        database_url=f"sqlite:///{copy}",
        video_backend="motion",
        video_annotated_dir=str(tmp_path / "video_annotated"),
        seed_on_startup=False,
    )


@pytest.fixture
def client(settings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def logged(client) -> TestClient:
    r = client.post("/login", data={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}, follow_redirects=False)
    assert r.status_code == 303
    return client
