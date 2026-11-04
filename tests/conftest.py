import pytest

from medmonitor.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        mock_azure=True,
        database_path=str(tmp_path / "alerts.db"),
    )
