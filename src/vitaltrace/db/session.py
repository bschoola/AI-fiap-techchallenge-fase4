"""Engine e fábrica de sessões."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from vitaltrace.db.orm import Base


class Database:
    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        kwargs: dict = {"future": True}
        if parsed.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False}
            if parsed.database in (None, "", ":memory:"):
                kwargs["poolclass"] = StaticPool  # memória compartilhada entre threads
            else:
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        else:
            kwargs["pool_pre_ping"] = True
        self.engine = create_engine(url, **kwargs)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
