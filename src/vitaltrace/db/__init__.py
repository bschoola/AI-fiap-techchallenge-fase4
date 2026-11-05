"""Camada de persistência (SQLAlchemy 2): Postgres no Docker, SQLite em dev/testes."""

from vitaltrace.db.session import Database

__all__ = ["Database"]
