"""SQLite database connection manager."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.db.schema import SCHEMA_SQL
from src.utils.logger import get_logger

log = get_logger(__name__)


class Database:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call .initialize() first.")
        return self._conn

    def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        log.info("database_initialized", path=self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("database_closed")

    def __enter__(self) -> Database:
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
