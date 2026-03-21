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
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._migrate_model_results_add_blob()
        self._migrate_time_local_to_24h()
        self._migrate_clean_resolution_url_dots()
        self._migrate_daily_peaks_add_summary_high()
        self._migrate_bets_add_llm_source()
        self._conn.commit()
        log.info("database_initialized", path=self._db_path)

    def _migrate_model_results_add_blob(self) -> None:
        """Add model_blob BLOB column to model_results if missing (one-time idempotent)."""
        assert self._conn is not None
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(model_results)").fetchall()}
        if "model_blob" not in cols:
            self._conn.execute("ALTER TABLE model_results ADD COLUMN model_blob BLOB")
            self._conn.commit()
            log.info("migrated_model_results_add_blob")

    def _migrate_time_local_to_24h(self) -> None:
        """One-time migration: convert AM/PM time_local strings to 24h 'HH:MM' format.

        Idempotent — rows already in HH:MM format are skipped by the WHERE clause.
        """
        assert self._conn is not None
        result = self._conn.execute(
            "SELECT COUNT(*) FROM hourly_observations WHERE time_local LIKE '% AM' OR time_local LIKE '% PM'"
        ).fetchone()
        count = result[0] if result else 0
        if count == 0:
            return

        self._conn.execute("""
            UPDATE hourly_observations
            SET time_local = printf('%02d:%02d',
                CASE
                    WHEN substr(time_local, -2) = 'AM'
                         AND CAST(substr(time_local, 1, instr(time_local, ':') - 1) AS INT) = 12 THEN 0
                    WHEN substr(time_local, -2) = 'AM'
                         THEN CAST(substr(time_local, 1, instr(time_local, ':') - 1) AS INT)
                    WHEN substr(time_local, -2) = 'PM'
                         AND CAST(substr(time_local, 1, instr(time_local, ':') - 1) AS INT) = 12 THEN 12
                    WHEN substr(time_local, -2) = 'PM'
                         THEN CAST(substr(time_local, 1, instr(time_local, ':') - 1) AS INT) + 12
                    ELSE hour
                END,
                CAST(substr(time_local, instr(time_local, ':') + 1, 2) AS INT)
            )
            WHERE time_local LIKE '% AM' OR time_local LIKE '% PM'
        """)
        log.info("migrated_time_local_to_24h", rows=count)

    def _migrate_daily_peaks_add_summary_high(self) -> None:
        """Add summary_high_temp REAL column to daily_peaks if missing (idempotent)."""
        assert self._conn is not None
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(daily_peaks)").fetchall()}
        if "summary_high_temp" not in cols:
            self._conn.execute("ALTER TABLE daily_peaks ADD COLUMN summary_high_temp REAL")
            self._conn.commit()
            log.info("migrated_daily_peaks_add_summary_high")

    def _migrate_bets_add_llm_source(self) -> None:
        """Add llm_source TEXT column to bets if missing (idempotent).

        Existing Claude bets get DEFAULT 'claude' automatically.
        """
        assert self._conn is not None
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(bets)").fetchall()}
        if "llm_source" not in cols:
            self._conn.execute("ALTER TABLE bets ADD COLUMN llm_source TEXT DEFAULT 'claude'")
            self._conn.commit()
            log.info("migrated_bets_add_llm_source")

    def _migrate_clean_resolution_url_dots(self) -> None:
        """Strip trailing dots/punctuation from wu_url_path values caused by early regex bug.

        Idempotent — only touches rows where wu_url_path ends with a dot.
        """
        assert self._conn is not None
        # Check table exists first (may be a fresh DB before schema runs)
        tables = {r[0] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "city_resolution_urls" not in tables:
            return
        result = self._conn.execute(
            "SELECT COUNT(*) FROM city_resolution_urls WHERE wu_url_path LIKE '%.'"
        ).fetchone()
        count = result[0] if result else 0
        if count == 0:
            return
        self._conn.execute("""
            UPDATE city_resolution_urls
            SET wu_url_path = rtrim(wu_url_path, './,''\")')
            WHERE wu_url_path LIKE '%.'
        """)
        self._conn.execute("""
            UPDATE city_resolution_urls
            SET wu_url = 'https://www.wunderground.com/history/daily/' || wu_url_path
            WHERE wu_url_path IS NOT NULL
        """)
        self._conn.commit()
        log.info("migrated_clean_resolution_url_dots", rows=count)

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
