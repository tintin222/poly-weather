"""Data access layer for poly-weather database."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from src.db.database import Database


class WeatherRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_hourly_observation(
        self,
        city_key: str,
        date: str,
        time_local: str,
        hour: int,
        temperature: float,
        dew_point: float | None = None,
        humidity: float | None = None,
        wind_speed: float | None = None,
        wind_gust: float | None = None,
        wind_direction: str | None = None,
        pressure: float | None = None,
        precip_rate: float | None = None,
        condition: str | None = None,
    ) -> None:
        self._db.conn.execute(
            """INSERT INTO hourly_observations
               (city_key, date, time_local, hour, temperature, dew_point,
                humidity, wind_speed, wind_gust, wind_direction,
                pressure, precip_rate, condition, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(city_key, date, time_local)
               DO UPDATE SET
                 temperature = excluded.temperature,
                 dew_point = excluded.dew_point,
                 humidity = excluded.humidity,
                 wind_speed = excluded.wind_speed,
                 wind_gust = excluded.wind_gust,
                 wind_direction = excluded.wind_direction,
                 pressure = excluded.pressure,
                 precip_rate = excluded.precip_rate,
                 condition = excluded.condition,
                 scraped_at = excluded.scraped_at
            """,
            (
                city_key, date, time_local, hour, temperature,
                dew_point, humidity, wind_speed, wind_gust,
                wind_direction, pressure, precip_rate, condition,
                int(time.time()),
            ),
        )
        self._db.conn.commit()

    def upsert_hourly_observations_batch(
        self, city_key: str, date: str, observations: list[dict[str, Any]]
    ) -> int:
        now = int(time.time())
        rows = [
            (
                city_key, date, obs["time_local"], obs["hour"], obs["temperature"],
                obs.get("dew_point"), obs.get("humidity"), obs.get("wind_speed"),
                obs.get("wind_gust"), obs.get("wind_direction"), obs.get("pressure"),
                obs.get("precip_rate"), obs.get("condition"), now,
            )
            for obs in observations
        ]
        self._db.conn.executemany(
            """INSERT INTO hourly_observations
               (city_key, date, time_local, hour, temperature, dew_point,
                humidity, wind_speed, wind_gust, wind_direction,
                pressure, precip_rate, condition, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(city_key, date, time_local)
               DO UPDATE SET
                 temperature = excluded.temperature,
                 dew_point = excluded.dew_point,
                 humidity = excluded.humidity,
                 wind_speed = excluded.wind_speed,
                 wind_gust = excluded.wind_gust,
                 wind_direction = excluded.wind_direction,
                 pressure = excluded.pressure,
                 precip_rate = excluded.precip_rate,
                 condition = excluded.condition,
                 scraped_at = excluded.scraped_at
            """,
            rows,
        )
        self._db.conn.commit()
        return len(rows)

    def get_hourly_observations(
        self, city_key: str, date: str
    ) -> list[sqlite3.Row]:
        return self._db.conn.execute(
            """SELECT * FROM hourly_observations
               WHERE city_key = ? AND date = ?
               ORDER BY hour""",
            (city_key, date),
        ).fetchall()

    def get_historical_hourly(
        self,
        city_key: str,
        month: int,
        day_start: int,
        day_end: int,
        years_back: int = 3,
    ) -> list[sqlite3.Row]:
        """Get historical hourly data for a city around a specific month/day window."""
        placeholders = ",".join("?" * years_back)
        current_year = time.localtime().tm_year
        years = [current_year - i for i in range(years_back)]

        return self._db.conn.execute(
            f"""SELECT * FROM hourly_observations
                WHERE city_key = ?
                  AND CAST(substr(date, 6, 2) AS INTEGER) = ?
                  AND CAST(substr(date, 9, 2) AS INTEGER) BETWEEN ? AND ?
                  AND CAST(substr(date, 1, 4) AS INTEGER) IN ({placeholders})
                ORDER BY date, hour""",
            (city_key, month, day_start, day_end, *years),
        ).fetchall()

    def upsert_daily_peak(
        self, city_key: str, date: str, peak_temp: float,
        peak_hour: int, observation_count: int,
    ) -> None:
        self._db.conn.execute(
            """INSERT INTO daily_peaks
               (city_key, date, peak_temp, peak_hour, observation_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(city_key, date)
               DO UPDATE SET
                 peak_temp = excluded.peak_temp,
                 peak_hour = excluded.peak_hour,
                 observation_count = excluded.observation_count,
                 computed_at = excluded.computed_at
            """,
            (city_key, date, peak_temp, peak_hour, observation_count, int(time.time())),
        )
        self._db.conn.commit()

    def get_daily_peak(self, city_key: str, date: str) -> sqlite3.Row | None:
        return self._db.conn.execute(
            "SELECT * FROM daily_peaks WHERE city_key = ? AND date = ?",
            (city_key, date),
        ).fetchone()

    def get_hourly_paginated(
        self,
        city_key: str,
        page: int = 1,
        page_size: int = 50,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[list[sqlite3.Row], int]:
        conditions = ["city_key = ?"]
        params: list[Any] = [city_key]
        if date_from:
            conditions.append("date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("date <= ?")
            params.append(date_to)
        where = " AND ".join(conditions)
        total = self._db.conn.execute(
            f"SELECT COUNT(*) as cnt FROM hourly_observations WHERE {where}", params
        ).fetchone()["cnt"]
        offset = (page - 1) * page_size
        rows = self._db.conn.execute(
            f"""SELECT * FROM hourly_observations WHERE {where}
                ORDER BY date DESC, hour DESC
                LIMIT ? OFFSET ?""",
            [*params, page_size, offset],
        ).fetchall()
        return rows, total

    def get_city_stats(self, city_key: str) -> dict:
        row = self._db.conn.execute(
            """SELECT
                COUNT(*) as total_rows,
                AVG(temperature) as avg_temp,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                MIN(date) as min_date,
                MAX(date) as max_date
               FROM hourly_observations WHERE city_key = ?""",
            (city_key,),
        ).fetchone()
        return dict(row) if row else {}

    def get_recent_daily_peaks(self, city_key: str, limit: int = 30) -> list[sqlite3.Row]:
        return self._db.conn.execute(
            """SELECT * FROM daily_peaks WHERE city_key = ?
               ORDER BY date ASC LIMIT ?""",
            (city_key, limit),
        ).fetchall()

    def get_historical_peaks(
        self, city_key: str, month: int, day_start: int, day_end: int,
        years_back: int = 3,
    ) -> list[sqlite3.Row]:
        current_year = time.localtime().tm_year
        years = [current_year - i for i in range(years_back)]
        placeholders = ",".join("?" * years_back)
        return self._db.conn.execute(
            f"""SELECT * FROM daily_peaks
                WHERE city_key = ?
                  AND CAST(substr(date, 6, 2) AS INTEGER) = ?
                  AND CAST(substr(date, 9, 2) AS INTEGER) BETWEEN ? AND ?
                  AND CAST(substr(date, 1, 4) AS INTEGER) IN ({placeholders})
                ORDER BY date""",
            (city_key, month, day_start, day_end, *years),
        ).fetchall()


class MarketRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_event(
        self,
        event_id: str,
        city_key: str,
        date: str,
        title: str,
        slug: str,
        status: str = "active",
        winning_bracket: str | None = None,
        total_volume: float | None = None,
        total_liquidity: float | None = None,
        resolution_source: str | None = None,
    ) -> None:
        self._db.conn.execute(
            """INSERT INTO polymarket_events
               (event_id, city_key, date, title, slug, status,
                winning_bracket, total_volume, total_liquidity,
                resolution_source, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id)
               DO UPDATE SET
                 status = excluded.status,
                 winning_bracket = excluded.winning_bracket,
                 total_volume = excluded.total_volume,
                 total_liquidity = excluded.total_liquidity,
                 resolution_source = excluded.resolution_source,
                 updated_at = excluded.updated_at
            """,
            (
                event_id, city_key, date, title, slug, status,
                winning_bracket, total_volume, total_liquidity,
                resolution_source, int(time.time()),
            ),
        )
        self._db.conn.commit()

    def upsert_bracket(
        self,
        event_id: str,
        market_id: str,
        condition_id: str,
        bracket_label: str,
        clob_token_id_yes: str | None = None,
        clob_token_id_no: str | None = None,
        current_yes_price: float | None = None,
        current_no_price: float | None = None,
        volume: float | None = None,
        tick_size: str | None = None,
        neg_risk: bool = False,
    ) -> int:
        cursor = self._db.conn.execute(
            """INSERT INTO polymarket_brackets
               (event_id, market_id, condition_id, bracket_label,
                clob_token_id_yes, clob_token_id_no, current_yes_price,
                current_no_price, volume, tick_size, neg_risk, price_updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id, market_id) DO UPDATE SET
                 current_yes_price = excluded.current_yes_price,
                 current_no_price = excluded.current_no_price,
                 volume = excluded.volume,
                 price_updated_at = excluded.price_updated_at
            """,
            (
                event_id, market_id, condition_id, bracket_label,
                clob_token_id_yes, clob_token_id_no, current_yes_price,
                current_no_price, volume, tick_size, int(neg_risk),
                int(time.time()),
            ),
        )
        self._db.conn.commit()
        return cursor.lastrowid or 0

    def get_active_events(self, date: str | None = None) -> list[sqlite3.Row]:
        if date:
            return self._db.conn.execute(
                "SELECT * FROM polymarket_events WHERE status = 'active' AND date = ?",
                (date,),
            ).fetchall()
        return self._db.conn.execute(
            "SELECT * FROM polymarket_events WHERE status = 'active'"
        ).fetchall()

    def get_brackets_for_event(self, event_id: str) -> list[sqlite3.Row]:
        return self._db.conn.execute(
            "SELECT * FROM polymarket_brackets WHERE event_id = ? ORDER BY bracket_label",
            (event_id,),
        ).fetchall()

    def record_bet(
        self,
        event_id: str,
        bracket_id: int,
        city_key: str,
        date: str,
        bracket_label: str,
        side: str,
        token_id: str,
        price: float,
        size: float,
        cost: float,
        current_max_temp: float,
        peak_hour: int,
        model_confidence: float,
        expected_value: float,
        order_id: str | None = None,
        order_status: str | None = None,
        is_paper: bool = True,
    ) -> int:
        cursor = self._db.conn.execute(
            """INSERT INTO bets
               (event_id, bracket_id, city_key, date, bracket_label, side,
                token_id, price, size, cost, order_id, order_status,
                current_max_temp, peak_hour, model_confidence, expected_value,
                is_paper, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, bracket_id, city_key, date, bracket_label, side,
                token_id, price, size, cost, order_id, order_status,
                current_max_temp, peak_hour, model_confidence, expected_value,
                int(is_paper), int(time.time()),
            ),
        )
        self._db.conn.commit()
        return cursor.lastrowid or 0

    def get_bets_for_date(self, date: str) -> list[sqlite3.Row]:
        return self._db.conn.execute(
            "SELECT * FROM bets WHERE date = ? ORDER BY created_at", (date,)
        ).fetchall()

    def get_pending_bets_for_event(self, event_id: str) -> list[sqlite3.Row]:
        return self._db.conn.execute(
            "SELECT * FROM bets WHERE event_id = ? AND won IS NULL",
            (event_id,),
        ).fetchall()

    def resolve_bet(self, bet_id: int, won: int, pnl: float) -> None:
        self._db.conn.execute(
            "UPDATE bets SET won = ?, pnl = ?, resolved_at = ? WHERE id = ?",
            (won, round(pnl, 2), int(time.time()), bet_id),
        )
        self._db.conn.commit()

    def get_daily_exposure(self, date: str) -> float:
        row = self._db.conn.execute(
            "SELECT COALESCE(SUM(cost), 0) as total FROM bets WHERE date = ?", (date,)
        ).fetchone()
        return float(row["total"]) if row else 0.0


class JobRunRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_run(
        self,
        job_type: str,
        city_key: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        cursor = self._db.conn.execute(
            """INSERT INTO job_runs (job_type, city_key, status, started_at, metadata_json)
               VALUES (?, ?, 'running', ?, ?)""",
            (job_type, city_key, int(time.time()), json.dumps(metadata) if metadata else None),
        )
        self._db.conn.commit()
        return cursor.lastrowid or 0

    def finish_run(
        self,
        run_id: int,
        status: str,
        records_processed: int | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._db.conn.execute(
            """UPDATE job_runs
               SET status = ?, finished_at = ?, records_processed = ?,
                   error_message = ?, metadata_json = COALESCE(?, metadata_json)
               WHERE id = ?""",
            (
                status, int(time.time()), records_processed, error_message,
                json.dumps(metadata) if metadata else None, run_id,
            ),
        )
        self._db.conn.commit()

    def list_runs(
        self,
        job_type: str | None = None,
        city_key: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        conditions = []
        params: list[Any] = []
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)
        if city_key:
            conditions.append("city_key = ?")
            params.append(city_key)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._db.conn.execute(
            f"SELECT * FROM job_runs {where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self._db.conn.execute(
            "SELECT * FROM job_runs WHERE id = ?", (run_id,)
        ).fetchone()

    def get_summary_stats(self) -> dict:
        total = self._db.conn.execute("SELECT COUNT(*) as v FROM job_runs").fetchone()["v"]
        succeeded = self._db.conn.execute(
            "SELECT COUNT(*) as v FROM job_runs WHERE status = 'success'"
        ).fetchone()["v"]
        cutoff = int(time.time()) - 86400
        last_24h = self._db.conn.execute(
            "SELECT COUNT(*) as v FROM job_runs WHERE started_at >= ?", (cutoff,)
        ).fetchone()["v"]
        success_rate = round(succeeded / total * 100, 1) if total > 0 else 0.0
        return {"total_runs": total, "success_rate": success_rate, "runs_last_24h": last_24h}


class ModelResultRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save_result(self, city_key: str, result: dict, model_blob: bytes | None = None) -> int:
        cursor = self._db.conn.execute(
            """INSERT INTO model_results
               (city_key, model_name, trained_at, n_train, n_val, n_test,
                train_rmse, val_rmse, test_rmse, train_r2, test_r2, results_json, model_blob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                city_key,
                result.get("model_name"),
                int(time.time()),
                result.get("n_train"),
                result.get("n_val"),
                result.get("n_test"),
                result.get("train_rmse"),
                result.get("val_rmse"),
                result.get("test_rmse"),
                result.get("train_r2"),
                result.get("test_r2"),
                json.dumps(result),
                model_blob,
            ),
        )
        self._db.conn.commit()
        return cursor.lastrowid or 0

    def get_latest_results(self, city_key: str) -> list[dict]:
        """Get the most recent result for each model_name for the given city."""
        rows = self._db.conn.execute(
            """SELECT results_json FROM model_results
               WHERE id IN (
                   SELECT MAX(id) FROM model_results
                   WHERE city_key = ?
                   GROUP BY model_name
               )
               ORDER BY model_name""",
            (city_key,),
        ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row["results_json"]))
            except Exception:
                pass
        return results

    def get_best_model_blob(self, city_key: str) -> tuple[str, bytes] | None:
        """Return (model_name, pickle_blob) for the city's best model (highest test_auc).

        Returns None if no model with a blob is stored for this city.
        """
        rows = self._db.conn.execute(
            """SELECT model_name, results_json, model_blob FROM model_results
               WHERE city_key = ? AND model_blob IS NOT NULL
               AND id IN (
                   SELECT MAX(id) FROM model_results
                   WHERE city_key = ? AND model_blob IS NOT NULL
                   GROUP BY model_name
               )""",
            (city_key, city_key),
        ).fetchall()
        if not rows:
            return None
        # Pick best by test_auc (classification) or test_rmse (regression fallback)
        best = None
        best_score = -1.0
        for row in rows:
            try:
                r = json.loads(row["results_json"])
                score = float(r.get("test_auc") or 0)
                if score > best_score:
                    best_score = score
                    best = (row["model_name"], bytes(row["model_blob"]))
            except Exception:
                continue
        return best

    def get_trained_city_keys(self) -> list[str]:
        """Return all city_keys that have at least one stored model result."""
        rows = self._db.conn.execute(
            "SELECT DISTINCT city_key FROM model_results ORDER BY city_key"
        ).fetchall()
        return [r["city_key"] for r in rows]
