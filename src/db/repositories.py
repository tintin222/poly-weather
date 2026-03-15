"""Data access layer for poly-weather database."""

from __future__ import annotations

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

    def get_daily_exposure(self, date: str) -> float:
        row = self._db.conn.execute(
            "SELECT COALESCE(SUM(cost), 0) as total FROM bets WHERE date = ?", (date,)
        ).fetchone()
        return float(row["total"]) if row else 0.0
