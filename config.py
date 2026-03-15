from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    db_path: str = "./data/poly_weather.db"

    # Polymarket credentials
    poly_private_key: str = ""
    poly_api_key: str = ""
    poly_api_secret: str = ""
    poly_api_passphrase: str = ""
    poly_signature_type: int = 0  # 0=EOA, 1=POLY_PROXY

    # Trading
    paper_mode: bool = True
    max_bet_size: float = 50.0
    max_daily_exposure: float = 200.0
    min_confidence: float = 0.85
    min_expected_value: float = 0.03

    # Polling intervals
    weather_poll_interval_minutes: int = 30
    market_poll_interval_minutes: int = 5

    # Active cities
    active_cities: list[str] = ["Ankara"]

    # Bot mode
    bot_auto_start: bool = False  # If True, runs automated bot loop in background

    # Logging
    log_level: str = "INFO"


settings = Settings()
