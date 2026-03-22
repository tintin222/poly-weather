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
    initial_balance: float = 1000.0     # Paper trading starting balance
    bet_size: float = 10.0              # Fixed bet amount per trade
    max_daily_exposure: float = 200.0   # Max total bets per day
    min_edge: float = 0.05             # Min edge (our_prob - market_price) to bet
    min_confidence: float = 0.85        # Legacy: min peak-passed confidence
    min_expected_value: float = 0.03    # Legacy: min EV

    # Polling intervals
    weather_poll_interval_minutes: int = 30
    market_poll_interval_minutes: int = 5

    # Active cities
    active_cities: list[str] = ["Ankara"]

    # Bot mode
    bot_auto_start: bool = True  # Runs automated paper trading bot in background
    # Orchestrator: CDF+ML mechanical betting bot. Disabled by default — the LLM bots
    # (Claude/GPT/Gemini) use richer reasoning and multi-source data and consistently
    # outperform the mechanical model. Enabling this alongside LLMs risks conflicting
    # bracket positions (different bets on the same city/date). Set to True only for
    # A/B testing or if all LLM bots are disabled.
    orchestrator_enabled: bool = False

    # LLM advisors
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Logging
    log_level: str = "INFO"


settings = Settings()
