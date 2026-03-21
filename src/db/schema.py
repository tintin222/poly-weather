"""Database schema definitions for poly-weather."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cities (
    city_key       TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    latitude       REAL NOT NULL,
    longitude      REAL NOT NULL,
    unit           TEXT NOT NULL CHECK(unit IN ('celsius', 'fahrenheit')),
    timezone       TEXT NOT NULL,
    wu_station_id  TEXT NOT NULL,
    wu_url_path    TEXT NOT NULL,
    is_active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS hourly_observations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    city_key       TEXT NOT NULL,
    date           TEXT NOT NULL,
    time_local     TEXT NOT NULL,
    hour           INTEGER NOT NULL,
    temperature    REAL NOT NULL,
    dew_point      REAL,
    humidity       REAL,
    wind_speed     REAL,
    wind_gust      REAL,
    wind_direction TEXT,
    pressure       REAL,
    precip_rate    REAL,
    condition      TEXT,
    scraped_at     INTEGER NOT NULL,
    UNIQUE(city_key, date, time_local)
);

CREATE INDEX IF NOT EXISTS idx_hourly_city_date
    ON hourly_observations(city_key, date);

CREATE TABLE IF NOT EXISTS daily_peaks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    city_key          TEXT NOT NULL,
    date              TEXT NOT NULL,
    peak_temp         REAL NOT NULL,
    peak_hour         INTEGER NOT NULL,
    observation_count INTEGER NOT NULL,
    summary_high_temp REAL,
    computed_at       INTEGER NOT NULL,
    UNIQUE(city_key, date)
);

CREATE INDEX IF NOT EXISTS idx_peaks_city_date
    ON daily_peaks(city_key, date);

CREATE TABLE IF NOT EXISTS polymarket_events (
    event_id          TEXT PRIMARY KEY,
    city_key          TEXT NOT NULL,
    date              TEXT NOT NULL,
    title             TEXT NOT NULL,
    slug              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    winning_bracket   TEXT,
    total_volume      REAL,
    total_liquidity   REAL,
    resolution_source TEXT,
    updated_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_city_date
    ON polymarket_events(city_key, date);
CREATE INDEX IF NOT EXISTS idx_events_status
    ON polymarket_events(status);

CREATE TABLE IF NOT EXISTS polymarket_brackets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id          TEXT NOT NULL REFERENCES polymarket_events(event_id),
    market_id         TEXT NOT NULL,
    condition_id      TEXT NOT NULL,
    bracket_label     TEXT NOT NULL,
    clob_token_id_yes TEXT,
    clob_token_id_no  TEXT,
    current_yes_price REAL,
    current_no_price  REAL,
    volume            REAL,
    tick_size         TEXT,
    neg_risk          INTEGER NOT NULL DEFAULT 0,
    price_updated_at  INTEGER,
    UNIQUE(event_id, market_id)
);

CREATE INDEX IF NOT EXISTS idx_brackets_event
    ON polymarket_brackets(event_id);
CREATE INDEX IF NOT EXISTS idx_brackets_token
    ON polymarket_brackets(clob_token_id_yes);

CREATE TABLE IF NOT EXISTS bets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id          TEXT NOT NULL,
    bracket_id        INTEGER NOT NULL,
    city_key          TEXT NOT NULL,
    date              TEXT NOT NULL,
    bracket_label     TEXT NOT NULL,
    side              TEXT NOT NULL,
    token_id          TEXT NOT NULL,
    price             REAL NOT NULL,
    size              REAL NOT NULL,
    cost              REAL NOT NULL,
    order_id          TEXT,
    order_status      TEXT,
    current_max_temp  REAL NOT NULL,
    peak_hour         INTEGER NOT NULL,
    model_confidence  REAL NOT NULL,
    expected_value    REAL NOT NULL,
    won               INTEGER,
    pnl               REAL,
    resolved_at       INTEGER,
    is_paper          INTEGER NOT NULL DEFAULT 1,
    created_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bets_event ON bets(event_id);
CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(date);

CREATE TABLE IF NOT EXISTS paper_account (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    initial_balance REAL NOT NULL DEFAULT 1000.0,
    bet_size        REAL NOT NULL DEFAULT 10.0,
    max_daily_exposure REAL NOT NULL DEFAULT 200.0,
    min_edge        REAL NOT NULL DEFAULT 0.05,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS job_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type          TEXT NOT NULL,
    city_key          TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    started_at        INTEGER NOT NULL,
    finished_at       INTEGER,
    records_processed INTEGER,
    error_message     TEXT,
    metadata_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_runs_started ON job_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_runs_type_city ON job_runs(job_type, city_key);

CREATE TABLE IF NOT EXISTS model_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    city_key     TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    trained_at   INTEGER NOT NULL,
    n_train      INTEGER,
    n_val        INTEGER,
    n_test       INTEGER,
    train_rmse   REAL,
    val_rmse     REAL,
    test_rmse    REAL,
    train_r2     REAL,
    test_r2      REAL,
    results_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_results_city ON model_results(city_key, trained_at DESC);

CREATE TABLE IF NOT EXISTS city_resolution_urls (
    city_key         TEXT PRIMARY KEY,
    wu_url           TEXT,
    wu_url_path      TEXT,
    previous_url     TEXT,
    changed_at       INTEGER,
    last_checked_at  INTEGER,
    check_count      INTEGER NOT NULL DEFAULT 0,
    source_slug      TEXT
);

CREATE TABLE IF NOT EXISTS llm_analysis_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_key        TEXT NOT NULL,
    date            TEXT NOT NULL,
    analysis_time   INTEGER NOT NULL,
    hour_local      INTEGER NOT NULL,
    peak_estimate   REAL,
    peak_confidence REAL,
    analysis_text   TEXT,
    recs_json       TEXT,
    bets_executed   INTEGER NOT NULL DEFAULT 0,
    model           TEXT NOT NULL DEFAULT 'claude-haiku-4-5-20251001'
);
CREATE INDEX IF NOT EXISTS idx_llm_log_city_date ON llm_analysis_log(city_key, date);
CREATE INDEX IF NOT EXISTS idx_llm_log_time ON llm_analysis_log(analysis_time DESC);
"""
