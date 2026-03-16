"""City registry with Weather Underground station mappings.

Ported from copy-cat/src/weather/cities.ts with added WU-specific metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    key: str
    display_name: str
    lat: float
    lon: float
    unit: str  # 'celsius' | 'fahrenheit'
    timezone: str
    wu_station_id: str
    wu_url_path: str


# Polymarket temperature markets use these cities.
# US cities resolve in Fahrenheit, all others in Celsius.
# Coordinates target the weather station each market resolves against
# (Weather Underground / NOAA stations near major airports).
CITIES: dict[str, CityConfig] = {
    "NYC": CityConfig(
        key="NYC", display_name="New York City",
        lat=40.7731, lon=-73.8803, unit="fahrenheit",
        timezone="America/New_York",
        wu_station_id="KLGA", wu_url_path="us/ny/new-york-city/KLGA",
    ),
    "Dallas": CityConfig(
        key="Dallas", display_name="Dallas",
        lat=32.8998, lon=-97.0403, unit="fahrenheit",
        timezone="America/Chicago",
        wu_station_id="KDAL", wu_url_path="us/tx/dallas/KDAL",
    ),
    "Atlanta": CityConfig(
        key="Atlanta", display_name="Atlanta",
        lat=33.6407, lon=-84.4277, unit="fahrenheit",
        timezone="America/New_York",
        wu_station_id="KATL", wu_url_path="us/ga/atlanta/KATL",
    ),
    "Miami": CityConfig(
        key="Miami", display_name="Miami",
        lat=25.7959, lon=-80.2870, unit="fahrenheit",
        timezone="America/New_York",
        wu_station_id="KMIA", wu_url_path="us/fl/miami/KMIA",
    ),
    "Chicago": CityConfig(
        key="Chicago", display_name="Chicago",
        lat=41.9742, lon=-87.9073, unit="fahrenheit",
        timezone="America/Chicago",
        wu_station_id="KORD", wu_url_path="us/il/chicago/KORD",
    ),
    "Seattle": CityConfig(
        key="Seattle", display_name="Seattle",
        lat=47.4502, lon=-122.3088, unit="fahrenheit",
        timezone="America/Los_Angeles",
        wu_station_id="KSEA", wu_url_path="us/wa/seatac/KSEA",
    ),
    "Toronto": CityConfig(
        key="Toronto", display_name="Toronto",
        lat=43.6777, lon=-79.6248, unit="celsius",
        timezone="America/Toronto",
        wu_station_id="CYYZ", wu_url_path="ca/toronto/CYYZ",
    ),
    "London": CityConfig(
        key="London", display_name="London",
        lat=51.5053, lon=0.0553, unit="celsius",
        timezone="Europe/London",
        wu_station_id="EGLC", wu_url_path="gb/london/EGLC",
    ),
    "Paris": CityConfig(
        key="Paris", display_name="Paris",
        lat=49.0097, lon=2.5479, unit="celsius",
        timezone="Europe/Paris",
        wu_station_id="LFPG", wu_url_path="fr/paris/LFPG",
    ),
    "Munich": CityConfig(
        key="Munich", display_name="Munich",
        lat=48.3537, lon=11.7750, unit="celsius",
        timezone="Europe/Berlin",
        wu_station_id="EDDM", wu_url_path="de/munich/EDDM",
    ),
    "Ankara": CityConfig(
        key="Ankara", display_name="Ankara",
        lat=40.1281, lon=32.9951, unit="celsius",
        timezone="Europe/Istanbul",
        wu_station_id="LTAC", wu_url_path="tr/%C3%A7ubuk/LTAC",
    ),
    "Tel Aviv": CityConfig(
        key="Tel Aviv", display_name="Tel Aviv",
        lat=32.0114, lon=34.8867, unit="celsius",
        timezone="Asia/Jerusalem",
        wu_station_id="LLBG", wu_url_path="il/tel-aviv/LLBG",
    ),
    "Seoul": CityConfig(
        key="Seoul", display_name="Seoul",
        lat=37.5583, lon=126.7906, unit="celsius",
        timezone="Asia/Seoul",
        wu_station_id="RKSS", wu_url_path="kr/seoul/RKSS",
    ),
    "Tokyo": CityConfig(
        key="Tokyo", display_name="Tokyo",
        lat=35.5533, lon=139.7811, unit="celsius",
        timezone="Asia/Tokyo",
        wu_station_id="RJTT", wu_url_path="jp/tokyo/RJTT",
    ),
    "Hong Kong": CityConfig(
        key="Hong Kong", display_name="Hong Kong",
        lat=22.3080, lon=113.9185, unit="celsius",
        timezone="Asia/Hong_Kong",
        wu_station_id="VHHH", wu_url_path="hk/hong-kong/VHHH",
    ),
    "Shanghai": CityConfig(
        key="Shanghai", display_name="Shanghai",
        lat=31.1443, lon=121.8083, unit="celsius",
        timezone="Asia/Shanghai",
        wu_station_id="ZSPD", wu_url_path="cn/shanghai/ZSPD",
    ),
    "Singapore": CityConfig(
        key="Singapore", display_name="Singapore",
        lat=1.3644, lon=103.9915, unit="celsius",
        timezone="Asia/Singapore",
        wu_station_id="WSSS", wu_url_path="sg/singapore/WSSS",
    ),
    "Buenos Aires": CityConfig(
        key="Buenos Aires", display_name="Buenos Aires",
        lat=-34.5592, lon=-58.4156, unit="celsius",
        timezone="America/Argentina/Buenos_Aires",
        wu_station_id="SABE", wu_url_path="ar/buenos-aires/SABE",
    ),
    "Sao Paulo": CityConfig(
        key="Sao Paulo", display_name="Sao Paulo",
        lat=-23.6273, lon=-46.6566, unit="celsius",
        timezone="America/Sao_Paulo",
        wu_station_id="SBSP", wu_url_path="br/sao-paulo/SBSP",
    ),
    "Wellington": CityConfig(
        key="Wellington", display_name="Wellington",
        lat=-41.3272, lon=174.8053, unit="celsius",
        timezone="Pacific/Auckland",
        wu_station_id="NZWN", wu_url_path="nz/wellington/NZWN",
    ),
    "Lucknow": CityConfig(
        key="Lucknow", display_name="Lucknow",
        lat=26.7606, lon=80.8893, unit="celsius",
        timezone="Asia/Kolkata",
        wu_station_id="VILK", wu_url_path="in/lucknow/VILK",
    ),
    "Auckland": CityConfig(
        key="Auckland", display_name="Auckland",
        lat=-37.0082, lon=174.7850, unit="celsius",
        timezone="Pacific/Auckland",
        wu_station_id="NZAA", wu_url_path="nz/auckland/NZAA",
    ),
}


# Slug-to-key mapping for cities whose slug form doesn't match their key
_SLUG_ALIASES: dict[str, str] = {
    "nyc": "NYC",
    "new york city": "NYC",
    "new-york-city": "NYC",
    "sao paulo": "Sao Paulo",
    "são paulo": "Sao Paulo",
    "tel aviv": "Tel Aviv",
    "tel-aviv": "Tel Aviv",
    "hong kong": "Hong Kong",
    "hong-kong": "Hong Kong",
    "buenos aires": "Buenos Aires",
    "buenos-aires": "Buenos Aires",
}


def get_city(name: str) -> CityConfig | None:
    """Look up a city by key (case-insensitive), with slug alias support."""
    if name in CITIES:
        return CITIES[name]
    lower = name.lower()
    # Check aliases first
    if lower in _SLUG_ALIASES:
        return CITIES.get(_SLUG_ALIASES[lower])
    for key, city in CITIES.items():
        if key.lower() == lower:
            return city
    return None


def get_active_cities(active_keys: list[str]) -> list[CityConfig]:
    """Return CityConfig objects for the given keys."""
    result = []
    for key in active_keys:
        city = get_city(key)
        if city:
            result.append(city)
    return result
