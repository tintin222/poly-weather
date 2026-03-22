# Weather Data Integration Spec — Accurate Historical Temperature Collection

## Problem Statement

When comparing weather data sources for the same station and day, there is a known discrepancy between **summary daily max/min temperatures** and **hourly observation tables**.

**Example observed on Weather Underground (EGLC — London City Airport, 2026-03-21):**

- **Summary table** reports: Actual High = **11°C**
- **Hourly table** shows max: **9°C** (at 11:20 AM / 11:50 AM)
- The chart on the same page also peaks around **9°C**

### Root Cause

The two values come from **different data pipelines**:

1. **Summary "Actual" high (11°C)** — Sourced from the station's official daily climate report (SYNOP or daily summary). This is based on a **dedicated min/max thermometer or continuous sensor** that captures the absolute peak, even if it lasted only seconds.

2. **Hourly table max (9°C)** — Based on **METAR observations taken at fixed intervals** (every 30 or 60 minutes). These are point-in-time snapshots. If the temperature spiked to 11°C between observation times (e.g., at 12:05 PM), the 11:50 AM reading caught 9°C and the 12:20 PM reading caught the value after it dropped back. The peak was missed.

3. **Incomplete day** — On the current day, the hourly table may only have data up to the latest observation, while the summary may already reflect a more recent peak.

This is a fundamental limitation of METAR/snapshot-based data: it almost always **underreports the true daily max** and **overreports the true daily min**.

### Same issue on metar-taf.com

The same pattern appears on metar-taf.com/history/KSEA. The METAR history table shows temperature readings at ~30 minute intervals. Any temperature peaks occurring between those intervals are lost.

---

## Data Source Options Evaluated

### 1. METAR via Aviation Weather Center API (Global, Free)

- **Endpoint:** `https://aviationweather.gov/api/data/metar?ids=KSEA&format=json`
- **Resolution:** 30–60 minute snapshots
- **Coverage:** Global (any ICAO station)
- **History:** Up to 15 days
- **Rate limit:** 100 req/min
- **Auth:** None required
- **Limitation:** Snapshot data — misses peaks between observations
- **Docs:** https://aviationweather.gov/data/api/

### 2. METAR via metar-taf.com (Global, Web Scraping)

- **URL:** `https://metar-taf.com/history/{ICAO_CODE}`
- **Resolution:** 30–60 minute snapshots (same METAR data)
- **Coverage:** Global
- **History:** Back to 2007
- **Auth:** None, but Cloudflare-protected (needs headless browser)
- **Limitation:** Same snapshot limitation as above. Requires Puppeteer/Playwright for scraping.
- **Data format:** HTML table with columns: Time, Code, Weather, Temp, Visibility, Ceiling, Wind, METAR/SPECI

### 3. Weather Underground (Global, Web Scraping)

- **URL:** `https://www.wunderground.com/history/daily/{country}/{city}/{ICAO}/date/{YYYY-M-D}`
- **Resolution:** 30 minute snapshots in hourly table
- **Coverage:** Global + Personal Weather Stations (PWS) network
- **Note:** Has both a summary table (from daily climate report) and an hourly table (from METAR). The summary table has the accurate daily max/min, but the hourly table has the granular data. The two don't match (this is the problem we're solving).
- **Auth:** API key required for programmatic access (IBM Weather Company API)

### 4. IEM ASOS 1-Minute Data (US Only, Free) ✅ RECOMMENDED

- **Endpoint:** `https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py`
- **Resolution:** Every 1 minute (sensor samples every 10 seconds, averages per minute)
- **Coverage:** US ASOS stations only (all major US airports)
- **History:** Back to ~2000 for most stations, up to previous day
- **Auth:** None required
- **Rate limit:** None documented, be reasonable
- **Limitation:** US airports only. Data available up to previous day (not real-time). Temperature reported in whole degrees Fahrenheit only.
- **Docs:** https://mesonet.agron.iastate.edu/request/asos/1min.phtml

### 5. Synoptic Data High-Frequency ASOS (US, API)

- **Endpoint:** Synoptic Data Mesonet API
- **Resolution:** 1 minute (real-time stream)
- **Station ID format:** Station ID with `1M` suffix (e.g., `KSEA1M`)
- **Auth:** API key required (free tier available)
- **Note:** Was unavailable Oct 2023 – Jan 2026, now restored
- **Docs:** https://docs.synopticdata.com/services/high-frequency-asos

### 6. UK Met Office Weather DataHub (UK Only)

- **Endpoint:** https://datahub.metoffice.gov.uk/
- **Resolution:** Hourly SYNOP observations publicly; 1-minute data exists internally but not freely exposed via API
- **Coverage:** UK automatic weather stations
- **Auth:** API key required
- **Limitation:** Public API only gives hourly data. 1-minute data requires research access request.

### 7. Open-Meteo (Global, Free)

- **Endpoint:** https://open-meteo.com/en/docs/ukmo-api
- **Resolution:** Hourly (reanalysis/model data, not raw observations)
- **Coverage:** Global
- **Auth:** None required
- **Limitation:** Model/reanalysis data, not raw station observations. Hourly resolution only.

---

## Recommended Solution: IEM ASOS 1-Minute API

For US airports, the IEM ASOS 1-minute data is the best option. It captures temperature every minute, so you never miss a peak.

### API Specification

**Base URL:**
```
https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `station` | string | Yes | FAA station ID without K prefix (e.g., `SEA`, `JFK`, `ORD`, `LAX`, `ATL`) |
| `tz` | string | Yes | Timezone for output timestamps. `UTC` or IANA tz like `America/Los_Angeles` |
| `year1` | int | Yes | Start year |
| `month1` | int | Yes | Start month (1-12) |
| `day1` | int | Yes | Start day (1-31) |
| `hour1` | int | Yes | Start hour (0-23) |
| `minute1` | int | Yes | Start minute (0-59) |
| `year2` | int | Yes | End year |
| `month2` | int | Yes | End month (1-12) |
| `day2` | int | Yes | End day (1-31) |
| `hour2` | int | Yes | End hour (0-23) |
| `minute2` | int | Yes | End minute (0-59) |
| `vars` | string | Yes | Comma-separated variable codes (see below) |
| `sample` | string | No | Sampling interval: `1min`, `5min`, `10min`, `20min`, `1hour`. Default: `1min` |
| `what` | string | Yes | `download` (returns as file attachment) or `view` (returns as plain text in browser) |
| `delim` | string | No | Delimiter: `comma`, `tab`, `space`. Default: `comma` |
| `gis` | string | No | Include lat/lon columns: `yes` or `no`. Default: `no` |

**Available Variables:**

| Code | Description | Unit |
|------|-------------|------|
| `tmpf` | Air Temperature | °F (whole degrees) |
| `dwpf` | Dew Point Temperature | °F (whole degrees) |
| `sknt` | Wind Speed | knots |
| `drct` | Wind Direction | degrees |
| `gust_drct` | 5-second Gust Wind Direction | degrees |

### Example curl

```bash
# Get 1-minute temperatures for Seattle on March 19, 2026
curl -o ksea_1min.csv \
  'https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=SEA&tz=UTC&year1=2026&month1=3&day1=19&hour1=0&minute1=0&year2=2026&month2=3&day2=20&hour2=0&minute2=0&vars=tmpf&sample=1min&what=download&delim=comma&gis=no'
```

### Response Format

```csv
station,station_name,valid(UTC),tmpf
SEA,SEATTLE-TACOMA INTL,2026-03-19 00:11,54
SEA,SEATTLE-TACOMA INTL,2026-03-19 00:12,54
SEA,SEATTLE-TACOMA INTL,2026-03-19 00:13,54
SEA,SEATTLE-TACOMA INTL,2026-03-19 01:06,53
SEA,SEATTLE-TACOMA INTL,2026-03-19 02:11,52
SEA,SEATTLE-TACOMA INTL,2026-03-19 08:16,49
...
```

- ~1,440 rows per full day (one per minute, minus any gaps)
- Temperature is in **whole degrees Fahrenheit** — convert to Celsius: `(°F - 32) × 5/9`
- Timestamps follow the timezone specified in the `tz` parameter

### Verified Sample Data (2026-03-19, KSEA)

| Metric | Value |
|--------|-------|
| Total records | 1,258 |
| Time range | 2026-03-19 00:11 UTC → 2026-03-20 09:06 UTC |
| Min temperature | 49°F (9.4°C) at 08:16 UTC |
| Max temperature | 54°F (12.2°C) at 00:11 UTC |
| METAR max for same period | 12°C (rounded from 11.7°C) — matches |
| METAR min for same period | 9°C (rounded from 9.4°C) — matches |

### Python Implementation

```python
import requests
import pandas as pd
from datetime import datetime


def get_asos_1min(
    station: str,
    start_date: str,
    end_date: str,
    tz: str = "UTC",
    variables: list[str] = None,
    sample: str = "1min",
) -> pd.DataFrame:
    """
    Fetch 1-minute ASOS observation data from Iowa Environmental Mesonet.

    Args:
        station: FAA station ID without K prefix (e.g., 'SEA', 'JFK', 'ORD')
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD' (exclusive — data up to midnight of this date)
        tz: Timezone for timestamps. Default 'UTC'. Use IANA tz names like 'America/Los_Angeles'
        variables: List of variable codes. Default ['tmpf']. Options: tmpf, dwpf, sknt, drct, gust_drct
        sample: Sampling interval. Default '1min'. Options: 1min, 5min, 10min, 20min, 1hour

    Returns:
        pandas DataFrame with columns per requested variable plus datetime and station info.
        Temperature columns are additionally provided in Celsius.

    Raises:
        requests.HTTPError: If the API request fails
        ValueError: If the response contains no data
    """
    if variables is None:
        variables = ["tmpf"]

    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "station": station,
        "tz": tz,
        "year1": s.year, "month1": s.month, "day1": s.day,
        "hour1": 0, "minute1": 0,
        "year2": e.year, "month2": e.month, "day2": e.day,
        "hour2": 0, "minute2": 0,
        "vars": ",".join(variables),
        "sample": sample,
        "what": "download",
        "delim": "comma",
        "gis": "no",
    }

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    if df.empty:
        raise ValueError(f"No data returned for station {station} between {start_date} and {end_date}")

    # Rename columns for clarity
    col_renames = {"valid(UTC)": "datetime_utc"}
    if tz != "UTC":
        col_renames = {f"valid({tz})": "datetime_local"}
    df = df.rename(columns=col_renames)

    # Add Celsius conversions if temperature columns present
    if "tmpf" in df.columns:
        df["tmpc"] = round((df["tmpf"] - 32) * 5 / 9, 1)
    if "dwpf" in df.columns:
        df["dwpc"] = round((df["dwpf"] - 32) * 5 / 9, 1)

    return df


def get_daily_summary(df: pd.DataFrame, temp_col: str = "tmpf") -> dict:
    """
    Compute daily summary statistics from 1-minute data.

    Returns dict with max, min, mean, and their timestamps.
    """
    dt_col = "datetime_utc" if "datetime_utc" in df.columns else "datetime_local"
    return {
        "max_f": int(df[temp_col].max()),
        "min_f": int(df[temp_col].min()),
        "mean_f": round(df[temp_col].mean(), 1),
        "max_c": round((df[temp_col].max() - 32) * 5 / 9, 1),
        "min_c": round((df[temp_col].min() - 32) * 5 / 9, 1),
        "mean_c": round((df[temp_col].mean() - 32) * 5 / 9, 1),
        "max_at": df.loc[df[temp_col].idxmax(), dt_col],
        "min_at": df.loc[df[temp_col].idxmin(), dt_col],
        "record_count": len(df),
    }


# --- Usage Example ---
if __name__ == "__main__":
    df = get_asos_1min("SEA", "2026-03-19", "2026-03-20")
    summary = get_daily_summary(df)

    print(f"Records:  {summary['record_count']}")
    print(f"Max temp: {summary['max_f']}°F ({summary['max_c']}°C) at {summary['max_at']}")
    print(f"Min temp: {summary['min_f']}°F ({summary['min_c']}°C) at {summary['min_at']}")
    print(f"Mean:     {summary['mean_f']}°F ({summary['mean_c']}°C)")
    print()
    print(df.head(10).to_string(index=False))
```

### Node.js / TypeScript Implementation

```typescript
interface ASOSRecord {
  datetime_utc: string;
  temperature_f: number;
  temperature_c: number;
}

interface DailySummary {
  max_f: number;
  min_f: number;
  mean_f: number;
  max_c: number;
  min_c: number;
  mean_c: number;
  max_at: string;
  min_at: string;
  record_count: number;
}

async function getASOS1Min(
  station: string,
  startDate: string, // YYYY-MM-DD
  endDate: string,   // YYYY-MM-DD
): Promise<ASOSRecord[]> {
  const s = new Date(startDate);
  const e = new Date(endDate);

  const params = new URLSearchParams({
    station,
    tz: "UTC",
    year1: String(s.getFullYear()),
    month1: String(s.getMonth() + 1),
    day1: String(s.getDate()),
    hour1: "0", minute1: "0",
    year2: String(e.getFullYear()),
    month2: String(e.getMonth() + 1),
    day2: String(e.getDate()),
    hour2: "0", minute2: "0",
    vars: "tmpf",
    sample: "1min",
    what: "download",
    delim: "comma",
    gis: "no",
  });

  const url = `https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?${params}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const text = await resp.text();
  const lines = text.trim().split("\n");

  // Skip header, parse CSV
  return lines.slice(1)
    .filter((line) => line.startsWith(station))
    .map((line) => {
      const parts = line.split(",");
      const tmpf = parseInt(parts[3]);
      return {
        datetime_utc: parts[2],
        temperature_f: tmpf,
        temperature_c: Math.round(((tmpf - 32) * 5) / 9 * 10) / 10,
      };
    });
}

function getDailySummary(records: ASOSRecord[]): DailySummary {
  const temps = records.map((r) => r.temperature_f);
  const maxF = Math.max(...temps);
  const minF = Math.min(...temps);
  const meanF = temps.reduce((a, b) => a + b, 0) / temps.length;

  return {
    max_f: maxF,
    min_f: minF,
    mean_f: Math.round(meanF * 10) / 10,
    max_c: Math.round(((maxF - 32) * 5) / 9 * 10) / 10,
    min_c: Math.round(((minF - 32) * 5) / 9 * 10) / 10,
    mean_c: Math.round(((meanF - 32) * 5) / 9 * 10) / 10,
    max_at: records.find((r) => r.temperature_f === maxF)!.datetime_utc,
    min_at: records.find((r) => r.temperature_f === minF)!.datetime_utc,
    record_count: records.length,
  };
}

// --- Usage ---
// const data = await getASOS1Min("SEA", "2026-03-19", "2026-03-20");
// const summary = getDailySummary(data);
// console.log(summary);
```

---

## Fallback: METAR API for Non-US Stations

For stations outside the US (like EGLC London City), 1-minute data is not freely available. Use the Aviation Weather Center METAR API as a fallback, understanding that it gives 30-60 minute snapshots only.

**Endpoint:**
```
https://aviationweather.gov/api/data/metar?ids=EGLC&format=json&hours=24
```

**Response (JSON):**
```json
[
  {
    "icaoId": "EGLC",
    "reportTime": "2026-03-21T11:50:00Z",
    "temp": 9,
    "dewp": 0,
    "wdir": 70,
    "wspd": 6,
    "wgst": null,
    "visib": "10+",
    "altim": 1019.4,
    "rawOb": "METAR EGLC 211150Z 07006KT ..."
  }
]
```

**Key parameters:**
- `ids` — ICAO code (e.g., `EGLC`, `KSEA`, `KJFK`)
- `format` — `json`, `xml`, or `raw`
- `hours` — Hours of history to retrieve (max ~15 days worth)
- `date` — Specific date for historical data (format: `YYYYMMDD_HHmm`)

**Docs:** https://aviationweather.gov/data/api/

---

## Architecture Decision

| Scenario | Data Source | Resolution | Notes |
|----------|-----------|------------|-------|
| US airport, historical (>1 day ago) | IEM ASOS 1-min | 1 min | Best accuracy, free |
| US airport, near real-time | Synoptic Data HF-ASOS | 1 min | Needs API key |
| US airport, current conditions | aviationweather.gov METAR | 30-60 min | Free, instant |
| Non-US airport, any | aviationweather.gov METAR | 30-60 min | Free, global |
| UK station, high-res | Met Office DataHub | 1 hour (public) | API key required |
| Any location, model data | Open-Meteo | 1 hour | Free, not raw obs |

### Recommended Strategy

```
if station is US airport:
    if need historical data (yesterday or older):
        → use IEM ASOS 1-minute API
    else if need current/today:
        → use aviationweather.gov METAR API (30-min)
        → optionally supplement with Synoptic Data HF-ASOS (1-min, needs key)
else:
    → use aviationweather.gov METAR API (global, 30-min)
    → accept that daily max/min from METAR may differ from official climate reports
```
