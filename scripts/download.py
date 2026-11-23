"""Download GB demand (NESO, 2018-2026), NESO's archived day-ahead forecasts, and weather (Open-Meteo, no key).

Weather comes twice: the ERA5-based archive (what actually happened) and Open-Meteo's "previous runs" archive of
forecasts issued one day earlier (`*_previous_day1`), so the validation and test years can be scored with the weather
information a day-ahead forecaster would really have had.
"""
import json
import time
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
API = "https://api.neso.energy/api/3/action/package_show?id="
YEARS = range(2018, 2027)
# five large population centres, weighted by rough metro population (millions)
CITIES = {"London": (51.51, -0.13, 9.0), "Birmingham": (52.49, -1.89, 2.9), "Manchester": (53.48, -2.24, 2.8),
          "Glasgow": (55.86, -4.25, 1.7), "Leeds": (53.80, -1.55, 1.9)}
# where embedded (sub-transmission) solar and wind actually are: solar mostly in southern England, wind mostly in
# Scotland, northern England and the south-west coast
SOLAR_SITES = {"Exeter": (50.72, -3.53), "Southampton": (50.90, -1.40), "Norwich": (52.63, 1.30),
               "Swindon": (51.56, -1.78), "Peterborough": (52.57, -0.24), "Truro": (50.26, -5.05)}
WIND_SITES = {"Aberdeen": (57.15, -2.09), "Inverness": (57.48, -4.22), "Dumfries": (55.07, -3.61),
              "Kendal": (54.33, -2.75), "Lincoln": (53.23, -0.54), "Newquay": (50.41, -5.08)}
VARS = "temperature_2m,shortwave_radiation,wind_speed_10m,cloud_cover"


def get(url, path, **kw):
    if path.exists():
        return
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120, **kw); r.raise_for_status()
            if path.suffix == ".json" and (b'"error"' in r.content[:200] or len(r.content) < 1000):
                raise RuntimeError(r.content[:200])
            path.write_bytes(r.content); print("saved", path.name, len(r.content)); return
        except Exception as e:  # transient network errors
            print("retry", path.name, e); time.sleep(30 * (attempt + 1))
    raise RuntimeError(url)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    demand = requests.get(API + "historic-demand-data", timeout=60).json()["result"]["resources"]
    for r in demand:
        for y in YEARS:
            if r["url"].endswith((f"demanddata_{y}.csv", f"demanddataupdate_{y}.csv")):
                get(r["url"], RAW / f"demand_{y}.csv")
    for r in requests.get(API + "1-day-ahead-demand-forecast", timeout=60).json()["result"]["resources"]:
        if "archive" in r["url"]:
            get(r["url"], RAW / "neso_day_ahead_archive.csv")
    for city, (lat, lon, _) in CITIES.items():
        p = dict(latitude=lat, longitude=lon, hourly=VARS, timezone="UTC")
        get("https://archive-api.open-meteo.com/v1/archive", RAW / f"weather_actual_{city}.json",
            params={**p, "start_date": "2018-01-01", "end_date": "2026-11-20"})
        day1 = ",".join(f"{v}_previous_day1" for v in VARS.split(","))
        get("https://previous-runs-api.open-meteo.com/v1/forecast", RAW / f"weather_dayahead_{city}.json",
            params={**p, "hourly": day1, "start_date": "2024-11-01", "end_date": "2026-11-02"})
    for group, sites in (("solar", SOLAR_SITES), ("wind", WIND_SITES)):
        for site, (lat, lon) in sites.items():
            p = dict(latitude=lat, longitude=lon, hourly=VARS, timezone="UTC")
            get("https://archive-api.open-meteo.com/v1/archive", RAW / f"weather_actual_{site}.json",
                params={**p, "start_date": "2018-01-01", "end_date": "2026-11-20"})
            day1 = ",".join(f"{v}_previous_day1" for v in VARS.split(","))
            get("https://previous-runs-api.open-meteo.com/v1/forecast", RAW / f"weather_dayahead_{site}.json",
                params={**p, "hourly": day1, "start_date": "2024-11-01", "end_date": "2026-11-02"})
            time.sleep(2)
    (RAW / "cities.json").write_text(json.dumps(CITIES))
    (RAW / "sites.json").write_text(json.dumps({"solar": list(SOLAR_SITES), "wind": list(WIND_SITES)}))


if __name__ == "__main__":
    main()
