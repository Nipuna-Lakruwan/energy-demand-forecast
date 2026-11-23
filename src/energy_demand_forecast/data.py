"""Build one half-hourly table: GB national demand (ND), calendar, and population-weighted weather."""
import glob
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WVARS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]


def load_demand(raw: Path = RAW) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "EMBEDDED_SOLAR_CAPACITY", "EMBEDDED_WIND_CAPACITY",
                                             "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_WIND_GENERATION"])
                   for f in sorted(glob.glob(str(raw / "demand_*.csv")))])
    d["date"] = pd.to_datetime(d.SETTLEMENT_DATE, format="mixed").dt.normalize()
    d = d.rename(columns={"SETTLEMENT_PERIOD": "period", "ND": "nd", "EMBEDDED_SOLAR_CAPACITY": "solar_cap",
                          "EMBEDDED_WIND_CAPACITY": "wind_cap", "EMBEDDED_SOLAR_GENERATION": "solar_gen",
                          "EMBEDDED_WIND_GENERATION": "wind_gen"})
    d = d[["date", "period", "nd", "solar_cap", "wind_cap", "solar_gen", "wind_gen"]]
    d = d.drop_duplicates(["date", "period"], keep="last").sort_values(["date", "period"]).reset_index(drop=True)
    # settlement periods count half-hours from local (UK) midnight; convert to an absolute UTC time
    local_midnight = d.date.dt.tz_localize("Europe/London", ambiguous=False, nonexistent="shift_forward")
    d["ts"] = (local_midnight + pd.to_timedelta((d.period - 1) * 30, unit="min")).dt.tz_convert("UTC")
    # "underlying" demand = what consumers used, before small solar/wind below the grid offset part of it
    d["underlying"] = d.nd + d.solar_gen + d.wind_gen
    d["solar_lf"] = d.solar_gen / d.solar_cap   # load factor: output per unit of capacity (stationary over years)
    d["wind_lf"] = d.wind_gen / d.wind_cap
    return d


def load_weather(kind: str, raw: Path = RAW) -> dict[str, pd.DataFrame]:
    """kind='actual' (ERA5 archive) or 'dayahead' (forecast issued the day before). Returns three half-hourly
    tables indexed by UTC time: 'demand' (population-weighted mean of five cities), 'solar' and 'wind' (plain means
    over sites where embedded solar / wind capacity is concentrated)."""
    cities = json.loads((raw / "cities.json").read_text())
    sites = json.loads((raw / "sites.json").read_text())
    out = {"demand": _mean(kind, {c: w for c, (_, _, w) in cities.items()}, raw)}
    for group in ("solar", "wind"):
        out[group] = _mean(kind, {s: 1.0 for s in sites[group]}, raw)
    return out


def _mean(kind: str, weights: dict, raw: Path) -> pd.DataFrame:
    total = sum(weights.values())
    acc = None
    for city, w in weights.items():
        h = json.loads((raw / f"weather_{kind}_{city}.json").read_text())["hourly"]
        df = pd.DataFrame({v: h[v if kind == "actual" else f"{v}_previous_day1"] for v in WVARS},
                          index=pd.to_datetime(h["time"]).tz_localize("UTC"), dtype=float)
        acc = df * (w / total) if acc is None else acc + df * (w / total)
    half = acc.resample("30min").interpolate("time")
    return half


def uk_holidays(years) -> set:
    hs = holidays.UK(subdiv="ENG", years=years) | holidays.UK(subdiv="SCT", years=years)
    return {pd.Timestamp(d) for d in hs}
