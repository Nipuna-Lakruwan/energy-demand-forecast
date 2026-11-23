"""Features for a day-ahead forecast issued at about 08:00 UK time on day D-1 for all half-hours of day D.

Information allowed at issue time: demand up to the end of day D-2, demand on D-1 from midnight to 07:00
(periods 1-14), calendar facts, embedded capacities as of D-2, and a weather *forecast* for day D.
Weather for past days is the actual weather.
"""
import numpy as np
import pandas as pd

from .data import WVARS, uk_holidays


def _lag(df, col, days):
    key = df.set_index(["date", "period"])[col]
    return key.reindex(pd.MultiIndex.from_arrays([df.date - pd.Timedelta(days=days), df.period])).to_numpy()


def build(demand: pd.DataFrame, weather_actual: dict, weather_target: dict | None = None) -> pd.DataFrame:
    """One row per (date, period). `weather_target` supplies weather for the target half-hour (pass the day-ahead
    forecast to be honest; defaults to actual weather, which is what training uses). Lag features are computed for
    both national demand (`nd_*`) and underlying demand (`u_*`)."""
    df = demand.copy()
    weather_target = weather_actual if weather_target is None else weather_target
    wt = weather_target["demand"].reindex(df.ts)
    for v in WVARS:
        df[v] = wt[v].to_numpy()
    ws, ww = weather_target["solar"].reindex(df.ts), weather_target["wind"].reindex(df.ts)
    df["s_radiation"], df["s_cloud"], df["s_temp"] = ws.shortwave_radiation.to_numpy(), ws.cloud_cover.to_numpy(), ws.temperature_2m.to_numpy()
    df["w_speed"] = ww.wind_speed_10m.to_numpy()
    df["s_rad_day_mean"] = df.groupby("date").s_radiation.transform("mean")
    df["hdd"] = np.clip(15.5 - df.temperature_2m, 0, None)
    df["cdd"] = np.clip(df.temperature_2m - 20, 0, None)
    df["temp_day_mean"] = df.groupby("date").temperature_2m.transform("mean")
    df["rad_day_mean"] = df.groupby("date").shortwave_radiation.transform("mean")

    for col, p in (("nd", "nd"), ("underlying", "u")):
        for lag in (2, 7, 14):
            df[f"{p}_lag{lag}d"] = _lag(df, col, lag)
        day = df.groupby("date")[col].agg(["mean", "max", "min"])
        for stat in ("mean", "max", "min"):
            df[f"{p}_d2_{stat}"] = day[stat].reindex(df.date - pd.Timedelta(days=2)).to_numpy()
        early = df[df.period <= 14].groupby("date")[col].mean()
        df[f"{p}_d1_early_mean"] = early.reindex(df.date - pd.Timedelta(days=1)).to_numpy()
        df[f"{p}_d2_early_mean"] = early.reindex(df.date - pd.Timedelta(days=2)).to_numpy()

    wa = weather_actual["demand"].reindex(df.ts - pd.Timedelta(days=2))
    df["temp_lag2d"] = wa.temperature_2m.to_numpy()
    df["temp_delta_2d"] = df.temperature_2m - df.temp_lag2d

    caps = df.groupby("date")[["solar_cap", "wind_cap"]].mean()
    for c in ("solar_cap", "wind_cap"):
        df[f"{c}_d2"] = caps[c].reindex(df.date - pd.Timedelta(days=2)).to_numpy()

    hol = uk_holidays(range(df.date.dt.year.min(), df.date.dt.year.max() + 1))
    df["holiday"] = df.date.isin(hol).astype(int)
    df["holiday_lag2d"] = (df.date - pd.Timedelta(days=2)).isin(hol).astype(int)
    df["holiday_lag7d"] = (df.date - pd.Timedelta(days=7)).isin(hol).astype(int)
    df["dow"] = df.date.dt.dayofweek
    doy = df.date.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["xmas"] = ((df.date.dt.month == 12) & (df.date.dt.day >= 20) | (df.date.dt.month == 1) & (df.date.dt.day <= 3)).astype(int)
    df["year_frac"] = df.date.dt.year + doy / 366
    return df


CALENDAR = ["period", "dow", "holiday", "holiday_lag2d", "holiday_lag7d", "xmas", "doy_sin", "doy_cos", "year_frac"]
WEATHER = ["hdd", "cdd", "temp_day_mean", "rad_day_mean", "temp_lag2d", "temp_delta_2d"] + WVARS


def lag_features(p: str) -> list[str]:
    return [f"{p}_lag2d", f"{p}_lag7d", f"{p}_lag14d", f"{p}_d2_mean", f"{p}_d2_max", f"{p}_d2_min",
            f"{p}_d1_early_mean", f"{p}_d2_early_mean"]


# direct model of national demand (the simple approach)
FEATURES = CALENDAR + WEATHER + lag_features("nd") + ["solar_cap_d2", "wind_cap_d2"]
# decomposed approach: underlying demand, plus solar and wind load factors
FEATURES_U = CALENDAR + WEATHER + lag_features("u")
FEATURES_SOLAR = ["period", "doy_sin", "doy_cos", "s_radiation", "s_cloud", "s_temp", "s_rad_day_mean",
                  "shortwave_radiation", "cloud_cover"]
FEATURES_WIND = ["period", "doy_sin", "doy_cos", "w_speed", "wind_speed_10m", "temperature_2m"]
