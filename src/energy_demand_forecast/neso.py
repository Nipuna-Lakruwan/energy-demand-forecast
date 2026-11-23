"""Compare with NESO's own day-ahead forecasts, which are published only at "cardinal points": fixed times (F),
peaks within a window (P) and troughs within a window (T)."""
import numpy as np
import pandas as pd

from .data import RAW


def load_neso(raw=RAW) -> pd.DataFrame:
    a = pd.read_csv(raw / "neso_day_ahead_archive.csv")
    a = a[a.DAYSAHEAD == 1].sort_values("FORECAST_TIMESTAMP")
    a = a.drop_duplicates(["TARGETDATE", "CARDINALPOINT"], keep="first")  # first issue of the day-ahead run
    a["date"] = pd.to_datetime(a.TARGETDATE)
    return a


def period_of(hhmm: int) -> int:
    return max(int(hhmm) // 100 * 2 + (int(hhmm) % 100) // 30, 1)


def cardinal(frame: pd.DataFrame, neso: pd.DataFrame, col: str) -> pd.DataFrame:
    """Apply each cardinal-point definition to `frame[col]` (indexed by date, period); returns one row per point."""
    x = frame.set_index(["date", "period"])[col]
    out = []
    for r in neso.itertuples():
        ps, pe = period_of(r.CP_ST_TIME), period_of(r.CP_END_TIME)
        try:
            w = x.loc[r.date].loc[ps:max(ps, pe)]
        except KeyError:
            continue
        if len(w) == 0 or w.isna().any():
            continue
        v = w.iloc[0] if r.CP_TYPE == "F" else (w.max() if r.CP_TYPE == "P" else w.min())
        out.append((r.date, r.CARDINALPOINT, v, r.FORECASTDEMAND))
    return pd.DataFrame(out, columns=["date", "point", col, "neso"])
