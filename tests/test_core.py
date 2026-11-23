import numpy as np
import pandas as pd

from energy_demand_forecast.features import FEATURES, FEATURES_U, build
from energy_demand_forecast.metrics import day_bootstrap_mape, pinball, scores
from energy_demand_forecast.neso import cardinal, period_of


def toy(days=30):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    d = pd.DataFrame([(dt, p) for dt in dates for p in range(1, 49)], columns=["date", "period"])
    d["nd"] = 20000 + 100 * d.period + d.date.dt.day * 10.0
    d["solar_cap"], d["wind_cap"], d["solar_gen"], d["wind_gen"] = 20000.0, 6000.0, 0.0, 1000.0
    d["underlying"] = d.nd + d.solar_gen + d.wind_gen
    d["ts"] = (d.date.dt.tz_localize("UTC") + pd.to_timedelta((d.period - 1) * 30, unit="min"))
    w = pd.DataFrame({"temperature_2m": 10.0, "shortwave_radiation": 0.0, "wind_speed_10m": 5.0, "cloud_cover": 50.0},
                     index=pd.date_range("2024-12-01", "2025-03-01", freq="30min", tz="UTC"))
    return d, {"demand": w, "solar": w, "wind": w}


def test_lags_only_use_past_information():
    d, w = toy()
    f = build(d, w).set_index(["date", "period"])
    row = f.loc[(pd.Timestamp("2025-01-20"), 30)]
    assert row.nd_lag2d == 20000 + 100 * 30 + 18 * 10
    assert row.nd_lag7d == 20000 + 100 * 30 + 13 * 10
    early_d1 = np.mean([20000 + 100 * p + 19 * 10 for p in range(1, 15)])
    assert np.isclose(row.nd_d1_early_mean, early_d1)  # D-1 only up to 07:00
    assert set(FEATURES + FEATURES_U) <= set(build(d, w).columns)
    assert row.u_lag2d == row.nd_lag2d + 1000


def test_metrics():
    y = np.array([100.0, 200.0]); P = np.array([[90, 100, 110], [150, 220, 230.0]])
    s = scores(y, P, naive=np.array([110.0, 180.0]))
    assert np.isclose(s["mape_pct"], 5.0) and np.isclose(s["mase_vs_week_ago"], 10 / 15)
    assert s["coverage_p10_p90"] == 1.0
    assert pinball(y, np.column_stack([y, y, y])) == 0
    lo, hi = day_bootstrap_mape(y, P[:, 1], np.array([1, 2]))
    assert 0 <= lo <= hi <= 10


def test_cardinal_points():
    assert period_of(30) == 1 and period_of(1230) == 25 and period_of(0) == 1
    frame = pd.DataFrame({"date": pd.Timestamp("2025-01-01"), "period": range(1, 49), "x": np.arange(48.0)})
    neso = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")] * 3, "CARDINALPOINT": ["a", "b", "c"],
                         "CP_TYPE": ["F", "P", "T"], "CP_ST_TIME": [100, 1000, 1000], "CP_END_TIME": [100, 1200, 1200],
                         "FORECASTDEMAND": [1, 2, 3]})
    out = cardinal(frame, neso, "x").set_index("point").x
    assert out["a"] == 1 and out["b"] == 23 and out["c"] == 19
