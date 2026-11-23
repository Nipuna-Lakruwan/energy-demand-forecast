"""Figures from results/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_demand_forecast.data import ROOT

FIG, RES = ROOT / "figures", ROOT / "results"
C = {"decomposed": "#DD8452", "direct_lightgbm": "#4C72B0", "naive": "#8C8C8C", "neso": "#55A868"}


def week(ax, p, start, title):
    w = p[(p.date >= start) & (p.date < pd.Timestamp(start) + pd.Timedelta(days=7))]
    t = w.ts.dt.tz_convert("Europe/London")
    ax.fill_between(t, w.decomposed_p10 / 1000, w.decomposed_p90 / 1000, color=C["decomposed"], alpha=.25, label="P10–P90")
    ax.plot(t, w.nd / 1000, color="k", lw=1.4, label="Actual")
    ax.plot(t, w.decomposed_p50 / 1000, color=C["decomposed"], lw=1.2, label="Forecast (P50)")
    ax.set(title=title, ylabel="GW"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz="Europe/London"))
    ax.tick_params(axis="x", labelsize=8)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    m = json.loads((RES / "metrics.json").read_text())
    p = pd.read_parquet(RES / "test_predictions.parquet")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    week(axes[0], p, "2026-01-12", "Winter week (test year)")
    week(axes[1], p, "2026-06-15", "Summer week (test year): midday dip = rooftop solar")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "example_weeks.png", dpi=150); plt.close(fig)

    p["month"] = p.date.dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, lab in (("naive", "Same time last week"), ("direct_lightgbm_p50", "Direct LightGBM"), ("decomposed_p50", "Decomposed")):
        e = (np.abs(p[col] - p.nd) / p.nd * 100).groupby(p.month).mean()
        ax.plot(e.index, e.values, "o-", label=lab, color=C[col.replace("_p50", "")])
    ax.set(ylabel="MAPE (%)", title="Error by month, test year (day-ahead, forecast weather)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "mape_by_month.png", dpi=150); plt.close(fig)

    cp = pd.read_parquet(RES / "cardinal_points.parquet")
    cp["month"] = cp.date.dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, lab in (("neso", "NESO (grid operator)"), ("decomposed_p50", "This project (decomposed)"), ("naive", "Same time last week")):
        e = (np.abs(cp[col] - cp.nd) / cp.nd * 100).groupby(cp.month).mean()
        ax.plot(e.index, e.values, "o-", label=lab, color=C[col.replace("_p50", "")])
    v = m["vs_neso_cardinal_points"]
    ax.set(ylabel="MAPE at cardinal points (%)",
           title=f"Vs. NESO's day-ahead forecast: {v['neso']:.2f}% vs {v['decomposed_p50']:.2f}% over {v['days']} days")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "vs_neso.png", dpi=150); plt.close(fig)

    # decomposition on the sunniest test day
    day = (p.groupby("date").solar_gen.max()).idxmax()
    w = p[p.date == day]; t = w.ts.dt.tz_convert("Europe/London")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, (w.nd + w.solar_gen + w.wind_gen) / 1000, color="#4C72B0", label="Underlying demand (ND + embedded solar + wind)")
    ax.fill_between(t, w.nd / 1000, (w.nd + w.solar_gen) / 1000, color="#F2C14E", alpha=.6, label="Embedded solar")
    ax.plot(t, w.nd / 1000, color="k", label="National demand (what the grid sees)")
    ax.plot(t, w.decomposed_p50 / 1000, color=C["decomposed"], ls="--", label="Forecast")
    ax.set(ylabel="GW", title=f"Why decompose: {pd.Timestamp(day):%d %b %Y}, sunniest test day")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="Europe/London")); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "decomposition_day.png", dpi=150); plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
