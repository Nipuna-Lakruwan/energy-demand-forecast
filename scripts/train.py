"""Train and evaluate the forecasters; write results/metrics.json and test predictions.

Split by date: train 2018-01-15 .. 2024-10-31, validation 2024-11-01 .. 2025-10-31, test 2025-11-01 .. 2026-10-31.
Validation and test use weather forecasts issued the day before; training uses actual weather.
"""
import json

import numpy as np
import pandas as pd

from energy_demand_forecast.data import ROOT, load_demand, load_weather
from energy_demand_forecast.features import FEATURES, FEATURES_SOLAR, FEATURES_U, FEATURES_WIND, build
from energy_demand_forecast.metrics import day_bootstrap_mape, scores
from energy_demand_forecast.models import Decomposed, DirectQuantile
from energy_demand_forecast.neso import cardinal, load_neso

OUT, MOD = ROOT / "results", ROOT / "models"
TRAIN_START, VAL_START, TEST_START, TEST_END = "2018-01-15", "2024-11-01", "2025-11-01", "2026-10-31"
NEED = sorted(set(FEATURES + FEATURES_U + FEATURES_SOLAR + FEATURES_WIND + ["nd", "underlying", "solar_lf", "wind_lf"]))


def main() -> None:
    OUT.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
    d, wa, wf = load_demand(), load_weather("actual"), load_weather("dayahead")
    f_act = build(d, wa).dropna(subset=NEED)
    f_fc = build(d, wa, wf).dropna(subset=NEED)
    tr = f_act[(f_act.date >= TRAIN_START) & (f_act.date < VAL_START)]
    va = f_fc[(f_fc.date >= VAL_START) & (f_fc.date < TEST_START)]
    te = f_fc[(f_fc.date >= TEST_START) & (f_fc.date <= TEST_END)]
    te_oracle = f_act.loc[f_act.index.intersection(te.index)]
    trva = pd.concat([tr, f_act[(f_act.date >= VAL_START) & (f_act.date < TEST_START)]])
    res = {"rows": {"train": len(tr), "val": len(va), "test": len(te)}, "validation": {}, "test": {}}

    # ---- validation ----
    cand = {}
    for leaves in (31, 127):
        m = DirectQuantile(leaves).fit(tr, va)
        res["validation"][f"direct/leaves{leaves}"] = {**scores(va.nd.to_numpy(), m.predict(va), va.nd_lag7d.to_numpy()), "iters": m.iters}
        cand[("direct", leaves)] = (res["validation"][f"direct/leaves{leaves}"]["mape_pct"], m.iters)
        m = Decomposed(leaves).fit(tr, va).calibrate(va)
        res["validation"][f"decomposed/leaves{leaves}"] = {**scores(va.nd.to_numpy(), m.predict(va), va.nd_lag7d.to_numpy()), "iters": m.iters}
        cand[("decomposed", leaves)] = (res["validation"][f"decomposed/leaves{leaves}"]["mape_pct"], m.iters)
    for k, v in res["validation"].items():
        print(f"val {k:22s} MAPE {v['mape_pct']:.2f}% cov {v['coverage_p10_p90']:.2f} iters {v['iters']}", flush=True)
    best = {kind: min(((l, *cand[(kind, l)]) for l in (31, 127)), key=lambda t: t[1]) for kind in ("direct", "decomposed")}
    res["chosen"] = {k: {"leaves": v[0], "iters": v[2]} for k, v in best.items()}

    # conformal residuals come from the train-only decomposed model on validation (data it never saw)
    calib = Decomposed(best["decomposed"][0]).fit(tr, iters=best["decomposed"][2]).calibrate(va)

    # ---- refit on train+val, score test once ----
    scale = lambda it: {k: int(v * 1.1) for k, v in it.items()}
    direct = DirectQuantile(best["direct"][0]).fit(trva, iters=scale(best["direct"][2]))
    dec = Decomposed(best["decomposed"][0]).fit(trva, iters=scale(best["decomposed"][2]))
    dec.q = calib.q
    y, naive, days = te.nd.to_numpy(), te.nd_lag7d.to_numpy(), te.date.to_numpy()
    preds = {"seasonal_naive": np.column_stack([naive] * 3), "direct_lightgbm": direct.predict(te),
             "decomposed": dec.predict(te)}
    yo = te_oracle.nd.to_numpy()
    for name, P in preds.items():
        s = scores(y, P, naive); s["mape_ci95"] = day_bootstrap_mape(y, P[:, 1], days)
        if name == "seasonal_naive":
            for k in ("coverage_p10_p90", "width_p10_p90_mw", "pinball_mw"):
                s.pop(k)
        res["test"][name] = s
        print(f"TEST {name:18s} MAPE {s['mape_pct']:.2f}% {s['mape_ci95']} MASE {s['mase_vs_week_ago']:.3f} cov {s.get('coverage_p10_p90')}", flush=True)
    res["test"]["decomposed_actual_weather"] = {"mape_pct": scores(yo, dec.predict(te_oracle), te_oracle.nd_lag7d.to_numpy())["mape_pct"]}
    u, s_, w_ = dec.components(te)
    res["test"]["decomposed_components"] = {
        "underlying_mape_pct": float(np.mean(np.abs(u - te.underlying) / te.underlying) * 100),
        "solar_mae_mw": float(np.mean(np.abs(s_ - te.solar_gen))), "wind_mae_mw": float(np.mean(np.abs(w_ - te.wind_gen)))}
    print("components", res["test"]["decomposed_components"], "oracle weather", res["test"]["decomposed_actual_weather"], flush=True)

    out = te[["date", "period", "ts", "nd", "solar_gen", "wind_gen"]].copy()
    for name in ("direct_lightgbm", "decomposed"):
        out[f"{name}_p10"], out[f"{name}_p50"], out[f"{name}_p90"] = preds[name].T
    out["naive"] = naive
    out.to_parquet(OUT / "test_predictions.parquet", index=False)

    # ---- NESO comparison at cardinal points ----
    neso = load_neso(); neso = neso[(neso.date >= TEST_START) & (neso.date <= TEST_END)]
    cp = cardinal(out, neso, "nd")
    for col in ("decomposed_p50", "direct_lightgbm_p50", "naive"):
        cp = cp.merge(cardinal(out, neso, col).drop(columns="neso"), on=["date", "point"])
    cmp = {"points": int(len(cp)), "days": int(cp.date.nunique())}
    for col in ("neso", "decomposed_p50", "direct_lightgbm_p50", "naive"):
        cmp[col] = float(np.mean(np.abs(cp[col] - cp.nd) / cp.nd) * 100)
    cp["d"] = (np.abs(cp.decomposed_p50 - cp.nd) - np.abs(cp.neso - cp.nd)) / cp.nd * 100
    g = cp.groupby("date").d.agg(["sum", "count"]); rng = np.random.default_rng(0)
    draws = [g["sum"].values[i].sum() / g["count"].values[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
    cmp["decomposed_minus_neso_pts"] = float(cp.d.mean())
    cmp["ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    cmp["share_of_points_we_beat_neso"] = float((cp.d < 0).mean())
    res["vs_neso_cardinal_points"] = cmp
    cp.to_parquet(OUT / "cardinal_points.parquet", index=False)
    print("NESO", json.dumps(cmp), flush=True)

    imp = dec.models["u"].booster_.feature_importance("gain")
    res["underlying_feature_gain_share"] = dict(sorted(zip(FEATURES_U, (imp / imp.sum()).round(4).tolist()), key=lambda kv: -kv[1]))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    for k, m in dec.models.items():
        m.booster_.save_model(str(MOD / f"decomposed_{k}.txt"))
    (MOD / "config.json").write_text(json.dumps({"features_underlying": FEATURES_U, "features_solar": FEATURES_SOLAR,
        "features_wind": FEATURES_WIND, "conformal_relative_residual_quantiles_by_4h_block": dec.q,
        "nd = underlying - clip(solar_lf,0,1)*solar_cap_d2 - clip(wind_lf,0,1)*wind_cap_d2": True}, indent=2, default=float))
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
