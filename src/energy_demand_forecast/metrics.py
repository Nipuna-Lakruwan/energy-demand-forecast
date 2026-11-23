"""Point and probabilistic forecast metrics, with a day-block bootstrap."""
import numpy as np

from .models import QUANTILES


def pinball(y, P):
    q = np.array(QUANTILES)[None, :]
    e = y[:, None] - P
    return float(np.maximum(q * e, (q - 1) * e).mean())


def scores(y, P, naive) -> dict:
    p50 = P[:, 1]
    return {"mape_pct": float(np.mean(np.abs(p50 - y) / y) * 100), "mae_mw": float(np.mean(np.abs(p50 - y))),
            "mase_vs_week_ago": float(np.mean(np.abs(p50 - y)) / np.mean(np.abs(naive - y))),
            "pinball_mw": pinball(y, P), "coverage_p10_p90": float(np.mean((y >= P[:, 0]) & (y <= P[:, 2]))),
            "width_p10_p90_mw": float(np.mean(P[:, 2] - P[:, 0]))}


def day_bootstrap_mape(y, p50, days, n=1000, seed=0):
    """95% CI for MAPE, resampling whole days (half-hours within a day are strongly correlated)."""
    ape = np.abs(p50 - y) / y
    uniq, inv = np.unique(days, return_inverse=True)
    s, c = np.bincount(inv, ape), np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = [s[i].sum() / c[i].sum() * 100 for i in (rng.integers(0, len(uniq), len(uniq)) for _ in range(n))]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
