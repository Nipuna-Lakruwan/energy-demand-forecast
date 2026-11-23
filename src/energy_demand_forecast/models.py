"""Forecasters producing P10/P50/P90 for national demand (ND).

- DirectQuantile: three LightGBM quantile models on ND directly.
- Decomposed: ND = underlying demand - embedded solar - embedded wind. Underlying demand gets a LightGBM (L1) model;
  solar and wind get load-factor models (output per MW of capacity, which is stable as capacity grows), multiplied by
  the capacity known on day D-2. Intervals come from split-conformal calibration on validation residuals,
  separately for each 4-hour block of the day.
"""
import lightgbm as lgb
import numpy as np

from .features import FEATURES, FEATURES_SOLAR, FEATURES_U, FEATURES_WIND

QUANTILES = (0.1, 0.5, 0.9)
PARAMS = dict(learning_rate=0.05, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              verbose=-1, n_jobs=4)


def _fit(X, y, objective, iters, leaves=31, alpha=None, seed=0, X_val=None, y_val=None):
    kw = {"alpha": alpha} if alpha is not None else {}
    m = lgb.LGBMRegressor(objective=objective, n_estimators=iters, num_leaves=leaves, random_state=seed, **PARAMS, **kw)
    if X_val is not None:
        m.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(X, y)
    return m


class DirectQuantile:
    def __init__(self, leaves=31):
        self.leaves, self.iters = leaves, {}

    def fit(self, tr, va=None, iters=None):
        self.models = {}
        for q in QUANTILES:
            n = (iters or {}).get(q, 3000)
            m = _fit(tr[FEATURES], tr.nd, "quantile", n, self.leaves, alpha=q,
                     X_val=None if va is None else va[FEATURES], y_val=None if va is None else va.nd)
            self.iters[q] = m.best_iteration_ if va is not None else n
            self.models[q] = m
        return self

    def predict(self, df):
        return np.sort(np.column_stack([self.models[q].predict(df[FEATURES]) for q in QUANTILES]), axis=1)


class Decomposed:
    BLOCKS = 6  # 4-hour blocks of the day for conformal calibration

    def __init__(self, leaves=31):
        self.leaves, self.iters = leaves, {}

    def fit(self, tr, va=None, iters=None):
        it = iters or {}
        specs = {"u": (FEATURES_U, "underlying", "l1"), "solar": (FEATURES_SOLAR, "solar_lf", "l2"),
                 "wind": (FEATURES_WIND, "wind_lf", "l2")}
        self.models = {}
        for k, (F, col, obj) in specs.items():
            m = _fit(tr[F], tr[col], obj, it.get(k, 3000), self.leaves,
                     X_val=None if va is None else va[F], y_val=None if va is None else va[col])
            self.iters[k] = m.best_iteration_ if va is not None else it.get(k, 3000)
            self.models[k] = m
        return self

    def components(self, df):
        u = self.models["u"].predict(df[FEATURES_U])
        s = np.clip(self.models["solar"].predict(df[FEATURES_SOLAR]), 0, 1) * df.solar_cap_d2.to_numpy()
        w = np.clip(self.models["wind"].predict(df[FEATURES_WIND]), 0, 1) * df.wind_cap_d2.to_numpy()
        return u, s, w

    def point(self, df):
        u, s, w = self.components(df)
        return u - s - w

    def calibrate(self, df):
        """Store the 10th/90th percentile of relative residuals per block, from data the model was not fitted on."""
        r = (df.nd.to_numpy() - self.point(df)) / df.nd.to_numpy()
        b = ((df.period.to_numpy() - 1) // 8).clip(0, self.BLOCKS - 1)
        self.q = {k: (np.quantile(r[b == k], 0.1), np.quantile(r[b == k], 0.9)) for k in range(self.BLOCKS)}
        return self

    def predict(self, df):
        p = self.point(df)
        b = ((df.period.to_numpy() - 1) // 8).clip(0, self.BLOCKS - 1)
        lo = np.array([self.q[k][0] for k in b]); hi = np.array([self.q[k][1] for k in b])
        # relative residual r = (y - p) / y  =>  y = p / (1 - r)
        return np.column_stack([p / (1 - lo), p, p / (1 - hi)])
