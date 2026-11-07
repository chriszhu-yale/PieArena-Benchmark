"""Baseline pairwise ranking methods compared against BBTL (Appendix: model comparison).

  SequentialElo     online Elo on continuous scores y in [0, 1]; order-dependent.
  LogisticBTL       Bradley-Terry-Luce MLE on binarized outcomes 1{y > 0.5}; order-invariant.
  CalibratedLogBTL  LogisticBTL plus Platt scaling to predict continuous pie shares.

All expose ``.fit(df)`` (columns ``lm_i``, ``lm_j``, ``y``) and ``.predict(lm_i, lm_j)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class SequentialElo:
    """Online Elo update on continuous-score outcomes.

    Standard chess Elo update is r_i ← r_i + K(S_i − E_i) where
    S_i ∈ {0, 0.5, 1} is the actual score and E_i = σ((r_i−r_j)/scale)
    is the expected score. We extend to continuous S_i = y_i ∈ [0, 1]
    (the observed pie share); this is the standard "rating-on-points"
    extension used by Glicko-style systems.

    Order-DEPENDENT: rating depends on the order of fit().
    """

    def __init__(self, K: float = 32.0, scale: float = 400.0, init: float = 1500.0):
        self.K = K
        self.scale = scale
        self.init = init
        self.ratings: Dict[str, float] = {}
        self.history: list = []

    def _expected(self, r_i: float, r_j: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_j - r_i) / self.scale))

    def fit(self, df: pd.DataFrame, lm_i: str = "lm_i", lm_j: str = "lm_j",
            y_col: str = "y") -> "SequentialElo":
        for _, row in df.iterrows():
            i, j = str(row[lm_i]), str(row[lm_j])
            r_i = self.ratings.get(i, self.init)
            r_j = self.ratings.get(j, self.init)
            E_i = self._expected(r_i, r_j)
            S_i = float(row[y_col])
            delta = self.K * (S_i - E_i)
            self.ratings[i] = r_i + delta
            self.ratings[j] = r_j - delta  # zero-sum
        return self

    def predict(self, lm_i: str, lm_j: str) -> float:
        return self._expected(
            self.ratings.get(lm_i, self.init),
            self.ratings.get(lm_j, self.init),
        )

    @property
    def leaderboard(self) -> pd.DataFrame:
        rows = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        return pd.DataFrame([{"lm": k, "rating": v} for k, v in rows])


@dataclass
class LogisticBTLResult:
    thetas: pd.DataFrame   # lm, theta, se, ci_lower, ci_upper
    log_likelihood: float
    n_obs: int
    n_params: int
    converged: bool


class LogisticBTL:
    """Classical BTL on binarized outcomes: Pr(i wins) = sigmoid(theta_i - theta_j); ties dropped.

    No first-mover or scenario-role effects. Order-invariant (batch likelihood)
    with Hessian-based CIs, but discards the continuous outcome information.
    """

    def __init__(self, reference: str):
        self.reference = reference
        self.result: Optional[LogisticBTLResult] = None

    def fit(self, df: pd.DataFrame, lm_i: str = "lm_i", lm_j: str = "lm_j",
            y_col: str = "y") -> "LogisticBTL":
        y = df[y_col].to_numpy(dtype=float)
        mask = np.abs(y - 0.5) > 1e-9
        df_use = df[mask]
        wins_i = (y[mask] > 0.5).astype(float)

        models = sorted(pd.unique(pd.concat([df_use[lm_i], df_use[lm_j]])))
        if self.reference not in models:
            raise ValueError(f"Reference '{self.reference}' not found.")
        model_idx = {m: i for i, m in enumerate(models)}
        ref_idx = model_idx[self.reference]
        free = np.array([k for k in range(len(models)) if k != ref_idx])
        n_free = len(free)
        full_to_free = -np.ones(len(models), dtype=int)
        full_to_free[free] = np.arange(n_free)
        i_idx = df_use[lm_i].map(model_idx).to_numpy()
        j_idx = df_use[lm_j].map(model_idx).to_numpy()
        pos_i, pos_j = full_to_free[i_idx], full_to_free[j_idx]
        valid_i, valid_j = pos_i >= 0, pos_j >= 0

        def nll_grad(theta_free):
            theta = np.zeros(len(models))
            theta[free] = theta_free
            p = np.clip(expit(theta[i_idx] - theta[j_idx]), 1e-12, 1 - 1e-12)
            ll = np.sum(wins_i * np.log(p) + (1 - wins_i) * np.log(1 - p))
            err = wins_i - p  # d ll / d eta
            grad = np.zeros(n_free)
            np.add.at(grad, pos_i[valid_i], err[valid_i])
            np.add.at(grad, pos_j[valid_j], -err[valid_j])
            return -ll, -grad

        res = minimize(nll_grad, np.zeros(n_free), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-9})

        # standard errors from a finite-difference Hessian of the analytic gradient
        eps = 1e-4
        H = np.zeros((n_free, n_free))
        _, g0 = nll_grad(res.x)
        for k in range(n_free):
            xp = res.x.copy()
            xp[k] += eps
            H[:, k] = (nll_grad(xp)[1] - g0) / eps
        H = 0.5 * (H + H.T)
        try:
            cov = np.linalg.inv(H + 1e-10 * np.eye(n_free))
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        theta_full, se_full = np.zeros(len(models)), np.zeros(len(models))
        theta_full[free], se_full[free] = res.x, se
        self.result = LogisticBTLResult(
            thetas=pd.DataFrame({"lm": models, "theta": theta_full, "se": se_full,
                                 "ci_lower": theta_full - 1.96 * se_full,
                                 "ci_upper": theta_full + 1.96 * se_full}),
            log_likelihood=-res.fun, n_obs=int(mask.sum()), n_params=n_free, converged=bool(res.success),
        )
        return self

    def predict(self, lm_i: str, lm_j: str) -> float:
        if self.result is None:
            raise ValueError("Call .fit() first.")
        th = self.result.thetas.set_index("lm")["theta"]
        if lm_i not in th.index or lm_j not in th.index:
            return 0.5
        return float(expit(th[lm_i] - th[lm_j]))

    @property
    def leaderboard(self) -> pd.DataFrame:
        if self.result is None:
            raise ValueError("Call .fit() first.")
        return self.result.thetas.sort_values("theta", ascending=False)


class CalibratedLogBTL:
    """Logistic BTL + Platt scaling for predicting continuous pie shares.

    Two-stage fit:
      Stage 1: standard LogBTL on binarized win/loss → θ̂.
      Stage 2: fit (a, b) on continuous y by minimizing
               soft cross-entropy ℓ(a,b) = −Σ [y log σ(a + b·Δθ̂) + (1−y) log(1−σ(...))],
               where Δθ̂ = θ̂_i − θ̂_j is held fixed from Stage 1.

    Prediction:
      ŷ(i, j) = σ(â + b̂ · (θ̂_i − θ̂_j))   ∈ (0, 1)   ← bounded by construction

    What this baseline isolates:
      * θ̂ still comes from a binary likelihood (Stage 1 throws away
        continuous info on how lopsided each match was).
      * The single (a, b) pair calibrates score-scale globally — it
        cannot model first-mover (γ), scenario-role (φ_s), or
        scenario-precision (κ_s) effects.
      * Any remaining gap to BBTL is therefore attributable to those
        structural components, not to a "binary vs continuous prediction"
        mismatch. This makes the comparison apples-to-apples.
    """

    def __init__(self, reference: str):
        self.btl = LogisticBTL(reference=reference)
        self.a: float = 0.0
        self.b: float = 1.0
        self.theta: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, lm_i: str = "lm_i", lm_j: str = "lm_j",
            y_col: str = "y") -> "CalibratedLogBTL":
        # Stage 1: standard LogBTL (uses binarized y internally)
        self.btl.fit(df, lm_i=lm_i, lm_j=lm_j, y_col=y_col)
        self.theta = self.btl.result.thetas.set_index("lm")["theta"]

        # Stage 2: calibrate (a, b) on continuous y
        delta = (df[lm_i].map(self.theta).to_numpy(dtype=float)
                 - df[lm_j].map(self.theta).to_numpy(dtype=float))
        y = df[y_col].to_numpy(dtype=float)
        # Numerical safety: confine y away from {0, 1}
        y_safe = np.clip(y, 1e-6, 1 - 1e-6)

        def nll(ab):
            a, b = ab
            mu = expit(a + b * delta)
            mu = np.clip(mu, 1e-12, 1 - 1e-12)
            return -np.sum(y_safe * np.log(mu) + (1 - y_safe) * np.log(1 - mu))

        res = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, lm_i_val: str, lm_j_val: str) -> float:
        if self.theta is None:
            raise ValueError("Call .fit() first.")
        if lm_i_val not in self.theta.index or lm_j_val not in self.theta.index:
            return 0.5
        delta = float(self.theta[lm_i_val] - self.theta[lm_j_val])
        return float(expit(self.a + self.b * delta))

    @property
    def leaderboard(self) -> pd.DataFrame:
        if self.theta is None:
            raise ValueError("Call .fit() first.")
        return self.btl.leaderboard
