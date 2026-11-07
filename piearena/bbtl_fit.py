"""
BBTL — Beta-Generalized Bradley-Terry-Luce model (NeurIPS 2026 paper).

Model:
    y_k* ~ Beta(μ_k κ_{s_k}, (1 - μ_k) κ_{s_k})
    logit(μ_k) = (θ_{i_k} - θ_{j_k}) + γ x^{(F)}_k + φ_{s_k}
    log(κ_{s_k}) = α_0 + α_{s_k}                     (per-scenario precision)

Estimation: scipy.optimize.minimize on the negative log-likelihood with
analytic gradients (digamma-based). Two specifications are supported:
  - 'constant':       single κ
  - 'per_scenario':   κ varies by scenario (3 separate κ_s)

Identifiability: θ[reference] = 0 (anchor); first scenario coded as
baseline for both φ and α (so we estimate (S-1) of each as free).

We use a hand-written fitter rather than statsmodels.othermod.betareg.BetaModel
because the latter does not support a separate design matrix for the
precision sub-model (per-scenario precision is required here).
A direct analytic NLL + gradient gives exact control over parameter layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
SCENARIOS = ("Hire_Talent", "IP_License", "Service_Quote")  # cross-play scenarios in the paper
SV_EPS = 1e-12  # numerical safety floor on y*
LOG_KAPPA_LO, LOG_KAPPA_HI = -6.0, 12.0  # bounds on log(κ_s)


def _logistic_safe(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, SV_EPS, 1 - SV_EPS)
    return np.log(y / (1.0 - y))


def smithson_verkuilen(d: np.ndarray) -> np.ndarray:
    """y* = (y(N-1) + 1/2)/N with y = (d+1)/2."""
    y = (d + 1.0) / 2.0
    N = len(y)
    if N <= 1:
        return y
    return (y * (N - 1) + 0.5) / N


def _who_first_to_sign(who_first: pd.Series) -> np.ndarray:
    out = np.empty(len(who_first), dtype=float)
    for k, v in enumerate(who_first.astype(object).to_numpy()):
        s = str(v).strip().lower()
        if s in {"side1", "i", "true", "1"}:
            out[k] = +1.0
        elif s in {"side2", "j", "false", "0"}:
            out[k] = -1.0
        else:
            raise ValueError(f"Cannot parse who_first: {v}")
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BBTLResult:
    thetas: pd.DataFrame                 # lm, theta, se, ci_lower, ci_upper
    gamma: float
    gamma_se: float
    gamma_ci: Tuple[float, float]
    scenario_effects: pd.DataFrame       # scenario, phi, se, ci_lower, ci_upper
    kappa: pd.DataFrame                  # scenario, kappa, se, ci_lower, ci_upper
    cov_all: pd.DataFrame
    log_likelihood: float
    n_params: int
    n_obs: int
    converged: bool
    n_iter: int
    spec: str                            # 'constant' or 'per_scenario'
    models: List[str]
    reference: str


# ---------------------------------------------------------------------------
# Negative log-likelihood (analytic gradient)
# ---------------------------------------------------------------------------
def _build_design(
    df: pd.DataFrame,
    reference: str,
    lm_col_i: str,
    lm_col_j: str,
    scenario_col: str,
    who_first_col: str,
    scenarios: tuple = SCENARIOS,
):
    """Build the design matrices/indices used inside the NLL."""
    models = sorted(pd.unique(pd.concat([df[lm_col_i], df[lm_col_j]])))
    if reference not in models:
        raise ValueError(f"Reference '{reference}' not found.")
    model_to_idx = {m: i for i, m in enumerate(models)}
    ref_idx = model_to_idx[reference]
    free_models = [m for m in models if m != reference]
    Nfree = len(free_models)

    lm_i_idx = df[lm_col_i].map(model_to_idx).to_numpy()
    lm_j_idx = df[lm_col_j].map(model_to_idx).to_numpy()

    scen_to_idx = {sv: s for s, sv in enumerate(scenarios)}
    s_idx = df[scenario_col].astype(str).map(scen_to_idx).to_numpy()
    if np.any(pd.isna(s_idx)):
        raise ValueError("Unknown scenarios in data.")
    s_idx = s_idx.astype(int)

    xF = _who_first_to_sign(df[who_first_col])
    return models, ref_idx, free_models, Nfree, lm_i_idx, lm_j_idx, s_idx, xF


def _nll_grad(params, ystar,
              lm_i_idx, lm_j_idx, ref_idx, Nfree,
              s_idx, xF, S, spec):
    """
    Negative log-likelihood and analytic gradient.

    Layout of params:
      [θ_-r (Nfree), γ (1), φ_1..φ_S (all S, no baseline), log_κ ...]
    All S φ_s are identifiable because the linear predictor has no
    intercept term — the anchor θ[ref]=0 fully pins location.
    For spec='constant', log_κ has 1 element.
    For spec='per_scenario', log_κ has S elements (one per scenario).
    """
    # unpack
    theta_free = params[:Nfree]
    gamma = params[Nfree]
    phi = params[Nfree + 1: Nfree + 1 + S]              # all S free
    rest = params[Nfree + 1 + S:]
    if spec == "constant":
        log_kappa = np.full(S, rest[0])
    else:
        log_kappa = np.clip(rest, LOG_KAPPA_LO, LOG_KAPPA_HI)
    kappa = np.exp(log_kappa)  # length S

    # full θ vector
    theta = np.zeros(Nfree + 1)
    free_positions = [i for i in range(Nfree + 1) if i != ref_idx]
    theta[free_positions] = theta_free
    # θ[ref_idx] = 0 by anchor

    # linear predictor
    eta = (theta[lm_i_idx] - theta[lm_j_idx]) + gamma * xF + phi[s_idx]
    mu = _logistic_safe(eta)
    mu = np.clip(mu, SV_EPS, 1 - SV_EPS)
    kap_k = kappa[s_idx]
    a = mu * kap_k
    b = (1.0 - mu) * kap_k

    log_y = np.log(np.clip(ystar, SV_EPS, 1.0))
    log_1my = np.log(np.clip(1.0 - ystar, SV_EPS, 1.0))

    ll_per = (gammaln(kap_k) - gammaln(a) - gammaln(b)
              + (a - 1.0) * log_y + (b - 1.0) * log_1my)
    nll = -float(np.sum(ll_per))

    # ----- Gradients -----
    # dμ/dη = μ(1-μ)
    dmu_deta = mu * (1.0 - mu)

    # ∂ℓ_k / ∂μ_k  (for fixed κ_k)
    # = κ [ψ((1-μ)κ) - ψ(μκ) + log(y) - log(1-y)]
    dl_dmu = kap_k * (digamma(b) - digamma(a) + log_y - log_1my)

    # ∂ℓ_k / ∂η_k = dl_dmu * dmu_deta
    dl_deta = dl_dmu * dmu_deta

    # ----- gradient w.r.t. θ_free, γ, φ -----
    grad = np.zeros_like(params)

    # θ contribution: each obs k contributes +dl_deta[k] to lm_i, -dl_deta[k] to lm_j
    # but only for entries that are NOT the anchor.
    free_idx_arr = np.array([i for i in range(Nfree + 1) if i != ref_idx])
    full_to_free = -np.ones(Nfree + 1, dtype=int)
    for pos, full in enumerate(free_idx_arr):
        full_to_free[full] = pos

    pos_i = full_to_free[lm_i_idx]
    pos_j = full_to_free[lm_j_idx]
    valid_i = pos_i >= 0
    valid_j = pos_j >= 0

    g_theta = np.zeros(Nfree)
    np.add.at(g_theta, pos_i[valid_i], dl_deta[valid_i])
    np.add.at(g_theta, pos_j[valid_j], -dl_deta[valid_j])
    grad[:Nfree] = g_theta

    # γ
    grad[Nfree] = float(np.sum(dl_deta * xF))

    # all S φ_s
    g_phi = np.zeros(S)
    for s in range(S):
        mask = s_idx == s
        g_phi[s] = float(np.sum(dl_deta[mask]))
    grad[Nfree + 1: Nfree + 1 + S] = g_phi

    # ----- gradient w.r.t. log_κ -----
    # ∂ℓ_k / ∂κ_k = ψ(κ) - μ ψ(μκ) - (1-μ)ψ((1-μ)κ) + μ log y + (1-μ) log(1-y)
    # ∂ℓ_k / ∂log κ_k = κ_k * (above)
    dl_dkappa = (digamma(kap_k) - mu * digamma(a) - (1 - mu) * digamma(b)
                 + mu * log_y + (1 - mu) * log_1my)
    dl_dlogk = kap_k * dl_dkappa

    if spec == "constant":
        grad[-1] = float(np.sum(dl_dlogk))
    else:
        for s in range(S):
            mask = s_idx == s
            grad[-(S - s)] = float(np.sum(dl_dlogk[mask]))

    # we minimize NLL → flip sign
    return nll, -grad


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_bbtl(
    df: pd.DataFrame,
    reference: str,
    spec: Literal["constant", "per_scenario"] = "per_scenario",
    include_gamma: bool = True,
    include_phi: bool = True,
    lm_col_i: str = "lm_i",
    lm_col_j: str = "lm_j",
    share_col_i: str = "p_i",
    share_col_j: str = "p_j",
    scenario_col: str = "scenario",
    who_first_col: str = "who_first",
    scenarios: tuple = SCENARIOS,
    max_iter: int = 1000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> BBTLResult:
    """Fit BBTL on Filter-A-cleaned data.

    Set include_gamma=False to constrain γ=0 (no first-mover effect).
    Set include_phi=False to constrain all φ_s=0 (no scenario-role effects).
    These flags enable component-by-component ablation studies.
    """
    pi = df[share_col_i].to_numpy(dtype=float)
    pj = df[share_col_j].to_numpy(dtype=float)
    assert np.all((pi >= 0) & (pi <= 1)) and np.all((pj >= 0) & (pj <= 1)), \
        "Apply Filter A first (p_i, p_j must be in [0,1])."
    d = pi - pj
    ystar = smithson_verkuilen(d)

    (models, ref_idx, free_models, Nfree,
     lm_i_idx, lm_j_idx, s_idx, xF) = _build_design(
        df, reference, lm_col_i, lm_col_j, scenario_col, who_first_col, scenarios
    )
    S = len(scenarios)

    # ----- initial values -----
    # θ_free: 0  ; γ: 0 ; φ_free: 0 ; log_κ: log(MoM)
    mu_bar = float(np.mean(ystar))
    var_y = float(np.var(ystar))
    kappa_init = max(mu_bar * (1 - mu_bar) / var_y - 1.0, 1.0)
    log_k_init = np.log(kappa_init)

    n_log_k = 1 if spec == "constant" else S
    P = Nfree + 1 + S + n_log_k
    x0 = np.zeros(P)
    if n_log_k == 1:
        x0[-1] = log_k_init
    else:
        x0[-S:] = log_k_init

    # ablation bounds: pin γ and/or φ_s to 0 if not included
    bounds = [(None, None)] * Nfree
    bounds += [(0.0, 0.0)] if not include_gamma else [(None, None)]
    bounds += [(0.0, 0.0)] * S if not include_phi else [(None, None)] * S
    bounds += [(LOG_KAPPA_LO, LOG_KAPPA_HI)] * n_log_k

    res = minimize(
        _nll_grad, x0,
        args=(ystar, lm_i_idx, lm_j_idx, ref_idx, Nfree,
              s_idx, xF, S, spec),
        jac=True, method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol, "disp": verbose},
    )
    converged = bool(res.success)
    n_iter = int(res.nit)
    nll = float(res.fun)

    # ----- post-fit: recover parameter vector + Hessian (numerical) -----
    params = res.x
    # numerical Hessian via finite differences for SE
    # use a small step; the gradient is analytic so this is well-conditioned
    eps = 1e-4
    H = np.zeros((P, P))
    g0 = res.jac
    for p in range(P):
        x_pert = params.copy()
        x_pert[p] += eps
        _, g_pert = _nll_grad(x_pert, ystar, lm_i_idx, lm_j_idx,
                              ref_idx, Nfree, s_idx, xF, S, spec)
        # NLL Hessian column = (g(x+ε) - g(x))/ε; but our _nll_grad returns
        # the *negated* gradient (because we minimize NLL = -ℓ).
        # Observed Fisher info = +Hessian of NLL.
        H[:, p] = (g_pert - g0) / eps
    # symmetrize
    H = 0.5 * (H + H.T)
    # invert with light Tikhonov for numerical stability
    try:
        cov = np.linalg.inv(H + 1e-10 * np.eye(P))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    # ----- unpack with SEs -----
    z95 = 1.96
    theta_free_hat = params[:Nfree]
    gamma_hat = float(params[Nfree])
    phi_hat = params[Nfree + 1: Nfree + 1 + S]
    if spec == "constant":
        log_k_hat = np.full(S, params[-1])
    else:
        log_k_hat = params[-S:]
    kappa_hat = np.exp(log_k_hat)

    # full θ
    free_positions = [i for i in range(Nfree + 1) if i != ref_idx]
    theta_full = np.zeros(Nfree + 1)
    se_theta_full = np.zeros(Nfree + 1)
    theta_full[free_positions] = theta_free_hat
    se_theta_full[free_positions] = se[:Nfree]

    thetas_df = pd.DataFrame({
        "lm": models, "theta": theta_full, "se": se_theta_full,
        "ci_lower": theta_full - z95 * se_theta_full,
        "ci_upper": theta_full + z95 * se_theta_full,
    })

    se_g = float(se[Nfree])
    g_ci = (gamma_hat - z95 * se_g, gamma_hat + z95 * se_g)

    se_phi = se[Nfree + 1: Nfree + 1 + S]
    scen_df = pd.DataFrame({
        "scenario": list(scenarios),
        "phi": phi_hat,
        "se": se_phi,
        "ci_lower": phi_hat - z95 * se_phi,
        "ci_upper": phi_hat + z95 * se_phi,
    })

    # κ SEs via delta method: SE(κ) = κ * SE(log κ)
    if spec == "constant":
        se_logk = np.full(S, se[-1])
    else:
        se_logk = se[-S:]
    se_kappa = kappa_hat * se_logk
    kappa_df = pd.DataFrame({
        "scenario": list(scenarios),
        "kappa": kappa_hat,
        "se": se_kappa,
        "ci_lower": np.exp(log_k_hat - z95 * se_logk),
        "ci_upper": np.exp(log_k_hat + z95 * se_logk),
    })

    par_names = ([f"theta[{m}]" for m in free_models]
                 + ["gamma"]
                 + [f"phi[{scenarios[s]}]" for s in range(S)]
                 + ([f"log_kappa[{scenarios[s]}]" for s in range(S)]
                    if spec == "per_scenario" else ["log_kappa"]))
    cov_df = pd.DataFrame(cov, index=par_names, columns=par_names)

    return BBTLResult(
        thetas=thetas_df, gamma=gamma_hat, gamma_se=se_g, gamma_ci=g_ci,
        scenario_effects=scen_df, kappa=kappa_df, cov_all=cov_df,
        log_likelihood=-nll, n_params=P, n_obs=len(df),
        converged=converged, n_iter=n_iter, spec=spec,
        models=models, reference=reference,
    )
