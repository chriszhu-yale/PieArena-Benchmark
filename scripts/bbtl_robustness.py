"""Reproduce the BBTL robustness checks of the paper from the released dataset.

    python scripts/bbtl_robustness.py                    # all three checks (~25 s)
    python scripts/bbtl_robustness.py nuisance           # Table 4 (gamma, phi, kappa) + LR test for per-scenario kappa
    python scripts/bbtl_robustness.py cv                 # Table 2: 10-fold CV of BBTL vs. Elo/BTL baselines
    python scripts/bbtl_robustness.py coverage           # Table 3: bootstrap coverage of the 95% Wald CIs (B = 500)
    python scripts/bbtl_robustness.py coverage --B 100   # faster, noisier coverage

All fits use ``dataset/lm_data/paper_crossplay_outcomes.csv`` with the filtering
of ``reproduce_leaderboard.py`` (13-model pool; verified deals with pie shares in
[0, 1]). The LR test, the CV and the coverage check use the pooled integrated
sample (both modes, N = 5,007 plays); Table 4 uses the six slice-by-mode fits of
the appendix leaderboards.

CV folds, bootstrap draws and Sequential Elo depend on the order of the plays.
The plays are first sorted into the order of the balanced extraction analysed in
the paper (scenario, pro before base mode, unordered model pair, direction, run),
so results do not depend on the CSV's row order. With the default seeds, Table 4
and the LR test match the paper at the printed precision, Table 3 matches
exactly, and Table 2 agrees to within 1e-4 (the paper prints the BBTL-gamma-phi
MSE, 0.039648, as 0.0397). Other SciPy versions can move CV values by about 1e-4
(optimizer tolerance).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from piearena import data as D  # noqa: E402
from piearena.bbtl_fit import fit_bbtl  # noqa: E402
from piearena.elo_baseline import CalibratedLogBTL, LogisticBTL, SequentialElo  # noqa: E402

MODES = {"base_mode": "Base", "pro_mode": "Pro"}
SEED = 2026  # CV folds and bootstrap


def load_plays() -> pd.DataFrame:
    """Plays entering the paper's BBTL fits, in canonical order, as a BBTL fit frame (plus ``mode``)."""
    runs = D.filter_a(D.in_pool(pd.read_csv(ROOT / "dataset" / "lm_data" / "paper_crossplay_outcomes.csv")))
    s1, s2 = runs.side1_model.to_numpy(), runs.side2_model.to_numpy()
    key = pd.DataFrame({"scenario": runs.scenario, "mode": runs["mode"].map({"pro_mode": 0, "base_mode": 1}),
                        "lo": np.where(s1 < s2, s1, s2), "hi": np.where(s1 < s2, s2, s1),
                        "reverse": s1 > s2, "run": runs.run_number}, index=runs.index)
    return D.fit_frame(runs.loc[key.sort_values(list(key.columns), kind="stable").index])


def fit(df: pd.DataFrame, spec: str = "per_scenario", **kw):
    with np.errstate(over="ignore"):  # unused scenarios of a slice have unbounded kappa CIs
        return fit_bbtl(df.reset_index(drop=True), reference=D.REFERENCE_MODEL, spec=spec, **kw)


# --------------------------------------------------------------------------- Table 4 + LR test
def nuisance(plays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tag = dict(zip(D.SLICES["Integrated"], ("HT", "IL", "SQ")))
    print("\nTable 4: BBTL nuisance parameters (Hessian SEs in parentheses; HT = Hire_Talent, IL = IP_License, "
          "SQ = Service_Quote)")
    print(f"  {'slice / mode':<20}{'gamma':>16}" + "".join(f"{'phi[' + t + ']':>16}" for t in tag.values())
          + "".join(f"{'kappa[' + t + ']':>11}" for t in tag.values()))
    for slice_name, scen in D.SLICES.items():
        for mode, label in MODES.items():
            res = fit(plays[(plays["mode"] == mode) & plays.scenario.isin(scen)],
                      spec="per_scenario" if len(scen) > 1 else "constant")  # one scenario: single kappa
            phi = res.scenario_effects.set_index("scenario").loc[list(scen)]
            kappa = res.kappa.set_index("scenario").loc[list(scen), "kappa"]
            rows.append({"slice": slice_name, "mode": label, "n": res.n_obs,
                         "gamma": res.gamma, "gamma_se": res.gamma_se,
                         **{f"phi[{s}]": phi.phi[s] for s in scen}, **{f"phi_se[{s}]": phi.se[s] for s in scen},
                         **{f"kappa[{s}]": kappa[s] for s in scen}})
            cells = [f"{phi.phi[s]:+.3f} ({phi.se[s]:.3f})" if s in scen else "---" for s in D.SLICES["Integrated"]]
            cells += [f"{kappa[s]:.2f}" if s in scen else "---" for s in D.SLICES["Integrated"]]
            print(f"  {slice_name + ', ' + label:<20}{f'{res.gamma:+.3f} ({res.gamma_se:.3f})':>16}"
                  + "".join(f"{c:>16}" for c in cells[:3]) + "".join(f"{c:>11}" for c in cells[3:]))

    # LR test: constant kappa (null) vs. per-scenario kappa, pooled integrated fit
    alt, null = fit(plays), fit(plays, spec="constant")
    lam, ddf = 2 * (alt.log_likelihood - null.log_likelihood), alt.n_params - null.n_params
    d_aic = (2 * null.n_params - 2 * null.log_likelihood) - (2 * alt.n_params - 2 * alt.log_likelihood)
    p = chi2.sf(lam, ddf)  # underflows to 0 for the paper's Lambda
    print(f"\nLR test, per-scenario vs. constant kappa (pooled integrated, N = {alt.n_obs:,}): Lambda = {lam:.1f}, "
          f"df = {ddf}, p {f'= {p:.1e}' if p > 0 else '< 1e-300'}, dAIC = {d_aic:.1f}")
    rows.append({"slice": "Integrated", "mode": "Pooled", "n": alt.n_obs, "lr_stat": lam, "lr_df": ddf, "d_aic": d_aic})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 2
def stratified_folds(strata: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Test-fold index of each play; same assignment as sklearn's StratifiedKFold(shuffle=True, random_state=seed)."""
    rng = np.random.RandomState(seed)
    _, first, inverse = np.unique(strata, return_index=True, return_inverse=True)
    code = np.argsort(np.argsort(first))[inverse]  # classes numbered by order of appearance
    alloc = np.array([np.bincount(np.sort(code)[i::n_folds], minlength=len(first)) for i in range(n_folds)])
    folds = np.empty(len(strata), dtype=int)
    for k in range(len(first)):
        f = np.repeat(np.arange(n_folds), alloc[:, k])
        rng.shuffle(f)
        folds[code == k] = f
    return folds


def predict_bbtl(res, df: pd.DataFrame) -> np.ndarray:
    """Fitted mean share of side 1: logistic((theta_i - theta_j) + gamma x_F + phi_s)."""
    theta = res.thetas.set_index("lm")["theta"]
    phi = res.scenario_effects.set_index("scenario")["phi"]
    x_first = np.where(df.who_first.str.lower().str.startswith("side1"), 1.0, -1.0)
    eta = (df.lm_i.map(theta).to_numpy() - df.lm_j.map(theta).to_numpy()
           + res.gamma * x_first + df.scenario.map(phi).to_numpy())
    return 1.0 / (1.0 + np.exp(-eta))


def cross_validate(plays: pd.DataFrame, n_folds: int = 10) -> pd.DataFrame:
    """Held-out MSE/MAE of side 1's pie share, folds stratified by scenario."""
    plays = plays.assign(y=(plays.p_i - plays.p_j + 1.0) / 2.0)  # = p_i; outcome column of the baselines
    folds = stratified_folds(plays.scenario.to_numpy(), n_folds, SEED)
    rows = []
    for k in range(n_folds):
        train, test = plays[folds != k], plays[folds == k]  # train keeps the canonical order (Elo is order-dependent)
        pairs = list(zip(test.lm_i, test.lm_j))
        elo = SequentialElo(K=32.0, scale=400.0).fit(train)
        btl = LogisticBTL(reference=D.REFERENCE_MODEL).fit(train)
        cal = CalibratedLogBTL(reference=D.REFERENCE_MODEL).fit(train)
        preds = {
            "Sequential Elo": [elo.predict(i, j) for i, j in pairs],
            "Logistic BTL": [btl.predict(i, j) for i, j in pairs],
            "Calibrated LogBTL": [cal.predict(i, j) for i, j in pairs],
            "BBTL-gamma-phi": predict_bbtl(fit(train, include_gamma=False, include_phi=False), test),
            "Full BBTL": predict_bbtl(fit(train), test),
        }
        for method, pred in preds.items():
            err = test.y.to_numpy() - np.asarray(pred)
            rows.append({"fold": k + 1, "method": method, "mse": np.mean(err ** 2), "mae": np.mean(np.abs(err))})
    cv = pd.DataFrame(rows)
    agg = cv.groupby("method", sort=False).agg(mse=("mse", "mean"), mse_se=("mse", "std"),
                                               mae=("mae", "mean"), mae_se=("mae", "std"))
    agg[["mse_se", "mae_se"]] /= np.sqrt(n_folds)  # SE of the mean over folds
    print(f"\nTable 2: {n_folds}-fold CV on N = {len(plays):,} plays (mean +- SE over folds)")
    print(f"  {'method':<20}{'CV MSE':>20}{'CV MAE':>20}")
    for m, r in agg.iterrows():
        print(f"  {m:<20}{f'{r.mse:.5f} +- {r.mse_se:.5f}':>20}{f'{r.mae:.5f} +- {r.mae_se:.5f}':>20}")
    return agg.reset_index()


# --------------------------------------------------------------------------- Table 3
def coverage(plays: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Share of bootstrap refits whose 95% Wald CI covers the full-sample theta (non-parametric bootstrap)."""
    cols = ["lm_i", "lm_j", "p_i", "p_j", "scenario", "who_first"]
    plays = plays.reset_index(drop=True)
    theta_hat = fit(plays[cols]).thetas.set_index("lm")["theta"]
    # resample plays within each (side-1 model, side-2 model, scenario, mode) cell, so every replicate keeps
    # the balanced design; cells are visited in sorted order with pro before base mode
    by_cell = plays.groupby(["lm_i", "lm_j", "scenario", plays["mode"].eq("base_mode")])
    cells = [np.asarray(ix) for ix in by_cell.indices.values()]
    rng = np.random.default_rng(SEED)
    hits = pd.Series(0, index=theta_hat.index)
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in cells])
        ci = fit(plays.iloc[take][cols]).thetas.set_index("lm")
        hits += ((ci.ci_lower <= theta_hat) & (theta_hat <= ci.ci_upper)).astype(int)
    cov = (hits / n_boot).rename("coverage").drop(D.REFERENCE_MODEL)  # the anchor (theta = 0) is covered trivially
    print(f"\nTable 3: coverage of the 95% Wald CIs, B = {n_boot} non-parametric bootstrap refits "
          f"({len(cells)} cells, N = {len(plays):,})")
    for m, c in cov.items():
        print(f"  {m:<32} {c:.3f}")
    print(f"  {D.REFERENCE_MODEL:<32} anchor")
    print(f"  mean over the {len(cov)} non-anchor LMs = {cov.mean():.3f}, range [{cov.min():.3f}, {cov.max():.3f}]")
    return cov.reset_index()


def main() -> None:
    checks = ("nuisance", "cv", "coverage")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tables", nargs="*", metavar="{nuisance,cv,coverage}", help="checks to run (default: all)")
    ap.add_argument("--B", type=int, default=500,
                    help="bootstrap replicates for coverage (default: %(default)s, ~25 s)")
    ap.add_argument("--out", help="directory for one CSV per check (default: print only)")
    args = ap.parse_args()
    if set(args.tables) - set(checks):  # argparse cannot combine choices with an empty nargs="*" default
        ap.error(f"unknown check(s) {sorted(set(args.tables) - set(checks))}; choose from {', '.join(checks)}")

    plays = load_plays()
    results = {}
    for table in args.tables or checks:
        if table == "nuisance":
            results[table] = nuisance(plays)
        elif table == "cv":
            results[table] = cross_validate(plays)
        else:
            results[table] = coverage(plays, args.B)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for table, df in results.items():
            df.to_csv(Path(args.out) / f"bbtl_{table}.csv", index=False)
        print(f"\nwrote {', '.join(f'bbtl_{t}.csv' for t in results)} to {args.out}")


if __name__ == "__main__":
    main()
