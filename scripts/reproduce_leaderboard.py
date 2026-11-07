"""Reproduce the cross-play BBTL leaderboards of the paper from the released dataset.

    python scripts/reproduce_leaderboard.py                 # exact paper numbers
    python scripts/reproduce_leaderboard.py --from-tables   # refit on the per-pair tables

Fits: every slice (Integrated, Multi-Issue, Single-Issue) in base and pro mode, the pooled main-text
leaderboard, then the per-scenario leaderboards of Hire Talent and IP License in base and pro mode, each
fitted on that scenario alone with a single kappa (Service Quote alone is the Single-Issue slice).

By default the fits use ``dataset/lm_data/paper_crossplay_outcomes.csv`` (the
outcomes recorded for the balanced design analysed in the paper). With
``--from-tables`` they use the per-pair ``pair_runs_table.csv`` rows flagged
``paper_analysis``, whose outcomes are recomputed from the released transcripts;
the two differ only where ``transcript_consistent`` is False in the outcome file.
Estimates agree with the paper's tables to within one unit in the 4th decimal
(optimizer tolerance across SciPy versions).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from piearena import data as D  # noqa: E402
from piearena.bbtl_fit import fit_bbtl  # noqa: E402

MODES = {"base_mode": "Base", "pro_mode": "Pro"}
PER_SCENARIO = {"Hire Talent": "Hire_Talent", "IP License": "IP_License"}  # Service Quote = Single-Issue slice


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-tables", action="store_true", help="use per-pair tables instead of the outcome file")
    ap.add_argument("--out", default="leaderboards.csv", help="CSV with all fitted skills (default: %(default)s)")
    args = ap.parse_args()

    if args.from_tables:
        runs = D.paper_runs(D.load_runs())
    else:
        runs = pd.read_csv(ROOT / "dataset" / "lm_data" / "paper_crossplay_outcomes.csv")
    fit = D.fit_frame(D.filter_a(D.in_pool(runs)))

    fits = [(slice_name, MODES[mode], fit[(fit["mode"] == mode) & fit.scenario.isin(scen)])
            for slice_name, scen in D.SLICES.items() for mode in MODES]
    fits.append(("Integrated", "Pooled", fit))  # main-text leaderboard: both modes pooled
    fits += [(name, MODES[mode], fit[(fit["mode"] == mode) & (fit.scenario == scen)])  # per-scenario leaderboards
             for name, scen in PER_SCENARIO.items() for mode in MODES]
    rows = []
    for slice_name, mode, sub in fits:
        spec = "per_scenario" if sub.scenario.nunique() > 1 else "constant"  # one scenario: single kappa
        with np.errstate(over="ignore"):  # unused scenarios of a slice have unbounded kappa CIs
            res = fit_bbtl(sub.reset_index(drop=True), reference=D.REFERENCE_MODEL, spec=spec)
        lb = res.thetas.sort_values("theta", ascending=False).reset_index(drop=True)
        lb.insert(0, "mode", mode)
        lb.insert(0, "slice", slice_name)
        lb["rank"] = range(1, len(lb) + 1)
        rows.append(lb)
        print(f"\n{slice_name} ({mode}), N = {len(sub)}, gamma = {res.gamma:+.4f}")
        for r in lb.itertuples():
            print(f"  {r.rank:>2}. {r.lm:<32} {r.theta:+.4f}  [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]")
    pd.concat(rows, ignore_index=True).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
