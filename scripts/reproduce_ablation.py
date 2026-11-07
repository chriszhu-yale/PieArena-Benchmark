"""Reproduce the paper's harness ablation (SnyderMed mirror-play) from the released dataset.

    python scripts/reproduce_ablation.py              # ablation table and summary (~1 s)
    python scripts/reproduce_ablation.py --out DIR    # also write the tables as CSV

Reproduces
  * the table "Per-model scaffolding ablation on SnyderMed mirror-play (12 LMs)": for State Tracking
    only vs. Base and for Full Harness vs. Base, the one-sided Mann-Whitney U test (H1: the harness
    condition yields a higher total pie), its Benjamini-Hochberg q-value across the 12 LMs, Cliff's delta
    and the significance mark (q < 0.05); rows sorted by Cliff's delta of the Full Harness comparison;
  * the summary in the text: number of LMs with q < 0.05 per comparison and their mean q;
  * the per-model mean total pie in each condition and its change from Base (absolute and normalized by
    P* = 71,500).

Data: the mirror-play tables
``dataset/lm_data/transcripts/mirror_play/{base_mode,memory_only_mode,pro_mode}/SnyderMed/*/pair_runs_table.csv``,
rows flagged ``paper_analysis``. Conditions: Base = ``base_mode``, State Tracking only = ``memory_only_mode``,
Full Harness = ``pro_mode``. The 12 LMs are the ranked LMs with flagged runs in all three conditions
(20 runs per model and condition; 10 State-Tracking runs for GPT-5 and o4-Mini-High). As in the
mirror-play tables, every flagged run with a recorded total pie enters the comparison. The tests and
Cliff's delta are invariant to normalizing the total pie by P*.
"""
from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from piearena import data as D  # noqa: E402
from reproduce_mirror_play import P_STAR, SHORT, bh_qvalues, cliffs_delta  # noqa: E402

CONDITIONS = {"base_mode": "Base", "memory_only_mode": "State Tracking only", "pro_mode": "Full Harness"}
COMPARISONS = (("memory_only_mode", "State Tracking only vs. Base"), ("pro_mode", "Full Harness vs. Base"))
ALPHA = 0.05


def signed2(x: float) -> str:
    """Two decimals with a sign, halves rounded away from zero as in the paper (Cliff's delta -0.125 -> -0.13)."""
    return f"{Decimal(repr(float(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+.2f}"


def load() -> pd.DataFrame:
    runs = D.paper_runs(D.load_runs("mirror_play", modes=tuple(CONDITIONS)))
    runs = runs[runs.side1_model.isin(D.FINAL_POOL)].assign(model=lambda d: d.side1_model)
    present = runs.groupby("model")["mode"].nunique()
    runs = runs[runs.model.isin(present[present == len(CONDITIONS)].index)]
    assert runs.model.nunique() == 12, "expected the 12 LMs present in all three conditions"
    return runs.sort_values(["mode", "model", "run_number"], kind="stable")


def pie(runs: pd.DataFrame, model: str, mode: str) -> np.ndarray:
    return runs.loc[(runs.model == model) & (runs["mode"] == mode), "total_pie"].dropna().to_numpy()


def ablation_table(runs: pd.DataFrame) -> pd.DataFrame:
    models = sorted(runs.model.unique())
    out = pd.DataFrame({"model": models})
    for mode, _ in COMPARISONS:
        tag = "st" if mode == "memory_only_mode" else "full"
        rows = [(cliffs_delta(pie(runs, m, mode), pie(runs, m, "base_mode")),
                 mannwhitneyu(pie(runs, m, mode), pie(runs, m, "base_mode"), alternative="greater").pvalue)
                for m in models]
        out[f"{tag}_delta"], out[f"{tag}_p"] = zip(*rows)
        out[f"{tag}_q"] = bh_qvalues(out[f"{tag}_p"])  # FDR across the 12 LMs
        out[f"{tag}_sig"] = out[f"{tag}_q"] < ALPHA
    out = out.sort_values("full_delta", ascending=False, kind="stable").reset_index(drop=True)

    print("\nTable: Per-model scaffolding ablation on SnyderMed mirror-play (12 LMs); one-sided Mann-Whitney U "
          "against Base,\n       BH q-values across the 12 LMs, Cliff's delta; * marks q < 0.05")
    print(f"  {'':<19}{'State Tracking only vs. Base':^24}{'Full Harness vs. Base':^24}")
    print(f"  {'Model':<19}" + f"{'q':>9}{'delta':>9}{'Sig.':>6}" * 2)
    for r in out.itertuples():
        print(f"  {SHORT[r.model]:<19}{r.st_q:>9.3f}{signed2(r.st_delta):>9}{'*' if r.st_sig else '':>6}"
              f"{r.full_q:>9.3f}{signed2(r.full_delta):>9}{'*' if r.full_sig else '':>6}")
    for tag, (_, label) in zip(("st", "full"), COMPARISONS):
        sig = out[out[f"{tag}_sig"]]
        print(f"  {label}: {len(sig)}/{len(out)} LMs with q < 0.05 (mean q = {sig[f'{tag}_q'].mean():.3f}); "
              f"mean Cliff's delta over the 12 LMs = {out[f'{tag}_delta'].mean():.3f}")
    return out


def condition_means(runs: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for m in order:
        means = {mode: pie(runs, m, mode).mean() for mode in CONDITIONS}
        rows.append({"model": m, **{f"n_{mode}": len(pie(runs, m, mode)) for mode in CONDITIONS},
                     **{f"mean_{mode}": v for mode, v in means.items()},
                     **{f"change_{mode}": means[mode] - means["base_mode"] for mode, _ in COMPARISONS}})
    out = pd.DataFrame(rows)
    print("\nMean total pie by condition (runs with a recorded total pie) and change from Base; "
          f"normalized change = change / {P_STAR:,.0f}")
    print(f"  {'Model':<19}{'Base':>10}{'State Tr.':>11}{'Full':>10}{'ST - Base':>11}{'Full - Base':>12}"
          f"{'(norm.)':>9}{'(norm.)':>9}")
    for r in out.itertuples():
        print(f"  {SHORT[r.model]:<19}{r.mean_base_mode:>10,.1f}{r.mean_memory_only_mode:>11,.1f}{r.mean_pro_mode:>10,.1f}"
              f"{r.change_memory_only_mode:>+11,.1f}{r.change_pro_mode:>+12,.1f}"
              f"{r.change_memory_only_mode / P_STAR:>+9.3f}{r.change_pro_mode / P_STAR:>+9.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="directory for the CSV files (default: print only)")
    args = ap.parse_args()

    runs = load()
    table = ablation_table(runs)
    means = condition_means(runs, list(table.model))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(args.out) / "ablation_tests.csv", index=False)
        means.to_csv(Path(args.out) / "ablation_condition_means.csv", index=False)
        print(f"\nwrote ablation_tests.csv, ablation_condition_means.csv to {args.out}")


if __name__ == "__main__":
    main()
