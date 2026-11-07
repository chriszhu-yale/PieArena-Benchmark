"""Reproduce the paper's SnyderMed mirror-play results and human-human baseline from the released dataset.

    python scripts/reproduce_mirror_play.py              # all tables (~10 s)
    python scripts/reproduce_mirror_play.py --out DIR    # also write one CSV per table

Reproduces
  * the human-human SnyderMed baseline (n = 23 pairs): mean, SD and median of the normalized
    total pie and of the total pie (main text and notes of the mirror-play tables);
  * the table "Mirror-play LM agent performance vs. human baseline (two-sided tests)", base and pro
    mode, and its main-text subset "Mirror-play performance vs. human baseline in SnyderMed";
  * the table "Scaffolding effect: pro vs. base (one-sided tests)".

Data: ``dataset/human_data/snydermed_human_human.csv`` (one row per agreed human-human contract, in
file order) and the mirror-play tables
``dataset/lm_data/transcripts/mirror_play/{base_mode,pro_mode}/SnyderMed/*/pair_runs_table.csv``,
rows flagged ``paper_analysis`` (20 runs per model and mode, taken in run order).

Definitions, as in the paper's analysis:
  * outcome: normalized total pie = total pie / P*, with P* = 71,500 the largest achievable total pie
    in SnyderMed. Every flagged LM run with a recorded total pie enters the statistics (this includes
    a few runs whose agreement was not verified); "n deals" counts the verified agreements;
  * Mean +- CI: mean with the half-width of a 95% percentile bootstrap CI of the mean (10,000
    resamples, generator ``numpy.random.default_rng(42)`` restarted for every sample);
  * Diff +- CI: difference of means (model - human, or pro - base) with the half-width of a 95%
    percentile bootstrap CI (10,000 draws that resample both samples in turn, same seed);
  * half-widths are (hi - lo) / 2 of the CI endpoints rounded to three decimals, as in the paper
    (ties rounded up); ``--out`` writes the unrounded endpoints;
  * tests: Mann-Whitney U (SciPy defaults: asymptotic, tie and continuity corrected), two-sided against
    the human baseline and one-sided (pro > base) for the scaffolding effect; Benjamini-Hochberg q-values
    within each mode (vs. human) or across the 13 LMs (pro vs. base); Sig. means q < 0.05;
  * Cliff's delta = P(X > Y) - P(X < Y), X drawn from the model (or pro) sample, Y from the human (or base)
    sample. Rows are sorted by Cliff's delta, largest first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from piearena import data as D  # noqa: E402

HUMAN_CSV = ROOT / "dataset" / "human_data" / "snydermed_human_human.csv"
P_STAR = 71_500.0  # largest achievable total pie in SnyderMed (normalized_total_pie = total_pie / P_STAR)
SEED, N_BOOT, ALPHA = 42, 10_000, 0.05
MODES = {"base_mode": "Base", "pro_mode": "Pro"}
SHORT = {  # model ids of the dataset -> short names used in the paper
    "anthropic/claude-opus-4.5": "Claude-Opus-4.5", "anthropic/claude-sonnet-4.5": "Claude-Sonnet-4.5",
    "baidu/ernie-4.5-vl-424b-a47b": "ERNIE-4.5", "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-3-pro-preview": "Gemini-3-Pro", "openai/gpt-4.1": "GPT-4.1", "openai/gpt-5": "GPT-5",
    "openai/gpt-5.2": "GPT-5.2", "openai/o4-mini-high": "o4-Mini-High", "x-ai/grok-3": "Grok-3",
    "x-ai/grok-3-mini-beta": "Grok-3-Mini", "x-ai/grok-4": "Grok-4", "x-ai/grok-4.1-fast": "Grok-4.1-Fast",
}
MAIN_TEXT_SUBSET = ("openai/gpt-5", "openai/gpt-5.2", "x-ai/grok-3", "x-ai/grok-4.1-fast",
                    "baidu/ernie-4.5-vl-424b-a47b", "openai/gpt-4.1", "anthropic/claude-opus-4.5",
                    "google/gemini-2.5-flash")


# --------------------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(means, 100 * ALPHA / 2)), float(np.percentile(means, 100 * (1 - ALPHA / 2)))


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% percentile bootstrap CI of mean(x) - mean(y); each draw resamples x, then y."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs[b] = np.mean(bx) - np.mean(by)
    return float(np.percentile(diffs, 100 * ALPHA / 2)), float(np.percentile(diffs, 100 * (1 - ALPHA / 2)))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sign(x[:, None] - y[None, :]).sum() / (len(x) * len(y)))


def bh_qvalues(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (step-up, capped at 1)."""
    order = np.argsort(p.to_numpy(), kind="stable")
    ranked = p.to_numpy()[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty(len(p))
    q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return pd.Series(q, index=p.index)


def half(lo: float, hi: float) -> str:
    """Half-width of [lo, hi] as printed in the paper: from the endpoints at 3 decimals, ties rounded up."""
    k = round(float(f"{hi:.3f}") * 1000) - round(float(f"{lo:.3f}") * 1000)  # width in thousandths
    return f"{(k + 1) // 2 / 1000:.3f}"


# --------------------------------------------------------------------------- data
def load_human() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_CSV)
    assert np.allclose(h.total_pie / P_STAR, h.normalized_total_pie), "P* does not match normalized_total_pie"
    return h


def load_mirror() -> pd.DataFrame:
    runs = D.paper_runs(D.load_runs("mirror_play", modes=tuple(MODES)))
    assert (runs.side1_model == runs.side2_model).all()
    runs = runs.sort_values(["mode", "side1_model", "run_number"], kind="stable")
    assert (runs.groupby(["mode", "side1_model"]).size() == 20).all(), "expected 20 paper runs per model and mode"
    return runs.assign(model=runs.side1_model, ratio=runs.total_pie / P_STAR)


def samples(runs: pd.DataFrame, mode: str) -> dict[str, tuple[np.ndarray, int]]:
    """model -> (normalized total pie of the runs with a recorded total pie, number of verified deals)."""
    out = {}
    for model, g in runs[runs["mode"] == mode].groupby("model", sort=True):
        out[model] = (g.ratio.dropna().to_numpy(), int(g.verified_agreement.astype(bool).sum()))
    return out


# --------------------------------------------------------------------------- tables
def human_baseline(h: pd.DataFrame) -> dict:
    r, pie = (h.total_pie / P_STAR).to_numpy(), h.total_pie.to_numpy()
    stats = {"n": len(h), "ratio_mean": r.mean(), "ratio_sd": r.std(ddof=1), "ratio_median": np.median(r),
             "pie_mean": pie.mean(), "pie_sd": pie.std(ddof=1), "pie_median": np.median(pie)}
    print(f"\nHuman-human SnyderMed baseline: n = {stats['n']} pairs")
    print(f"  normalized total pie: mean = {stats['ratio_mean']:.3f}, SD = {stats['ratio_sd']:.3f}, "
          f"median = {stats['ratio_median']:.3f}")
    print(f"  total pie:            mean = ${stats['pie_mean']:,.0f}, SD = ${stats['pie_sd']:,.2f}, "
          f"median = ${stats['pie_median']:,.0f}")
    return stats


def sig_label(q: float, diff: float, one_sided: bool = False) -> str:
    if q >= ALPHA:
        return "---"
    return "up" if (one_sided or diff > 0) else "down"


def vs_human(runs: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    human = (h.total_pie / P_STAR).to_numpy()
    rows = []
    for mode, label in MODES.items():
        part = []
        for model, (x, n_deals) in samples(runs, mode).items():
            lo, hi = bootstrap_ci(x)
            dlo, dhi = bootstrap_diff_ci(x, human)
            part.append({"mode": label, "model": model, "n_deals": n_deals, "n_used": len(x),
                         "mean": x.mean(), "ci_lo": lo, "ci_hi": hi, "median": np.median(x),
                         "diff": x.mean() - human.mean(), "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                         "cliffs_delta": cliffs_delta(x, human),
                         "p": mannwhitneyu(x, human, alternative="two-sided").pvalue})
        part = pd.DataFrame(part)
        part["q"] = bh_qvalues(part.p)  # FDR within each mode
        rows.append(part.sort_values("cliffs_delta", ascending=False, kind="stable"))
    res = pd.concat(rows, ignore_index=True)
    res["sig"] = [sig_label(q, d) for q, d in zip(res.q, res["diff"])]

    print("\nTable: Mirror-play LM agent performance vs. human baseline (two-sided tests); normalized total pie")
    print(f"  {'Model':<19}{'n deals':>8}{'Mean +- CI':>17}{'Median':>8}{'Diff (Mean +- CI)':>19}"
          f"{'Cliffs d':>10}{'q':>7}  Sig.")
    for tag, label in zip("AB", MODES.values()):
        print(f"  ({tag}) {label} Mode")
        for r in res[res["mode"] == label].itertuples():
            print(f"  {SHORT[r.model]:<19}{r.n_deals:>8}{f'{r.mean:.3f} +- {half(r.ci_lo, r.ci_hi)}':>17}"
                  f"{r.median:>8.3f}{f'{r.diff:+.3f} +- {half(r.diff_ci_lo, r.diff_ci_hi)}':>19}"
                  f"{r.cliffs_delta:>10.3f}{r.q:>7.3f}  {r.sig}")
    return res


def main_text_subset(res: pd.DataFrame) -> pd.DataFrame:
    wide = res.pivot(index="model", columns="mode", values=["mean", "sig"])
    sub = wide.loc[list(MAIN_TEXT_SUBSET)]
    print("\nMain-text table: Mirror-play performance vs. human baseline in SnyderMed (representative subset)")
    print(f"  {'Model':<19}{'Base Mean':>10}{'Sig.':>6}{'Pro Mean':>10}{'Sig.':>6}")
    for model, r in sub.iterrows():
        print(f"  {SHORT[model]:<19}{r[('mean', 'Base')]:>10.3f}{r[('sig', 'Base')]:>6}"
              f"{r[('mean', 'Pro')]:>10.3f}{r[('sig', 'Pro')]:>6}")
    sub.columns = [f"{c}_{m.lower()}" for c, m in sub.columns]
    return sub.reset_index()


def pro_vs_base(runs: pd.DataFrame) -> pd.DataFrame:
    base, pro = samples(runs, "base_mode"), samples(runs, "pro_mode")
    rows = []
    for model in sorted(set(base) & set(pro)):
        (xp, n_p), (xb, n_b) = pro[model], base[model]
        plo, phi = bootstrap_ci(xp)
        blo, bhi = bootstrap_ci(xb)
        dlo, dhi = bootstrap_diff_ci(xp, xb)
        rows.append({"model": model, "pro_n_deals": n_p, "base_n_deals": n_b,
                     "pro_mean": xp.mean(), "pro_ci_lo": plo, "pro_ci_hi": phi,
                     "base_mean": xb.mean(), "base_ci_lo": blo, "base_ci_hi": bhi,
                     "diff": xp.mean() - xb.mean(), "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                     "cliffs_delta": cliffs_delta(xp, xb),
                     "p": mannwhitneyu(xp, xb, alternative="greater").pvalue})
    res = pd.DataFrame(rows)
    res["q"] = bh_qvalues(res.p)  # FDR across models
    res = res.sort_values("cliffs_delta", ascending=False, kind="stable").reset_index(drop=True)
    res["sig"] = [sig_label(q, d, one_sided=True) for q, d in zip(res.q, res["diff"])]

    print("\nTable: Scaffolding effect, pro vs. base (one-sided tests, H1: pro > base); normalized total pie")
    print(f"  {'Model':<19}{'Pro n':>6}{'Base n':>7}{'Pro (Mean +- CI)':>18}{'Base (Mean +- CI)':>19}"
          f"{'Diff (Mean +- CI)':>19}{'Cliffs d':>10}{'q':>7}  Sig.")
    for title, part in (("Significant scaffolding benefit (q < 0.05)", res[res.q < ALPHA]),
                        ("No significant scaffolding benefit (q >= 0.05)", res[res.q >= ALPHA])):
        print(f"  {title}")
        for r in part.itertuples():
            print(f"  {SHORT[r.model]:<19}{r.pro_n_deals:>6}{r.base_n_deals:>7}"
                  f"{f'{r.pro_mean:.3f} +- {half(r.pro_ci_lo, r.pro_ci_hi)}':>18}"
                  f"{f'{r.base_mean:.3f} +- {half(r.base_ci_lo, r.base_ci_hi)}':>19}"
                  f"{f'{r.diff:+.3f} +- {half(r.diff_ci_lo, r.diff_ci_hi)}':>19}"
                  f"{r.cliffs_delta:>10.3f}{r.q:>7.3f}  {r.sig}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="directory for one CSV per table (default: print only)")
    args = ap.parse_args()

    human, runs = load_human(), load_mirror()
    results = {"human_baseline": pd.DataFrame([human_baseline(human)])}
    results["mirror_vs_human"] = vs_human(runs, human)
    results["mirror_vs_human_main_text"] = main_text_subset(results["mirror_vs_human"])
    results["pro_vs_base"] = pro_vs_base(runs)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            df.to_csv(Path(args.out) / f"{name}.csv", index=False)
        print(f"\nwrote {', '.join(f'{n}.csv' for n in results)} to {args.out}")


if __name__ == "__main__":
    main()
