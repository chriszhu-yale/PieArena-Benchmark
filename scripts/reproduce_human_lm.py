"""Reproduce the paper's human-LM study tables from the released human data.

    python scripts/reproduce_human_lm.py               # tables, power analysis, main-text numbers
    python scripts/reproduce_human_lm.py --draws 200   # also the spread of the statistics over 200 redraws

Prints, next to the published values, the human-data appendix tables (Main Street and Hire Talent experimental
conditions, overview statistics and human-LM tests; post-hoc power and Bayes factors) and the related main-text
numbers (session counts, deal rates, pie shares, the SnyderMed human-human summary). Demographics are not released.

The primary analyses use double-balanced subsets of the deals (Main Street: 17 per LM mode x first-mover cell; Hire
Talent: 10 per student role x first-mover cell), marked by the ``balanced_subset`` column. Point estimates, tests,
effect sizes and Bayes factors do not depend on row order. The bootstrap intervals resample rows by position from the
paper's seeded streams, so they equal the paper's only in the paper's analysis order (``balanced_order``, if present);
otherwise the subset is taken in session-id order. Without ``balanced_subset`` the subsets are redrawn with the
paper's sampler (pandas ``sample(n=k, random_state=42)`` per cell, then ``sample(frac=1, random_state=42)``) from the
deals in session-id order, which gives a different subset.
"""
from __future__ import annotations

import argparse
import re
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parent.parent
HUMAN = ROOT / "dataset" / "human_data"
N_BOOT = 10_000
MS_PIE = 200_000  # Main Street: total pie, fixed by the single-issue scenario design
MS_CELLS = [("Base", "LM"), ("Base", "Student"), ("Pro", "LM"), ("Pro", "Student")]  # (mode, first mover)
HT_CELLS = [("Recruiter", "LM"), ("Recruiter", "Student"), ("Candidate", "LM"), ("Candidate", "Student")]
# Bootstrap intervals: (lower, upper) block of a seeded RandomState stream of 10,000-resample blocks, in the order in
# which the paper's analysis consumed the stream (Main Street: every mean CI uses block 0 of seed 42).
MS_SEED, MS_BLOCKS = 42, {"mean": (0, 0), "human_median": (1, 2), "lm_median": (3, 4)}
HT_SEED, HT_BLOCKS = 2026, {"total": (4, 5), "share": (6, 7), "human_median": (10, 11), "lm_median": (12, 13),
                            "human_mean": (14, 15), "lm_mean": (16, 17)}

PAPER = {  # published values
    "ms_conditions": [50, 39, 41, 48, 23, 18, 27, 21, 34, 34, 34, 34, 17, 17, 17, 17],
    "ht_conditions": [29, 26, 29, 26, 17, 12, 12, 14, 20, 20, 20, 20, 10, 10, 10, 10],
    "ms_overview": [("79,449", "[70,934, 88,338]"), ("120,551", "[111,662, 129,066]"), ("200,000", "---"),
                    ("39.7%", "[35.5%, 44.2%]"), ("75,000", "[70,000, 85,000]"),
                    ("125,000", "[115,000, 130,000]")],
    "ht_overview": [("28,388", "[23,025, 33,588]"), ("22,682", "[18,400, 27,015]"), ("51,070", "[48,630, 53,370]"),
                    ("53.8%", "[44.5%, 62.9%]"), ("28,500", "[21,250, 35,000]"), ("22,750", "[18,000, 27,750]")],
    "ms_tests": [("34", "83,721", "116,279", "p=0.0186; r_rb=-0.484"),
                 ("34", "75,176", "124,824", "p<1e-4; r_rb=-0.825"),
                 ("34/34", "124,824", "116,279", "p=0.2089; delta=0.115")],
    "ht_tests": [("40", "28,388", "22,682", "p=0.2674; r_rb=0.201"),
                 ("40", "53.8%", "50.0%", "p=0.4008; r_rb=0.152")],
    "power": [("68", "-0.559", ">0.99 at observed d", "BF10 = 977 (LM)"),
              ("40", "+0.186", "0.85 at d=0.5; d*=0.465 at 0.80", "BF01 = 3.09 (parity)")],
}


# ----------------------------------------------------------------------------------------------------- data
def load(human=HUMAN):
    ms = pd.read_csv(human / "main_street_sessions.csv").sort_values("session_id", ignore_index=True)
    ht = pd.read_csv(human / "hire_talent_sessions.csv").sort_values("session_id", ignore_index=True)
    hh = pd.read_csv(human / "snydermed_human_human.csv")
    ms["mode"] = np.where((ms.use_memory == 1) & (ms.use_plan == 1), "Pro", "Base")
    ht["role"] = ht.student_role.map({"side1": "Recruiter", "side2": "Candidate"})
    for df in (ms, ht):
        df["first"] = np.where(df.student_goes_first == 1, "Student", "LM")
    return ms, ht, hh


def deals_of(df):
    deals = df[df.deal_reached == 1].copy()
    assert deals.student_true_points.notna().all() and deals.ai_true_points.notna().all()
    return deals


def balanced(deals, by, cells, seed=42):
    """Double-balanced subset: k = smallest cell size deals per cell, cells concatenated in order, then shuffled."""
    k = min(((deals[by[0]] == a) & (deals[by[1]] == b)).sum() for a, b in cells)
    parts = [deals[(deals[by[0]] == a) & (deals[by[1]] == b)].sample(n=k, random_state=seed) for a, b in cells]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def subset(deals, by, cells):
    """The paper's subset from the released columns, in analysis order if given; else a redraw. Returns (frame, how)."""
    if "balanced_subset" not in deals:
        return balanced(deals, by, cells), "redrawn with the paper's sampler (no balanced_subset column)"
    bal = deals[deals.balanced_subset == 1]
    k = len(bal) // 4
    assert [int(((bal[by[0]] == a) & (bal[by[1]] == b)).sum()) for a, b in cells] == [k] * 4
    if "balanced_order" in bal:
        assert sorted(bal.balanced_order) == list(range(1, len(bal) + 1))
        return bal.sort_values("balanced_order"), "balanced_subset, in the paper's order (balanced_order)"
    return bal.sort_values("session_id"), "balanced_subset, in session-id order (no balanced_order column)"


# ----------------------------------------------------------------------------------------------- statistics
def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation (W+ - W-) / (n(n+1)/2) over the nonzero differences."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / (len(d) * (len(d) + 1) / 2)


def wilcoxon(diff):
    """Two-sided Wilcoxon signed-rank test (SciPy defaults) with its rank-biserial effect size."""
    return stats.wilcoxon(np.asarray(diff, dtype=float), alternative="two-sided").pvalue, rank_biserial(diff)


def cliffs_delta(x, y):
    return float(np.sign(np.subtract.outer(np.asarray(x, float), np.asarray(y, float))).mean())


def boot_blocks(seed, n, count):
    rs = np.random.RandomState(seed)
    return [rs.randint(0, n, size=(N_BOOT, n)) for _ in range(count)]


def boot_ci(x, blocks, lo_hi, stat=np.mean):
    """Percentile 95% interval; the lower bound from one resample block and the upper bound from another."""
    x = np.asarray(x, dtype=float)
    return (np.percentile(stat(x[blocks[lo_hi[0]]], axis=1), 2.5),
            np.percentile(stat(x[blocks[lo_hi[1]]], axis=1), 97.5))


def cohens_d(diff):
    return float(np.mean(diff) / np.std(diff, ddof=1))


def jzs_bf01(x, r=np.sqrt(2) / 2):
    """JZS Bayes factor BF01 for a one-sample (paired) t-test, Cauchy(0, r) prior on the effect size."""
    n = len(x)
    t = np.mean(x) / (np.std(x, ddof=1) / np.sqrt(n))

    def integrand(g):
        s = np.sqrt(1 + n * g)
        return np.exp(stats.t.logpdf(t / s, n - 1) - np.log(s) + stats.invgamma.logpdf(g, a=0.5, scale=r ** 2 / 2))

    p_h1, _ = integrate.quad(integrand, 1e-10, np.inf, limit=200)
    return float(np.exp(stats.t.logpdf(t, n - 1)) / p_h1)


def wilcoxon_power(n, d, n_sim=10_000, seed=42, alpha=0.05):
    """Simulated power of the two-sided Wilcoxon signed-rank test for paired differences drawn from N(d, 1)."""
    x = np.random.RandomState(seed).normal(d, 1.0, size=(n_sim, n))
    return float(np.mean(stats.wilcoxon(x, axis=1, alternative="two-sided").pvalue < alpha))


def min_detectable_d(n, target=0.80, lo=0.01, hi=2.0, tol=0.01):
    while hi - lo > tol:  # bisection on the simulated power curve
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if wilcoxon_power(n, mid) < target else (lo, mid)
    return (lo + hi) / 2


def ms_stats(bal):
    s, a = bal.student_true_points.to_numpy(float), bal.ai_true_points.to_numpy(float)
    base, pro = (bal["mode"] == "Base").to_numpy(), (bal["mode"] == "Pro").to_numpy()
    out = {"human_mean": s.mean(), "lm_mean": a.mean(), "share": (s / MS_PIE).mean(),
           "human_median": np.median(s), "lm_median": np.median(a), "d": cohens_d(s - a)}
    for name, m in (("base", base), ("pro", pro)):
        out[f"{name}_human"], out[f"{name}_lm"] = s[m].mean(), a[m].mean()
        out[f"{name}_p"], out[f"{name}_rrb"] = wilcoxon(s[m] - a[m])
    out["mwu_p"] = stats.mannwhitneyu(a[pro], a[base], alternative="greater").pvalue
    out["delta"] = cliffs_delta(a[pro], a[base])
    return out


def ht_stats(bal):
    s, a = bal.student_true_points.to_numpy(float), bal.ai_true_points.to_numpy(float)
    share = bal.student_share.to_numpy(float)
    out = {"human_mean": s.mean(), "lm_mean": a.mean(), "total": bal.total_points.mean(), "share": share.mean(),
           "human_median": np.median(s), "lm_median": np.median(a), "d": cohens_d(s - a)}
    out["p"], out["rrb"] = wilcoxon(s - a)
    out["share_p"], out["share_rrb"] = wilcoxon(share - 0.5)
    return out


# ------------------------------------------------------------------------------------------------- printing
def fmt_p(p):
    return "p<1e-4" if p < 1e-4 else f"p={p:.4f}"


def ci(lo_hi, f="{:,.0f}"):
    return "[" + f.format(lo_hi[0]) + ", " + f.format(lo_hi[1]) + "]"


def table(title, header, widths, rows, paper_rows):
    """Print rows in the paper's layout; under a row that differs from the paper, print the paper's row."""
    line = lambda cells: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    print(f"\n{title}\n{line(header)}\n" + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    same = total = 0
    for row, ref in zip(rows, paper_rows):
        if ref is None:  # section label
            print(row[0])
            continue
        n_cells = len(row) - 1
        hits = sum(str(c) == str(r) for c, r in zip(row[1:], ref))
        same, total = same + hits, total + n_cells
        print(line(row) + ("   = paper" if hits == n_cells else ""))
        if hits < n_cells:
            print(line(["    paper"] + list(ref)))
    print(f"[{same}/{total} cells equal to the paper]")
    return same, total


def conditions_table(title, full, bal, groups, paper):
    rows, refs, it = [], [], iter(np.reshape(paper, (2, -1)).T)
    for label, items in groups:
        rows.append([label])
        refs.append(None)
        for name, mask_full, mask_bal in items:
            rows.append([name, int(mask_full.sum()), int(mask_bal.sum())])
            refs.append(list(next(it)))
    return table(title, ["Condition", f"Full cleaned (n={len(full)})", f"Double-balanced (n={len(bal)})"],
                 [26, 24, 26], rows, refs)


def ci_gap(rows, refs):
    """Print how far the 95% CI bounds are from the paper's: max and median |difference| (surplus; share in pp)."""
    num = lambda s: [float(x.replace(",", "")) for x in re.findall(r"-?[\d,]*\.?\d+", s)]
    gaps, fmt = {"surplus": [], "share (pp)": []}, {"surplus": "{:,.0f}", "share (pp)": "{:.1f}"}
    for row, (_, ref) in zip(rows, refs):
        if ref != "---":
            gaps["share (pp)" if "%" in ref else "surplus"] += [abs(a - b) for a, b in zip(num(row[2]), num(ref))]
    if any(v > 0 for g in gaps.values() for v in g):
        print("  95% CI bounds vs paper, |difference|: " + "; ".join(
            f"{unit} max {fmt[unit].format(max(g))}, median {fmt[unit].format(np.median(g))} ({len(g)} bounds)"
            for unit, g in gaps.items() if g))


def check(label, ours, paper):
    ok = str(ours) == str(paper)
    print(f"  {label:<58} {str(ours):>18}   paper {str(paper):>10}   {'=' if ok else 'DIFF'}")
    return int(ok), 1


def subset_note(deals, by, cells, k):
    sizes = [int(((deals[by[0]] == a) & (deals[by[1]] == b)).sum()) for a, b in cells]
    parts = [f"{a}/{b} first {n}->{k}{' (whole)' if n == k else ''}" for (a, b), n in zip(cells, sizes)]
    n_subsets = np.prod([float(comb(n, k)) for n in sizes])
    print(f"  cells (deals -> balanced): {'; '.join(parts)}; possible subsets: {n_subsets:.3g}")


# --------------------------------------------------------------------------------------------------- tables
def main_street(ms, power):
    deals = deals_of(ms)
    bal, how = subset(deals, ("mode", "first"), MS_CELLS)
    print("\n" + "=" * 100 + f"\nMAIN STREET (human seller vs GPT-5 buyer)\n  balanced subset: {how}")
    if "balanced_subset" not in deals:
        subset_note(deals, ("mode", "first"), MS_CELLS, len(bal) // 4)
    score = []
    first = [("LM first", ms["first"] == "LM", bal["first"] == "LM"),
             ("Student first", ms["first"] == "Student", bal["first"] == "Student")]
    mode = [("Pro", ms["mode"] == "Pro", bal["mode"] == "Pro"), ("Base", ms["mode"] == "Base", bal["mode"] == "Base")]
    cells = [(f"{m}, {f} first", (ms["mode"] == m) & (ms["first"] == f), (bal["mode"] == m) & (bal["first"] == f))
             for m, f in [("Pro", "LM"), ("Pro", "Student"), ("Base", "LM"), ("Base", "Student")]]
    score.append(conditions_table("Table: Main Street experimental conditions", ms, bal,
                                  [("First mover", first), ("LM mode", mode),
                                   ("2x2 design (Mode x First mover)", cells)], PAPER["ms_conditions"]))

    st = ms_stats(bal)
    s, a = bal.student_true_points.to_numpy(float), bal.ai_true_points.to_numpy(float)
    blocks = boot_blocks(MS_SEED, len(bal), 5)
    rows = [["Human mean surplus", f"{st['human_mean']:,.0f}", ci(boot_ci(s, blocks, MS_BLOCKS["mean"]))],
            ["LM mean surplus", f"{st['lm_mean']:,.0f}", ci(boot_ci(a, blocks, MS_BLOCKS["mean"]))],
            ["Total pie", f"{(s + a).mean():,.0f}", "---"],
            ["Human share of pie", f"{st['share']:.1%}",
             ci(boot_ci(s / MS_PIE, blocks, MS_BLOCKS["mean"]), "{:.1%}")],
            ["Human median surplus", f"{st['human_median']:,.0f}",
             ci(boot_ci(s, blocks, MS_BLOCKS["human_median"], np.median))],
            ["LM median surplus", f"{st['lm_median']:,.0f}", ci(boot_ci(a, blocks, MS_BLOCKS["lm_median"], np.median))]]
    score.append(table(f"Table: Main Street overview statistics (double-balanced dataset, n={len(bal)})",
                       ["Metric", "Value", "95% CI"], [22, 10, 22], rows, PAPER["ms_overview"]))
    ci_gap(rows, PAPER["ms_overview"])

    nb, npro = int((bal["mode"] == "Base").sum()), int((bal["mode"] == "Pro").sum())
    rows = [["Human vs LM (Base)", nb, f"{st['base_human']:,.0f}", f"{st['base_lm']:,.0f}", "Wilcoxon (two-sided)",
             f"{fmt_p(st['base_p'])}; r_rb={st['base_rrb']:.3f}"],
            ["Human vs LM (Pro)", npro, f"{st['pro_human']:,.0f}", f"{st['pro_lm']:,.0f}", "Wilcoxon (two-sided)",
             f"{fmt_p(st['pro_p'])}; r_rb={st['pro_rrb']:.3f}"],
            ["LM (Pro) vs LM (Base)", f"{npro}/{nb}", f"{st['pro_lm']:,.0f}", f"{st['base_lm']:,.0f}",
             "MWU (one-sided; Pro>Base)", f"{fmt_p(st['mwu_p'])}; delta={st['delta']:.3f}"]]
    refs = [(r[0], r[1], r[2], t, r[3]) for r, t in
            zip(PAPER["ms_tests"], ["Wilcoxon (two-sided)", "Wilcoxon (two-sided)", "MWU (one-sided; Pro>Base)"])]
    score.append(table(f"Table: Main Street (double-balanced, n={len(bal)}): human-LM comparisons and scaffolding",
                       ["Comparison", "n", "Group A Mean", "Group B Mean", "Test (H1)", "p / Effect"],
                       [22, 6, 12, 12, 26, 24], rows, refs))
    pw = None
    if power:
        diff = s - a
        pw = {"n": len(bal), "d": cohens_d(diff), "power": wilcoxon_power(len(bal), abs(cohens_d(diff))),
              "bf10": 1 / jzs_bf01(diff), "p": wilcoxon(diff)[0]}
    return deals, bal, st, pw, score


def hire_talent(ht, power):
    deals = deals_of(ht)
    bal, how = subset(deals, ("role", "first"), HT_CELLS)
    print("\n" + "=" * 100 + f"\nHIRE TALENT (human vs GPT-5 in pro mode)\n  balanced subset: {how}")
    if "balanced_subset" not in deals:
        subset_note(deals, ("role", "first"), HT_CELLS, len(bal) // 4)
    score = []
    role = [("Recruiter (side1)", ht.role == "Recruiter", bal.role == "Recruiter"),
            ("Candidate (side2)", ht.role == "Candidate", bal.role == "Candidate")]
    first = [("LM first", ht["first"] == "LM", bal["first"] == "LM"),
             ("Student first", ht["first"] == "Student", bal["first"] == "Student")]
    cells = [(f"{r}, {f} first", (ht.role == r) & (ht["first"] == f), (bal.role == r) & (bal["first"] == f))
             for r, f in HT_CELLS]
    score.append(conditions_table("Table: Hire Talent experimental conditions", ht, bal,
                                  [("Student role", role), ("First mover", first),
                                   ("2x2 design (Role x First mover)", cells)], PAPER["ht_conditions"]))

    st = ht_stats(bal)
    s, a = bal.student_true_points.to_numpy(float), bal.ai_true_points.to_numpy(float)
    blocks = boot_blocks(HT_SEED, len(bal), 18)
    rows = [["Human mean surplus", f"{st['human_mean']:,.0f}", ci(boot_ci(s, blocks, HT_BLOCKS["human_mean"]))],
            ["LM mean surplus", f"{st['lm_mean']:,.0f}", ci(boot_ci(a, blocks, HT_BLOCKS["lm_mean"]))],
            ["Total pie (mean)", f"{st['total']:,.0f}", ci(boot_ci(bal.total_points, blocks, HT_BLOCKS["total"]))],
            ["Human share of pie", f"{st['share']:.1%}",
             ci(boot_ci(bal.student_share, blocks, HT_BLOCKS["share"]), "{:.1%}")],
            ["Human median surplus", f"{st['human_median']:,.0f}",
             ci(boot_ci(s, blocks, HT_BLOCKS["human_median"], np.median))],
            ["LM median surplus", f"{st['lm_median']:,.0f}", ci(boot_ci(a, blocks, HT_BLOCKS["lm_median"], np.median))]]
    score.append(table(f"Table: Hire Talent overview statistics (double-balanced dataset, n={len(bal)})",
                       ["Metric", "Value", "95% CI"], [22, 10, 22], rows, PAPER["ht_overview"]))
    ci_gap(rows, PAPER["ht_overview"])

    rows = [["Human vs LM", len(bal), f"{st['human_mean']:,.0f}", f"{st['lm_mean']:,.0f}", "Wilcoxon (two-sided)",
             f"{fmt_p(st['p'])}; r_rb={st['rrb']:.3f}"],
            ["Human share vs 50%", len(bal), f"{st['share']:.1%}", "50.0%", "Wilcoxon (two-sided)",
             f"{fmt_p(st['share_p'])}; r_rb={st['share_rrb']:.3f}"]]
    refs = [(r[0], r[1], r[2], "Wilcoxon (two-sided)", r[3]) for r in PAPER["ht_tests"]]
    score.append(table(f"Table: Hire Talent (double-balanced, n={len(bal)}): human-LM surplus comparison",
                       ["Comparison", "n", "Mean A", "Mean B", "Test (H1)", "p / Effect"],
                       [22, 6, 12, 12, 26, 24], rows, refs))
    pw = None
    if power:
        diff = s - a
        pw = {"n": len(bal), "d": cohens_d(diff), "power05": wilcoxon_power(len(bal), 0.5),
              "dstar": min_detectable_d(len(bal)), "bf01": jzs_bf01(diff)}
    return deals, bal, st, pw, score


def power_table(ms_pw, ht_pw):
    power = ">0.99 at observed d" if ms_pw["power"] > 0.99 else f"{ms_pw['power']:.3f} at observed d"
    rows = [["Main Street (1 issue)", ms_pw["n"], f"{ms_pw['d']:+.3f}", power, f"BF10 = {ms_pw['bf10']:.0f} (LM)"],
            ["Hire Talent (7 issues)", ht_pw["n"], f"{ht_pw['d']:+.3f}",
             f"{ht_pw['power05']:.2f} at d=0.5; d*={ht_pw['dstar']:.3f} at 0.80",
             f"BF01 = {ht_pw['bf01']:.2f} ({'parity' if ht_pw['bf01'] > 1 else 'difference'})"]]
    return table("Table: Post-hoc power and Bayesian analyses for the human-LM scenarios",
                 ["Scenario", "n", "Cohen's d", "Power / d*", "Bayes factor"], [22, 4, 10, 34, 22], rows,
                 PAPER["power"])


def main_text(ms, ht, hh, ms_st, ht_st, ms_pw, ht_pw):
    print("\nMain-text and appendix-text numbers")
    ms_deals, ht_deals = int((ms.deal_reached == 1).sum()), int((ht.deal_reached == 1).sum())
    out = [check("Human interaction sessions (89 + 55 + 23)", len(ms) + len(ht) + len(hh), 167),
           check("SnyderMed human-human pairs", len(hh), 23),
           check("SnyderMed mean normalized total pie", f"{hh.normalized_total_pie.mean():.3f}", "0.874"),
           check("SnyderMed SD normalized total pie", f"{hh.normalized_total_pie.std():.3f}", "0.085"),
           check("SnyderMed median normalized total pie", f"{hh.normalized_total_pie.median():.3f}", "0.857"),
           check("Main Street sessions", len(ms), 89),
           check("Main Street deals", ms_deals, 82),
           check("Main Street deal rate", f"{ms_deals / len(ms):.1%}", "92.1%"),
           check("Main Street: all students seller (side2), all LMs GPT-5",
                 bool((ms.student_role == "side2").all() and (ms.ai_model == "openai/gpt-5").all()), True),
           check("Main Street students' share of pie (balanced)", f"{ms_st['share']:.1%}", "39.7%"),
           check("Main Street agent's share of pie (balanced)", f"{1 - ms_st['share']:.1%}", "60.3%"),
           check("Main Street Base: p", fmt_p(ms_st["base_p"]), "p=0.0186"),
           check("Main Street Base: r_rb", f"{ms_st['base_rrb']:.3f}", "-0.484"),
           check("Main Street Pro: p", fmt_p(ms_st["pro_p"]), "p<1e-4"),
           check("Main Street Pro: r_rb", f"{ms_st['pro_rrb']:.3f}", "-0.825"),
           check("Main Street Pro vs Base: p (one-sided MWU)", fmt_p(ms_st["mwu_p"]), "p=0.2089"),
           check("Main Street Pro vs Base: Cliff's delta", f"{ms_st['delta']:.3f}", "0.115"),
           check("Hire Talent sessions", len(ht), 55),
           check("Hire Talent deals", ht_deals, 51),
           check("Hire Talent deal rate", f"{ht_deals / len(ht):.1%}", "92.7%"),
           check("Hire Talent: all sessions GPT-5 in pro mode",
                 bool(((ht.use_memory == 1) & (ht.use_plan == 1)).all() and (ht.ai_model == "openai/gpt-5").all()),
                 True),
           check("Hire Talent students' share of pie (balanced)", f"{ht_st['share']:.1%}", "53.8%"),
           check("Hire Talent agent's share of pie (balanced)", f"{1 - ht_st['share']:.1%}", "46.2%"),
           check("Hire Talent human vs LM: p", fmt_p(ht_st["p"]), "p=0.2674"),
           check("Hire Talent human vs LM: r_rb", f"{ht_st['rrb']:.3f}", "0.201")]
    if ms_pw and ht_pw:
        out += [check("Power text, Main Street: Cohen's d", f"{ms_pw['d']:.3f}", "-0.559"),
                check("Power text, Main Street: p", fmt_p(ms_pw["p"]), "p<1e-4"),
                check("Power text, Main Street: BF10", f"{ms_pw['bf10']:.0f}", "977"),
                check("Power text, Hire Talent: power at d=0.5", f"{ht_pw['power05']:.0%}", "85%"),
                check("Power text, Hire Talent: d* at 80% power", f"{ht_pw['dstar']:.3f}", "0.465"),
                check("Power text, Hire Talent: observed d", f"{ht_pw['d']:.3f}", "0.186"),
                check("Power text, Hire Talent: BF01", f"{ht_pw['bf01']:.2f}", "3.09"),
                check("Power text, Hire Talent: BF01 rounded", f"{ht_pw['bf01']:.1f}x", "3.1x")]
    hits = sum(o[0] for o in out)
    print(f"[{hits}/{len(out)} numbers equal to the paper]")
    return hits, len(out)


def draw_summary(deals, by, cells, stat_fn, keys, n_draws, title):
    """Spread of balanced-subset statistics over seeded draws 0..n_draws-1 (same sampler, other seeds)."""
    df = pd.DataFrame([stat_fn(balanced(deals, by, cells, seed=s)) for s in range(n_draws)])
    print(f"\n{title}: {n_draws} seeded balanced draws from the released deals (seeds 0-{n_draws - 1})")
    print(f"  {'statistic':<24}{'2.5%':>10}{'median':>10}{'97.5%':>10}{'paper':>10}{'draws <= paper':>16}")
    for key, label, f, paper, alpha in keys:
        q = np.percentile(df[key], [2.5, 50, 97.5])
        shown = f"<{alpha:g}" if alpha and paper == alpha else f.format(paper)
        note = f"   p<{alpha:g} in {(df[key] < alpha).mean():.1%} of draws" if alpha else ""
        print(f"  {label:<24}" + "".join(f"{f.format(v):>10}" for v in q) + f"{shown:>10}"
              + f"{(df[key] <= paper).mean():>16.1%}" + note)


def all_deals_line(ms_deals, ht_deals):
    """The same tests on all deals, without balancing (not reported in the paper)."""
    ms, ht = ms_stats(ms_deals), ht_stats(ht_deals)
    nb, npro = int((ms_deals["mode"] == "Base").sum()), int((ms_deals["mode"] == "Pro").sum())
    print(f"\nAll deals, no balancing (not reported in the paper):\n"
          f"  Main Street Base (n={nb}): {fmt_p(ms['base_p'])}, r_rb={ms['base_rrb']:.3f}; "
          f"Pro (n={npro}): {fmt_p(ms['pro_p'])}, r_rb={ms['pro_rrb']:.3f}; "
          f"Pro>Base: {fmt_p(ms['mwu_p'])}, delta={ms['delta']:.3f}; human share {ms['share']:.1%}\n"
          f"  Hire Talent (n={len(ht_deals)}): {fmt_p(ht['p'])}, r_rb={ht['rrb']:.3f}; "
          f"share vs 50%: {fmt_p(ht['share_p'])}; human share {ht['share']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draws", type=int, default=0, help="seeded redraws to summarise (default: %(default)s)")
    ap.add_argument("--no-power", action="store_true", help="skip the simulation-based power analysis")
    ap.add_argument("--human-data", type=Path, default=HUMAN, help="folder of the human CSVs (default: %(default)s)")
    args = ap.parse_args()

    ms, ht, hh = load(args.human_data)
    ms_deals, _, ms_st, ms_pw, ms_score = main_street(ms, not args.no_power)
    ht_deals, _, ht_st, ht_pw, ht_score = hire_talent(ht, not args.no_power)
    score = ms_score + ht_score
    if not args.no_power:
        print("\n" + "=" * 100)
        score.append(power_table(ms_pw, ht_pw))
    score.append(main_text(ms, ht, hh, ms_st, ht_st, ms_pw, ht_pw))
    print(f"\nTotal: {sum(s[0] for s in score)}/{sum(s[1] for s in score)} printed numbers equal to the paper")

    if args.draws > 0:
        f0, f3, f4 = "{:,.0f}", "{:.3f}", "{:.4f}"
        draw_summary(ms_deals, ("mode", "first"), MS_CELLS, ms_stats,
                     [("human_mean", "human mean surplus", f0, 79449, None),
                      ("share", "human share of pie", f3, 0.397, None),
                      ("base_p", "Base: Wilcoxon p", f4, 0.0186, 0.05), ("base_rrb", "Base: r_rb", f3, -0.484, None),
                      ("pro_p", "Pro: Wilcoxon p", f4, 1e-4, 1e-4), ("pro_rrb", "Pro: r_rb", f3, -0.825, None),
                      ("mwu_p", "Pro>Base: MWU p", f4, 0.2089, 0.05),
                      ("delta", "Pro>Base: Cliff's delta", f3, 0.115, None),
                      ("d", "Cohen's d (paired)", f3, -0.559, None)], args.draws, "Main Street")
        draw_summary(ht_deals, ("role", "first"), HT_CELLS, ht_stats,
                     [("human_mean", "human mean surplus", f0, 28388, None),
                      ("lm_mean", "LM mean surplus", f0, 22682, None), ("total", "total pie (mean)", f0, 51070, None),
                      ("share", "human share of pie", f3, 0.538, None), ("p", "Wilcoxon p", f4, 0.2674, 0.05),
                      ("rrb", "r_rb", f3, 0.201, None), ("share_p", "share vs 50%: p", f4, 0.4008, 0.05),
                      ("share_rrb", "share vs 50%: r_rb", f3, 0.152, None),
                      ("d", "Cohen's d (paired)", f3, 0.186, None)], args.draws, "Hire Talent")
        all_deals_line(ms_deals, ht_deals)


if __name__ == "__main__":
    main()
