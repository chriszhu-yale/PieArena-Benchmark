"""Load the released dataset (``dataset/lm_data/transcripts``) for ranking and profiling."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRANSCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "lm_data" / "transcripts"
REFERENCE_MODEL = "google/gemini-2.5-flash"   # BBTL anchor (theta = 0)
FINAL_POOL = (                                # the 13 LMs ranked in the paper
    "openai/gpt-5", "openai/gpt-5.2", "openai/gpt-4.1", "openai/o4-mini-high",
    "google/gemini-3-pro-preview", "google/gemini-2.5-flash",
    "anthropic/claude-opus-4.5", "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4", "x-ai/grok-4.1-fast", "x-ai/grok-3", "x-ai/grok-3-mini-beta",
    "baidu/ernie-4.5-vl-424b-a47b",
)
SLICES = {  # scenario groups of the cross-play leaderboards
    "Integrated": ("Hire_Talent", "IP_License", "Service_Quote"),
    "Multi-Issue": ("Hire_Talent", "IP_License"),
    "Single-Issue": ("Service_Quote",),
}


def load_runs(play_type: str = "cross_play", modes=("base_mode", "pro_mode"), root: Path = TRANSCRIPTS,
              files: tuple[str, ...] = ("pair_runs_table.csv",)) -> pd.DataFrame:
    """Concatenate per-pair tables under ``root/<play_type>/<mode>/<scenario>/<pair>/``.

    Adds ``play_type``, ``mode`` and ``pair_dir`` columns. With several ``files``
    (e.g. ``reputation_score.csv``) they are merged on ``run_number`` per pair.
    """
    frames = []
    for mode in modes:
        for pair_dir in sorted((Path(root) / play_type / mode).glob(f"*/*/{files[0]}")):
            df = pd.read_csv(pair_dir)
            for extra in files[1:]:
                path = pair_dir.parent / extra
                if path.exists():
                    other = pd.read_csv(path)
                    df = df.merge(other[[c for c in other.columns if c == "run_number" or c not in df.columns]],
                                  on="run_number", how="left")
            df["play_type"], df["mode"], df["pair_dir"] = play_type, mode, pair_dir.parent.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def paper_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Runs used for the paper's analyses (balanced design: 6 runs per ordered pairing, scenario and mode)."""
    return df[df["paper_analysis"].astype(bool)]


def filter_a(df: pd.DataFrame) -> pd.DataFrame:
    """Drop runs without a verified deal and BATNA-violating runs (pie shares outside [0, 1])."""
    df = df.dropna(subset=["p_i", "p_j"])
    return df[df.p_i.between(0, 1) & df.p_j.between(0, 1)]


def in_pool(df: pd.DataFrame, pool=FINAL_POOL) -> pd.DataFrame:
    return df[df.side1_model.isin(pool) & df.side2_model.isin(pool)]


def fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Columns expected by :func:`piearena.bbtl_fit.fit_bbtl` and the baselines."""
    return pd.DataFrame({
        "lm_i": df.side1_model.astype(str), "lm_j": df.side2_model.astype(str),
        "p_i": df.p_i.astype(float), "p_j": df.p_j.astype(float),
        "scenario": df.scenario.astype(str), "who_first": df.who_first.astype(str),
        "mode": df["mode"].astype(str),
    }).reset_index(drop=True)
