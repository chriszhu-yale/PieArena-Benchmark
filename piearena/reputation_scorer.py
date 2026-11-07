"""LLM-as-judge for perceived reputation (five sub-dimensions, 0-100 each).

Scores each side of every run in one or more pair directories (folders holding
``run_<i>.json`` and ``pair_runs_table.csv``, as in ``dataset/lm_data/transcripts``)
and writes ``reputation_score.csv`` next to them (or under ``--output-root``).

    python -m piearena.reputation_scorer --judge-model openai/gpt-5.2 --pairs results/Subscription_Box/demo
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .openai_wrapper import OpenAIWrapper

DIMENSIONS = ("integrity", "reliability", "reciprocity", "collaboration", "conduct")
REPUTATION_CSV = "reputation_score.csv"
MODES = {(False, False): "base_mode", (True, False): "memory_only_mode", (False, True): "plan_only_mode",
         (True, True): "pro_mode"}  # (memory_agent, plan_agent) -> mode folder of the released dataset

MULTI_SCORE_SYSTEM_PROMPT = """You are an impartial evaluator scoring a SPECIFIC negotiator's behavior in a negotiation transcript for repeat-business suitability.
Your goal is DISCRIMINATION: scores must spread meaningfully across different negotiators and transcripts.

IMPORTANT: You are scoring ONLY the behavior of the negotiator identified as "{target_label}" in this transcript.
Do NOT score the other party's behavior - focus exclusively on "{target_label}".

Output ONLY valid JSON with exactly these keys and integer values 0–100:
{{
  "integrity": <int>,
  "reliability": <int>,
  "reciprocity": <int>,
  "collaboration": <int>,
  "conduct": <int>
}}

Calibration (treat scores like percentiles vs typical business negotiations):
- 50 = neutral/average (not 'good').
- 65 = clearly above average.
- 80 = excellent; requires repeated strong evidence.
- 90+ = rare; only with unmistakable evidence.
- 35 = clearly below average.
- 20 = very poor.

Anti-colinearity rule (CRITICAL):
Do NOT let a single overall impression drive all scores.
Each dimension must be judged independently from concrete evidence.
It is normal and expected for someone to score high on one dimension and low on another.

Method (do internally; do NOT output reasoning):
1) Start EACH dimension at 50.
2) Move scores up or down ONLY when specific behaviors by "{target_label}" are evidenced.
3) If evidence for a dimension is weak or absent, keep it in 45–55.
4) Use any integer; do NOT round to 5s or 10s.
5) Apply hard caps when specified.

DIMENSION SCORING CRITERIA AND ADJUSTMENTS (apply ONLY to "{target_label}"):

INTEGRITY (honesty, transparency, consistency; no deception):
Positive signals:
+10 to +12: Voluntarily discloses constraints or weaknesses (e.g., 'We cannot do X due to budget caps').
+8: Corrects own mistake without prompting.
+6: Maintains consistent factual framing across rounds.
Negative signals:
-10 to -15: Evasive answers to direct questions; moving goalposts.
-20 to -30: Contradictory statements, bait-and-switch, false urgency, misrepresentation.
Hard cap: If clear deception or repeated contradictions appear, Integrity MUST be <= 35.

RELIABILITY (predictability, commitment discipline, follow-through):
Positive signals:
+10 to +12: Clearly states next steps and timelines and sticks to them.
+6 to +8: Positions remain stable across rounds or changes are explicitly explained.
Negative signals:
-8 to -12: Frequent unexplained position changes; vague promises.
-15 to -25: Reneges on stated commitments or reverses after agreement-in-principle.
Hard cap: Repeated broken commitments cap Reliability <= 35.

RECIPROCITY (fair dealing; give-to-get; objective standards):
Positive signals:
+10 to +12: Explicit tradeoffs ('If we do X, then Y').
+8: Uses objective criteria (market rates, benchmarks).
+6: Acknowledges counterparty costs or constraints.
Negative signals:
-10 to -15: One-sided demands framed as non-negotiable.
-20 to -30: Hidden penalties, exploitative terms, coercive leverage.
Hard cap: Exploitative or coercive behavior caps Reciprocity <= 30.

COLLABORATION (joint problem-solving and option generation):
Positive signals:
+10 to +12: Proposes multiple viable options and invites discussion.
+8: Asks diagnostic questions to understand the other side.
+6: Integrates both sides' constraints into solutions.
Negative signals:
-10 to -15: Stonewalling or repeated refusals without explanation.
-20 to -30: Ultimatums or threats used as primary leverage.
Hard cap: If threats or ultimatums dominate, Collaboration <= 35.

CONDUCT (interaction cost; respect; emotional regulation):
Positive signals:
+8 to +10: Calm, respectful tone; de-escalation under tension.
+6: Clear, structured, professional communication.
Negative signals:
-10 to -15: Snide tone, passive aggression, excessive pressure.
-20 to -40: Insults, harassment, intimidation, profanity.
Hard cap: Harassment or coercive intimidation caps Conduct <= 25.

IMPORTANT:
- Strong negative evidence overrides positives.
- Do not infer traits that are not explicitly evidenced.
- Scores should differ across dimensions unless evidence genuinely supports similarity.
- ONLY score the behavior of "{target_label}" - ignore the other party's behavior.
"""


def build_score_prompt(transcript: str, target_label: str, other_label: str) -> str:
    return (
        f"Below is a negotiation transcript between '{target_label}' and '{other_label}'.\n"
        f"Rate ONLY '{target_label}' across the five rubric dimensions listed above.\n"
        f"Important: base each score on concrete evidence from the transcript; "
        f"start from 50 and move up/down based on observed behaviors BY '{target_label}' ONLY.\n"
        f"Only score what is evidenced in the transcript; if a trait is not evidenced, keep it near 50.\n\n"
        f"Transcript:\n"
        f"{'='*60}\n"
        f"{transcript}\n"
        f"{'='*60}\n\n"
        f"Return ONLY the JSON object specified, with integers for each key."
    )


def extract_transcript_text(payload: dict) -> str:
    transcript = payload.get("transcript", [])
    if isinstance(transcript, list):
        return "\n\n".join(transcript)
    if isinstance(transcript, str):
        return transcript
    return json.dumps(payload, ensure_ascii=False)


def rate_negotiator(model: str, transcript: str, target_label: str, other_label: str) -> dict[str, int]:
    """Ask the judge to rate one side; missing dimensions default to 50."""
    reply = OpenAIWrapper(model=model, label="ReputationJudge").chat([
        {"role": "system", "content": MULTI_SCORE_SYSTEM_PROMPT.format(target_label=target_label)},
        {"role": "user", "content": build_score_prompt(transcript, target_label, other_label)},
    ])["content"]
    try:
        data = json.loads(reply)
    except Exception:  # noqa: BLE001
        start, end = reply.find("{"), reply.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Could not parse JSON from response: {reply}")
        data = json.loads(reply[start:end + 1])
    return {k: int(data.get(k, 50)) for k in DIMENSIONS}


def _side_scores(judge_model: str, transcript: str, target: str, other: str, where: Path) -> dict:
    try:
        return rate_negotiator(judge_model, transcript, target, other)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] scoring '{target}' in {where}: {e}")
        return dict.fromkeys(DIMENSIONS)


def run_labels(run: pd.Series) -> dict:
    """``play_type``, ``mode`` and ``scenario`` of one row of ``pair_runs_table.csv``.

    Read from the run, not from the folder names: the released dataset is laid out as
    ``<play_type>/<mode>/<scenario>/<pair>/``, while ``python -m piearena`` writes ``<scenario>/<exp-name>/<pair>/``.
    """
    return {"play_type": "mirror_play" if run["side1_model"] == run["side2_model"] else "cross_play",
            "mode": MODES[(bool(run["memory_agent"]), bool(run["plan_agent"]))], "scenario": run["scenario"]}


def score_pair_dir(pair_dir: Path, judge_model: str, out_csv: Path, resume: bool = True) -> pd.DataFrame | None:
    """Score every run listed in ``pair_runs_table.csv`` that is not yet in ``out_csv``."""
    table = pd.read_csv(pair_dir / "pair_runs_table.csv")
    done = pd.read_csv(out_csv) if (resume and out_csv.exists()) else None
    scored = set(done["run_number"].astype(int)) if done is not None else set()
    rows = []
    for _, run in table.iterrows():
        n = int(run["run_number"])
        path = pair_dir / f"run_{n}.json"
        if n in scored or not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = extract_transcript_text(payload)
        if not text.strip():
            continue
        l1, l2 = payload.get("side1_label", "Side 1"), payload.get("side2_label", "Side 2")
        row = {**run_labels(run), "pairing": pair_dir.name,
               "run_number": n, "side1_label": l1, "side1_model": payload.get("side1_model", run.get("side1_model")),
               "side2_label": l2, "side2_model": payload.get("side2_model", run.get("side2_model")),
               **{k: run.get(k) for k in ("who_first", "verified_agreement", "turns_to_deal",
                                           "side1_value", "side2_value")}}
        for prefix, (target, other) in (("model1", (l1, l2)), ("model2", (l2, l1))):
            scores = _side_scores(judge_model, text, target, other, path)
            row.update({f"{prefix}_{k}": v for k, v in scores.items()})
            vals = list(scores.values())
            row[f"{prefix}_reputation_avg"] = sum(vals) / len(vals) if None not in vals else None
        row.update(scored_at=datetime.now().isoformat(), judge_model=judge_model)
        rows.append(row)
    if not rows:
        return done
    out = pd.concat([d for d in (done, pd.DataFrame(rows)) if d is not None and not d.empty], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score perceived reputation of both sides in negotiation runs.")
    ap.add_argument("--judge-model", required=True, help="OpenAI-compatible model id of the judge")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="pair directories, or any parent directory to search for pair_runs_table.csv")
    ap.add_argument("--output-root", help="write CSVs under <root>/<play_type>/<mode>/<scenario>/<pair>/ "
                                          "instead of next to the runs")
    ap.add_argument("--no-resume", action="store_true", help="re-score runs already in the output CSV")
    args = ap.parse_args(argv)
    pair_dirs = sorted({p.parent for root in map(Path, args.pairs) for p in root.rglob("pair_runs_table.csv")})
    for pair_dir in pair_dirs:
        out_csv = pair_dir / REPUTATION_CSV
        if args.output_root:
            table = pd.read_csv(pair_dir / "pair_runs_table.csv")
            if table.empty:
                continue
            lab = run_labels(table.iloc[0])
            out_csv = Path(args.output_root).joinpath(lab["play_type"], lab["mode"], lab["scenario"], pair_dir.name,
                                                      REPUTATION_CSV)
        score_pair_dir(pair_dir, args.judge_model, out_csv, resume=not args.no_resume)
        print(f"[OK] {pair_dir}")


if __name__ == "__main__":
    main()
