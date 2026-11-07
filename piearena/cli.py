"""Command-line entry point: ``python -m piearena --scenario ... --models ...``."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .runner import run_pair_n_runs
from .scenarios import load_scenario


def resolve_harness(use_memory: bool | None, use_plan: bool | None) -> tuple[bool, bool]:
    """No flag -> full harness (pro mode); otherwise unspecified modules are off."""
    if use_memory is None and use_plan is None:
        return True, True
    return bool(use_memory), bool(use_plan)


def pairings(models: list[str], selfplay_only: bool) -> list[tuple[str, str]]:
    if len(models) == 1 or selfplay_only:
        return [(m, m) for m in models]
    return [(a, b) for a in models for b in models if a != b]  # both side assignments


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="piearena", description="Run automated LM negotiation experiments.")
    ap.add_argument("--scenario", required=True, metavar="NAME|PATH",
                    help="scenario name in dataset/scenarios/ or a path to a YAML file")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model ids (space- or comma-separated); all ordered pairs are run")
    ap.add_argument("--n-runs", type=int, required=True, help="negotiations per ordered pairing")
    ap.add_argument("--rounds", type=int, required=True, help="rounds per negotiation")
    ap.add_argument("--output-dir", default="results", help="root directory for results (default: results)")
    ap.add_argument("--exp-name", help="results go to <output-dir>/<scenario>/<exp-name> (re-running resumes); "
                                       "default: a timestamp")
    ap.add_argument("--pair-mode", action="store_true", help="run only the ordered pair given by two --models")
    ap.add_argument("--selfplay-only", action="store_true", help="run only (M, M) mirror-play pairings")
    ap.add_argument("--use-memory", dest="use_memory", action="store_true", help="enable state tracking")
    ap.add_argument("--no-use-memory", dest="use_memory", action="store_false", help="disable state tracking")
    ap.add_argument("--use-plan", dest="use_plan", action="store_true", help="enable the planning module")
    ap.add_argument("--no-use-plan", dest="use_plan", action="store_false", help="disable the planning module")
    ap.set_defaults(use_memory=None, use_plan=None)
    ap.add_argument("--verbose", action="store_true", help="print the dialogue as it happens")
    ap.add_argument("--prompt-verbosity", action="store_true", help="print every prompt sent to the models")
    args = ap.parse_args(argv)

    scenario_name, config = load_scenario(args.scenario, rounds=args.rounds)
    models = [m.strip() for token in args.models for m in token.split(",") if m.strip()]
    if args.pair_mode:
        if len(models) != 2:
            raise SystemExit("--pair-mode requires exactly two --models (side1 side2).")
        pairs = [tuple(models)]
    else:
        pairs = pairings(models, args.selfplay_only)
    exp = args.exp_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / scenario_name / ("" if exp == "null" else exp)
    use_memory, use_plan = resolve_harness(args.use_memory, args.use_plan)
    print(f"{len(pairs)} pairing(s) x {args.n_runs} runs -> {out_dir}")
    for side1, side2 in pairs:
        print(f"\n🔁 {side1} (side 1) vs {side2} (side 2)")
        run_pair_n_runs(side1, side2, config, out_dir, args.n_runs, scenario_name, use_memory=use_memory,
                        use_plan=use_plan, verbose=args.verbose, prompt_verbosity=args.prompt_verbosity)
