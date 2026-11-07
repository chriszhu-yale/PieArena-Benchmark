# PieArena

Code for **PieArena: Ranking and Profiling Language Agents in Realistic Negotiation Scenarios** (NeurIPS 2026, Evaluations & Datasets track).

Chris Zhu, Sasha Cui, Will Sanok Dufallo, Runzhi Jin, Zhen Xu, Linjun Zhang, Daylian M. Cain

PieArena evaluates language models (LMs) in two-party negotiations with private information and deterministic payoffs. This repository contains the negotiation harness, the shared-intentionality agentic harness ("pro" mode), the LLM judges for lies and perceived reputation, the BBTL skill-ranking model, the web application used for human–LM sessions, scenarios in the PieArena format, and scripts that reproduce the paper's main results.

The dataset (transcripts, per-run tables, judge outputs and human data) is hosted on Harvard Dataverse: https://doi.org/10.7910/DVN/UYI2O5. Its documentation is in [`dataset/README.md`](dataset/README.md).

## Repository layout

```
.
├── piearena/                  # Python package
│   ├── cli.py                 # `python -m piearena ...`: run model pairings on a scenario
│   ├── runner.py              # one negotiation episode; repeated runs of one ordered pairing
│   ├── prompts.py             # harness prompts (turns, confirmation, JSON repair, state tracking, planning)
│   ├── scenarios.py           # scenario loading and scoring conventions
│   ├── scorers/               # deterministic deal verification and scoring
│   ├── liar_judge_core.py     # LLM judge: lies
│   ├── reputation_scorer.py   # LLM judge: perceived reputation
│   ├── bbtl_fit.py            # Beta-Generalized Bradley–Terry–Luce (BBTL) model
│   ├── elo_baseline.py        # Elo and logistic BTL baselines used for comparison
│   ├── data.py                # loaders for the dataset
│   ├── openai_wrapper.py      # OpenAI-compatible client with retries (e.g. OpenRouter)
│   └── lm_human/              # web application for human–LM sessions
├── scenarios/                 # scenarios in the PieArena format
├── scripts/
│   ├── reproduce_leaderboard.py   # cross-play leaderboards, per slice and per scenario
│   ├── bbtl_robustness.py         # BBTL nuisance parameters, LR test, cross-validation, CI coverage
│   ├── reproduce_human_lm.py      # human–LM tables: conditions, outcomes, tests, power
│   ├── reproduce_mirror_play.py   # human–human baseline, mirror play vs. humans, pro-mode effect
│   └── reproduce_ablation.py      # harness ablation
├── dataset/                   # dataset documentation, Croissant metadata, outcome tables and human data
├── requirements.txt
└── setup/environment.yml
```

## Setup

```bash
conda env create -f setup/environment.yml && conda activate piearena
# or: pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY (and OPENAI_BASE_URL for OpenRouter)
```

## Running negotiations

All ordered pairings of the given models, six rounds, six runs per pairing, full agentic harness:

```bash
python -m piearena --scenario scenarios/Subscription_Box.yaml --rounds 6 --n-runs 6 \
  --models openai/gpt-5 anthropic/claude-opus-4.5 --exp-name demo
```

- `--scenario` takes the path of a scenario YAML; see [`scenarios/`](scenarios/).
- Base mode (no harness): `--no-use-memory --no-use-plan`; state tracking only: `--use-memory`.
- `--pair-mode` runs only the given ordered pair; `--selfplay-only` runs mirror-play (M vs M).
- Results go to `results/<scenario>/<exp-name>/<side1>_vs_<side2>/` with the same files as the dataset: `run_<i>.json`, `pair_runs_table.csv`, `lies_report.jsonl`. Re-running a command resumes it.

Perceived reputation (LLM judge) for finished runs:

```bash
python -m piearena.reputation_scorer --judge-model openai/gpt-5.2 --pairs results/Subscription_Box/demo
```

## Reproducing the paper's results

```bash
python scripts/reproduce_leaderboard.py
python scripts/bbtl_robustness.py   # about 25 s
python scripts/reproduce_human_lm.py
python scripts/reproduce_mirror_play.py
python scripts/reproduce_ablation.py
```

- `reproduce_leaderboard.py` fits BBTL on the cross-play outcomes in `dataset/lm_data/paper_crossplay_outcomes.csv` for every slice (integrated, multi-issue, single-issue) and mode, the pooled main-text leaderboard and the per-scenario leaderboards.
- `bbtl_robustness.py` reproduces the BBTL nuisance parameters and likelihood-ratio test, the cross-validation against the baselines, and the CI coverage study.
- `reproduce_human_lm.py` reproduces the human–LM tables (conditions, outcomes, tests and power analysis) from `dataset/human_data/`; bootstrap intervals agree up to resampling noise.
- `reproduce_mirror_play.py` reproduces the human–human baseline, the mirror-play comparison with it and the effect of pro mode; `reproduce_ablation.py` reproduces the harness ablation. Both read the per-pair tables in `dataset/lm_data/transcripts/`, which come from the Dataverse archive (unzip `lm_data_transcripts.zip` into `dataset/`).

Other results can be computed from the per-run tables, judge outputs and human data in the dataset (see [`dataset/README.md`](dataset/README.md)).

LM outputs are not deterministic, and several provider model versions have since been retired. The released transcripts are the record of the runs analyzed in the paper; model identifiers are listed in the paper's appendix.

## Adding scenarios

PieArena runs any two-party negotiation written in its scenario format and scores it automatically; the files in [`scenarios/`](scenarios/) show the format. The web application in `piearena/lm_human/` runs human–LM sessions on the same files.

We hope the community will use PieArena as an open, extensible platform: to build richer negotiation scenarios, and to collect human–LM negotiation data from more populations and with more model families.

## Citation

```bibtex
@inproceedings{zhu2026piearena,
  title     = {PieArena: Ranking and Profiling Language Agents in Realistic Negotiation Scenarios},
  author    = {Zhu, Chris and Cui, Sasha and Sanok Dufallo, Will and Jin, Runzhi and Xu, Zhen and Zhang, Linjun and Cain, Daylian M.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code and scenario files: Apache License 2.0 (`LICENSE`, `NOTICE`). Dataset: CC BY 4.0 (`dataset/LICENSE`).
