# PieArena Dataset

Dataset for **PieArena: Ranking and Profiling Language Agents in Realistic Negotiation Scenarios** (NeurIPS 2026, Evaluations & Datasets track). Code: https://github.com/chriszhu-yale/PieArena-Benchmark.

The dataset contains LM–LM negotiation transcripts with per-run outcomes and judge outputs, the outcome table behind the paper's cross-play leaderboards, and per-session human negotiation data. With the code, these files support reproducing the paper's main results.

## Layout

```
dataset/
├── README.md
├── LICENSE                                  # CC BY 4.0
├── croissant_metadata.json                  # Croissant metadata (core and Responsible AI fields)
├── lm_data/
│   ├── paper_crossplay_outcomes.csv         # outcomes behind the paper's cross-play leaderboards
│   └── transcripts/<play_type>/<mode>/<scenario>/<side1>_vs_<side2>/
│       ├── run_<i>.json                     # one negotiation
│       ├── pair_runs_table.csv              # one row per run
│       ├── lies_report.jsonl                # lie-judge output, one record per run
│       └── reputation_score.csv             # reputation-judge scores, one row per run (cross-play)
└── human_data/                              # per-session human negotiation data
```

On Dataverse, `lm_data/transcripts/` is distributed as `lm_data_transcripts.zip`; unzip it in this folder to obtain the layout above.

- `play_type`: `cross_play` (two different models) or `mirror_play` (a model against itself).
- `mode`: `base_mode` (no scaffolding), `pro_mode` (state tracking and planning), `memory_only_mode` (state tracking only, used in the harness ablation).
- `scenario`: `Hire_Talent`, `IP_License` and `Service_Quote` in cross-play, `SnyderMed` in mirror-play. The paper's appendix describes each scenario.

## LM–LM data

### `run_<i>.json`

The transcript (`transcript`), the models and role labels of both sides, the contract each side reported (`side1_offer`, `side2_offer`), the deterministic verdict (`deal_summary`), who opened (`who_goes_first`), the round of agreement (`turns_to_deal`), a parsing trace (`deal_trace`), and the harness configuration (`memory_agent`, `plan_agent`). Runs of a pairing are numbered consecutively within its folder.

### `pair_runs_table.csv`

One row per `run_<i>.json` (matched by `run_number`), computed from the run file:

| Column(s) | Meaning |
|---|---|
| `who_first` | side that opened every round (`side1`/`side2`) |
| `verified_agreement`, `misunderstanding`, `all_valid` | deal verified by both sides and the scorer / confirmation failed / terms satisfy the hard constraints |
| `side1_value`, `side2_value`, `total_pie` | surplus of each side over its BATNA, and their sum |
| `p_i`, `p_j` | pie shares of side 1 and side 2 |
| `side*_value_correct`, `side*_signed_error` | whether each side computed its own payoff correctly (self-reported minus true) |
| `side*_above_batna` | deal at or above the side's BATNA |
| `side*_used_json_fixer` | the contract needed LLM repair to parse |
| `side*_lie_count`, `liar_judge_fallback` | lies per side from the lie judge |
| `paper_analysis` | run is used in the analyses of the paper's main results |

`paper_analysis` marks the runs used in the analyses of the paper's main results.

### `paper_crossplay_outcomes.csv`

The outcomes of the balanced cross-play design, from which the paper's leaderboards are computed (`python scripts/reproduce_leaderboard.py` in the code repository). Rows refer to pair folders and run numbers of this dataset. `transcript_consistent` marks whether the released transcript of a run is the one from which its recorded outcome was computed.

### Judge outputs

- `lies_report.jsonl`: per run, the lies attributed to each side (`round`, `quote`, `reason` with category A/B/C), counts, and the judge model.
- `reputation_score.csv`: per run, 0–100 scores of each side (`model1_*` for side 1, `model2_*` for side 2) on integrity, reliability, reciprocity, collaboration and conduct, their average, and the judge model.

## Human data (`human_data/`)

Per-session data without personal identifiers. `session_id` is a random identifier. In the two human–LM files, `balanced_subset` is 1 for the sessions in the double-balanced subsets used for the paper's primary human–LM analyses (equal numbers of sessions per agent mode or student role and first speaker), and 0 otherwise.

| File | Description |
|---|---|
| `main_street_sessions.csv` | human–LM Main Street sessions: roles, agent configuration, first speaker, outcome, and each side's points and share |
| `hire_talent_sessions.csv` | human–LM Hire Talent sessions, with the same columns except `scenario_name` |
| `snydermed_human_human.csv` | human–human SnyderMed outcomes, one row per agreed contract: the contract terms (total salary over the contract, start date, division, city, housing bonus, signing bonus), each side's surplus over its BATNA (`side1_value` for the recruiter, `side2_value` for the candidate), `total_pie`, `normalized_total_pie` (total pie divided by the largest achievable total pie) and `all_valid` |

## Intended use, limitations, and known biases

**Intended use.** Evaluation research on language agents: comparing models on negotiation outcomes, estimating skill rankings, and profiling behavior (deception, output validity, numerical accuracy, deal rate, perceived reputation).

**Limitations.**
- Two-party negotiations only.
- Real negotiations may involve payoffs that are harder to quantify than the deterministic payoff functions used here.
- The human baseline consists of graduate business-school students.
- Lie rate and perceived reputation come from LLM judges, so residual judge bias may remain.

**Known biases.** Scenarios are in English with North American business contexts. The model pool reflects models available through the OpenRouter API at the time of data collection.

**Misuse.** Do not use the dataset to train or tune deceptive or adversarial negotiation agents deployed against people who do not know they face an agent. The behavioral metrics are research instruments, not certifications of trustworthiness.

## Ethics and human data

The human-subjects study was reviewed by the authors' institutional review board and determined to be exempt, and participants gave informed consent. The human data files contain no personally identifying information.

## License

CC BY 4.0 (see `LICENSE`).
