"""Negotiation episodes between two LMs, and repeated runs for one ordered model pairing."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import jsonschema
import pandas as pd

from . import prompts as P
from .liar_judge_core import LIAR_JUDGE_DEFAULT_MODEL, liar_judge_run
from .openai_wrapper import OpenAIWrapper
from .scenarios import VALUE_FIELD, spec_for, side_values

JSON_FIXER_MODEL = "openai/gpt-5"
PAIR_RUNS_TABLE = "pair_runs_table.csv"
LIES_REPORT = "lies_report.jsonl"
_SELF_REPORTED = {VALUE_FIELD}


def _other(side: str) -> str:
    return "side2" if side == "side1" else "side1"


def _debug(enabled: bool, title: str, *blocks: str) -> None:
    if enabled:
        print(f"\n================ {title} ================")
        for b in blocks:
            print(b)


def pairing_dirname(side1_model: str, side2_model: str) -> str:
    return f"{side1_model.replace('/', '-')}_vs_{side2_model.replace('/', '-')}"


# -----------------------------------------------------------------------------
# Structured-deal parsing (with an LLM JSON fixer as fallback)
# -----------------------------------------------------------------------------
def _fixer_schema(schema: dict) -> dict:
    """Copy of the contract schema in which every field may also be the string 'invalid'."""
    fixed = copy.deepcopy(schema)
    for name, field in list(fixed.get("properties", {}).items()):
        if not isinstance(field, dict):
            continue
        if field.get("type") == "string" and "enum" in field:
            if "invalid" not in field["enum"]:
                field["enum"].append("invalid")
        elif field.get("type") in ("number", "boolean", "string"):
            fixed["properties"][name] = {"anyOf": [copy.deepcopy(field), {"type": "string", "enum": ["invalid"]}]}
    return fixed


def parse_deal_json(text: str, schema: dict, fixer: OpenAIWrapper, verbose: bool = False,
                    prompt_verbosity: bool = False) -> tuple[dict, dict]:
    """Parse a deal JSON; on failure ask the fixer model. Returns ``(deal, meta)``."""
    meta = {"used_fixer": False, "schema_strict_ok_after_fix": False, "contains_invalid_placeholder": False}
    try:
        data = json.loads(text)
        jsonschema.validate(instance=data, schema=schema)
        meta["schema_strict_ok_after_fix"] = True
        meta["contains_invalid_placeholder"] = any(v == "invalid" for v in data.values() if isinstance(v, str))
        return data, meta
    except Exception:  # noqa: BLE001
        pass

    meta["used_fixer"] = True
    fixer_schema = _fixer_schema(schema)
    system = P.json_fixer_system(json.dumps(fixer_schema))
    _debug(prompt_verbosity, "JSON FIXER PROMPT", "-- system --", system, "-- user --", text)
    fixed = fixer.chat([{"role": "system", "content": system}, {"role": "user", "content": text}],
                       response_format={"schema": fixer_schema, "strict": True})["content"].strip()
    _debug(prompt_verbosity, "JSON FIXER OUTPUT", fixed)
    try:
        data = json.loads(fixed)
    except json.JSONDecodeError:
        if verbose:
            print("❌ JSON fixer output is not valid JSON:\n", fixed)
        meta["contains_invalid_placeholder"] = True
        return {k: "invalid" for k in schema.get("properties", {})}, meta
    meta["contains_invalid_placeholder"] = any(v == "invalid" for v in data.values() if isinstance(v, str))
    try:
        jsonschema.validate(instance=data, schema=schema)
        meta["schema_strict_ok_after_fix"] = True
    except jsonschema.exceptions.ValidationError:
        pass
    return data, meta


# -----------------------------------------------------------------------------
# One negotiation episode
# -----------------------------------------------------------------------------
class _Side:
    def __init__(self, key: str, cfg: dict, model: str):
        self.key, self.model = key, model
        self.label, self.batna = cfg["label"], cfg.get("batna")
        self.system, self.context = cfg["system_prompt"], cfg["context_prompt"]
        self.opening = cfg["initial_offer_prompt"]
        self.agent = OpenAIWrapper(model=model, label=self.label)
        self.state = self.plan = self.last_plan = ""
        self._memory_agent = self._plan_agent = None

    def memory_agent(self) -> OpenAIWrapper:
        self._memory_agent = self._memory_agent or OpenAIWrapper(self.model, f"Memory-{self.label}")
        return self._memory_agent

    def plan_agent(self) -> OpenAIWrapper:
        self._plan_agent = self._plan_agent or OpenAIWrapper(self.model, f"Planner-{self.label}")
        return self._plan_agent


def run_one_experiment(config: dict, scenario_name: str, side1_model: str, side2_model: str,
                       first_turn: str = "side1", use_memory: bool = True, use_plan: bool = True,
                       verbose: bool = False, prompt_verbosity: bool = False) -> dict:
    """Run one fixed-horizon negotiation and return the transcript, offers, and verdict.

    Each round both sides speak once (``first_turn`` opens every round). A side
    accepts by emitting ``$DEAL_REACHED$`` plus a JSON contract; the other side
    then confirms (or reports ``$DEAL_MISUNDERSTANDING$``) and the deal is
    scored deterministically. ``use_memory``/``use_plan`` switch on the state
    tracking and planning modules of the agentic harness ("pro" mode = both).
    """
    spec = spec_for(scenario_name, config)
    n_rounds = config["num_rounds"]
    schema = json.loads(config["json_schema"])
    schema_text = "\n" + json.dumps(schema, indent=2)
    turn_template = P.turn_template(schema_text, spec.value_field)
    sides = {"side1": _Side("side1", config["side1"], side1_model),
             "side2": _Side("side2", config["side2"], side2_model)}
    order = [first_turn, _other(first_turn)]
    trace = {"first_deal_side": None, **{k: dict.fromkeys(
        ("deal_reached_token", "used_json_fixer", "schema_strict_ok_after_fix", "contains_invalid_placeholder",
         "constraints_ok", "misunderstanding_sent", "confirmation_consistent_with_first")) for k in sides}}
    transcript: list[str] = []
    deal = None  # (signaling side, round, message)
    if verbose:
        print(f"\n🟢 Starting negotiation [{side1_model} vs {side2_model}]\n")

    for round_id in range(1, n_rounds + 1):
        for position, key in enumerate(order):
            side = sides[key]
            base = side.opening if (round_id == 1 and position == 0) else turn_template
            history = "\n\n".join(transcript)
            prompt = P.unescape_braces(f"{side.context}\n\n{base}".format(
                history=P.escape_braces(history), current_round=round_id, total_rounds=n_rounds,
                rounds_remaining=n_rounds - round_id,
                turn_position="going first" if position == 0 else "going second",
                turn_action="start" if position == 0 else "finish"))
            active = bool(transcript)  # the harness modules have nothing to work on before the first message
            window = P.last_rounds(transcript, k_rounds=2)
            if use_memory and active:
                user = P.state_tracker_user(side.label, round_id, n_rounds, side.system, side.context,
                                            side.state, window, 2)
                side.state = (side.memory_agent().chat([{"role": "system", "content": P.STATE_TRACKER_SYSTEM},
                                                        {"role": "user", "content": user}])["content"] or "").strip()
                _debug(prompt_verbosity, f"STATE UPDATE for {side.label}", user, "-- new state --", side.state)
            else:
                side.state = ""
            if use_plan and active:
                user = P.planner_user(side.label, round_id, n_rounds, side.system, side.context,
                                      side.state, side.last_plan)
                side.plan = side.plan_agent().chat([{"role": "system", "content": P.PLANNER_SYSTEM},
                                                    {"role": "user", "content": user}])["content"].strip()
                side.last_plan = side.plan
                _debug(prompt_verbosity, f"STRATEGY for {side.label}", user, "-- new strategy --", side.plan)
            else:
                side.plan = ""
            full_prompt = P.with_state_and_plan(prompt, side.state, side.plan, use_memory, use_plan)
            _debug(prompt_verbosity, f"PROMPT to {side.label}", "-- system --", side.system, "-- user --", full_prompt)
            reply = side.agent.chat([{"role": "system", "content": side.system},
                                     {"role": "user", "content": full_prompt}])["content"]
            transcript.append(f"{side.label} (round {round_id}):\n{reply}")
            if verbose:
                print(f"{'🟩' if key == 'side1' else '🟦'} {side.label} ({side.model}):\n{reply}\n")
            if P.is_deal_token(reply):
                trace["first_deal_side"] = key
                trace[key]["deal_reached_token"] = True
                deal = (key, round_id, reply)
                break
        if deal:
            break

    history = "\n\n".join(transcript)
    if deal is None:
        finals = {"side1": {"deal_reached_token": False}, "side2": {"deal_reached_token": False}}
        verdict = {"deal_reached_token": False, "deal_failed_token": False, "verified_agreement": False,
                   "misunderstanding": None, "all_valid": None, "timed_out": True, "failed_at_round": n_rounds}
    else:
        signaler, _, deal_msg = deal
        other = sides[_other(signaler)]
        body = P.confirmation_prompt(history, schema_text, spec.value_field)
        conf_prompt = body if spec.value_field is None else f"{other.context}\n\n{body}"
        _debug(prompt_verbosity, f"PROMPT to {other.label}", "-- system --", other.system, "-- user --", conf_prompt)
        confirmation = other.agent.chat([{"role": "system", "content": other.system},
                                         {"role": "user", "content": conf_prompt}])["content"]
        if confirmation.startswith(P.MISUNDERSTANDING_TOKEN):
            trace[other.key]["misunderstanding_sent"] = True
            finals = {k: {"deal_reached_token": k == signaler} for k in sides}
            verdict = {"deal_reached_token": True, "verified_agreement": False,
                       "misunderstanding": True, "all_valid": None}
        else:
            trace[other.key]["deal_reached_token"] = True
            trace[other.key]["misunderstanding_sent"] = False
            fixer = OpenAIWrapper(model=JSON_FIXER_MODEL, label="JSONFixer", allow_response_format=True)
            finals = {}
            for key, text in ((signaler, deal_msg), (other.key, confirmation)):
                start = text.find("{")
                finals[key], meta = parse_deal_json(text[start:] if start != -1 else text, schema, fixer,
                                                    verbose, prompt_verbosity)
                trace[key]["used_json_fixer"] = meta["used_fixer"]
                trace[key]["schema_strict_ok_after_fix"] = meta["schema_strict_ok_after_fix"]
                trace[key]["contains_invalid_placeholder"] = meta["contains_invalid_placeholder"]
            first, second = ({k: v for k, v in finals[k].items() if k not in _SELF_REPORTED}
                             for k in (signaler, other.key))
            trace[other.key]["confirmation_consistent_with_first"] = first == second
            finals = {k: finals[k] for k in ("side1", "side2")}
            for f in finals.values():
                f["deal_reached_token"] = True
            if spec.scorer is not None:
                verdict = spec.scorer(finals["side1"], finals["side2"], True)
            else:
                verdict = {"deal_reached_token": True, "verified_agreement": False, "misunderstanding": False,
                           "all_valid": None, "timed_out": False, "deal_failed_token": False,
                           "failed_at_round": None}

    turns_to_deal = turns_to_fail = None
    if deal is not None:
        turns_to_deal = round(deal[1] + (0.1 if deal[0] == first_turn else 0.2), 1)
    else:
        turns_to_fail = round(n_rounds + 0.2, 1)
    return {
        "transcript": transcript,
        "side1_label": sides["side1"].label,
        "side1_model": side1_model,
        "side1_offer": finals["side1"],
        "side2_label": sides["side2"].label,
        "side2_model": side2_model,
        "side2_offer": finals["side2"],
        "who_goes_first": sides[first_turn].label,
        "deal_summary": verdict,
        "turns_to_deal": turns_to_deal,
        "turns_to_fail": turns_to_fail,
        "deal_trace": trace,
        "memory_agent": bool(use_memory),
        "plan_agent": bool(use_plan),
    }


# -----------------------------------------------------------------------------
# Repeated runs for one ordered pairing
# -----------------------------------------------------------------------------
def pair_table_row(result: dict, scenario_name: str, config: dict, run_number: int, first_turn: str) -> dict:
    """One row of ``pair_runs_table.csv`` (everything except the lie-judge columns)."""
    spec = spec_for(scenario_name, config)
    verdict = result.get("deal_summary", {}) or {}
    trace = result.get("deal_trace", {}) or {}
    t1, t2 = trace.get("side1") or {}, trace.get("side2") or {}
    v1, v2 = side_values(verdict, spec, config["side1"].get("batna"), config["side2"].get("batna"))
    total = p_i = p_j = None
    if v1 is not None and v2 is not None:
        total = v1 + v2
        if isinstance(total, (int, float)) and total != 0:
            p_i, p_j = v1 / total, v2 / total
    acc = [verdict.get(k) for k in spec.accuracy_keys] if spec.accuracy_keys else [None] * 4
    s1, s2 = result["side1_model"], result["side2_model"]
    row = {
        "scenario": scenario_name,
        "pairing": f"{s1}_vs_{s2}",
        "run_number": run_number,
        "who_first": first_turn,
        "side1_label": config["side1"]["label"],
        "side1_model": s1,
        "side2_label": config["side2"]["label"],
        "side2_model": s2,
        "verified_agreement": verdict.get("verified_agreement", False),
        "misunderstanding": verdict.get("misunderstanding", None),
        "first_deal_side": trace.get("first_deal_side"),
        "both_reached": bool(t1.get("deal_reached_token", False) and t2.get("deal_reached_token", False)),
        "turns_to_deal": result.get("turns_to_deal"),
        "side1_value": v1,
        "side2_value": v2,
        "total_pie": total,
        "p_i": p_i,
        "p_j": p_j,
        "side1_used_json_fixer": t1.get("used_json_fixer"),
        "side2_used_json_fixer": t2.get("used_json_fixer"),
        "side1_value_correct": acc[0],
        "side2_value_correct": acc[1],
        "side1_signed_error": acc[2],
        "side2_signed_error": acc[3],
        "all_valid": verdict.get("all_valid"),
        "side1_above_batna": None if v1 is None else v1 >= 0,
        "side2_above_batna": None if v2 is None else v2 >= 0,
        "memory_agent": result.get("memory_agent"),
        "plan_agent": result.get("plan_agent"),
    }
    return row


def _judged_runs(pair_dir: Path) -> dict[int, dict]:
    """Lie-judge records already present for this pairing, keyed by run number."""
    path = pair_dir / LIES_REPORT
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["run_number"]: r for r in records}


def run_pair_n_runs(side1_model: str, side2_model: str, scenario_config: dict, output_root: Path,
                    n_runs: int, scenario_name: str, use_memory: bool = True, use_plan: bool = True,
                    judge_model: str = LIAR_JUDGE_DEFAULT_MODEL, verbose: bool = False,
                    prompt_verbosity: bool = False) -> pd.DataFrame:
    """Run one ordered pairing ``n_runs`` times; write ``run_<i>.json``, ``lies_report.jsonl``
    and ``pair_runs_table.csv`` into ``output_root/<side1>_vs_<side2>/``.

    Side 1 opens the first ``ceil(n_runs/2)`` runs and side 2 the rest. Existing
    ``run_<i>.json`` files are reused, so re-running a command resumes it.
    """
    pair_dir = Path(output_root) / pairing_dirname(side1_model, side2_model)
    pair_dir.mkdir(parents=True, exist_ok=True)
    judged = _judged_runs(pair_dir)
    rows = []
    for run in range(1, n_runs + 1):
        first_turn = "side1" if run <= math.ceil(n_runs / 2) else "side2"
        path = pair_dir / f"run_{run}.json"
        try:
            result, cached = json.loads(path.read_text()), True
        except (OSError, ValueError):
            result, cached = run_one_experiment(scenario_config, scenario_name, side1_model, side2_model,
                                                first_turn, use_memory, use_plan, verbose, prompt_verbosity), False
            path.write_text(json.dumps(result, indent=2))
        row = pair_table_row(result, scenario_name, scenario_config, run, first_turn)
        lies = (judged.get(run) if cached else None) or liar_judge_run(
            run_result=result, scenario_config=scenario_config, judge_model=judge_model, save_dir=pair_dir,
            pairing=f"{side1_model}_vs_{side2_model}", run_number=run, verbose=verbose)
        row.update(side1_lie_count=lies["side1_lie_count"], side2_lie_count=lies["side2_lie_count"],
                   liar_judge_fallback=lies["judge_fallback"])
        rows.append(row)
        print(f"run {run}/{n_runs}: verified={row['verified_agreement']} p_i={row['p_i']} "
              f"lies={row['side1_lie_count']}/{row['side2_lie_count']}")
    df = pd.DataFrame(rows)
    df.to_csv(pair_dir / PAIR_RUNS_TABLE, index=False)
    return df
