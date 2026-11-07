"""Scenario loading and the per-scenario conventions used by the runner.

A scenario YAML defines ``json_schema`` (the contract emitted on agreement) and
``side1``/``side2`` blocks (``label``, ``batna``, ``system_prompt``,
``context_prompt``, ``initial_offer_prompt``), and optionally a ``scoring``
block with machine-readable payoffs (see ``scorers/generic.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import yaml

from .scorers.generic import VALUE_FIELD, verify_and_score

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "dataset" / "scenarios"


@dataclass(frozen=True)
class ScenarioSpec:
    value_field: str | None = VALUE_FIELD                   # self-reported value key (None: not asked)
    scorer: Callable | None = None                          # deterministic verifier/scorer
    value_keys: tuple[str, str] | None = None               # verdict keys with each side's points
    accuracy_keys: tuple[str, str, str, str] | None = None  # (correct1, correct2, error1, error2)
    price_based: bool = False                               # single-issue price: values via BATNAs


SPECS: dict = {}  # no scenario-specific scorers are released; see scorers/generic.py


def spec_for(name: str, config: dict | None = None) -> ScenarioSpec:
    """Registered scenarios keep their spec; others are scored from their ``scoring`` block if they have
    one, and otherwise are run but not scored (deals are never verified)."""
    if name in SPECS:
        return SPECS[name]
    scoring = (config or {}).get("scoring")
    if not scoring:
        return ScenarioSpec()
    single = scoring.get("kind") == "single_issue"  # price only: values come from the price and the BATNAs
    return ScenarioSpec(None if single else VALUE_FIELD, partial(verify_and_score, scoring),
                        ("side1_points", "side2_points"),
                        ("side1_points_correct", "side2_points_correct", "side1_points_error", "side2_points_error"),
                        price_based=single)


def load_scenario(name_or_path: str, rounds: int | None = None) -> tuple[str, dict]:
    """Return ``(scenario_name, config)``; a bare name is resolved against ``dataset/scenarios/``."""
    path = Path(name_or_path)
    if "/" not in name_or_path and not name_or_path.endswith(".yaml"):
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)
    if rounds is not None:
        config["num_rounds"] = rounds
    return path.stem, config


def side_values(verdict: dict, spec: ScenarioSpec, batna1, batna2) -> tuple:
    """Each side's surplus over its BATNA, or ``(None, None)`` if the run is not scorable."""
    if not verdict:
        return None, None
    if spec.price_based:  # buyer is side1, seller is side2
        price = verdict.get("final_price")
        if isinstance(price, (int, float)) and batna1 is not None and batna2 is not None:
            return batna1 - price, price - batna2
        return None, None
    if spec.value_keys is None:
        return None, None
    v1, v2 = (verdict.get(k) for k in spec.value_keys)
    if v1 is None or v2 is None:
        return None, None
    return v1 - (0 if batna1 is None else batna1), v2 - (0 if batna2 is None else batna2)
