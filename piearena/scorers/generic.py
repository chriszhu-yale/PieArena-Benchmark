"""Scenarios with a machine-readable ``scoring`` block: deterministic deal verification and scoring.

The block lists each contract issue (the ``json_schema`` key it fills) with each side's payoff,
the BATNAs, optional interaction terms and hard constraints. ``kind`` is ``multi_issue`` (each side
also reports its own total as ``VALUE_FIELD``) or ``single_issue`` (price only). The verdict uses the
same conventions as the per-scenario scorers (verified_agreement, all_valid, per-side values and
self-report accuracy); ``piearena.scenarios.spec_for`` uses it for every unregistered scenario that
has the block.
"""
from __future__ import annotations

from typing import Any

SIDES = ("side1", "side2")
VALUE_FIELD = "total_value_of_deal_to_me"  # each side's self-reported value in every multi_issue contract


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def piecewise(points: list, x: float) -> float:
    """Linear interpolation through sorted (x, value) points, extrapolating with the end slopes."""
    pts = sorted((float(a), float(b)) for a, b in points)
    if len(pts) == 1:
        return pts[0][1]
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        i = next(k for k in range(1, len(pts)) if x <= pts[k][0])
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def issue_value(issue: dict, side: str, x) -> float:
    """One side's payoff from one issue at value ``x``."""
    spec = issue[side]
    if issue["type"] == "categorical":
        return float(spec[str(x)])
    if issue["type"] == "boolean":
        return float(spec[bool(x)])
    if "piecewise" in spec:
        return piecewise(spec["piecewise"], float(x))
    return float(spec.get("coef", 0)) * float(x) + float(spec.get("const", 0))


def side_value(scoring: dict, side: str, deal: dict) -> float:
    """Total value to ``side`` of a complete contract (keys as in the json_schema)."""
    v = float(scoring.get("constants", {}).get(side, 0))
    for issue in scoring["issues"]:
        v += issue_value(issue, side, deal[issue["key"]])
    for term in scoring.get("interactions", []):  # number issue whose rate depends on another issue's option
        by = deal[term["by"]]
        coef = term[side].get(by if isinstance(by, bool) else str(by), 0)
        v += float(coef) * float(deal[term["number"]])
    return v


def _holds(cond: dict, deal: dict) -> bool:
    x = deal.get(cond["key"])
    if "equals" in cond:
        return x == cond["equals"]
    if "in" in cond:
        return x in cond["in"]
    if "max" in cond:
        return _num(x) and x <= cond["max"]
    if "min" in cond:
        return _num(x) and x >= cond["min"]
    raise ValueError(f"condition needs equals/in/max/min: {cond}")


def constraint_ok(rule: dict, deal: dict) -> bool:
    """Whether a complete contract satisfies one hard constraint."""
    kind = rule["rule"]
    if kind == "max":
        return deal[rule["key"]] <= rule["value"]
    if kind == "min":
        return deal[rule["key"]] >= rule["value"]
    if kind == "requires":  # "then" must hold whenever "if" holds (always, without "if")
        if "if" in rule and not _holds(rule["if"], deal):
            return True
        return _holds(rule["then"], deal)
    if kind == "forbids":
        return not _holds(rule["if"], deal)
    raise ValueError(f"unknown constraint rule: {kind}")


def constraints_by_side(scoring: dict, deal: dict) -> dict:
    """{'side1': bool, 'side2': bool}: all of that side's constraints hold (``both`` counts for each)."""
    out = {s: True for s in SIDES}
    for rule in scoring.get("constraints", []):
        if not constraint_ok(rule, deal):
            for s in (SIDES if rule["side"] == "both" else (rule["side"],)):
                out[s] = False
    return out


def normalize(issue: dict, x):
    """The contract value as the scorer reads it, or None if it is malformed: a label outside the menu,
    a non-boolean, a non-number, or a number outside the issue's bounds. Numbers written as strings
    ("16,000", "$16000") and the strings "true"/"false" are read leniently."""
    if issue["type"] == "categorical":
        x = x.strip() if isinstance(x, str) else x
        return x if isinstance(x, str) and x in issue["side1"] else None
    if issue["type"] == "boolean":
        if isinstance(x, str) and x.strip().lower() in ("true", "false"):
            return x.strip().lower() == "true"
        return x if isinstance(x, bool) else None
    if isinstance(x, str):
        try:
            x = float(x.strip().lstrip("$").replace(",", ""))
        except ValueError:
            return None
    if not _num(x) or x != x:  # x != x: NaN
        return None
    lo, hi = issue.get("min"), issue.get("max")
    return x if (lo is None or x >= lo) and (hi is None or x <= hi) else None


def verify_and_score(scoring: dict, side1_final: dict, side2_final: dict, deal_reached: bool) -> dict:
    """Verdict for one run: agreement on every issue, validity under the hard constraints, true values,
    and, for multi_issue scenarios, the accuracy of each side's self-reported value (``VALUE_FIELD``).

    A contract that both sides filed identically but with a malformed value (see ``normalize``) is an
    agreement on an invalid contract: ``all_valid`` is false and ``malformed`` lists the keys."""
    keys = [i["key"] for i in scoring["issues"]]
    vf = None if scoring.get("kind") == "single_issue" else VALUE_FIELD
    tol = float(scoring.get("report_tolerance", 1))
    result: dict[str, Any] = {"deal_reached_token": bool(deal_reached), "verified_agreement": False,
                              "misunderstanding": False, "all_valid": None, "full_agreement": False,
                              "side1_constraint_ok": None, "side2_constraint_ok": None,
                              "side1_points": None, "side2_points": None, "total_points": None,
                              "side1_points_correct": None, "side2_points_correct": None,
                              "side1_points_error": None, "side2_points_error": None, "malformed": []}
    if not deal_reached:
        return result
    finals = (side1_final, side2_final)
    if any(v == "invalid" for f in finals for v in f.values()):
        result["all_valid"] = False
        return result
    agreed = {}
    for issue in scoring["issues"]:
        k = issue["key"]
        raw = (side1_final.get(k), side2_final.get(k))
        a, b = (normalize(issue, x) for x in raw)
        if a is not None and b is not None:
            same = a == b
        else:  # identical malformed entries still count as agreeing on that (invalid) term
            same = raw[0] is not None and raw[0] == raw[1]
            if same:
                result["malformed"].append(k)
        result[f"{k}_agreement"] = same
        agreed[k] = (a if a is not None else raw[0]) if same else None
        result[k] = agreed[k]
    result["full_agreement"] = all(result[f"{k}_agreement"] for k in keys)
    if not result["full_agreement"]:
        return result
    if result["malformed"]:
        result["all_valid"] = False
        return result
    ok = constraints_by_side(scoring, agreed)
    result["side1_constraint_ok"], result["side2_constraint_ok"] = ok["side1"], ok["side2"]
    result["all_valid"] = ok["side1"] and ok["side2"]
    vals = [side_value(scoring, s, agreed) for s in SIDES]
    result["side1_points"], result["side2_points"] = (round(v, 2) for v in vals)
    result["total_points"] = round(vals[0] + vals[1], 2)
    result["verified_agreement"] = bool(result["all_valid"])
    if vf:
        for s, f, v in zip(SIDES, finals, vals):
            claim = f.get(vf)
            if _num(claim):
                result[f"{s}_points_error"] = round(claim - v, 2)
                result[f"{s}_points_correct"] = abs(claim - v) <= tol
    return result
