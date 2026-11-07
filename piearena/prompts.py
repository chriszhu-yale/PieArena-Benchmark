"""Prompt templates used by the negotiation harness.

The wording (including its quirks) is exactly what produced the released
transcripts; do not edit it if you want runs comparable to the paper.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Per-turn instructions. Placeholders in {braces} are filled at every turn with
# str.format, so literal braces (the JSON schema, the transcript) are escaped.
# ---------------------------------------------------------------------------
_TURN_HEAD = (
    "CURRENT ROUND INFORMATION:\n"
    "It is now round {current_round}/{total_rounds}. You are {turn_position} this round, so it is your turn to {turn_action} the round. "
    "You have {rounds_remaining} rounds remaining after this one.\n\n"
    "<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{history}\n\n"
    "<END NEGOTIATION TRANSCRIPT>\n\n"
    "OUTPUT INSTRUCTIONS:\n"
    "1. Reminder: you can either continue the negotiation or accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output. If you do not reach a deal by the end of the last round, you get your BATNA.\n\n"
    "2. OUTPUT OPTIONS (YOU MUST CHOOSE ONE OR THE OTHER):\n"
    "a) Continue the negotiation by outputting your negotiation message to {turn_action} round {current_round}\n"
    "b) Accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output on its own line (DO NOT put any other text before it), immediately followed by a newline, and then the deal terms you are accepting in the following JSON format. "
)
_TURN_TAIL = (
    "Do not put anything in your output besides the token '$DEAL_REACHED$' and the correctly formatted JSON (Do NOT output the schema. Output only a filled JSON instance matching the schema). Do not add any commentary, explanations, or markdown.\n"
    "JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION B):\n"
)


def escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def unescape_braces(s: str) -> str:
    return s.replace("{{", "{").replace("}}", "}")


def turn_template(schema_text: str, value_field: str | None) -> str:
    """Instructions appended to a side's context on every turn except the opening one."""
    fill = "" if value_field is None else (
        f"Fill in '{value_field}' by calculating your value using your private scoring rules described in your given prompts. ")
    return _TURN_HEAD + fill + _TURN_TAIL + escape_braces(schema_text)


def confirmation_prompt(history: str, schema_text: str, value_field: str | None) -> str:
    """Asked of the side that did NOT emit $DEAL_REACHED$."""
    if value_field is None:
        return (
            "DEAL CONFIRMATION:\n"
            "The other side indicated that they accept the most recent terms offered by you and specified the deal terms that they are accepting in their most recent message. "
            "Look at the complete negotiation transcript below to confirm that the terms specified by the other side in their last message are faithful to the proposal you made.\n\n"
            "COMPLETE NEGOTIATION TRANSCRIPT:\n"
            "<BEGIN TRANSCRIPT>\n"
            f"{history}\n"
            "<END TRANSCRIPT>\n\n"
            "OUTPUT INSTRUCTIONS: CHOOSE ONE OF THE FOLLOWING TWO OPTIONS\n"
            "OPTION 1: IF the terms of the deal specified by the other side are INDEED CORRECT then output the agreed upon deal in the following JSON format.\n"
            f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION 1):\n{schema_text}\n"
            "OPTION 2: IF the terms of the deal specified by the other side are NOT correct then output ONLY the token '$DEAL_MISUNDERSTANDING$'. Do not bother to output any other text or fill in the JSON.\n\n"
        )
    return (
        "DEAL CONFIRMATION:\n"
        "The other side indicated that they accept the most recent terms offered by you and specified the deal terms that they are accepting in their most recent message. "
        "Look at the complete negotiation transcript below to confirm that the terms specified by the other side in their last message are faithful to the latest proposal you made.\n\n"
        "COMPLETE NEGOTIATION TRANSCRIPT:\n"
        "<BEGIN TRANSCRIPT>\n"
        f"{history}\n"
        "<END TRANSCRIPT>\n\n"
        "OUTPUT INSTRUCTIONS: CHOOSE ONE OF THE FOLLOWING TWO OPTIONS\n"
        f"OPTION 1: IF the terms of the deal specified by the other side are INDEED CORRECT then output the agreed upon deal in the following JSON format. Fill in '{value_field}' by calculating your own value using your private scoring rules described in your given prompts.\n"
        f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION 1):\n{schema_text}\n"
        "OPTION 2: IF the terms of the deal specified by the other side are NOT correct then output ONLY the token '$DEAL_MISUNDERSTANDING$'. Do not bother to output any other text or fill in the JSON.\n"
        "IMPORTANT: In both options, completely ignore the other side's 'total_value_of_deal_to_me'. You must never read, use, or compare it when producing your output.\n\n "
    )


# ---------------------------------------------------------------------------
# Deal-token detection
# ---------------------------------------------------------------------------
_DEAL_TOKEN_RE = re.compile(
    r"""^\s*(?:```[a-zA-Z]*\s*)?[\$]+DEAL_REACHED[\$]+(?:\s*```)?(?:\s|$)""",
    re.IGNORECASE | re.VERBOSE,
)
MISUNDERSTANDING_TOKEN = "$DEAL_MISUNDERSTANDING$"


def is_deal_token(msg: str) -> bool:
    return bool(_DEAL_TOKEN_RE.match(msg or ""))


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------
def json_fixer_system(fixer_schema_raw: str) -> str:
    return (
        "You are an expert assistant at repairing JSON. You are given text that was *intended* to be valid JSON but isn't. "
        "Rewrite it into syntactically correct JSON that preserves all intended keys, values and data types. "
        "Strict rules:\n"
        "1. Respond with JSON ONLY – no markdown, explanations or extra characters.\n"
        "2. Do NOT add or remove required fields. Ignore any schema/metainfo keys such as 'type', 'properties', 'required', 'additionalProperties' if they appear in the input.\n"
        "3. Ensure numbers are plain digits – no commas, underscores, or other separators.\n"
        "4. Prefer concrete values present in the input. If a value is clearly provided (even inside a schema-like 'properties' block), extract it and place it at the top level under the correct field name.\n"
        "5. If a required field is truly missing or the value is irrecoverably invalid, set that field to the string 'invalid'.\n"
        "6. IMPORTANT: If the input looks like a JSON Schema (e.g., has top-level 'type' and 'properties'), DO NOT output a schema. Instead, construct a single JSON INSTANCE by taking values for each required field from inside 'properties' (or from nearby text). Example: input with \"properties\": {\"salary_per_month\": 13750, ...} => output: {\"salary_per_month\": 13750, ...}.\n"
        "7. Your output MUST conform to the following JSON schema (use it to validate and repair the JSON):\n"
        f"{fixer_schema_raw}\n"
    )


# ---------------------------------------------------------------------------
# Shared-intentionality harness: state tracking (memory) and planning modules
# ---------------------------------------------------------------------------
STATE_TRACKER_SYSTEM = (
    "You are a state tracking module for an AI negotiator. "
    "Produce a CONCISE, ACTIONABLE negotiation state that survives limited transcript windows.\n\n"
    "REQUIRED SECTIONS (keep each section ≤2 lines):\n"
    "OFFERS: [Them: <explicit numeric/boolean terms>; Us: <explicit numeric/boolean terms>; use old→new if changed]\n"
    "PATTERNS: [Concession/firmness behaviors observed across rounds]\n"
    "PRIORITIES: [Their priorities/red lines/BATNA if explicit; mark uncertain with 'Hypothesis:']\n"
    "OPPORTUNITIES: [Cross-issue trades, value-creation levers, give↔get pairs]\n"
    "INSIGHTS: [Durable facts and round-to-round deductions you will carry forward]\n\n"
    "HARD RULES:\n"
    "1) PERSISTENCE — Keep long-lived facts (BATNA, red lines, stable priorities, accepted constraints) even if they don’t reappear in the latest window. Do NOT drop them unless contradicted.\n"
    "2) OFFERS COVERAGE — Always include the opponent’s latest explicit terms (Them: …) AND our latest standing terms (Us: …). If no explicit numeric/boolean terms exist for a side, write '(none yet)'. Normalize numbers (e.g., '$150k', '12 months').\n"
    "3) FACT vs HYPOTHESIS — Only mark as FACT if explicitly stated; otherwise prefix with 'Hypothesis:'.\n"
    "4) CANONICALIZATION — Strip prose; compress to atomic items, use consistent units/symbols; show deltas as old→new.\n"
    "5) DEDUP & DELTA — Remove stale/duplicates; keep only the latest state; emphasize what materially changed.\n"
    "6) NO VERBATIM QUOTES — Summarize; do not copy transcript sentences.\n"
    "7) BINDING — When recording concessions, record the coupled give↔get condition if stated or implied (e.g., 'signing bonus −2k ↔ earlier start').\n"
    "8) SAFETY NET — If transcript is thin/ambiguous, prefer carrying forward last known stable state over inventing new facts.\n"
    "\n"
    "Output exactly these five sections, in this order. Keep total length tight."
    "Keep each section to 1-2 lines max. Focus on critical information only.\n"
)


def state_tracker_user(label: str, round_id: int, total_rounds: int, rules: str, facts: str,
                       prev_state: str, window: str, k_rounds: int) -> str:
    return (
        f"=== Identity ===\n"
        f"Role: State tracker of {label}\n"
        f"Round: {round_id}/{total_rounds}\n\n"
        f"=== Scenario Objective & Rules ===\n{rules}\n\n"
        f"=== Scenario Facts & Value model ===\n{facts}\n\n"
        f"=== Previous State ===\n{prev_state or '(empty)'}\n\n"
        f"=== Recent Transcript Window (last {k_rounds} rounds) ===\n{window}\n\n"
        f"=== Output State ===\n"
        "Update the negotiation state now, following HARD RULES."
    )


PLANNER_SYSTEM = (
    "You are a strategic planning module for an AI negotiator. "
    "Generate a SMART and ASSERTIVE plan for THIS round based on current state and context.\n\n"
    "PLANNING RULES:\n"
    "- READ STATE — Use OFFERS, PATTERNS, PRIORITIES, OPPORTUNITIES, and INSIGHTS from the State exactly as the source of truth.\n"
    "- ROUND GOAL — Set a concrete, realistic target for THIS round.\n"
    "- TRADE-ACROSS-ISSUES — Do not concede on high-priority terms without gaining on other issues.\n"
    "- ADAPT & NON-REPETITION — If last round's plan didn't move the opponent, change tactics.\n"
    "- CLOSING OPTIONS — If the opponent's current terms already satisfy your core priorities, stay above BATNA, "
    "and do not cross your red lines, you may consider proposing acceptance; otherwise push toward your round target.\n"
    "- ZOPA AWARENESS — Only invoke ZOPA framing if it improves your outcome.\n"
    "- BREVITY — Be crisp and operational: no fluff, no transcript restatement.\n\n"
    "OUTPUT SKELETON (≤10 lines, short bullets):\n"
    "- ROUND GOAL: <one clear numeric/boolean target for this round (key term or concise combined terms), or 'consider accepting current offer'>\n"
    "- KEY LEVERS: <2–3 trade levers from PRIORITIES/OPPORTUNITIES; if accepting, note 'already satisfied'>\n"
    "- TACTICS: <2–3 concrete moves this round; if accepting, write 'consider proposing acceptance'>\n"
    "- OFFER SCAFFOLD: <one main package based on OFFERS; if accepting, refer to opponent’s last offer>\n"
    "- RISK & RESPONSES: <if-then counterplans; if accepting, '(not applicable)'>\n"
)


def planner_user(label: str, round_id: int, total_rounds: int, rules: str, facts: str,
                 state: str, last_plan: str) -> str:
    return (
        f"=== Identity ===\n"
        f"Role: Strategy module of {label}\n"
        f"Round: {round_id}/{total_rounds}\n\n"
        f"=== Scenario Rules & Objective ===\n{rules}\n\n"
        f"=== Scenario Facts & Value model ===\n{facts}\n\n"
        f"=== Current State ===\n{state or '(empty)'}\n\n"
        f"=== Previous Strategy ===\n{last_plan or 'N/A'}\n\n"
        f"=== Output Strategy ===\n"
    )


def with_state_and_plan(prompt: str, state: str, plan: str, use_memory: bool, use_plan: bool) -> str:
    """Prepend the harness outputs (when enabled) to a side's turn prompt."""
    parts = []
    if use_memory and state and state.strip():
        parts.append(f"--- NEGOTIATION STATE TRACKING ---\n{state}")
    if use_plan and plan and plan.strip():
        parts.append(f"--- STRATEGY FOR THIS ROUND ---\n{plan}")
    parts.append(f"--- MAIN PROMPT ---\n{prompt}")
    return "\n\n".join(parts).strip()


_ROUND_HEADER_RE = re.compile(r"^.+? \(round (\d+)\):\n", re.S)


def last_rounds(transcript: list[str], k_rounds: int = 2) -> str:
    """Messages from the most recent ``k_rounds`` rounds, in original order."""
    items = []
    for i, t in enumerate(transcript):
        m = _ROUND_HEADER_RE.match(t)
        if m:
            items.append((int(m.group(1)), i, t))
    if not items:
        return ""
    keep = set(sorted({r for r, _, _ in items})[-k_rounds:])
    return "\n\n".join(t for r, _, t in sorted(items, key=lambda x: x[1]) if r in keep)
