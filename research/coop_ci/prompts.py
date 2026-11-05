"""Prompt templates.

Rules, map, menu, orders, observation and valid actions are identical across ego conditions.
A0 / A1 / A1c differ only in the OUTPUT block. A2 / A3 / A4 are A0 plus one labelled
PARTNER INFORMATION block. Non-oracle prompts never contain the partner's persona, its
committed macro or target, or its private thought (tested).
"""
from __future__ import annotations

from .env import CoopEnv
from .planner import grammar_for

INJECT_HEADER = "PARTNER INFORMATION (given to you)"


def rules(env: CoopEnv) -> str:
    view = ("You and your partner see the whole kitchen: both cooks, what each holds, every pot and every counter."
            if env.view_radius is None else
            f"You see only cells within {env.view_radius} cells of yourself (Chebyshev distance).")
    orders = ("There is one open order at a time." if env.n_open == 1 else
              f"There are {env.n_open} open orders at a time; any of them can be served.")
    return f"""\
You are one of two cooks in a grid-world kitchen (Overcooked). You and your partner share one score.
- Both cooks know the open orders. {orders} After a correct delivery that order closes and the next one opens.
- A soup needs exactly 3 ingredients that match an open order (a multiset; the order of adding does not matter).
- A pot starts cooking by itself once it holds 3 ingredients; cooking takes 20 steps. Ingredients cannot be taken out of a pot.
- You hold at most one item. To take something else, first get rid of what you hold: put it into a pot, deliver it, place it on a free counter, or put an ingredient or empty plate back on its pile ("put back").
- With an empty plate in hand, take a cooked soup from the pot ("plate soup"), then deliver it at the serving window X.
- Correct delivery: +20. Wrong soup: -20.
- {view}
- You cannot talk to your partner. You can only act and watch what they do.
- Each step both cooks act at the same time. Time is limited."""


def _common(env: CoopEnv) -> list[str]:
    return [
        rules(env),
        "MAP (fixed; x grows to the right, y grows downward):\n" + env.static_map_text(),
        f"MENU: {' | '.join(env.menu())}",
        "ACTIONS (macro-actions; walking is automatic):\n" + grammar_for(env),
    ]


OUTPUT_SPECS = {
    "A0": 'Reply with JSON only: {{"plan": "<your plan, at most 30 words>", "action": "<one action>"}}',
    "A1": (
        'Reply with JSON only: {{"partner_current_subgoal": "<what your partner is doing right now>", '
        '"evidence": "<what you observed that shows it>", '
        '"partner_tendency": "<how this partner usually works, as far as you can tell>", '
        '"my_complementary_action_reason": "<why your action complements your partner>", '
        '"action": "<one action>"}}. Use at most {w} words in total.'
    ),
    "A1c": (
        'Reply with JSON only: {{"reasoning": "<step-by-step reasoning about what to do next, at most {w} words>", '
        '"action": "<one action>"}}'
    ),
}
OUTPUT_OF = {"A0": "A0", "A1": "A1", "A1c": "A1c", "A2": "A0", "A3": "A0", "A4": "A0"}
CONDITIONS = tuple(OUTPUT_OF)


MEMORY_HEADER = "MEMORY (your own notes from earlier episodes; they may be incomplete or out of date)"


def ego_system_prompt(env: CoopEnv, condition: str, words: int, partner_id: str, profile_label: str = "",
                      memory_text: str = None) -> str:
    """`memory_text` (Exp B/C): None = no memory block at all (Exp A); "" = the block with 'none yet'."""
    parts = _common(env) + [f"Your partner today: {partner_id}."]
    if memory_text is not None:
        parts.append(f"{MEMORY_HEADER}:\n{memory_text.strip() or '(none yet)'}")
    if condition in ("A2", "A4"):
        parts.append(f"{INJECT_HEADER}: Partner profile: {profile_label}")
    parts.append(OUTPUT_SPECS[OUTPUT_OF[condition]].format(w=words))
    return "\n\n".join(parts)


def decision_user_prompt(obs_text: str, valid: list[str], last_result: str, injected: str = "") -> str:
    inj = f"{INJECT_HEADER}: {injected}\n" if injected else ""
    return (f"{obs_text}\n\n{inj}Result of your last action: {last_result}\n"
            f"VALID_ACTIONS: {' | '.join(valid)}\nChoose your next action.")


def human_system_prompt(env: CoopEnv, persona_text: str) -> str:
    return "\n\n".join(_common(env) + [
        "You are playing a PERSON with a consistent working style. Stay in character for the whole "
        "episode, even when another style would score more.\n"
        f"YOUR WORKING STYLE: {persona_text}",
        'Reply with JSON only: {"thought": "<at most 20 words>", "action": "<one action>"}',
    ])
