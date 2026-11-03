"""Prompt templates.

Everything that is not the manipulated variable is identical across conditions:
rules, map, menu, valid-action list and observation format.
Only the OUTPUT block differs (direct / cot / tom_free / tom_slots / tom_2nd).
"""
from __future__ import annotations

from .env import CoopEnv
from .planner import grammar_for

RULES = """\
You are one of two cooks in a grid-world kitchen (Overcooked). You and your partner share one score.
- A soup needs exactly 3 ingredients that match the CURRENT ORDER (a multiset, order of adding does not matter).
- A pot starts cooking by itself once it holds 3 ingredients; cooking takes 20 steps. Ingredients cannot be taken out of a pot.
- With an empty plate in hand, take a ready soup from the pot ("plate soup"), then deliver it at the serving window X.
- Correct delivery: +20. Wrong soup: -20. After a correct delivery a new order is drawn from the menu.
- The current order is shown only on the order display R, readable only when you are within 2 cells of it (Chebyshev distance). You see only cells within 2 cells of yourself.
- You cannot talk to your partner. You can only act and watch what they do.
- Each step both cooks act at the same time. Time is limited."""

OUTPUT_SPECS = {
    "direct": 'Reply with JSON only: {{"action": "<one action>"}}',
    "cot": (
        'Reply with JSON only: {{"reasoning": "<step-by-step reasoning about what to do next, at most {w} words>", '
        '"action": "<one action>"}}'
    ),
    "tom_free": (
        'Reply with JSON only: {{"partner_reasoning": "<reason about your partner: what they know (can they see the order?), '
        'what they are trying to do, what kind of partner they are, and what they will do next; then what that implies for you; '
        'at most {w} words>", "action": "<one action>"}}'
    ),
    "tom_slots": (
        'Reply with JSON only: {{"recipe_belief": {{<each menu item>: <probability>}}, '
        '"partner_type": "<one-sentence description of how this partner works>", '
        '"partner_next_action": "<the action you expect from your partner next>", "action": "<one action>"}}'
    ),
    "tom_2nd": (
        'Reply with JSON only: {{"recipe_belief": {{<each menu item>: <probability>}}, '
        '"partner_type": "<one-sentence description of how this partner works>", '
        '"partner_next_action": "<the action you expect from your partner next>", '
        '"partner_expects_me_to": "<the action your partner expects from you now>", "action": "<one action>"}}'
    ),
}


def agent_system_prompt(env: CoopEnv, me: int, condition: str, reasoning_words: int,
                        partner_id: str, oracle_persona_text: str = "") -> str:
    parts = [
        RULES,
        "MAP (fixed; x grows to the right, y grows downward):\n" + env.static_map_text(),
        f"MENU: {' | '.join(env.menu())}",
        f"You are agent {me}. Your partner today is '{partner_id}'.",
        "ACTIONS (macro-actions; walking is automatic):\n" + grammar_for(env),
    ]
    if oracle_persona_text:
        parts.append(f"How your partner works (given to you): {oracle_persona_text}")
    parts.append(OUTPUT_SPECS[condition].format(w=reasoning_words))
    return "\n\n".join(parts)


def decision_user_prompt(obs_text: str, valid: list[str], last_result: str) -> str:
    return (
        f"{obs_text}\n\nResult of your last action: {last_result}\n"
        f"VALID_ACTIONS: {' | '.join(valid)}\n"
        "Choose your next action."
    )


def human_system_prompt(env: CoopEnv, me: int, persona_text: str, partner_label: str = "your teammate") -> str:
    return "\n\n".join([
        RULES,
        "MAP (fixed; x grows to the right, y grows downward):\n" + env.static_map_text(),
        f"MENU: {' | '.join(env.menu())}",
        f"You are agent {me}. You are playing a PERSON with a consistent working style. "
        f"Stay in character for the whole episode, even when another style would score more.\n"
        f"YOUR WORKING STYLE: {persona_text}",
        "ACTIONS (macro-actions; walking is automatic):\n" + grammar_for(env),
        'Reply with JSON only: {"thought": "<at most 20 words>", "action": "<one action>"}',
    ])


def probe_prompt(menu: list[str], persona_options: dict[str, str]) -> str:
    opts = " ".join(f"({k}) {v}" for k, v in persona_options.items())
    return (
        "PROBE (a side question; it does not change the game and you do not act now).\n"
        f"MENU: {' | '.join(menu)}\n"
        "1) What is the current order? Give a probability for every menu item.\n"
        f"2) Which description fits your partner best? Options: {opts}\n"
        "3) What will your partner do next?\n"
        'Reply with JSON only: {"recipe_probs": {<menu item>: <probability>}, "partner_type": "<option letter>", '
        '"partner_next_action": "<action>"}'
    )
