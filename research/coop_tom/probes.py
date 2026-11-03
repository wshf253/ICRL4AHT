"""Forked belief probes.

A probe re-uses the agent's exact system prompt and observation, asks for its belief,
and is NOT appended to anything the agent sees later. Asking inside the decision
prompt would itself be a ToM prompt and contaminate the `direct` condition.
"""
from __future__ import annotations

from .agents import LLMAgentPolicy, stable_seed
from .env import CoopEnv, multiset_contains
from .llm import parse_json
from .planner import valid_actions
from .prompts import decision_user_prompt, probe_prompt


def run_probe(policy: LLMAgentPolicy, env: CoopEnv, i: int, persona_options: dict, true_persona_letter: str,
              trigger: str) -> dict:
    sys = policy.system_prompt(env, i)
    user = decision_user_prompt(env.observation_text(i, policy.oracle_recipe, policy.n_events), valid_actions(env, i), "(probe)")
    user += "\n\n" + probe_prompt(env.menu(), persona_options)
    r = policy.backend.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                            max_tokens=300, seed=stable_seed(policy.ep_seed, env.t, i, "probe", trigger))
    policy.usage.add(r)
    js = parse_json(r.text) or {}
    return score_probe(env, i, js, true_persona_letter, trigger, r.text)


def score_probe(env: CoopEnv, i: int, js: dict, true_persona_letter: str, trigger: str, raw: str = "") -> dict:
    menu = env.menu()
    truth = env.recipe_str(env.current_recipe)
    probs = js.get("recipe_probs") or {}
    p = {}
    for k in menu:
        try:
            p[k] = max(0.0, float(probs.get(k, 0.0)))
        except (TypeError, ValueError):
            p[k] = 0.0
    z = sum(p.values())
    p = {k: (v / z if z > 0 else 1.0 / len(menu)) for k, v in p.items()}
    pot = env.pots[0]
    # What the agent knows about the pot: current view, else its last sighting.
    if env.in_view(i, pot):
        contents = env.pot_status(pot)[1]
    else:
        seen = env.last_seen[i].get(pot)
        contents = seen[1][1] if seen else []
    consistent = {env.recipe_str(r) for r in env.recipes if multiset_contains(r, contents)}
    top = max(p, key=p.get)
    ptype = str(js.get("partner_type", "")).strip().strip("()").upper()[:1]
    return {
        "t": env.t, "trigger": trigger, "truth": truth, "top1": top, "top1_correct": top == truth,
        "p_true": round(p.get(truth, 0.0), 4),
        "violation_mass": round(sum(v for k, v in p.items() if k not in consistent), 4),
        "consistent_size": len(consistent),
        "partner_type": ptype, "partner_type_correct": bool(true_persona_letter) and ptype == true_persona_letter,
        "partner_next_action": str(js.get("partner_next_action", "")), "parsed": bool(js), "raw": raw[:500],
    }
