"""Experiment runner: expands a YAML config into episodes, runs them, logs JSONL.

Grid mode only in this first version: conditions x personas x episodes, every episode
independent (Exp 0, 1, 2, 7). Stream mode with evolving memory (Exp 3-6) comes later.

Paired design: an episode's seed depends only on (config seed, persona-agnostic episode
index); every condition sees the same recipe sequence and start positions.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import yaml

from .agents import LLMAgentPolicy, LLMHumanPolicy, Policy, ScriptedCook, ScriptedPartner, stable_seed
from .env import CoopEnv, multiset_contains
from .llm import Backend, Usage, build_backend
from .planner import Executor, parse_macro
from .probes import run_probe

PARTNER_EVENT_KINDS = ("pot_add", "counter_place", "counter_pick", "deliver", "button")
_tls = threading.local()
_write_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, upd: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_env(env_cfg: dict) -> CoopEnv:
    """One CoopEnv per (thread, env config): avoids re-jitting every episode."""
    key = json.dumps(env_cfg, sort_keys=True)
    cache = getattr(_tls, "envs", None)
    if cache is None:
        cache = _tls.envs = {}
    if key not in cache:
        cache[key] = CoopEnv(**env_cfg)
    return cache[key]


class Backends:
    def __init__(self, specs: dict):
        self.specs, self._cache, self._lock = specs, {}, threading.Lock()

    def get(self, name: str) -> Backend:
        with self._lock:
            if name not in self._cache:
                if name not in self.specs:
                    raise KeyError(f"backend '{name}' not in backends.yaml")
                self._cache[name] = build_backend(name, self.specs[name])
            return self._cache[name]


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def run_episode(cfg: dict, cond: dict, persona_id: str, partner_id: str, seed: int, backends: Backends,
                personas: dict, out_dir: str, episode_id: str, env_over: dict = None) -> dict:
    t_start = time.time()
    env_cfg = deep_update(cfg["env"], env_over)
    env = get_env(env_cfg)
    env.reset(seed)
    slots = cfg.get("slots", {"agent": 1, "partner": 0})
    ia, ip = slots["agent"], slots["partner"]
    persona = personas[persona_id]
    usage = {"agent": Usage(), "partner": Usage()}

    # --- policies --- #
    if cond.get("policy") == "scripted_cook":
        agent: Policy = ScriptedCook(oracle_recipe=cond.get("oracle_recipe", False), patience=cond.get("patience", 30))
    else:
        agent = LLMAgentPolicy(
            backends.get(cond.get("agent_backend", cfg["backends"]["agent"])), usage["agent"],
            condition=cond.get("prompt", "direct"), reasoning_words=cond.get("reasoning_words", cfg.get("reasoning_words", 80)),
            oracle_recipe=cond.get("oracle_recipe", False),
            oracle_persona_text=persona["label"] if cond.get("oracle_persona") else "",
            partner_id=partner_id,
        )
    pb = cond.get("partner_backend", cfg["backends"]["partner"])
    partner: Policy = ScriptedPartner(persona["scripted"]) if pb == "scripted" else LLMHumanPolicy(
        backends.get(pb), usage["partner"], persona["text"])
    pols = {ia: agent, ip: partner}
    for i, p in pols.items():
        p.episode_start(env, i, seed)

    probe_cfg = cfg.get("probe", {}) if isinstance(agent, LLMAgentPolicy) else {}
    probe_on = set(probe_cfg.get("triggers", [])) if probe_cfg.get("enabled") else set()
    opt_ids = probe_cfg.get("options", [])
    letters = [chr(65 + k) for k in range(len(opt_ids))]
    persona_options = {L: personas[pid]["label"] for L, pid in zip(letters, opt_ids)}
    true_letter = dict(zip(opt_ids, letters)).get(persona_id, "")

    ex = {i: Executor(env, i, salt=seed) for i in pols}
    last_result = {i: "none (episode start)" for i in pols}
    trace, probes, decisions = [], [], {ia: 0, ip: 0}
    stats = {"parse_fail": {ia: 0, ip: 0}, "fallback": {ia: 0, ip: 0}, "macro_fail": {ia: 0, ip: 0},
             "pot_changed": {ia: 0, ip: 0},
             "wait_steps": {ia: {"ingredient": 0, "cook_serve": 0}, ip: {"ingredient": 0, "cook_serve": 0}},
             "signal_ok_steps": 0}
    pool = ThreadPoolExecutor(max_workers=2)

    def decide(i):
        return pols[i].decide(env, i, last_result[i])

    while not env.done:
        # 1) probes requested at decision points
        if "decision" in probe_on or "decision_ingredient_phase" in probe_on:
            if ex[ia].idle():
                st, contents, _ = env.pot_status(env.pots[0])
                if "decision" in probe_on or (st == "idle" and len(contents) < 3):
                    probes.append(run_probe(agent, env, ia, persona_options, true_letter, "decision"))
        # 2) decisions for idle agents (in parallel), up to 2 attempts per step
        for attempt in range(2):
            idle = [i for i in pols if ex[i].idle()]
            if not idle:
                break
            futs = {i: pool.submit(decide, i) for i in idle}
            for i, f in futs.items():
                d = f.result()
                decisions[i] += 1
                stats["parse_fail"][i] += int(not d.parse_ok)
                stats["fallback"][i] += int(d.fallback)
                m = parse_macro(d.action, env) or parse_macro("wait 1", env)
                ex[i].start(m)
                rec = {"type": "decision", "t": env.t, "agent": i, "role": pols[i].role, "action": str(m),
                       "fields": {k: v for k, v in (d.fields or {}).items() if k != "action"}, "raw": d.raw[:1500],
                       "tokens": d.tokens}
                if ex[i].ms.status == "failed":
                    stats["macro_fail"][i] += 1
                    last_result[i] = f"'{m}' could not start ({ex[i].ms.reason})"
                    rec["failed"] = ex[i].ms.reason
                trace.append(rec)
        for i in pols:  # anything still idle after 2 attempts just waits
            if ex[i].idle():
                ex[i].start(parse_macro("wait 1", env))
        # 3) step
        pot_st, pot_c, _ = env.pot_status(env.pots[0])
        phase = "ingredient" if (pot_st == "idle" and len(pot_c) < 3) else "cook_serve"
        for i in pols:
            if ex[i].ms and ex[i].ms.macro.kind == "wait":
                stats["wait_steps"][i][phase] += 1
        actions = {i: ex[i].act() for i in pols}
        # A waiting agent standing where the other one needs to go steps aside.
        for i in pols:
            for j in pols:
                if i != j and ex[i].intended == env.agent_pos(j) and ex[j].ms and ex[j].ms.macro.kind == "wait":
                    side = ex[j].sidestep(avoid={env.agent_pos(i)})
                    if side is not None:
                        actions[j] = side
        _, evs = env.step(actions)
        for i in pols:
            ex[i].after_step()
        for e in evs:
            trace.append({"type": "event", **asdict(e)})
        if persona["scripted"].get("kind") == "signal":
            stats["signal_ok_steps"] += int(_signal_ok(env, persona["scripted"]))
        # 4) outcomes, interrupts, probes
        for i in pols:
            ms = ex[i].ms
            if ms and ms.status == "done":
                last_result[i] = f"'{ms.macro}' finished"
            elif ms and ms.status == "failed":
                key = "pot_changed" if ms.reason.startswith("the pot changed") else "macro_fail"
                stats[key][i] += 1
                last_result[i] = f"'{ms.macro}' failed ({ms.reason})"
                ms.status = "failed_reported"
            other_evs = [e for e in evs if e.agent != i and e.kind in PARTNER_EVENT_KINDS
                         and (env.in_view(i, e.pos) or env.in_view(i, env.agent_pos(e.agent)))]
            if other_evs and cfg.get("replan_on_partner_event", True) and ms and ms.status == "running" and ms.macro.kind == "wait":
                ms.status = "done"
                last_result[i] = f"'{ms.macro}' interrupted: you saw your partner act"
        partner_pot = [e for e in evs if e.agent == ip and e.kind == "pot_add"
                       and (env.in_view(ia, e.pos) or env.in_view(ia, env.agent_pos(ip)))]
        if partner_pot and "partner_pot_event" in probe_on:
            probes.append(run_probe(agent, env, ia, persona_options, true_letter, "partner_pot_event"))
    pool.shutdown()

    for p in probes:
        trace.append({"type": "probe", **p})
    rec = {
        "episode_id": episode_id, "condition": cond["name"], "persona": persona_id, "partner_id": partner_id,
        "seed": seed, "env": env_cfg, "score": env.score,
        "n_correct": sum(1 for e in env.events if e.kind == "deliver" and e.correct),
        "n_wrong": sum(1 for e in env.events if e.kind == "deliver" and not e.correct),
        "soups": soup_metrics(env, ia, ip, probes),
        "decisions": {"agent": decisions[ia], "partner": decisions[ip]},
        "parse_fail": {"agent": stats["parse_fail"][ia], "partner": stats["parse_fail"][ip]},
        "fallback": {"agent": stats["fallback"][ia], "partner": stats["fallback"][ip]},
        "macro_fail": {"agent": stats["macro_fail"][ia], "partner": stats["macro_fail"][ip]},
        "pot_changed_replans": {"agent": stats["pot_changed"][ia], "partner": stats["pot_changed"][ip]},
        "wait_steps": {"agent": stats["wait_steps"][ia], "partner": stats["wait_steps"][ip]},
        "probe_summary": summarize_probes(probes),
        "adherence": adherence(env, ip, persona, stats, env.max_steps),
        "usage": {k: v.as_dict() for k, v in usage.items()},
        "wall_s": round(time.time() - t_start, 1),
    }
    write_jsonl(os.path.join(out_dir, "episodes.jsonl"), rec)
    safe = episode_id.replace("|", "__").replace("/", "_")
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    with open(os.path.join(out_dir, "traces", f"{safe}.jsonl"), "w") as f:
        for r in trace:
            f.write(json.dumps(r, default=str) + "\n")
    return rec


def _signal_ok(env: CoopEnv, spec: dict) -> bool:
    conv = {k: tuple(v) for k, v in spec["convention"].items()}
    want = conv.get(env.recipe_str(env.current_recipe))
    return all((env.cell(c)[1] != 0) == (c == want) for c in conv.values())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def soup_metrics(env: CoopEnv, ia: int, ip: int, probes: list) -> list:
    out = []
    for s in env.soups:
        sizes, contents, singleton_t = [], [], None
        for t, a, ing in s.adds:
            contents.append(ing)
            n = len(env.consistent_recipes(contents))
            sizes.append(n)
            if n == 1 and singleton_t is None:
                singleton_t = t
        robot_adds = [(t, ing) for t, a, ing in s.adds if a == ia]
        first = robot_adds[0][0] if robot_adds else None
        before = [ing for t, a, ing in s.adds if first is not None and (t < first or (t == first and a != ia))]
        size_before = len(env.consistent_recipes(before)) if first is not None else None
        prior = [p for p in probes if first is not None and p["t"] <= first and p["t"] >= s.start_t]
        pac = prior[-1] if prior else None
        cum, wrong_add = [], False
        for t, a, ing in s.adds:
            cum.append(ing)
            if a == ia and not multiset_contains(s.target, cum):
                wrong_add = True
        out.append({
            "idx": s.idx, "target": env.recipe_str(s.target), "contents": [env.ing_name(c) for c in s.contents],
            "adds": [(t, "agent" if a == ia else "partner", env.ing_name(ing)) for t, a, ing in s.adds],
            "partner_adds": sum(1 for _, a, _ in s.adds if a == ip), "agent_adds": len(robot_adds),
            "start_t": s.start_t, "full_t": s.full_t, "delivered_t": s.delivered_t, "correct": s.correct,
            "agent_commit_t": first, "consistent_size_at_commit": size_before,
            "premature_commit": (size_before > 1) if size_before is not None else None,
            "singleton_t": singleton_t,
            "over_wait": (first - singleton_t) if (first is not None and singleton_t is not None and singleton_t <= first) else None,
            "agent_wrong_add": wrong_add,
            "probe_at_commit": {"top1_correct": pac["top1_correct"], "p_true": pac["p_true"]} if pac else None,
            "partner_first_ingredient_minority": _minority_first(s),
        })
    return out


def _minority_first(s) -> Optional[bool]:
    if not s.adds or len(set(s.target)) == 1:
        return None
    return s.target.count(s.adds[0][2]) == 1


def summarize_probes(probes: list) -> dict:
    if not probes:
        return {}
    n = len(probes)
    return {
        "n": n,
        "top1_acc": round(sum(p["top1_correct"] for p in probes) / n, 4),
        "p_true": round(sum(p["p_true"] for p in probes) / n, 4),
        "violation_mass": round(sum(p["violation_mass"] for p in probes) / n, 4),
        "partner_type_acc": round(sum(p["partner_type_correct"] for p in probes) / n, 4),
        "parsed": round(sum(p["parsed"] for p in probes) / n, 4),
    }


def adherence(env, ip, persona, stats, T) -> dict:
    sc = persona["scripted"]
    out = {"partner_wait_frac": round(sum(stats["wait_steps"][ip].values()) / max(1, T), 4)}
    if sc.get("kind") == "demo":
        closed = [s for s in env.soups if s.full_t is not None]
        if closed:
            out["partner_adds_per_soup"] = round(sum(sum(1 for _, a, _ in s.adds if a == ip) for s in closed) / len(closed), 3)
            out["expected_partner_adds"] = sc.get("n_add", 3)
    if sc.get("kind") == "signal":
        out["signal_ok_frac"] = round(stats["signal_ok_steps"] / max(1, T), 4)
    return out


def episode_summary(env: CoopEnv, i: int, max_chars: int = 3000) -> str:
    """What the agent experienced, from its own perspective (no hidden ground truth)."""
    lines = [f"Episode finished. Team score {env.score:+.0f}."]
    for s in env.soups:
        adds = ", ".join(f"{env.ing_name(ing)} ({'you' if a == i else 'partner'}, t={t})" for t, a, ing in s.adds)
        outcome = "not delivered" if s.correct is None else ("delivered: CORRECT" if s.correct else "delivered: WRONG")
        lines.append(f"Soup {s.idx + 1}: added in this order: {adds} -> {outcome}.")
    obs = [env.event_text(e, i) for e in env.observed[i]]
    lines.append("Timeline of what you observed:\n" + "\n".join(obs[-60:]))
    return "\n".join(lines)[-max_chars:]


def write_jsonl(path, rec):
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Experiment expansion
# --------------------------------------------------------------------------- #
def run_grid(cfg, personas, backends, out_dir, workers, done_ids):
    units = []
    for cond in cfg["conditions"]:
        for pid in cfg["personas"]:
            for ep in range(cfg["episodes_per_cell"]):
                eid = f"{cond['name']}|{pid}|{ep}"
                if eid in done_ids:
                    continue
                seed = stable_seed(cfg.get("seed", 0), "ep", ep)
                units.append((cond, pid, ep, eid, seed))

    def _one(u):
        cond, pid, ep, eid, seed = u
        return run_episode(cfg, cond, pid, cfg.get("partner_ids", {}).get(pid, pid), seed, backends, personas,
                           out_dir, eid)

    _run_pool(units, _one, workers)


def _run_pool(units, fn, workers):
    if workers <= 1:
        for u in units:
            _report(fn(u))
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, u) for u in units]
        for f in as_completed(futs):
            _report(f.result())


def _report(r):
    if isinstance(r, dict):
        print(f"{r['episode_id']}: score {r['score']:+.0f} (correct {r['n_correct']}, wrong {r['n_wrong']}), "
              f"calls agent/partner {r['usage']['agent']['calls']}/{r['usage']['partner']['calls']}, {r['wall_s']}s", flush=True)
    elif r is not None:
        print(f"condition done: {r}", flush=True)
