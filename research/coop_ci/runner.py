"""Episode engine and grid runner.

`Episode` steps one ego + one partner through an env with the macro executors, records the
trace and the metrics of IMPLEMENTATION_PLAN section 5, and supports snapshot()/restore() so Gate 0
(validate.py) and saved-context regret can branch from the same state.

Grid mode: cells (layout x personas) x conditions x episodes; every episode independent.
Paired design: an episode's seed depends only on (config seed, episode index); every condition
sees the same order sequence. Seats are balanced: the ego is agent 0 on even episode indices
and agent 1 on odd ones.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import yaml

from .agents import LLMEgoPolicy, LLMHumanPolicy
from .env import A_STAY, ACTION_VEC, MOVE_ACTIONS, CoopEnv, decode_ingredients, is_ingredient
from .llm import Backend, Usage, build_backend
from .planner import NOT_INFRA, Executor, parse_macro
from .policy import Policy, stable_seed
from .scripted import complies, ego_policy, opportunity, persona_policy, useful_actions

PARTNER_EVENT_KINDS = ("pot_add", "counter_place", "counter_pick", "deliver", "plate_soup")
NO_DUP = ("wait", "move")
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


def ego_seat(ep_index: int) -> int:
    return ep_index % 2


def partner_token(*parts) -> str:
    """Opaque partner id (never reveals the persona)."""
    return f"P-{stable_seed('partner', *parts) % 65536:04x}"


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def _zero_counts() -> dict:
    return {"decisions": Counter(), "parse_fail": Counter(), "fallback": Counter(), "fail": {0: Counter(), 1: Counter()},
            "wait_steps": Counter(), "ego_idle_useful": 0, "blocking_steps": 0, "dup_subtask_steps": 0,
            "dup_ingredient": 0, "surplus_plate": 0, "wrong_soup_add": Counter(), "comp_opp": 0, "comp_match": 0,
            "steps": 0, "decision_points": Counter(), "unrecovered": Counter()}


class Episode:
    def __init__(self, env: CoopEnv, seed: int, ego: int, ego_policy: Policy, partner_policy: Policy,
                 persona: dict, replan_on_partner_event: bool = True, record_trace: bool = True, salt: int = 0):
        self.env, self.seed, self.ego, self.partner = env, seed, ego, 1 - ego
        self.pols = {ego: ego_policy, 1 - ego: partner_policy}
        self.persona = persona
        self.replan = replan_on_partner_event
        self.record_trace = record_trace
        self.ex = {i: Executor(env, i, salt=seed * 7 + salt) for i in (0, 1)}
        for i, p in self.pols.items():
            p.episode_start(env, i, seed, salt)
        self.last_result = {i: "none (episode start)" for i in (0, 1)}
        self.trace: list = []
        self.c = _zero_counts()
        self._dup_prev: set = set()
        self.t0, self.score0 = env.t, env.score
        self._pool: Optional[ThreadPoolExecutor] = None
        self._llm = any(isinstance(p, (LLMEgoPolicy, LLMHumanPolicy)) for p in self.pols.values())

    # ---- branching ---- #
    def snapshot(self) -> dict:
        return {"env": self.env.snapshot(), "ex": {i: e.get_state() for i, e in self.ex.items()},
                "pols": {i: p.get_state() for i, p in self.pols.items()},
                "last_result": dict(self.last_result), "c": copy.deepcopy(self.c), "dup_prev": set(self._dup_prev)}

    def restore(self, snap: dict, reset_counts: bool = True):
        self.env.restore(snap["env"])
        for i, e in self.ex.items():
            e.set_state(snap["ex"][i])
        for i, p in self.pols.items():
            p.set_state(snap["pols"][i])
        self.last_result = dict(snap["last_result"])
        self._dup_prev = set(snap["dup_prev"])
        self.trace = []
        if reset_counts:
            self.c, self.t0, self.score0 = _zero_counts(), self.env.t, self.env.score
        else:
            self.c = copy.deepcopy(snap["c"])

    def replace_policy(self, seat: int, policy: Policy, salt: int = 0):
        """Swap the policy of `seat` at a branch point; it decides at the next step."""
        policy.episode_start(self.env, seat, self.seed, salt)
        self.pols[seat] = policy
        self.ex[seat].ms = None
        self.last_result[seat] = "none (episode start)"
        self._llm = any(isinstance(p, (LLMEgoPolicy, LLMHumanPolicy)) for p in self.pols.values())

    # ---- stepping ---- #
    def _context(self, i: int) -> dict:
        """Runner-side information; only the A3 ego condition reads it."""
        j = 1 - i
        ms = self.ex[j].ms
        doing = None
        if ms is not None and ms.status == "running":
            tgt = ms.target
            where = f" at {tgt}" if tgt is not None else ""
            doing = f"{ms.macro}{where} [{ms.subgoal[1]}]"
        return {"partner_doing": doing}

    def _decide(self, i: int):
        return self.pols[i].decide(self.env, i, self.last_result[i], self._context(i))

    def _commit(self, i: int, d, attempt: int):
        env, c = self.env, self.c
        c["decisions"][i] += 1
        c["parse_fail"][i] += int(not d.parse_ok)
        c["fallback"][i] += int(d.fallback)
        m = parse_macro(d.action, env) or parse_macro("wait 1", env)
        allowed, ok = None, None
        if i == self.partner:
            allowed = opportunity(env, i, self.persona)
            if allowed is not None:
                ok = complies(env, m, allowed)
                c["comp_opp"] += 1
                c["comp_match"] += int(ok)
        self.ex[i].start(m)
        ms = self.ex[i].ms
        if self.record_trace:
            rec = {"type": "decision", "t": env.t, "agent": i, "role": "ego" if i == self.ego else "partner",
                   "policy": self.pols[i].name, "action": str(m), "subgoal": list(ms.subgoal),
                   "target": ms.target, "fields": {k: v for k, v in (d.fields or {}).items() if k != "action"},
                   "raw": d.raw[:1500], "tokens": d.tokens, "attempt": attempt, "parse_ok": d.parse_ok,
                   "holding": env.item_text(env.inventory(i))}
            if allowed is not None:  # evaluator-only compliance audit
                rec["opportunity"] = {"tendency": sorted(allowed), "complied": ok}
            if d.private:
                rec["private"] = d.private  # trace only; never shown to the other cook
            if ms.status == "failed":
                rec["failed"] = {"code": ms.code, "reason": ms.reason}
            self.trace.append(rec)
        if ms.status == "failed":
            c["fail"][i][ms.code] += 1
            self.last_result[i] = f"'{m}' could not start ({ms.reason})"
            ms.status = "failed_reported"

    def decide_idle(self):
        first_idle = [i for i in (0, 1) if self.ex[i].idle()]
        for i in first_idle:
            self.c["decision_points"][i] += 1
        for attempt in range(2):
            idle = [i for i in (0, 1) if self.ex[i].idle()]
            if not idle:
                break
            needs_partner_first = isinstance(self.pols[self.ego], LLMEgoPolicy) and self.pols[self.ego].condition == "A3"
            if needs_partner_first and self.partner in idle and self.ego in idle:
                # A3 shows the partner's committed macro: let the partner commit first.
                self._commit(self.partner, self._decide(self.partner), attempt)
                idle = [self.ego]
            if self._llm and len(idle) > 1:
                self._pool = self._pool or ThreadPoolExecutor(max_workers=2)
                futs = {i: self._pool.submit(self._decide, i) for i in idle}
                ds = {i: f.result() for i, f in futs.items()}
            else:
                ds = {i: self._decide(i) for i in idle}
            for i in idle:
                self._commit(i, ds[i], attempt)
        for i in (0, 1):  # anything still idle after 2 attempts just waits
            if self.ex[i].idle():
                if i in first_idle:
                    self.c["unrecovered"][i] += 1
                self.ex[i].start(parse_macro("wait 1", self.env))

    def step(self):
        env, c = self.env, self.c
        self.decide_idle()
        ego, par = self.ego, self.partner
        sg = {i: self.ex[i].ms.subgoal[0] for i in (0, 1)}
        for i in (0, 1):
            if self.ex[i].ms.macro.kind == "wait":
                c["wait_steps"][i] += 1
        if self.ex[ego].ms.macro.kind == "wait" and useful_actions(env, ego, sg[par]):
            c["ego_idle_useful"] += 1
        if sg[0] == sg[1] and sg[0] not in NO_DUP:
            c["dup_subtask_steps"] += 1
        actions = {i: self.ex[i].act() for i in (0, 1)}
        for i in (0, 1):  # a waiting cook standing where the other needs to go steps aside
            j = 1 - i
            if self.ex[i].intended == env.agent_pos(j) and self.ex[j].ms and self.ex[j].ms.macro.kind == "wait":
                side = self.ex[j].sidestep(avoid={env.agent_pos(i)})
                if side is not None:
                    actions[j] = side
        pos0 = {i: env.agent_pos(i) for i in (0, 1)}
        _, evs = env.step(actions)
        c["steps"] += 1
        blocked = False
        for i in (0, 1):
            a = actions.get(i, A_STAY)
            if a in MOVE_ACTIONS:
                dx, dy = ACTION_VEC[a]
                tgt = (pos0[i][0] + dx, pos0[i][1] + dy)
                blocked |= tgt == pos0[1 - i] and env.agent_pos(i) == pos0[i]
        c["blocking_steps"] += int(blocked)
        for i in (0, 1):
            self.ex[i].after_step()
        self._waste(evs)
        if self.record_trace:
            for e in evs:
                self.trace.append({"type": "event", **asdict(e)})
        for i in (0, 1):
            ms = self.ex[i].ms
            if ms and ms.status == "done":
                self.last_result[i] = f"'{ms.macro}' finished" + (f" ({ms.note})" if ms.note else "")
            elif ms and ms.status == "failed":
                c["fail"][i][ms.code] += 1
                self.last_result[i] = f"'{ms.macro}' failed ({ms.reason})"
                ms.status = "failed_reported"
            seen = [e for e in evs if e.agent != i and e.kind in PARTNER_EVENT_KINDS
                    and (env.in_view(i, e.pos) or env.in_view(i, env.agent_pos(e.agent)))]
            if seen and self.replan and ms and ms.status == "running" and ms.macro.kind == "wait":
                ms.status = "done"
                self.last_result[i] = f"'{ms.macro}' interrupted: you saw your partner act"

    def _waste(self, evs):
        env, c = self.env, self.c
        plan = {pp["name"]: pp for pp in env.pot_plan()}
        for e in evs:
            if e.kind == "pot_add" and plan[e.extra["pot"]]["target"] is None:
                c["wrong_soup_add"][e.agent] += 1
            if e.kind in ("fetch", "counter_pick") and e.item == "an empty plate":
                demand = sum(1 for p in env.pots if env.pot_status(p)[0] in ("cooking", "ready"))
                c["surplus_plate"] += int(env.plates_out() > demand)
        need = env.missing_ingredients()
        held = [decode_ingredients(env.inventory(i))[0] if is_ingredient(env.inventory(i)) else None for i in (0, 1)]
        dup = {held[0]} if held[0] is not None and held[0] == held[1] and need[held[0]] < 2 else set()
        c["dup_ingredient"] += len(dup - self._dup_prev)
        self._dup_prev = dup

    def run(self, until_t: int):
        while self.env.t < min(until_t, self.env.max_steps):
            self.step()

    def close(self):
        if self._pool:
            self._pool.shutdown()
            self._pool = None

    # ---- metrics over [t0, now] ---- #
    def metrics(self) -> dict:
        env, c, ego, par = self.env, self.c, self.ego, self.partner
        T = max(1, c["steps"])
        dels = [e for e in env.events if e.kind == "deliver" and e.t > self.t0]
        good = [e for e in dels if e.correct]
        ready_idle = []
        for s in env.soups:
            if s.ready_t is not None and s.ready_t >= self.t0:
                ready_idle.append((s.plated_t if s.plated_t is not None else env.t) - s.ready_t)
        fails = {r: dict(c["fail"][i]) for r, i in (("ego", ego), ("partner", par))}
        infra = {r: sum(v for k, v in fails[r].items() if k not in NOT_INFRA) + c["parse_fail"][i]
                 for r, i in (("ego", ego), ("partner", par))}
        dec = {"ego": c["decisions"][ego], "partner": c["decisions"][par]}
        # After the retry policy (2 attempts per step): decision points that ended in a forced wait,
        # plus failures of macros that did start (timeout, target gone, deadlock).
        pts = {"ego": c["decision_points"][ego], "partner": c["decision_points"][par]}
        after = {r: c["unrecovered"][i] + sum(fails[r].get(k, 0) for k in ("timeout", "target_gone", "deadlock"))
                 for r, i in (("ego", ego), ("partner", par))}
        return {
            "return": env.score - self.score0, "n_correct": len(good), "n_wrong": len(dels) - len(good),
            "first_correct_t": (good[0].t - self.t0) if good else None, "steps": c["steps"],
            "ready_idle_steps": sum(ready_idle), "ego_idle_while_useful": c["ego_idle_useful"],
            "wait_frac": {"ego": round(c["wait_steps"][ego] / T, 4), "partner": round(c["wait_steps"][par] / T, 4)},
            "blocking_rate": round(c["blocking_steps"] / T, 4), "dup_subtask_rate": round(c["dup_subtask_steps"] / T, 4),
            "waste": {"dup_ingredient": c["dup_ingredient"], "surplus_plate": c["surplus_plate"],
                      "wrong_soup_add": {"ego": c["wrong_soup_add"][ego], "partner": c["wrong_soup_add"][par]},
                      "counter_items_end": len(env.counter_items())},
            "compliance": {"opportunities": c["comp_opp"], "matches": c["comp_match"],
                           "rate": round(c["comp_match"] / c["comp_opp"], 4) if c["comp_opp"] else None},
            "decisions": dec, "failures": fails,
            "parse_fail": {"ego": c["parse_fail"][ego], "partner": c["parse_fail"][par]},
            "fallback": {"ego": c["fallback"][ego], "partner": c["fallback"][par]},
            "infra_rate": {r: round(infra[r] / dec[r], 4) if dec[r] else 0.0 for r in dec},
            "decision_points": pts,
            "infra_after_retry": {r: round(after[r] / pts[r], 4) if pts[r] else 0.0 for r in pts},
            "deadlocks": fails["ego"].get("deadlock", 0) + fails["partner"].get("deadlock", 0),
        }


# --------------------------------------------------------------------------- #
# Policy factory
# --------------------------------------------------------------------------- #
def make_ego(cond: dict, persona_id: str, personas: dict, backends: Backends, usage: Usage, partner_id: str,
             default_backend: str) -> Policy:
    kind = cond.get("ego", "llm")
    if kind == "llm":
        c = cond.get("condition", "A0")
        label = ""
        if c == "A2":
            label = personas[persona_id]["label"]
        elif c == "A4":
            label = personas[mismatched(personas, persona_id)]["label"]
        return LLMEgoPolicy(backends.get(cond.get("backend", default_backend)), usage, condition=c,
                            words=cond.get("words", 80), partner_id=partner_id, profile_label=label)
    return ego_policy(personas, kind, persona_id)


def mismatched(personas: dict, pid: str) -> str:
    fam = personas[pid]["family"]
    others = sorted(k for k, v in personas.items() if v.get("family") == fam and k != pid)
    return others[0]


def make_partner(backend_name: str, persona_id: str, personas: dict, backends: Backends, usage: Usage) -> Policy:
    if backend_name == "scripted":
        return persona_policy(personas, persona_id)
    return LLMHumanPolicy(backends.get(backend_name), usage, personas[persona_id]["text"])


def run_episode(cfg: dict, cell: dict, cond: dict, persona_id: str, ep: int, backends: Backends, personas: dict,
                out_dir: str) -> dict:
    t_start = time.time()
    env_cfg = deep_update(cfg.get("env", {}), {"layout": cell["layout"], **cell.get("env", {})})
    env = get_env(env_cfg)
    seed = stable_seed(cfg.get("seed", 0), "ep", ep)
    env.reset(seed)
    seat = ego_seat(ep)
    pid_token = partner_token(cfg.get("seed", 0), persona_id, cell["layout"])
    usage = {"ego": Usage(), "partner": Usage()}
    ego = make_ego(cond, persona_id, personas, backends, usage["ego"], pid_token, cfg["backends"]["ego"])
    partner = make_partner(cond.get("partner", cfg["backends"]["partner"]), persona_id, personas, backends,
                           usage["partner"])
    epi = Episode(env, seed, seat, ego, partner, personas[persona_id],
                  replan_on_partner_event=cfg.get("replan_on_partner_event", True))
    try:
        epi.run(env.max_steps)
    finally:
        epi.close()
    eid = f"{cond['name']}|{cell['layout']}|{persona_id}|{ep}"
    rec = {"episode_id": eid, "condition": cond["name"], "layout": cell["layout"], "persona": persona_id,
           "family": personas[persona_id].get("family"), "partner_id": pid_token, "ep": ep, "seed": seed,
           "ego_seat": seat, "ego_policy": ego.name, "partner_policy": partner.name, "env": env_cfg,
           **epi.metrics(), "soups": soup_records(env, seat),
           "usage": {k: v.as_dict() for k, v in usage.items()}, "wall_s": round(time.time() - t_start, 1)}
    write_jsonl(os.path.join(out_dir, "episodes.jsonl"), rec)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    with open(os.path.join(out_dir, "traces", eid.replace("|", "__") + ".jsonl"), "w") as f:
        for r in epi.trace:
            f.write(json.dumps(r, default=str) + "\n")
    return rec


def soup_records(env: CoopEnv, ego: int) -> list:
    return [{"idx": s.idx, "pot": s.pot, "contents": [env.ing_name(c) for c in s.contents],
             "adds": [(t, "ego" if a == ego else "partner", env.ing_name(k)) for t, a, k in s.adds],
             "start_t": s.start_t, "full_t": s.full_t, "ready_t": s.ready_t, "plated_t": s.plated_t,
             "plated_by": None if s.plated_by is None else ("ego" if s.plated_by == ego else "partner"),
             "delivered_t": s.delivered_t, "correct": s.correct} for s in env.soups]


def write_jsonl(path, rec):
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Grid expansion
# --------------------------------------------------------------------------- #
def grid_units(cfg: dict) -> list:
    units = []
    for cell in cfg["cells"]:
        for cond in cfg["conditions"]:
            for pid in cell["personas"]:
                for ep in range(cfg["episodes_per_cell"]):
                    units.append((cell, cond, pid, ep, f"{cond['name']}|{cell['layout']}|{pid}|{ep}"))
    return units


def run_grid(cfg, personas, backends, out_dir, workers, done_ids=frozenset()):
    units = [u for u in grid_units(cfg) if u[4] not in done_ids]

    def one(u):
        cell, cond, pid, ep, _ = u
        return run_episode(cfg, cell, cond, pid, ep, backends, personas, out_dir)

    if workers <= 1:
        for u in units:
            _report(one(u))
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one, u) for u in units]):
            _report(f.result())


def _report(r):
    comp = r["compliance"]["rate"]
    print(f"{r['episode_id']}: return {r['return']:+.0f} (correct {r['n_correct']}, wrong {r['n_wrong']}), "
          f"compliance {'-' if comp is None else f'{comp:.2f}'}, infra ego/partner "
          f"{r['infra_rate']['ego']:.2f}/{r['infra_rate']['partner']:.2f} "
          f"(after retry {r['infra_after_retry']['ego']:.2f}/{r['infra_after_retry']['partner']:.2f}), "
          f"calls {r['usage']['ego']['calls']}/{r['usage']['partner']['calls']}, {r['wall_s']}s", flush=True)
