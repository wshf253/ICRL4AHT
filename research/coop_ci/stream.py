"""Stream mode (Exp B/C): one ego keeps memory across a scheduled sequence of episodes.

A stream is a list of episodes, e.g. A A A | A-heldout x2 | B B B | A-return x2. Partners A and B
are two personas of one family (swapped on odd streams, so each persona is A in half the streams),
each with an opaque id. Everything except the memory condition is identical across conditions:
episode seeds (stream, index), seats, orders, partners. Episodes within a (stream, condition)
chain run in order; chains run in parallel. After each episode the memory is saved, so a stopped
run resumes from the last finished episode (`--resume`).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import LLMEgoPolicy
from .llm import Usage
from .memory import Memory, episode_summary
from .policy import stable_seed, trace_name
from .runner import Episode, deep_update, get_env, make_partner, partner_token, soup_records, write_jsonl


def schedule(cfg: dict, phases: list = None, k0: int = 0) -> list[dict]:
    """Expand a phase list into one entry per episode. `partner` names the partner id (A or B);
    `persona` (default: the same role) names whose persona that id plays, so Exp C can switch the
    persona behind an unchanged id."""
    out = []
    for ph in (phases if phases is not None else cfg["schedule"]):
        for j in range(ph["n"]):
            out.append({"k": k0 + len(out), "phase": ph["phase"], "partner": ph["partner"],
                        "persona": ph.get("persona", ph["partner"]), "j": j,
                        "update": bool(ph.get("update", False)),
                        "recipes": ph.get("recipes", cfg.get("train_recipes"))})
    return out


def stream_roles(cfg: dict, s: int) -> dict:
    a, b = cfg["family_pair"]
    if s % 2:
        a, b = b, a
    return {"A": a, "B": b}


def chain_dir(out_dir, cond, s):
    return os.path.join(out_dir, "memory", cond["name"], f"stream_{s}")


def make_memory(cfg, cond, backends) -> tuple:
    usage = Usage()
    mc = cfg.get("memory", {})
    mem = Memory(cond.get("memory", "none"), budget=mc.get("budget", 1500),
                 backend=backends.get(cfg["backends"].get("memory", cfg["backends"]["ego"])), usage=usage,
                 rule=cond.get("rule", "standard"), m=mc.get("m", 2))
    return mem, usage


def run_chain(cfg, cond, s, personas, backends, out_dir, done_ids, episodes=None, mem=None, mem_usage=None,
              extra=None) -> list:
    """Run the episodes of one (stream, condition) in order, saving the memory after each episode."""
    roles = stream_roles(cfg, s)
    ids = {r: partner_token(cfg.get("seed", 0), "stream", s, r) for r in roles}
    if mem is None:
        mem, mem_usage = make_memory(cfg, cond, backends)
    cdir = chain_dir(out_dir, cond, s)
    recs = []
    for ep in episodes if episodes is not None else schedule(cfg):
        k = ep["k"]
        eid = f"{cond['name']}|s{s}|{k:02d}|{ep['phase']}"
        state_path = os.path.join(cdir, f"after_{k:02d}.json")
        if eid in done_ids and os.path.exists(state_path):
            mem.set_state(json.load(open(state_path)))  # resume: memory as it was after this episode
            continue
        recs.append(run_stream_episode(cfg, cond, s, ep, roles, ids, personas, backends, mem, mem_usage,
                                       out_dir, eid, extra))
        mem.save(state_path)
    return recs


def run_stream_episode(cfg, cond, s, ep, roles, ids, personas, backends, mem, mem_usage, out_dir, eid,
                       extra=None) -> dict:
    t_start = time.time()
    pid = roles[ep.get("persona", ep["partner"])]
    partner_id = ids[ep["partner"]]
    env_cfg = deep_update(cfg.get("env", {}), {"layout": cfg["layout"]})
    if cfg.get("counterbalance_display") and (s // 2) % 2:
        # Streams 2-3, 6-7, ... list tomato first everywhere (crossed with which persona is A), so a
        # listing-order bias of the model cannot favour one persona.
        env_cfg["display_order"] = [1, 0]
    if ep["recipes"]:
        env_cfg["recipes"] = ep["recipes"]
    env = get_env(env_cfg)
    seed = stable_seed(cfg.get("seed", 0), "stream", s, ep["k"])
    env.reset(seed)
    seat = (s + ep["k"]) % 2
    usage = {"ego": Usage(), "partner": Usage()}
    mem_text = (f"Partner {partner_id}: {personas[pid]['label']}" if mem.kind == "oracle"
                else mem.render(partner_id))
    version_in = mem.state["version"]
    ego = LLMEgoPolicy(backends.get(cond.get("backend", cfg["backends"]["ego"])), usage["ego"],
                       condition=cfg.get("ego", {}).get("condition", "A0"), words=cfg.get("ego", {}).get("words", 80),
                       partner_id=partner_id, memory_text=mem_text)
    partner = make_partner(cfg["backends"]["partner"], pid, personas, backends, usage["partner"])
    epi = Episode(env, seed, seat, ego, partner, personas[pid],
                  replan_on_partner_event=cfg.get("replan_on_partner_event", True))
    try:
        epi.run(env.max_steps)
    finally:
        epi.close()
    label = f"E{ep['k'] + 1}"
    upd = {"status": "frozen (no update in this phase)"}
    if ep["update"]:
        summ = episode_summary(env, seat, partner_id, label)
        tokens0 = (mem_usage.prompt_tokens, mem_usage.completion_tokens)
        upd = mem.update(summ, partner_id, label, (cfg.get("seed", 0), cond["name"], s, ep["k"]))
        upd["summary"] = summ
        upd["update_tokens"] = (mem_usage.prompt_tokens - tokens0[0], mem_usage.completion_tokens - tokens0[1])
        os.makedirs(chain_dir(out_dir, cond, s), exist_ok=True)
        write_jsonl(os.path.join(chain_dir(out_dir, cond, s), "history.jsonl"), {"k": ep["k"], **upd})
    rec = {"episode_id": eid, "mode": "stream", "condition": cond["name"], "memory_kind": mem.kind,
           "memory_rule": mem.rule, **(extra or {}),
           "stream": s, "k": ep["k"], "phase": ep["phase"], "partner_role": ep["partner"],
           "persona_role": ep.get("persona", ep["partner"]), "persona": pid,
           "family": personas[pid].get("family"), "partner_id": partner_id, "layout": cfg["layout"],
           "recipes": ep["recipes"], "seed": seed, "ego_seat": seat, "ego_policy": ego.name,
           "partner_policy": partner.name, "env": env_cfg,
           "memory_in": {"version": version_in, "chars": len(mem_text), "text": mem_text},
           "memory_update": {k: v for k, v in upd.items() if k in ("status", "version_out", "chars", "update_tokens")},
           **epi.metrics(), "soups": soup_records(env, seat),
           "usage": {k: v.as_dict() for k, v in usage.items()}, "wall_s": round(time.time() - t_start, 1)}
    write_jsonl(os.path.join(out_dir, "episodes.jsonl"), rec)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    with open(os.path.join(out_dir, "traces", trace_name(eid) + ".jsonl"), "w") as f:
        for r in epi.trace:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"{eid} ({pid}): return {rec['return']:+.0f} (correct {rec['n_correct']}, wrong {rec['n_wrong']}), "
          f"memory in {len(mem_text)} chars v{version_in}, update {upd.get('status')}, {rec['wall_s']}s", flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Exp C: revision. One shared prefix per stream (the partner note is learnt once), then every
# regime x update rule continues from a clone of that note.
# --------------------------------------------------------------------------- #
def run_revision(cfg, personas, backends, out_dir, workers, done_ids=frozenset()):
    prefix_cond = {"name": "prefix", "memory": "partner", "rule": "standard"}
    prefix = schedule(cfg, cfg["prefix"])

    def do_prefix(s):
        mem, usage = make_memory(cfg, prefix_cond, backends)
        run_chain(cfg, prefix_cond, s, personas, backends, out_dir, done_ids, prefix, mem, usage,
                  {"regime": "prefix", "rule": "standard"})
        return s, mem.get_state()

    chains = [(regime, rule) for regime in cfg["regimes"] for rule in cfg["rules"]]

    def do_chain(s, state, regime, rule):
        cond = {"name": f"{regime}/{rule}", "memory": "partner", "rule": rule}
        mem, usage = make_memory(cfg, cond, backends)
        mem.set_state(state)  # clone of the prefix note
        eps = schedule(cfg, cfg["regimes"][regime], k0=len(prefix))
        return run_chain(cfg, cond, s, personas, backends, out_dir, done_ids, eps, mem, usage,
                         {"regime": regime, "rule": rule})

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        states = [f.result() for f in as_completed([pool.submit(do_prefix, s) for s in range(cfg["streams"])])]
        futs = [pool.submit(do_chain, s, st, regime, rule) for s, st in states for regime, rule in chains]
        for f in as_completed(futs):
            f.result()


def stream_units(cfg) -> list:
    return [(cond, s) for s in range(cfg["streams"]) for cond in cfg["conditions"]]


def run_streams(cfg, personas, backends, out_dir, workers, done_ids=frozenset()):
    units = stream_units(cfg)
    if workers <= 1:
        for cond, s in units:
            run_chain(cfg, cond, s, personas, backends, out_dir, done_ids)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_chain, cfg, cond, s, personas, backends, out_dir, done_ids) for cond, s in units]
        for f in as_completed(futs):
            f.result()
