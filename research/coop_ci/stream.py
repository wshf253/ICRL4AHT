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
from .policy import stable_seed
from .runner import Episode, deep_update, get_env, make_partner, partner_token, soup_records, write_jsonl


def schedule(cfg: dict) -> list[dict]:
    """Expand the phase list into one entry per episode."""
    out = []
    for ph in cfg["schedule"]:
        for j in range(ph["n"]):
            out.append({"k": len(out), "phase": ph["phase"], "partner": ph["partner"], "j": j,
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


def run_chain(cfg, cond, s, personas, backends, out_dir, done_ids) -> list:
    roles = stream_roles(cfg, s)
    ids = {r: partner_token(cfg.get("seed", 0), "stream", s, r) for r in roles}
    mem_usage = Usage()
    mem = Memory(cond.get("memory", "none"), budget=cfg.get("memory", {}).get("budget", 1500),
                 backend=backends.get(cfg["backends"].get("memory", cfg["backends"]["ego"])), usage=mem_usage)
    cdir = chain_dir(out_dir, cond, s)
    recs = []
    for ep in schedule(cfg):
        k = ep["k"]
        eid = f"{cond['name']}|s{s}|{k:02d}|{ep['phase']}"
        state_path = os.path.join(cdir, f"after_{k:02d}.json")
        if eid in done_ids and os.path.exists(state_path):
            mem.set_state(json.load(open(state_path)))  # resume: memory as it was after this episode
            continue
        recs.append(run_stream_episode(cfg, cond, s, ep, roles, ids, personas, backends, mem, mem_usage,
                                       out_dir, eid))
        mem.save(state_path)
    return recs


def run_stream_episode(cfg, cond, s, ep, roles, ids, personas, backends, mem, mem_usage, out_dir, eid) -> dict:
    t_start = time.time()
    pid = roles[ep["partner"]]
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
           "stream": s, "k": ep["k"], "phase": ep["phase"], "partner_role": ep["partner"], "persona": pid,
           "family": personas[pid].get("family"), "partner_id": partner_id, "layout": cfg["layout"],
           "recipes": ep["recipes"], "seed": seed, "ego_seat": seat, "ego_policy": ego.name,
           "partner_policy": partner.name, "env": env_cfg,
           "memory_in": {"version": version_in, "chars": len(mem_text), "text": mem_text},
           "memory_update": {k: v for k, v in upd.items() if k in ("status", "version_out", "chars", "update_tokens")},
           **epi.metrics(), "soups": soup_records(env, seat),
           "usage": {k: v.as_dict() for k, v in usage.items()}, "wall_s": round(time.time() - t_start, 1)}
    write_jsonl(os.path.join(out_dir, "episodes.jsonl"), rec)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    with open(os.path.join(out_dir, "traces", eid.replace("|", "__") + ".jsonl"), "w") as f:
        for r in epi.trace:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"{eid} ({pid}): return {rec['return']:+.0f} (correct {rec['n_correct']}, wrong {rec['n_wrong']}), "
          f"memory in {len(mem_text)} chars v{version_in}, update {upd.get('status')}, {rec['wall_s']}s", flush=True)
    return rec


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
