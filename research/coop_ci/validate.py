"""Gate 0: is each scenario partner-sensitive? (scripted, no ego LLM; CPU)

  python -m research.coop_ci.validate research/coop_ci/configs/gate0_validation.yaml --workers 8
  python -m research.coop_ci.validate <config> --partner-backend qwen3.8-27b-partner   # LLM-human rollouts

For each layout x family x partner persona:
1. Matched states: run a warm-up ego (neutral) with the scripted twin from `seeds` seeds and
   snapshot at every t0 (episode start included). The ego seat alternates with the seed index.
2. Branch rollouts from every state: the ego follows one fixed allocation policy (each persona's
   jobs in the family, plus neutral) for `horizon` steps against the partner (scripted twin, or the
   LLM human with sampled rollouts). complementary(X) = the complement's jobs, conflicting(X) = X's.
3. Checks (paired bootstrap over states):
   allocation matters   complementary - conflicting return > 0, CI excluding 0, for every partner type
   partner-sensitive    the best allocation differs between the family's partner types
   horizon              >= 2 correct dishes in `horizon` steps under the complementary policy
Also a scripted-pair sanity check (M1/M2 acceptance): complementary ego + twin for a full episode
(dishes, deadlocks, infrastructure-failure rate, persona compliance of the twin).

Writes <out>/gate0_<stamp>/{report.md, rollouts.jsonl, sanity.jsonl, config.resolved.json}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from statistics import mean

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .env import CONFIG_DIR, CoopEnv  # noqa: E402
from .llm import Usage  # noqa: E402
from .policy import stable_seed  # noqa: E402
from .runner import Backends, Episode, load_yaml, make_partner  # noqa: E402
from .scripted import ego_policy, persona_policy  # noqa: E402
from .stats import boot_ci, fmt, paired_diff  # noqa: E402

_TLS = threading.local()


def _env(layout: str, max_steps: int) -> CoopEnv:
    """One env per (thread, layout): LLM-partner units run in threads and must not share one."""
    envs = getattr(_TLS, "envs", None)
    if envs is None:
        envs = _TLS.envs = {}
    key = (layout, max_steps)
    if key not in envs:
        envs[key] = CoopEnv(layout, max_steps=max_steps)
    return envs[key]


def family_personas(personas: dict, fam: str) -> list[str]:
    return sorted(k for k, v in personas.items() if v.get("family") == fam)


# --------------------------------------------------------------------------- #
# Work units
# --------------------------------------------------------------------------- #
def gate0_unit(u: dict) -> list[dict]:
    """All branch rollouts from the states of one (layout, family, persona, seed)."""
    personas = u["personas"]
    env = _env(u["layout"], u["max_steps"])
    seed = stable_seed(u["seed0"], "gate0", u["seed_idx"])
    env.reset(seed)
    seat = u["seed_idx"] % 2
    backends = Backends(u["backend_specs"]) if u["partner"] != "scripted" else None
    usage = Usage()
    pid = u["persona"]
    epi = Episode(env, seed, seat, ego_policy(personas, "neutral"),
                  make_partner(u["partner"], pid, personas, backends, usage), personas[pid], record_trace=False)
    snaps = []
    for t0 in sorted(u["t0"]):
        epi.run(t0)
        snaps.append((t0, epi.snapshot()))
    allocs = [f"as:{p}" for p in family_personas(personas, u["family"])] + ["neutral"]
    out = []
    for t0, snap in snaps:
        for alloc in allocs:
            for r in range(u["rollouts"]):
                epi.restore(snap)
                for e in epi.ex.values():
                    e.salt = seed * 7 + 1000 * (r + 1)
                if u["partner"] != "scripted":
                    epi.pols[epi.partner].salt = r + 1   # sampled LLM-human rollouts
                epi.replace_policy(seat, ego_policy(personas, alloc), salt=r)
                epi.run(t0 + u["horizon"])
                m = epi.metrics()
                out.append({"layout": u["layout"], "family": u["family"], "persona": pid, "seed_idx": u["seed_idx"],
                            "t0": t0, "state": f"{u['seed_idx']}@{t0}", "alloc": alloc[3:] if alloc.startswith("as:") else alloc,
                            "rollout": r, "ego_seat": seat, **m})
    epi.close()
    return out


def sanity_unit(u: dict) -> dict:
    personas, pid = u["personas"], u["persona"]
    env = _env(u["layout"], u["steps"])
    seed = stable_seed(u["seed0"], "sanity", u["seed_idx"])
    env.reset(seed)
    seat = u["seed_idx"] % 2
    epi = Episode(env, seed, seat, ego_policy(personas, "complementary", pid), persona_policy(personas, pid),
                  personas[pid], record_trace=False)
    epi.run(u["steps"])
    return {"layout": u["layout"], "family": u["family"], "persona": pid, "seed_idx": u["seed_idx"], **epi.metrics()}


def _pool(fn, units, workers, threads=False):
    if workers <= 1:
        return [fn(u) for u in units]
    if threads:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, units))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, units, chunksize=1))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def per_state(rows, key="return") -> dict:
    """(layout, family, persona, alloc) -> {state: mean over rollouts}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["layout"], r["family"], r["persona"], r["alloc"])][r["state"]].append(r[key])
    return {k: {s: mean(v) for s, v in d.items()} for k, d in acc.items()}


def report(rows, sanity, cfg, personas) -> tuple[str, list]:
    ret, dishes = per_state(rows, "return"), per_state(rows, "n_correct")
    lines = [f"# Gate 0: scenario validation ({cfg['name']})", "",
             f"Partner: `{cfg['partner']}`. States per persona: {cfg['seeds']} seeds x t0 {cfg['t0']}; "
             f"{cfg['rollouts']} rollouts per state; horizon {cfg['horizon']} steps. Return = native reward "
             "(+20 correct, -20 wrong). CIs: paired bootstrap over states.", ""]
    verdicts = []
    for layout, fam in sorted({(r["layout"], r["family"]) for r in rows}):
        fps = family_personas(personas, fam)
        allocs = fps + ["neutral"]
        lines += [f"## {layout} / {fam}", "", "Mean return per state (dishes) by ego allocation:", "",
                  "| partner \\ ego | " + " | ".join(allocs) + " |", "|---|" + "---|" * len(allocs)]
        for pid in fps:
            cells = []
            for a in allocs:
                v = ret.get((layout, fam, pid, a), {})
                dv = dishes.get((layout, fam, pid, a), {})
                cells.append(f"{fmt(mean(v.values()))} ({fmt(mean(dv.values()), 2)})" if v else "-")
            lines.append(f"| {pid} | " + " | ".join(cells) + " |")
        lines += ["", "| partner | complementary | conflicting | comp - conf (95% CI) | dishes (comp) | allocation matters | horizon ok |",
                  "|---|---|---|---|---|---|---|"]
        ok_alloc, ok_hor, best = True, True, {}
        for pid in fps:
            comp, conf = personas[pid]["complement"], pid
            a, b = ret[(layout, fam, pid, comp)], ret[(layout, fam, pid, conf)]
            d, (lo, hi), n = paired_diff(a, b)
            dish = mean(dishes[(layout, fam, pid, comp)].values())
            m_ok, h_ok = lo > 0, dish >= 2
            ok_alloc &= m_ok
            ok_hor &= h_ok
            best[pid] = max(fps, key=lambda q: mean(ret[(layout, fam, pid, q)].values()))
            lines.append(f"| {pid} | {fmt(mean(a.values()))} | {fmt(mean(b.values()))} | {fmt(d)} [{fmt(lo)}, {fmt(hi)}] (n={n}) "
                         f"| {fmt(dish, 2)} | {'yes' if m_ok else 'NO'} | {'yes' if h_ok else 'NO'} |")
        sens = len(set(best.values())) == len(fps)
        passed = ok_alloc and sens and ok_hor
        lines += ["", f"Best allocation per partner type: {best} -> partner-sensitive: {'yes' if sens else 'NO'}.",
                  f"**Gate 0 {'PASSED' if passed else 'FAILED'}** for {layout} / {fam}.", ""]
        # Waste/delay diagnostics, complementary vs conflicting
        diag = defaultdict(list)
        for r in rows:
            if r["layout"] == layout and r["family"] == fam:
                tag = "comp" if r["alloc"] == personas[r["persona"]]["complement"] else ("conf" if r["alloc"] == r["persona"] else None)
                if tag:
                    diag[tag].append(r)
        lines += ["| ego | dup ingredient | surplus plate | wrong-soup adds | ready-idle steps | ego idle while useful | blocking rate | dup subtask rate |",
                  "|---|---|---|---|---|---|---|---|"]
        for tag in ("comp", "conf"):
            rs = diag[tag]
            if rs:
                lines.append(f"| {tag} | {fmt(mean(r['waste']['dup_ingredient'] for r in rs), 2)} | "
                             f"{fmt(mean(r['waste']['surplus_plate'] for r in rs), 2)} | "
                             f"{fmt(mean(r['waste']['wrong_soup_add']['ego'] + r['waste']['wrong_soup_add']['partner'] for r in rs), 2)} | "
                             f"{fmt(mean(r['ready_idle_steps'] for r in rs))} | {fmt(mean(r['ego_idle_while_useful'] for r in rs))} | "
                             f"{fmt(mean(r['blocking_rate'] for r in rs), 3)} | {fmt(mean(r['dup_subtask_rate'] for r in rs), 3)} |")
        lines.append("")
        verdicts.append({"layout": layout, "family": fam, "passed": passed, "allocation_matters": ok_alloc,
                         "partner_sensitive": sens, "horizon_ok": ok_hor, "best": best})
    if sanity:
        lines += ["## Scripted-pair sanity (complementary ego + twin, full episodes)", "",
                  "| layout | persona | dishes mean (min) | wrong | deadlocks | infra rate ego/partner | twin compliance |",
                  "|---|---|---|---|---|---|---|"]
        g = defaultdict(list)
        for s in sanity:
            g[(s["layout"], s["persona"])].append(s)
        for (layout, pid), rs in sorted(g.items()):
            opp = sum(r["compliance"]["opportunities"] for r in rs)
            mat = sum(r["compliance"]["matches"] for r in rs)
            lines.append(f"| {layout} | {pid} | {fmt(mean(r['n_correct'] for r in rs), 2)} ({min(r['n_correct'] for r in rs)}) | "
                         f"{sum(r['n_wrong'] for r in rs)} | {sum(r['deadlocks'] for r in rs)} | "
                         f"{fmt(mean(r['infra_rate']['ego'] for r in rs), 3)}/{fmt(mean(r['infra_rate']['partner'] for r in rs), 3)} | "
                         f"{fmt(mat / opp if opp else None, 3)} ({mat}/{opp}) |")
        lines.append("")
    lines += ["## Decision rule", "",
              "A family that fails on a layout is not used for Exp A on that layout (IMPLEMENTATION_PLAN M3):",
              "change the layout, menu or horizon and rerun.", ""]
    for v in verdicts:
        lines.append(f"- {v['layout']} / {v['family']}: {'use' if v['passed'] else 'do not use'}"
                     + ("" if v["passed"] else f" (allocation matters: {v['allocation_matters']}, partner-sensitive: "
                        f"{v['partner_sensitive']}, horizon: {v['horizon_ok']})"))
    return "\n".join(lines) + "\n", verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--backends", default=os.path.join(CONFIG_DIR, "backends.yaml"))
    ap.add_argument("--partner-backend", help="scripted (default) or an LLM backend for the LLM human")
    ap.add_argument("--layouts", help="comma-separated subset")
    ap.add_argument("--families", help="comma-separated subset")
    ap.add_argument("--seeds", type=int, help="override number of seeds")
    ap.add_argument("--rollouts", type=int, help="override rollouts per state")
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/coop_ci")
    a = ap.parse_args(argv)

    cfg = load_yaml(a.config)
    personas = load_yaml(os.path.join(CONFIG_DIR, cfg.get("personas_file", "personas.yaml")))
    cfg["partner"] = a.partner_backend or cfg.get("partner", "scripted")
    cfg["seeds"] = a.seeds or cfg["seeds"]
    cfg["rollouts"] = a.rollouts or cfg["rollouts"]
    fams = cfg["families"]
    if a.families:
        fams = {k: v for k, v in fams.items() if k in a.families.split(",")}
    specs = load_yaml(a.backends) if cfg["partner"] != "scripted" else {}
    units, sanity_units = [], []
    for fam, fcfg in fams.items():
        layouts = [l for l in fcfg["layouts"] if not a.layouts or l in a.layouts.split(",")]
        for layout in layouts:
            for pid in family_personas(personas, fam):
                for s in range(cfg["seeds"]):
                    units.append({"layout": layout, "family": fam, "persona": pid, "seed_idx": s, "seed0": cfg.get("seed", 0),
                                  "t0": cfg["t0"], "horizon": cfg["horizon"], "rollouts": cfg["rollouts"],
                                  "max_steps": max(cfg["t0"]) + cfg["horizon"], "partner": cfg["partner"],
                                  "personas": personas, "backend_specs": specs})
                if not a.no_sanity and cfg.get("sanity"):
                    for s in range(cfg["sanity"]["seeds"]):
                        sanity_units.append({"layout": layout, "family": fam, "persona": pid, "seed_idx": s,
                                             "seed0": cfg.get("seed", 0), "steps": cfg["sanity"]["steps"], "personas": personas})
    n_roll = len(units) * len(cfg["t0"]) * 3 * cfg["rollouts"]
    print(f"Gate 0: {len(units)} (layout, persona, seed) units, ~{n_roll} branch rollouts of {cfg['horizon']} steps; "
          f"partner={cfg['partner']}; sanity episodes {len(sanity_units)}", flush=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(a.out, f"{cfg['name']}__{cfg['partner']}__{stamp}_{os.getpid()}")
    os.makedirs(out_dir, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out_dir, "config.resolved.json"), "w") as f:
        json.dump({"config": cfg, "families": fams, "git_commit": commit, "argv": sys.argv,
                   "personas": personas, "backend": specs.get(cfg["partner"])}, f, indent=2)
    threads = cfg["partner"] != "scripted"   # LLM rollouts are I/O bound
    rows = [r for rs in _pool(gate0_unit, units, a.workers, threads) for r in rs]
    with open(os.path.join(out_dir, "rollouts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    sanity = _pool(sanity_unit, sanity_units, a.workers) if sanity_units else []
    with open(os.path.join(out_dir, "sanity.jsonl"), "w") as f:
        for r in sanity:
            f.write(json.dumps(r, default=str) + "\n")
    text, verdicts = report(rows, sanity, cfg, personas)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"-> {out_dir}/report.md")
    return out_dir, verdicts


if __name__ == "__main__":
    main()
