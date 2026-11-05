"""Run one experiment config: grid mode (cells x conditions x episodes, Exp A, M2) or stream mode
(memory kept across a scheduled sequence of episodes, Exp B/C).

  python -m research.coop_ci.run research/coop_ci/configs/m2_compliance.yaml --workers 8
  python -m research.coop_ci.run <config> --partner-backend scripted --episodes 1   # offline plumbing
  python -m research.coop_ci.run <config> --ego-backend mock --dry-run

Writes <out>/<name>__<partner>__<stamp>_<pid>/: episodes.jsonl, traces/, prompts/ (the full prompt
templates of every condition and persona used), config.resolved.json.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

# JAX must not grab the GPU vLLM is serving from; the env is tiny and runs on CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .env import CONFIG_DIR, CoopEnv  # noqa: E402
from .llm import OpenAICompatBackend  # noqa: E402
from .prompts import ego_system_prompt, human_system_prompt  # noqa: E402
from .runner import Backends, deep_update, grid_units, load_yaml, mismatched, run_grid  # noqa: E402
from .stream import run_streams, schedule  # noqa: E402


def load_config(path: str) -> dict:
    cfg = load_yaml(path)
    if "base" in cfg:
        base = load_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), cfg["base"]))
        cfg = deep_update(base, {k: v for k, v in cfg.items() if k != "base"})
    return cfg


def write_prompts(cfg: dict, personas: dict, out_dir: str):
    d = os.path.join(out_dir, "prompts")
    os.makedirs(d, exist_ok=True)
    for cell in cfg["cells"]:
        env = CoopEnv(cell["layout"], **{k: v for k, v in {**cfg.get("env", {}), **cell.get("env", {})}.items()
                                         if k != "layout"}).reset(0)
        for pid in cell["personas"]:
            with open(os.path.join(d, f"human__{cell.get('name', cell['layout'])}__{pid}.txt"), "w") as f:
                f.write(human_system_prompt(env, personas[pid]["text"]))
        for cond in cfg["conditions"]:
            if cond.get("ego", "llm") != "llm":
                continue
            c = cond.get("condition", "A0")
            pid = cell["personas"][0]
            label = {"A2": personas[pid]["label"], "A4": personas[mismatched(personas, pid)]["label"]}.get(c, "")
            with open(os.path.join(d, f"ego__{cond['name']}__{cell.get('name', cell['layout'])}.txt"), "w") as f:
                f.write(ego_system_prompt(env, c, cond.get("words", 80), "P-xxxx", label))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--backends", default=os.path.join(CONFIG_DIR, "backends.yaml"))
    ap.add_argument("--ego-backend", help="override backends.ego (e.g. frontier, mock)")
    ap.add_argument("--partner-backend", help="override backends.partner (scripted or an LLM backend)")
    ap.add_argument("--memory-backend", help="override backends.memory (the model that writes the notes)")
    ap.add_argument("--episodes", type=int, help="override episodes_per_cell (grid mode)")
    ap.add_argument("--streams", type=int, help="override the number of streams (stream mode)")
    ap.add_argument("--conditions", help="comma-separated subset of condition names")
    ap.add_argument("--layouts", help="comma-separated subset of cell layouts")
    ap.add_argument("--personas", help="comma-separated subset of personas")
    ap.add_argument("--max-steps", type=int, help="override env.max_steps")
    ap.add_argument("--workers", type=int, default=4, help="parallel episodes; vLLM batches concurrent requests")
    ap.add_argument("--out", default="outputs/coop_ci", help="parent folder for run directories")
    ap.add_argument("--resume", help="existing run dir: skip episodes already in its episodes.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the expanded plan and exit")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.ego_backend:  # the memory writer follows the ego unless it was set to something else
        if cfg["backends"].get("memory") in (None, cfg["backends"]["ego"]):
            cfg["backends"]["memory"] = a.ego_backend
        cfg["backends"]["ego"] = a.ego_backend
    if a.memory_backend:
        cfg["backends"]["memory"] = a.memory_backend
    if a.partner_backend:
        cfg["backends"]["partner"] = a.partner_backend
    if a.max_steps:
        cfg.setdefault("env", {})["max_steps"] = a.max_steps
    if a.conditions:
        keep = a.conditions.split(",")
        unknown = set(keep) - {c["name"] for c in cfg["conditions"]}
        if unknown:
            ap.error(f"unknown conditions {sorted(unknown)}")
        cfg["conditions"] = [c for c in cfg["conditions"] if c["name"] in keep]
    stream_mode = cfg.get("mode") == "stream"
    if a.streams:
        cfg["streams"] = a.streams
    if a.layouts and not stream_mode:
        cfg["cells"] = [c for c in cfg["cells"] if c["layout"] in a.layouts.split(",")]
    if a.personas and not stream_mode:
        keep = a.personas.split(",")
        cfg["cells"] = [dict(c, personas=[p for p in c["personas"] if p in keep]) for c in cfg["cells"]]
        cfg["cells"] = [c for c in cfg["cells"] if c["personas"]]
    if a.episodes:
        cfg["episodes_per_cell"] = a.episodes

    personas = load_yaml(os.path.join(CONFIG_DIR, cfg.get("personas_file", "personas.yaml")))
    used_personas = set(cfg["family_pair"]) if stream_mode else {p for c in cfg["cells"] for p in c["personas"]}
    missing = sorted(used_personas - set(personas))
    if missing:
        ap.error(f"unknown personas {missing}")
    specs = load_yaml(a.backends)
    if stream_mode:
        n_ep = len(schedule(cfg))
        print(f"{cfg['name']}: {cfg['streams']} streams x {len(cfg['conditions'])} memory conditions x {n_ep} "
              f"episodes = {cfg['streams'] * len(cfg['conditions']) * n_ep} episodes on {cfg['layout']}; "
              f"phases {[p['phase'] + 'x' + str(p['n']) for p in cfg['schedule']]}; ego={cfg['backends']['ego']}, "
              f"partner={cfg['backends']['partner']}", flush=True)
    else:
        n = len(grid_units(cfg))
        print(f"{cfg['name']}: {len(cfg['cells'])} cells x {len(cfg['conditions'])} conditions x "
              f"{cfg['episodes_per_cell']} episodes = {n} episodes, ego={cfg['backends']['ego']}, "
              f"partner={cfg['backends']['partner']}", flush=True)
    if a.dry_run:
        print(json.dumps(cfg, indent=2))
        return None

    backends = Backends(specs)
    used = set()
    if stream_mode or any(c.get("ego", "llm") == "llm" for c in cfg["conditions"]):
        used.add(cfg["backends"]["ego"])
    if stream_mode and any(c.get("memory") in ("general", "partner") for c in cfg["conditions"]):
        used.add(cfg["backends"].get("memory", cfg["backends"]["ego"]))
    if cfg["backends"]["partner"] != "scripted":
        used.add(cfg["backends"]["partner"])
    for name in sorted(used):
        b = backends.get(name)
        if isinstance(b, OpenAICompatBackend):  # fail before any episode when URL/model is wrong
            served = b.models()
            if b.model not in served:
                sys.exit(f"backend '{name}': model {b.model!r} not served at {b.base_url}; available {served}")

    if a.resume:
        out_dir = a.resume
        p = os.path.join(out_dir, "episodes.jsonl")
        done = {json.loads(l)["episode_id"] for l in open(p) if l.strip()} if os.path.exists(p) else set()
        print(f"resuming {out_dir}: {len(done)} episodes already done")
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(a.out, f"{cfg['name']}__{cfg['backends']['partner']}__{stamp}_{os.getpid()}")
        if os.path.exists(out_dir):
            sys.exit(f"{out_dir} exists; choose another --out")
        done = set()
    os.makedirs(out_dir, exist_ok=True)
    if not stream_mode:
        write_prompts(cfg, personas, out_dir)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out_dir, "config.resolved.json"), "w") as f:
        json.dump({"config": cfg, "backends": {k: specs.get(k) for k in used}, "personas": personas,
                   "chat_template_kwargs_override": os.environ.get("COOP_CHAT_TEMPLATE_KWARGS"),
                   "git_commit": commit, "argv": sys.argv}, f, indent=2)

    if stream_mode:
        run_streams(cfg, personas, backends, out_dir, a.workers, done)
    else:
        run_grid(cfg, personas, backends, out_dir, a.workers, done)
    print(f"done -> {out_dir}\nanalyse: python -m research.coop_ci.analyze {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
