"""Run one cooperative-intelligence experiment config.

  python -m research.coop_tom.run research/coop_tom/configs/exp0_smoke.yaml --workers 8
  python -m research.coop_tom.run <config> --agent-backend mock --episodes 1     # offline plumbing
  python -m research.coop_tom.run <config> --agent-backend frontier              # stage 2
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

# JAX must not grab the GPU vLLM is serving from; the env is tiny and runs on CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .llm import OpenAICompatBackend  # noqa: E402
from .runner import Backends, deep_update, load_yaml, run_grid  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def load_config(path: str) -> dict:
    cfg = load_yaml(path)
    if "base" in cfg:
        base = load_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), cfg["base"]))
        cfg = deep_update(base, {k: v for k, v in cfg.items() if k != "base"})
    return cfg


def plan(cfg: dict) -> int:
    return len(cfg["conditions"]) * len(cfg["personas"]) * cfg["episodes_per_cell"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--backends", default=os.path.join(CONFIG_DIR, "backends.yaml"))
    ap.add_argument("--agent-backend", help="override backends.agent (e.g. frontier, mock)")
    ap.add_argument("--partner-backend", help="override backends.partner (scripted or an LLM backend)")
    ap.add_argument("--episodes", type=int, help="override episodes_per_cell")
    ap.add_argument("--conditions", help="comma-separated subset of condition names")
    ap.add_argument("--personas", help="comma-separated subset of personas")
    ap.add_argument("--max-steps", type=int, help="override env.max_steps")
    ap.add_argument("--workers", type=int, default=4, help="parallel episodes; vLLM batches concurrent requests")
    ap.add_argument("--out", default="outputs/coop_tom", help="parent folder for run directories")
    ap.add_argument("--resume", help="existing run dir: skip episodes already in its episodes.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the expanded plan and exit")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.agent_backend:
        cfg["backends"]["agent"] = a.agent_backend
    if a.partner_backend:
        cfg["backends"]["partner"] = a.partner_backend
    if a.max_steps:
        cfg["env"]["max_steps"] = a.max_steps
    if a.conditions:
        keep = a.conditions.split(",")
        unknown = set(keep) - {c["name"] for c in cfg["conditions"]}
        if unknown:
            ap.error(f"unknown conditions {sorted(unknown)}")
        cfg["conditions"] = [c for c in cfg["conditions"] if c["name"] in keep]
    if a.personas:
        cfg["personas"] = a.personas.split(",")
    if a.episodes:
        cfg["episodes_per_cell"] = a.episodes
    if cfg.get("mode", "grid") != "grid":
        ap.error("only grid mode is implemented in this version")

    personas = load_yaml(os.path.join(CONFIG_DIR, cfg.get("personas_file", "personas.yaml")))
    missing = [p for p in cfg["personas"] if p not in personas]
    if missing:
        ap.error(f"unknown personas {missing}")
    specs = load_yaml(a.backends)
    print(f"{cfg['name']}: {len(cfg['conditions'])} conditions x {len(cfg['personas'])} personas x "
          f"{cfg['episodes_per_cell']} episodes = {plan(cfg)} episodes, agent={cfg['backends']['agent']}, "
          f"partner={cfg['backends']['partner']}, max_steps={cfg['env']['max_steps']}", flush=True)
    if a.dry_run:
        print(json.dumps(cfg, indent=2))
        return None

    backends = Backends(specs)
    llm_used = any(c.get("policy") != "scripted_cook" for c in cfg["conditions"])
    for role in ("agent", "partner"):
        name = cfg["backends"][role]
        if name == "scripted" or (role == "agent" and not llm_used):
            continue
        b = backends.get(name)
        if isinstance(b, OpenAICompatBackend):  # fail before any episode when URL/model is wrong
            served = b.models()
            if b.model not in served:
                sys.exit(f"{role} backend '{name}': model {b.model!r} not served at {b.base_url}; available {served}")

    if a.resume:
        out_dir = a.resume
        p = os.path.join(out_dir, "episodes.jsonl")
        done = {json.loads(l)["episode_id"] for l in open(p) if l.strip()} if os.path.exists(p) else set()
        print(f"resuming {out_dir}: {len(done)} episodes already done")
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(a.out, f"{cfg['name']}__{cfg['backends']['agent']}__{stamp}_{os.getpid()}")
        if os.path.exists(out_dir):
            sys.exit(f"{out_dir} exists; choose another --out")
        done = set()
    os.makedirs(out_dir, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    used = {cfg["backends"]["agent"], cfg["backends"]["partner"]}
    with open(os.path.join(out_dir, "config.resolved.json"), "w") as f:
        json.dump({"config": cfg, "backends": {k: specs.get(k) for k in used if k in specs},
                   "chat_template_kwargs_override": os.environ.get("COOP_CHAT_TEMPLATE_KWARGS"),
                   "git_commit": commit, "argv": sys.argv}, f, indent=2)

    run_grid(cfg, personas, backends, out_dir, a.workers, done)
    print(f"done -> {out_dir}\nanalyse: python -m research.coop_tom.analyze {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
