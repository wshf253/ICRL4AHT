"""Summarise a run directory (episodes.jsonl) into the tables the RQs need.

  python -m research.coop_tom.analyze outputs/coop_tom/<run_dir> [--ref hid_hid]

Writes <run_dir>/analysis/*.csv and prints markdown tables:
  summary        per condition x persona [x phase]: score (95% bootstrap CI), deliveries, commit timing, probes, cost
  paired         per condition vs --ref on matched episodes (same seed): mean difference (95% CI)
  q0b            commit-time decomposition: inference error vs utilization error
  phases         failures by phase (ingredient/parallel vs cook_serve/serial) and waiting steps
  curve          stream runs: score by episode index within each phase
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from statistics import mean


def boot_ci(xs, n=2000, seed=0):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def fmt(x, d=2):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def phase_of(eid):
    parts = eid.split("|")
    return parts[1] if len(parts) == 5 else ""


def match_key(r):
    parts = r["episode_id"].split("|")
    return "|".join(parts[1:])  # everything except the condition name


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", help="reference condition for paired differences")
    a = ap.parse_args()
    eps = [json.loads(l) for l in open(os.path.join(a.run_dir, "episodes.jsonl")) if l.strip()]
    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)

    # ---------------- summary ---------------- #
    groups = defaultdict(list)
    for r in eps:
        groups[(r["condition"], r["persona"], phase_of(r["episode_id"]))].append(r)
    summ = []
    for (c, p, ph), rs in sorted(groups.items()):
        sc = [r["score"] for r in rs]
        lo, hi = boot_ci(sc)
        soups = [s for r in rs for s in r["soups"]]
        pr = [r["probe_summary"] for r in rs if r.get("probe_summary")]
        summ.append({
            "condition": c, "persona": p, "phase": ph, "n": len(rs),
            "score": fmt(mean(sc), 1), "ci95": f"[{fmt(lo, 1)}, {fmt(hi, 1)}]",
            "correct": fmt(mean(r["n_correct"] for r in rs)), "wrong": fmt(mean(r["n_wrong"] for r in rs)),
            "premature": fmt(mean_or_none([float(s["premature_commit"]) for s in soups if s["premature_commit"] is not None])),
            "over_wait": fmt(mean_or_none([s["over_wait"] for s in soups]), 1),
            "agent_wrong_add": fmt(mean_or_none([float(s["agent_wrong_add"]) for s in soups])),
            "probe_top1": fmt(mean_or_none([x.get("top1_acc") for x in pr])),
            "probe_p_true": fmt(mean_or_none([x.get("p_true") for x in pr])),
            "violation": fmt(mean_or_none([x.get("violation_mass") for x in pr])),
            "persona_acc": fmt(mean_or_none([x.get("partner_type_acc") for x in pr])),
            "agent_calls": fmt(mean(r["usage"]["agent"]["calls"] for r in rs), 0),
            "agent_tok": fmt(mean(r["usage"]["agent"]["prompt_tokens"] + r["usage"]["agent"]["completion_tokens"] for r in rs), 0),
            "parse_fail": fmt(mean(r["parse_fail"]["agent"] / max(1, r["decisions"]["agent"]) for r in rs), 3),
            "thinking": fmt(mean(r["usage"]["agent"].get("thinking", 0) for r in rs), 1),
            "truncated": fmt(mean(r["usage"]["agent"].get("truncated", 0) for r in rs), 1),
            "pot_changed": fmt(mean(r.get("pot_changed_replans", {}).get("agent", 0) for r in rs), 1),
            "wall_s": fmt(mean(r["wall_s"] for r in rs), 0),
            "partner_adherence": json.dumps({k: v for k, v in rs[0].get("adherence", {}).items()}) if rs else "",
        })
    write_csv(os.path.join(out, "summary.csv"), summ)
    print("## Summary\n" + md(summ, ["condition", "persona", "phase", "n", "score", "ci95", "correct", "wrong",
                                     "premature", "over_wait", "agent_wrong_add", "probe_top1", "violation",
                                     "persona_acc", "agent_calls", "parse_fail", "thinking", "truncated",
                                     "pot_changed", "wall_s"]))

    # ---------------- paired ---------------- #
    if a.ref:
        by = defaultdict(dict)
        for r in eps:
            by[match_key(r)][r["condition"]] = r["score"]
        rows = []
        for c in sorted({r["condition"] for r in eps} - {a.ref}):
            diffs = [v[c] - v[a.ref] for v in by.values() if c in v and a.ref in v]
            if diffs:
                lo, hi = boot_ci(diffs)
                rows.append({"condition": c, "vs": a.ref, "n_pairs": len(diffs), "mean_diff": fmt(mean(diffs), 1),
                             "ci95": f"[{fmt(lo, 1)}, {fmt(hi, 1)}]"})
        write_csv(os.path.join(out, "paired.csv"), rows)
        print("\n## Paired differences (same seeds)\n" + md(rows, ["condition", "vs", "n_pairs", "mean_diff", "ci95"]))

    # ---------------- Q0-b ---------------- #
    q = defaultdict(lambda: defaultdict(int))
    for r in eps:
        for s in r["soups"]:
            pac = s.get("probe_at_commit")
            if s["agent_commit_t"] is None or pac is None:
                continue
            ok = bool(s["correct"]) and not s["agent_wrong_add"]
            key = "probe_right_soup_ok" if pac["top1_correct"] and ok else (
                "utilization_error" if pac["top1_correct"] else ("inference_error" if not ok else "probe_wrong_soup_ok"))
            q[(r["condition"], r["persona"])][key] += 1
    qrows = [{"condition": c, "persona": p, **{k: d.get(k, 0) for k in
              ["probe_right_soup_ok", "utilization_error", "inference_error", "probe_wrong_soup_ok"]}}
             for (c, p), d in sorted(q.items())]
    write_csv(os.path.join(out, "q0b.csv"), qrows)
    if qrows:
        print("\n## Q0-b: commit-time decomposition (soups where the agent added an ingredient)\n"
              + md(qrows, list(qrows[0].keys())))

    # ---------------- phases (Q0-c) ---------------- #
    prow = []
    for (c, p, ph), rs in sorted(groups.items()):
        soups = [s for r in rs for s in r["soups"]]
        prow.append({
            "condition": c, "persona": p, "phase": ph,
            "parallel_fail_wrong_add": sum(s["agent_wrong_add"] for s in soups),
            "serial_fail_full_not_delivered": sum(1 for s in soups if s["full_t"] is not None and s["delivered_t"] is None),
            "wrong_delivered": sum(1 for s in soups if s["correct"] is False),
            "agent_wait_ingredient": fmt(mean(r["wait_steps"]["agent"]["ingredient"] for r in rs), 1),
            "agent_wait_cook_serve": fmt(mean(r["wait_steps"]["agent"]["cook_serve"] for r in rs), 1),
        })
    write_csv(os.path.join(out, "phases.csv"), prow)
    print("\n## Q0-c: failures by phase\n" + md(prow, list(prow[0].keys())))

    # ---------------- robot action mix (from traces) ---------------- #
    verbs = ["fetch", "put", "plate", "deliver", "place", "pick", "move", "wait"]
    mix = defaultdict(lambda: defaultdict(int))
    for r in eps:
        safe = r["episode_id"].replace("|", "__").replace("/", "_")
        path = os.path.join(a.run_dir, "traces", f"{safe}.jsonl")
        if not os.path.exists(path):
            continue
        g = mix[(r["condition"], r["persona"])]
        for line in open(path):
            d = json.loads(line)
            if d.get("type") != "decision" or d.get("role") != "agent":
                continue
            g["decisions"] += 1
            g[d["action"].split()[0]] += 1
            g["could_not_start"] += "failed" in d
    arows = []
    for (c, p), g in sorted(mix.items()):
        n = g["decisions"] or 1
        arows.append({"condition": c, "persona": p, "decisions": g["decisions"],
                      **{f"{v}%": fmt(100 * g[v] / n, 0) for v in verbs},
                      "could_not_start%": fmt(100 * g["could_not_start"] / n, 0)})
    if arows:
        write_csv(os.path.join(out, "actions.csv"), arows)
        print("\n## Robot action mix (% of decisions; could_not_start = precondition failed)\n"
              + md(arows, list(arows[0].keys())))

    # ---------------- why robot actions could not start ---------------- #
    fails = defaultdict(lambda: defaultdict(int))
    for r in eps:
        safe = r["episode_id"].replace("|", "__").replace("/", "_")
        path = os.path.join(a.run_dir, "traces", f"{safe}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            d = json.loads(line)
            if d.get("type") == "decision" and d.get("role") == "agent" and "failed" in d:
                key = (d["action"].split()[0] if not d["action"].startswith(("place", "pick", "move"))
                       else " ".join(d["action"].split()[:2]), d["failed"],
                       d.get("holding", "?"), d.get("pot", "?").split(" [")[0])
                fails[(r["condition"], r["persona"])][key] += 1
    frows = []
    for (c, p), g in sorted(fails.items()):
        total = sum(g.values())
        for (act, why, holding, pot), n in sorted(g.items(), key=lambda kv: -kv[1])[:8]:
            frows.append({"condition": c, "persona": p, "action": act, "reason": why, "holding": holding,
                          "pot": pot, "n": n, "share%": fmt(100 * n / total, 0)})
    if frows:
        write_csv(os.path.join(out, "start_failures.csv"), frows)
        print("\n## Robot actions that could not start (top reasons; holding/pot logged from this version on)\n"
              + md(frows, list(frows[0].keys())))

    # ---------------- learning curves (stream) ---------------- #
    if any(phase_of(r["episode_id"]) for r in eps):
        curve = defaultdict(list)
        order = defaultdict(int)
        for r in eps:  # episodes.jsonl is appended in run order within a condition
            c, ph = r["condition"], phase_of(r["episode_id"])
            idx = order[(c, ph)]
            order[(c, ph)] += 1
            curve[(c, ph, idx)].append(r["score"])
        crow = [{"condition": c, "phase": ph, "episode_idx": i, "score": fmt(mean(v), 1)} for (c, ph, i), v in sorted(curve.items())]
        write_csv(os.path.join(out, "curve.csv"), crow)
        print(f"\nlearning curve -> {os.path.join(out, 'curve.csv')}")


if __name__ == "__main__":
    main()
