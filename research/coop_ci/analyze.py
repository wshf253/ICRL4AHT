"""Summarise a grid run directory (episodes.jsonl).

  python -m research.coop_ci.analyze outputs/coop_ci/<run_dir> [--ref A0]

Prints markdown tables and writes <run_dir>/analysis/*.csv:
  summary     per condition x layout x persona: return (95% bootstrap CI over episodes), dishes,
              partner compliance, infrastructure-failure rate (flagged above 2% of decisions),
              dependency delay, coordination waste, cost
  paired      per condition vs --ref on matched episodes (same layout, persona, episode index)
  failures    failure codes per role (pot_changed / taken_by_partner are not infrastructure)
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from statistics import mean

from .stats import boot_ci, fmt, paired_diff

INFRA_FLAG = 0.02


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def summary_rows(eps):
    g = defaultdict(list)
    for r in eps:
        g[(r["condition"], r["layout"], r["persona"])].append(r)
    rows = []
    for (c, lay, p), rs in sorted(g.items()):
        ret = [r["return"] for r in rs]
        lo, hi = boot_ci(ret)
        opp = sum(r["compliance"]["opportunities"] for r in rs)
        mat = sum(r["compliance"]["matches"] for r in rs)
        dec_e = sum(r["decisions"]["ego"] for r in rs)
        dec_p = sum(r["decisions"]["partner"] for r in rs)
        inf_e = sum(r["infra_rate"]["ego"] * r["decisions"]["ego"] for r in rs) / max(1, dec_e)
        inf_p = sum(r["infra_rate"]["partner"] * r["decisions"]["partner"] for r in rs) / max(1, dec_p)
        flag = " !" if max(inf_e, inf_p) > INFRA_FLAG else ""
        rows.append({
            "condition": c, "layout": lay, "persona": p, "n": len(rs),
            "return": f"{fmt(mean(ret))} [{fmt(lo)}, {fmt(hi)}]",
            "correct": fmt(mean(r["n_correct"] for r in rs), 2), "wrong": fmt(mean(r["n_wrong"] for r in rs), 2),
            "compliance": fmt(mat / opp if opp else None, 3),
            "infra_ego/partner": f"{fmt(inf_e, 3)}/{fmt(inf_p, 3)}{flag}",
            "ego_idle_useful": fmt(mean(r["ego_idle_while_useful"] for r in rs)),
            "ready_idle": fmt(mean(r["ready_idle_steps"] for r in rs)),
            "dup_subtask": fmt(mean(r["dup_subtask_rate"] for r in rs), 3),
            "blocking": fmt(mean(r["blocking_rate"] for r in rs), 3),
            "dup_ing": fmt(mean(r["waste"]["dup_ingredient"] for r in rs), 2),
            "wrong_adds": fmt(mean(r["waste"]["wrong_soup_add"]["ego"] + r["waste"]["wrong_soup_add"]["partner"] for r in rs), 2),
            "calls_ego/partner": f"{fmt(mean(r['usage']['ego']['calls'] for r in rs), 0)}/{fmt(mean(r['usage']['partner']['calls'] for r in rs), 0)}",
            "out_tok_ego": fmt(mean(r["usage"]["ego"]["completion_tokens"] / max(1, r["usage"]["ego"]["calls"]) for r in rs), 0),
            "thinking": sum(r["usage"]["ego"]["thinking"] + r["usage"]["partner"]["thinking"] for r in rs),
            "truncated": sum(r["usage"]["ego"]["truncated"] + r["usage"]["partner"]["truncated"] for r in rs),
            "wall_s": fmt(mean(r["wall_s"] for r in rs), 0),
        })
    return rows


def paired_rows(eps, ref):
    by = defaultdict(dict)
    for r in eps:
        by[r["condition"]][(r["layout"], r["persona"], r["ep"])] = r["return"]
    if ref not in by:
        return []
    rows = []
    for c in sorted(by):
        if c == ref:
            continue
        for lay, p in sorted({(k[0], k[1]) for k in by[c]}):
            a = {k: v for k, v in by[c].items() if k[:2] == (lay, p)}
            b = {k: v for k, v in by[ref].items() if k[:2] == (lay, p)}
            d, (lo, hi), n = paired_diff(a, b)
            rows.append({"condition": c, "vs": ref, "layout": lay, "persona": p, "n": n,
                         "diff": f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]"})
    return rows


def failure_rows(eps):
    g = defaultdict(Counter)
    for r in eps:
        for role in ("ego", "partner"):
            g[(r["condition"], role)].update(r["failures"][role])
            g[(r["condition"], role)]["parse_fail"] += r["parse_fail"][role]
    return [{"condition": c, "role": role, "codes": ", ".join(f"{k} {v}" for k, v in cnt.most_common()) or "-"}
            for (c, role), cnt in sorted(g.items())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", help="reference condition for paired differences")
    a = ap.parse_args(argv)
    eps = [json.loads(l) for l in open(os.path.join(a.run_dir, "episodes.jsonl")) if l.strip()]
    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    summ = summary_rows(eps)
    write_csv(os.path.join(out, "summary.csv"), summ)
    print("## Summary\n")
    print(md(summ, list(summ[0].keys())))
    print(f"\n`!` = infrastructure failures above {INFRA_FLAG:.0%} of decisions (flagged, not dropped).\n")
    if a.ref:
        pr = paired_rows(eps, a.ref)
        write_csv(os.path.join(out, "paired.csv"), pr)
        if pr:
            print(f"## Paired return difference vs {a.ref}\n")
            print(md(pr, list(pr[0].keys())) + "\n")
    fr = failure_rows(eps)
    write_csv(os.path.join(out, "failures.csv"), fr)
    print("## Failure codes\n")
    print(md(fr, ["condition", "role", "codes"]))
    return summ


if __name__ == "__main__":
    main()
