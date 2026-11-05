"""Summarise a grid run directory (episodes.jsonl).

  python -m research.coop_ci.analyze outputs/coop_ci/<run_dir> [--ref A0]

Prints markdown tables and writes <run_dir>/analysis/*.csv:
  summary     per condition x layout x persona: return (95% bootstrap CI over episodes), dishes,
              partner compliance, infrastructure-failure rate (flagged above 2% of decisions),
              dependency delay, coordination waste, cost
  paired      per condition vs --ref on matched episodes (same layout, persona, episode index)
  failures    failure codes per role (pot_changed / taken_by_partner are not infrastructure)
  start_failures     (from traces) chosen actions that could not start: action, code, what was held
  compliance_misses  (from traces) what the partner chose when its tendency applied but it did not follow it
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


def mean_known(xs):
    """Mean over the episodes that recorded the field; None ('-') when none did (older runs)."""
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def clause_rate(rs, key):
    n = sum(r.get("clauses", {}).get(key, {}).get("n", 0) for r in rs)
    h = sum(r.get("clauses", {}).get(key, {}).get("hit", 0) for r in rs)
    return h / n if n else None


def fill_from_traces(eps, run_dir):
    """Counter placements and put-backs per cook, recovered from trace events for runs recorded
    before these fields existed (ready_both_hold needs per-step state and cannot be recovered)."""
    d = os.path.join(run_dir, "traces")
    for r in eps:
        w = r["waste"]
        if "counter_places" in w:
            continue
        p = os.path.join(d, r["episode_id"].replace("|", "__") + ".jsonl")
        if not os.path.exists(p):
            continue
        cnt = {k: {"ego": 0, "partner": 0} for k in ("counter_places", "put_backs")}
        for line in open(p):
            e = json.loads(line)
            if e.get("type") == "event" and e["kind"] in ("counter_place", "put_back"):
                role = "ego" if e["agent"] == r["ego_seat"] else "partner"
                cnt["counter_places" if e["kind"] == "counter_place" else "put_backs"][role] += 1
        w.update(cnt)


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

        def after(role):  # older runs have no after-retry numbers
            pts = sum(r.get("decision_points", {}).get(role, 0) for r in rs)
            n = sum(r.get("infra_after_retry", {}).get(role, 0) * r.get("decision_points", {}).get(role, 0) for r in rs)
            return n / pts if pts else None
        ae, ap = after("ego"), after("partner")
        worst = max(x for x in (ae, ap, 0.0) if x is not None) if ae is not None else max(inf_e, inf_p)
        flag = " !" if worst > INFRA_FLAG else ""
        rows.append({
            "condition": c, "layout": lay, "persona": p, "n": len(rs),
            "return": f"{fmt(mean(ret))} [{fmt(lo)}, {fmt(hi)}]",
            "correct": fmt(mean(r["n_correct"] for r in rs), 2), "wrong": fmt(mean(r["n_wrong"] for r in rs), 2),
            "compliance": fmt(mat / opp if opp else None, 3),
            "failed_attempts_ego/partner": f"{fmt(inf_e, 3)}/{fmt(inf_p, 3)}",
            "infra_after_retry_ego/partner": f"{fmt(ae, 3)}/{fmt(ap, 3)}{flag}",
            "ego_idle_useful": fmt(mean(r["ego_idle_while_useful"] for r in rs)),
            "ready_idle": fmt(mean(r["ready_idle_steps"] for r in rs)),
            "ready_both_hold": fmt(mean_known(r.get("ready_both_hold_steps") for r in rs)),
            "dup_subtask": fmt(mean(r["dup_subtask_rate"] for r in rs), 3),
            "blocking": fmt(mean(r["blocking_rate"] for r in rs), 3),
            "dup_ing": fmt(mean(r["waste"]["dup_ingredient"] for r in rs), 2),
            "plate_when_ready_ego/partner": "/".join(fmt(clause_rate(rs, f"{role}:plate_ready"), 2) for role in ("ego", "partner")),
            "over_fetch_partner": fmt(clause_rate(rs, "partner:over_fetch"), 2),
            "counters_end": fmt(mean(r["waste"]["counter_items_end"] for r in rs), 1),
            "placed/put_back_ego": "/".join(fmt(mean_known(r["waste"].get(k, {}).get("ego") for r in rs), 1)
                                            for k in ("counter_places", "put_backs")),
            "placed/put_back_partner": "/".join(fmt(mean_known(r["waste"].get(k, {}).get("partner") for r in rs), 1)
                                                for k in ("counter_places", "put_backs")),
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


def read_traces(run_dir, eps=None):
    """(episode meta, decision records) per episode, from traces/<episode_id>.jsonl. With `eps` the
    meta comes from the episode records (stream ids do not encode layout and persona)."""
    d = os.path.join(run_dir, "traces")
    meta = {r["episode_id"].replace("|", "__"): {"condition": r["condition"], "layout": r["layout"],
                                                 "persona": r["persona"]} for r in eps or []}
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        key = fn[:-len(".jsonl")]
        parts = key.split("__")
        if key in meta:
            m = meta[key]
        elif len(parts) == 4:
            m = dict(zip(("condition", "layout", "persona", "ep"), parts))
        else:
            continue
        recs = [json.loads(l) for l in open(os.path.join(d, fn)) if l.strip()]
        out.append((m, [r for r in recs if r.get("type") == "decision"]))
    return out


def start_failure_rows(traces, top=12):
    """Why chosen actions could not start: role x action verb x failure code x what was held."""
    cnt, tot, example = Counter(), Counter(), {}
    for _, decs in traces:
        for r in decs:
            tot[r["role"]] += 1
            if r.get("failed"):
                verb = " ".join(w for w in r["action"].split() if not w.isdigit())
                k = (r["role"], verb, r["failed"]["code"], r.get("holding", ""))
                cnt[k] += 1
                example.setdefault(k, f"{r['action']}: {r['failed']['reason']}"[:140])
    return [{"role": k[0], "action": k[1], "code": k[2], "holding": k[3], "n": v,
             "share_of_role_decisions": fmt(v / max(1, tot[k[0]]), 3), "example": example[k]}
            for k, v in cnt.most_common(top)]


def compliance_miss_rows(traces):
    """At decisions where the persona's tendency applied: what the partner chose instead."""
    g = defaultdict(Counter)
    for meta, decs in traces:
        for r in decs:
            o = r.get("opportunity")
            if o is not None:
                key = (meta["layout"], meta["persona"], "/".join(o["tendency"]) if len(o["tendency"]) < 3 else "any ingredient")
                g[key]["_n"] += 1
                if not o["complied"]:
                    g[key][r["subgoal"][0] if r.get("subgoal") else r["action"]] += 1
    rows = []
    for (lay, p, tend), c in sorted(g.items()):
        n = c.pop("_n")
        miss = sum(c.values())
        rows.append({"layout": lay, "persona": p, "tendency": tend, "opportunities": n,
                     "compliance": fmt(1 - miss / n, 3),
                     "chosen instead": ", ".join(f"{k} {v}" for k, v in c.most_common(6)) or "-"})
    return rows


PHASE_ORDER = ("A", "A_heldout", "B", "A_return")
READOUT = {"A": "familiar partner (learning)", "A_heldout": "transfer: held-out task, frozen notes",
           "B": "unfamiliar partner", "A_return": "retention: partner A again, frozen notes"}


def stream_rows(eps, ref):
    """Stream runs (Exp B/C): resampling unit = stream (episodes within a stream are not independent)."""
    per = defaultdict(lambda: defaultdict(list))  # (cond, phase) -> stream -> returns
    for r in eps:
        per[(r["condition"], r["phase"])][r["stream"]].append(r["return"])
    conds = sorted({r["condition"] for r in eps})
    phases = [p for p in PHASE_ORDER if any(r["phase"] == p for r in eps)] + sorted(
        {r["phase"] for r in eps} - set(PHASE_ORDER))
    rows, prow = [], []
    for c in conds:
        row = {"condition": c, "memory": next(r["memory_kind"] for r in eps if r["condition"] == c)}
        for ph in phases:
            m = {s: mean(v) for s, v in per[(c, ph)].items()}
            lo, hi = boot_ci(list(m.values()))
            row[ph] = f"{fmt(mean(m.values()) if m else None)} [{fmt(lo)}, {fmt(hi)}]"
            if ref and c != ref:
                b = {s: mean(v) for s, v in per[(ref, ph)].items()}
                d, (dl, dh), n = paired_diff(m, b)
                prow.append({"condition": c, "vs": ref, "phase": ph, "readout": READOUT.get(ph, ""), "streams": n,
                             "diff": f"{fmt(d)} [{fmt(dl)}, {fmt(dh)}]"})
        rows.append(row)
    curve = []
    for c in conds:
        ks = sorted({r["k"] for r in eps if r["condition"] == c})
        row = {"condition": c}
        for k in ks:
            v = [r["return"] for r in eps if r["condition"] == c and r["k"] == k]
            ph = next(r["phase"] for r in eps if r["k"] == k)
            row[f"E{k + 1} {ph}"] = fmt(mean(v), 0)
        curve.append(row)
    mem = []
    for c in conds:
        rs = [r for r in eps if r["condition"] == c]
        st = Counter(r["memory_update"]["status"].split(" ")[0] for r in rs)
        mem.append({"condition": c, "chars_in (mean, by phase)": ", ".join(
            f"{ph} {fmt(mean(r['memory_in']['chars'] for r in rs if r['phase'] == ph), 0)}" for ph in phases),
            "updates": ", ".join(f"{k} {v}" for k, v in st.most_common()),
            "update_tokens_out": fmt(mean_known(r["memory_update"].get("update_tokens", [None, None])[1] for r in rs), 0)})
    return rows, prow, curve, mem


def note_samples(eps) -> str:
    """The notes each learning condition carried into the first A_return episode (partner A, all
    updates done) and into the last B episode, next to the true personas (evaluator only)."""
    out = ["# Notes carried by the ego (true personas shown for the evaluator only)\n"]
    conds = sorted({r["condition"] for r in eps if r["memory_kind"] in ("general", "partner", "raw")})
    for s in sorted({r["stream"] for r in eps}):
        rs = {(r["condition"], r["phase"], r["k"]): r for r in eps if r["stream"] == s}
        ret = min((k for (c, ph, k) in rs if ph == "A_return"), default=None)
        lastb = max((k for (c, ph, k) in rs if ph == "B"), default=None)
        for c in conds:
            for k, who in ((ret, "partner A"), (lastb, "partner B")):
                r = rs.get(next((key for key in rs if key[0] == c and key[2] == k), None))
                if r is None:
                    continue
                out.append(f"## stream {s} / {c} / into E{k + 1} ({who} = {r['persona']}, {r['partner_id']})\n")
                out.append("```\n" + (r["memory_in"]["text"] or "(empty)") + "\n```\n")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", help="reference condition for paired differences")
    a = ap.parse_args(argv)
    eps = [json.loads(l) for l in open(os.path.join(a.run_dir, "episodes.jsonl")) if l.strip()]
    fill_from_traces(eps, a.run_dir)
    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    if any(r.get("mode") == "stream" for r in eps):
        ref = a.ref or "B0"
        rows, prow, curve, mem = stream_rows(eps, ref)
        for name, data in (("stream_phases", rows), ("stream_paired", prow), ("stream_curve", curve), ("memory", mem)):
            write_csv(os.path.join(out, f"{name}.csv"), data)
        print("## Mean return per phase (95% CI over streams)\n")
        print(md(rows, list(rows[0].keys())))
        print("\nPhases: " + "; ".join(f"{k} = {v}" for k, v in READOUT.items()) + "\n")
        if prow:
            print(f"## Paired difference vs {ref} (per stream)\n")
            print(md(prow, list(prow[0].keys())) + "\n")
        print("## Return by episode (mean over streams)\n")
        print(md(curve, list(curve[0].keys())) + "\n")
        print("## Memory\n")
        print(md(mem, list(mem[0].keys())) + "\n")
        notes = note_samples(eps)
        with open(os.path.join(out, "notes.md"), "w") as f:
            f.write(notes)
        print(f"Final notes per stream (what the ego carried into the return / last B episode): {out}/notes.md\n")
    summ = summary_rows(eps)
    write_csv(os.path.join(out, "summary.csv"), summ)
    print("## Summary\n")
    print(md(summ, list(summ[0].keys())))
    print(f"\n`failed_attempts` = share of replies whose action could not start or failed (each retry counts). "
          f"`infra_after_retry` = share of decision points still failed after the 2-attempt retry policy "
          f"(forced wait, timeout, target gone, deadlock). `!` = above {INFRA_FLAG:.0%} (flagged, not dropped; "
          f"runs without after-retry numbers are flagged on failed attempts).\n")
    if a.ref and not any(r.get("mode") == "stream" for r in eps):
        pr = paired_rows(eps, a.ref)
        write_csv(os.path.join(out, "paired.csv"), pr)
        if pr:
            print(f"## Paired return difference vs {a.ref}\n")
            print(md(pr, list(pr[0].keys())) + "\n")
    fr = failure_rows(eps)
    write_csv(os.path.join(out, "failures.csv"), fr)
    print("## Failure codes\n")
    print(md(fr, ["condition", "role", "codes"]))
    traces = read_traces(a.run_dir, eps)
    sf = start_failure_rows(traces)
    write_csv(os.path.join(out, "start_failures.csv"), sf)
    if sf:
        print("\n## Actions that could not start (top)\n")
        print(md(sf, list(sf[0].keys())))
    cm = compliance_miss_rows(traces)
    write_csv(os.path.join(out, "compliance_misses.csv"), cm)
    if cm:
        print("\n## Persona compliance: what the partner chose when its tendency applied but it did not follow it\n")
        print(md(cm, list(cm[0].keys())))
    return summ


if __name__ == "__main__":
    main()
