"""Summarise a run directory (episodes.jsonl): grid runs (Exp A, M2), stream runs (Exp B), revision runs (Exp C).

  python -m research.coop_ci.analyze outputs/coop_ci/<run_dir> [--ref A0]
  python -m research.coop_ci.analyze <run_dir> --show paired,strata        # only these tables
  python -m research.coop_ci.analyze <run_dir> --show all --cols all       # everything
  python -m research.coop_ci.analyze <run_dir> --sections                  # list the tables

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

from .memory import note_verdict
from .policy import trace_name
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
        p = os.path.join(d, trace_name(r["episode_id"]) + ".jsonl")
        if not os.path.exists(p):
            continue
        cnt = {k: {"ego": 0, "partner": 0} for k in ("counter_places", "put_backs")}
        fetch = {"ego": Counter(), "partner": Counter()}
        for line in open(p):
            e = json.loads(line)
            if e.get("type") != "event":
                continue
            role = "ego" if e["agent"] == r["ego_seat"] else "partner"
            if e["kind"] in ("counter_place", "put_back"):
                cnt["counter_places" if e["kind"] == "counter_place" else "put_backs"][role] += 1
            elif e["kind"] == "fetch":
                fetch[role][(e.get("item") or "?").split()[-1]] += 1   # "an onion" -> onion, "an empty plate" -> plate
        if "counter_places" not in w:
            w.update(cnt)
        r["fetches"] = fetch


def onion_share(rs, role):
    """Share of onions among the ingredients a cook took from the piles."""
    o = sum(r.get("fetches", {}).get(role, {}).get("onion", 0) for r in rs)
    t = sum(r.get("fetches", {}).get(role, {}).get("tomato", 0) for r in rs)
    return o / (o + t) if o + t else None


PREF = {"onion_supplier": 0, "tomato_supplier": 1}


def first_listed(r) -> str:
    """'first' if the partner supplies the ingredient listed first in the prompts, else 'second'
    (Qwen over-plays the first-listed supplier, see expA_order_check)."""
    order = (r.get("env") or {}).get("display_order") or [0, 1]
    if r["persona"] not in PREF:
        return "-"
    return "first" if order[0] == PREF[r["persona"]] else "second"


def listed_rows(eps):
    """Return by condition x whether the partner supplies the first-listed ingredient."""
    g = defaultdict(list)
    for r in eps:
        g[(r["condition"], first_listed(r))].append(r["return"])
    rows = []
    for c in sorted({k[0] for k in g}):
        row = {"condition": c}
        for w in ("first", "second"):
            v = g.get((c, w), [])
            lo, hi = boot_ci(v)
            row[f"partner supplies the {w}-listed ingredient"] = f"{fmt(mean(v) if v else None)} [{fmt(lo)}, {fmt(hi)}] (n={len(v)})"
        rows.append(row)
    return rows


def summary_rows(eps):
    g = defaultdict(list)
    for r in eps:
        lay = r["layout"] + (" (tomato first)" if (r.get("env") or {}).get("display_order") == [1, 0] else "")
        g[(r["condition"], lay, r["persona"])].append(r)
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
            "onion_share_of_fetches_ego/partner": "/".join(fmt(onion_share(rs, role), 2) for role in ("ego", "partner")),
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
    meta = {trace_name(r["episode_id"]): {"condition": r["condition"], "layout": r["layout"],
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


def early_return(r, horizon):
    """Native return from deliveries in the first `horizon` steps (from the soup records)."""
    return sum((20 if s["correct"] else -20) for s in r.get("soups", [])
               if s.get("delivered_t") is not None and s["delivered_t"] <= horizon)


def stream_rows(eps, ref, value=None):
    """Stream runs (Exp B/C): resampling unit = stream (episodes within a stream are not independent).
    `value(r)` picks the outcome (default: the episode return)."""
    value = value or (lambda r: r["return"])
    per = defaultdict(lambda: defaultdict(list))  # (cond, phase) -> stream -> values
    for r in eps:
        per[(r["condition"], r["phase"])][r["stream"]].append(value(r))
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
            v = [value(r) for r in eps if r["condition"] == c and r["k"] == k]
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


def note_accuracy_rows(eps, personas_pref):
    """Per learning condition: does the note the ego carries say the right thing about the partner?"""
    rows = []
    for c in sorted({r["condition"] for r in eps if r["memory_kind"] == "partner"}):
        g = defaultdict(Counter)
        for r in eps:
            if r["condition"] != c or not r["memory_in"]["text"]:
                continue
            v = note_verdict(r["memory_in"]["text"])
            truth = personas_pref.get(r["persona"])
            g[r["phase"]]["right" if v == truth else ("wrong" if v in ("onion", "tomato") else v)] += 1
        for ph in [p for p in PHASE_ORDER if p in g]:
            n = sum(g[ph].values())
            rows.append({"condition": c, "phase": ph, "notes": n, "right": g[ph]["right"], "wrong": g[ph]["wrong"],
                         "both/none": g[ph]["both"] + g[ph]["none"], "right_share": fmt(g[ph]["right"] / n, 2)})
    return rows


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


ING = {"onion_supplier": "onion", "tomato_supplier": "tomato"}


def revision_rows(eps):
    """Exp C tables: returns after the change, revision cost vs the stable regime, recovery, and what the
    note says (A's ingredient or the new persona's) episode by episode. Resampling unit: stream."""
    pre = defaultdict(list)
    a_ing = {}
    for r in eps:
        if r.get("regime") == "prefix":
            pre[r["stream"]].append(r["return"])
            a_ing[r["stream"]] = ING.get(r["persona"])
    post = [r for r in eps if r.get("regime") not in (None, "prefix")]
    chains = sorted({(r["regime"], r["rule"]) for r in post})
    ks = sorted({r["k"] for r in post})
    by = defaultdict(dict)  # (regime, rule) -> stream -> {k: rec}
    for r in post:
        by[(r["regime"], r["rule"])].setdefault(r["stream"], {})[r["k"]] = r
    curve, total, cost, notes = [], [], [], []
    pre_mean = {s: mean(v) for s, v in pre.items() if v}
    for reg, rule in chains:
        row = {"regime": reg, "rule": rule, "prefix (mean)": fmt(mean(pre_mean.values()) if pre_mean else None, 0)}
        for k in ks:
            v = [d[k]["return"] for d in by[(reg, rule)].values() if k in d]
            row[f"E{k + 1}"] = fmt(mean(v) if v else None, 0)
        curve.append(row)
        sums = {s: sum(x["return"] for x in d.values()) for s, d in by[(reg, rule)].items() if len(d) == len(ks)}
        lo, hi = boot_ci(list(sums.values()))
        trow = {"regime": reg, "rule": rule, "streams": len(sums),
                "post-change return (sum)": f"{fmt(mean(sums.values()) if sums else None)} [{fmt(lo)}, {fmt(hi)}]"}
        static = {s: sum(x["return"] for x in d.values()) for s, d in by[(reg, "static")].items() if len(d) == len(ks)}
        if rule != "static" and static:
            dd, (dl, dh), n = paired_diff(sums, static)
            trow["vs static (paired)"] = f"{fmt(dd)} [{fmt(dl)}, {fmt(dh)}]"
        else:
            trow["vs static (paired)"] = "-"
        # recovery: first post episode from which 2 consecutive episodes reach 90% of the stream's prefix mean
        rec = []
        for s, d in by[(reg, rule)].items():
            seq = [d[k]["return"] for k in ks if k in d]
            thr = 0.9 * pre_mean.get(s, 0)
            hit = next((j + 1 for j in range(len(seq) - 1) if seq[j] >= thr and seq[j + 1] >= thr), len(seq) + 1)
            rec.append(hit)
        trow["episodes to recover (mean)"] = fmt(mean(rec) if rec else None, 1)
        total.append(trow)
        if reg != "stable":
            stable = {s: sum(x["return"] for x in d.values()) for s, d in by[("stable", rule)].items() if len(d) == len(ks)}
            dd, (dl, dh), n = paired_diff(stable, sums)
            cost.append({"regime": reg, "rule": rule, "streams": n,
                         "revision cost = stable - regime (sum over post episodes)": f"{fmt(dd)} [{fmt(dl)}, {fmt(dh)}]"})
        nrow = {"regime": reg, "rule": rule}
        for k in ks:
            c = Counter()
            for s, d in by[(reg, rule)].items():
                if k in d:
                    v = note_verdict(d[k]["memory_in"]["text"])
                    c["A" if v == a_ing.get(s) else ("new" if v in ("onion", "tomato") else "-")] += 1
            nrow[f"into E{k + 1}"] = f"A {c['A']} / new {c['new']} / - {c['-']}"
        notes.append(nrow)
    return curve, total, cost, notes


SECTIONS = {
    # Exp C (revision runs)
    "c_curve": "Exp C: return by episode (mean over streams; the first episodes are the shared prefix)",
    "c_total": "Exp C: return after the change (sum over the post-prefix episodes, 95% CI over streams)",
    "c_cost": "Exp C: revision cost = stable minus regime, same rule (paired by stream)",
    "c_notes": "Exp C: what the note says going into each episode: A's ingredient / the new persona's / neither",
    # Exp B (stream runs)
    "phases": "Mean return per phase (95% CI over streams)",
    "paired": "Paired difference vs the reference condition (grid: matched episodes; streams: per stream)",
    "strata": "Paired difference per stream, split by whether partner A is the hard (first-listed) supplier",
    "curve": "Return by episode (mean over streams)",
    "early": "Paired difference per stream, return in the first --early steps only",
    "memory": "Memory size and update status",
    "notes": "Partner notes: does the note name the partner's ingredient? (+ notes.md)",
    # any run
    "listed": "Return by whether the partner supplies the ingredient listed first in the prompts",
    "summary": "Per condition x layout x persona (columns: --cols)",
    "failures": "Failure codes per role",
    "start_failures": "Actions that could not start (from traces)",
    "compliance": "Persona compliance misses (from traces)",
}
DEFAULT_SHOW = {"revision": ["c_curve", "c_total", "c_cost", "c_notes"],
                "stream": ["phases", "paired", "early", "memory", "notes"],
                "grid": ["paired", "listed", "summary"]}
ALIASES = {"diagnostics": ["failures", "start_failures", "compliance"], "expc": DEFAULT_SHOW["revision"],
           "expb": ["phases", "paired", "strata", "curve", "early", "memory", "notes"]}
DEFAULT_COLS = ["condition", "layout", "persona", "n", "return", "correct", "wrong", "compliance",
                "infra_after_retry_ego/partner", "ready_idle", "onion_share_of_fetches_ego/partner",
                "calls_ego/partner"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Summarise a run directory. Every table is written to <run_dir>/analysis/*.csv; "
                    "--show picks which ones are printed.")
    ap.add_argument("run_dir")
    ap.add_argument("--ref", help="reference condition for paired differences (streams: default B0)")
    ap.add_argument("--early", type=int, default=100, help="stream runs: the early window, in steps")
    ap.add_argument("--show", help="comma-separated sections to print (or 'all'; aliases: expb, expc, diagnostics). "
                                   "Default depends on the run type; --sections lists them")
    ap.add_argument("--cols", help="summary columns to print, comma-separated, or 'all' (default: a compact set)")
    ap.add_argument("--sections", action="store_true", help="list the sections and exit")
    a = ap.parse_args(argv)
    if a.sections:
        for k, v in SECTIONS.items():
            print(f"{k:15s} {v}")
        print("aliases: " + "; ".join(f"{k} = {','.join(v)}" for k, v in ALIASES.items()))
        return None
    eps = [json.loads(l) for l in open(os.path.join(a.run_dir, "episodes.jsonl")) if l.strip()]
    fill_from_traces(eps, a.run_dir)
    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    kind = "revision" if any(r.get("regime") for r in eps) else (
        "stream" if any(r.get("mode") == "stream" for r in eps) else "grid")
    if not a.show:
        show = DEFAULT_SHOW[kind]
    elif a.show == "all":
        show = list(SECTIONS)
    else:
        show = []
        for x in a.show.split(","):
            show += ALIASES.get(x.strip(), [x.strip()])
        unknown = [x for x in show if x not in SECTIONS]
        if unknown:
            ap.error(f"unknown sections {unknown}; use --sections")
    tables: dict = {}   # section -> list of (title, rows) or text

    def add(sec, name, title, rows):
        if rows:
            write_csv(os.path.join(out, f"{name}.csv"), rows)
            tables.setdefault(sec, []).append((title, rows))

    if kind == "revision":
        curve, total, cost, notes = revision_rows(eps)
        add("c_curve", "revision_curve", SECTIONS["c_curve"], curve)
        add("c_total", "revision_total", SECTIONS["c_total"], total)
        add("c_cost", "revision_cost", SECTIONS["c_cost"], cost)
        add("c_notes", "revision_notes", SECTIONS["c_notes"] + " (keyword heuristic)", notes)
    elif kind == "stream":
        ref = a.ref or "B0"
        rows, prow, curve, mem = stream_rows(eps, ref)
        add("phases", "stream_phases", SECTIONS["phases"], rows)
        add("paired", "stream_paired", f"Paired difference vs {ref} (per stream)", prow)
        hard = {r["stream"]: first_listed(r) == "first" for r in eps if r.get("partner_role") == "A" and r["k"] == 0}
        for label, keep in (("A is the hard partner (first-listed supplier)", True), ("A is the other partner", False)):
            sub = [r for r in eps if hard.get(r["stream"]) is keep]
            if sub:
                _, sp, _, _ = stream_rows(sub, ref)
                add("strata", f"stream_paired_{'hardA' if keep else 'easyA'}",
                    f"Paired difference vs {ref}, streams where {label} ({len({r['stream'] for r in sub})} streams)", sp)
        add("curve", "stream_curve", SECTIONS["curve"], curve)
        _, eprow, _, _ = stream_rows(eps, ref, value=lambda r: early_return(r, a.early))
        add("early", "stream_paired_early", f"Paired difference vs {ref}, return in the first {a.early} steps only", eprow)
        add("memory", "memory", SECTIONS["memory"], mem)
        acc = note_accuracy_rows(eps, {"onion_supplier": "onion", "tomato_supplier": "tomato"})
        add("notes", "note_accuracy", SECTIONS["notes"].split(" (+")[0] + " (keyword heuristic)", acc)
        with open(os.path.join(out, "notes.md"), "w") as f:
            f.write(note_samples(eps))
        tables.setdefault("notes", []).append(
            (None, f"Final notes per stream (what the ego carried into the return / last B episode): {out}/notes.md"))
    else:
        if a.ref:
            add("paired", "paired", f"Paired return difference vs {a.ref}", paired_rows(eps, a.ref))
    if any(first_listed(r) != "-" for r in eps):
        add("listed", "listed_first", SECTIONS["listed"], listed_rows(eps))
    summ = summary_rows(eps)
    write_csv(os.path.join(out, "summary.csv"), summ)
    cols = list(summ[0].keys()) if a.cols == "all" else (
        [c for c in a.cols.split(",") if c in summ[0]] if a.cols else [c for c in DEFAULT_COLS if c in summ[0]])
    tables["summary"] = [("Summary (all columns: --cols all; full table in analysis/summary.csv)",
                          [{c: r[c] for c in cols} for r in summ])]
    add("failures", "failures", SECTIONS["failures"], failure_rows(eps))
    if {"start_failures", "compliance"} & set(show):
        traces = read_traces(a.run_dir, eps)
        add("start_failures", "start_failures", SECTIONS["start_failures"], start_failure_rows(traces))
        add("compliance", "compliance_misses", SECTIONS["compliance"], compliance_miss_rows(traces))

    for sec in show:
        for title, rows in tables.get(sec, []):
            if title is None:
                print(rows + "\n")
                continue
            print(f"## {title}\n")
            print(md(rows, list(rows[0].keys())) + "\n")
            if sec == "phases":
                print("Phases: " + "; ".join(f"{k} = {v}" for k, v in READOUT.items()) + "\n")
            if sec == "summary":
                print("`infra_after_retry` = share of decision points still failed after the 2-attempt retry "
                      f"(`!` above {INFRA_FLAG:.0%}).\n")
    hidden = [k for k in tables if k not in show]
    if hidden:
        print(f"(not shown: {', '.join(hidden)}; --show <section> or --show all; CSVs in {out})")
    return summ


if __name__ == "__main__":
    main()
