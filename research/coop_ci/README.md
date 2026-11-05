# Cooperative intelligence on OvercookedV2

Can a frozen LLM cook, cooperating without messages, (1) infer its partner's current subgoal and
stable tendencies from behaviour, (2) turn that into complementary action, and (3) improve across
encounters through outer-loop memory, revising it when the partner changes? (Design:
`IMPLEMENTATION_PLAN.md` of the planning chat, 2026-11-04.) This package covers milestones
**M1–M3**: environment, personas and scripted policies, and Gate 0. Exp A–C come next.

What changed from the previous version (`claude/coop-tom-exp`, now in `configs/superseded/`): **the
order is no longer hidden.** Both cooks see the open orders and the whole kitchen; only the
partner's tendencies and intentions are private. Hiding the recipe turned the task into recipe
guessing (a signalling game); inference from behaviour is needed here only because there are no
messages.

## Setting

- OvercookedV2 (this repo's `envs/overcooked_v2`), text observations, macro-actions with a BFS
  executor (`planner.py`), no message channel, full view (`view_radius: null`).
- Layouts in `configs/layouts.yaml`, all mirror-symmetric (equal access for both seats, tested).
  One open order per pot; any open order can be served. Return = deliveries only (+20 / −20).
- Seats balanced: the ego is agent 0 on even episode/seed indices, agent 1 on odd ones. Prompts say
  "you" / "partner", never agent indices.
- Personas (`configs/personas.yaml`): conditional tendencies with the same competence and speed.
  F1 `onion_supplier` / `tomato_supplier`, F2 `loader` / `server`. Each has an LLM-human text, a
  one-line label (used only by the oracle conditions A2/A4), a scripted twin and a compliance predicate.
- Partner = LLM-simulated human. For now it is the **same Qwen3.8-27B** as the ego, with a different
  prompt (persona + private `thought`, which stays in the trace). The plan prefers another model
  family; `backends.yaml: qwen3.8-27b-partner` is the one place to change that.
- Ego conditions (prompts implemented and leak-tested; runs come in Exp A): A0 standard `plan` ·
  A1 structured partner reasoning · A1c token-matched generic reasoning · A2 true profile · A3 the
  partner's committed macro · A4 a mismatched profile from the same family.

Logged per episode: return, dishes, first correct dish, ready-soup idle, ego idle while useful,
coordination waste (duplicate ingredient, surplus plate, wrong-soup add, items left on counters),
blocking and duplicated-subtask rates (OverForge definitions; duplicates exclude wait/move),
compliance, failure codes, infrastructure-failure rate (flagged above 2%), calls and tokens.
Failure codes: `precheck:<what>`, `timeout`, `target_gone`, `deadlock` (infrastructure), and
`pot_changed` / `taken_by_partner` (replan / contention, not infrastructure).

## Gate 0 (scripted, CPU, ~4 min)

`validate.py` builds 30 matched states per layout × persona (10 seeds × t0 ∈ {0, 20, 40}, warm-up
with a neutral ego), then branches 150-step rollouts in which the ego follows each allocation
policy: complementary(X) = the complement persona's jobs, conflicting(X) = X's own jobs, plus a
neutral greedy cook. A layout × family passes if complementary − conflicting > 0 with a paired
bootstrap CI over states excluding 0 for both partner types, the best allocation differs by partner
type, and ≥ 2 dishes are reached under the complementary policy.

Result (scripted twins, seed 0; complementary − conflicting return per 150 steps, 95% CI; one dish = 20):

| layout | F1 vs onion_supplier | F1 vs tomato_supplier | F2 vs loader | F2 vs server | verdict |
|---|---|---|---|---|---|
| shared_1pot | +10.9 [7.1, 15.1] | +9.6 [5.8, 13.8] | | | **F1 passes** |
| split_1pot | +7.8 [3.1, 13.3] | +4.2 [1.8, 7.1] | | | **F1 passes** |
| split_2pot | +3.8 [0.7, 7.3] | +3.3 [0.0, 7.3] | +2.2 [−1.8, 6.4] | +1.3 [−3.3, 6.2] | **F1 passes** (barely), F2 fails |
| shared_2pot | | | −3.3 [−7.6, 0.9] | +14.2 [8.9, 19.8] | F2 fails |
| lanes_2pot | +0.2 [−3.6, 3.8] | +4.0 [−0.0, 7.8] | +3.8 [−0.7, 8.2] | +4.7 [0.0, 9.6] | both fail |

Read-off: partner type matters in F1 (best allocation flips with the partner on every passing
layout), but the effect is about half a dish per 150 steps, and a partner-blind greedy cook
(`neutral`) recovers part of it (e.g. shared_1pot 61–63 vs 66–67). F2 fails as defined: a loader's
fallback (plate when nothing needs loading) makes two loaders almost as good as loader + server.
Scripted-pair sanity: 5–10 dishes per 300 steps, 0 wrong, 0 deadlocks, infrastructure failures
≤ 0.1% of decisions, twin compliance 100% on every layout × persona.

## M2: LLM-human compliance (Qwen3.8-27B, first run)

| layout | persona | compliance | partner actions that could not start |
|---|---|---|---|
| shared_2pot | loader | 0.80 | 5% |
| shared_2pot | server | 0.96 | 14% |
| split_2pot | onion_supplier | **0.50** | 9% |
| split_2pot | tomato_supplier | **0.37** | 9% |

F1 was below the 0.70 bar, so its wording was revised (v2, logged under `revisions` in
`personas.yaml`): an explicit habit ("you are the one who fetches onions ... even when a tomato is
needed too") and "open orders" instead of "the current order". The F1 check now runs on the
layouts that passed Gate 0. Changes for every cook: each pot line says which open orders its
contents still fit (or "wrong soup"); the LLM human may write up to 160 tokens (3 replies were
cut off at 120). `analyze` now also lists the actions that could not start and, per persona,
what the partner chose when its tendency applied but it did not follow it.

Why partner actions could not start (same run): mostly holding the wrong item, i.e. an empty plate
while trying to fetch an ingredient (79), plate a pot that was still filling (31) or put the plate
down (24), and an ingredient while trying to fetch a plate (38). Changes: the rules now say a cook
holds one item and must put it down first; `place on counter x y` on a cell that is not a free
counter uses the nearest free counter and says so. Two rates are reported: `failed_attempts`
(every reply whose action failed, retries included) and `infra_after_retry` (decision points still
failed after the fixed 2-attempt retry policy: forced wait, timeout, target gone, deadlock). The
2% flag applies to `infra_after_retry`.

### M2 rerun (persona v2, one-item rule, counter fallback)

| layout | persona | compliance | failed attempts | infra after retry | dishes / 300 steps |
|---|---|---|---|---|---|
| shared_1pot | onion_supplier | 0.97 | 2.3% | 0.2% | 6.0 |
| shared_1pot | tomato_supplier | 0.91 | 1.8% | 0.0% | 6.4 |
| split_1pot | onion_supplier | 1.00 | 1.5% | 0.0% | 4.8 |
| split_1pot | tomato_supplier | 0.88 | 1.2% | 0.0% | 5.0 |
| shared_2pot | loader | 0.81 | 2.4% | 0.2% | 9.2 |
| shared_2pot | server | 0.88 | 6.8% | 0.2% | 8.4 |

M2 passes: every persona is above 0.70 and infrastructure failures after retry are at most 0.2%.
The remaining F1 misses are mostly the other ingredient (tomato_supplier fetched an onion 4-6
times) or a short wait. The LLM human with a scripted neutral ego makes about as many dishes as
the scripted pair (shared_1pot 6.0-6.4 vs 6.7). Next: Gate 0 with the LLM human as partner.

## Gate 0 with the LLM human (Qwen3.8-27B partner, F1)

| layout | vs onion_supplier | vs tomato_supplier | complementary − neutral (onion / tomato) | verdict |
|---|---|---|---|---|
| shared_1pot | +15.8 [10.2, 21.8] | +13.8 [8.2, 19.8] | +7.4 / +0.4 | **passes** |
| split_1pot | +14.2 [10.2, 18.2] | +10.4 [6.7, 14.4] | +7.3 / +5.1 | **passes** |

The partner sensitivity survives (and grows) with a real LLM partner: complementary − conflicting
is +10 to +16 per 150 steps (about 0.5-0.8 dish), against +4 to +11 with the scripted twin;
conflicting allocations cause more wrong-soup adds (1.5-1.9 vs 0.1-0.3) and duplicate ingredients.
A partner-blind greedy cook recovers part of the gap, so the headroom for partner information over
`neutral` is about +5 to +7, except against tomato_supplier on shared_1pot (~0).

## Exp A pilot, first run (6 episodes per cell; mean return per 300 steps)

| condition | shared / onion | shared / tomato | split / onion | split / tomato | mean |
|---|---|---|---|---|---|
| A0 standard | 30.0 | 113.3 | 33.3 | 86.7 | 65.8 |
| A1 structured partner reasoning | 36.7 | 110.0 | 56.7 | 83.3 | 71.7 |
| A1c generic reasoning | 73.3 | 90.0 | 63.3 | 86.7 | 78.3 |
| A2 true profile | 106.7 | 113.3 | 76.7 | 73.3 | 92.5 |
| A3 partner's committed macro | 86.7 | 100.0 | 60.0 | 40.0 | 71.7 |
| A4 wrong profile | 46.7 | 90.0 | 36.7 | 76.7 | 62.5 |
| ref complementary / neutral / conflicting | 136.7 / 103.3 / 70.0 | 126.7 / 136.7 / 66.7 | 100 / 83.3 / 66.7 | 110 / 96.7 / 70 | 118.3 / 105 / 68.3 |

A0 does well with tomato_supplier and badly with onion_supplier; the true profile (A2) closes most
of the onion gap and the wrong one (A4) does not. But the runs were distorted by a dead end that is
not about the partner: OvercookedV2 has no way to get rid of an item except a pot, a counter or the
serving window. Once the 10 counters of shared_1pot filled up, a cook holding an unneeded
ingredient (or a wrong dish) could not do anything else (ready soups waited 116-161 steps in A0
vs onion_supplier; "no free counter" was the top start failure; one scripted reference was stuck
with a wrong dish for 378 decisions). Fix: **put back** (an extension, used the same way by every
cook): interacting with an ingredient's pile (or the plate pile) while holding it returns it.
Scripted cooks use it only when no counter is free, and serve a wrong dish (-20) when it cannot be
set aside; scripted Gate 0 results are unchanged. A1/A1c output caps were raised (words x 4 + 160
tokens, still equal for both) after 53 truncated A1c replies. The pilot now uses 10 episodes per cell.

## Exp A pilot, second run (with put back; 10 episodes per cell; mean return [95% CI])

| condition | shared / onion | shared / tomato | split / onion | split / tomato |
|---|---|---|---|---|
| A0 | 20 [6, 34] | 98 [70, 124] | 20 [4, 42] | 68 [46, 88] |
| A1 | 44 | 84 | 30 | 70 |
| A1c | 46 | 104 | 46 | 68 |
| A2 true profile | **78 [54, 104]** | 88 | **78 [60, 96]** | 76 |
| A3 partner's macro | 50 | 106 | 80 | 58 |
| A4 wrong profile | 24 | 96 | 24 | 32 [8, 58] |
| ref complementary / neutral / conflicting | 134 / 122 / 94 | 136 / 134 / 112 | 104 / 94 / 68 | 110 / 106 / 82 |

Read-off: (1) A0 fails against onion_supplier only (20 vs 68-98 against tomato_supplier). Its
ready soups sat 183-202 of 300 steps: both cooks hold an ingredient for the next soup while the soup
waits, and nobody plates (top start failure: "put in pot" on a cooking/ready pot, 1,119 times).
The complement of a supplier is "fetch the other ingredient, then plate and serve"; A0 does not
find it. (2) The true profile fixes most of that (A2 +58 on both layouts); a wrong profile does not
(A4 ~ A0) and hurts against tomato_supplier on split_1pot (32 vs 68). (3) Against tomato_supplier A0
already does the right thing and A2 adds nothing. (4) A1 ~ A1c: no gain from partner-structured
reasoning beyond generic reasoning. (5) Every LLM condition stays well below the scripted
references (best LLM cell 106 vs 104-136), so execution also limits. Counters still fill up (items
left at the end: 5-15 in LLM conditions vs 0-3 for scripted); `analyze` now also reports counter
placements and put-backs per cook, and `ready_both_hold` (steps with a ready soup while both cooks
hold an ingredient).

## Exp B: transfer and retention through outer-loop memory (`expB_memory.yaml`, stream mode)

One stream = one ego that keeps text notes across 10 episodes on shared_1pot:
`A x3` (learn partner A) -> `A_heldout x2` (pure orders onion x3 / tomato x3, so the learnt division of
labour no longer fits) -> `B x3` (a new partner id with the other persona) -> `A_return x2`.
Notes are updated after A and B episodes only; held-out and return episodes use frozen notes (strict
transfer track). Partners A/B swap on odd streams. Every condition gets the same seeds, seats, orders and
partners; only the ego's MEMORY block differs (budget 1500 characters):

| condition | memory |
|---|---|
| B0 | none (the block says "(none yet)") |
| B1 | general lessons; the update prompt forbids partner-specific claims |
| B2 | structured note per partner id: tendencies, conditions of applicability, evidence for/against (episode labels), confidence, open questions |
| B3 | raw history: most recent episode summaries with this partner, no LLM rewrite |

Ego planner = A0 (A1 gave no gain over A0 or A1c in Exp A). Episode summaries contain only what the ego
saw (who added what, plated, served, outcomes, action counts). Every memory version, update prompt and raw
reply is saved under `memory/<condition>/stream_<s>/`; a stopped run resumes with `--resume`.
`analyze` (stream runs) prints the return per phase with 95% CIs over streams, paired differences vs B0
per phase (transfer = A_heldout, unfamiliar partner = B, retention = A_return), the return by episode,
and memory size and update failures.

```bash
DP_SIZE=2 MAX_NUM_SEQS=32 lab-run --new --spare --gpus 2 --cpus 32 --mem 160G -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/expB_memory.yaml --workers 64
```

### Exp B, first run (6 streams; mean return per phase, paired difference vs B0 with 95% CI over streams)

| condition | A (learning) | A_heldout (transfer) | B (new partner) | A_return (retention) |
|---|---|---|---|---|
| B0 none | 77.8 | 125.0 | 88.9 | 78.3 |
| B1 general | +4.4 [-11, 22] | 0.0 [-20, 27] | -2.2 [-33, 24] | +10.0 [-18, 40] |
| B2 partner note | +7.8 [-19, 37] | +10.0 [-12, 43] | -17.8 [-36, 0] | +1.7 [-25, 23] |
| B3 raw history | -20.0 [-53, 13] | -11.7 [-38, 13] | -1.1 [-19, 20] | +6.7 [-43, 50] |

No difference is resolved with 6 streams (per-stream phase means vary by +-30). By persona, memory
matters only against onion_supplier: B1 cut the steps a ready soup waited from 113 (B0) to 44 and the
hold-and-wait steps from 49 to 14 (return 61 -> 77), B2 to 74 / 29 (return 65); against
tomato_supplier B0 is already best (121). So far the gain looks like generic task learning (B1 >= B2),
which the plan's decision rule reads as "strengthen controls". Next run: 12 streams (6 more, resumed in
the same run directory) and the privileged control `B2_oracle` (the true persona label as the note),
which shows what a correct partner note can do inside a stream; `analyze` writes the notes each
condition carried (analysis/notes.md) to check what B1/B2 actually learnt.

### Exp B, 12 streams (paired difference vs B0, 95% CI over streams)

| condition | A (learning) | A_heldout | B (new partner) | A_return |
|---|---|---|---|---|
| B1 general | +2.2 [-11, 17] | +1.7 [-13, 18] | +10.6 [-11, 28] | -1.7 [-18, 16] |
| B2 partner note | +5.0 [-13, 26] | +2.5 [-17, 23] | -12.8 [-32, 6] | -6.7 [-28, 12] |
| B2_oracle (true label) | +10.0 [-8, 28] | +5.8 [-5, 18] | +8.3 [-7, 24] | +3.3 [-7, 14] |

Still unresolved. By persona (all phases): against onion_supplier B0 59, B2 64, B1 75, B2_oracle 76;
against tomato_supplier all 109-120. Even the true label as the note gains only ~+17 against
onion_supplier and nothing against tomato_supplier; general lessons (B1) do as well as the true label,
and the learnt partner notes (B2) do worse than both (B2 note updates also failed to parse 11/72 times).
Changes: memory-update reply cap 700 -> 1000 tokens; `analyze` scores whether each B2 note names the
partner's ingredient (keyword heuristic, `note_accuracy.csv`); `counterbalance_display: true` lists tomato
first in half the streams (crossed with which persona is A), to be switched on if the listing-order check
confirms a position bias.

### Listing-order check (`expA_order_check.yaml`, A0 return, 10 episodes per cell)

| prompts list | vs onion_supplier | vs tomato_supplier |
|---|---|---|
| onion first (default) | **34** | 96 |
| tomato first | 110 | **24** |

The asymmetry is a position bias of Qwen: the LLM partner over-plays the persona of whichever
ingredient is listed first (over-fetching it when no pot needs it: 0.72 vs 0.41 with onion first,
0.51 vs 0.34 with tomato first), keeps its hands full and never plates, and the episode collapses into
hold-and-wait. The ego's own split stays complementary either way. The true profile (A2) helps exactly
in that cell (+42 / +54) and much less in the other (+12 / -2). All experiment configs now set
`counterbalance_display: true` (grid: episode pairs; streams: stream pairs, crossed with which persona is
A), and `analyze` reports the return by whether the partner supplies the first-listed ingredient.

## Run on the server

```bash
cd ~/work/icrl4aht
```
```bash
git fetch origin
```
```bash
git switch claude/coop-ci
```
```bash
git pull --ff-only origin claude/coop-ci
```

Gate 0 with scripted partners (no GPU):

```bash
uv run --locked python -m research.coop_ci.validate research/coop_ci/configs/gate0_validation.yaml --workers 8
```

M2 compliance of the LLM human (Qwen3.8-27B plays each persona against a scripted neutral ego;
10 episodes per persona; revise any persona's wording below 70%):

```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 120G -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/m2_compliance.yaml --workers 8
```
```bash
uv run --locked python -m research.coop_ci.analyze outputs/coop_ci/<run_dir>
```

Exp A pilot, closed loop (A0 A1 A1c A2 A3 A4 + scripted references; 2 layouts x 2 personas x 6
episodes; ~216 episodes):

```bash
MAX_NUM_SEQS=32 lab-run --new --spare --gpus 1 --cpus 24 --mem 120G -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/expA_pilot.yaml --workers 32
```
```bash
uv run --locked python -m research.coop_ci.analyze outputs/coop_ci/<run_dir> --ref A0
```

Gate 0 with the LLM human as the partner (3 sampled rollouts per state):

```bash
MAX_NUM_SEQS=32 COOP_ENTRY=validate lab-run --new --spare --gpus 1 --cpus 24 --mem 120G -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/gate0_validation.yaml --partner-backend qwen3.8-27b-partner --families F1 --layouts shared_1pot,split_1pot --no-sanity --workers 32
```

More throughput: one served model handles both roles, so a second model is never needed.
On one GPU, raise `MAX_NUM_SEQS` (and `--workers` to match) as far as the KV cache allows (vLLM
logs the KV-cache size and the maximum concurrency at startup). On several GPUs, `DP_SIZE=<n>`
serves n copies behind one endpoint (`lab-run --gpus <n>`, `--workers` = n x `MAX_NUM_SEQS`);
`TP_SIZE` splits one copy over GPUs and only matters for models that do not fit on one card.

Serving variables (`LLM_MODEL_PATH`, `LLM_SERVE_ENV`, `MAX_NUM_SEQS`, ...) and the thinking check
are unchanged from the previous version: `enable_thinking: false` is verified at server start.

## Not in this version

Saved-context regret and prediction probes (Exp A diagnostics), Exp C (revision), memory interventions, F3/F4
personas, frontier API runs. **F5** (optional) is the only test called ToM: the partner gets a
partial view, something recipe-relevant changes where it cannot see it (e.g. a pot is filled), and
the ego must predict the partner acting on its outdated belief (a false-belief test).
