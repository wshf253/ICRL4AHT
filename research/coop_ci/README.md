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
lab-run --new --gpus 1 -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/m2_compliance.yaml --workers 8
```
```bash
uv run --locked python -m research.coop_ci.analyze outputs/coop_ci/<run_dir>
```

Gate 0 with the LLM human as the partner (3 sampled rollouts per state):

```bash
COOP_ENTRY=validate lab-run --new --gpus 1 -- bash scripts/run_qwen38_coop.sh research/coop_ci/configs/gate0_validation.yaml --partner-backend qwen3.8-27b-partner --families F1 --workers 16
```

Serving variables (`LLM_MODEL_PATH`, `LLM_SERVE_ENV`, `MAX_NUM_SEQS`, ...) and the thinking check
are unchanged from the previous version: `enable_thinking: false` is verified at server start.

## Not in this version

Exp A runs (closed loop, saved-context regret, prediction probes), Exp B/C (memory), F3/F4
personas, frontier API runs. **F5** (optional) is the only test called ToM: the partner gets a
partial view, something recipe-relevant changes where it cannot see it (e.g. a pot is filled), and
the ego must predict the partner acting on its outdated belief (a false-belief test).
