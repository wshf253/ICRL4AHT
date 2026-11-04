# Cooperative intelligence on OvercookedV2 (first version)

Can an LLM agent (the "robot" cook) infer its partner's hidden state from behaviour alone, and
act on it? This is the first runnable version of the paper's experiments, ported from the
`coop-tom-ovc2` planning harness into this repo. It covers what has to be tested first:

| Step | Config | What it answers |
|---|---|---|
| 1 | serving + `check` | Is Qwen3.8-27B's thinking really off? Does JSON come back? |
| 2 | `exp0_smoke` | Task competence without ToM (robot sees the order), parse failures, calls and wall time per episode |
| 3 | `exp1_diagnosis` | Q0: order hidden/given × persona hidden/given; probe accuracy, inference vs utilization error, failures by phase |
| 4 | `exp2_tom_rationale` | T1–T4: direct / cot / tom_free / tom_slots / tom_2nd (supported; run after Exp 1) |

Not in this version (next): memory evolving across episodes (Exp 3–6), an LLM-simulated human
partner from another model family (the hook exists: `--partner-backend partner-llm`), frontier
API runs (`--agent-backend frontier` once filled in), PARTNR.

## Scenario (Demo Cook Simple)

```
     x=01234567890
  y=0  WWWWWR2W0WW      partner = agent 0 (left room): sees the order display R
  y=1  0......W..B      robot   = agent 1 (right room): never sees R, sees the shared pot P
  y=2  W......P..X      plates B and serving window X only on the robot's side
  y=3  1......W..B      ingredients: 0 onion, 1 tomato, 2 broccoli
  y=4  WWWWWR2W1WW
```

- Menu `mixed_01`: all 3-multisets of onion/tomato, so one ingredient leaves 2–3 candidate orders.
- Wrong delivery −20. After a correct delivery the next order comes from a sequence fixed by the
  episode seed: every condition sees the same orders (paired design).
- No message channel. The robot learns the order only by watching what goes into the pot.
- Macro-actions (`fetch onion`, `put in pot`, `plate soup`, `deliver`, `wait n`, ...); a BFS
  executor walks. The LLM is called when a macro ends; a `wait` ends early when the robot sees
  its partner act.
- Partner personas (scripted twins, exact ground truth): `does_all` (adds all three; robot only
  serves), `starter` (adds one, robot must finish on ambiguous evidence), `starter_minority`,
  `slow`, `minority_first`.

Invariants (tested): the order reaches the robot's prompt only in oracle conditions; probes are
forked side calls never fed back; across prompt conditions only the output block differs.

## Run on the server

```bash
cd ~/work/icrl4aht
```
```bash
git fetch origin
```
```bash
git switch claude/coop-tom-exp
```
```bash
git pull --ff-only origin claude/coop-tom-exp
```

Offline plumbing check (CPU, no model; seconds):

```bash
JAX_PLATFORMS=cpu uv run --locked python -m research.coop_tom.run research/coop_tom/configs/exp1_diagnosis.yaml --agent-backend mock --episodes 1 --out outputs/coop_tom_selftest
```

Exp 0 with Qwen3.8-27B (serves the model in the job, checks thinking, runs, stops the server):

```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen38_coop.sh research/coop_tom/configs/exp0_smoke.yaml --workers 8
```

Then a small Exp 1 before the full grid (5 instead of 30 episodes per cell):

```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen38_coop.sh research/coop_tom/configs/exp1_diagnosis.yaml --episodes 5 --workers 8
```

Analysis (no GPU):

```bash
uv run --locked python -m research.coop_tom.analyze outputs/coop_tom/<run_dir> --ref hid_hid
```

Serving settings (environment variables of `run_qwen38_coop.sh`):

| Variable | Default | Note |
|---|---|---|
| `LLM_MODEL_PATH` | `/lab/ckpt/jsj412/Qwen3.8-27B` | checkpoint directory (must contain `config.json`) |
| `LLM_SERVE_ENV` | `~/work/llm-serving/.venv-qwen35` | vLLM venv; Qwen3.8 may need a newer vLLM |
| `ICRL_LLM_MODEL` | `qwen3.8-27b` | served model name |
| `MAX_NUM_SEQS` | 16 | concurrent sequences; `--workers` above this only queues |
| `GPU_MEM` | 0.85 | `--gpu-memory-utilization` |
| `REASONING_PARSER` | `qwen3` | empty to disable |
| `LANGUAGE_MODEL_ONLY` | 1 | set 0 if vLLM rejects `--language-model-only` for this checkpoint |
| `COOP_CHAT_TEMPLATE_KWARGS` | chosen by the check | JSON, e.g. `{"enable_thinking": false}` |
| `ICRL_LLM_BASE_URL` | unset | if set, use that endpoint and do not start a server |

**Thinking check.** `python -m research.coop_tom.check` sends a tiny JSON task with each
candidate `chat_template_kwargs` and keeps the first that returns no reasoning text. Its report is
saved as `thinking_check.jsonl` next to the vLLM log; the script stops if no candidate works.
Verified on Qwen3.8-27B: `{"enable_thinking": false}` turns thinking off; `reasoning_effort`
only accepts `xhigh` (default) / `medium` / `low`, so "none" is rejected with HTTP 400.

## What to read off first

Exp 0 (`analyze` summary): `parse_fail` (should be ~0), `thinking` and `truncated` (must be 0),
`agent_calls` and `wall_s` per episode (calibrate `max_steps` / `episodes_per_cell` / probes),
and score vs `ref_scripted_oracle`.

Exp 1: score per condition × persona with 95% CIs; paired differences vs `hid_hid`; probe
`top1` / `violation` / `persona_acc`; the Q0-b table (probe right but soup wrong = utilization
error, probe wrong = inference error); waiting by phase. `probe.triggers` with
`decision_ingredient_phase` roughly doubles agent calls; `[partner_pot_event]` alone is cheaper.

Rule-based references (scripted partner, seeds 0–7, 300 steps), measured offline:

| persona | `ref_rule` (no order) | `ref_rule_oracle` (knows the order) |
|---|---|---|
| does_all | 80.0 | 137.5 |
| minority_first | 80.0 | 145.0 |
| slow | 60.0 | 142.5 |
| starter | **−60.0** (always guesses wrong) | 137.5 |
| starter_minority | −15.0 | 150.0 |

`ref_rule` waits for evidence and commits to the first consistent order after 30 idle steps; it
never learns from a WRONG delivery, and orders change only after correct ones, so against
`starter` it repeats the same wrong guess all episode. Closing this gap is what inference
(and later memory) should do.

## Changes relative to the planning harness

- Runs on this repo's `envs/overcooked_v2`; no `jaxmarl==0.2` or `openai` dependency
  (stdlib HTTP client). `cramped_room_v2` → `cramped_room_up` for Exp 0.
- Fixed soup bookkeeping: a soup record now ends when it is plated, not delivered (the next
  soup's first ingredient was logged into the previous soup).
- `put in pot` is cancelled if the pot changed while the agent walked (e.g. the partner added
  an ingredient); the agent then decides again with what it sees (`pot_changed_replans`).
  Same rule for every agent; it needs no knowledge of the order.
- Scripted partner: walks to the pot and looks before adding (it never adds an off-order
  ingredient, as its persona states), and after seeing a soup plated waits for the new order
  instead of starting the next soup for the old one (before: 14% of its adds).
- `ref_rule` patience counts from the start of the current soup; `ref_rule_oracle` sets aside a
  dish it knows is wrong instead of serving it.
- Thinking detection (`thinking`, `truncated` per episode) and the serving-time check.
