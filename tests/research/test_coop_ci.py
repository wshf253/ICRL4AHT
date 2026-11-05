"""Offline tests for the cooperative-intelligence harness (no model, no GPU)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from research.coop_ci.agents import LLMEgoPolicy, LLMHumanPolicy
from research.coop_ci.env import A_STAY, CONFIG_DIR, CoopEnv, layouts, station_distances
from research.coop_ci.llm import LLMResult, MockBackend, Usage
from research.coop_ci.planner import Executor, parse_macro, valid_actions
from research.coop_ci.policy import stable_seed
from research.coop_ci.prompts import CONDITIONS, INJECT_HEADER
from research.coop_ci.runner import Episode, load_yaml, mismatched
from research.coop_ci.scripted import ego_policy, persona_policy

PERSONAS = load_yaml(os.path.join(CONFIG_DIR, "personas.yaml"))
LAYOUTS = sorted(layouts())


def scripted_episode(layout, pid, kind="complementary", seed=0, steps=300, seat=None):
    env = CoopEnv(layout, max_steps=steps).reset(seed)
    seat = seed % 2 if seat is None else seat
    epi = Episode(env, seed, seat, ego_policy(PERSONAS, kind, pid), persona_policy(PERSONAS, pid), PERSONAS[pid])
    epi.run(steps)
    return env, epi


# --------------------------------------------------------------------------- #
# M1: environment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts_parse_and_give_both_seats_equal_access(layout):
    env = CoopEnv(layout).reset(0)
    assert env.num_agents == 2 and env.pots and env.goals and env.plate_piles and set(env.piles) == {0, 1}
    d0, d1 = (station_distances(env, env.agent_pos(i)) for i in (0, 1))
    assert d0 == d1 and None not in d0.values(), (d0, d1)
    assert env.n_open == len(env.pots)


def test_paired_seeds_and_orders_known_to_both():
    a, b = CoopEnv("split_2pot").reset(123), CoopEnv("split_2pot").reset(123)
    assert a.recipe_seq == b.recipe_seq and CoopEnv("split_2pot").reset(124).recipe_seq != a.recipe_seq
    for i in (0, 1):
        obs = a.observation_text(i)
        assert "Open orders" in obs and a.recipe_str(a.orders[0]) in obs and a.recipe_str(a.orders[1]) in obs
        assert "agent 0" not in obs.lower() and "agent 1" not in obs.lower() and "Partner: at" in obs


def test_multi_pot_soups_are_attributed_to_the_pot_and_the_deliverer():
    env, epi = scripted_episode("split_2pot", "onion_supplier", seed=2)
    delivered = [s for s in env.soups if s.delivered_t is not None]
    assert len(delivered) >= 4 and {s.pot for s in delivered} == {"pot_0", "pot_1"}
    assert sum(s.correct for s in delivered) == epi.metrics()["n_correct"]
    for s in delivered:
        assert s.ready_t <= s.plated_t <= s.delivered_t and s.plated_by == s.delivered_by
        assert len(s.adds) == 3 and sorted(k for _, _, k in s.adds) == sorted(s.contents)


def test_snapshot_restore_gives_an_identical_trajectory():
    env = CoopEnv("shared_2pot").reset(5)
    epi = Episode(env, 5, 1, ego_policy(PERSONAS, "neutral"), persona_policy(PERSONAS, "loader"), PERSONAS["loader"])
    epi.run(30)
    snap = epi.snapshot()

    def roll():
        epi.run(90)
        return ([e.__dict__ for e in env.events], env.score, [env.agent_pos(i) for i in (0, 1)],
                [(s.pot, s.contents, s.delivered_t) for s in env.soups], env.orders)

    first = roll()
    epi.restore(snap)
    assert env.t == 30
    assert roll() == first


def test_pot_names_and_pot_arguments():
    env = CoopEnv("shared_2pot").reset(0)
    assert env.pot_name == {(4, 2): "pot_0", (6, 2): "pot_1"}
    assert str(parse_macro("put in pot_1", env)) == "put in pot_1" and parse_macro("put in pot 1", env).pot == "pot_1"
    assert parse_macro("plate soup pot_0", env).pot == "pot_0" and parse_macro("put in pot_7", env) is None
    assert str(parse_macro("put in pot", env)) == "put in pot"


def test_put_in_pot_replans_when_the_pot_changed():
    env = CoopEnv("shared_1pot").reset(0)
    st = env.state  # agent 0 at (3, 2): put it next to the pot at (4, 2), facing right, holding an onion
    st = st.replace(agents=st.agents.replace(pos=st.agents.pos.replace(x=st.agents.pos.x.at[0].set(4)),
                                             dir=st.agents.dir.at[0].set(2), inventory=st.agents.inventory.at[0].set(4)))
    env.state = st
    env._sync()
    ex = Executor(env, 0)
    ex.start(parse_macro("put in pot", env))
    assert ex.ms.status == "running" and ex.ms.subgoal == ("load", "load:pot_0")
    env.state = env.state.replace(grid=env.state.grid.at[2, 5, 1].set(16))  # partner added a tomato meanwhile
    env._sync()
    assert ex.act() == A_STAY
    assert ex.ms.status == "failed" and ex.ms.code == "pot_changed"


def test_failure_codes_and_reasons():
    env = CoopEnv("shared_1pot").reset(0)
    ex = Executor(env, 0)
    ex.start(parse_macro("plate soup", env))
    assert ex.ms.code == "precheck:no_plate"
    env.state = env.state.replace(agents=env.state.agents.replace(inventory=env.state.agents.inventory.at[0].set(1)))
    env._sync()  # empty plate in hand, pot empty
    ex.start(parse_macro("plate soup", env))
    assert ex.ms.code == "precheck:no_target" and "cooking or ready" in ex.ms.reason and "filling 0/3" in ex.ms.reason
    ex.start(parse_macro("move to 50 2", env))
    assert ex.ms.code == "precheck:outside"
    ex.start(parse_macro("move to 5 2", env))  # the pot: walks next to it
    assert ex.ms.status == "running"


def test_mirror_livelock_is_broken():
    # Regression: two cooks in one column kept detouring in mirror image until both timed out.
    env = CoopEnv("lanes_2pot", max_steps=300).reset(stable_seed(0, "sanity", 14))
    epi = Episode(env, 0, 0, ego_policy(PERSONAS, "complementary", "loader"), persona_policy(PERSONAS, "loader"),
                  PERSONAS["loader"], record_trace=False)
    epi.run(300)
    m = epi.metrics()
    assert m["n_correct"] >= 4 and not m["failures"]["ego"].get("timeout") and m["deadlocks"] == 0, m


# --------------------------------------------------------------------------- #
# M2: personas, scripted policies, compliance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pid,layout", [("onion_supplier", "split_1pot"), ("tomato_supplier", "shared_1pot"),
                                        ("loader", "shared_2pot"), ("server", "split_2pot")])
def test_scripted_twins_comply_and_the_pair_is_competent(pid, layout):
    env, epi = scripted_episode(layout, pid, seed=1)
    m = epi.metrics()
    assert m["compliance"]["opportunities"] >= 5 and m["compliance"]["rate"] >= 0.95, m["compliance"]
    assert m["n_correct"] >= 4 and m["n_wrong"] == 0 and m["deadlocks"] == 0
    assert max(m["infra_rate"].values()) <= 0.02, m["failures"]


def test_complement_policies_are_the_other_persona():
    for pid, p in PERSONAS.items():
        assert PERSONAS[p["complement"]]["complement"] == pid and PERSONAS[p["complement"]]["family"] == p["family"]
        assert ego_policy(PERSONAS, "complementary", pid).jobs == PERSONAS[p["complement"]]["scripted"]["jobs"]
        assert ego_policy(PERSONAS, "conflicting", pid).jobs == p["scripted"]["jobs"]


class ScriptedLLM(MockBackend):
    """Partner replies carry a sentinel thought and a distinctive committed macro."""

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None):
        if "YOUR WORKING STYLE" in messages[0]["content"]:
            with self._lock:
                self.calls.append(messages)
            return LLMResult(json.dumps({"thought": "SENTINEL-THOUGHT-42", "action": "move to 9 3"}), finish_reason="stop")
        return super().chat(messages, max_tokens, seed, json_mode)


@pytest.mark.parametrize("layout", ["shared_1pot", "split_2pot"])
def test_no_partner_information_leaks_into_non_oracle_prompts(layout):
    secret = [s for p in PERSONAS.values() for s in (p["label"], p["text"])] + list(PERSONAS)
    prompts = {}
    for cond in CONDITIONS:
        pid = "onion_supplier"
        label = {"A2": PERSONAS[pid]["label"], "A4": PERSONAS[mismatched(PERSONAS, pid)]["label"]}.get(cond, "")
        be = ScriptedLLM()
        env = CoopEnv(layout, max_steps=40).reset(3)
        ego = LLMEgoPolicy(be, Usage(), condition=cond, partner_id="P-1a2b", profile_label=label)
        partner = LLMHumanPolicy(be, Usage(), PERSONAS[pid]["text"])
        epi = Episode(env, 3, 0, ego, partner, PERSONAS[pid])
        epi.run(40)
        ego_calls = [c for c in be.calls if "YOUR WORKING STYLE" not in c[0]["content"]]
        assert ego_calls
        prompts[cond] = ["\n".join(m["content"] for m in c) for c in ego_calls]
        assert any(r.get("private", {}).get("thought") == "SENTINEL-THOUGHT-42" for r in epi.trace)
    for cond, texts in prompts.items():
        for t in texts:
            assert "SENTINEL-THOUGHT-42" not in t and "agent 0" not in t.lower() and "agent 1" not in t.lower()
            assert ("Partner is currently doing" in t) == (cond == "A3")
            if cond == "A3":
                assert "move to 9 3" in t.split("Partner is currently doing", 1)[1].split("\n", 1)[0]
            assert (INJECT_HEADER in t) == (cond in ("A2", "A3", "A4"))
            leaked = [s for s in secret if s in t]
            if cond == "A2":
                assert leaked == [PERSONAS["onion_supplier"]["label"]]
            elif cond == "A4":
                assert leaked == [PERSONAS["tomato_supplier"]["label"]]
            else:
                assert leaked == [], (cond, leaked)


def test_ego_conditions_differ_only_in_output_or_injected_block():
    env = CoopEnv("split_2pot").reset(0)
    sys = {c: LLMEgoPolicy(MockBackend(), Usage(), condition=c,
                           profile_label="X" if c in ("A2", "A4") else "").system_prompt(env) for c in CONDITIONS}
    head = {c: s.split("Reply with JSON only")[0] for c, s in sys.items()}
    assert head["A0"] == head["A1"] == head["A1c"] == head["A3"]
    assert head["A2"].replace(f"\n\n{INJECT_HEADER}: Partner profile: X", "") == head["A0"]
    a1, a1c = (LLMEgoPolicy(MockBackend(), Usage(), condition=c) for c in ("A1", "A1c"))
    assert a1.max_tokens() == a1c.max_tokens()  # token-matched control


def test_llm_partner_is_stateless_given_env_and_executor_state():
    env = CoopEnv("shared_1pot").reset(0)
    be = MockBackend()
    p = LLMHumanPolicy(be, Usage(), PERSONAS["loader"]["text"])
    p.episode_start(env, 1, 7)
    st = p.get_state()
    assert set(st) == {"persona_text", "n_events", "name", "ep_seed", "salt"}
    d1 = p.decide(env, 1, "none", {})
    d2 = p.decide(env, 1, "none", {})
    assert be.calls[0] == be.calls[1] and d1.action == d2.action and p.get_state() == st


# --------------------------------------------------------------------------- #
# M3: Gate 0 and the CLI
# --------------------------------------------------------------------------- #
def test_gate0_runs_and_reports(tmp_path):
    from research.coop_ci import validate
    cfg = tmp_path / "g.yaml"
    cfg.write_text("name: g\nseeds: 2\nt0: [0, 10]\nrollouts: 1\nhorizon: 40\npartner: scripted\n"
                   "families:\n  F1: {layouts: [shared_1pot]}\nsanity: {seeds: 1, steps: 60}\n")
    out, verdicts = validate.main([str(cfg), "--workers", "1", "--out", str(tmp_path)])
    text = open(os.path.join(out, "report.md")).read()
    assert "shared_1pot / F1" in text and "Scripted-pair sanity" in text and len(verdicts) == 1
    rows = [json.loads(l) for l in open(os.path.join(out, "rollouts.jsonl"))]
    assert len(rows) == 2 * 2 * 2 * 3 and {r["alloc"] for r in rows} == {"onion_supplier", "tomato_supplier", "neutral"}
    assert {r["ego_seat"] for r in rows} == {0, 1}


class FakeVLLM(BaseHTTPRequestHandler):
    """Mirrors the served Qwen3.8 template: enable_thinking switches thinking, and
    reasoning_effort outside xhigh/medium/low is rejected with HTTP 400."""
    requests = []

    def log_message(self, *_):
        pass

    def respond(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.respond({"data": [{"id": "qwen3.8-27b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.requests.append(body)
        kwargs = body.get("chat_template_kwargs") or {}
        effort = kwargs.get("reasoning_effort")
        if effort is not None and effort not in ("xhigh", "medium", "low") and kwargs.get("enable_thinking", True):
            return self.respond({"error": {"message": f"Unexpected reasoning effort {effort}."}}, code=400)
        thinking = kwargs.get("enable_thinking", True)
        check_task = '"wait 1"' in body["messages"][-1]["content"]
        reply = {"action": "wait 1"} if check_task else {"thought": "fetching", "action": "fetch onion"}
        msg = {"content": json.dumps(reply),
               "reasoning_content": "Let me think about the kitchen..." if thinking else None}
        self.respond({"choices": [{"message": msg, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeVLLM.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("ICRL_LLM_BASE_URL", url)
    yield url
    server.shutdown()
    server.server_close()
    thread.join()


def test_thinking_check_picks_a_working_switch(fake_vllm, capsys, tmp_path):
    from research.coop_ci import check
    report = tmp_path / "check.jsonl"
    check.main(["--base-url", fake_vllm, "--model", "qwen3.8-27b", "--pick", "--report", str(report)])
    out = capsys.readouterr()
    assert json.loads(out.out.strip()) == {"enable_thinking": False}


def test_cli_compliance_run_end_to_end(fake_vllm, tmp_path, monkeypatch):
    from research.coop_ci import analyze, run
    monkeypatch.setenv("COOP_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    out = run.main([os.path.join(CONFIG_DIR, "m2_compliance.yaml"), "--episodes", "1", "--max-steps", "30",
                    "--layouts", "shared_1pot", "--personas", "onion_supplier", "--workers", "1", "--out", str(tmp_path)])
    rec = json.loads(open(os.path.join(out, "episodes.jsonl")).readline())
    assert rec["usage"]["partner"]["calls"] > 0 and rec["usage"]["ego"]["calls"] == 0
    assert rec["usage"]["partner"]["thinking"] == 0 and rec["compliance"]["opportunities"] > 0
    body = FakeVLLM.requests[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 160
    assert "YOUR WORKING STYLE" in body["messages"][0]["content"]
    trace = [json.loads(l) for l in open(os.path.join(out, "traces", os.listdir(os.path.join(out, "traces"))[0]))]
    assert any(r.get("private", {}).get("thought") == "fetching" for r in trace)
    assert os.path.exists(os.path.join(out, "prompts", "human__shared_1pot__onion_supplier.txt"))
    analyze.main([out])
    assert os.path.exists(os.path.join(out, "analysis", "summary.csv"))


def test_cli_rejects_a_model_that_is_not_served(fake_vllm, tmp_path, monkeypatch):
    from research.coop_ci import run
    monkeypatch.setenv("ICRL_LLM_MODEL", "some-other-model")
    with pytest.raises(SystemExit):
        run.main([os.path.join(CONFIG_DIR, "m2_compliance.yaml"), "--episodes", "1", "--out", str(tmp_path)])


def test_valid_actions_name_pots_on_two_pot_layouts():
    env = CoopEnv("split_2pot").reset(0)
    va = valid_actions(env, 0)
    assert "fetch onion" in va and "fetch plate" in va and not any(a.startswith("put in") for a in va)


def test_pots_say_which_open_orders_they_fit():
    env = CoopEnv("split_2pot").reset(3)  # orders: onion+tomato+tomato, onion+onion+tomato
    p0 = env.pots[0]
    env.state = env.state.replace(grid=env.state.grid.at[p0[1], p0[0], 1].set(16 + 16))  # two tomatoes
    env._sync()
    assert "fits order 1" in env.observation_text(0) and "fits order 1 or 2" not in env.observation_text(0)
    env.state = env.state.replace(grid=env.state.grid.at[p0[1], p0[0], 1].set(16 + 16 + 16))  # three tomatoes
    env._sync()
    assert "fits no open order (wrong soup)" in env.observation_text(1)


def test_analyze_reports_start_failures_and_compliance_misses(tmp_path):
    from research.coop_ci import analyze
    be = ScriptedLLM()
    env = CoopEnv("shared_1pot", max_steps=60).reset(1)
    epi = Episode(env, 1, 0, ego_policy(PERSONAS, "neutral"), LLMHumanPolicy(be, Usage(), PERSONAS["onion_supplier"]["text"]),
                  PERSONAS["onion_supplier"])
    epi.run(60)
    os.makedirs(tmp_path / "traces")
    with open(tmp_path / "traces" / "c__shared_1pot__onion_supplier__0.jsonl", "w") as f:
        for r in epi.trace:
            f.write(json.dumps(r, default=str) + "\n")
    rows = analyze.compliance_miss_rows(analyze.read_traces(str(tmp_path)))
    assert rows and rows[0]["tendency"] == "onion" and "move" in rows[0]["chosen instead"]  # always 'move to 9 3'


def test_place_on_a_named_non_counter_uses_the_nearest_free_counter():
    env = CoopEnv("shared_1pot").reset(0)
    env.state = env.state.replace(agents=env.state.agents.replace(inventory=env.state.agents.inventory.at[0].set(4)))
    env._sync()
    ex = Executor(env, 0)
    ex.start(parse_macro("place on counter 1 0", env))  # (1, 0) is an onion pile
    assert ex.ms.status == "running" and ex.ms.target in env.counters and "nearest free one" in ex.ms.note


def test_infra_after_retry_counts_forced_waits():
    class AlwaysInvalid(MockBackend):
        def chat(self, messages, max_tokens=None, seed=None, json_mode=None):
            return LLMResult(json.dumps({"thought": "x", "action": "deliver"}), finish_reason="stop")
    env = CoopEnv("shared_1pot", max_steps=20).reset(0)
    epi = Episode(env, 0, 0, ego_policy(PERSONAS, "neutral"), LLMHumanPolicy(AlwaysInvalid(), Usage(), "x"),
                  PERSONAS["loader"])
    epi.run(20)
    m = epi.metrics()
    assert m["infra_after_retry"]["partner"] == 1.0 and m["infra_after_retry"]["ego"] == 0.0
    assert m["decision_points"]["partner"] == 20


def test_gate0_with_an_llm_partner_streams_results_and_reports_partial_runs(tmp_path, capsys):
    from research.coop_ci import validate
    cfg = tmp_path / "g.yaml"
    cfg.write_text("name: g\nseeds: 2\nt0: [0, 10]\nrollouts: 2\nhorizon: 20\npartner: scripted\n"
                   "families:\n  F1: {layouts: [shared_1pot]}\n")
    out, _ = validate.main([str(cfg), "--partner-backend", "mock", "--workers", "3", "--no-sanity",
                            "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "warm-ups" in printed and "rollouts]" in printed
    rows = [json.loads(l) for l in open(os.path.join(out, "rollouts.jsonl"))]
    assert len(rows) == 2 * 2 * 2 * 3 * 2
    # The branches of one state start from the same snapshot (identical warm-up).
    first = [r for r in rows if r["state"] == "0@10" and r["persona"] == "onion_supplier"]
    assert len(first) == 6 and {r["steps"] for r in first} == {20}
    with open(os.path.join(out, "rollouts.jsonl"), "w") as f:  # simulate a run still in progress
        for r in rows[:5]:
            f.write(json.dumps(r) + "\n")
    validate.main([str(cfg), "--report-only", out])
    assert "Interim report from 5 rollouts" in open(os.path.join(out, "report_interim.md")).read()


def test_put_back_returns_an_item_to_its_pile():
    env = CoopEnv("shared_1pot").reset(0)
    env.state = env.state.replace(agents=env.state.agents.replace(inventory=env.state.agents.inventory.at[0].set(4)))
    env._sync()  # agent 0 holds an onion
    assert "put back" in valid_actions(env, 0)
    ex = Executor(env, 0)
    ex.start(parse_macro("put it back", env))
    assert ex.ms.status == "running" and ex.ms.target in env.piles[0]
    for _ in range(10):
        if ex.idle():
            break
        env.step({0: ex.act()})
        ex.after_step()
    assert ex.ms.status == "done" and env.inventory(0) == 0
    assert env.events[-1].kind == "put_back" and "back on its pile" in env.observation_text(1)
    ex.start(parse_macro("put back", env))
    assert ex.ms.code == "precheck:no_item"


def test_analyze_recovers_placements_from_traces_of_older_runs(tmp_path):
    from research.coop_ci import analyze
    rec = {"episode_id": "A0|shared_1pot|onion_supplier|0", "ego_seat": 1, "waste": {"counter_items_end": 2}}
    os.makedirs(tmp_path / "traces")
    with open(tmp_path / "traces" / "A0__shared_1pot__onion_supplier__0.jsonl", "w") as f:
        for agent, kind in ((1, "counter_place"), (0, "counter_place"), (0, "put_back"), (1, "fetch")):
            f.write(json.dumps({"type": "event", "kind": kind, "agent": agent}) + "\n")
    analyze.fill_from_traces([rec], str(tmp_path))
    assert rec["waste"]["counter_places"] == {"ego": 1, "partner": 1}
    assert rec["waste"]["put_backs"] == {"ego": 0, "partner": 1}
    assert analyze.mean_known([None, None]) is None


def test_side_checks_for_plating_and_over_fetching():
    env, epi = scripted_episode("shared_1pot", "onion_supplier", seed=4)
    cl = epi.metrics()["clauses"]
    assert cl["partner:over_fetch"]["n"] > 0 and cl["partner:over_fetch"]["hit"] == 0  # the twin never over-fetches
    # Scripted cooks fetch a plate while the soup cooks, so a ready soup rarely waits for one.
    plate = [v for k, v in cl.items() if k.endswith(":plate_ready")]
    assert not plate or sum(v["hit"] for v in plate) / sum(v["n"] for v in plate) >= 0.9


def test_no_plating_ego_leaves_plating_to_the_partner():
    env, epi = scripted_episode("shared_1pot", "onion_supplier", kind="no_plating", seed=2)
    m = epi.metrics()
    assert m["n_correct"] >= 3  # the scripted twin plates (it fetches a plate while the soup cooks)
    assert not any(e.kind == "fetch" and e.item == "an empty plate" and e.agent == epi.ego for e in env.events)
