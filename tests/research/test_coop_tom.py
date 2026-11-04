"""Offline tests for the cooperative-intelligence harness (no model, no GPU)."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from research.coop_tom.env import A_STAY, CoopEnv, multiset_contains
from research.coop_tom.llm import OpenAICompatBackend
from research.coop_tom.planner import Executor, parse_macro
from research.coop_tom.run import CONFIG_DIR, load_config
from research.coop_tom.runner import Backends, load_yaml, run_episode

PERSONAS = load_yaml(os.path.join(CONFIG_DIR, "personas.yaml"))
SPECS = load_yaml(os.path.join(CONFIG_DIR, "backends.yaml"))


def exp1(**over):
    cfg = load_config(os.path.join(CONFIG_DIR, "exp1_diagnosis.yaml"))
    cfg["env"]["max_steps"] = over.pop("max_steps", 120)
    cfg["backends"]["agent"] = "mock"
    cfg.update(over)
    return cfg


def episode(cfg, cond_name, persona, tmp_path, seed=7, backends=None):
    cond = next(c for c in cfg["conditions"] if c["name"] == cond_name)
    backends = backends or Backends(SPECS)
    os.makedirs(tmp_path, exist_ok=True)
    rec = run_episode(cfg, cond, persona, persona, seed, backends, PERSONAS, str(tmp_path), f"{cond_name}|{persona}|0")
    return rec, backends


def test_demo_cook_information_asymmetry():
    env = CoopEnv("demo_cook_simple", "mixed_01", max_steps=50).reset(0)
    assert env.visible_order(1) is None and env.visible_order(0) == env.current_recipe
    # The robot's room never comes within view of an order display.
    for x, y in [(8, 1), (8, 2), (8, 3), (9, 1), (9, 2), (9, 3)]:
        assert not any(max(abs(x - p[0]), abs(y - p[1])) <= env.view_radius for p in env.indicators)


def test_order_reaches_the_robot_prompt_only_in_oracle_conditions(tmp_path):
    cfg = exp1(max_steps=80)
    for cond, leaks in (("hid_hid", False), ("ord_hid", True)):
        _, b = episode(cfg, cond, "does_all", tmp_path / cond)
        decisions = [c for c in b.get("mock").calls if "PROBE" not in c[-1]["content"]]
        assert decisions
        texts = ["\n".join(m["content"] for m in c) for c in decisions]
        assert all(("Current order (given to you)" in t) == leaks for t in texts)
        assert not any("Order display (visible now)" in t for t in texts)


def test_probes_are_forked_and_never_fed_back(tmp_path):
    _, b = episode(exp1(max_steps=80), "hid_hid", "starter", tmp_path)
    calls = b.get("mock").calls
    probes = [c for c in calls if "PROBE" in c[-1]["content"]]
    decisions = [c for c in calls if "PROBE" not in c[-1]["content"]]
    assert probes and decisions
    assert not any("PROBE" in m["content"] for c in decisions for m in c)


def test_conditions_share_seeds_and_order_sequences():
    a = CoopEnv("demo_cook_simple", "mixed_01").reset(123)
    b = CoopEnv("demo_cook_simple", "mixed_01").reset(123)
    assert a.recipe_seq == b.recipe_seq
    assert CoopEnv("demo_cook_simple", "mixed_01").reset(124).recipe_seq != a.recipe_seq


@pytest.mark.parametrize("persona", ["does_all", "starter", "slow", "starter_minority", "minority_first"])
def test_oracle_reference_and_scripted_partner_are_sound(persona, tmp_path):
    cfg = exp1(max_steps=300, probe={"enabled": False})
    rec, _ = episode(cfg, "ref_rule_oracle", persona, tmp_path, seed=3)
    assert rec["n_correct"] >= 3 and rec["n_wrong"] == 0, rec
    # The scripted partner follows its persona: it never adds an ingredient that leaves
    # the pot inconsistent with the order it was cooking for.
    for soup in rec["soups"]:
        target = soup["target"].split("+")
        so_far = []
        for _, who, ing in soup["adds"]:
            so_far.append(ing)
            if who == "partner":
                assert multiset_contains(target, so_far), soup


def test_partner_waits_for_the_new_order_after_a_soup_is_plated(tmp_path):
    cfg = exp1(max_steps=300, probe={"enabled": False})
    for persona in ("does_all", "starter"):
        rec, _ = episode(cfg, "ref_rule_oracle", persona, tmp_path / persona, seed=5)
        trace = [json.loads(l) for l in open(tmp_path / persona / "traces" / f"ref_rule_oracle__{persona}__0.jsonl")]
        plated = False
        for e in (r for r in trace if r["type"] == "event"):
            plated = (plated or e["kind"] == "plate_soup") and e["kind"] != "deliver"
            assert not (plated and e["kind"] == "pot_add" and e["agent"] == 0), e  # would use the old order


def test_rule_reference_waits_for_evidence(tmp_path):
    cfg = exp1(max_steps=300, probe={"enabled": False})
    rec, _ = episode(cfg, "ref_rule", "does_all", tmp_path, seed=3)
    assert rec["n_wrong"] == 0 and rec["n_correct"] >= 3, rec


def test_put_in_pot_replans_when_the_pot_changed():
    env = CoopEnv("demo_cook_simple", "mixed_01").reset(0)
    st = env.state
    # Robot next to the pot, facing it (left), holding an onion.
    st = st.replace(agents=st.agents.replace(dir=st.agents.dir.at[1].set(3),
                                             inventory=st.agents.inventory.at[1].set(4)))
    env.state = st
    env._sync()
    ex = Executor(env, 1)
    ex.start(parse_macro("put in pot", env))
    assert ex.ms.status == "running"
    env.state = env.state.replace(grid=env.state.grid.at[2, 7, 1].set(16))  # partner added a tomato meanwhile
    env._sync()
    assert ex.act() == A_STAY
    assert ex.ms.status == "failed" and ex.ms.reason.startswith("the pot changed")


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
        msg = {"content": json.dumps({"action": "wait 1"}),
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
    from research.coop_tom import check
    report = tmp_path / "check.jsonl"
    check.main(["--base-url", fake_vllm, "--model", "qwen3.8-27b", "--pick", "--report", str(report)])
    out = capsys.readouterr()
    assert json.loads(out.out.strip()) == {"enable_thinking": False}  # minimal switch preferred
    rows = [json.loads(l) for l in report.read_text().splitlines()]
    assert [r["passes"] for r in rows[:3]] == [True, False, True]
    assert "HTTP 400" in rows[1]["error"]           # reasoning_effort "none" rejected
    assert rows[3]["thinking"] and not rows[3]["error"] and "warning" not in out.err  # detector verified


def test_cli_end_to_end_with_served_model(fake_vllm, tmp_path, monkeypatch):
    from research.coop_tom import analyze, run
    monkeypatch.setenv("COOP_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    out = run.main([os.path.join(CONFIG_DIR, "exp0_smoke.yaml"), "--episodes", "1", "--max-steps", "30",
                    "--conditions", "llm_direct", "--personas", "does_all", "--workers", "1", "--out", str(tmp_path)])
    rec = json.loads(open(os.path.join(out, "episodes.jsonl")).readline())
    assert rec["usage"]["agent"]["calls"] > 0 and rec["usage"]["agent"]["thinking"] == 0
    body = FakeVLLM.requests[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"] == {"type": "json_object"} and "seed" in body and body["top_k"] == 20
    resolved = json.load(open(os.path.join(out, "config.resolved.json")))
    assert resolved["chat_template_kwargs_override"] == '{"enable_thinking": false}'
    monkeypatch.setattr("sys.argv", ["analyze", out])
    analyze.main()
    assert os.path.exists(os.path.join(out, "analysis", "summary.csv"))


def test_cli_rejects_a_model_that_is_not_served(fake_vllm, tmp_path, monkeypatch):
    from research.coop_tom import run
    monkeypatch.setenv("ICRL_LLM_MODEL", "some-other-model")
    with pytest.raises(SystemExit):
        run.main([os.path.join(CONFIG_DIR, "exp0_smoke.yaml"), "--episodes", "1", "--out", str(tmp_path)])
