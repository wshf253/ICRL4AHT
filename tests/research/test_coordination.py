import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from research.coordination.scenario import (LAYOUTS, create_env, initial_state, state_hash,
                                           partner_information, pot_positions)
from research.coordination.observation import decode_observation, planner_view
from research.coordination.controller import candidates, preferred_task
from research.coordination.llm import ChatClient, ModelError, parse_choice
from research.coordination.run import trial, summarize, views


@pytest.fixture(scope='module')
def env():
    return create_env(80, 'full')


def test_paired_states_and_initial_choice_opportunity(env):
    for seed in (0, 1, 2):
        obs, s = initial_state(env, seed)
        assert state_hash(s) == state_hash(initial_state(env, seed)[1])
        for i, v in enumerate(views(env, obs, s)):
            menu = candidates(v, env.layout.static_objects)
            assert {'prep', 'delivery'} <= {c['kind'] for c in menu}
            assert preferred_task(menu, 'prep', v['self']['pos'], env.layout.static_objects)['kind'] == 'prep'
            assert preferred_task(menu, 'delivery', v['self']['pos'], env.layout.static_objects)['kind'] == 'delivery'
    assert partner_information('prep', 'correct') == partner_information('delivery', 'opposite')
    assert partner_information('prep', 'none') == partner_information('delivery', 'none')


def test_partial_observation_has_no_hidden_dynamic_state_leak():
    e = create_env(20, 'partial')
    obs, s = initial_state(e, 0)
    pose = (int(s.agents.pos.x[0]), int(s.agents.pos.y[0]))
    # x=6 is outside agent 0's radius-2 view, as is partner at (5,3).
    s2 = s.replace(grid=s.grid.at[1, 6, 1].set(15),
                   agents=s.agents.replace(pos=s.agents.pos.replace(
                       x=s.agents.pos.x.at[1].set(5), y=s.agents.pos.y.at[1].set(3))))
    s3 = s2.replace(grid=s2.grid.at[1, 6, 1].set(0),
                    agents=s2.agents.replace(inventory=s2.agents.inventory.at[1].set(15)))
    def encoded(state):
        return planner_view(decode_observation(e.get_obs(state)['agent_0'],
                            e.layout.static_objects, pose, n=1, view_size=2),
                            e.layout.static_objects)
    assert encoded(s2) == encoded(s3)
    assert encoded(s2)['visible_partner'] == []
    station = next(v for v in encoded(s2)['stations'] if v['pos'] == [6, 1])
    assert station['contents'] is None and not station['observed_now']


def test_choice_validation():
    menu = [{'id': 'wait'}]
    good = {'task_id': 'wait', 'partner_prediction': 'prep', 'reason': 'test'}
    assert parse_choice(json.dumps(good), menu) == good
    with pytest.raises(ModelError):
        parse_choice(json.dumps({**good, 'task_id': 'invented'}), menu)
    with pytest.raises(ModelError):
        parse_choice('<think>still thinking', menu)


def test_scripted_pairing_and_deliveries(env, tmp_path):
    args = SimpleNamespace(backend='scripted', steps=80, max_job_steps=32,
                           observation='full', video=False)
    rows = []
    for condition in ('none', 'correct', 'opposite'):
        rows.append(trial(args, env, None, 0, 'delivery', condition, tmp_path/condition))
    assert all(r['status'] == 'completed' for r in rows), rows
    assert all(r['correct_deliveries'] > 0 for r in rows), rows
    assert len({r['shared_return'] for r in rows}) == 1  # mock doesn't use description
    assert (tmp_path/'none/steps.jsonl').read_bytes() == (tmp_path/'correct/steps.jsonl').read_bytes()
    contexts = []
    for condition in ('none', 'correct', 'opposite'):
        first = json.loads((tmp_path/condition/'decisions.jsonl').read_text().splitlines()[0])
        context = first['context']
        context.pop('partner_information')
        contexts.append(context)
    assert contexts[0] == contexts[1] == contexts[2]
    summary = summarize(tmp_path)
    assert summary['paired_effects'][0]['correct_minus_none'] == 0
    assert summary['failed_trials'] == 0


def test_http_adapter_and_error_accounting(env, tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def respond(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.respond({'data': [{'id': 'mock-qwen'}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            context = json.loads(body['messages'][1]['content'])
            task = next((c for c in context['candidates'] if c['kind'] == 'prep'), context['candidates'][0])
            choice = {'task_id': task['id'], 'partner_prediction': 'delivery', 'reason': 'HTTP integration test'}
            self.respond({'choices': [{'message': {'content': json.dumps(choice)}, 'finish_reason': 'stop'}]})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ChatClient(f'http://127.0.0.1:{server.server_port}/v1', 'mock-qwen')
        assert client.check_model() == ['mock-qwen']
        args = SimpleNamespace(backend='llm', steps=80, max_job_steps=32,
                               observation='full', video=False)
        result = trial(args, env, client, 0, 'delivery', 'correct', tmp_path/'http')
        assert result['status'] == 'completed', result
        assert result['correct_deliveries'] > 0
        assert requests and requests[0]['chat_template_kwargs'] == {'enable_thinking': False}
        context = json.loads(requests[0]['messages'][1]['content'])
        assert 'partner_profile' not in context and 'partner_current_kind' not in context
        class Broken:
            def choose(self, context, seed, audit):
                raise ModelError('intentional failure')
        failed = trial(args, env, Broken(), 0, 'delivery', 'none', tmp_path/'failed')
        assert failed['status'] == 'error' and failed['steps'] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_executor_breaks_symmetric_navigation_livelock(tmp_path):
    # Regression: identical deterministic executors used to step into the same
    # free cell forever (seed 0 delivery: one delivery in 200 steps) or yield and
    # return in lockstep when each stood on the other's only access cell.
    e = create_env(200, 'full')
    args = SimpleNamespace(backend='scripted', steps=200, max_job_steps=32,
                           observation='full', video=False)
    for seed, profile in ((0, 'delivery'), (1, 'delivery'), (0, 'prep'), (2, 'prep')):
        r = trial(args, e, None, seed, profile, 'none', tmp_path/f'{seed}_{profile}')
        assert r['status'] == 'completed', r
        assert r['correct_deliveries'] >= 5, r
        rows = [json.loads(l) for l in (tmp_path/f'{seed}_{profile}/steps.jsonl').read_text().splitlines()]
        stuck = longest = 0
        prev = None
        for row in rows:
            moving = all(row['actions'][f'agent_{i}'] < 4 for i in range(2))
            stuck = stuck+1 if moving and row['positions'] == prev else 0
            longest = max(longest, stuck)
            prev = row['positions']
        assert longest < 8, (seed, profile, longest)


def test_layout_selection_and_initialization():
    from research.coordination.scenario import SUPPORTED_LAYOUTS
    for name in SUPPORTED_LAYOUTS:
        for start in ('controlled', 'empty'):
            e = create_env(20, 'full', name, start)
            _, state = initial_state(e, 0)
            grid = np.asarray(state.grid)
            timers = grid[:, :, 2]
            assert not np.any(timers[np.asarray(e.layout.static_objects) != 5])
            assert np.count_nonzero(timers) == (start == 'controlled')
            assert state_hash(state) == state_hash(initial_state(e, 0)[1])
    with pytest.raises(ValueError):
        create_env(layout_name='cramped_room_up')


def test_explicit_planner_state_encoding():
    from research.coordination.observation import planner_agent, planner_contents
    def inv(n, plate=False, cooked=False):
        return dict(ingredients=[n], plate=plate, cooked=cooked)
    for value, expected in [(inv(0), 'empty'), (inv(1), 'ingredient_0'),
                             (inv(0, True), 'empty_plate'), (inv(3, True, True), 'cooked_soup_on_plate')]:
        raw = dict(pos=[1, 2], direction=0, inventory=value)
        encoded = planner_agent(raw)
        assert encoded['holding'] == expected
        assert encoded['ingredient_counts'] == {'ingredient_0': value['ingredients'][0]}
        assert 'inventory' not in encoded
        assert raw['inventory'] == value
    for count, timer, cooked, status in [(0, 0, False, 'empty'), (1, 0, False, 'filling'),
                                        (3, 19, False, 'cooking'), (3, 0, True, 'ready')]:
        c = planner_contents(dict(**inv(count, cooked=cooked), timer=timer), 'pot')
        assert c['status'] == status and c['remaining_cook_steps'] == timer
    assert planner_contents(None, 'pot') is None


def test_role_matrix_pairing_and_censored_times():
    from research.coordination.role_matrix import aggregate
    rows = []
    for partner in ('prep', 'delivery'):
        for ego in ('prep', 'delivery'):
            rows.append(dict(partner_profile=partner, ego_profile=ego, seed=0,
                status='completed', initial_state_sha256='same', shared_return=20 if ego=='prep' else 40,
                correct_deliveries=1 if ego=='prep' else 2,
                target_reached=ego=='delivery', target_delivery_step=80 if ego=='delivery' else None))
    result = aggregate(rows, [0])
    assert all(p['delivery_minus_prep_return'] == 20 for p in result['paired_effects'])
    assert all(p['delivery_minus_prep_target_steps'] is None for p in result['paired_effects'])
    rows[0]['status'] = 'error'
    assert len(aggregate(rows, [0])['excluded_pairs']) == 1

def _pilot_initial_grid(env, seed):
    # The original two_pot_small initializer, kept verbatim as a reference.
    from envs.overcooked_v2.common import DynamicObject
    _, state = env.reset(jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    cooking_x = int(rng.choice([2, 4]))
    timer = int(rng.integers(5, 11))
    count = int(rng.integers(0, 3))
    ing = int(DynamicObject.ingredient(0))
    grid = state.grid.at[0, cooking_x, 1].set(3 * ing)
    grid = grid.at[0, cooking_x, 2].set(timer)
    return grid.at[0, 6 - cooking_x, 1].set(count * ing)


def test_default_layout_initial_states_unchanged(env):
    for seed in range(6):
        assert np.array_equal(initial_state(env, seed)[1].grid, _pilot_initial_grid(env, seed))


@pytest.mark.parametrize('layout', sorted(LAYOUTS))
def test_layouts_pair_states_and_offer_both_jobs(layout):
    for observation in ('full', 'partial'):
        e = create_env(40, observation, layout)
        pots = pot_positions(e.layout.static_objects)
        for seed in (0, 1, 2):
            obs, s = initial_state(e, seed)
            assert state_hash(s) == state_hash(initial_state(e, seed)[1])
            timers = [int(s.grid[y, x, 2]) for x, y in pots]
            assert sum(t > 0 for t in timers) == 1  # exactly one pot already cooking
            if observation == 'full':
                for v in views(e, obs, s):
                    kinds = {c['kind'] for c in candidates(v, e.layout.static_objects)}
                    assert {'prep', 'delivery'} <= kinds


def test_screen_reports_information_gap():
    from research.coordination.screen import screen_layout
    r = screen_layout('long_three_pot', seeds=[0], steps=200)
    assert set(r['information_gap']) == {'prep', 'delivery'}
    # Screened design property of this layout: correct vs opposite information
    # changes the best simple ego rule's return by at least one soup each way.
    assert min(r['information_gap'].values()) >= 20, r
    assert r['max_failed_move_rate'] < 0.1, r
