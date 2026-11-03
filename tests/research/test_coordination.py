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
        create_env(layout_name='fun_coordination')  # needs counter hand-offs


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



# ---------------------------------------------------------------- multi-recipe (Stage 1)
MIXED = 'long_three_pot_mixed'


def _counts(value, n=2):
    return [(int(value) >> (2 + 2*i)) & 3 for i in range(n)]


def test_recipe_comes_from_observation_and_is_remembered():
    from research.coordination.run import recipe_trackers
    full = create_env(40, 'full', MIXED)
    for seed in range(4):
        obs, s = initial_state(full, seed)
        assert [v['recipe'] for v in views(full, obs, s)] == [_counts(s.recipe)] * 2
    part = create_env(40, 'partial', MIXED)
    obs, s = initial_state(part, 0)
    trackers = recipe_trackers(part)
    v = views(part, obs, s, trackers)
    # Indicator at x=0 is outside both radius-2 views at the start.
    assert v[0]['recipe'] is None and v[0]['known_recipe'] is None
    assert v[0]['recipe_info']['source'] == 'unknown'
    assert any(c['id'].startswith('inspect_recipe') for c in candidates(v[0], part.layout.static_objects))
    moved = s.replace(agents=s.agents.replace(pos=s.agents.pos.replace(x=s.agents.pos.x.at[0].set(1))))
    v = views(part, part.get_obs(moved), moved, trackers)
    assert v[0]['known_recipe'] == _counts(s.recipe) and v[0]['recipe_info']['source'] == 'observed_now'
    v = views(part, obs, s, trackers)  # back out of view: remembered, not re-read
    assert v[0]['recipe'] is None and v[0]['recipe_info']['source'] == 'remembered'
    assert v[0]['known_recipe'] == _counts(s.recipe)


def test_multi_recipe_initial_states_are_recipe_consistent():
    from research.coordination.controller import still_possible
    e = create_env(40, 'full', MIXED)
    pots = pot_positions(e.layout.static_objects)
    seen = set()
    for seed in range(6):
        _, s = initial_state(e, seed)
        recipe = _counts(s.recipe)
        assert recipe == [[2, 1], [1, 2]][seed % 2]  # recipes balanced by seed
        seen.add(tuple(recipe))
        cooking = [p for p in pots if int(s.grid[p[1], p[0], 2]) > 0]
        assert len(cooking) == 1 and _counts(s.grid[cooking[0][1], cooking[0][0], 1]) == recipe
        for x, y in pots:
            if (x, y) != cooking[0]:
                contents = _counts(s.grid[y, x, 1])
                assert sum(contents) <= 2 and still_possible(contents, recipe)
    assert len(seen) == 2


def _view(pot_contents, recipe, holding=None, pot=(3, 0), cooked=False, timer=0):
    inv = {'plate': holding in ('plate', 'dish'), 'cooked': holding == 'dish',
           'ingredients': list(holding) if isinstance(holding, list) else [0, 0]}
    return {'self': {'pos': [3, 1], 'direction': 0, 'inventory': inv}, 'others': [],
            'cells': {pot: {'plate': False, 'cooked': cooked, 'ingredients': list(pot_contents), 'timer': timer}},
            'recipe': recipe, 'known_recipe': recipe}


def test_multi_recipe_menu_and_scripted_choice():
    static = create_env(20, 'full', MIXED).layout.static_objects
    v = _view([2, 0], [2, 1])
    menu = candidates(v, static)
    ids = {c['id'] for c in menu}
    assert {'prep_3_0_ing0', 'prep_3_0_ing1'} <= ids  # both offered; planner must choose
    task = preferred_task(menu, 'prep', v['self']['pos'], static, v)
    assert task['id'] == 'prep_3_0_ing1'  # scripted rule adds only what the pot needs
    held = candidates(_view([2, 0], [2, 1], holding=[0, 1]), static)
    assert {c['id'] for c in held if c['kind'] == 'prep'} == {'prep_3_0_ing1'}
    wrong = _view([3, 0], [2, 1], cooked=True)
    wrong_menu = candidates(wrong, static)
    assert {'serve_3_0', 'discard_3_0'} <= {c['id'] for c in wrong_menu}
    assert preferred_task(wrong_menu, 'delivery', [3, 1], static, wrong)['id'] == 'discard_3_0'
    right = _view([2, 1], [2, 1], cooked=True)
    assert 'discard_3_0' not in {c['id'] for c in candidates(right, static)}
    # Single-ingredient menus keep their original ids.
    single = create_env(20, 'full').layout.static_objects
    one = {'self': {'pos': [2, 1], 'direction': 0, 'inventory': {'plate': False, 'cooked': False, 'ingredients': [0]}},
           'others': [], 'cells': {(2, 0): {'plate': False, 'cooked': False, 'ingredients': [1], 'timer': 0}},
           'recipe': None, 'known_recipe': [3]}
    assert 'prep_2_0' in {c['id'] for c in candidates(one, single)}


def test_redundant_prep_released_but_wrong_choice_kept():
    from research.coordination.controller import Executor
    static = create_env(20, 'full', MIXED).layout.static_objects
    task = {'id': 'prep_3_0_ing0', 'kind': 'prep', 'target': [3, 0], 'ingredient': 0}
    ex = Executor(static, 0)
    ex.assign(task, _view([1, 0], [2, 1]))       # needed when chosen
    ex.observe(_view([2, 0], [2, 1]))            # partner added it first
    assert ex.job is None and ex.last_outcome['outcome'] == 'target_changed'
    ex.assign(task, _view([2, 0], [2, 1]))       # unneeded from the start
    ex.observe(_view([2, 0], [2, 1]))
    assert ex.job is not None                    # executed as chosen; mistake stays visible


def _step_until(env, state, obs, ex, task, steps=80):
    from research.coordination.run import successful_events, recipe_trackers
    from research.coordination.scenario import primitive_key
    transition = jax.jit(env.step_env)
    trackers = recipe_trackers(env)
    events, total = [], 0.0
    v = views(env, obs, state, trackers)
    ex.assign(task, v[0])
    for t in range(steps):
        v = views(env, obs, state, trackers)
        ex.observe(v[0])
        if ex.job is None:
            break
        acts = {'agent_0': ex.action(v[0]), 'agent_1': 4}
        before = state
        obs, state, r, _, _ = transition(primitive_key(0, t), state, acts)
        total += float(r['agent_0'])
        events += successful_events(before, state, acts, env.layout.static_objects)
    return state, events, total


def test_discard_puts_wrong_soup_on_counter_without_penalty():
    from research.coordination.controller import Executor
    from research.coordination.scenario import encode_ingredients
    e = create_env(120, 'full', MIXED, 'empty')
    _, s = initial_state(e, 0)                    # recipe [2, 1]
    s = s.replace(grid=s.grid.at[0, 3, 1].set(encode_ingredients([0, 0, 0]) | 2))  # cooked wrong soup
    s, events, total = _step_until(e, s, e.get_obs(s), Executor(e.layout.static_objects, 0),
                                   {'id': 'discard_3_0', 'kind': 'recovery', 'target': [3, 0]})
    types = [ev['type'] for ev in events]
    assert 'dish_discarded' in types and 'wrong_delivery' not in types
    assert total == 0 and int(s.grid[0, 3, 1]) == 0  # pot freed at no cost


def test_wrong_delivery_detected_and_penalized():
    from research.coordination.controller import Executor
    from research.coordination.scenario import encode_ingredients
    e = create_env(60, 'full', MIXED, 'empty')
    _, s = initial_state(e, 0)                    # recipe [2, 1]
    wrong_dish = encode_ingredients([0, 0, 0]) | 3
    s = s.replace(agents=s.agents.replace(inventory=s.agents.inventory.at[0].set(wrong_dish)))
    s, events, total = _step_until(e, s, e.get_obs(s), Executor(e.layout.static_objects, 0),
                                   {'id': 'deliver', 'kind': 'delivery', 'target': None})
    assert [ev['type'] for ev in events] == ['wrong_delivery'] and total == -20


def test_multi_recipe_trial_uses_recipe_prompt_and_schema(tmp_path):
    from research.coordination.llm import SYSTEM_MULTI_RECIPE
    class Recorder:
        systems, contexts = [], []
        def choose(self, context, seed, audit, system=None):
            self.systems.append(system)
            self.contexts.append(context)
            task = preferred_task(context['candidates'], 'prep', [5, 2], e.layout.static_objects)
            return {'task_id': task['id'], 'partner_prediction': 'prep', 'reason': 'test'}
    e = create_env(80, 'full', MIXED)
    args = SimpleNamespace(backend='llm', steps=80, max_job_steps=32, observation='full', video=False)
    rec = Recorder()
    r = trial(args, e, rec, 1, 'delivery', 'correct', tmp_path/'mixed')
    assert r['status'] == 'completed', r
    assert set(rec.systems) == {SYSTEM_MULTI_RECIPE}
    first = rec.contexts[0]['observation']
    assert first['schema_version'] == 3 and 'public_recipe' not in first
    assert first['recipe']['ingredient_counts'] == {'ingredient_0': 1, 'ingredient_1': 2}
    assert r['recipe_counts'] == [1, 2]
    assert r['shared_return'] == 20 * (r['correct_deliveries'] - r['wrong_deliveries'])


def test_screen_mixed_layout_information_gap():
    from research.coordination.screen import screen_layout
    r = screen_layout(MIXED, seeds=[0, 1], steps=200)
    assert min(r['information_gap'].values()) >= 20, r
    assert r['max_failed_move_rate'] < 0.1, r


def test_diagnose_reports_trials_pairs_and_llm_errors(tmp_path, capsys):
    import sys
    from research.coordination import diagnose
    e = create_env(40, 'full', MIXED)
    args = SimpleNamespace(backend='scripted', steps=40, max_job_steps=32, observation='full', video=False)
    for condition in ('none', 'correct', 'opposite'):
        trial(args, e, None, 0, 'prep', condition, tmp_path/f'seed0_prep_{condition}')
    first = json.loads((tmp_path/'seed0_prep_none/decisions.jsonl').read_text().splitlines()[0])
    bad = {'attempt': 1, 'sim_step': 0, 'error': 'Invalid task_id',
           'request': {'messages': [{}, {'content': json.dumps(first['context'])}]},
           'response': {'choices': [{'message': {'content': '{"task_id": "invented"}'}}]}}
    (tmp_path/'seed0_prep_none/llm.jsonl').write_text(json.dumps(bad)+'\n')
    errors = diagnose.llm_errors(str(tmp_path))
    assert errors[0]['model_output'] == '{"task_id": "invented"}'
    assert errors[0]['valid_task_ids'] == [c['id'] for c in first['context']['candidates']]
    assert diagnose.paired(diagnose.trial_rows(str(tmp_path)))[0]['correct_minus_none'] == 0
    sys.argv = ['diagnose', str(tmp_path)]
    diagnose.main()
    assert 'LLM errors: 1' in capsys.readouterr().out
