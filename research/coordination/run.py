"""Paired RQ3 interventions: no/correct/opposite partner information.

Real results require --backend llm. --backend scripted is plumbing validation only.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import jax
import numpy as np

from .scenario import (PROFILES, CONDITIONS, LAYOUT_TEXT, create_env, initial_state,
                       state_hash, primitive_key, partner_information)
from .observation import decode_observation, planner_view, inventory_kind
from .controller import Executor, candidates, preferred_task
from .llm import ChatClient


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def positive(value):
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return v


def views(env, obs, state):
    # This is the ONLY privileged-to-policy boundary: public map and OWN pose.
    # Inventory, partner location and all dynamic cells are decoded from obs.
    return [decode_observation(obs[f'agent_{i}'], env.layout.static_objects,
                              (int(state.agents.pos.x[i]), int(state.agents.pos.y[i])),
                              n=env.layout.num_ingredients, view_size=env.agent_view_size)
            for i in range(2)]


def successful_events(before, after, acts, static):
    """Evaluator-only primitive action events; never passed to the planner."""
    events = []
    directions = [(0,-1), (0,1), (1,0), (-1,0)]
    for i in range(2):
        if int(acts[f'agent_{i}']) != 5:
            continue
        inv0, inv1 = int(before.agents.inventory[i]), int(after.agents.inventory[i])
        dx, dy = directions[int(before.agents.dir[i])]
        x = int(before.agents.pos.x[i])+dx
        y = int(before.agents.pos.y[i])+dy
        target = int(static[y, x])
        if inv0 == 15 and inv1 == 0 and target == 4:
            events.append({'agent': i, 'type': 'correct_delivery'})
        if inv0 == 4 and inv1 == 0 and target == 5:
            events.append({'agent': i, 'type': 'ingredient_added'})
    return events


def source_digest():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def trial(args, env, client, seed, profile, condition, path):
    path.mkdir(parents=True, exist_ok=False)
    obs, state = initial_state(env, seed)
    init_hash = state_hash(state)
    controllers = [Executor(env.layout.static_objects, i, args.max_job_steps, seed) for i in range(2)]
    transition = jax.jit(env.step_env)
    history, decisions = [], []
    counters = Counter()
    returns = 0.0
    writer = None
    last_step = 0
    finished = False
    summary = {'seed': seed, 'partner_profile': profile, 'condition': condition,
               'initial_state_sha256': init_hash, 'backend': args.backend,
               'observation': args.observation, 'status': 'running'}
    dump(path/'summary.json', summary)
    try:
        with (path/'steps.jsonl').open('w') as steps_log, (path/'decisions.jsonl').open('w') as choices_log, (path/'llm.jsonl').open('w') as llm_log:
            def audit(record):
                record['sim_step'] = last_step
                llm_log.write(json.dumps(record)+'\n')
                llm_log.flush()
                counters['llm_attempts'] += 1
                counters['llm_errors'] += int('error' in record)
            if args.video:
                import imageio.v2 as imageio
                from scripts.preview_teammates import frame
                writer = imageio.get_writer(path/'rollout.mp4', fps=8, codec='libx264', macro_block_size=2)
                writer.append_data(frame(state, 'RQ3 controlled task', 0, 0,
                                               f'DEBUG full-state video; policy observations: {args.observation}'))
            for step in range(args.steps):
                last_step = step
                current = views(env, obs, state)
                for i in range(2):
                    controllers[i].observe(current[i])
                if controllers[1].job is None:
                    menu = candidates(current[1], env.layout.static_objects)
                    task = preferred_task(menu, profile, current[1]['self']['pos'], env.layout.static_objects)
                    controllers[1].assign(task, current[1])
                if controllers[0].job is None:
                    if controllers[0].last_outcome:
                        history.append(controllers[0].last_outcome)
                    menu = candidates(current[0], env.layout.static_objects)
                    context = {
                        'step': step, 'steps_remaining': args.steps-step,
                        'observation': planner_view(current[0], env.layout.static_objects),
                        'partner_information': partner_information(profile, condition),
                        'recent_own_jobs': history[-4:], 'candidates': menu,
                    }
                    if args.backend == 'llm':
                        choice = client.choose(context, seed, audit)
                    else:
                        # Intentionally ignores intervention text: validates mechanics,
                        # never manufactures a positive partner-information effect.
                        task = preferred_task(menu, 'prep', current[0]['self']['pos'], env.layout.static_objects)
                        choice = {'task_id': task['id'], 'partner_prediction': 'wait',
                                  'reason': 'Scripted plumbing check; ignores partner information.'}
                    task = next(c for c in menu if c['id'] == choice['task_id'])
                    controllers[0].assign(task, current[0])
                    actual = controllers[1].job.task['kind']
                    opportunity = {'prep', 'delivery'} <= {c['kind'] for c in menu}
                    # Ground truth is logged AFTER the planner call; absent from context.
                    record = {'step': step, 'context': context, 'choice': choice,
                              'chosen_kind': task['kind'], 'partner_current_kind': actual,
                              'partner_prediction_correct': choice['partner_prediction'] == actual,
                              'both_work_types_available': opportunity}
                    choices_log.write(json.dumps(record)+'\n')
                    choices_log.flush()
                    decisions.append(record)
                    history.append({'step': step, 'selected_task': task['id'],
                                    'visible_partner': current[0]['others']})
                acts = {f'agent_{i}': controllers[i].action(current[i]) for i in range(2)}
                jobs = [controllers[i].job.task for i in range(2)]
                same_target = jobs[0]['target'] is not None and jobs[0]['target'] == jobs[1]['target']
                # Conservative duplicate demand: same soup, or both preparing the
                # final missing ingredient for one pot. Not all joint prep is waste.
                demand_conflict = False
                if same_target and jobs[0]['kind'] == jobs[1]['kind']:
                    x, y = jobs[0]['target']
                    packed = int(state.grid[y, x, 1]) >> 2
                    demand_conflict = jobs[0]['kind'] == 'delivery' or (
                        jobs[0]['kind'] == 'prep' and (packed & 3) == 2)
                before = state
                obs, state, rewards, done, _ = transition(primitive_key(seed, step), state, acts)
                returns += float(rewards['agent_0'])
                counters['duplicate_demand_steps'] += int(demand_conflict)
                events = successful_events(before, state, acts, env.layout.static_objects)
                for event in events:
                    counters[f"agent_{event['agent']}_{event['type']}"] += 1
                static = env.layout.static_objects
                for i in range(2):
                    counters[f'agent_{i}_stay_steps'] += int(acts[f'agent_{i}'] == 4)
                    # Navigation diagnostic: a move toward free floor that did not
                    # change position (agent collision/blocked). Turns are excluded.
                    a = int(acts[f'agent_{i}'])
                    if a < 4:
                        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[a]
                        x0, y0 = int(before.agents.pos.x[i]), int(before.agents.pos.y[i])
                        if int(static[y0+dy, x0+dx]) == 0 and (x0, y0) == (
                                int(state.agents.pos.x[i]), int(state.agents.pos.y[i])):
                            counters[f'agent_{i}_failed_move_steps'] += 1
                record = {'step': step+1, 'actions': acts, 'jobs': jobs,
                          'rewards': {k: float(v) for k, v in rewards.items()},
                          'events': events, 'duplicate_demand': demand_conflict,
                          'positions': [[int(state.agents.pos.x[i]), int(state.agents.pos.y[i])] for i in range(2)],
                          'done': bool(done['__all__'])}
                steps_log.write(json.dumps(record)+'\n')
                if writer is not None:
                    writer.append_data(frame(state, 'RQ3 controlled task', step+1, returns,
                                                   f'DEBUG full-state video; policy observations: {args.observation}'))
                last_step = step+1
                if record['done']:
                    finished = True
                    break
        if not finished:
            raise RuntimeError('Episode did not terminate at horizon')
        summary['status'] = 'completed'
    except Exception as e:
        summary['status'] = 'error'
        summary['error'] = f'{type(e).__name__}: {e}'
        (path/'error.log').write_text(traceback.format_exc())
    finally:
        if writer is not None:
            writer.close()
        summary.update({'steps': last_step, 'shared_return': returns, 'metrics': dict(counters),
                        'correct_deliveries': sum(counters[f'agent_{i}_correct_delivery'] for i in range(2)),
                        'decision_count': len(decisions),
                        'choice_opportunities': sum(d['both_work_types_available'] for d in decisions),
                        'first_choice': decisions[0]['choice']['task_id'] if decisions else None,
                        'partner_prediction_accuracy': np.mean([d['partner_prediction_correct'] for d in decisions]).item() if decisions else None})
        dump(path/'summary.json', summary)
    return summary


def summarize(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('*/summary.json'))]
    pairs, excluded = [], []
    groups = {(r['seed'], r['partner_profile']) for r in rows}
    for seed, profile in sorted(groups):
        group = {r['condition']: r for r in rows if r['seed'] == seed and r['partner_profile'] == profile}
        if set(group) != set(CONDITIONS) or any(r['status'] != 'completed' for r in group.values()):
            excluded.append({'seed': seed, 'profile': profile, 'reason': 'missing or failed condition'})
            continue
        if len({r['initial_state_sha256'] for r in group.values()}) != 1:
            raise RuntimeError('Conditions did not share the same initial state')
        pairs.append({'seed': seed, 'profile': profile,
                      'correct_minus_none': group['correct']['shared_return']-group['none']['shared_return'],
                      'correct_minus_opposite': group['correct']['shared_return']-group['opposite']['shared_return'],
                      'first_choice_changed_correct_vs_opposite': group['correct']['first_choice'] != group['opposite']['first_choice']})
    result = {'trials': rows, 'paired_effects': pairs, 'excluded_pairs': excluded,
              'completed_trials': sum(r['status'] == 'completed' for r in rows),
              'failed_trials': sum(r['status'] != 'completed' for r in rows),
              'note': 'Paired diagnostic effects, not statistical significance. Scripted runs are plumbing checks only. '
                      'This tests supplied partner information, not continual learning.'}
    dump(out/'results.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('llm', 'scripted'), default='llm')
    p.add_argument('--base-url', default=os.environ.get('ICRL_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--model', default=os.environ.get('ICRL_LLM_MODEL', 'qwen35-27b'))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--steps', type=positive, default=200)
    p.add_argument('--max-job-steps', type=positive, default=32)
    p.add_argument('--observation', choices=('full', 'partial'), default='full')
    p.add_argument('--max-tokens', type=positive, default=512)
    p.add_argument('--timeout', type=positive, default=120)
    p.add_argument('--video', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if any(s < 0 or s >= 2**32 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be unique integers in [0, 2**32)')
    if len(set(args.profiles)) != len(args.profiles) or len(set(args.conditions)) != len(args.conditions):
        p.error('profiles/conditions must not contain duplicates')
    if args.out.exists():
        p.error('Output path exists. Choose a fresh --out to avoid overwriting experiments.')
    args.out.mkdir(parents=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    metadata = {**vars(args), 'out': str(args.out), 'git_commit': revision,
                'source_sha256': source_digest(), 'layout': LAYOUT_TEXT,
                'public_information': ['static map', 'fixed recipe 0,0,0', 'own pose'],
                'partner_access': args.observation, 'ego_access': args.observation,
                'intervention': 'supplied partner-priority description only',
                'method_scope': 'RQ3 controlled oracle-information diagnostic; no learned memory'}
    dump(args.out/'config.json', metadata)
    client = ChatClient(args.base_url, args.model, args.max_tokens, args.timeout) if args.backend == 'llm' else None
    if client is not None:
        # Fail before any rollout when URL/model is misconfigured.
        client.check_model()
    env = create_env(args.steps, args.observation)
    failed = False
    for seed in args.seeds:
        for profile in args.profiles:
            for condition in args.conditions:
                path = args.out/f'seed{seed}_{profile}_{condition}'
                result = trial(args, env, client, seed, profile, condition, path)
                print(json.dumps({k: result.get(k) for k in ('seed', 'partner_profile', 'condition', 'status', 'shared_return', 'error')}), flush=True)
                failed |= result['status'] != 'completed'
    result = summarize(args.out)
    print(f"Completed {result['completed_trials']} trials; failed {result['failed_trials']}. Results: {args.out/'results.json'}")
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
