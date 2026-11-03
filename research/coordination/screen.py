"""Screen RQ3 layouts BEFORE LLM runs: can partner information change the score?

No model is called. For each partner profile t, two fixed ego rules are rolled
out from the same paired initial states the experiment uses: prep-first and
delivery-first (the partner's own rule with the other priority). Let BR(t) be
the better of the two against t. Then

    information_gap(t) = R(t, BR(t)) - R(t, BR(opposite of t))

is what an ego gains by acting on correct rather than opposite information,
using only these two simple rules. It is 0 when the same rule is best for both
partner types, i.e. the layout gives the correct/opposite manipulation no
room to matter. It is a diagnostic, not an upper bound on what an LLM can do.

A layout passes when information_gap >= --min-gap for every profile and the
failed-move rate stays below --max-failed-rate (navigation must not dominate).
Fix the criteria and the chosen layout before running the model.
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from .controller import Executor, candidates, preferred_task
from .run import views, successful_events, failed_moves, recipe_trackers
from .scenario import LAYOUTS, SUPPORTED_LAYOUTS, PROFILES, create_env, initial_state, primitive_key


def rollout(env, transition, seed, ego_rule, partner_profile, max_job_steps=32):
    obs, state = initial_state(env, seed)
    static = env.layout.static_objects
    executors = [Executor(static, i, max_job_steps, seed) for i in range(2)]
    trackers = recipe_trackers(env)
    rules = (ego_rule, partner_profile)
    shared_return, deliveries, wrong, failed, steps = 0.0, 0, 0, 0, 0
    for step in range(env.max_steps):
        current = views(env, obs, state, trackers)
        for i in range(2):
            executors[i].observe(current[i])
            if executors[i].job is None:
                menu = candidates(current[i], static)
                executors[i].assign(preferred_task(menu, rules[i], current[i]['self']['pos'], static,
                                                   current[i]), current[i])
        acts = {f'agent_{i}': executors[i].action(current[i]) for i in range(2)}
        before = state
        obs, state, rewards, done, _ = transition(primitive_key(seed, step), state, acts)
        shared_return += float(rewards['agent_0'])
        events = successful_events(before, state, acts, static)
        deliveries += sum(e['type'] == 'correct_delivery' for e in events)
        wrong += sum(e['type'] == 'wrong_delivery' for e in events)
        failed += len(failed_moves(before, state, acts, static))
        steps = step+1
        if done['__all__']:
            break
    return {'shared_return': shared_return, 'correct_deliveries': deliveries, 'wrong_deliveries': wrong,
            'failed_move_rate': failed / (2*steps)}


def screen_layout(layout, seeds, steps, max_job_steps=32):
    env = create_env(steps, 'full', layout)
    transition = jax.jit(env.step_env)
    table = {}
    for partner in PROFILES:
        for ego in PROFILES:
            runs = [rollout(env, transition, s, ego, partner, max_job_steps) for s in seeds]
            table[f'partner_{partner}/ego_{ego}_first'] = {
                'mean_return': float(np.mean([r['shared_return'] for r in runs])),
                'per_seed_return': [r['shared_return'] for r in runs],
                'mean_failed_move_rate': float(np.mean([r['failed_move_rate'] for r in runs])),
            }
    def ret(partner, ego):
        return table[f'partner_{partner}/ego_{ego}_first']['mean_return']
    best = {t: max(PROFILES, key=lambda e: ret(t, e)) for t in PROFILES}
    gaps = {}
    for t in PROFILES:
        other = 'delivery' if t == 'prep' else 'prep'
        gaps[t] = ret(t, best[t]) - ret(t, best[other])
    return {'layout': layout, 'seeds': list(seeds), 'steps': steps, 'table': table,
            'best_fixed_ego_rule': best, 'information_gap': gaps,
            'max_failed_move_rate': max(v['mean_failed_move_rate'] for v in table.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layouts', nargs='+', choices=SUPPORTED_LAYOUTS, default=sorted(LAYOUTS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--max-job-steps', type=int, default=32)
    p.add_argument('--min-gap', type=float, default=20.0, help='one soup = 20 reward')
    p.add_argument('--max-failed-rate', type=float, default=0.10)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    results = []
    for layout in args.layouts:
        r = screen_layout(layout, args.seeds, args.steps, args.max_job_steps)
        r['passes'] = (min(r['information_gap'].values()) >= args.min_gap
                       and r['max_failed_move_rate'] < args.max_failed_rate)
        results.append(r)
        cells = '  '.join(f"{k.replace('partner_', 'P=').replace('/ego_', ' E=').replace('_first', '')}:"
                          f"{v['mean_return']:.1f}" for k, v in r['table'].items())
        gaps = ' '.join(f'{t}:{g:+.1f}' for t, g in r['information_gap'].items())
        print(f"{layout:16s} {cells} | gap {gaps} | failed-move {r['max_failed_move_rate']:.3f} "
              f"| {'PASS' if r['passes'] else 'fail'}", flush=True)
    if args.out:
        args.out.write_text(json.dumps({'criteria': {'min_gap': args.min_gap,
                                                     'max_failed_rate': args.max_failed_rate},
                                        'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
