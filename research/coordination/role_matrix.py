"""Compare two scripted robot priorities against both partner priorities; no LLM."""
import argparse
import json
from pathlib import Path
import subprocess
from statistics import mean

from .scenario import PROFILES, SUPPORTED_LAYOUTS, create_env
from .run import trial, dump, positive, source_digest


def timing(path, target):
    deliveries = []
    for line in (path / 'steps.jsonl').read_text().splitlines():
        row = json.loads(line)
        deliveries.extend(row['step'] for event in row['events']
                          if event['type'] == 'correct_delivery')
    return {'delivery_steps': deliveries,
            'first_delivery_step': deliveries[0] if deliveries else None,
            'target_deliveries': target,
            'target_reached': len(deliveries) >= target,
            'target_delivery_step': deliveries[target-1] if len(deliveries) >= target else None}


def aggregate(rows, seeds):
    cells, pairs, excluded = [], [], []
    for partner in PROFILES:
        for ego in PROFILES:
            group = [r for r in rows if r['partner_profile'] == partner and r['ego_profile'] == ego]
            good = [r for r in group if r['status'] == 'completed']
            reached = [r for r in good if r['target_reached']]
            cells.append({'partner_profile': partner, 'ego_profile': ego,
                          'completed': len(good), 'failed': len(group)-len(good),
                          'mean_return': mean(r['shared_return'] for r in good) if good else None,
                          'mean_deliveries': mean(r['correct_deliveries'] for r in good) if good else None,
                          'target_reached_trials': len(reached),
                          'mean_target_step_among_reached': mean(r['target_delivery_step'] for r in reached) if reached else None})
        for seed in seeds:
            group = {r['ego_profile']: r for r in rows if r['partner_profile'] == partner and r['seed'] == seed}
            if len(group) != 2 or any(r['status'] != 'completed' for r in group.values()):
                excluded.append({'seed': seed, 'partner_profile': partner})
                continue
            a, b = group['prep'], group['delivery']
            if a['initial_state_sha256'] != b['initial_state_sha256']:
                raise ValueError('Unmatched initial states')
            pairs.append({'seed': seed, 'partner_profile': partner,
                          'delivery_minus_prep_return': b['shared_return']-a['shared_return'],
                          'delivery_minus_prep_target_steps': b['target_delivery_step']-a['target_delivery_step']
                          if a['target_reached'] and b['target_reached'] else None})
    return {'cells': cells, 'paired_effects': pairs, 'excluded_pairs': excluded,
            'trials': rows,
            'note': 'Two heuristic priorities only, not an oracle or ToM result. Unreached target times are null, not zero. Empty resets may share identical states across seeds.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', choices=SUPPORTED_LAYOUTS, default='counter_circuit')
    p.add_argument('--start', choices=('empty', 'controlled'), default='empty')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--steps', type=positive, default=400)
    p.add_argument('--target-deliveries', type=positive, default=4)
    p.add_argument('--max-job-steps', type=positive, default=32)
    p.add_argument('--observation', choices=('full', 'partial'), default='full')
    p.add_argument('--video', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error('Seeds must be unique integers in [0, 2**32)')
    if args.out.exists():
        p.error('Choose a fresh output directory')
    args.out.mkdir(parents=True)
    args.backend = 'scripted'
    dump(args.out/'config.json', {**vars(args), 'out': str(args.out),
         'source_sha256': source_digest(),
         'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
         'scope': '2 robot priorities x 2 partner priorities; same executor; no LLM'})
    env = create_env(args.steps, args.observation, args.layout, args.start)
    rows = []
    for seed in args.seeds:
        for partner in PROFILES:
            for ego in PROFILES:
                args.ego_profile = ego
                path = args.out/f'seed{seed}_partner-{partner}_robot-{ego}'
                row = trial(args, env, None, seed, partner, 'none', path)
                row.update(timing(path, args.target_deliveries))
                row['partner_prediction_accuracy'] = None  # no predictor in this experiment
                dump(path/'summary.json', row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('seed', 'partner_profile', 'ego_profile', 'status', 'shared_return', 'target_delivery_step')}), flush=True)
    result = aggregate(rows, args.seeds)
    dump(args.out/'role_matrix.json', result)
    print(json.dumps(result['cells'], indent=2))
    if any(r['status'] != 'completed' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
