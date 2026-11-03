"""Read-only report for an RQ3 experiment folder: per-trial outcomes, per-condition
totals, paired effects, and every failed LLM response with its valid task ids.

    python -m research.coordination.diagnose outputs/rq3_<run>/experiment
"""
import argparse
from collections import defaultdict
import glob
import json
import os
from statistics import mean


def metric(row, key):
    return sum(row['metrics'].get(f'agent_{i}_{key}', 0) for i in (0, 1))


def trial_rows(run):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run, '*', 'summary.json')))]


def llm_errors(run):
    errors = []
    for path in sorted(glob.glob(os.path.join(run, '*', 'llm.jsonl'))):
        for line in open(path):
            record = json.loads(line)
            if 'error' not in record:
                continue
            context = json.loads(record['request']['messages'][1]['content'])
            try:
                content = record['response']['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                content = None
            errors.append({'trial': os.path.basename(os.path.dirname(path)), 'sim_step': record.get('sim_step'),
                           'attempt': record['attempt'], 'error': record['error'], 'model_output': content,
                           'holding': context['observation']['self']['holding'],
                           'valid_task_ids': [c['id'] for c in context['candidates']]})
    return errors


def paired(rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['seed'], r['partner_profile'])][r['condition']] = r
    pairs = []
    for (seed, profile), g in sorted(groups.items()):
        ok = {c: r['shared_return'] for c, r in g.items() if r['status'] == 'completed'}
        pairs.append({'seed': seed, 'profile': profile,
                      'correct_minus_none': ok['correct'] - ok['none'] if {'correct', 'none'} <= set(ok) else None,
                      'correct_minus_opposite': ok['correct'] - ok['opposite'] if {'correct', 'opposite'} <= set(ok) else None})
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', help='experiment folder containing seed*_*/summary.json')
    args = p.parse_args()
    rows = trial_rows(args.run)
    if not rows:
        raise SystemExit(f'No trials found under {args.run}')
    print(f"{'trial':28s} {'ret':>5s} {'ok':>3s} {'bad':>3s} {'unneed':>6s} {'disc':>4s} {'pred':>5s} {'dec':>3s}  first_choice  status")
    for r in rows:
        acc = r.get('partner_prediction_accuracy')
        print(f"s{r['seed']}_{r['partner_profile']}_{r['condition']:<16s} {r['shared_return']:5.0f} "
              f"{r['correct_deliveries']:3d} {r.get('wrong_deliveries', 0):3d} "
              f"{metric(r, 'unneeded_ingredient_added'):6d} {metric(r, 'dish_discarded'):4d} "
              f"{'-' if acc is None else f'{acc:.2f}':>5s} {r['decision_count']:3d}  {r['first_choice']}  {r['status']}")
    print('\nPer condition (completed trials only):')
    by = defaultdict(list)
    for r in rows:
        if r['status'] == 'completed':
            by[r['condition']].append(r)
    for condition, rs in sorted(by.items()):
        print(f"  {condition:9s} n={len(rs)} mean_return={mean(r['shared_return'] for r in rs):6.1f} "
              f"wrong_deliveries={sum(r.get('wrong_deliveries', 0) for r in rs)} "
              f"unneeded_ingredients={sum(metric(r, 'unneeded_ingredient_added') for r in rs)} "
              f"discards={sum(metric(r, 'dish_discarded') for r in rs)}")
    print('\nPaired effects (None = a needed condition failed or is missing):')
    pairs = paired(rows)
    for pair in pairs:
        print(f"  seed {pair['seed']} {pair['profile']:8s} correct-none={pair['correct_minus_none']} "
              f"correct-opposite={pair['correct_minus_opposite']}")
    for key in ('correct_minus_none', 'correct_minus_opposite'):
        values = [pair[key] for pair in pairs if pair[key] is not None]
        if values:
            print(f"  mean {key} = {mean(values):+.1f} over {len(values)} pairs")
    errors = llm_errors(args.run)
    print(f'\nLLM errors: {len(errors)}')
    for e in errors:
        print(f"  {e['trial']} step={e['sim_step']} attempt={e['attempt']} error={e['error']} holding={e['holding']}")
        print(f"    model output: {e['model_output']!r}")
        print(f"    valid ids:    {e['valid_task_ids']}")


if __name__ == '__main__':
    main()
