"""Read-only report for an RQ3 experiment folder: per-trial outcomes, per-condition
totals, paired effects, and every failed LLM response with its valid task ids.

    python -m research.coordination.diagnose outputs/rq3_<run>/experiment
"""
import argparse
from collections import defaultdict
import glob
import json
import os
from statistics import mean, stdev


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


def described_type(profile, condition):
    """Partner type the planner was told about: the true one, the other one, or none."""
    if condition == 'none':
        return 'none'
    return profile if condition == 'correct' else ('delivery' if profile == 'prep' else 'prep')


def decision_stats(run):
    """Per (true partner, condition): what the ego chose across ALL decisions.

    duplicate = ego chose the same work kind (prep/delivery) the partner was doing
    at that moment. Compares told-about type with ego choices to tell
    complementing (opposite kind) from mirroring (same kind as described)."""
    groups = defaultdict(lambda: defaultdict(int))
    for path in sorted(glob.glob(os.path.join(run, '*', 'decisions.jsonl'))):
        summary = json.load(open(os.path.join(os.path.dirname(path), 'summary.json')))
        key = (summary['partner_profile'], summary['condition'])
        for line in open(path):
            d = json.loads(line)
            g = groups[key]
            g['decisions'] += 1
            g[f"chose_{d['chosen_kind']}"] += 1
            if d['chosen_kind'] in ('prep', 'delivery') and d['partner_current_kind'] in ('prep', 'delivery'):
                g['work_vs_work'] += 1
                g['duplicate'] += d['chosen_kind'] == d['partner_current_kind']
            if d.get('both_work_types_available'):
                g['opportunities'] += 1
                g['opportunity_chose_delivery'] += d['chosen_kind'] == 'delivery'
    return groups


def classify_decision(d):
    """Was useful, uncovered prep work on the menu, and what did the ego pick?

    useful prep: the target pot is empty/filling and adding the ingredient keeps
    it able to match the recipe (single-recipe layouts: any empty/filling pot).
    uncovered: the visible partner is not holding that ingredient while the pot
    needs exactly one more of it (a heuristic: the partner may target another pot).
    Returns (useful_uncovered_available, chosen_class)."""
    obs = d['context']['observation']
    recipe = (obs.get('recipe') or obs.get('public_recipe') or {}).get('ingredient_counts')
    pots = {tuple(st['pos']): st['contents'] for st in obs['stations']
            if st['kind'] == 'pot' and st['contents'] is not None}
    partner_holds = {a['holding'] for a in obs.get('visible_partner', [])}

    def prep_class(c):
        pot = pots.get(tuple(c['target']))
        if pot is None or pot['status'] not in ('empty', 'filling'):
            return 'unneeded_prep'
        if 'ingredient' not in c:
            return 'useful_prep'
        if recipe is None:
            return 'unknown_prep'
        name = f"ingredient_{c['ingredient']}"
        need = recipe.get(name, 0) - pot['ingredient_counts'].get(name, 0)
        if need <= 0:
            return 'unneeded_prep'
        return 'covered_prep' if need == 1 and name in partner_holds else 'useful_prep'

    available = any(c['kind'] == 'prep' and prep_class(c) == 'useful_prep' for c in d['context']['candidates'])
    chosen = next(c for c in d['context']['candidates'] if c['id'] == d['choice']['task_id'])
    if chosen['kind'] == 'prep':
        cls = prep_class(chosen)
        cls = 'useful_prep' if cls == 'covered_prep' else cls
    elif chosen['kind'] == 'delivery' and chosen['target'] is not None:
        pot = pots.get(tuple(chosen['target']))
        cls = 'serve_ready' if pot is not None and pot['status'] == 'ready' else 'serve_early'
    else:
        cls = chosen['kind']  # delivery of a held dish, recovery, inspect, wait
    return available, cls


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
        for profile in (None, 'prep', 'delivery'):
            values = [pair[key] for pair in pairs if pair[key] is not None and profile in (None, pair['profile'])]
            if not values:
                continue
            se = stdev(values) / len(values) ** 0.5 if len(values) > 1 else float('nan')
            print(f"  mean {key} = {mean(values):+.1f} (SE {se:.1f}) over {len(values)} pairs"
                  f"{'' if profile is None else f' [{profile} partner]'}")
    print('\nEgo decisions by true partner and condition (all decisions):')
    print(f"  {'partner':9s} {'condition':9s} {'told':9s} {'dec':>4s} {'prep%':>6s} {'deliv%':>6s} "
          f"{'other%':>6s} {'dup%':>5s} {'deliv%|both':>11s} {'pred':>5s} {'return':>6s}")
    stats = decision_stats(args.run)
    for (profile, condition), g in sorted(stats.items()):
        n = g['decisions'] or 1
        rs = [r for r in rows if r['partner_profile'] == profile and r['condition'] == condition]
        accs = [r['partner_prediction_accuracy'] for r in rs if r.get('partner_prediction_accuracy') is not None]
        other = n - g['chose_prep'] - g['chose_delivery']
        dup = f"{100*g['duplicate']/g['work_vs_work']:.0f}" if g['work_vs_work'] else '-'
        both = f"{100*g['opportunity_chose_delivery']/g['opportunities']:.0f}" if g['opportunities'] else '-'
        print(f"  {profile:9s} {condition:9s} {described_type(profile, condition):9s} {g['decisions']:4d} "
              f"{100*g['chose_prep']/n:6.0f} {100*g['chose_delivery']/n:6.0f} {100*other/n:6.0f} {dup:>5s} "
              f"{both:>11s} {mean(accs) if accs else float('nan'):5.2f} {mean(r['shared_return'] for r in rs):6.1f}")
    print('  dup% = ego chose the work kind the partner was doing; deliv%|both = delivery share when both kinds were offered.')
    print('  Complementing: deliv%|both rises when told the partner prioritizes prep. Mirroring: it rises when told delivery.')
    print('\nWhere the time goes, per trial means (ego = agent 0, partner = agent 1):')
    print(f"  {'partner':9s} {'condition':9s} {'wait%':>5s} {'recov%':>6s} {'insp%':>5s} {'ego_idle':>8s} "
          f"{'ego_ing':>7s} {'ego_unneed':>10s} {'ego_dlv':>7s} {'ptn_ing':>7s} {'ptn_unneed':>10s} {'ptn_dlv':>7s}")
    for (profile, condition), g in sorted(stats.items()):
        n = g['decisions'] or 1
        rs = [r for r in rows if r['partner_profile'] == profile and r['condition'] == condition]
        per = lambda agent, key: mean(r['metrics'].get(f'agent_{agent}_{key}', 0) for r in rs)
        print(f"  {profile:9s} {condition:9s} {100*g['chose_wait']/n:5.0f} {100*g['chose_recovery']/n:6.0f} "
              f"{100*g['chose_inspect']/n:5.0f} {per(0, 'stay_steps'):8.1f} {per(0, 'ingredient_added'):7.1f} "
              f"{per(0, 'unneeded_ingredient_added'):10.2f} {per(0, 'correct_delivery'):7.1f} "
              f"{per(1, 'ingredient_added'):7.1f} {per(1, 'unneeded_ingredient_added'):10.2f} {per(1, 'correct_delivery'):7.1f}")
    print('  ego_idle = steps the ego issued stay (waiting at a station or in a wait job); recov = discard/put_down.')
    print('\nChoices when a useful, uncovered prep job was on the menu (% of those decisions):')
    print(f"  {'partner':9s} {'condition':9s} {'avail%':>6s} {'n':>4s} {'use_prep':>8s} {'bad_prep':>8s} "
          f"{'serve_early':>11s} {'serve_rdy':>9s} {'wait':>5s} {'other':>5s}")
    groups = defaultdict(lambda: defaultdict(int))
    for path in sorted(glob.glob(os.path.join(args.run, '*', 'decisions.jsonl'))):
        summary = json.load(open(os.path.join(os.path.dirname(path), 'summary.json')))
        g = groups[(summary['partner_profile'], summary['condition'])]
        for line in open(path):
            available, cls = classify_decision(json.loads(line))
            g['decisions'] += 1
            if available:
                g['available'] += 1
                g[cls] += 1
    for key, g in sorted(groups.items()):
        n = g['available']
        pct = lambda k: f"{100*g[k]/n:.0f}" if n else '-'
        other = n - sum(g[k] for k in ('useful_prep', 'unneeded_prep', 'serve_early', 'serve_ready', 'wait'))
        print(f"  {key[0]:9s} {key[1]:9s} {100*n/(g['decisions'] or 1):6.0f} {n:4d} {pct('useful_prep'):>8s} "
              f"{pct('unneeded_prep'):>8s} {pct('serve_early'):>11s} {pct('serve_ready'):>9s} {pct('wait'):>5s} "
              f"{(f'{100*other/n:.0f}' if n else '-'):>5s}")
    print('  serve_early = plate a soup that is still cooking (the agent waits at the pot).')
    print(f"\nCorrect soups discarded: {sum(metric(r, 'correct_soup_discarded') for r in rows)}; "
          f"LLM feedback retries: {sum(r['metrics'].get('llm_feedback_retries', 0) for r in rows)}, "
          f"recovered: {sum(r['metrics'].get('llm_recovered_by_feedback', 0) for r in rows)}")
    errors = llm_errors(args.run)
    print(f'\nLLM errors: {len(errors)}')
    for e in errors:
        print(f"  {e['trial']} step={e['sim_step']} attempt={e['attempt']} error={e['error']} holding={e['holding']}")
        print(f"    model output: {e['model_output']!r}")
        print(f"    valid ids:    {e['valid_task_ids']}")


if __name__ == '__main__':
    main()
