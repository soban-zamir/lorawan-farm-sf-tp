"""Shared helpers for the experiment scripts: running farms in parallel, caching and loading results."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from farmlora import run_farm  # noqa: E402
from farmlora.stats import paired  # noqa: E402

RUNS = os.path.join(ROOT, 'runs')       # all new results go here (one JSON file per farm)
EXTRA = {}                              # set by --days (smoke tests only)


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--mode', choices=['full', 'quick'], default='full',
                    help="'full' reproduces the reference results (10 farms per scenario); 'quick' uses 2 farms for a first look")
    ap.add_argument('--jobs', type=int, default=max(1, min(8, os.cpu_count() or 2)),
                    help='number of farms simulated in parallel (default: up to 8 cores)')
    ap.add_argument('--days', type=float, default=None,
                    help='simulate fewer days (smoke test only; results go to runs_test_<days>d/, not runs/)')
    args = ap.parse_args()
    if args.days:
        global RUNS
        RUNS = os.path.join(ROOT, f'runs_test_{args.days:g}d')
        EXTRA['sim_days'] = args.days
    return args


def farm_path(part, scn, seed, runs=None):
    return os.path.join(runs or RUNS, part, scn, f'seed{seed:04d}.json')


def _run_one(part, scn, overrides, seed, methods, keep_map_for, runs, extra):
    # runs/extra are passed explicitly: parallel workers import this module fresh
    from farmlora import CFG
    path = farm_path(part, scn, seed, runs)
    if os.path.exists(path):
        return 0.0
    t0 = time.time()
    out = run_farm({**CFG, **overrides, **extra}, seed, methods, keep_map_for=keep_map_for)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    os.replace(tmp, path)                       # write-then-rename: never leaves a half-written file
    return time.time() - t0


def run_tasks(part, tasks, jobs, cost=None):
    """tasks: list of (scenario, overrides, seed, methods[, keep_map_for]). Finished farms are skipped,
    so an interrupted run can simply be started again."""
    tasks = [t if len(t) == 5 else (*t, ()) for t in tasks]
    todo = [t for t in tasks if not os.path.exists(farm_path(part, t[0], t[2]))]
    cost = cost or {'mix_1000': 5.0, 'mix_50': 0.3, 'collisions_on': 1.3}
    todo.sort(key=lambda t: -cost.get(t[0], 1.0) * len(t[3]))     # biggest jobs first: keeps all cores busy
    print(f'[{part}] {len(tasks)} farm runs in total, {len(tasks) - len(todo)} already done, '
          f'{len(todo)} to run on {jobs} cores')
    t0 = time.time()
    if todo:
        if jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=jobs, verbose=10)(delayed(_run_one)(part, *t, RUNS, EXTRA) for t in todo)
        else:
            for n, t in enumerate(todo, 1):
                dt = _run_one(part, *t, RUNS, EXTRA)
                print(f'{n:3d}/{len(todo)}  {t[0]:22s} seed {t[2]}  {dt:5.0f} s   total {(time.time()-t0)/60:5.1f} min')
    missing = sum(not os.path.exists(farm_path(part, t[0], t[2])) for t in tasks)
    print(f'[{part}] finished in {(time.time()-t0)/60:.1f} min. Missing: {missing}')


def load_part(part):
    """All results of one part: {(scenario, seed, method): summary}."""
    out = {}
    base = os.path.join(RUNS, part)
    if not os.path.isdir(base):
        return out
    for scn in sorted(os.listdir(base)):
        d = os.path.join(base, scn)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                with open(os.path.join(d, fn), encoding='utf-8') as f:
                    for m, s in json.load(f).items():
                        out[(scn, int(fn[4:8]), m)] = s
    return out


class Results:
    """Look up a farm result in several parts, the first part that has it wins
    (a method that was not re-run in a later part is taken from an earlier one: same farm, same method)."""
    def __init__(self, *parts):
        self.parts = [load_part(p) for p in parts]

    def get(self, scn, seed, m):
        for p in self.parts:
            if (scn, seed, m) in p:
                return p[(scn, seed, m)]
        return None

    def col(self, scn, m, key, seeds, scale=1.0):
        out = []
        for s in seeds:
            r = self.get(scn, s, m)
            out.append(r[key] * scale if (r is not None and key in r) else np.nan)
        return np.array(out, float)

    def pair(self, scn, seeds, key, a, b, scale=1.0):
        return paired(self.col(scn, a, key, seeds, scale), self.col(scn, b, key, seeds, scale))


def cmp_line(res, scn, seeds, b, a='ours'):
    dp, cp, pp, _ = res.pair(scn, seeds, 'pdr', a, b, 100)
    de, ce, pe, rel = res.pair(scn, seeds, 'edel', a, b)
    return (f'  {a} vs {b:6s}: dPDR {dp:+6.2f}±{cp:4.2f} pp (p={pp:.1e}) | '
            f'dEnergy {de:+7.2f}±{ce:5.2f} mJ ({rel:+6.1f} %, p={pe:.1e})')


def save_report(part, name, lines):
    text = '\n'.join(lines)
    print(text)
    os.makedirs(os.path.join(RUNS, part), exist_ok=True)
    path = os.path.join(RUNS, part, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\nSaved to', path)
