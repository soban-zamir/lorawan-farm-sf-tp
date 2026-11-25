"""Main study: standard ADR, Always SF12, learning alone, no look-ahead and the proposed method in every
scenario, plus delivery per sensor type and per period after start-up.

    python experiments/run_main.py            # full: 115 farm runs (10 farms per scenario, 5 with 1000 sensors)
    python experiments/run_main.py --mode quick

Output: runs/main/<scenario>/seed*.json and runs/main/RESULTS_SUMMARY.txt
"""
import numpy as np

import _common
from _common import Results, cmp_line, parse_args, run_tasks, save_report
from farmlora import CFG, EVAL_SEED_BASE, LABELS, SCENARIOS, TYPE_NAMES, seed_count

METHODS_MAIN = ['adr', 'sf12', 'alone', 'now', 'ours']
PERIODS = [(0, 1, '0-1 h'), (1, 6, '1-6 h'), (6, 24, '6-24 h'), (24, 72, 'days 2-3')]


def tasks(mode):
    return [(scn, ov, EVAL_SEED_BASE + i, METHODS_MAIN, ('ours',) if i == 0 else ())
            for scn, ov in SCENARIOS.items() for i in range(seed_count(scn, mode))]


def report(mode, T):
    res = Results('main')
    seeds = {}
    for scn, ov, seed, ms, _ in T:
        seeds.setdefault(scn, []).append(seed)
    cfg = {**CFG, **_common.EXTRA}
    L = ['=' * 30 + ' RESULTS SUMMARY (main study) ' + '=' * 30,
         f'mode={mode}  evaluation seeds from {EVAL_SEED_BASE}  sensors={cfg["n_sensors"]}  days={cfg["sim_days"]}  '
         f'interval={cfg["report_interval_s"]:.0f}s  target={cfg["target_pdr"]}  farm={cfg["farm_size_m"]/1000:.0f}km']
    for scn in SCENARIOS:
        sd = seeds[scn]
        L.append(f'\n## {scn}  (farms={len(sd)})')
        L.append(f'{"method":20s} {"PDR %":>15s} {"mJ/delivered":>16s} {"replies/sens/h":>15s} {"gwTX % allowed":>15s} '
                 f'{"blocked %":>13s} {"sensors@target %":>17s}')
        for m in METHODS_MAIN:
            c = lambda k, s=1.0: res.col(scn, m, k, sd, s)  # noqa: E731
            L.append(f'{LABELS[m]:20s} {mci2(c("pdr", 100))} {mci2(c("edel"), 8)} {mci2(c("dl_per_sensor_h"), 7)} '
                     f'{mci2(c("gw_tx_pct"), 7, 1)} {mci2(c("blocked_pct"), 5)} {mci2(c("sensors_at_target", 100), 9)}')
        for b in ['adr', 'sf12', 'alone', 'now']:
            L.append(cmp_line(res, scn, sd, b))
    main = seeds['main_mix_200']
    L.append('\n## main_mix_200 by sensor type (PDR % / mJ per delivered)')
    for m in METHODS_MAIN:
        L.append(f'{LABELS[m]:20s} ' + '  '.join(
            f'{n}: {np.nanmean(res.col("main_mix_200", m, f"pdr_{n}", main, 100)):6.2f} / '
            f'{np.nanmean(res.col("main_mix_200", m, f"edel_{n}", main)):6.2f}' for n in TYPE_NAMES.values()))
    L.append('\n## main_mix_200 delivery per period after start-up (%), mean over farms and hours')
    for m in ['adr', 'sf12', 'ours']:
        hp = np.array([res.get('main_mix_200', s, m)['hourly_pdr'] for s in main]) * 100
        L.append(f'{LABELS[m]:20s} ' + '  '.join(f'{lab}: {hp[:, a:b].mean():5.1f}' for a, b, lab in PERIODS))
    L.append('\n## hours until delivery first reaches 94 % (main_mix_200)')
    for m in METHODS_MAIN:
        L.append(f'{LABELS[m]:20s} {mci2(res.col("main_mix_200", m, "reach_h", main))}')
    L.append('\n## our method internals (mean over farms)')
    for scn in SCENARIOS:
        ss = [res.get(scn, s, 'ours') for s in seeds[scn]]
        L.append(f'{scn:17s} map corr {np.nanmean([x["map_corr"] for x in ss]):5.2f} | cells learned '
                 f'{100*np.mean([x["map_cells_learned"] for x in ss]):5.1f} % | stationary decisions '
                 f'{100*np.mean([x["stationary_share"] for x in ss]):5.1f} % | mean target '
                 f'{np.mean([x["tau_mean"] for x in ss]):.3f}')
    L.append('=' * 90)
    save_report('main', 'RESULTS_SUMMARY.txt', L)


def mci2(v, w=7, d=2):
    from farmlora.stats import ci95
    return f'{np.nanmean(v):{w}.{d}f} ± {ci95(v):5.2f}'


if __name__ == '__main__':
    args = parse_args(__doc__)
    T = tasks(args.mode)
    run_tasks('main', T, args.jobs)
    report(args.mode, T)
