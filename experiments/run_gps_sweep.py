"""GPS-rule margin sweep: the GPS rule with margins of 4-16 dB on the main scenario,
compared with our method on the same farms. Run run_main.py first.

    python experiments/run_gps_sweep.py
Output: runs/gps_sweep/... and runs/gps_sweep/GPS_SWEEP_SUMMARY.txt
"""
import numpy as np

from _common import Results, parse_args, run_tasks, save_report
from farmlora import EVAL_SEED_BASE, seed_count
from farmlora.stats import mci, paired

MARGINS = [4, 6, 8, 10, 11, 12, 13, 14, 16]       # dB; 10 dB is the default margin


def tasks(mode):
    return [(f'margin{float(mg):04.1f}', dict(gps_rule_margin_db=float(mg)), EVAL_SEED_BASE + i, ['gps'])
            for mg in MARGINS for i in range(seed_count('main_mix_200', mode))]


def report(mode, T):
    res = Results('gps_sweep', 'references', 'main')
    seeds = sorted({t[2] for t in T})
    ours_pdr = res.col('main_mix_200', 'ours', 'pdr', seeds, 100)
    ours_e = res.col('main_mix_200', 'ours', 'edel', seeds)
    L = ['=' * 28 + ' GPS-RULE MARGIN SWEEP ' + '=' * 28,
         f'scenario=main_mix_200  farms={len(seeds)}  mode={mode}',
         f'Our method on the same farms: PDR {mci(ours_pdr)} %  |  E {mci(ours_e)} mJ per delivered', '',
         'margin |       PDR %       |   mJ per delivered  | sensors at target % |  replies/h  | blocked % |'
         ' ours vs GPS rule: dPDR pp, dEnergy % (paired t-test)']
    pdr_mean = {}
    for mg in MARGINS:
        scn = f'margin{float(mg):04.1f}'
        P = res.col(scn, 'gps', 'pdr', seeds, 100); E = res.col(scn, 'gps', 'edel', seeds)
        pdr_mean[mg] = np.nanmean(P)
        dp, _, pp, _ = paired(ours_pdr, P)
        _, _, pe, rel = paired(ours_e, E)
        L.append(f'{mg:5.1f}  | {mci(P)} | {mci(E)} | {mci(res.col(scn, "gps", "sensors_at_target", seeds, 100))} | '
                 f'{mci(res.col(scn, "gps", "dl_per_sensor_h", seeds))} | {np.nanmean(res.col(scn, "gps", "blocked_pct", seeds)):6.2f}   | '
                 f'dPDR {dp:+5.2f} (p={pp:.1e}), dE {rel:+6.1f} % (p={pe:.1e})')
    first95 = next((mg for mg in MARGINS if pdr_mean[mg] >= 95.0), None)
    L.append('')
    L.append('No margin up to 16 dB reaches 95 % delivery on average.' if first95 is None else
             f'Smallest margin reaching the 95 % target: {first95} dB (PDR {pdr_mean[first95]:.2f} %)')
    save_report('gps_sweep', 'GPS_SWEEP_SUMMARY.txt', L)


if __name__ == '__main__':
    args = parse_args(__doc__)
    T = tasks(args.mode)
    run_tasks('gps_sweep', T, args.jobs)
    report(args.mode, T)
