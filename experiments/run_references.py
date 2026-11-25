"""Reference methods and robustness: GPS rule, prior only and oracle in every scenario, the two
model-mismatch channels, the sensitivity of the method to its own settings, the receive energy of the
replies and the spread of per-sensor delivery.

Run run_main.py first: methods that are not re-run here are taken from the main study (same farms).

    python experiments/run_references.py
Output: runs/references/<scenario>/seed*.json and runs/references/REFERENCES_SUMMARY.txt
"""
import numpy as np

from _common import Results, cmp_line, parse_args, run_tasks, save_report
from farmlora import EVAL_SEED_BASE, LABELS, MISMATCH_SCENARIOS, SCENARIOS, SENSITIVITY_SCENARIOS, seed_count
from farmlora.stats import mci

ALL = ['adr', 'sf12', 'gps', 'prior', 'alone', 'now', 'ours', 'oracle']


def tasks(mode):
    n = 2 if mode == 'quick' else 10
    T = []
    for scn, ov in SCENARIOS.items():                     # new references on all evaluation farms
        for i in range(seed_count(scn, mode)):
            T.append((scn, ov, EVAL_SEED_BASE + i, ALL if scn == 'main_mix_200' else ['gps', 'prior', 'oracle']))
    for scn, ov in MISMATCH_SCENARIOS.items():
        for i in range(n):
            T.append((scn, ov, EVAL_SEED_BASE + i, ALL))
    for scn, ov in SENSITIVITY_SCENARIOS.items():
        for i in range(min(5, n)):
            T.append((scn, ov, EVAL_SEED_BASE + i, ['ours']))
    return T


def report(mode, T):
    res = Results('references', 'main')
    seeds = {}
    for scn, ov, seed, ms in T:
        seeds.setdefault(scn, []).append(seed)

    def row(scn, m, sd, last):
        c = lambda k, s=1.0: res.col(scn, m, k, sd, s)  # noqa: E731
        return (f'{LABELS[m]:26s} PDR {mci(c("pdr", 100))} | E {mci(c("edel"))} | '
                f'at target {mci(c("sensors_at_target", 100))} | ' + last(c))

    L = ['=' * 30 + ' REFERENCE METHODS AND ROBUSTNESS ' + '=' * 30, f'mode={mode}  (main-study results used where a method was not re-run)']
    L.append('\n### A. All original scenarios: references vs our method (PDR % | mJ per delivered | sensors at target %)')
    for scn in SCENARIOS:
        sd = sorted(set(seeds[scn]))
        L.append(f'\n## {scn} (farms={len(sd)})')
        for m in ['adr', 'sf12', 'gps', 'prior', 'ours', 'oracle']:
            L.append(row(scn, m, sd, lambda c: f'replies/h {mci(c("dl_per_sensor_h"))}'))
        for b in ['gps', 'prior', 'oracle']:
            L.append(cmp_line(res, scn, sd, b))

    L.append('\n### B. Model-mismatch scenarios (all methods)')
    for scn in MISMATCH_SCENARIOS:
        sd = sorted(set(seeds[scn]))
        L.append(f'\n## {scn} (farms={len(sd)})')
        for m in ALL:
            L.append(row(scn, m, sd, lambda c: f'blocked {mci(c("blocked_pct"))}'))
        for b in ['adr', 'sf12', 'gps', 'prior', 'alone', 'now', 'oracle']:
            L.append(cmp_line(res, scn, sd, b))

    L.append('\n### C. Sensitivity (our method, main scenario, same farms; base = frozen settings)')
    sd5 = sorted(set(seeds[list(SENSITIVITY_SCENARIOS)[0]]))

    def srow(lab, scn):
        c = lambda k, s=1.0: res.col(scn, 'ours', k, sd5, s)  # noqa: E731
        return (f'{lab:42s} PDR {mci(c("pdr", 100))} | E {mci(c("edel"))} | at target {mci(c("sensors_at_target", 100))} | '
                f'replies/h {mci(c("dl_per_sensor_h"))} | blocked {mci(c("blocked_pct"))}')
    L.append(srow('base (B=4, saving 20 %, hysteresis 0.7)', 'main_mix_200'))
    for scn, ov in SENSITIVITY_SCENARIOS.items():
        L.append(srow(f'{scn} {ov}', scn))

    L.append('\n### D. Main scenario: receive energy of replies and spread of per-sensor delivery (re-run)')
    sdm = sorted(set(seeds['main_mix_200']))
    for m in ALL:
        rx = res.col('main_mix_200', m, 'rx_mj_per_del', sdm); e = res.col('main_mix_200', m, 'edel', sdm)
        L.append(f'{LABELS[m]:26s} RX mJ/delivered {mci(rx, 3)} ({100*np.nanmean(rx/e):4.2f} % of TX) | '
                 f'per-sensor PDR (all days) p10/p25/p50 = ' +
                 '/'.join(f'{100*np.nanmean(res.col("main_mix_200", m, f"ps_all_p{q}", sdm)):5.1f}' for q in (10, 25, 50)) +
                 ' | last day p10/p25/p50 = ' +
                 '/'.join(f'{100*np.nanmean(res.col("main_mix_200", m, f"ps_day_p{q}", sdm)):5.1f}' for q in (10, 25, 50)))

    L.append('\n### E. Energy per delivered message over time (main scenario)')
    for m in ['adr', 'sf12', 'ours']:
        vals = []
        for a, b, lab in [(0, 1, '0-1 h'), (1, 6, '1-6 h'), (6, 24, '6-24 h'), (24, 72, 'days 2-3')]:
            num = den = 0.0
            for s in sdm:
                r = res.get('main_mix_200', s, m)
                hp, he = np.array(r['hourly_pdr'][a:b]), np.array(r['hourly_edel'][a:b])
                num += (hp * he).sum(); den += hp.sum()
            vals.append(f'{lab}: {num / max(den, 1e-9):6.1f} mJ')
        L.append(f'{LABELS[m]:26s} ' + '  '.join(vals))
    L.append('=' * 76)
    save_report('references', 'REFERENCES_SUMMARY.txt', L)


if __name__ == '__main__':
    args = parse_args(__doc__)
    T = tasks(args.mode)
    run_tasks('references', T, args.jobs)
    report(args.mode, T)
