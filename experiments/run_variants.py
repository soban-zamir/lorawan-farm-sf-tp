"""ADR variants and stress tests: G-ADR and EMA-ADR in every scenario, half-duplex loss halved and doubled,
collar body shadowing and per-sensor delivery percentiles per sensor type.
Run run_main.py and run_references.py first.

    python experiments/run_variants.py
Output: runs/variants/... and runs/variants/VARIANTS_SUMMARY.txt
"""
import numpy as np

from _common import Results, cmp_line, parse_args, run_tasks, save_report
from farmlora import (BODY_SCENARIOS, EVAL_SEED_BASE, HALF_DUPLEX_SCENARIOS, LABELS, MISMATCH_SCENARIOS,
                      SCENARIOS, TYPE_NAMES, seed_count)
from farmlora.stats import mci


def tasks(mode):
    n = 2 if mode == 'quick' else 10
    T = []
    for scn, ov in SCENARIOS.items():
        for i in range(seed_count(scn, mode)):
            ms = ['adr', 'ours', 'gadr', 'ema'] if scn == 'main_mix_200' else ['gadr', 'ema']
            T.append((scn, ov, EVAL_SEED_BASE + i, ms))
    for scn, ov in MISMATCH_SCENARIOS.items():
        for i in range(n):
            T.append((scn, ov, EVAL_SEED_BASE + i, ['gadr', 'ema']))
    for scn, ov in HALF_DUPLEX_SCENARIOS.items():
        for i in range(n):
            T.append((scn, ov, EVAL_SEED_BASE + i, ['adr', 'gps', 'prior', 'ours', 'gadr', 'ema']))
    for scn, ov in BODY_SCENARIOS.items():
        for i in range(n):
            T.append((scn, ov, EVAL_SEED_BASE + i,
                      ['adr', 'sf12', 'gps', 'prior', 'alone', 'now', 'ours', 'gadr', 'ema']))
    return T


def report(mode, T):
    res = Results('variants', 'references', 'main')
    seeds = {}
    for scn, ov, seed, ms in T:
        seeds.setdefault(scn, []).append(seed)

    def row(scn, m, sd):
        c = lambda k, s=1.0: res.col(scn, m, k, sd, s)  # noqa: E731
        return (f'{LABELS[m]:26s} PDR {mci(c("pdr", 100))} | E {mci(c("edel"))} | at target {mci(c("sensors_at_target", 100))} | '
                f'replies/h {mci(c("dl_per_sensor_h"))} | blocked {mci(c("blocked_pct"))}')

    L = ['=' * 30 + ' ADR VARIANTS AND STRESS TESTS ' + '=' * 30, f'mode={mode}  (earlier results used where a method was not re-run)']
    L.append('\n### A. ADR variants from the literature vs standard ADR and our method')
    for scn in list(SCENARIOS) + list(MISMATCH_SCENARIOS):
        sd = sorted(set(seeds[scn]))
        L.append(f'\n## {scn} (farms={len(sd)})')
        for m in ['adr', 'gadr', 'ema', 'ours']:
            L.append(row(scn, m, sd))
        for b in ['adr', 'gadr', 'ema']:
            L.append(cmp_line(res, scn, sd, b))
    sdm = sorted(set(seeds['main_mix_200']))
    L.append('\n## main_mix_200 by sensor type (PDR % / mJ per delivered)')
    for m in ['adr', 'gadr', 'ema', 'ours']:
        L.append(f'{LABELS[m]:26s} ' + '  '.join(
            f'{nm}: {np.nanmean(res.col("main_mix_200", m, f"pdr_{nm}", sdm, 100)):6.2f} / '
            f'{np.nanmean(res.col("main_mix_200", m, f"edel_{nm}", sdm)):6.2f}' for nm in TYPE_NAMES.values()))

    L.append('\n### B. Half-duplex loss scaled (true loss only; the server model is unchanged). Base = main scenario')
    for scn, lab in [('main_mix_200', 'x1 (base)'), ('hd_half', 'x0.5'), ('hd_double', 'x2')]:
        sd = sdm if scn == 'main_mix_200' else sorted(set(seeds[scn]))
        L.append(f'\n## half-duplex loss {lab} (farms={len(sd)})')
        for m in ['adr', 'gps', 'prior', 'gadr', 'ema', 'ours']:
            L.append(row(scn, m, sd))
        for b in ['adr', 'gps', 'prior', 'gadr', 'ema']:
            L.append(cmp_line(res, scn, sd, b))
    L.append('  (Always-SF12 sends no replies, so it is unaffected: see the main scenario)')

    L.append('\n### C. Collar body shadowing (cows: 0-20 dB extra loss depending on heading relative to the gateway)')
    for scn in BODY_SCENARIOS:
        sd = sorted(set(seeds[scn]))
        L.append(f'\n## {scn} (farms={len(sd)})')
        for m in ['adr', 'sf12', 'gps', 'prior', 'alone', 'now', 'gadr', 'ema', 'ours']:
            L.append(row(scn, m, sd) + f' | cows PDR {np.nanmean(res.col(scn, m, "pdr_cow", sd, 100)):6.2f}')
        for b in ['adr', 'sf12', 'gps', 'prior', 'alone', 'now', 'gadr', 'ema']:
            L.append(cmp_line(res, scn, sd, b))

    L.append('\n### D. Per-sensor delivery percentiles per sensor type (main scenario, mean over farms)')
    for m in ['adr', 'gadr', 'ema', 'ours']:
        for nm in TYPE_NAMES.values():
            a = '/'.join(f'{100*np.nanmean(res.col("main_mix_200", m, f"ps_all_{nm}_p{q}", sdm)):5.1f}' for q in (5, 10, 50))
            b = '/'.join(f'{100*np.nanmean(res.col("main_mix_200", m, f"ps_day_{nm}_p{q}", sdm)):5.1f}' for q in (5, 10, 50))
            L.append(f'{LABELS[m]:22s} {nm:8s} all days p5/p10/p50 = {a} | last day p5/p10/p50 = {b}')
    L.append('=' * 84)
    save_report('variants', 'VARIANTS_SUMMARY.txt', L)


if __name__ == '__main__':
    args = parse_args(__doc__)
    T = tasks(args.mode)
    run_tasks('variants', T, args.jobs)
    report(args.mode, T)
