"""Quick check (a few seconds) that your installation gives exactly the reference numbers.

It simulates a short run (6 hours) on development farm 123 with every method and compares delivery ratio and
energy per delivered message with recorded reference values.

    python experiments/self_check.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from farmlora import CFG, LABELS, METHODS, run_farm  # noqa: E402

# (delivery ratio, mJ per delivered message) on farm 123 after 0.25 days
REFERENCE = {
    'adr':    (0.6917222222, 103.24861457),
    'sf12':   (0.9884722222, 265.94042434),
    'alone':  (0.9381111111, 206.78683525),
    'now':    (0.9396666667, 144.67385007),
    'ours':   (0.9234722222, 149.71300948),
    'gps':    (0.9363611111, 214.48516717),
    'prior':  (0.9399166667, 199.76543429),
    'gadr':   (0.8643888889, 138.78764702),
    'ema':    (0.7378333333, 84.9579945),
    'oracle': (0.967, 56.0620763),
}

if __name__ == '__main__':
    t0 = time.time()
    res = run_farm({**CFG, 'sim_days': 0.25}, 123, METHODS + ['oracle'])
    ok = True
    for m, (pdr, edel) in REFERENCE.items():
        s = res[m]
        same = abs(s['pdr'] - pdr) < 1e-8 and abs(s['edel'] - edel) < 1e-5 * max(1.0, edel)
        ok &= same
        print(f'{LABELS[m]:26s} PDR {100*s["pdr"]:6.2f} %   mJ/delivered {s["edel"]:7.2f}   '
              f'{"OK" if same else "DIFFERENT (expected %.2f %% / %.2f mJ)" % (100*pdr, edel)}')
    print(f'\n{"All methods reproduce the reference values." if ok else "Some values differ: check the package versions in requirements.txt."}'
          f'  ({time.time()-t0:.0f} s)')
    sys.exit(0 if ok else 1)
