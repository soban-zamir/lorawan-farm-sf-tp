"""farmlora: online learning of LoRaWAN spreading factor and transmit power for mobile agricultural sensors.

A LoRaWAN farm simulator and a network-server method that learns the best setting for every sensor.

Quick use:
    from farmlora import CFG, run_farm
    out = run_farm({**CFG, 'sim_days': 0.5}, seed=123, methods=['adr', 'ours'])
    print(out['ours']['pdr'], out['ours']['edel'])
"""
import os

# one maths thread per process: much faster for many tiny calculations, and lets farms run in parallel
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import warnings  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)

from .config import (CFG, TYPE_NAMES, EVAL_SEED_BASE, DEV_SEEDS, SCENARIOS, MISMATCH_SCENARIOS,  # noqa: E402
                     SENSITIVITY_SCENARIOS, HALF_DUPLEX_SCENARIOS, BODY_SCENARIOS, seed_count)
from .environment import make_env  # noqa: E402
from .simulate import (METHODS, LABELS, simulate, simulate_oracle, summarize, summarize_all,  # noqa: E402
                       run_farm)

__all__ = ['CFG', 'TYPE_NAMES', 'EVAL_SEED_BASE', 'DEV_SEEDS', 'SCENARIOS', 'MISMATCH_SCENARIOS',
           'SENSITIVITY_SCENARIOS', 'HALF_DUPLEX_SCENARIOS', 'BODY_SCENARIOS', 'seed_count', 'make_env',
           'METHODS', 'LABELS', 'simulate', 'simulate_oracle', 'summarize', 'summarize_all', 'run_farm']
