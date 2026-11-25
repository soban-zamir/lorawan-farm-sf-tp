# LoRaWAN SF and TP selection for mobile agricultural sensors

A LoRaWAN farm simulator and an online learning method that chooses the spreading factor (SF) and
transmit power (TP) of fixed and moving sensors, such as soil sensors, cow collars and tractor trackers.

The method runs on the network server, needs no training data and no change to the sensors. After every
received message it:

1. **learns** a shared link model (Bayesian logistic regression) and a coverage map of the farm from the
   messages of all sensors,
2. **predicts** where each moving sensor will be at its next message (Kalman filter),
3. **decides** the cheapest setting that still reaches the delivery target (Thompson sampling), and sends a
   new setting only when it is needed or worthwhile, within a limited reply budget.

## Repository layout

```
farmlora/                     the package
  config.py                   all settings (CFG) and scenarios
  radio.py                    LoRa time on air, sensitivity, energy, path loss
  environment.py              farm, sensor movement, terrain, fading
  learning.py                 Bayesian logistic regression, Kalman filter, Thompson sampling
  method.py                   the proposed network-server method
  baselines.py                standard ADR, Always SF12, G-ADR, EMA-ADR, GPS rule, prior only
  gateway.py                  EU868 duty cycle, receive windows, collisions
  simulate.py                 simulation loop, oracle, per-farm summary
  stats.py                    confidence intervals and paired t-tests
experiments/
  self_check.py               quick check of the installation (a few seconds)
  run_main.py                 main study
  run_references.py           reference methods, model mismatch, sensitivity
  run_gps_sweep.py            GPS-rule margin sweep
  run_variants.py             ADR variants, half-duplex scaling, body shadowing
  plot_learning_curve.py      delivery over time
results/                      reference output of the experiment scripts
```

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
python experiments/self_check.py
```

`self_check.py` should print `OK` for every method.

## Running the experiments

```bash
python experiments/run_main.py
python experiments/run_references.py
python experiments/run_gps_sweep.py
python experiments/run_variants.py
python experiments/plot_learning_curve.py
```

Run them in this order: each script reuses the results of the earlier ones. Every script writes one JSON
file per simulated farm to `runs/` and a summary text file at the end. A finished farm is never simulated
again, so an interrupted script can simply be started again. Compare your summaries with the files in
`results/`.

Options for all scripts:

* `--jobs N`: number of farms simulated in parallel (default: up to 8 cores)
* `--mode quick`: 2 farms per scenario instead of 10, for a first look
* `--days D`: simulate only `D` days (smoke test; results go to `runs_test_<D>d/`)

On an 8-core desktop, each experiment script takes about 30 minutes.

## Using the package

```python
from farmlora import CFG, run_farm

res = run_farm({**CFG, 'sim_days': 1.0}, seed=1000, methods=['adr', 'ours', 'oracle'])
for m, s in res.items():
    print(m, f"delivery {100*s['pdr']:.2f} %, {s['edel']:.1f} mJ per delivered message")
```

Change a value in a copy of `CFG` to simulate your own farm.

## Reproducibility

* Evaluation farms use seeds 1000–1009 (1000–1004 for 1000 sensors).
* Every method sees exactly the same farm, movement and radio conditions, so methods are compared in pairs.
* Each method has its own random stream, so adding a method never changes the results of the others.

## Author

Muhammad Soban Zamir

## License

MIT (see `LICENSE`).
