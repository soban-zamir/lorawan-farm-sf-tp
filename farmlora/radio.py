"""Radio physics and battery cost: sensitivities, time on air, energy table, path loss and the features of
the link model."""
import math

import numpy as np

from .config import CFG

SENS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}   # dBm, 125 kHz
SNR_REQ = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}       # dB (used by ADR)
TX_CURRENT_MA = {2: 22.0, 4: 24.0, 6: 26.0, 8: 29.0, 10: 32.0, 12: 36.0, 14: 44.0}  # whole node, SX1276-class radio

SENS_ARR = np.array([SENS[s] for s in range(7, 13)])
SNR_REQ_ARR = np.array([SNR_REQ[s] for s in range(7, 13)])


def toa_bytes(sf, nbytes, cfg=CFG, crc=1):
    """Semtech LoRa time-on-air formula (seconds) for a PHY payload of nbytes."""
    bw = cfg['bw_hz']
    de = 1 if (sf >= 11 and bw == 125e3) else 0   # low data-rate optimisation
    h, cr = 0, cfg['coding_rate']
    tsym = (2 ** sf) / bw
    num = 8 * nbytes - 4 * sf + 28 + 16 * crc - 20 * h
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (cfg['preamble_symbols'] + 4.25) * tsym + n_payload * tsym


def time_on_air(sf, cfg=CFG):
    return toa_bytes(sf, cfg['payload_bytes'] + cfg['mac_overhead_bytes'], cfg)


def build_tables(cfg=CFG):
    """Energy table (E = V * I(TP) * time on air) and the 42 actions sorted from cheapest to most expensive."""
    toa = np.array([time_on_air(s, cfg) for s in range(7, 13)])
    e_tab = np.zeros((6, 15))                       # e_tab[sf-7, tp] in mJ
    for i, s in enumerate(range(7, 13)):
        for tp in cfg['tp_list']:
            e_tab[i, tp] = cfg['voltage'] * TX_CURRENT_MA[tp] * toa[i]   # V * mA * s = mJ
    acts = [(s, tp, e_tab[s - 7, tp]) for s in cfg['sf_list'] for tp in cfg['tp_list']]
    acts.sort(key=lambda a: a[2])
    act_sf = np.array([a[0] for a in acts])
    act_tp = np.array([a[1] for a in acts])
    act_e = np.array([a[2] for a in acts])
    return toa, e_tab, act_sf, act_tp, act_e


TOA, E_TAB, ACT_SF, ACT_TP, ACT_E = build_tables(CFG)
ACT_INDEX = np.full((13, 15), -1)                 # (sf, tp) -> action index
ACT_INDEX[ACT_SF, ACT_TP] = np.arange(len(ACT_SF))
# reply air-times (downlinks have no payload CRC)
DL_CMD = np.array([toa_bytes(s, CFG['dl_cmd_bytes'], crc=0) for s in range(7, 13)])
DL_EMPTY = np.array([toa_bytes(s, CFG['dl_empty_bytes'], crc=0) for s in range(7, 13)])
DL_SETUP = np.array([toa_bytes(s, CFG['dl_setup_bytes'], crc=0) for s in range(7, 13)])
# action features used by the link model
ACT_TPF = ACT_TP / 10.0
ACT_SFF = (-SENS_ARR[ACT_SF - 7] - 130.0) / 10.0


def path_loss(d, cfg=CFG):
    """Mean log-distance path loss. With cfg['pl_model'] == 'two_slope' (model-mismatch scenario):
    exponent pl_n1 up to d0 (1 km) and pl_n2 beyond."""
    if cfg.get('pl_model') == 'two_slope':
        d = np.maximum(d, 10.0)
        x = np.log10(d / cfg['d0_m'])
        return cfg['pl_d0_db'] + 10 * np.where(x < 0, cfg['pl_n1'], cfg['pl_n2']) * x
    d = np.maximum(d, 10.0)
    return cfg['pl_d0_db'] + 10 * cfg['path_loss_exp'] * np.log10(d / cfg['d0_m'])


def features(tp, sf, d):
    """Feature vector of the link model: x = [1, TP/10, log10(d/1 km), (-S(SF) - 130)/10]."""
    d = np.maximum(d, 10.0)
    return np.stack([np.ones_like(d, dtype=float), tp / 10.0, np.log10(d / 1000.0),
                     (-SENS_ARR[sf - 7] - 130.0) / 10.0], axis=-1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
