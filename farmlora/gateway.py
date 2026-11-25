"""Gateway constraints: EU868 duty cycle of the replies and uplink collisions."""
import numpy as np

from .radio import TOA


class Gateway:
    """EU868 reply scheduling: token buckets for the duty cycle of RX1 sub-bands (g1, g) and RX2."""
    def __init__(self, cfg):
        self.cap = np.array([cfg['rx1_duty'], cfg['rx1_duty'], cfg['rx2_duty']]) * 3600.0
        self.rate = np.array([cfg['rx1_duty'], cfg['rx1_duty'], cfg['rx2_duty']])
        self.tokens = self.cap.copy(); self.t = 0.0
        self.rx2_sf = cfg['rx2_sf']

    def send(self, t, sf_up, channel, table):
        """Try to send one reply after an uplink at time t. Returns air-time used (0 if no budget)."""
        self.tokens = np.minimum(self.cap, self.tokens + self.rate * max(t - self.t, 0.0)); self.t = max(self.t, t)
        rx1 = (0 if channel < 3 else 1, table[sf_up - 7])
        rx2 = (2, table[self.rx2_sf - 7])
        for band, air in (sorted([rx1, rx2], key=lambda o: o[1])):   # use the shorter window first
            if self.tokens[band] >= air:
                self.tokens[band] -= air
                return air
        return 0.0


def collided(env, k, sf, prx, cfg):
    """Messages lost to a collision: same channel and SF, overlapping in time, and the other message
    is not at least `capture_db` weaker."""
    t0 = env['t'][:, k]; t1 = t0 + TOA[sf - 7]
    key = env['channel'][:, k] * 6 + (sf - 7)
    col = np.zeros(len(sf), bool)
    for g in np.unique(key):
        idx = np.flatnonzero(key == g)
        if len(idx) < 2:
            continue
        ov = (t0[idx][:, None] < t1[idx][None, :]) & (t0[idx][None, :] < t1[idx][:, None])
        np.fill_diagonal(ov, False)
        strong = prx[idx][None, :] > prx[idx][:, None] - cfg['capture_db']
        col[idx] = (ov & strong).any(1)
    return col
