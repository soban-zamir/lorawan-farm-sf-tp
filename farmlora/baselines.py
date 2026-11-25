"""Baselines and reference methods.

- ADRServer:       standard ADR (Semtech recommended algorithm): best SNR of the last 20 uplinks, 10 dB margin
- SF12Server:      Always-SF12, the safest setting for every message, no replies
- GADRServer:      G-ADR (Farhad et al., Sensors 2020): mean of the last 20 SNRs within one standard deviation
- EMAADRServer:    EMA-ADR (Farhad et al., Sensors 2020): exponential moving average of the SNR (beta = 0.7)
- GPSRuleServer:   textbook link budget at the reported GPS position with a fixed margin, no learning
- PriorOnlyServer: the proposed method without any learning (prior model, no map, fixed target, no sampling)
The oracle is in simulate.simulate_oracle.
"""
import numpy as np

from .config import CFG
from .method import OurServer
from .learning import pick_action
from .radio import ACT_E, ACT_INDEX, ACT_SF, ACT_TP, SENS_ARR, SNR_REQ_ARR

# the GPS rule always uses the textbook link budget, whatever the scenario's true channel is
TEXTBOOK = dict(pl_d0_db=CFG['pl_d0_db'], path_loss_exp=CFG['path_loss_exp'], d0_m=CFG['d0_m'])


class ADRServer:
    """Semtech ADR: max SNR of the last 20 uplinks, 10 dB margin, 3 dB per step."""
    uses_adr = True

    def __init__(self, env, cfg, rng):
        self.cfg = cfg; N = env['N']; self.H = cfg['adr_history']
        self.hist = np.full((N, self.H), -np.inf); self.n = np.zeros(N, int)

    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        c, H = self.cfg, self.H
        tpmin, tpmax = min(c['tp_list']), max(c['tp_list'])
        self.hist[D, self.n[D] % H] = prx[D] - c['noise_floor_dbm']
        self.n[D] += 1
        rdy = D[self.n[D] >= H]
        if rdy.size == 0:
            return
        margin = self.hist[rdy].max(1) - SNR_REQ_ARR[sf[rdy] - 7] - c['adr_margin_db']
        nstep = np.floor(margin / 3).astype(int)
        s, p = sf[rdy].copy(), tp[rdy].copy()
        k1 = np.minimum(np.maximum(nstep, 0), s - 7); s -= k1; nstep -= k1
        k2 = np.minimum(np.maximum(nstep, 0), (p - tpmin) // 2); p -= 2 * k2; nstep -= k2
        k3 = np.minimum(np.maximum(-nstep, 0), (tpmax - p) // 2); p += 2 * k3
        changed = rdy[(s != sf[rdy]) | (p != tp[rdy])]
        sf[rdy], tp[rdy] = s, p
        self.n[changed] = 0; self.hist[changed] = -np.inf


class SF12Server:
    """Always the safest setting: SF12 at 14 dBm, ADR switched off (never needs a reply)."""
    uses_adr = False

    def __init__(self, env, cfg, rng):
        pass

    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        pass


def adr_apply(srv, rdy, est, sf, tp):
    """Standard ADR margin rule (identical to ADRServer) applied to an SNR estimate `est`. Returns changed sensors."""
    c = srv.cfg
    tpmin, tpmax = min(c['tp_list']), max(c['tp_list'])
    margin = est - SNR_REQ_ARR[sf[rdy] - 7] - c['adr_margin_db']
    nstep = np.floor(margin / 3).astype(int)
    s, p = sf[rdy].copy(), tp[rdy].copy()
    k1 = np.minimum(np.maximum(nstep, 0), s - 7); s -= k1; nstep -= k1
    k2 = np.minimum(np.maximum(nstep, 0), (p - tpmin) // 2); p -= 2 * k2; nstep -= k2
    k3 = np.minimum(np.maximum(-nstep, 0), (tpmax - p) // 2); p += 2 * k3
    changed = rdy[(s != sf[rdy]) | (p != tp[rdy])]
    sf[rdy], tp[rdy] = s, p
    return changed


class GADRServer(ADRServer):
    """G-ADR: average of the last 20 SNR values that lie within one standard deviation of their mean."""
    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        c, H = self.cfg, self.H
        self.hist[D, self.n[D] % H] = prx[D] - c['noise_floor_dbm']
        self.n[D] += 1
        rdy = D[self.n[D] >= H]
        if rdy.size == 0:
            return
        v = self.hist[rdy]
        mu = v.mean(1, keepdims=True)
        sd = v.std(1, ddof=1, keepdims=True)
        inside = np.abs(v - mu) <= sd + 1e-9
        est = (v * inside).sum(1) / np.maximum(inside.sum(1), 1)
        changed = adr_apply(self, rdy, est, sf, tp)
        self.n[changed] = 0; self.hist[changed] = -np.inf


class EMAADRServer(ADRServer):
    """EMA-ADR: SNR smoothed by an exponential moving average (beta = 0.7); decides from the 2nd SNR after a change."""
    def __init__(self, env, cfg, rng):
        super().__init__(env, cfg, rng)
        N = env['N']
        self.S = np.zeros(N); self.cnt = np.zeros(N, int)
        self.beta = cfg.get('ema_beta', 0.7); self.nmin = cfg.get('ema_min_obs', 2)

    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        y = prx[D] - self.cfg['noise_floor_dbm']
        self.S[D] = np.where(self.cnt[D] == 0, y, self.beta * y + (1 - self.beta) * self.S[D])
        self.cnt[D] += 1
        rdy = D[self.cnt[D] >= self.nmin]
        if rdy.size == 0:
            return
        changed = adr_apply(self, rdy, self.S[rdy], sf, tp)
        self.cnt[changed] = 0


class GPSRuleServer(OurServer):
    """No learning: textbook link budget (path loss at 1 km = 130 dB, exponent 2.7) at the reported GPS position.
    Cheapest setting with at least `gps_rule_margin_db` margin; same reply handling as our method."""
    def __init__(self, env, cfg, rng):
        super().__init__(env, cfg, rng, pooled=False, lookahead=False)
        self.priority = np.zeros(self.N)

    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        c, gw = self.cfg, self.env['gw']
        d = np.maximum(np.linalg.norm(self.env['gps'][D, k] - gw, axis=1), 10.0)
        pl = TEXTBOOK['pl_d0_db'] + 10 * TEXTBOOK['path_loss_exp'] * np.log10(d / TEXTBOOK['d0_m'])
        margin = ACT_TP[None, :] - pl[:, None] - SENS_ARR[ACT_SF - 7][None, :]
        ok = margin >= c['gps_rule_margin_db']
        a_new = np.where(ok.any(1), ok.argmax(1), len(ACT_SF) - 1)       # actions are sorted cheapest first
        n = np.arange(len(D)); cur = ACT_INDEX[sf[D], tp[D]]
        unsafe = margin[n, cur] < c['gps_rule_margin_db']
        good_down = ACT_E[a_new] <= (1 - c['switch_saving']) * ACT_E[cur]
        a = np.where(unsafe, a_new, np.where(good_down, a_new, cur))
        sf[D], tp[D] = ACT_SF[a], ACT_TP[a]
        self.priority = np.zeros(self.N)
        self.priority[D] = np.where(unsafe, 1 + (c['gps_rule_margin_db'] - margin[n, cur]) / 10,
                                    np.where(good_down, 0.5 * (1 - ACT_E[a_new] / ACT_E[cur]), 0.0))

    def final_state(self):
        return {}


class PriorOnlyServer(OurServer):
    """The proposed method without link learning: prior model, no coverage map, fixed 95 % target, no sampling.
    Motion prediction and the reply rules stay the same."""
    def __init__(self, env, cfg, rng):
        super().__init__(env, cfg, rng, pooled=True, lookahead=True)

    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        c = self.cfg
        q_dec = self.collision_factor(sf_h[:, k].astype(int)) * self.busy_factor()
        self.observations(k, D, sf, tp, sf_h, tp_h)       # bookkeeping only: nothing is learned
        g_now = self.env['gps'][D, k]
        self.update_motion(D, g_now, self.env['t'][D, k])
        cand = self.candidates(D, g_now)
        P_m = self.delivery_prob(self.mu[np.zeros(len(D), int)], cand, q_dec, False)
        a_new = pick_action(P_m, self.tau[D])
        n = np.arange(len(D)); cur = ACT_INDEX[sf[D], tp[D]]
        thr = np.minimum(self.tau[D], 0.98 * P_m.max(1))
        lt = np.log(thr / (1 - thr)) + c['hysteresis_logit']
        thr_hi = np.minimum(1 / (1 + np.exp(-lt)), 0.99 * P_m.max(1))
        safe_cur = P_m[n, cur] >= thr
        good_down = (ACT_E[a_new] <= (1 - c['switch_saving']) * ACT_E[cur]) & (P_m[n, a_new] >= thr_hi)
        a = np.where(~safe_cur, pick_action(P_m, thr_hi), np.where(good_down, a_new, cur))
        sf[D], tp[D] = ACT_SF[a], ACT_TP[a]
        self.priority = np.zeros(self.N)
        self.priority[D] = np.where(~safe_cur, 1 + (thr - P_m[n, cur]),
                                    np.where(good_down, 0.5 * (1 - ACT_E[a_new] / ACT_E[cur]), 0.0))

    def final_state(self):
        return {}
