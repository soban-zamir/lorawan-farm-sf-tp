"""The proposed method: network-server selection of SF and TP after every received message."""
import numpy as np

from .learning import KalmanCV, link_terms, pick_action, sample_weights, update_shared
from .radio import ACT_E, ACT_INDEX, ACT_SF, ACT_SFF, ACT_TP, ACT_TPF, TOA, features, sigmoid


class OurServer:
    """The proposed method: learn (shared link model, coverage map, per-sensor target), predict (Kalman filter),
    decide (Thompson sampling, keep-or-switch rule, reply budget).

    pooled=False gives the "learning alone" variant (one model per sensor, no shared map);
    lookahead=False gives the "no look-ahead" variant (plans for the current position).
    """
    uses_adr = True

    def __init__(self, env, cfg, rng, pooled=True, lookahead=True):
        self.env, self.cfg, self.rng = env, cfg, rng
        N = env['N']; self.N = N
        self.pooled, self.lookahead = pooled, lookahead
        self.last_k = np.full(N, -1); self.last_gps = np.zeros((N, 2)); self.last_t = np.zeros(N)
        self.kf = KalmanCV(N, cfg)
        M = 1 if pooled else N
        self.mu = np.tile(np.array(cfg['prior_mean'], float), (M, 1))
        self.prec = np.tile(np.diag(1.0 / np.array(cfg['prior_std'], float) ** 2), (M, 1, 1))
        self.cell = cfg['map_cell_m']; self.ng = int(cfg['farm_size_m'] / self.cell) + 1
        self.b_mu = np.zeros(self.ng * self.ng)
        self.b_prec = np.full(self.ng * self.ng, 1.0 / cfg['map_prior_std'] ** 2)
        nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        self.nb = np.array(nb)
        self.nb_w = np.array([1.0 if (dx, dy) == (0, 0) else cfg['map_neighbor_w'] for dx, dy in nb])
        self.tau = np.full(N, cfg['target_pdr'])
        K = cfg['motion_K']; self.K = K
        self.alpha = np.full(N, cfg['motion_alpha0'])
        self.suy = np.zeros(N); self.suu = np.zeros(N)
        self.resid = np.zeros((N, K, 2)); self.rptr = np.zeros(N, int)
        self.n_stationary = 0; self.n_decisions = 0
        self.p_busy = None                               # set by the simulation loop from the gateway's reply schedule

    # ---------- helpers ----------
    def cells(self, pos):
        ij = np.clip(np.floor(pos / self.cell).astype(int), 0, self.ng - 1)
        return ij[..., 0] * self.ng + ij[..., 1]

    def collision_factor(self, sf_now):
        """Chance of no collision on each SF (pure ALOHA), from the server's own SF assignments."""
        c = self.cfg
        if not c['collisions']:
            return np.ones(6)
        counts = np.bincount(sf_now - 7, minlength=6)
        return np.exp(-2 * counts * TOA / (c['report_interval_s'] * c['n_channels']))

    def busy_factor(self):
        """Chance that the gateway is not busy sending replies, per SF (from its own reply schedule)."""
        return (1 - self.p_busy) if self.p_busy is not None else np.ones(6)

    # ---------- learn ----------
    def observations(self, k, D, sf, tp, sf_h, tp_h):
        """Delivered now (y=1) + every message lost since the last delivery (y=0, found from sequence gaps)."""
        t, gps = self.env['t'], self.env['gps']
        g_now = gps[D, k]
        o_i, o_sf, o_tp, o_p, o_y = [D], [sf[D]], [tp[D]], [g_now], [np.ones(len(D))]
        gaps = k - self.last_k[D] - 1
        for i in D[gaps > 0]:
            js = np.arange(self.last_k[i] + 1, k)
            if self.last_k[i] >= 0:     # position of a lost message: interpolate between GPS fixes
                frac = (t[i, js] - self.last_t[i]) / max(t[i, k] - self.last_t[i], 1e-6)
                pp = self.last_gps[i] + frac[:, None] * (gps[i, k] - self.last_gps[i])
            else:
                pp = np.repeat(gps[i, k][None], len(js), 0)
            o_i.append(np.full(len(js), i)); o_sf.append(sf_h[i, js].astype(int)); o_tp.append(tp_h[i, js].astype(int))
            o_p.append(pp); o_y.append(np.zeros(len(js)))
        self.last_k[D] = k; self.last_gps[D] = g_now; self.last_t[D] = t[D, k]
        return (np.concatenate(o_i), np.concatenate(o_sf).astype(int), np.concatenate(o_tp).astype(float),
                np.concatenate(o_p), np.concatenate(o_y))

    def update_map(self, X, pos, y, q, w):
        c = self.cfg
        z0 = X @ w
        ij = np.clip(np.floor(pos / self.cell).astype(int), 0, self.ng - 1)
        nij = np.clip(ij[:, None, :] + self.nb[None], 0, self.ng - 1)
        cells = (nij[..., 0] * self.ng + nij[..., 1]).ravel()
        wts = np.tile(self.nb_w, len(y)); oidx = np.repeat(np.arange(len(y)), 9)
        uc, inv = np.unique(cells, return_inverse=True)
        b0, P0 = self.b_mu[uc], self.b_prec[uc]
        b = b0.copy()
        for it in range(3):
            _, score, info = link_terms(z0[oidx] + b[inv], y[oidx], q[oidx])
            H = P0 + np.bincount(inv, wts * info, len(uc))
            if it == 2:
                break
            g = P0 * (b - b0) - np.bincount(inv, wts * score, len(uc))
            b = b - g / H
        self.b_mu[uc] = np.clip(b, -6, 6); self.b_prec[uc] = H * c['map_forget']

    # ---------- decide ----------
    def update_motion(self, D, g_now, t_now):
        """Learn how each sensor's movement continues: next displacement ~ alpha * (velocity * dt)."""
        c = self.cfg
        has = self.kf.init[D]
        if has.any():
            i = D[has]
            dt = t_now[has] - self.kf.t[i]
            u = self.kf.s[i, 2:] * dt[:, None]
            e = g_now[has] - self.kf.s[i, :2]
            r = e - self.alpha[i, None] * u
            self.resid[i, self.rptr[i] % self.K] = r; self.rptr[i] += 1
            f = c['motion_forget']; prior = 100.0
            self.suy[i] = f * self.suy[i] + (u * e).sum(1)
            self.suu[i] = f * self.suu[i] + (u * u).sum(1)
            self.alpha[i] = np.clip((self.suy[i] + c['motion_alpha0'] * prior) / (self.suu[i] + prior), -1.0, 1.2)
        self.kf.update(D, g_now, t_now)

    def candidates(self, D, g_now):
        """Likely positions for each of the next H messages: (n, H, K, 2)."""
        c = self.cfg; L = c['farm_size_m']; H = c['plan_horizon']
        if not self.lookahead:
            return g_now[:, None, None, :]
        pos, vel = self.kf.s[D, :2], self.kf.s[D, 2:]
        nres = np.minimum(self.rptr[D], self.K)
        rms = np.sqrt((self.resid[D] ** 2).sum(-1).sum(-1) / np.maximum(nres, 1))
        lim = 3 * c['gps_sigma_m'] + c['stationary_extra_m']
        move = np.linalg.norm(vel, axis=1) * c['report_interval_s']
        stationary = (move < lim) & (rms < lim) & (nres >= 3)
        self.n_stationary += int(stationary.sum()); self.n_decisions += len(D)
        h = np.arange(1, H + 1)[None, :, None, None]
        base = pos[:, None, None, :] + self.alpha[D, None, None, None] * vel[:, None, None, :] * c['report_interval_s'] * h
        cand = base + self.resid[D][:, None, :, :] * np.sqrt(h)    # errors grow with the horizon
        cand[stationary] = pos[stationary][:, None, None, :]       # stationary mode: own location, no look-ahead
        return np.clip(cand, 0, L)

    def delivery_prob(self, w, cand, q_sf, sample_map):
        """(n, 42): for each action, the worst over the next H messages of the average over likely positions."""
        gw = self.env['gw']
        logd = np.log10(np.maximum(np.linalg.norm(cand - gw, axis=-1), 10) / 1000)      # (n, H, K)
        if self.pooled:
            cc = self.cells(cand)
            b = self.b_mu[cc]
            if sample_map:
                b = b + self.rng.standard_normal(cc.shape) / np.sqrt(self.b_prec[cc])
        else:
            b = np.zeros(logd.shape)
        z = (w[:, None, None, [0]] + w[:, None, None, [1]] * ACT_TPF + w[:, None, None, [2]] * logd[..., None]
             + w[:, None, None, [3]] * ACT_SFF + b[..., None])                          # (n, H, K, 42)
        return (sigmoid(z) * q_sf[ACT_SF - 7]).mean(2).min(1)

    # ---------- one step ----------
    def step(self, k, D, sf, tp, prx, sf_h, tp_h):
        c, gw = self.cfg, self.env['gw']
        q_col = self.collision_factor(sf_h[:, k].astype(int))
        q_busy = self.busy_factor()
        oi, osf, otp, opos, oy = self.observations(k, D, sf, tp, sf_h, tp_h)
        X = features(otp, osf, np.linalg.norm(opos - gw, axis=1))
        oq = (q_col * q_busy)[osf - 7]                   # learning: losses to load are not blamed on the link
        midx = np.zeros(len(oi), int) if self.pooled else oi
        off = self.b_mu[self.cells(opos)] if self.pooled else None
        update_shared(self.mu, self.prec, midx, X, oy, c, offset=off, q=oq)
        if self.pooled:
            self.update_map(X, opos, oy, oq, self.mu[0])
        # controller: each sensor's real delivery (including losses to network load) should reach the target
        np.add.at(self.tau, oi, c['calib_eta'] * (c['target_pdr'] - oy))
        self.tau = np.clip(self.tau, c['calib_min'], c['calib_max'])
        g_now = self.env['gps'][D, k]
        self.update_motion(D, g_now, self.env['t'][D, k])
        cand = self.candidates(D, g_now)
        midx_d = np.zeros(len(D), int) if self.pooled else D
        q_dec = q_col * q_busy                            # decisions count every way a message can be lost
        P_s = self.delivery_prob(sample_weights(self.mu, self.prec, midx_d, self.rng, c), cand, q_dec, True)
        a_new = pick_action(P_s, self.tau[D])
        # change only when worth it (every change costs a gateway reply):
        #  - current setting no longer safe (mean model)      -> switch
        #  - new setting >= switch_saving cheaper AND safe     -> switch
        #  - otherwise                                         -> keep
        #  hysteresis: a new setting must be safe with a margin (thr_hi), the current one only above thr,
        #  so a sensor does not flip back and forth at a boundary
        P_m = self.delivery_prob(self.mu[midx_d], cand, q_dec, False)
        n = np.arange(len(D))
        cur = ACT_INDEX[sf[D], tp[D]]
        thr = np.minimum(self.tau[D], 0.98 * P_m.max(1))
        lt = np.log(thr / (1 - thr)) + c['hysteresis_logit']
        thr_hi = np.minimum(1 / (1 + np.exp(-lt)), 0.99 * P_m.max(1))
        safe_cur = P_m[n, cur] >= thr
        good_down = (ACT_E[a_new] <= (1 - c['switch_saving']) * ACT_E[cur]) & (P_m[n, a_new] >= thr_hi)
        a_up = pick_action(P_m, thr_hi)
        a = np.where(~safe_cur, a_up, np.where(good_down, a_new, cur))
        sf[D], tp[D] = ACT_SF[a], ACT_TP[a]
        # priority for the gateway's reply budget: safety first (the more unsafe, the higher), then the biggest savings
        self.priority = np.zeros(self.N)
        self.priority[D] = np.where(~safe_cur, 1 + (thr - P_m[n, cur]),
                                    np.where(good_down, 0.5 * (1 - ACT_E[a_new] / ACT_E[cur]), 0.0))

    def final_state(self):
        out = dict(stationary_share=self.n_stationary / max(self.n_decisions, 1),
                   tau_mean=float(self.tau.mean()), alpha_mean=float(self.alpha.mean()))
        if self.pooled:
            out['mu'] = self.mu[0].tolist()
        return out
