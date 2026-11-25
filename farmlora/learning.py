"""Online learning parts of the proposed method: Kalman filter for the motion
model, Bayesian logistic regression (Laplace approximation) with guard rails, Thompson sampling of the
weights, and the cheapest-action rule."""
import numpy as np

from .radio import sigmoid


class KalmanCV:
    """Constant-velocity Kalman filter for every sensor at once. State = [x, y, vx, vy]."""
    def __init__(self, N, cfg):
        self.s = np.zeros((N, 4)); self.P = np.zeros((N, 4, 4))
        self.t = np.zeros(N); self.init = np.zeros(N, bool)
        self.q = cfg['kf_q']; self.r = max(cfg['gps_sigma_m'], 1.0) ** 2; self.v0 = cfg['kf_v0']

    def update(self, idx, z, t):
        new = ~self.init[idx]
        if new.any():
            i = idx[new]
            self.s[i] = np.c_[z[new], np.zeros((len(i), 2))]
            self.P[i] = np.diag([self.r, self.r, self.v0 ** 2, self.v0 ** 2])
            self.t[i] = t[new]; self.init[i] = True
        old = ~new
        if old.any():
            i, zz = idx[old], z[old]
            dt = np.maximum(t[old] - self.t[i], 1e-3)
            F = np.tile(np.eye(4), (len(i), 1, 1)); F[:, 0, 2] = dt; F[:, 1, 3] = dt
            Q = np.zeros((len(i), 4, 4))
            a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
            Q[:, 0, 0] = Q[:, 1, 1] = a; Q[:, 2, 2] = Q[:, 3, 3] = c
            Q[:, 0, 2] = Q[:, 2, 0] = Q[:, 1, 3] = Q[:, 3, 1] = b
            Q *= self.q
            s = np.einsum('nij,nj->ni', F, self.s[i])
            P = F @ self.P[i] @ F.transpose(0, 2, 1) + Q
            S = P[:, :2, :2] + self.r * np.eye(2)
            K = P[:, :, :2] @ np.linalg.inv(S)
            s = s + np.einsum('nij,nj->ni', K, zz - s[:, :2])
            P = P - K @ P[:, :2, :]
            self.s[i], self.P[i], self.t[i] = s, P, t[old]


def enforce_signs(w, cfg):
    """Physics: more TP / higher SF never hurt, more distance never helps."""
    w = w.copy(); m = cfg['min_weight']
    w[:, 1] = np.maximum(w[:, 1], m); w[:, 3] = np.maximum(w[:, 3], m); w[:, 2] = np.minimum(w[:, 2], -m)
    return w


def link_terms(z, y, q):
    """Bernoulli model P(delivered) = sigmoid(z) * q  (q = chance of no collision).
    Returns p, score (d logL / dz) and Fisher information per observation."""
    p = sigmoid(z)
    pi = np.clip(p * q, 1e-9, 1 - 1e-9)
    score = y * (1 - p) - (1 - y) * p * q * (1 - p) / (1 - pi)
    info = q * p * (1 - p) ** 2 / (1 - pi)
    return p, score, info


def update_shared(mu, prec, midx, X, y, cfg, offset=None, q=None):
    """Online Bayesian logistic regression (Laplace approximation, a few Newton steps)."""
    offset = np.zeros(len(y)) if offset is None else offset
    q = np.ones(len(y)) if q is None else q
    models, loc = np.unique(midx, return_inverse=True)
    mu0, P0 = mu[models], prec[models]
    w = mu0.copy()
    for it in range(cfg['newton_iters'] + 1):
        _, score, info = link_terms(np.einsum('nd,nd->n', X, w[loc]) + offset, y, q)
        H = P0.copy()
        np.add.at(H, loc, info[:, None, None] * X[:, :, None] * X[:, None, :])
        if it == cfg['newton_iters']:
            break
        g = np.einsum('mij,mj->mi', P0, w - mu0)
        np.add.at(g, loc, -X * score[:, None])
        w = w - np.linalg.solve(H, g[..., None])[..., 0]
    mu[models] = enforce_signs(w, cfg)
    prec[models] = H * cfg['forget_lambda']


def sample_weights(mu, prec, midx, rng, cfg):
    cov = np.linalg.inv(prec[midx])
    cov = 0.5 * (cov + cov.transpose(0, 2, 1)) + 1e-9 * np.eye(4)
    Lc = np.linalg.cholesky(cov)
    return enforce_signs(mu[midx] + np.einsum('nij,nj->ni', Lc, rng.standard_normal((len(midx), 4))), cfg)


def pick_action(P, thr):
    """P: (n, 42) delivery probabilities for actions sorted cheapest first.
    Cheapest action with P >= thr; if the target is unreachable, cheapest within 2 % of the best."""
    thr = np.minimum(thr[:, None], 0.98 * P.max(1, keepdims=True))
    return (P >= thr).argmax(1)
