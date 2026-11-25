"""The virtual farm: sensors, movement, GPS, terrain, fading and channel variants.

Everything here is independent of the method, so all methods see exactly the same farm, movements and
radio conditions for a given seed.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .radio import path_loss


def cow_paths(rng, t, L, cfg):
    n, R = t.shape
    p = np.zeros((n, R, 2))
    cur = rng.uniform(0.05 * L, 0.95 * L, (n, 2))
    heading = rng.uniform(0, 2 * np.pi, n)
    speed = rng.uniform(*cfg['cow_speed'], n)
    grazing = rng.random(n) < 0.5
    p[:, 0] = cur
    for k in range(1, R):
        dt = t[:, k] - t[:, k - 1]
        u = rng.random(n)
        grazing = np.where(grazing, u > cfg['cow_p_stop'], u < cfg['cow_p_start'])
        heading = heading + rng.normal(0, cfg['cow_turn_sd'], n)
        step = np.where(grazing, speed * dt, 0.0)
        cur = cur + step[:, None] * np.c_[np.cos(heading), np.sin(heading)]
        for dim in (0, 1):                       # bounce off the farm fence
            lo, hi = cur[:, dim] < 0, cur[:, dim] > L
            cur[lo, dim] = -cur[lo, dim]
            cur[hi, dim] = 2 * L - cur[hi, dim]
            out = lo | hi
            heading[out] = (np.pi - heading[out]) if dim == 0 else -heading[out]
        p[:, k] = cur
    return p


def tractor_path(rng, t_end, L, cfg):
    """Waypoints: drive to a field, plough it row by row, pause, drive to the next field."""
    pos = rng.uniform(0, L, 2)
    wp, tw = [pos.copy()], [0.0]
    state = {'t': 0.0, 'pos': pos}

    def go(to, v):
        dist = np.linalg.norm(to - state['pos'])
        state['t'] += max(dist / v, 1e-3)
        state['pos'] = to.copy()
        wp.append(to.copy()); tw.append(state['t'])

    while state['t'] < t_end:
        fw = rng.uniform(*cfg['field_len']); fd = rng.uniform(*cfg['field_depth'])
        along_x = rng.random() < 0.5
        ax_, ay_ = (fw, fd) if along_x else (fd, fw)
        x0, y0 = rng.uniform(0, L - ax_), rng.uniform(0, L - ay_)
        go(np.array([x0, y0]), cfg['tractor_transit_speed'])
        v = rng.uniform(*cfg['tractor_field_speed'])
        for r in range(max(2, int(fd // cfg['row_spacing']))):
            off = r * cfg['row_spacing']
            a, b = (0.0, fw) if r % 2 == 0 else (fw, 0.0)
            start = np.array([x0 + a, y0 + off]) if along_x else np.array([x0 + off, y0 + a])
            end = np.array([x0 + b, y0 + off]) if along_x else np.array([x0 + off, y0 + b])
            if r > 0:
                go(start, v)
            go(end, v)
        state['t'] += rng.uniform(*cfg['tractor_pause_s'])      # pause in place
        wp.append(state['pos'].copy()); tw.append(state['t'])
    return np.array(wp), np.array(tw)


def make_base_env(cfg, seed):
    """Sensors, movement, GPS, terrain map, fading and channels for one farm."""
    rng = np.random.default_rng([seed, 1])
    L, N, T = cfg['farm_size_m'], cfg['n_sensors'], cfg['report_interval_s']
    R = int(round(cfg['sim_days'] * 86400 / T))
    n_fixed = int(round(N * cfg['frac_fixed']))
    n_cow = min(int(round(N * cfg['frac_cow'])), N - n_fixed)
    types = np.array([0] * n_fixed + [1] * n_cow + [2] * (N - n_fixed - n_cow))
    gw = np.array([L / 2, L / 2])

    offs = rng.uniform(0, T, N)
    t = np.arange(R)[None, :] * T + offs[:, None] + rng.uniform(-cfg['jitter_s'], cfg['jitter_s'], (N, R))
    t = np.maximum(t, 0.0)

    pos = np.zeros((N, R, 2))
    fi = np.flatnonzero(types == 0)
    pos[fi] = rng.uniform(0, L, (len(fi), 1, 2))
    ci = np.flatnonzero(types == 1)
    if len(ci):
        pos[ci] = cow_paths(rng, t[ci], L, cfg)
    for i in np.flatnonzero(types == 2):
        wp, tw = tractor_path(rng, t[i, -1] + T, L, cfg)
        pos[i, :, 0] = np.interp(t[i], tw, wp[:, 0])
        pos[i, :, 1] = np.interp(t[i], tw, wp[:, 1])

    gps = pos + rng.normal(0, cfg['gps_sigma_m'], pos.shape)
    dist = np.linalg.norm(pos - gw, axis=-1)

    g = cfg['shadow_grid_m']
    ng = int(L / g) + 1
    smap = gaussian_filter(rng.standard_normal((ng, ng)), sigma=cfg['shadow_corr_m'] / g, mode='wrap')
    smap = smap / smap.std() * cfg['shadow_sigma_db']
    ix = np.clip(np.round(pos[..., 0] / g).astype(int), 0, ng - 1)
    iy = np.clip(np.round(pos[..., 1] / g).astype(int), 0, ng - 1)
    shadow = smap[ix, iy]
    fading = rng.normal(0, cfg['fading_sigma_db'], (N, R))

    base_loss = path_loss(dist, cfg) + shadow - fading - cfg['antenna_gain_db']   # Prx = TP - base_loss
    channel = rng.integers(0, cfg['n_channels'], (N, R))
    return dict(types=types, t=t, pos=pos, gps=gps, dist=dist, gw=gw, base_loss=base_loss,
                channel=channel, shadow_map=smap, R=R, N=N)


def make_env(cfg, seed):
    """One farm for one scenario. On top of make_base_env (no extra random numbers are drawn):
    - time-varying channel: cfg['tv_extra_db'] of extra loss during cfg['tv_hours'] every day;
    - collar body shadowing: up to cfg['body_loss_db'] of extra loss for cows, the most when a cow
      faces away from the gateway;
    - 'mean_loss': the true average link without fading (used by the oracle)."""
    env = make_base_env(cfg, seed)
    g = cfg['shadow_grid_m']; sm = env['shadow_map']; ng = sm.shape[0]
    ix = np.clip(np.round(env['pos'][..., 0] / g).astype(int), 0, ng - 1)
    iy = np.clip(np.round(env['pos'][..., 1] / g).astype(int), 0, ng - 1)
    extra = np.zeros_like(env['base_loss'])
    if cfg.get('tv_extra_db', 0.0):
        h0, h1 = cfg['tv_hours']
        hour = (env['t'] / 3600.0) % 24
        extra = np.where((hour >= h0) & (hour < h1), cfg['tv_extra_db'], 0.0)
        env['base_loss'] = env['base_loss'] + extra
    env['mean_loss'] = path_loss(env['dist'], cfg) + sm[ix, iy] + extra - cfg['antenna_gain_db']
    A = cfg.get('body_loss_db', 0.0)
    if A:
        ci = np.flatnonzero(env['types'] == 1)
        if len(ci):
            p = env['pos'][ci]                                             # (n, R, 2) true positions
            d = np.diff(p, axis=1)
            ang = np.where(np.linalg.norm(d, axis=-1) > 0.5, np.arctan2(d[..., 1], d[..., 0]), np.nan)
            ang = pd.DataFrame(ang.T).ffill().bfill().values.T             # a resting cow keeps its last heading
            ang = np.nan_to_num(np.concatenate([ang[:, :1], ang], axis=1))
            away = np.arctan2(p[..., 1] - env['gw'][1], p[..., 0] - env['gw'][0])
            body = A * (1 + np.cos(ang - away)) / 2                        # 0 dB facing the gateway, A facing away
            env['base_loss'][ci] += body
            env['mean_loss'][ci] += body
    return env
