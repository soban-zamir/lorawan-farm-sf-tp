"""Simulation loop, oracle and per-farm summary.

Random numbers: the farm uses default_rng([seed, 1]); each method uses default_rng([seed, 2, i]) with i its
position in METHODS. METHODS only ever grows at the end, so every method keeps its random numbers.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .baselines import ADRServer, EMAADRServer, GADRServer, GPSRuleServer, PriorOnlyServer, SF12Server
from .config import RX_CURRENT_MA, TYPE_NAMES
from .environment import make_env
from .gateway import Gateway, collided
from .method import OurServer
from .radio import ACT_SF, ACT_TP, DL_CMD, DL_EMPTY, DL_SETUP, E_TAB, SENS_ARR, TOA

METHODS = ['adr', 'sf12', 'alone', 'now', 'ours', 'gps', 'prior', 'gadr', 'ema']
LABELS = {'adr': 'Standard ADR', 'sf12': 'Always SF12/14dBm', 'alone': 'Ours, learns alone',
          'now': 'Ours, no look-ahead', 'ours': 'Our method', 'gps': 'GPS rule (no learning)',
          'prior': 'Prior only (no learning)', 'gadr': 'G-ADR', 'ema': 'EMA-ADR',
          'oracle': 'Oracle (true channel)'}
# methods with the same reply handling (one-off ADRParamSetupReq, reply budget, ADR_ACK 256/64)
OURS_LIKE = ('alone', 'now', 'ours', 'gps', 'prior')


def make_server(method, env, cfg, rng):
    if method == 'adr':
        return ADRServer(env, cfg, rng)
    if method == 'sf12':
        return SF12Server(env, cfg, rng)
    if method == 'alone':
        return OurServer(env, cfg, rng, pooled=False, lookahead=True)
    if method == 'now':
        return OurServer(env, cfg, rng, pooled=True, lookahead=False)
    if method == 'ours':
        return OurServer(env, cfg, rng, pooled=True, lookahead=True)
    if method == 'gps':
        return GPSRuleServer(env, cfg, rng)
    if method == 'prior':
        s = PriorOnlyServer(env, cfg, rng); s.pooled = False   # nothing is learned -> no map to save
        return s
    if method == 'gadr':
        return GADRServer(env, cfg, rng)
    if method == 'ema':
        return EMAADRServer(env, cfg, rng)
    raise ValueError(method)


def simulate(env, method, cfg, seed):
    """Simulate one method on one farm (the same loop for every method)."""
    rng = np.random.default_rng([seed, 2, METHODS.index(method)])
    N, R, T = env['N'], env['R'], cfg['report_interval_s']
    hd = cfg.get('hd_scale', 1.0)
    sf = np.full(N, cfg['init_sf']); tp = np.full(N, cfg['init_tp'])
    delivered = np.zeros((N, R), bool); energy = np.zeros((N, R), np.float32)
    sf_h = np.zeros((N, R), np.int8); tp_h = np.zeros((N, R), np.int8)
    server = make_server(method, env, cfg, rng)
    gwy = Gateway(cfg)
    uses_adr = server.uses_adr
    if method in OURS_LIKE:
        lim, dly = cfg['ours_ack_limit'], cfg['ours_ack_delay']
    else:
        lim, dly = cfg['adr_ack_limit'], cfg['adr_ack_delay']
    ack_cnt = np.zeros(N, int)
    setup_done = np.zeros(N, bool) if method in OURS_LIKE else np.ones(N, bool)
    tpmax = max(cfg['tp_list'])
    n_dl = 0; dl_air = 0.0; n_dl_needed = 0; n_dl_dropped = 0; n_blocked = 0
    prev_dl_count, prev_dl_air = 0, 0.0
    for k in range(R):
        tk = env['t'][:, k]
        prx = tp - env['base_loss'][:, k]
        ok = prx >= SENS_ARR[sf - 7]
        if cfg['collisions']:
            ok &= ~collided(env, k, sf, prx, cfg)
        if prev_dl_count and cfg.get('half_duplex', True):
            p_block = 1 - np.exp(-hd * (prev_dl_count * TOA[sf - 7] + prev_dl_air) / T)
            blocked = rng.random(N) < p_block
            n_blocked += int((ok & blocked).sum())
            ok &= ~blocked
        delivered[:, k] = ok
        energy[:, k] = E_TAB[sf - 7, tp]
        sf_h[:, k] = sf; tp_h[:, k] = tp
        if uses_adr:
            ack_cnt += 1
        D = np.flatnonzero(ok)
        new_sf, new_tp = sf.copy(), tp.copy()
        if isinstance(server, OurServer):              # the server's own loss model is NOT scaled
            p_now = 1 - np.exp(-(prev_dl_count * TOA + prev_dl_air) / T)
            server.p_busy = p_now if server.p_busy is None else 0.97 * server.p_busy + 0.03 * p_now
        if D.size:
            server.step(k, D, new_sf, new_tp, prx, sf_h, tp_h)
        k_dl_count, k_dl_air = 0, 0.0
        if uses_adr and D.size:
            change = (new_sf[D] != sf[D]) | (new_tp[D] != tp[D])
            ackreq = ack_cnt[D] >= lim
            setup = ~setup_done[D]
            if isinstance(server, OurServer):
                must = ackreq
                free = max(cfg['reply_budget'] - int(must.sum()), 0)
                prio = server.priority[D] + 0.75 * setup
                opt = np.flatnonzero((change | setup) & ~must)
                if len(opt) > free:
                    keep = opt[np.argsort(-prio[opt])[:free]]
                    drop = np.setdiff1d(opt, keep)
                    change[drop] = False; setup[drop] = False
                    new_sf[D[drop]], new_tp[D[drop]] = sf[D[drop]], tp[D[drop]]
            need = change | ackreq | setup
            for j in np.argsort(tk[D]):
                if not need[j]:
                    continue
                i = D[j]
                n_dl_needed += 1
                table = DL_SETUP if setup[j] else (DL_CMD if change[j] else DL_EMPTY)
                air = gwy.send(tk[i] + TOA[sf[i] - 7], sf[i], env['channel'][i, k], table)
                if air > 0:
                    n_dl += 1; dl_air += air; k_dl_count += 1; k_dl_air += air
                    ack_cnt[i] = 0; setup_done[i] = True
                    sf[i], tp[i] = new_sf[i], new_tp[i]
                else:
                    n_dl_dropped += 1
        if uses_adr:
            bo = (ack_cnt >= lim + dly) & ((ack_cnt - lim) % dly == 0)
            if bo.any():
                raise_tp = bo & (tp < tpmax)
                tp = np.where(raise_tp, tpmax, tp)
                sf = np.where(bo & ~raise_tp, np.minimum(sf + 1, 12), sf)
        prev_dl_count, prev_dl_air = k_dl_count, k_dl_air
    hours = R * T / 3600
    out = dict(delivered=delivered, energy=energy, sf=sf_h, tp=tp_h,
               mac=dict(dl_per_sensor_h=n_dl / (N * hours),
                        gw_tx_pct=100 * dl_air / (hours * 3600 * (2 * cfg['rx1_duty'] + cfg['rx2_duty'])),
                        blocked_pct=100 * n_blocked / (N * R),
                        dl_dropped_pct=100 * n_dl_dropped / max(n_dl_needed, 1),
                        dl_air_s=dl_air))
    st = server.final_state() if hasattr(server, 'final_state') else {}
    if st:
        out['state'] = st
    if getattr(server, 'pooled', False):
        out['map'] = server.b_mu; out['map_prec'] = server.b_prec; out['map_ng'] = server.ng
    return out


def simulate_oracle(env, cfg, seed):
    """Knows the true average link (path loss + terrain + any extra loss) at the actual transmit position.
    Every message gets the cheapest setting whose delivery chance under fading is >= the target. No replies."""
    N, R = env['N'], env['R']
    need = stats.norm.ppf(cfg['target_pdr']) * max(cfg['fading_sigma_db'], 1e-6)
    sens_act = SENS_ARR[ACT_SF - 7]
    delivered = np.zeros((N, R), bool); energy = np.zeros((N, R), np.float32)
    sf_h = np.zeros((N, R), np.int8); tp_h = np.zeros((N, R), np.int8)
    for k in range(R):
        m = ACT_TP[None, :] - env['mean_loss'][:, k, None] - sens_act[None, :]
        ok_a = m >= need
        a = np.where(ok_a.any(1), ok_a.argmax(1), len(ACT_SF) - 1)
        sf, tp = ACT_SF[a], ACT_TP[a]
        prx = tp - env['base_loss'][:, k]
        ok = prx >= SENS_ARR[sf - 7]
        if cfg['collisions']:
            ok &= ~collided(env, k, sf, prx, cfg)
        delivered[:, k] = ok; energy[:, k] = E_TAB[sf - 7, tp]; sf_h[:, k] = sf; tp_h[:, k] = tp
    return dict(delivered=delivered, energy=energy, sf=sf_h, tp=tp_h,
                mac=dict(dl_per_sensor_h=0.0, gw_tx_pct=0.0, blocked_pct=0.0, dl_dropped_pct=0.0, dl_air_s=0.0))


def summarize(env, res, cfg, keep_map=False):
    types, dlv, E = env['types'], res['delivered'], res['energy']
    N, R = dlv.shape
    rph = int(round(3600 / cfg['report_interval_s']))
    nh = R // rph

    def stats_for(mask):
        n_del = dlv[mask].sum()
        return float(n_del / max(dlv[mask].size, 1)), float(E[mask].sum() / max(n_del, 1))

    s = {}
    s['pdr'], s['edel'] = stats_for(np.ones(N, bool))
    for k, name in TYPE_NAMES.items():
        if (types == k).any():
            s[f'pdr_{name}'], s[f'edel_{name}'] = stats_for(types == k)
    hd = dlv[:, :nh * rph].reshape(N, nh, rph).sum((0, 2)).astype(float)
    he = E[:, :nh * rph].reshape(N, nh, rph).sum((0, 2)).astype(float)
    s['hourly_pdr'] = (hd / (N * rph)).tolist()
    s['hourly_edel'] = (he / np.maximum(hd, 1)).tolist()
    roll_d = pd.Series(hd).rolling(3, min_periods=1).sum().values
    roll_n = pd.Series(np.full(nh, N * rph)).rolling(3, min_periods=1).sum().values
    hp = roll_d / roll_n
    reach = np.flatnonzero(hp >= cfg['target_pdr'] - 0.01)
    s['reach_h'] = float(reach[0] + 1) if len(reach) else float('nan')   # hours to reach >= 94 % delivery
    s['sf_share'] = [float((res['sf'] == q).mean()) for q in range(7, 13)]
    per_sensor = dlv[:, -24 * rph:].mean(1)
    s['sensors_at_target'] = float((per_sensor >= cfg['target_pdr'] - 0.01).mean())
    s.update(res['mac'])
    if 'state' in res:
        s.update(res['state'])
    if 'map' in res:
        ng, cell = res['map_ng'], cfg['map_cell_m']
        cx = (np.arange(ng) + 0.5) * cell
        g = cfg['shadow_grid_m']; sm = env['shadow_map']
        ii = np.clip(np.round(cx / g).astype(int), 0, sm.shape[0] - 1)
        true = -sm[np.ix_(ii, ii)].ravel()
        learned = res['map']
        seen = res['map_prec'] > 1.0 / cfg['map_prior_std'] ** 2 * 3
        s['map_cells_learned'] = float(seen.mean())
        s['map_corr'] = float(np.corrcoef(true[seen], learned[seen])[0, 1]) if seen.sum() > 10 else float('nan')
        if keep_map:
            s['map'] = np.round(learned, 2).tolist(); s['map_seen'] = seen.tolist()
    return s


def summarize_all(env, res, cfg, keep_map=False):
    """summarize() plus receive energy of the replies and per-sensor delivery percentiles (all, per type)."""
    s = summarize(env, res, cfg, keep_map=keep_map)
    dlv = res['delivered']
    rph = int(round(3600 / cfg['report_interval_s']))
    ps_all = dlv.mean(1); ps_day = dlv[:, -24 * rph:].mean(1)
    for q in (10, 25, 50):
        s[f'ps_all_p{q}'] = float(np.percentile(ps_all, q))
        s[f'ps_day_p{q}'] = float(np.percentile(ps_day, q))
    s['rx_mj_per_del'] = float(cfg['voltage'] * RX_CURRENT_MA * res['mac']['dl_air_s'] / max(dlv.sum(), 1))
    for k, name in TYPE_NAMES.items():
        msk = env['types'] == k
        if msk.any():
            for q in (5, 10, 50):
                s[f'ps_all_{name}_p{q}'] = float(np.percentile(ps_all[msk], q))
                s[f'ps_day_{name}_p{q}'] = float(np.percentile(ps_day[msk], q))
    return s


def run_farm(cfg, seed, methods, keep_map_for=()):
    """Build one farm and run every method in `methods` on it ('oracle' included). Returns {method: summary}."""
    env = make_env(cfg, seed)
    out = {}
    for m in methods:
        res = simulate_oracle(env, cfg, seed) if m == 'oracle' else simulate(env, m, cfg, seed)
        out[m] = summarize_all(env, res, cfg, keep_map=(m in keep_map_for))
    return out
