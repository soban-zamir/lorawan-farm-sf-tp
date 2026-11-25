"""All simulation and method parameters in one place.

Every experiment starts from ``CFG`` and overrides only what its scenario changes.
"""

CFG = dict(
    # ---- farm and traffic ----
    farm_size_m=5000.0,          # square farm, gateway in the centre
    n_sensors=200,
    frac_fixed=0.4,              # soil sensors
    frac_cow=0.4,                # cow collars (the rest are tractors)
    report_interval_s=120.0,     # one message every 2 minutes (T)
    jitter_s=10.0,               # +/- random jitter on each send time
    sim_days=3.0,

    # ---- radio options (42 combinations) ----
    sf_list=[7, 8, 9, 10, 11, 12],
    tp_list=[2, 4, 6, 8, 10, 12, 14],
    bw_hz=125e3,
    payload_bytes=20,            # application payload
    mac_overhead_bytes=13,       # LoRaWAN header + MIC
    preamble_symbols=8,
    coding_rate=1,               # 4/5

    # ---- channel model: P_rx = TP - (130 + 10 n log10(d / 1 km) + X - F) ----
    pl_d0_db=130.0,              # path loss at 1 km (includes the antenna gains)
    d0_m=1000.0,
    path_loss_exp=2.7,
    shadow_sigma_db=4.0,         # spatially correlated terrain term X
    shadow_corr_m=75.0,          # correlation distance
    shadow_grid_m=25.0,
    fading_sigma_db=2.0,         # per-message fading F
    antenna_gain_db=0.0,
    noise_floor_dbm=-117.0,      # -174 + 10log10(125 kHz) + 6 dB noise figure (used by ADR's SNR)
    n_channels=8,                # 8-channel EU868 gateway: channels 0-2 in sub-band g1, 3-7 in sub-band g
    collisions=False,
    capture_db=6.0,

    # ---- gateway replies, EU868 ----
    rx1_duty=0.01,               # 1 % duty cycle on each RX1 sub-band
    rx2_duty=0.10,               # 10 % on the RX2 channel (869.525 MHz)
    rx2_sf=9,                    # RX2 data rate (SF9, as used by The Things Network)
    dl_cmd_bytes=17,             # reply carrying LinkADRReq (header 12 + command 5)
    dl_empty_bytes=12,           # empty reply (answer to ADRACKReq)
    dl_setup_bytes=14,           # one-off ADRParamSetupReq at the start (methods with a reply budget)

    # ---- GPS ----
    gps_sigma_m=5.0,

    # ---- device ----
    voltage=3.3,
    init_sf=12, init_tp=14,      # devices start at the safest setting
    adr_ack_limit=64,            # LoRaWAN default ADR_ACK_LIMIT / ADR_ACK_DELAY
    adr_ack_delay=32,

    # ---- standard ADR (Semtech) and its variants ----
    adr_history=20,
    adr_margin_db=10.0,
    ema_beta=0.7,                # EMA-ADR smoothing factor
    ema_min_obs=2,               # EMA-ADR decides from the 2nd SNR after a change

    # ---- GPS rule (reference without learning) ----
    gps_rule_margin_db=10.0,

    # ---- proposed method ----
    target_pdr=0.95,             # tau
    forget_lambda=0.999,
    newton_iters=3,
    prior_mean=[0.0, 3.0, -8.0, 3.0],   # textbook rural link budget (~ proportional to the channel model)
    prior_std=[1.5, 1.0, 2.0, 1.0],
    min_weight=0.1,              # guard rails: more TP / higher SF never hurt, more distance never helps
    kf_q=0.02,                   # Kalman filter process noise (m^2/s^3)
    kf_v0=3.0,                   # initial velocity uncertainty (m/s)
    map_cell_m=50.0,             # coverage-map patch size
    map_prior_std=1.5,           # prior spread of a patch correction (log-odds)
    map_neighbor_w=0.3,          # how much neighbouring patches learn from one observation
    map_forget=0.998,
    calib_eta=0.004,             # per-sensor target controller step (eta)
    calib_min=0.80, calib_max=0.995,
    motion_K=8,                  # recent prediction errors kept per sensor
    motion_forget=0.9,
    motion_alpha0=0.5,           # starting guess: movement continues at half speed
    stationary_extra_m=10.0,     # stationary if expected move and recent errors < 3 x GPS noise + this
    switch_saving=0.20,          # only switch to a cheaper setting if it saves at least this much
    plan_horizon=1,              # plan for the next message only
    hysteresis_logit=0.7,        # a new setting must be safe with a margin (95 % -> ~97.5 %)
    reply_budget=4,              # B: replies per report interval (gateway-wide)
    ours_ack_limit=256,          # set with ADRParamSetupReq
    ours_ack_delay=64,

    # ---- mobility ----
    cow_speed=(0.3, 1.0),
    cow_p_stop=0.10,             # per message: grazing -> resting
    cow_p_start=0.20,            # per message: resting -> grazing
    cow_turn_sd=0.4,             # radians per message
    tractor_field_speed=(2.0, 4.0),
    tractor_transit_speed=5.0,
    field_len=(400.0, 1000.0),   # row length
    field_depth=(200.0, 600.0),
    row_spacing=20.0,
    tractor_pause_s=(0.0, 3600.0),
)

TYPE_NAMES = {0: 'fixed', 1: 'cow', 2: 'tractor'}
EVAL_SEED_BASE = 1000            # evaluation farms: 1000, 1001, ... (never used while developing the method)
DEV_SEEDS = (123, 124)           # development farms
RX_CURRENT_MA = 11.5             # SX1276 receive current, for the energy of received replies

# Scenarios. Each entry only lists what differs from CFG.
SCENARIOS = {
    'main_mix_200':     dict(),                                   # 40 % fixed, 40 % cows, 20 % tractors
    'fixed_only':       dict(frac_fixed=1.0, frac_cow=0.0),
    'cows_only':        dict(frac_fixed=0.0, frac_cow=1.0),
    'tractors_only':    dict(frac_fixed=0.0, frac_cow=0.0),
    'mix_50':           dict(n_sensors=50),
    'mix_1000':         dict(n_sensors=1000),
    'gps_poor_25m':     dict(gps_sigma_m=25.0),
    'collisions_on':    dict(collisions=True),
    'robust_terrain6':  dict(shadow_sigma_db=6.0),
    'robust_pathloss3': dict(path_loss_exp=3.0),
    'robust_fading4':   dict(fading_sigma_db=4.0),
    'more_replies_8':   dict(reply_budget=8),
}
MISMATCH_SCENARIOS = {
    'mismatch_two_slope':    dict(pl_model='two_slope', pl_n1=2.0, pl_n2=3.3),
    'mismatch_time_varying': dict(tv_extra_db=4.0, tv_hours=(0.0, 6.0)),
}
SENSITIVITY_SCENARIOS = {        # our method only, 5 farms of the main scenario
    'sens_budget2':  dict(reply_budget=2),
    'sens_saving10': dict(switch_saving=0.10),
    'sens_saving30': dict(switch_saving=0.30),
    'sens_hyst035':  dict(hysteresis_logit=0.35),
    'sens_hyst14':   dict(hysteresis_logit=1.4),
}
HALF_DUPLEX_SCENARIOS = {        # true half-duplex loss scaled; the server's own model is unchanged
    'hd_half':   dict(hd_scale=0.5),
    'hd_double': dict(hd_scale=2.0),
}
BODY_SCENARIOS = {               # 0-20 dB heading-dependent body shadowing on the cow collars
    'body_shadow_cows': dict(body_loss_db=20.0),
}


def seed_count(scenario, mode='full'):
    """Number of evaluation farms per scenario: 10 (5 with 1000 sensors); 'quick' mode uses 2."""
    n = 2 if mode == 'quick' else 10
    return min(n, 5) if scenario == 'mix_1000' else n
