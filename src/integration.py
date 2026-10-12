"""Connect the JSBSim truth environment to the learned rollout model."""
import jax.numpy as jnp
import numpy as np
from src import config
from src.gatekeeper import DRSGatekeeper, GatekeeperParams
from src.backup import AltitudeHoldController, build_altitude_hold_policy_jax
from src.pid import build_pid_trajectory_policy_jax
from src.dynamics import (
    f16_kinematics_step_with_load_factors, load_nominal_weights, _terrain_elevation_ft_at,
)
from src.uncertainty import RuntimeUncertaintySampler

M_TO_FT = 3.28084
KTS_TO_FPS = 1.68781

def get_active_canyon_reference(env, pad_back_ft=500.0, pad_front_ft=1500.0):
    canyon = env.unwrapped.canyon
    if not (
        hasattr(canyon, "north_samples_ft")
        and hasattr(canyon, "width_samples_ft")
        and hasattr(canyon, "center_east_samples_ft")
    ):
        north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
        width_samples_ft = np.ones(256, dtype=np.float32) * 1000.0
        center_east_samples_ft = np.zeros(256, dtype=np.float32)
        centerline_heading_samples_rad = np.zeros(256, dtype=np.float32)
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    north_samples_ft = np.asarray(canyon.north_samples_ft, dtype=np.float32)
    width_samples_ft = np.asarray(canyon.width_samples_ft, dtype=np.float32)
    center_east_samples_ft = np.asarray(canyon.center_east_samples_ft, dtype=np.float32)
    centerline_heading_samples_rad = np.asarray(
        getattr(canyon, "centerline_heading_samples_rad", np.zeros_like(center_east_samples_ft)),
        dtype=np.float32,
    )

    start_info = getattr(env.unwrapped, "dem_start_info", None)
    canyon_span_ft = float(getattr(env.unwrapped, "canyon_span_ft", 0.0))
    if start_info is None or canyon_span_ft <= 0.0 or north_samples_ft.size < 2:
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    start_north_ft = float(start_info["local_north_ft"])
    min_north_ft = max(float(north_samples_ft[0]), start_north_ft - float(pad_back_ft))
    max_north_ft = min(float(north_samples_ft[-1]), start_north_ft + canyon_span_ft + float(pad_front_ft))
    mask = (north_samples_ft >= min_north_ft) & (north_samples_ft <= max_north_ft)
    if int(np.count_nonzero(mask)) < 2:
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    return (
        north_samples_ft[mask],
        width_samples_ft[mask],
        center_east_samples_ft[mask],
        centerline_heading_samples_rad[mask],
    )

def _estimate_progress_fraction_from_position(position_n_ft, position_e_ft, path_north_ft, path_east_ft):
    path_north_ft = np.asarray(path_north_ft, dtype=np.float64).reshape(-1)
    path_east_ft = np.asarray(path_east_ft, dtype=np.float64).reshape(-1)
    if path_north_ft.size < 2 or path_east_ft.size != path_north_ft.size:
        return float("nan")

    segment_n = np.diff(path_north_ft)
    segment_e = np.diff(path_east_ft)
    segment_len = np.hypot(segment_n, segment_e)
    total_len = float(np.sum(segment_len))
    if not np.isfinite(total_len) or total_len <= 1e-9:
        return float("nan")

    point_n = float(position_n_ft)
    point_e = float(position_e_ft)
    best_distance_sq = float("inf")
    best_progress_ft = 0.0
    cumulative_start_ft = 0.0

    for idx, seg_len in enumerate(segment_len):
        if seg_len <= 1e-9:
            cumulative_start_ft += float(seg_len)
            continue
        start_n = path_north_ft[idx]
        start_e = path_east_ft[idx]
        dn = segment_n[idx]
        de = segment_e[idx]
        t = ((point_n - start_n) * dn + (point_e - start_e) * de) / float(seg_len * seg_len)
        t = float(np.clip(t, 0.0, 1.0))
        proj_n = start_n + t * dn
        proj_e = start_e + t * de
        distance_sq = (point_n - proj_n) ** 2 + (point_e - proj_e) ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq = float(distance_sq)
            best_progress_ft = cumulative_start_ft + t * float(seg_len)
        cumulative_start_ft += float(seg_len)

    return float(np.clip(best_progress_ft / total_len, 0.0, 1.0))


def controller_state(env):
    state = env.get_full_state_dict()
    north, east = env.canyon.get_local_from_latlon(
        env.simulation.get_property_value("position/lat-gc-deg"),
        env.simulation.get_property_value("position/long-gc-deg"),
    )
    return dict(state, p_N=north, p_E=east, h=state["h"] - env.dem_start_elev_ft)


def state_vector(state):
    return jnp.asarray([state[k] for k in (
        "p_N", "p_E", "h", "u", "v", "w", "p", "q", "r",
        "phi", "theta", "psi", "ny", "nz",
    )], dtype=jnp.float32)


def build_empirical_features(canyon, active_names, north_samples_ft=None, width_samples_ft=None):
    """Pack one query in precisely the learned artifact's feature order."""
    north_np = canyon.north_samples_ft if north_samples_ft is None else north_samples_ft
    width_np = canyon.width_samples_ft if width_samples_ft is None else width_samples_ft
    north = jnp.asarray(north_np)
    widths = jnp.asarray(width_np)
    width_grad = jnp.asarray(np.gradient(width_np, north_np))

    def features(state, action, previous, step):
        del step
        u, v, w = state[3:6]
        speed_sq = u*u + v*v + w*w
        speed = jnp.sqrt(jnp.maximum(speed_sq, 1.0))
        values = {
            "alpha": jnp.arctan2(w, jnp.maximum(u, 1.0)),
            "beta": jnp.arcsin(jnp.clip(v / speed, -1.0, 1.0)),
            "mach": speed / 1116.45,
            "p": state[6], "q": state[7], "r": state[8],
            "delta_a": action[0], "delta_e": action[1],
            "delta_r": action[2], "delta_t": action[3],
            "prev_delta_a": previous[0], "prev_delta_e": previous[1],
            "prev_delta_r": previous[2], "prev_delta_t": previous[3],
            "qbar": jnp.maximum(0.5 * 0.0023769 * speed_sq, 1.0),
            "canyon_width": jnp.interp(state[0], north, widths),
            "canyon_width_grad": jnp.interp(state[0], north, width_grad),
        }
        return jnp.asarray([values[name] for name in active_names], dtype=jnp.float32)

    return features


def build_gatekeeper(env, initial_state, reference, seed):
    canyon = env.canyon
    north, widths, _, _ = get_active_canyon_reference(env)
    full_north = np.asarray(canyon.north_samples_ft)
    top = np.nanmin(canyon.ordered_dem_msl_m, axis=1) * M_TO_FT + canyon.wall_height_samples_ft
    target_altitude = float(np.max(np.interp(north, full_north, top))
                            - env.dem_start_elev_ft + config.BACKUP["altitude_margin_ft"])
    target_speed = config.BACKUP["target_speed_kts"] * KTS_TO_FPS
    backup = AltitudeHoldController(target_altitude_ft=target_altitude, target_speed_fps=target_speed)
    backup.reset(state_dict=initial_state)
    backup_policy = build_altitude_hold_policy_jax(target_altitude, target_speed)
    nominal_policy = build_pid_trajectory_policy_jax(reference, config=config.PID)
    W, B, powers, thrust = load_nominal_weights()
    terrain = jnp.asarray(canyon.ordered_dem_msl_m * M_TO_FT - env.dem_start_elev_ft)
    terrain_north, terrain_east = jnp.asarray(full_north), jnp.asarray(canyon.east_samples_ft)

    def dynamics(state, action, noise):
        return f16_kinematics_step_with_load_factors(state, action, W, B + noise, powers, thrust)

    def safety(state, _unused):
        ground = _terrain_elevation_ft_at(state[0], state[1], terrain_north, terrain_east, terrain)
        return state[2] - ground - env.terrain_collision_buffer_ft

    def terminal(state):
        u, v, w = state[3:6]
        speed = jnp.sqrt(jnp.maximum(u*u + v*v + w*w, 1.0))
        phi, theta = state[9], state[10]
        vertical_rate = u*jnp.sin(theta) - v*jnp.sin(phi)*jnp.cos(theta) - w*jnp.cos(phi)*jnp.cos(theta)
        speed_margin = config.BACKUP["terminal_speed_limit_kts"] * KTS_TO_FPS - speed
        climb_margin = vertical_rate - config.BACKUP["terminal_vertical_rate_min_fps"]
        # Preserve the original terminal score cap; this is no corridor constraint.
        return jnp.minimum(100.0, jnp.minimum(speed_margin, climb_margin))

    sampler = RuntimeUncertaintySampler(**config.UNCERTAINTY)
    settings = {k: v for k, v in config.GATEKEEPER.items() if k != "update_interval_steps"}
    gatekeeper = DRSGatekeeper(
        GatekeeperParams(**settings), dynamics, nominal_policy, backup_policy, safety, terminal,
        sampler, build_empirical_features(canyon, sampler.configure_active_features(), north, widths), seed=seed,
    )
    gatekeeper.reset(state_vector(initial_state))
    return gatekeeper, backup
