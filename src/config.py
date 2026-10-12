"""User-editable defaults for the public F-16 canyon experiment."""

from __future__ import annotations


KTS_TO_FPS = 1.68781


DEMO = {
    "seed": 1340,
    "nominal_start_fraction": 0.0,
    "nominal_end_fraction": 0.1,
    "max_steps": 1200,
    "reference_spacing_ft": 12.0,
    "output_dir": "output",
}


ENVIRONMENT = {
    "dem_path": "assets/terrain/black-canyon.tif",
    "dem_bbox": (38.52, 38.62, -107.78, -107.65),
    "dem_valley_rel_elev": 0.08,
    "dem_smoothing_window": 11,
    "dem_min_width_ft": 140.0,
    "dem_max_width_ft": 2200.0,
    "dem_start_pixel": (1400, 950),
    # None uses the saved reference's initial flight state; a value overrides it.
    "dem_start_heading_deg": None,
    "dem_render_mesh": True,
    "dem_use_proxy_canyon_bounds": False,
    "wall_margin_ft": 30.0,
    "target_altitude_ft": 500.0,
    "entry_altitude_ft": None,
    "min_altitude_ft": -500.0,
    "max_altitude_ft": 3000.0,
    "terrain_collision_buffer_ft": 10.0,
    "entry_speed_kts": None,
    "entry_roll_deg": None,
    "entry_pitch_deg": None,
    "entry_alpha_deg": None,
    "entry_beta_deg": None,
    "wind_sigma": 5.0,
    "canyon_span_ft": 9000.0,
}


PID = {
    "guidance": {
        "kp_xtrk": -0.01,
        "kd_xtrk": -0.01,
        "kp_z": 0.02,
        "kd_z": 0.01,
        "k_accel_to_alpha_scale": 0.2,
        "lookahead_dist_ft": 300.0,
    },
    "autopilot": {
        "alpha_p": 3.0,
        "alpha_i": 1.0,
        "q_p": 3.0,
        "q_d": 0.0,
        "phi_p": 5.0,
        "p_p": 2.0,
        "p_d": 0.0,
        "beta_p": 3.0,
        "beta_i": 1.5,
        "washout_tau": 1.0,
        "r_highpass_gain": 0.5,
        "v_p": 0.1,
        "v_i": 0.02,
    },
}


GATEKEEPER = {
    "M": 25,
    "T": 100,
    "N": 256,
    "delta": 0.1,
    "epsilon": 0.05,
    "beta": 1.0,
    "alpha": 0.0,
    "p": 1,
    "lipschitz_mode": "fixed",
    "lipschitz_constant": 50.0,
    "update_interval_steps": 10,
}


BACKUP = {
    "target_speed_kts": 300.0,
    "altitude_margin_ft": 200.0,
    "terminal_speed_limit_kts": 600.0,
    "terminal_vertical_rate_min_fps": 0.0,
    "roll_gain": -1.6,
    "roll_rate_gain": -0.45,
    "live_roll_limit": 1.0,
    "yaw_rate_gain": -0.25,
    "yaw_cmd_min": -0.15,
    "yaw_cmd_max": 0.15,
    "pitch_altitude_gain": -0.0022,
    "pitch_vertical_rate_gain": 0.004,
    "pitch_angle_gain": 0.70,
    "pitch_rate_gain": 0.30,
    "pitch_cmd_min": -1.0,
    "pitch_cmd_max": 1.0,
    "throttle_base": 0.64,
    "throttle_speed_gain": 0.01,
    "throttle_min": 0.0,
    "throttle_max": 1.0,
    "rollout_roll_limit": 0.35,
}


UNCERTAINTY = {
    "neighbor_count": 20,
    "max_pool_size": 512,
    "epsilon_w": 200.0,
    "alpha_threshold": 0.8,
}


VISUALIZATION = {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "gif_fps": 10,
    "gif_width": 640,
    "record_video": True,
    "record_gif": True,
}

