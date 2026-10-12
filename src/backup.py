"""Conservative altitude and speed backup controller."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from .config import BACKUP as DEFAULT_BACKUP_CONFIG
from .config import KTS_TO_FPS


def _resolve_backup_config(config: Mapping[str, float] | None = None) -> dict[str, float]:
    gains = dict(DEFAULT_BACKUP_CONFIG)
    if config is not None:
        gains.update(dict(config))
    return gains


class AltitudeHoldController:
    """Straight-flight backup with altitude, attitude, and speed feedback."""

    def __init__(
        self,
        target_altitude_ft: float | None = None,
        target_speed_fps: float | None = None,
        config: Mapping[str, float] | None = None,
    ):
        self.config = _resolve_backup_config(config)
        self.target_altitude_ft = 1000.0 if target_altitude_ft is None else float(target_altitude_ft)
        if target_speed_fps is None:
            target_speed_fps = float(self.config["target_speed_kts"]) * KTS_TO_FPS
        self.target_speed_fps = float(target_speed_fps)

    def reset(
        self,
        state_dict=None,
        target_altitude_ft=None,
        reference_trajectory=None,
        target_speed_fps=None,
        **kwargs,
    ):
        del state_dict, reference_trajectory, kwargs
        if target_altitude_ft is not None:
            self.target_altitude_ft = float(target_altitude_ft)
        if target_speed_fps is not None:
            self.target_speed_fps = float(target_speed_fps)

    def get_action(self, obs_or_state):
        h = float(obs_or_state["h"])
        altitude_error_ft = float(self.target_altitude_ft - h)

        roll_angle = float(obs_or_state["phi"])
        pitch_angle = float(obs_or_state["theta"])
        roll_rate = float(obs_or_state["p"])
        pitch_rate = float(obs_or_state["q"])
        yaw_rate = float(obs_or_state["r"])
        u = float(obs_or_state["u"])
        v = float(obs_or_state["v"])
        w = float(obs_or_state["w"])
        speed_fps = float(np.sqrt(max(u * u + v * v + w * w, 0.0)))
        v_rate = (
            u * np.sin(pitch_angle)
            - v * np.sin(roll_angle) * np.cos(pitch_angle)
            - w * np.cos(roll_angle) * np.cos(pitch_angle)
        )

        roll_cmd = np.clip(
            float(self.config["roll_gain"]) * roll_angle
            + float(self.config["roll_rate_gain"]) * roll_rate,
            -float(self.config["live_roll_limit"]),
            float(self.config["live_roll_limit"]),
        )
        yaw_cmd = np.clip(
            float(self.config["yaw_rate_gain"]) * yaw_rate,
            float(self.config["yaw_cmd_min"]),
            float(self.config["yaw_cmd_max"]),
        )
        pitch_cmd = np.clip(
            float(self.config["pitch_altitude_gain"]) * altitude_error_ft
            + float(self.config["pitch_vertical_rate_gain"]) * v_rate
            + float(self.config["pitch_angle_gain"]) * pitch_angle
            + float(self.config["pitch_rate_gain"]) * pitch_rate,
            float(self.config["pitch_cmd_min"]),
            float(self.config["pitch_cmd_max"]),
        )

        speed_error_fps = self.target_speed_fps - speed_fps
        throttle_cmd = np.clip(
            float(self.config["throttle_base"])
            + float(self.config["throttle_speed_gain"]) * speed_error_fps,
            float(self.config["throttle_min"]),
            float(self.config["throttle_max"]),
        )
        return np.asarray([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


def build_altitude_hold_policy_jax(
    target_altitude_ft: float,
    target_speed_fps: float,
    config: Mapping[str, float] | None = None,
):
    """Build a stateless JAX policy using the same backup gains as live control."""

    gains = _resolve_backup_config(config)
    target_altitude_ft = float(target_altitude_ft)
    target_speed_fps = float(target_speed_fps)

    def policy_fn(state_flat):
        h = state_flat[2]
        u = state_flat[3]
        v = state_flat[4]
        w = state_flat[5]
        roll_rate = state_flat[6]
        pitch_rate = state_flat[7]
        yaw_rate = state_flat[8]
        roll_angle = state_flat[9]
        pitch_angle = state_flat[10]

        speed_fps = jnp.sqrt(jnp.maximum(u * u + v * v + w * w, 0.0))
        altitude_error_ft = target_altitude_ft - h
        v_rate = (
            u * jnp.sin(pitch_angle)
            - v * jnp.sin(roll_angle) * jnp.cos(pitch_angle)
            - w * jnp.cos(roll_angle) * jnp.cos(pitch_angle)
        )

        roll_cmd = jnp.clip(
            float(gains["roll_gain"]) * roll_angle
            + float(gains["roll_rate_gain"]) * roll_rate,
            -float(gains["rollout_roll_limit"]),
            float(gains["rollout_roll_limit"]),
        )
        yaw_cmd = jnp.clip(
            float(gains["yaw_rate_gain"]) * yaw_rate,
            float(gains["yaw_cmd_min"]),
            float(gains["yaw_cmd_max"]),
        )
        pitch_cmd = jnp.clip(
            float(gains["pitch_altitude_gain"]) * altitude_error_ft
            + float(gains["pitch_vertical_rate_gain"]) * v_rate
            + float(gains["pitch_angle_gain"]) * pitch_angle
            + float(gains["pitch_rate_gain"]) * pitch_rate,
            float(gains["pitch_cmd_min"]),
            float(gains["pitch_cmd_max"]),
        )
        speed_error_fps = target_speed_fps - speed_fps
        throttle_cmd = jnp.clip(
            float(gains["throttle_base"]) + float(gains["throttle_speed_gain"]) * speed_error_fps,
            float(gains["throttle_min"]),
            float(gains["throttle_max"]),
        )
        return jnp.stack([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd]).astype(jnp.float32)

    return policy_fn

