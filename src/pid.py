from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import PID


DEFAULT_PID_TRAJ_GUIDANCE_GAINS = dict(PID["guidance"])
DEFAULT_PID_TRAJ_AUTOPILOT_GAINS = dict(PID["autopilot"])


def _resolve_pid_traj_gains(config=None):
    config = {} if config is None else dict(config)
    guidance_gains = dict(DEFAULT_PID_TRAJ_GUIDANCE_GAINS)
    guidance_gains.update(dict(config.get('guidance', {})))
    autopilot_gains = dict(DEFAULT_PID_TRAJ_AUTOPILOT_GAINS)
    autopilot_gains.update(dict(config.get('autopilot', {})))
    return guidance_gains, autopilot_gains

class PIDTrajectoryController:
    """State-feedback PID controller for a nominal local trajectory."""
    def __init__(self, env, reference_trajectory, config=None):
        self.env = env
        self.config = {} if config is None else dict(config)
        self._guidance_gains, self._autopilot_gains = _resolve_pid_traj_gains(self.config)

        north_ft = reference_trajectory['north_ft']
        east_ft = reference_trajectory['east_ft']
        alt_ft = reference_trajectory['altitude_ft']

        self.nominal_trajectory_data = {
            'enu': np.column_stack((east_ft, north_ft, alt_ft)),
            'heading_rad': reference_trajectory['heading_rad'],
            'phi_opt': reference_trajectory['phi_rad'],
            'alpha_opt': reference_trajectory['alpha_rad'],
            'v_opt': reference_trajectory['speed_fps']
        }

        self._recreate_control_loops()

    def _recreate_control_loops(self):
        self.guidance = SpatialGuidance(self._guidance_gains, self.nominal_trajectory_data)
        self.guidance.lla_to_enu = (
            lambda current_e, current_n, current_u, *_: np.asarray(
                [current_e, current_n, current_u], dtype=np.float64
            )
        )
        self.autopilot = F16Autopilot(self._autopilot_gains)

    def reset(self, initial_state=None):
        del initial_state
        self._recreate_control_loops()
        self.last_diagnostics = {}

    def get_action(self, obs=None):
        del obs
        dt = 1.0 / 30.0

        lat_rad = self.env.unwrapped.simulation.get_property_value("position/lat-gc-rad")
        lon_rad = self.env.unwrapped.simulation.get_property_value("position/long-gc-rad")
        alt = self.env.unwrapped.simulation.get_property_value("position/h-sl-ft")

        altitude_ref_ft = float(getattr(self.env.unwrapped, "dem_start_elev_ft", 0.0))
        h_rel = alt - altitude_ref_ft

        north_ft, east_ft = self.env.unwrapped.canyon.get_local_from_latlon(np.degrees(lat_rad), np.degrees(lon_rad))
        current_enu_ft = (east_ft, north_ft, h_rel)

        V = self.env.unwrapped.simulation.get_property_value("velocities/vt-fps")
        p = self.env.unwrapped.simulation.get_property_value("velocities/p-rad_sec")
        q = self.env.unwrapped.simulation.get_property_value("velocities/q-rad_sec")
        r = self.env.unwrapped.simulation.get_property_value("velocities/r-rad_sec")

        alpha = self.env.unwrapped.simulation.get_property_value("aero/alpha-rad")
        beta = self.env.unwrapped.simulation.get_property_value("aero/beta-rad")

        phi = self.env.unwrapped.simulation.get_property_value("attitude/phi-rad")

        phi_cmd, alpha_cmd, v_opt = self.guidance.update(
            current_enu_ft,
            (0, 0, 0),
            dt,
            current_phi=phi,
        )

        cmds = (phi_cmd, alpha_cmd, v_opt)
        states = (alpha, beta, phi, p, q, r, V)

        elevator, aileron, rudder, throttle = self.autopilot.update(cmds, states, dt)

        self.last_diagnostics = {}
        self.last_diagnostics.update(self.guidance.last_diagnostics)
        self.last_diagnostics.update(self.autopilot.last_diagnostics)

        return np.array([aileron, elevator, 0.0 * rudder, throttle], dtype=np.float32)


def build_pid_trajectory_policy_jax(reference_trajectory, config=None):
    """
    Build a JAX nominal policy for gatekeeper rollouts that applies state feedback
    on each simulated rollout state.

    Gatekeeper rollout state convention:
    [p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi, ny, nz]
    """
    guidance_gains, autopilot_gains = _resolve_pid_traj_gains(config)

    north_ft = np.asarray(reference_trajectory['north_ft'], dtype=np.float32).reshape(-1)
    east_ft = np.asarray(reference_trajectory['east_ft'], dtype=np.float32).reshape(-1)
    alt_ft = np.asarray(reference_trajectory['altitude_ft'], dtype=np.float32).reshape(-1)
    heading_rad = np.asarray(reference_trajectory['heading_rad'], dtype=np.float32).reshape(-1)
    phi_opt = np.asarray(reference_trajectory['phi_rad'], dtype=np.float32).reshape(-1)
    alpha_opt = np.asarray(reference_trajectory['alpha_rad'], dtype=np.float32).reshape(-1)
    v_opt = np.asarray(reference_trajectory['speed_fps'], dtype=np.float32).reshape(-1)

    sample_count = int(north_ft.size)
    if sample_count < 2:
        raise ValueError("PID trajectory nominal policy requires at least two trajectory samples.")
    if not (
        east_ft.size == sample_count
        and alt_ft.size == sample_count
        and heading_rad.size == sample_count
        and phi_opt.size == sample_count
        and alpha_opt.size == sample_count
        and v_opt.size == sample_count
    ):
        raise ValueError("PID trajectory nominal arrays must all have matching sample counts.")

    path_enu = np.column_stack((east_ft, north_ft, alt_ft)).astype(np.float32)
    segment_lengths = np.linalg.norm(np.diff(path_enu, axis=0), axis=1)
    segment_lengths = np.where(np.isfinite(segment_lengths), np.maximum(segment_lengths, 0.0), 0.0)
    path_s = np.concatenate([[0.0], np.cumsum(segment_lengths, dtype=np.float64)]).astype(np.float32)

    path_enu_jax = jnp.asarray(path_enu, dtype=jnp.float32)
    path_heading_jax = jnp.asarray(heading_rad, dtype=jnp.float32)
    phi_opt_jax = jnp.asarray(phi_opt, dtype=jnp.float32)
    alpha_opt_jax = jnp.asarray(alpha_opt, dtype=jnp.float32)
    v_opt_jax = jnp.asarray(v_opt, dtype=jnp.float32)
    path_s_jax = jnp.asarray(path_s, dtype=jnp.float32)

    lookahead_dist_ft = float(guidance_gains.get('lookahead_dist_ft', 500.0))
    kp_xtrk = float(guidance_gains.get('kp_xtrk', 0.0))
    kp_z = float(guidance_gains.get('kp_z', 0.0))
    k_accel_to_alpha_scale = float(guidance_gains.get('k_accel_to_alpha_scale', 0.0))

    alpha_p = float(autopilot_gains.get('alpha_p', 0.0))
    q_p = float(autopilot_gains.get('q_p', 0.0))
    phi_p = float(autopilot_gains.get('phi_p', 0.0))
    p_p = float(autopilot_gains.get('p_p', 0.0))
    v_p = float(autopilot_gains.get('v_p', 0.0))

    max_idx = sample_count - 1

    def _wrap_angle(angle_rad):
        return jnp.arctan2(jnp.sin(angle_rad), jnp.cos(angle_rad))

    def policy_fn(state_flat):
        p_n = state_flat[0]
        p_e = state_flat[1]
        h = state_flat[2]
        u = state_flat[3]
        v = state_flat[4]
        w = state_flat[5]
        p_rate = state_flat[6]
        q_rate = state_flat[7]
        phi = state_flat[9]

        speed = jnp.sqrt(jnp.maximum(u * u + v * v + w * w, 1.0))
        alpha = jnp.arctan2(w, jnp.maximum(u, 1.0))

        current_enu = jnp.stack([p_e, p_n, h]).astype(jnp.float32)
        diff = path_enu_jax - current_enu[None, :]
        idx = jnp.argmin(jnp.sum(jnp.square(diff), axis=1))

        target_s = path_s_jax[idx] + lookahead_dist_ft
        target_idx = jnp.searchsorted(path_s_jax, target_s, side='left')
        target_idx = jnp.clip(target_idx, idx, max_idx)

        closest_p = path_enu_jax[idx]
        trk_hdg = path_heading_jax[idx]
        d_east = current_enu[0] - closest_p[0]
        d_north = current_enu[1] - closest_p[1]
        e_xtrk = d_north * (-jnp.sin(trk_hdg)) + d_east * jnp.cos(trk_hdg)

        lookahead_alt = path_enu_jax[target_idx][2]
        closest_alt = closest_p[2]
        target_alt = jnp.maximum(closest_alt, lookahead_alt)

        e_z = target_alt - current_enu[2] + 100.0

        delta_lateral_accel = jnp.clip(kp_xtrk * e_xtrk, -4.0, 4.0)
        delta_vertical_accel = jnp.clip(kp_z * e_z, -2.0, 9.0)

        open_loop_vertical_accel = jnp.cos(phi_opt_jax[idx])
        open_loop_lateral_accel = jnp.sin(phi_opt_jax[target_idx])

        total_lateral_accel = delta_lateral_accel + open_loop_lateral_accel
        total_vertical_accel = delta_vertical_accel + open_loop_vertical_accel

        accel_direction = jnp.arctan2(total_lateral_accel, total_vertical_accel)
        phi_cmd = jnp.clip(accel_direction, -jnp.radians(120.0), jnp.radians(120.0))

        lift_y = jnp.sin(phi)
        lift_z = jnp.cos(phi)
        M_effective = total_lateral_accel * lift_y + total_vertical_accel * lift_z
        M_effective = jnp.maximum(0.0, M_effective)

        delta_lift_Gs = M_effective - 1.0

        alpha_cmd = alpha_opt_jax[idx] + delta_lift_Gs * k_accel_to_alpha_scale
        alpha_cmd = jnp.clip(alpha_cmd, -jnp.radians(5.0), jnp.radians(20.0))

        v_cmd = v_opt_jax[target_idx]

        q_cmd = jnp.clip(alpha_p * (alpha_cmd - alpha), -10.0, 10.0)
        elevator = -jnp.clip(q_p * (q_cmd - q_rate), -1.0, 1.0)

        phi_error = _wrap_angle(phi_cmd - phi)
        p_cmd = jnp.clip(phi_p * phi_error, -10.0, 10.0)
        aileron = jnp.clip(p_p * (p_cmd - p_rate), -1.0, 1.0)

        throttle = jnp.clip(v_p * (v_cmd - speed), 0.0, 1.0)

        return jnp.stack([aileron, elevator, jnp.zeros_like(aileron), throttle]).astype(jnp.float32)

    policy_fn_jit = jax.jit(policy_fn)
    return policy_fn_jit

class DiscretePID:
    """Discrete-time PID controller with anti-windup."""
    def __init__(self, kp, ki, kd, output_min=-1.0, output_max=1.0, wrap_angle=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.wrap_angle = wrap_angle
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, process_variable, dt):
        if dt <= 0:
            return 0.0

        # normalize angles if requested
        if self.wrap_angle:
            setpoint = np.arctan2(np.sin(setpoint), np.cos(setpoint))
            process_variable = np.arctan2(np.sin(process_variable), np.cos(process_variable))
            error = setpoint - process_variable
            error = np.arctan2(np.sin(error), np.cos(error))
        else:
            error = setpoint - process_variable

        # Proportional
        p_term = self.kp * error

        # Derivative (backward difference)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Temporary output without integral
        temp_out = p_term + self.integral + d_term

        # Anti-windup (conditional integration)
        # Only integrate if we are not saturated or if the integral will reduce saturation
        if not (temp_out >= self.output_max and error > 0) and not (temp_out <= self.output_min and error < 0):
             # Forward Euler integration
             self.integral += self.ki * error * dt

        # Final output
        output = p_term + self.integral + d_term
        output = np.clip(output, self.output_min, self.output_max)

        self.prev_error = error
        return output

class WashoutFilter:
    """High-pass washout filter for yaw rate (Dutch roll damping)."""
    def __init__(self, time_constant):
        self.tau = time_constant
        self.x = 0.0
        self.prev_input = 0.0

    def update(self, current_input, dt):
        if self.tau == 0:
            return current_input
        # Tustin or simple backward diff depending on discretisation
        # Simple high pass: y(k) = (tau/(tau+dt))*y(k-1) + (tau/(tau+dt))*(u(k)-u(k-1))
        alpha = self.tau / (self.tau + dt)
        filtered = alpha * self.x + alpha * (current_input - self.prev_input)
        self.x = filtered
        self.prev_input = current_input
        return filtered

class SpatialGuidance:
    """Translates 3D positional errors into aerodynamic corrections using lookahead and nz-scaling."""
    def __init__(self, gains, nominal_trajectory_data):
        self.kp_xtrk = gains.get('kp_xtrk', 0.05)
        self.kd_xtrk = gains.get('kd_xtrk', 0.01)
        self.kp_z = gains.get('kp_z', 0.01)
        self.kd_z = gains.get('kd_z', 0.005)
        self.k_accel_to_alpha_scale = gains.get('k_accel_to_alpha_scale', 0.0)
        self.lookahead_dist = gains.get('lookahead_dist_ft', 500.0)


        self.path_enu = nominal_trajectory_data['enu'] # (N, 3) East, North, Up
        self.path_heading = nominal_trajectory_data['heading_rad'] # (N,)
        self.alpha_opt = nominal_trajectory_data['alpha_opt'] # (N,)
        self.phi_opt = nominal_trajectory_data['phi_opt'] # (N,)
        self.v_opt = nominal_trajectory_data['v_opt'] # (N,)

        self.prev_e_xtrk = 0.0
        self.prev_e_z = 0.0

    def lla_to_enu(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2*f - f**2

        d_lat = np.radians(lat - ref_lat)
        d_lon = np.radians(lon - ref_lon)
        ref_lat_rad = np.radians(ref_lat)

        N = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
        M = a * (1 - e2) / (1 - e2 * np.sin(ref_lat_rad)**2)**1.5

        east = d_lon * N * np.cos(ref_lat_rad)
        north = d_lat * M
        up = alt - ref_alt

        return np.array([east, north, up])

    def find_closest_point(self, current_enu):
        distances = np.linalg.norm(self.path_enu - current_enu, axis=1)
        return np.argmin(distances)

    def get_lookahead_index(self, closest_idx):
        """Finds the index of the point lookahead_dist ahead on the path."""
        accumulated_dist = 0.0
        curr_idx = closest_idx
        max_idx = len(self.path_enu) - 1

        while accumulated_dist < self.lookahead_dist and curr_idx < max_idx:
            step_dist = np.linalg.norm(self.path_enu[curr_idx + 1] - self.path_enu[curr_idx])
            accumulated_dist += step_dist
            curr_idx += 1

        return curr_idx

    def update(self, current_lla, ref_lla, dt, current_phi=None):
        current_enu = self.lla_to_enu(*current_lla, *ref_lla)

        idx = self.find_closest_point(current_enu)
        target_idx = self.get_lookahead_index(idx)

        closest_p = self.path_enu[idx]
        trk_hdg = self.path_heading[idx]

        # Calculate errors relative to the closest point laterally
        d_east = current_enu[0] - closest_p[0]
        d_north = current_enu[1] - closest_p[1]

        e_xtrk = d_north * (-np.sin(trk_hdg)) + d_east * (np.cos(trk_hdg))

        # Asymmetrical vertical lookahead: anticipate pull-ups, but not dives
        lookahead_alt = self.path_enu[target_idx][2]
        closest_alt = closest_p[2]
        target_alt = max(closest_alt, lookahead_alt)

        e_z = target_alt - current_enu[2] + 100.0


        if dt > 0:
            e_xtrk_dot = (e_xtrk - self.prev_e_xtrk) / dt
            e_z_dot = (e_z - self.prev_e_z) / dt
        else:
            e_xtrk_dot = 0.0
            e_z_dot = 0.0

        self.prev_e_xtrk = e_xtrk
        self.prev_e_z = e_z

        # Roll guidance: error correction + lookahead feedforward
        delta_lateral_accel = np.clip(self.kp_xtrk * e_xtrk + self.kd_xtrk * e_xtrk_dot, -4.0, 4.0)

        # Pitch guidance: Load factor (nz) based altitude correction
        delta_vertical_accel = np.clip(self.kp_z * e_z + self.kd_z * e_z_dot, -2.0, 9.0)

        open_loop_vertical_accel = np.cos(self.phi_opt[idx])
        open_loop_lateral_accel = np.sin(self.phi_opt[target_idx])

        total_lateral_accel = delta_lateral_accel + open_loop_lateral_accel
        total_vertical_accel = delta_vertical_accel + open_loop_vertical_accel

        accel_direction = np.arctan2(total_lateral_accel, total_vertical_accel)
        phi_cmd = np.clip(accel_direction, -np.radians(120.0), np.radians(120.0))

        # Calculate the extra lift (Gs) needed by projecting the desired
        # acceleration vector onto the current lift vector direction.
        if current_phi is not None:
            # Lift vector components based on current bank angle
            lift_y = np.sin(current_phi)
            lift_z = np.cos(current_phi)
            M_effective = total_lateral_accel * lift_y + total_vertical_accel * lift_z
            # Prevent negative load factor demands in this logic (clip to 0 or 1g as needed)
            M_effective = max(0.0, M_effective)
        else:
            M_effective = np.sqrt(total_lateral_accel**2 + total_vertical_accel**2)

        delta_lift_Gs = M_effective - 1.0

        # Scale the nominal 1g lookahead alpha by the commanded load factor
        alpha_cmd = self.alpha_opt[idx] + delta_lift_Gs * self.k_accel_to_alpha_scale
        alpha_cmd = np.clip(alpha_cmd, -np.radians(5.0), np.radians(20.0))

        v_opt_val = self.v_opt[target_idx]

        self.last_diagnostics = {
            'e_xtrk': float(e_xtrk),
            'e_z': float(e_z),
            'e_xtrk_dot': float(e_xtrk_dot),
            'e_z_dot': float(e_z_dot),
            'closest_idx': int(idx),
            'target_idx': int(target_idx),
            'phi_ref': float(self.phi_opt[target_idx]),
            'alpha_ref': float(self.alpha_opt[target_idx]),
            'phi_cmd': float(phi_cmd),
            'alpha_cmd': float(alpha_cmd),
            'v_opt_val': float(v_opt_val)
        }

        return phi_cmd, alpha_cmd, v_opt_val
class F16Autopilot:
    """Translates guidance commands into actuator deflections."""
    def __init__(self, gains):
        # Pitch Axis (Elevator): outer PI on alpha, inner PD on q
        self.alpha_pi = DiscretePID(gains['alpha_p'], gains['alpha_i'], 0.0, -2.0, 2.0)
        self.q_pd = DiscretePID(gains['q_p'], 0.0, gains['q_d'], -1.0, 1.0)

        # Roll Axis (Aileron): outer P on phi, inner PD on p
        self.phi_p = DiscretePID(gains['phi_p'], 0.0, 0.0, -10.0, 10.0, wrap_angle=True)
        self.p_pd = DiscretePID(gains['p_p'], 0.0, gains['p_d'], -1.0, 1.0)

        # Yaw Axis (Rudder - Turn Coordination): PI on beta
        self.beta_pi = DiscretePID(gains['beta_p'], gains['beta_i'], 0.0, -1.0, 1.0)
        self.yaw_washout = WashoutFilter(time_constant=gains.get('washout_tau', 1.0))
        self.r_gain = gains.get('r_highpass_gain', 1.0)

        # Speed Axis (Throttle): PI on V
        self.v_pi = DiscretePID(gains['v_p'], gains['v_i'], 0.0, 0.0, 1.0)

    def update(self, cmds, states, dt):
        """
        cmds: (phi_cmd, alpha_cmd, v_opt)
        states: (alpha, beta, phi, p, q, r, V)
        """
        phi_cmd, alpha_cmd, v_opt = cmds
        alpha, beta, phi, p, q, r, V = states
        # Pitch Axis
        q_cmd = self.alpha_pi.update(alpha_cmd, alpha, dt)
        elevator = -self.q_pd.update(q_cmd, q, dt)

        # Roll Axis
        p_cmd = self.phi_p.update(phi_cmd, phi, dt)
        aileron = self.p_pd.update(p_cmd, p, dt)

        # Yaw Axis
        rudder_beta = self.beta_pi.update(0.0, beta, dt)
        r_highpass = self.yaw_washout.update(r, dt)
        rudder = np.clip(rudder_beta + self.r_gain * r_highpass, -1.0, 1.0)

        # Speed Axis
        throttle = self.v_pi.update(v_opt, V, dt)

        self.last_diagnostics = {
            'q_cmd': float(q_cmd),
            'e_alpha': float(alpha_cmd - alpha),
            'e_q': float(q_cmd - q),
            'elevator_cmd': float(elevator),
            'p_cmd': float(p_cmd),
            'e_phi': float(phi_cmd - phi),
            'e_p': float(p_cmd - p),
            'aileron_cmd': float(aileron),
            'rudder_beta': float(rudder_beta),
            'r_highpass': float(r_highpass),
            'rudder_cmd': float(rudder),
            'e_V': float(v_opt - V),
            'throttle_cmd': float(throttle)
        }

        return elevator, aileron, rudder, throttle

