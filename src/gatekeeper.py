"""Distributionally robust switching with state-conditioned coefficient residuals.

The sampled rollout stage runs in JAX; confidence bounds and commitment
selection run on the CPU. The F-16 safety and terminal checks are callbacks.
"""
from __future__ import annotations
from typing import Callable, NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.special import betaincinv
from src.uncertainty import JAXEmpiricalData, sample_empirical_jax

Array = jax.Array

class GatekeeperParams(NamedTuple):
    """
    Static hyperparameters for DRS-GK (Algorithm 2).

    Attributes
    ----------
    M : int
        Number of candidate switching times {0, ..., M-1}.
    T : int
        Rollout horizon (number of steps).
    N : int
        Number of noise trajectory samples per candidate switching time.
    delta : float
        Incorrectness tolerance.  The chosen switching time satisfies the DRCC
        with probability ≥ 1-δ.
    epsilon : float
        Per-decision failure tolerance (safety level).
    beta : float
        Wasserstein ambiguity-set radius.
    alpha : float
        Probability with which the backup policy renders C probabilistically
        controlled-invariant (PCI).  Used in the feasibility threshold
        (ε - α) / (1 - α).
    p : int
        Order of the Wasserstein distance (typically 1 or 2).
    lipschitz_mode : str
        How to compute the Lipschitz constant L_H.
        'autodiff' — estimate via jax.grad through the rollout (expensive).
        'fixed'    — use `lipschitz_constant` directly (cheap).
    lipschitz_constant : float
        Fixed L_H value used when lipschitz_mode='fixed'.
    """

    M: int
    T: int
    N: int
    delta: float
    epsilon: float
    beta: float
    alpha: float
    p: int = 1
    lipschitz_mode: str = "fixed"   # 'fixed' | 'autodiff'
    lipschitz_constant: float = 0.0

class GatekeeperState(NamedTuple):
    """
    Mutable runtime state for DRS-GK.

    Attributes
    ----------
    t : int
        Current timestep (absolute, not within any rollout).
    s_prev : int
        Switching time from the previous iteration.  If no new feasible time
        is found the algorithm falls back to this value.
    predicted_trajectories : Optional[np.ndarray]
        Visualization data.
    failure_mask : Optional[np.ndarray]
        Boolean mask of failures for the predicted trajectories.
    m_star : int
        Distance to switching point.
    q_bar_star : float
        Estimated failure probability bound for the chosen switching time.
    """

    t: int
    s_prev: int
    using_backup: bool = False
    predicted_trajectories: Optional[np.ndarray] = None
    failure_mask: Optional[np.ndarray] = None
    noise_l2: Optional[np.ndarray] = None
    noise_maxabs: Optional[np.ndarray] = None
    m_star: int = 0
    q_bar_star: float = 0.0
    is_reverting: bool = False
    plan_start_t: int = 0

def rollout_single(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """Uses an initial state, switch time, noise, and model functions to return the minimum
    safety margin and flight path.
    """
    def step_fn(carry, inputs):
        """Uses the current state, step, noise, and environment input to return the next state
        and safety margin.
        """
        state, step_idx = carry
        noise_step, env_param_step = inputs
        use_nominal = step_idx < switching_offset

        # Feedback nominal policy or a supplied action sequence
        if nominal_trajectory is not None:
            action_nominal = nominal_trajectory[step_idx]
        else:
            # Evaluate nominal feedback on the rollout state
            action_nominal = nominal_policy_fn(state)

        action_backup = backup_policy_fn(state)
        action = jnp.where(use_nominal, action_nominal, action_backup)
        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_param_step)
        # Return full state for debugging/visualization.
        return (next_state, step_idx + 1), (h_val, state)

    final_carry, (h_values, state_traj) = lax.scan(
        step_fn, (x0, 0), (noise_traj, env_params_traj), length=T
    )
    final_state = final_carry[0]
    state_traj = jnp.concatenate([state_traj, final_state[None, :]], axis=0)

    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), state_traj

def rollout_single_empirical(
    x0: Array,
    switching_offset: Array,
    rng_key: Array,
    env_params: Array,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    empirical_feature_fn: Callable,
    empirical_sampler_fn: Callable,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """Uses an initial state, switch time, model functions, and saved uncertainty to sample a
    flight and return its safety margin, noise, and path.
    """
    def step_fn(carry, subkey):
        """Uses the current state, previous action, and random key to sample noise and return the
        next state, safety margin, and noise.
        """
        state, step_idx, prev_action = carry
        use_nominal = step_idx < switching_offset

        # Feedback nominal policy or a supplied action sequence
        if nominal_trajectory is not None:
            action_nominal = nominal_trajectory[step_idx]
        else:
            # Evaluate nominal feedback on the rollout state
            action_nominal = nominal_policy_fn(state)

        action_backup = backup_policy_fn(state)
        action = jnp.where(use_nominal, action_nominal, action_backup)

        # Sample noise online (autoregressive)
        feat = empirical_feature_fn(state, action, prev_action, step_idx)
        noise_step = empirical_sampler_fn(feat, gate_id, subkey, uncertainty_data)

        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_params)
        # Return full state for debugging/visualization.
        return (next_state, step_idx + 1, action), (h_val, noise_step, state)

    if nominal_trajectory is not None:
        initial_prev_action = nominal_trajectory[0]
    else:
        initial_prev_action = nominal_policy_fn(x0)

    keys = jax.random.split(rng_key, T)
    final_carry, (h_values, noise_traj, state_traj) = lax.scan(
        step_fn,
        (x0, 0, initial_prev_action),
        keys,
        length=T,
    )
    final_state = final_carry[0]
    state_traj = jnp.concatenate([state_traj, final_state[None, :]], axis=0)

    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), noise_traj, state_traj

def rollout_with_grad(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Uses rollout inputs and fixed noise to return the safety margin and flight path together
    with the noise gradient.
    """
    def h_only(w):
        """Uses a noise sequence and the surrounding rollout inputs to return the safety margin
        and flight path.
        """
        # rollout_single returns (h_min, xy_traj)
        return rollout_single(
            x0, switching_offset, w, env_params_traj,
            dynamics_fn, nominal_policy_fn, backup_policy_fn,
            safety_fn, pcis_fn, T,
            nominal_trajectory=nominal_trajectory
        )
    # Use has_aux=True because h_only returns (scalar, auxiliary_data)
    (val, aux), grad = jax.value_and_grad(h_only, has_aux=True)(noise_traj)
    return (val, aux), grad

def build_rollout_all_empirical(
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    empirical_feature_fn: Callable,
    empirical_sampler_fn: Callable,
) -> Callable:
    """Uses model functions, settings, and saved uncertainty to return a function that evaluates
    all candidate flights.
    """
    def rollout_all(
        x0: Array,
        switching_offsets: Array,
        rng_matrix: Array,     # [M, N, 2] PRNG keys
        env_params_batch: Array, # [M, N, theta_dim]
        compute_grad: bool = False,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[Array, Optional[Array], Array, Array, Array]:
        """Uses a starting state, switch times, random keys, and environment inputs to return
        safety margins, optional gradients, paths, and noise measures.
        """
        if not compute_grad:
            # Fast value-only path: one batched scan over time for all [M, N]
            # rollouts, which reduces per-rollout scan overhead.
            M, N = rng_matrix.shape[:2]
            switching_offsets_b = switching_offsets[:, None]
            env_params_static = env_params_batch[..., 0, :] if env_params_batch.ndim == 4 else env_params_batch

            batched_feature_fn = jax.vmap(
                jax.vmap(empirical_feature_fn, in_axes=(0, 0, 0, None)),
                in_axes=(0, 0, 0, None),
            )
            batched_sampler_fn = jax.vmap(
                jax.vmap(lambda feat, key: empirical_sampler_fn(feat, gate_id, key, uncertainty_data), in_axes=(0, 0)),
                in_axes=(0, 0),
            )
            batched_dynamics_fn = jax.vmap(
                jax.vmap(dynamics_fn, in_axes=(0, 0, 0)),
                in_axes=(0, 0, 0),
            )
            batched_safety_fn = jax.vmap(
                jax.vmap(safety_fn, in_axes=(0, 0)),
                in_axes=(0, 0),
            )
            batched_pcis_fn = jax.vmap(
                jax.vmap(pcis_fn, in_axes=0),
                in_axes=0,
            )

            state0 = jnp.broadcast_to(x0, (M, N, x0.shape[0]))
            if nominal_trajectory is not None:
                initial_prev_action = jnp.broadcast_to(
                    nominal_trajectory[0],
                    (M, N, nominal_trajectory.shape[-1]),
                )
            else:
                initial_action = nominal_policy_fn(x0)
                initial_prev_action = jnp.broadcast_to(
                    initial_action,
                    (M, N, initial_action.shape[0]),
                )

            def step_fn(carry, step_idx):
                """Uses batched states, previous actions, random keys, and the step number to
                return updated states, safety margins, and noise measures.
                """
                state, prev_action, key_batch = carry
                use_nominal = step_idx < switching_offsets_b

                if nominal_trajectory is not None:
                    action_nominal = jnp.broadcast_to(
                        nominal_trajectory[step_idx],
                        prev_action.shape,
                    )
                else:
                    action_nominal = jax.vmap(jax.vmap(nominal_policy_fn))(state)

                action_backup = jax.vmap(jax.vmap(backup_policy_fn))(state)
                action = jnp.where(use_nominal[..., None], action_nominal, action_backup)

                split_keys = jax.vmap(jax.vmap(jax.random.split))(key_batch)
                next_keys = split_keys[..., 0, :]
                subkeys = split_keys[..., 1, :]

                feat = batched_feature_fn(state, action, prev_action, step_idx)
                noise_step = batched_sampler_fn(feat, subkeys)
                next_state = batched_dynamics_fn(state, action, noise_step)
                h_val = batched_safety_fn(next_state, env_params_static)
                state_snapshot = state
                noise_l2 = jnp.linalg.norm(noise_step, axis=-1)
                noise_maxabs = jnp.max(jnp.abs(noise_step), axis=-1)
                return (next_state, action, next_keys), (h_val, state_snapshot, noise_l2, noise_maxabs)

            final_carry, (h_values, state_steps, noise_l2_steps, noise_maxabs_steps) = lax.scan(
                step_fn,
                (state0, initial_prev_action, rng_matrix),
                jnp.arange(T, dtype=jnp.int32),
                length=T,
            )
            final_state = final_carry[0]
            X_all = jnp.concatenate(
                [jnp.moveaxis(state_steps, 0, 2), final_state[:, :, None, :]],
                axis=2,
            )
            H_matrix = jnp.minimum(
                batched_pcis_fn(final_state),
                jnp.min(h_values, axis=0),
            )
            noise_l2_all = jnp.moveaxis(noise_l2_steps, 0, 2)
            noise_maxabs_all = jnp.moveaxis(noise_maxabs_steps, 0, 2)
            return H_matrix, None, X_all, noise_l2_all, noise_maxabs_all

        # Gradient-enabled path: we need realized sampled_noise_batch.
        def _single_sample(m, key, th):
            """Uses a switch time, random key, and environment input to return one sampled
            flight's safety margin, noise, and path.
            """
            return rollout_single_empirical(
                x0, m, key, th, uncertainty_data, gate_id,
                dynamics_fn, nominal_policy_fn, backup_policy_fn,
                safety_fn, pcis_fn, T,
                empirical_feature_fn, empirical_sampler_fn,
                nominal_trajectory=nominal_trajectory
            )

        v_sample = jax.vmap(_single_sample, in_axes=(None, 0, 0))
        all_MN_sample = jax.vmap(v_sample, in_axes=(0, 0, 0))

        # [M, N], [M, N, T, noise_dim], and [M, N, T+1, state_dim]
        H_matrix, sampled_noise_batch, X_all = all_MN_sample(switching_offsets, rng_matrix, env_params_batch)

        # 2. Gradient Pass (Conditional)
        # Use the realized noise as if it were fixed input for the differentiable core
        def _single_grad(m, w, th):
            """Uses a switch time, fixed noise, and environment input to return the safety margin
            and path together with the noise gradient.
            """
            if th.ndim == 1:
                th = jnp.broadcast_to(th, (T, th.shape[-1]))
            return rollout_with_grad(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        v_grad = jax.vmap(_single_grad, in_axes=(None, 0, 0))
        all_MN_grad = jax.vmap(v_grad, in_axes=(0, 0, 0))

        # We re-run to get derivatives w.r.t. the fixed noise sequence
        _, Grad_matrix = all_MN_grad(switching_offsets, sampled_noise_batch, env_params_batch)
        noise_l2_all = jnp.linalg.norm(sampled_noise_batch, axis=-1)
        noise_maxabs_all = jnp.max(jnp.abs(sampled_noise_batch), axis=-1)
        return H_matrix, Grad_matrix, X_all, noise_l2_all, noise_maxabs_all

    return rollout_all


def compute_q_bar(k: int, N: int, rho: float) -> float:
    """Uses failure and sample counts plus a confidence setting to return an upper bound on
    failure probability.
    """
    return 1.0 if k == N else float(betaincinv(k + 1, N - k, 1 - rho))


def select_switch(failure_counts, params, t, previous_switch):
    """Uses failure counts, settings, and switch times to choose a switch and return its time,
    feasibility flags, bounds, and index.
    """
    rho = 1 - (1 - params.delta) ** (1 / params.M)
    bounds = np.asarray([compute_q_bar(int(k), params.N, rho) for k in failure_counts], dtype=np.float32)
    feasible = bounds <= (params.epsilon - params.alpha) / (1 - params.alpha)
    index = int(np.flatnonzero(feasible)[-1]) if feasible.any() else 0
    switch = t + index if feasible.any() else previous_switch
    return switch, feasible, bounds, index


class DRSGatekeeper:
    """F-16 Gatekeeper with empirical sampling and a retained switching plan."""

    def __init__(self, params, dynamics_fn, nominal_policy_fn, backup_policy_fn,
                 safety_fn, pcis_fn, uncertainty_model, empirical_feature_fn, seed=0):
        """Uses settings, model functions, saved uncertainty, and a seed to initialize the
        gatekeeper, with no return value.
        """
        self.params = params
        self.nominal_policy_fn = nominal_policy_fn
        self.backup_policy_fn = backup_policy_fn
        self.uncertainty_data = uncertainty_model.to_jax()
        self._seed = int(seed)
        self._key = jax.random.PRNGKey(self._seed)
        self._offsets = jnp.arange(params.M, dtype=jnp.int32)
        self._environment = jnp.zeros((params.M, params.N, 0), dtype=jnp.float32)
        self._rollout = jax.jit(build_rollout_all_empirical(
            dynamics_fn, nominal_policy_fn, backup_policy_fn, safety_fn, pcis_fn,
            params.T, self.uncertainty_data, 0, empirical_feature_fn, sample_empirical_jax,
        ), static_argnums=4)
        self.state = None

    def reset(self, x0_flat, t=0):
        """Accepts an initial state and time, ignores the state, resets the random key and
        switching state, and returns nothing.
        """
        del x0_flat
        self._key = jax.random.PRNGKey(self._seed)
        self.state = GatekeeperState(t=t, s_prev=t, using_backup=False, plan_start_t=t)

    def update(self, x_flat, nominal_trajectory=None):
        """Uses the current aircraft state and optional action sequence to evaluate switch times
        and return the updated gatekeeper state.
        """
        if self.state is None:
            raise RuntimeError("Reset the Gatekeeper before updating it.")
        params = self.params
        self._key, key = jax.random.split(self._key)
        keys = jax.random.split(key, params.M * params.N).reshape(params.M, params.N, 2)
        values, gradients, trajectories, noise_l2, noise_maxabs = self._rollout(
            x_flat, self._offsets, keys, self._environment,
            params.lipschitz_mode == "autodiff", nominal_trajectory,
        )
        lipschitz = params.lipschitz_constant
        if gradients is not None:
            lipschitz = float(jnp.max(jnp.linalg.norm(
                gradients.reshape(params.M, params.N, -1), axis=-1)))
        failures = values <= lipschitz * params.beta ** (1 / params.p)
        counts = np.asarray(jnp.sum(failures, axis=1))
        switch, feasible, bounds, index = select_switch(
            counts, params, self.state.t, self.state.s_prev)
        using_backup = self.state.t >= switch
        new_commitment = bool(feasible.any() and switch != self.state.s_prev)
        if new_commitment or self.state.predicted_trajectories is None:
            predicted = np.asarray(trajectories[index], dtype=np.float32)
            mask = np.asarray(failures[index], dtype=bool)
            nl2 = np.asarray(noise_l2[index], dtype=np.float32)
            nmax = np.asarray(noise_maxabs[index], dtype=np.float32)
            bound = float(bounds[index])
            plan_start = self.state.t
        else:
            predicted = self.state.predicted_trajectories
            mask = self.state.failure_mask
            nl2, nmax = self.state.noise_l2, self.state.noise_maxabs
            bound, plan_start = self.state.q_bar_star, self.state.plan_start_t
        self.state = GatekeeperState(
            t=self.state.t, s_prev=switch, using_backup=using_backup,
            predicted_trajectories=predicted, failure_mask=mask,
            noise_l2=nl2, noise_maxabs=nmax,
            m_star=index if new_commitment else switch - self.state.t,
            q_bar_star=bound, is_reverting=not bool(feasible.any()),
            plan_start_t=plan_start,
        )
        return self.state

    def tick(self):
        """Uses the stored state to advance the clock and update backup mode, with no return
        value.
        """
        if self.state is None:
            raise RuntimeError("Reset the Gatekeeper before advancing it.")
        t = self.state.t + 1
        self.state = self.state._replace(t=t, using_backup=t >= self.state.s_prev)
