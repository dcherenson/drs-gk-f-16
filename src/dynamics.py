"""Learned aerodynamic coefficients and rigid-body Gatekeeper dynamics."""
import jax.numpy as jnp
import numpy as np
from itertools import combinations_with_replacement

from pathlib import Path


COEFFICIENT_FEATURE_NAMES = (
    "alpha",
    "beta",
    "mach",
    "p",
    "q",
    "r",
    "delta_e",
    "delta_a",
    "delta_r",
)

COEFFICIENT_TARGET_NAMES = (
    "C_X",
    "C_Y",
    "C_Z",
    "C_L",
    "C_M",
    "C_N",
)

IXX = 9496.0
IYY = 55814.0
IZZ = 63100.0
IXZ = -982.0
INERTIA_DET = IXX * IZZ - IXZ * IXZ

WING_AREA_FT2 = 300.0
WING_SPAN_FT = 30.0
MEAN_AERODYNAMIC_CHORD_FT = 11.32
DEFAULT_MASS_LBS = 17400.0 + 230.0 + 2000.0
DEFAULT_MASS_SLUGS = DEFAULT_MASS_LBS / 32.174
STANDARD_SPEED_OF_SOUND_FPS = 1116.45
AIR_DENSITY_SLUG_FT3 = 0.0023769
MIN_QBAR_PSF = 1.0
DT = 1.0 / 30.0
G_FTPS2 = 32.174

# Kinematics / clipping constants (extracted from inline literals)
ALPHA_MIN_RAD = -0.35
ALPHA_MAX_RAD = 0.52
U_MIN_FOR_ALPHA = 1.0
V_SQ_MIN = 1.0
BETA_MIN_RAD = -0.35
BETA_MAX_RAD = 0.35
MACH_MIN = 0.2
MACH_MAX = 1.2
THETA_FOR_KIN_CLIP = 1.3962634
SAFE_COS_THETA_MIN = 0.1

UDOT_MIN = -250.0
UDOT_MAX = 250.0
VDOT_MIN = -250.0
VDOT_MAX = 250.0
WDOT_MIN = -250.0
WDOT_MAX = 250.0

P_DOT_MIN = -15.0
P_DOT_MAX = 15.0
Q_DOT_MIN = -10.0
Q_DOT_MAX = 10.0
R_DOT_MIN = -15.0
R_DOT_MAX = 15.0

PHI_DOT_LIMIT = 10.0
THETA_DOT_LIMIT = 10.0
PSI_DOT_LIMIT = 10.0

PN_DOT_LIMIT = 1200.0
PE_DOT_LIMIT = 1200.0
H_DOT_LIMIT = 400.0

P_CLIP_LIMIT = 6.0
Q_CLIP_LIMIT = 4.0
R_CLIP_LIMIT = 6.0

NY_MIN = -4.0
NY_MAX = 4.0
NZ_MIN = -3.0
NZ_MAX = 9.0


def _build_polynomial_powers(n_features, degree, include_bias):
    powers = []
    if include_bias:
        powers.append(np.zeros(n_features, dtype=np.int32))
    for deg in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(n_features), deg):
            row = np.zeros(n_features, dtype=np.int32)
            for idx in combo:
                row[idx] += 1
            powers.append(row)
    return np.asarray(powers, dtype=np.int32)


def load_nominal_weights():
    path = Path(__file__).parent / "assets" / "models" / "nominal_coeff_weights.npz"
    with np.load(path, allow_pickle=False) as data:
        W = np.asarray(data["W"], dtype=np.float32)
        B = np.asarray(data["B"], dtype=np.float32)

        if "feature_names" not in data:
            raise ValueError("nominal coefficient weights are missing feature metadata.")
        feature_names = tuple(str(x) for x in np.asarray(data["feature_names"]).reshape(-1))
        if feature_names != COEFFICIENT_FEATURE_NAMES:
            raise ValueError(
                "nominal coefficient weights have an incompatible feature set: "
                f"expected {COEFFICIENT_FEATURE_NAMES}, got {feature_names}."
            )

        if "target_names" not in data:
            raise ValueError("nominal coefficient weights are missing target metadata.")
        target_names = tuple(str(x) for x in np.asarray(data["target_names"]).reshape(-1))
        if target_names != COEFFICIENT_TARGET_NAMES:
            raise ValueError(
                "nominal coefficient weights have an incompatible target set: "
                f"expected {COEFFICIENT_TARGET_NAMES}, got {target_names}."
            )

        if "model_space" not in data:
            raise ValueError("nominal coefficient weights are missing model-space metadata.")
        model_space_values = np.asarray(data["model_space"]).reshape(-1)
        model_space = str(model_space_values[0]) if model_space_values.size else ""
        if model_space != "aerodynamic_coefficients":
            raise ValueError(
                "nominal coefficient weights have an incompatible model space: "
                f"expected 'aerodynamic_coefficients', got {model_space!r}."
            )

        if "poly_degree" not in data:
            raise ValueError("nominal coefficient weights are missing polynomial-degree metadata.")
        poly_degree = int(np.asarray(data["poly_degree"]).reshape(-1)[0])
        include_bias = bool(int(np.asarray(data["include_bias"]).reshape(-1)[0])) if "include_bias" in data else True
        poly_powers = _build_polynomial_powers(len(COEFFICIENT_FEATURE_NAMES), poly_degree, include_bias)
        expected_rows = int(poly_powers.shape[0])

        if W.ndim != 2 or W.shape[0] != expected_rows or W.shape[1] != len(COEFFICIENT_TARGET_NAMES):
            raise ValueError(
                "nominal coefficient weights have an incompatible polynomial shape: "
                f"expected ({expected_rows}, {len(COEFFICIENT_TARGET_NAMES)}), got {W.shape}."
            )
        if B.shape != (len(COEFFICIENT_TARGET_NAMES),):
            raise ValueError(
                "nominal coefficient weights have an incompatible bias shape: "
                f"expected ({len(COEFFICIENT_TARGET_NAMES)},), got {B.shape}."
            )
        if "throttle_force_coeffs" not in data:
            raise ValueError("nominal coefficient weights are missing throttle-force coefficients.")
        throttle_force_coeffs = np.asarray(data["throttle_force_coeffs"], dtype=np.float32).reshape(-1)
        if throttle_force_coeffs.size < 1:
            raise ValueError("nominal coefficient weights contain no throttle-force coefficients.")

    return (
        jnp.asarray(W),
        jnp.asarray(B),
        jnp.asarray(poly_powers, dtype=jnp.int32),
        jnp.asarray(throttle_force_coeffs, dtype=jnp.float32),
    )


def expand_poly(x, poly_powers):
    return jnp.prod(jnp.power(x[None, :], poly_powers), axis=1)


def throttle_force_term(delta_t, throttle_force_coeffs):
    coeffs = jnp.asarray(throttle_force_coeffs, dtype=jnp.float32).reshape(-1)
    term = jnp.asarray(0.0, dtype=jnp.float32)
    for power in range(coeffs.shape[0]):
        term = term + coeffs[power] * jnp.power(delta_t, power)
    return term


def moments_to_angular_rate_derivatives(p, q, r, L, M, N):
    h_x = IXX * p + IXZ * r
    h_y = IYY * q
    h_z = IXZ * p + IZZ * r

    cross_x = q * h_z - r * h_y
    cross_y = r * h_x - p * h_z
    cross_z = p * h_y - q * h_x

    rhs_x = L - cross_x
    rhs_y = M - cross_y
    rhs_z = N - cross_z

    p_dot = (IZZ * rhs_x - IXZ * rhs_z) / INERTIA_DET
    q_dot = rhs_y / IYY
    r_dot = (IXX * rhs_z - IXZ * rhs_x) / INERTIA_DET
    return p_dot, q_dot, r_dot


def f16_kinematics_step_with_load_factors(state, action, W, B, poly_powers, throttle_force_coeffs):
    state_core = state[:12]
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi = state_core

    delta_a = action[0]
    delta_e = action[1]
    delta_r = action[2]
    delta_t = action[3]

    alpha = jnp.clip(jnp.arctan2(w, jnp.maximum(u, U_MIN_FOR_ALPHA)), ALPHA_MIN_RAD, ALPHA_MAX_RAD)
    V_sq = u * u + v * v + w * w
    V = jnp.sqrt(jnp.maximum(V_sq, V_SQ_MIN))
    beta = jnp.clip(jnp.arcsin(jnp.clip(v / V, -1.0, 1.0)), BETA_MIN_RAD, BETA_MAX_RAD)
    mach = jnp.clip(V / STANDARD_SPEED_OF_SOUND_FPS, MACH_MIN, MACH_MAX)

    features = jnp.array([alpha, beta, mach, p, q, r, delta_e, delta_a, delta_r], dtype=jnp.float32)
    phi_vec = expand_poly(features, poly_powers)
    preds = jnp.dot(phi_vec, W) + B
    C_X, C_Y, C_Z, C_L, C_M, C_N = preds

    qbar_psf = jnp.maximum(0.5 * AIR_DENSITY_SLUG_FT3 * V_sq, MIN_QBAR_PSF)
    force_scale = qbar_psf * WING_AREA_FT2 / DEFAULT_MASS_SLUGS
    roll_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT
    pitch_moment_scale = qbar_psf * WING_AREA_FT2 * MEAN_AERODYNAMIC_CHORD_FT
    yaw_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT

    X = C_X * force_scale + throttle_force_term(delta_t, throttle_force_coeffs)
    Y = C_Y * force_scale
    Z = C_Z * force_scale
    L = C_L * roll_moment_scale
    M = C_M * pitch_moment_scale
    N = C_N * yaw_moment_scale

    theta_for_kin = jnp.clip(theta, -THETA_FOR_KIN_CLIP, THETA_FOR_KIN_CLIP)
    c_theta_safe = jnp.cos(theta_for_kin)
    c_theta_safe = jnp.sign(c_theta_safe) * jnp.maximum(jnp.abs(c_theta_safe), SAFE_COS_THETA_MIN)
    s_theta_safe = jnp.sin(theta_for_kin)

    u_dot = X + r * v - q * w - G_FTPS2 * s_theta_safe
    v_dot = Y + p * w - r * u + G_FTPS2 * jnp.sin(phi) * c_theta_safe
    w_dot = Z + q * u - p * v + G_FTPS2 * jnp.cos(phi) * c_theta_safe

    p_dot, q_dot, r_dot = moments_to_angular_rate_derivatives(p, q, r, L, M, N)

    u_dot = jnp.clip(u_dot, UDOT_MIN, UDOT_MAX)
    v_dot = jnp.clip(v_dot, VDOT_MIN, VDOT_MAX)
    w_dot = jnp.clip(w_dot, WDOT_MIN, WDOT_MAX)
    p_dot = jnp.clip(p_dot, P_DOT_MIN, P_DOT_MAX)
    q_dot = jnp.clip(q_dot, Q_DOT_MIN, Q_DOT_MAX)
    r_dot = jnp.clip(r_dot, R_DOT_MIN, R_DOT_MAX)

    t_theta = s_theta_safe / c_theta_safe
    phi_dot = p + q * jnp.sin(phi) * t_theta + r * jnp.cos(phi) * t_theta
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = (q * jnp.sin(phi) + r * jnp.cos(phi)) / c_theta_safe

    phi_dot = jnp.clip(phi_dot, -PHI_DOT_LIMIT, PHI_DOT_LIMIT)
    theta_dot = jnp.clip(theta_dot, -THETA_DOT_LIMIT, THETA_DOT_LIMIT)
    psi_dot = jnp.clip(psi_dot, -PSI_DOT_LIMIT, PSI_DOT_LIMIT)

    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    c_theta = c_theta_safe
    s_theta = s_theta_safe
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)

    p_N_dot = (
        u * (c_theta * c_psi)
        + v * (s_phi * s_theta * c_psi - c_phi * s_psi)
        + w * (c_phi * s_theta * c_psi + s_phi * s_psi)
    )
    p_E_dot = (
        u * (c_theta * s_psi)
        + v * (s_phi * s_theta * s_psi + c_phi * c_psi)
        + w * (c_phi * s_theta * s_psi - s_phi * c_psi)
    )
    h_dot = u * s_theta - v * (s_phi * c_theta) - w * (c_phi * c_theta)

    p_N_dot = jnp.clip(p_N_dot, -PN_DOT_LIMIT, PN_DOT_LIMIT)
    p_E_dot = jnp.clip(p_E_dot, -PE_DOT_LIMIT, PE_DOT_LIMIT)
    h_dot = jnp.clip(h_dot, -H_DOT_LIMIT, H_DOT_LIMIT)

    state_next_core = jnp.array(
        [
            p_N + DT * p_N_dot,
            p_E + DT * p_E_dot,
            h + DT * h_dot,
            u + DT * u_dot,
            v + DT * v_dot,
            w + DT * w_dot,
            jnp.clip(p + DT * p_dot, -P_CLIP_LIMIT, P_CLIP_LIMIT),
            jnp.clip(q + DT * q_dot, -Q_CLIP_LIMIT, Q_CLIP_LIMIT),
            jnp.clip(r + DT * r_dot, -R_CLIP_LIMIT, R_CLIP_LIMIT),
            phi + DT * phi_dot,
            theta + DT * theta_dot,
            psi + DT * psi_dot,
        ],
        dtype=jnp.float32,
    )

    ny = jnp.clip(Y / G_FTPS2, NY_MIN, NY_MAX)
    nz = jnp.clip(-Z / G_FTPS2, NZ_MIN, NZ_MAX)
    return jnp.concatenate([state_next_core, jnp.asarray([ny, nz], dtype=jnp.float32)], axis=0)


def _terrain_elevation_ft_at(p_n_ft, p_e_ft, north_axis_ft, east_axis_ft, terrain_elevation_ft):
    n_hi = jnp.clip(jnp.searchsorted(north_axis_ft, p_n_ft, side="right"), 1, north_axis_ft.shape[0] - 1)
    e_hi = jnp.clip(jnp.searchsorted(east_axis_ft, p_e_ft, side="right"), 1, east_axis_ft.shape[0] - 1)
    n_lo = n_hi - 1
    e_lo = e_hi - 1

    n0 = north_axis_ft[n_lo]
    n1 = north_axis_ft[n_hi]
    e0 = east_axis_ft[e_lo]
    e1 = east_axis_ft[e_hi]

    dn = jnp.where(jnp.abs(n1 - n0) > 1e-6, (p_n_ft - n0) / (n1 - n0), 0.0)
    de = jnp.where(jnp.abs(e1 - e0) > 1e-6, (p_e_ft - e0) / (e1 - e0), 0.0)
    dn = jnp.clip(dn, 0.0, 1.0)
    de = jnp.clip(de, 0.0, 1.0)

    z00 = terrain_elevation_ft[n_lo, e_lo]
    z01 = terrain_elevation_ft[n_lo, e_hi]
    z10 = terrain_elevation_ft[n_hi, e_lo]
    z11 = terrain_elevation_ft[n_hi, e_hi]

    z0 = (1.0 - de) * z00 + de * z01
    z1 = (1.0 - de) * z10 + de * z11
    return (1.0 - dn) * z0 + dn * z1
