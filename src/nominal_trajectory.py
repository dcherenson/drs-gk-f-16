"""Portable nominal F-16 trajectory loading and canyon interpolation.

The source trajectory is converted once into ``assets/models/nominal_trajectory.npz``.
The public runtime reads these numeric arrays without the trajectory optimizer
or its serialization dependencies.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


M_TO_FT = 3.28084
METERS_PER_DEG_LAT = 1e6 / 9.0
FPS_PER_KT = 1.68781
_FORMAT_VERSION = "drs-f16-nominal-trajectory-v1"
_DEFAULT_NOMINAL_PATH = Path(__file__).resolve().parent / "assets" / "models" / "nominal_trajectory.npz"
_TRAJECTORY_FIELDS = (
    "x_e_m",
    "y_e_m",
    "z_e_m",
    "speed_mps",
    "gamma_rad",
    "track_rad",
    "alpha_deg",
    "beta_deg",
    "bank_rad",
    "time_s",
    "roll_deg",
    "pitch_deg",
    "heading_deg",
)


def _progress_fraction_to_sample_index(progress_fraction: float, sample_count: int) -> int:
    """Map a [0, 1] progress fraction to a discrete trajectory sample index."""

    if int(sample_count) < 1:
        raise ValueError("sample_count must be >= 1.")

    progress = float(progress_fraction)
    if not np.isfinite(progress):
        raise ValueError("progress_fraction must be finite.")
    if progress < 0.0 or progress > 1.0:
        raise ValueError("progress_fraction must be within [0.0, 1.0].")
    if int(sample_count) == 1:
        return 0

    max_index = int(sample_count) - 1
    return int(np.clip(np.rint(progress * float(max_index)), 0, max_index))


def _read_nominal_array(data: Any, key: str, sample_count: int | None = None) -> np.ndarray:
    if key not in data:
        raise ValueError(f"nominal trajectory artifact is missing required field {key!r}")
    array = np.asarray(data[key])
    expected_shape = (sample_count,) if sample_count is not None else None
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"nominal trajectory field {key!r} has shape {array.shape}; expected {expected_shape}")
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"nominal trajectory field {key!r} must be a finite numeric vector")
    return np.array(array, dtype=np.float64, copy=True)


def _load_portable_nominal_trajectory(path: str | Path | None) -> SimpleNamespace:
    """Load the portable NPZ and expose a small source-compatible namespace."""

    trajectory_path = _DEFAULT_NOMINAL_PATH if path is None else Path(path).expanduser()
    if trajectory_path.suffix.lower() != ".npz":
        raise ValueError(
            "portable nominal trajectory must be an .npz file; "
            f"got {trajectory_path.name!r}"
        )
    try:
        loaded = np.load(trajectory_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing portable nominal trajectory: {trajectory_path}") from exc
    with loaded as data:
        if "format_version" not in data:
            raise ValueError("nominal trajectory artifact is missing format_version")
        version_array = np.asarray(data["format_version"])
        if version_array.shape != (1,) or version_array.dtype.kind not in "SU":
            raise ValueError("nominal trajectory format_version must be a one-element string array")
        if str(version_array[0]) != _FORMAT_VERSION:
            raise ValueError(f"unsupported nominal trajectory format: {version_array[0]!r}")
        if "sample_count" not in data:
            raise ValueError("nominal trajectory artifact is missing sample_count")
        count_array = np.asarray(data["sample_count"])
        if count_array.shape != (1,) or not np.issubdtype(count_array.dtype, np.integer):
            raise ValueError("nominal trajectory sample_count must be a one-element integer array")
        sample_count = int(count_array[0])
        if sample_count < 2:
            raise ValueError("nominal trajectory must contain at least two samples")
        arrays = {
            key: _read_nominal_array(data, key, sample_count)
            for key in _TRAJECTORY_FIELDS
        }

    return SimpleNamespace(
        x_e=arrays["x_e_m"],
        y_e=arrays["y_e_m"],
        z_e=arrays["z_e_m"],
        speed=arrays["speed_mps"],
        gamma=arrays["gamma_rad"],
        track=arrays["track_rad"],
        alpha=arrays["alpha_deg"],
        beta=arrays["beta_deg"],
        bank=arrays["bank_rad"],
        roll_deg=arrays["roll_deg"],
        pitch_deg=arrays["pitch_deg"],
        heading_deg=arrays["heading_deg"],
        other_fields={"time": arrays["time_s"]},
        source_path=trajectory_path,
    )


def _north_east_m_to_lat_lon(
    north_m: np.ndarray,
    east_m: np.ndarray,
    *,
    south_deg: float,
    north_deg: float,
    west_deg: float,
    east_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    datum_lat = 0.5 * (float(south_deg) + float(north_deg))
    datum_lon = 0.5 * (float(west_deg) + float(east_deg))
    cos_lat = float(np.cos(np.deg2rad(datum_lat)))

    lat_deg = np.asarray(north_m, dtype=np.float64) / METERS_PER_DEG_LAT + datum_lat
    lon_deg = np.asarray(east_m, dtype=np.float64) / (METERS_PER_DEG_LAT * cos_lat) + datum_lon
    return lat_deg, lon_deg




def _ordered_display_samples(
    north_ft: np.ndarray,
    east_ft: np.ndarray,
    altitude_ft: np.ndarray,
    *,
    spacing_ft: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    north = np.asarray(north_ft, dtype=np.float64).reshape(-1)
    east = np.asarray(east_ft, dtype=np.float64).reshape(-1)
    altitude = np.asarray(altitude_ft, dtype=np.float64).reshape(-1)

    finite_mask = np.isfinite(north) & np.isfinite(east) & np.isfinite(altitude)
    north = north[finite_mask]
    east = east[finite_mask]
    altitude = altitude[finite_mask]
    if north.size < 2:
        raise ValueError("Need at least two finite samples to build a display trajectory.")

    segment_lengths = np.hypot(np.diff(north), np.diff(east))
    keep_mask = np.concatenate([[True], segment_lengths > 1e-6])
    north = north[keep_mask]
    east = east[keep_mask]
    altitude = altitude[keep_mask]
    if north.size < 2:
        raise ValueError("Display trajectory collapsed after removing duplicate samples.")

    arc_length_ft = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(north), np.diff(east)))])
    if float(arc_length_ft[-1]) <= 1e-6:
        raise ValueError("Display trajectory has zero arc length.")

    spacing = float(max(spacing_ft, 1.0))
    sample_s = np.arange(0.0, float(arc_length_ft[-1]) + 0.5 * spacing, spacing)
    if sample_s.size < 2:
        sample_s = np.asarray([0.0, float(arc_length_ft[-1])], dtype=np.float64)

    display_north = np.interp(sample_s, arc_length_ft, north)
    display_east = np.interp(sample_s, arc_length_ft, east)
    display_altitude = np.interp(sample_s, arc_length_ft, altitude)
    return display_north, display_east, display_altitude


def _wrap_heading_deg(angle_deg: float) -> float:
    return float(np.mod(float(angle_deg), 360.0))


def _body_euler_deg_series_from_dyn(dyn: SimpleNamespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the precomputed JSBSim-convention body Euler angles."""

    try:
        roll = np.asarray(dyn.roll_deg, dtype=np.float64).reshape(-1)
        pitch = np.asarray(dyn.pitch_deg, dtype=np.float64).reshape(-1)
        heading = np.asarray(dyn.heading_deg, dtype=np.float64).reshape(-1)
    except AttributeError as exc:
        raise ValueError("portable nominal trajectory is missing precomputed body attitudes") from exc
    if not (roll.shape == pitch.shape == heading.shape) or not np.all(
        np.isfinite(np.concatenate([roll, pitch, heading]))
    ):
        raise ValueError("precomputed body attitude arrays must be finite and have equal lengths")
    return roll, pitch, heading


def _validate_canyon(canyon: Any) -> None:
    required = ("south", "north", "west", "east", "get_local_from_latlon")
    if not all(hasattr(canyon, attr) for attr in required):
        raise TypeError("canyon must provide DEM bounds and get_local_from_latlon().")


def load_nominal_initial_conditions(
    dyn_path: str | Path,
    *,
    canyon: Any,
    progress_fraction: float = 0.0,
) -> dict[str, float | int | tuple[int, int]]:
    """Extract JSBSim-friendly initial conditions from a portable NPZ."""

    dyn = _load_portable_nominal_trajectory(dyn_path)
    _validate_canyon(canyon)

    north_m_series = np.asarray(dyn.x_e, dtype=np.float64).reshape(-1)
    east_m_series = np.asarray(dyn.y_e, dtype=np.float64).reshape(-1)
    altitude_msl_ft_series = -np.asarray(dyn.z_e, dtype=np.float64).reshape(-1) * M_TO_FT
    speed_fps_series = np.asarray(dyn.speed, dtype=np.float64).reshape(-1) * M_TO_FT
    alpha_deg_series = np.asarray(dyn.alpha, dtype=np.float64).reshape(-1)
    beta_deg_series = np.asarray(dyn.beta, dtype=np.float64).reshape(-1)
    gamma_deg_series = np.degrees(np.asarray(dyn.gamma, dtype=np.float64).reshape(-1))
    track_deg_series = np.mod(np.degrees(np.asarray(dyn.track, dtype=np.float64).reshape(-1)), 360.0)
    bank_deg_series = np.degrees(np.asarray(dyn.bank, dtype=np.float64).reshape(-1))

    sample_count = int(north_m_series.size)
    if sample_count < 1:
        raise ValueError("Offline nominal trajectory must contain at least one sample.")
    _ = _progress_fraction_to_sample_index(progress_fraction, sample_count)
    progress_fraction = float(progress_fraction)

    for arr in (
        east_m_series,
        altitude_msl_ft_series,
        speed_fps_series,
        alpha_deg_series,
        beta_deg_series,
        gamma_deg_series,
        track_deg_series,
        bank_deg_series,
    ):
        if arr.shape != north_m_series.shape:
            raise ValueError("Offline nominal trajectory fields must have consistent sample counts.")

    path_step_m = np.hypot(np.diff(north_m_series), np.diff(east_m_series))
    path_step_m = np.where(np.isfinite(path_step_m), np.maximum(path_step_m, 0.0), 0.0)
    path_s_m = np.concatenate([[0.0], np.cumsum(path_step_m, dtype=np.float64)])
    if float(path_s_m[-1]) > 1e-9:
        target_s_m = float(np.clip(progress_fraction, 0.0, 1.0)) * float(path_s_m[-1])
        sample_index = int(np.clip(np.searchsorted(path_s_m, target_s_m, side="left"), 0, sample_count - 1))
    else:
        sample_index = _progress_fraction_to_sample_index(progress_fraction, sample_count)
    north_m = float(north_m_series[sample_index])
    east_m = float(east_m_series[sample_index])
    altitude_msl_ft = float(altitude_msl_ft_series[sample_index])
    speed_fps = float(speed_fps_series[sample_index])
    alpha_deg = float(alpha_deg_series[sample_index])
    beta_deg = float(beta_deg_series[sample_index])
    gamma_deg = float(gamma_deg_series[sample_index])
    track_deg = _wrap_heading_deg(track_deg_series[sample_index])
    bank_deg = float(bank_deg_series[sample_index])

    lat_deg_arr, lon_deg_arr = _north_east_m_to_lat_lon(
        np.asarray([north_m], dtype=np.float64),
        np.asarray([east_m], dtype=np.float64),
        south_deg=float(canyon.south),
        north_deg=float(canyon.north),
        west_deg=float(canyon.west),
        east_deg=float(canyon.east),
    )
    lat_deg = float(lat_deg_arr[0])
    lon_deg = float(lon_deg_arr[0])

    local_north_ft, local_east_ft = canyon.get_local_from_latlon(lat_deg, lon_deg)

    if hasattr(canyon, "_latlon_to_ordered_row_col"):
        row_ordered, col_float = canyon._latlon_to_ordered_row_col(lat_deg, lon_deg)
        if getattr(canyon, "fly_direction", "south_to_north") == "south_to_north":
            row_original = float(canyon.rows - 1) - float(row_ordered)
        else:
            row_original = float(row_ordered)
        pixel_x = int(np.clip(int(np.rint(col_float)), 0, int(canyon.cols) - 1))
        pixel_y = int(np.clip(int(np.rint(row_original)), 0, int(canyon.rows) - 1))
    else:
        pixel_x = 0
        pixel_y = 0

    if hasattr(canyon, "get_elevation_msl_ft_from_latlon"):
        terrain_msl_ft = float(canyon.get_elevation_msl_ft_from_latlon(lat_deg, lon_deg))
    else:
        terrain_msl_ft = float(canyon.get_pixel_info(pixel_x, pixel_y)["elevation_msl_ft"])

    roll_deg_series, pitch_deg_series, heading_deg_series = _body_euler_deg_series_from_dyn(dyn)
    if roll_deg_series.shape != north_m_series.shape:
        raise ValueError("Recovered body Euler angle series length does not match trajectory samples.")
    roll_deg = float(roll_deg_series[sample_index])
    pitch_deg = float(pitch_deg_series[sample_index])
    heading_deg = float(heading_deg_series[sample_index])

    return {
        "start_pixel": (pixel_x, pixel_y),
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "local_north_ft": float(local_north_ft),
        "local_east_ft": float(local_east_ft),
        "altitude_msl_ft": altitude_msl_ft,
        "terrain_msl_ft": terrain_msl_ft,
        "entry_altitude_ft": float(altitude_msl_ft - terrain_msl_ft),
        "speed_fps": speed_fps,
        "speed_kts": float(speed_fps / FPS_PER_KT),
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "heading_deg": heading_deg,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "gamma_deg": gamma_deg,
        "track_deg": track_deg,
        "bank_deg": bank_deg,
        "sample_index": int(sample_index),
        "sample_count": int(sample_count),
        "progress_fraction": float(progress_fraction),
        "source_dyn_path": str(dyn.source_path),
    }


def build_nominal_reference(
    dyn_path: str | Path,
    *,
    canyon: Any,
    altitude_ref_ft: float = 0.0,
    resample_spacing_ft: float = 12.0,
    end_fraction: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert a portable nominal trajectory into DEM-local flight references.

    ``end_fraction`` is applied to the raw trajectory time axis before the
    30-Hz interpolation, matching the source conversion.
    """

    dyn = _load_portable_nominal_trajectory(dyn_path)
    _validate_canyon(canyon)

    north_m = np.asarray(dyn.x_e, dtype=np.float64).reshape(-1)
    east_m = np.asarray(dyn.y_e, dtype=np.float64).reshape(-1)
    altitude_msl_ft = -np.asarray(dyn.z_e, dtype=np.float64).reshape(-1) * M_TO_FT
    speed_fps = np.asarray(dyn.speed, dtype=np.float64).reshape(-1) * M_TO_FT
    time_s = np.asarray(dyn.other_fields.get("time", np.arange(north_m.size, dtype=np.float64) / 30.0), dtype=np.float64)
    if time_s.shape != north_m.shape:
        raise ValueError("Offline nominal trajectory time samples must match the position sample count.")

    roll_deg, pitch_deg, heading_deg = _body_euler_deg_series_from_dyn(dyn)
    roll_rad = np.unwrap(np.deg2rad(roll_deg))
    pitch_rad = np.unwrap(np.deg2rad(pitch_deg))
    heading_rad = np.unwrap(np.deg2rad(heading_deg))
    alpha_deg = np.asarray(dyn.alpha, dtype=np.float64).reshape(-1)
    beta_deg = np.asarray(dyn.beta, dtype=np.float64).reshape(-1)
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)

    lat_deg, lon_deg = _north_east_m_to_lat_lon(
        north_m,
        east_m,
        south_deg=float(canyon.south),
        north_deg=float(canyon.north),
        west_deg=float(canyon.west),
        east_deg=float(canyon.east),
    )

    local_samples = np.asarray(
        [canyon.get_local_from_latlon(float(lat), float(lon)) for lat, lon in zip(lat_deg, lon_deg)],
        dtype=np.float64,
    )
    if local_samples.ndim != 2 or local_samples.shape[1] != 2:
        raise ValueError("canyon.get_local_from_latlon() must return north/east pairs")
    local_north_ft = local_samples[:, 0]
    local_east_ft = local_samples[:, 1]
    altitude_rel_ft = altitude_msl_ft - float(altitude_ref_ft)

    end_fraction = float(np.clip(end_fraction, 0.0, 1.0))
    if end_fraction < 1.0:
        t_start = float(time_s[0])
        t_end = float(time_s[-1])
        t_cutoff = t_start + end_fraction * (t_end - t_start)
        mask = time_s <= t_cutoff
        if int(np.count_nonzero(mask)) < 2:
            mask[:2] = True
        north_m = north_m[mask]
        east_m = east_m[mask]
        altitude_msl_ft = altitude_msl_ft[mask]
        speed_fps = speed_fps[mask]
        time_s = time_s[mask]
        roll_rad = roll_rad[mask]
        pitch_rad = pitch_rad[mask]
        heading_rad = heading_rad[mask]
        alpha_rad = alpha_rad[mask]
        beta_rad = beta_rad[mask]
        lat_deg = lat_deg[mask]
        lon_deg = lon_deg[mask]
        local_north_ft = local_north_ft[mask]
        local_east_ft = local_east_ft[mask]
        altitude_rel_ft = altitude_rel_ft[mask]

    display_north_ft, display_east_ft, display_altitude_ft = _ordered_display_samples(
        local_north_ft,
        local_east_ft,
        altitude_rel_ft,
        spacing_ft=resample_spacing_ft,
    )

    reference_dt_s = 1.0 / 30.0
    sample_time_s = np.arange(float(time_s[0]), float(time_s[-1]) + 0.5 * reference_dt_s, reference_dt_s)
    if sample_time_s.size < 2:
        sample_time_s = np.asarray([float(time_s[0]), float(time_s[-1])], dtype=np.float64)

    sample_north_ft = np.interp(sample_time_s, time_s, local_north_ft)
    sample_east_ft = np.interp(sample_time_s, time_s, local_east_ft)
    sample_altitude_ft = np.interp(sample_time_s, time_s, altitude_rel_ft)
    sample_speed_fps = np.interp(sample_time_s, time_s, speed_fps)
    sample_phi_rad = np.interp(sample_time_s, time_s, roll_rad)
    sample_theta_rad = np.interp(sample_time_s, time_s, pitch_rad)
    sample_psi_rad = np.interp(sample_time_s, time_s, heading_rad)
    sample_alpha_rad = np.interp(sample_time_s, time_s, alpha_rad)
    sample_beta_rad = np.interp(sample_time_s, time_s, beta_rad)

    canyon_north = np.asarray(canyon.north_samples_ft, dtype=np.float64)
    canyon_center = np.asarray(canyon.center_east_samples_ft, dtype=np.float64)
    canyon_width = np.asarray(canyon.width_samples_ft, dtype=np.float64)

    if hasattr(canyon, "left_half_samples_ft") and hasattr(canyon, "right_half_samples_ft"):
        left_half_ft = np.interp(
            sample_north_ft,
            canyon_north,
            np.asarray(canyon.left_half_samples_ft, dtype=np.float64),
        )
        right_half_ft = np.interp(
            sample_north_ft,
            canyon_north,
            np.asarray(canyon.right_half_samples_ft, dtype=np.float64),
        )
    else:
        half_width_ft = 0.5 * np.interp(sample_north_ft, canyon_north, canyon_width)
        left_half_ft = half_width_ft
        right_half_ft = half_width_ft

    canyon_center_ft = np.interp(sample_north_ft, canyon_north, canyon_center)
    left_wall_ft = canyon_center_ft - left_half_ft
    right_wall_ft = canyon_center_ft + right_half_ft
    symmetric_half_width_ft = np.minimum(
        np.maximum(sample_east_ft - left_wall_ft, 20.0),
        np.maximum(right_wall_ft - sample_east_ft, 20.0),
    )
    sample_width_ft = 2.0 * symmetric_half_width_ft

    return {
        "time_s": sample_time_s.astype(np.float32),
        "north_ft": sample_north_ft.astype(np.float32),
        "east_ft": sample_east_ft.astype(np.float32),
        "heading_rad": sample_psi_rad.astype(np.float32),
        "width_ft": sample_width_ft.astype(np.float32),
        "altitude_ft": sample_altitude_ft.astype(np.float32),
        "speed_fps": sample_speed_fps.astype(np.float32),
        "phi_rad": sample_phi_rad.astype(np.float32),
        "theta_rad": sample_theta_rad.astype(np.float32),
        "psi_rad": sample_psi_rad.astype(np.float32),
        "alpha_rad": sample_alpha_rad.astype(np.float32),
        "beta_rad": sample_beta_rad.astype(np.float32),
        "reference_states_ft_rad": np.column_stack(
            [
                sample_north_ft,
                sample_east_ft,
                sample_altitude_ft,
                sample_phi_rad,
                sample_theta_rad,
                sample_psi_rad,
            ]
        ).astype(np.float32),
        "display_north_ft": display_north_ft.astype(np.float32),
        "display_east_ft": display_east_ft.astype(np.float32),
        "display_altitude_ft": display_altitude_ft.astype(np.float32),
        "closed_loop": False,
        "source_dyn_path": str(dyn.source_path),
    }






__all__ = [
    "build_nominal_reference",
    "load_nominal_initial_conditions",
]
