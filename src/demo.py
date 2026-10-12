"""Run independent nominal and supervised flights through the saved canyon."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import numpy as np

from src import config
from src.artifacts import FlightRecorder, save_case_results, save_comparison
from src.canyon import DEMCanyon
from src.canyon_env import CanyonFlightEnv
from src.integration import (
    controller_state, state_vector, build_gatekeeper,
    _estimate_progress_fraction_from_position,
)
from src.nominal_trajectory import build_nominal_reference, load_nominal_initial_conditions
from src.pid import PIDTrajectoryController
from src.simulation import DemoCancelled

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent if (PACKAGE_ROOT.parent / "run.py").is_file() else Path.cwd()
TRAJECTORY_PATH = PACKAGE_ROOT / "assets/models/nominal_trajectory.npz"


def validate_configuration(max_steps):
    start, end = config.DEMO["nominal_start_fraction"], config.DEMO["nominal_end_fraction"]
    if not 0 <= start < end <= 1:
        raise ValueError("Trajectory fractions must satisfy 0 <= start < end <= 1.")
    if max_steps < 1:
        raise ValueError("The maximum step count must be positive.")
    gk = config.GATEKEEPER
    if any(not isinstance(gk[k], int) or gk[k] < 1 for k in ("M", "N", "T", "update_interval_steps", "p")):
        raise ValueError("Gatekeeper counts, update interval, and Wasserstein order must be positive integers.")
    if gk["M"] > gk["T"] or not 0 <= gk["alpha"] < gk["epsilon"] < 1 or not 0 < gk["delta"] < 1:
        raise ValueError("Require M <= T, 0 <= alpha < epsilon < 1, and 0 < delta < 1.")
    if gk["beta"] < 0 or gk["lipschitz_constant"] < 0 or gk["lipschitz_mode"] not in {"fixed", "autodiff"}:
        raise ValueError("Select fixed/autodiff Lipschitz mode with nonnegative radius and bound.")
    if any(config.VISUALIZATION[k] < 1 for k in ("width", "height", "fps", "gif_fps", "gif_width")):
        raise ValueError("Rendering dimensions and frame rates must be positive.")


def make_environment(headless, max_steps):
    settings = dict(config.ENVIRONMENT)
    dem_path = Path(settings["dem_path"]).expanduser()
    if not dem_path.is_absolute():
        dem_path = PACKAGE_ROOT / dem_path
    settings["dem_path"] = dem_path
    south, north, west, east = settings["dem_bbox"]
    canyon = DEMCanyon(
        dem_path, south=south, north=north, west=west, east=east,
        valley_rel_elev=settings["dem_valley_rel_elev"],
        smoothing_window=settings["dem_smoothing_window"],
        min_width_ft=settings["dem_min_width_ft"], max_width_ft=settings["dem_max_width_ft"],
        fly_direction="south_to_north", dem_start_pixel=settings["dem_start_pixel"],
    )
    initial = load_nominal_initial_conditions(
        TRAJECTORY_PATH, canyon=canyon, progress_fraction=config.DEMO["nominal_start_fraction"])
    for setting, saved in {
        "entry_speed_kts": "speed_kts", "entry_altitude_ft": "entry_altitude_ft",
        "dem_start_heading_deg": "heading_deg", "entry_roll_deg": "roll_deg",
        "entry_pitch_deg": "pitch_deg", "entry_alpha_deg": "alpha_deg", "entry_beta_deg": "beta_deg",
    }.items():
        if settings.get(setting) is None:
            settings[setting] = initial[saved]
    settings.update(dem_start_pixel=initial["start_pixel"], max_episode_steps=max_steps)
    env = CanyonFlightEnv(
        render_mode=None if headless else "human", **settings,
        render_width=config.VISUALIZATION["width"], render_height=config.VISUALIZATION["height"],
        render_fps=config.VISUALIZATION["fps"],
    )
    return env


def _gatekeeper_debug(gatekeeper):
    if gatekeeper is None or gatekeeper.state.predicted_trajectories is None:
        return None
    state = gatekeeper.state
    paths = state.predicted_trajectories
    return {
        "gk_trajectories": paths[:, :, :2], "gk_h_ft": paths[:, :, 2],
        "failure_mask": state.failure_mask, "s_t": state.s_prev, "plan_start_t": state.plan_start_t,
    }


def run_case(case, *, headless=False, max_steps=1200, output_root=None, record=True):
    """Run a fresh case and return its compact numerical results."""
    started = time.perf_counter()
    if output_root is None:
        output_root = PROJECT_ROOT / config.DEMO["output_dir"]
    output_dir = Path(output_root) / case
    env = make_environment(headless, max_steps)
    recorder = None
    try:
        _, info = env.reset(seed=config.DEMO["seed"])
        reference = build_nominal_reference(
            TRAJECTORY_PATH, canyon=env.canyon, altitude_ref_ft=env.dem_start_elev_ft,
            resample_spacing_ft=config.DEMO["reference_spacing_ft"],
            end_fraction=config.DEMO["nominal_end_fraction"],
        )
        controller = PIDTrajectoryController(env, reference, config=config.PID)
        controller.reset()
        gatekeeper = backup = None
        if case == "gatekeeper":
            gatekeeper, backup = build_gatekeeper(
                env, controller_state(env), reference, config.DEMO["seed"])
            print("Compiling the Gatekeeper on first update…", flush=True)
        if not headless:
            recorder = FlightRecorder(env, reference, output_dir, config.VISUALIZATION, enabled=record)
        states = [np.asarray(state_vector(controller_state(env)))]
        clearances, flags = [float(info["terrain_clearance_ft"])], [False]
        backup_count, termination, cancelled = 0, "time_limit", False
        progress = _estimate_progress_fraction_from_position(
            states[0][0], states[0][1], reference["north_ft"], reference["east_ft"])
        print(f"Starting PID / {case}, seed {config.DEMO['seed']}", flush=True)
        try:
            for step in range(max_steps):
                frame_started = time.perf_counter()
                state = controller_state(env)
                nominal_action = controller.get_action(state)
                action, using_backup = nominal_action, False
                if gatekeeper is not None:
                    if step % config.GATEKEEPER["update_interval_steps"] == 0:
                        gatekeeper.update(state_vector(state))
                    using_backup = bool(gatekeeper.state.using_backup)
                    if using_backup:
                        action = backup.get_action(state)
                        backup_count += 1
                if not np.isfinite(action).all():
                    raise RuntimeError("The controller returned a non-finite action.")
                _, _, terminated, truncated, info = env.step(action)
                next_state = np.asarray(state_vector(controller_state(env)))
                states.append(next_state)
                clearances.append(float(info["terrain_clearance_ft"]))
                flags.append(using_backup)
                progress = _estimate_progress_fraction_from_position(
                    next_state[0], next_state[1], reference["north_ft"], reference["east_ft"])
                termination = info["termination_reason"]
                if progress >= .95 and not (terminated or truncated):
                    termination = "mission_success"
                if not headless:
                    mode = "BACKUP" if using_backup else "TRACK"
                    env.set_hud_commands(mode_labels=("PID", case.upper(), mode))
                    env.set_caption(f"F-16 / {case.upper()} / PID")
                    frame = env.render(present=False)
                    frame = recorder.compose(frame, _gatekeeper_debug(gatekeeper))
                    env.present_frame(frame)
                    recorder.append(frame)
                if gatekeeper is not None:
                    gatekeeper.tick()
                if step % 100 == 0:
                    print(f"  step {step + 1}, progress {progress:.0%}, clearance {clearances[-1]:.0f} ft", flush=True)
                if terminated or truncated or termination == "mission_success":
                    break
                if not headless:
                    time.sleep(max(0, 1/config.VISUALIZATION["fps"] - (time.perf_counter() - frame_started)))
        except (DemoCancelled, KeyboardInterrupt):
            termination, cancelled = "cancelled", True
        if recorder is not None:
            recorder.close()
            recorder = None
        elapsed = time.perf_counter() - started
        summary = {
            "case": case, "controller": "pid", "seed": config.DEMO["seed"],
            "gatekeeper_enabled": gatekeeper is not None, "termination_reason": termination,
            "mission_success": termination == "mission_success", "steps": len(states) - 1,
            "simulation_duration_s": (len(states) - 1)/30, "elapsed_s": elapsed,
            "minimum_terrain_clearance_ft": min(clearances), "backup_steps_used": backup_count,
            "nominal_progress_fraction": float(progress), "headless": headless,
            "nominal_start_fraction": config.DEMO["nominal_start_fraction"],
            "nominal_end_fraction": config.DEMO["nominal_end_fraction"],
            "maximum_steps": max_steps,
        }
        save_case_results(output_dir, env.canyon, reference, states, clearances, flags, summary)
        print(f"Finished {case}: {termination} in {elapsed:.1f}s. Results: {output_dir}", flush=True)
        return dict(summary=summary, states=states, clearances=clearances, reference=reference,
                    canyon=env.canyon, collision_buffer_ft=env.terrain_collision_buffer_ft, cancelled=cancelled)
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nominal and DRS-Gatekeeper F-16 canyon flights.")
    parser.add_argument("--headless", action="store_true", help="Numerical run without a display or video.")
    parser.add_argument("--case", choices=("both", "nominal", "gatekeeper"), default="both")
    parser.add_argument("--max-steps", type=int, default=config.DEMO["max_steps"])
    parser.add_argument("--no-record", action="store_true", help="Display flights without saving MP4/GIF.")
    args = parser.parse_args(argv)
    validate_configuration(args.max_steps)
    print(f"JAX backend: {jax.default_backend()} ({jax.devices()[0].device_kind})", flush=True)
    output = Path(config.DEMO["output_dir"]).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in ("nominal", "gatekeeper") if args.case == "both" else (args.case,):
        results[case] = run_case(case, headless=args.headless,
                                max_steps=args.max_steps, output_root=output, record=not args.no_record)
        if results[case]["cancelled"]:
            break
    save_comparison(output, results)
    return 130 if any(result["cancelled"] for result in results.values()) else 0
