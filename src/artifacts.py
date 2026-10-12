"""Flight videos and compact result plots; no training data collection."""
from pathlib import Path
import json
import os
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FT_TO_M = 1 / 3.28084
RENDER_SCALE = 1e-3

class FlightRecorder:
    def __init__(self, env, reference, output_dir, settings, enabled=True):
        # Avoid imageio's pre-exec fork hook after JAX has started worker threads.
        os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
        self.env = env
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sim = env.simulation
        self._reference_north_ft = reference["display_north_ft"]
        self._reference_east_ft = reference["display_east_ft"]
        self._reference_altitude_ft = reference["display_altitude_ft"]
        self._writer = None
        self._gif_writer = None
        self.frame_count = 0
        if enabled and settings["record_video"]:
            self._writer = iio.get_writer(self.output_dir / "flight.mp4", format="ffmpeg",
                                         fps=settings["fps"], macro_block_size=1)
        if enabled and settings["record_gif"]:
            self._gif_writer = iio.get_writer(self.output_dir / "flight.gif", format="GIF",
                duration=1000 / settings.get("gif_fps", 10), loop=0)

    def compose(self, frame, planner_debug=None):
        frame = self._overlay_reference_trajectory(frame)
        frame = self._overlay_planner_debug(frame, planner_debug)
        return frame

    def append(self, frame):
        if self._writer is not None:
            self._writer.append_data(frame)
        stride = max(1, round(self.settings["fps"] / self.settings.get("gif_fps", 10)))
        if self._gif_writer is not None and self.frame_count % stride == 0:
            width = self.settings.get("gif_width", 640)
            image = Image.fromarray(frame)
            image = image.resize((width, round(image.height * width / image.width)))
            self._gif_writer.append_data(np.asarray(image))
        self.frame_count += 1

    def close(self):
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            if self._gif_writer is not None:
                self._gif_writer.close()

    @staticmethod
    def _project_world_points(world_points, view, projection, width, height):
        if world_points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        homogeneous = np.concatenate(
            [
                np.asarray(world_points, dtype=np.float32),
                np.ones((world_points.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        clip = (projection @ view @ homogeneous.T).T
        w = np.maximum(clip[:, 3:4], 1e-6)
        ndc = clip[:, :3] / w

        pixels = np.empty((world_points.shape[0], 2), dtype=np.float32)
        pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * float(width - 1)
        pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height - 1)

        valid = np.logical_and.reduce(
            [
                clip[:, 3] > 1e-6,
                ndc[:, 2] > -1.5,
                ndc[:, 2] < 1.5,
            ]
        )
        pixels[~valid] = np.nan
        return pixels

    def _trajectory_world_points(self, xy_ft, h_ft):
        env_unwrapped = self.env.unwrapped
        xy_ft = np.asarray(xy_ft, dtype=np.float32)
        h_ft = np.asarray(h_ft, dtype=np.float32)
        if xy_ft.ndim != 2 or xy_ft.shape[1] != 2:
            return np.zeros((0, 3), dtype=np.float32)

        if h_ft.ndim != 1 or h_ft.shape[0] != xy_ft.shape[0]:
            default_h_ft = float(self._sim.get_property_value("position/h-sl-ft"))
            h_ft = np.full((xy_ft.shape[0],), default_h_ft, dtype=np.float32)

        base_elev_ft = float(getattr(env_unwrapped, "dem_render_base_elev_ft", 0.0))
        start_elev_ft = float(getattr(env_unwrapped, "dem_start_elev_ft", 0.0))

        north_ft = xy_ft[:, 0]
        east_ft = xy_ft[:, 1]
        h_msl_ft = h_ft + start_elev_ft

        world = np.empty((xy_ft.shape[0], 3), dtype=np.float32)
        world[:, 0] = -east_ft * FT_TO_M * RENDER_SCALE
        world[:, 1] = (h_msl_ft - base_elev_ft) * FT_TO_M * RENDER_SCALE
        world[:, 2] = north_ft * FT_TO_M * RENDER_SCALE
        return world

    def _overlay_planner_debug(self, frame, planner_debug):
        if planner_debug is None:
            return frame

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            return frame

        gk_trajectories = np.asarray(
            planner_debug.get("gk_trajectories", np.zeros((0, 0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        gk_h_ft = np.asarray(
            planner_debug.get("gk_h_ft", np.zeros((0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        failure_mask = np.asarray(
            planner_debug.get("failure_mask", np.zeros((0,), dtype=bool)),
            dtype=bool,
        )

        if gk_trajectories.size == 0:
            return frame

        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        if gk_trajectories.size > 0:
            traj_len = int(gk_trajectories.shape[1]) if gk_trajectories.ndim == 3 else 0
            s_t = int(planner_debug.get("s_t", 0))
            plan_start_t = int(planner_debug.get("plan_start_t", 0))
            local_m = s_t - plan_start_t
            m_slice = int(np.clip(local_m, 0, max(traj_len - 1, 0)))

            for idx, traj_xy in enumerate(gk_trajectories):
                traj_h = gk_h_ft[idx] if idx < len(gk_h_ft) else np.zeros((traj_xy.shape[0],), dtype=np.float32)
                world_points = self._trajectory_world_points(traj_xy, traj_h)
                pixels = self._project_world_points(world_points, view, projection, width, height)
                valid_mask = np.all(np.isfinite(pixels), axis=1)

                nom_pixels = pixels[1 : m_slice + 2][valid_mask[1 : m_slice + 2]]
                if len(nom_pixels) >= 2:
                    draw.line([tuple(point) for point in nom_pixels], fill=(0, 255, 0, 255), width=1)

                is_failed = bool(failure_mask[idx]) if idx < len(failure_mask) else False
                backup_color = (255, 30, 30, 255) if is_failed else (0, 0, 255, 255)
                back_pixels = pixels[m_slice + 2 :][valid_mask[m_slice + 2 :]]
                if len(back_pixels) >= 2:
                    draw.line([tuple(point) for point in back_pixels], fill=backup_color, width=1)

                switch_idx = local_m + 1
                if 0 <= switch_idx < len(pixels) and valid_mask[switch_idx]:
                    x_pix, y_pix = pixels[switch_idx]
                    draw.ellipse((x_pix - 3, y_pix - 3, x_pix + 3, y_pix + 3), fill=(255, 235, 59, 210))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _overlay_reference_trajectory(self, frame):
        if self._reference_north_ft is None or self._reference_east_ft is None:
            return frame

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            return frame

        n_arr = np.asarray(self._reference_north_ft, dtype=np.float32).reshape(-1)
        e_arr = np.asarray(self._reference_east_ft, dtype=np.float32).reshape(-1)
        if n_arr.size < 2 or e_arr.size != n_arr.size:
            return frame

        h_arr = None
        if self._reference_altitude_ft is not None:
            h_ref = np.asarray(self._reference_altitude_ft, dtype=np.float32).reshape(-1)
            if h_ref.size == n_arr.size:
                h_arr = h_ref

        cur_n = None
        cur_e = None
        env_unwrapped = self.env.unwrapped
        canyon = getattr(env_unwrapped, "canyon", None)
        if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
            try:
                lat_deg = float(self._sim.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self._sim.get_property_value("position/long-gc-deg"))
                cur_n, cur_e = canyon.get_local_from_latlon(lat_deg, lon_deg)
                cur_n = float(cur_n)
                cur_e = float(cur_e)
            except Exception:
                cur_n = None
                cur_e = None
        if cur_n is None or cur_e is None:
            try:
                state = env_unwrapped.get_full_state_dict()
                cur_n = float(state.get("p_N", np.nan))
                cur_e = float(state.get("p_E", np.nan))
            except Exception:
                cur_n = np.nan
                cur_e = np.nan
            if not (np.isfinite(cur_n) and np.isfinite(cur_e)):
                return frame

        valid = np.isfinite(n_arr) & np.isfinite(e_arr)
        if not np.any(valid):
            return frame

        valid_idx = np.flatnonzero(valid)
        dn = n_arr[valid_idx] - float(cur_n)
        de = e_arr[valid_idx] - float(cur_e)
        nearest_valid = int(np.argmin(dn * dn + de * de))
        start_idx = int(valid_idx[nearest_valid])
        end_idx = int(min(start_idx + 150, n_arr.size - 1))
        if end_idx <= start_idx:
            return frame

        seg_n = n_arr[start_idx : end_idx + 1]
        seg_e = e_arr[start_idx : end_idx + 1]
        if h_arr is None:
            seg_h = np.full_like(seg_n, 0.0, dtype=np.float32)
        else:
            seg_h = h_arr[start_idx : end_idx + 1]
            if seg_h.size != seg_n.size:
                seg_h = np.full_like(seg_n, 0.0, dtype=np.float32)

        xy = np.column_stack([seg_n, seg_e])
        world_points = self._trajectory_world_points(xy, seg_h)
        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)
        pixels = self._project_world_points(world_points, view, projection, width, height)
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
        if pixels.shape[0] < 2:
            return frame

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.line([tuple(p) for p in pixels], fill=(255, 165, 0, 230), width=3)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_case_results(output_dir, canyon, reference, states, clearances, backup_flags, summary):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    states = np.asarray(states, dtype=np.float32)
    times = np.arange(states.shape[0], dtype=np.float32) / 30
    np.savez_compressed(directory / 'trajectory.npz', state=states, time_s=times,
                        terrain_clearance_ft=np.asarray(clearances), using_backup=np.asarray(backup_flags))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_terrain(axes[0], canyon, reference)
    axes[0].plot(states[:, 1], states[:, 0], color='crimson', label=summary['case'])
    axes[0].legend()
    axes[1].plot(times, states[:, 2], label='Aircraft altitude')
    axes[1].plot(times, states[:, 2] - np.asarray(clearances), color='sienna', label='Terrain')
    flags = np.asarray(backup_flags, dtype=bool)
    axes[1].scatter(times[flags], states[flags, 2], s=8, color='orange', label='Backup active')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Altitude relative to canyon entrance (ft)')
    axes[1].legend()
    fig.suptitle(f"{summary['controller'].upper()} / {summary['case']}: {summary['termination_reason']}")
    fig.savefig(directory / 'trajectory.png', dpi=140)
    plt.close(fig)


def _plot_terrain(ax, canyon, reference):
    ax.imshow(canyon.ordered_dem_msl_m, origin='lower', cmap='terrain',
              extent=(canyon.east_samples_ft[0], canyon.east_samples_ft[-1],
                      canyon.north_samples_ft[0], canyon.north_samples_ft[-1]), alpha=.75)
    ax.plot(reference['east_ft'], reference['north_ft'], '--', color='cyan', label='Saved reference')
    ax.set(xlabel='Local east (ft)', ylabel='Local north (ft)',
           xlim=(float(np.min(reference['east_ft'])) - 1000, float(np.max(reference['east_ft'])) + 1000),
           ylim=(float(np.min(reference['north_ft'])) - 500, float(np.max(reference['north_ft'])) + 500))


def save_comparison(output_dir, results):
    directory = Path(output_dir)
    (directory / 'comparison.json').write_text(json.dumps(
        {name: result['summary'] for name, result in results.items()}, indent=2, allow_nan=False) + '\n')
    first = next(iter(results.values()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_terrain(axes[0], first['canyon'], first['reference'])
    colors = {'nominal': 'crimson', 'gatekeeper': 'navy'}
    for name, result in results.items():
        states = np.asarray(result['states'])
        times = np.arange(len(states)) / 30
        axes[0].plot(states[:, 1], states[:, 0], color=colors[name], label=name)
        axes[1].plot(times, result['clearances'], color=colors[name], label=name)
    axes[0].legend()
    axes[1].axhline(first['collision_buffer_ft'], color='sienna', linestyle='--', label='Collision buffer')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Terrain clearance (ft)')
    axes[1].legend()
    fig.suptitle('Nominal flight and DRS-Gatekeeper')
    fig.savefig(directory / 'comparison.png', dpi=140)
    plt.close(fig)
