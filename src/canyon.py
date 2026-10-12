from pathlib import Path

import imageio.v3 as iio
import numpy as np
from collections import deque


FT_PER_DEG_LAT = 364000.0
M_TO_FT = 3.28084

# Module-level cache to avoid recomputing heavy BFS centerlines for the same DEM and start pixel
_DEM_CENTERLINE_CACHE = {}

class DEMCanyon:
    """Canyon width profile extracted from a DEM raster clip.

    The profile is sampled along the north-south axis of the DEM and converted to
    feet using the provided WGS84 bounding box.
    """

    def __init__(
        self,
        dem_path,
        south,
        north,
        west,
        east,
        valley_rel_elev=0.30,
        smoothing_window=31,
        min_width_ft=120.0,
        max_width_ft=5000.0,
        fly_direction="south_to_north",
        dem_start_pixel=None,
    ):
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

        if not (south < north and west < east):
            raise ValueError("Invalid DEM bbox; expected south<north and west<east")

        if not (0.05 <= valley_rel_elev <= 0.95):
            raise ValueError("valley_rel_elev must be between 0.05 and 0.95")

        if fly_direction not in {"south_to_north", "north_to_south"}:
            raise ValueError("fly_direction must be 'south_to_north' or 'north_to_south'")

        self.south = float(south)
        self.north = float(north)
        self.west = float(west)
        self.east = float(east)
        self.fly_direction = fly_direction

        dem = iio.imread(self.dem_path).astype(np.float32)
        if dem.ndim == 3:
            dem = dem[..., 0]

        dem[(~np.isfinite(dem)) | (dem < -1e20)] = np.nan
        if dem.ndim != 2 or dem.size == 0:
            raise ValueError(f"DEM array has invalid shape: {dem.shape}")

        rows, cols = dem.shape
        self.rows = int(rows)
        self.cols = int(cols)
        mid_lat = 0.5 * (south + north)

        feet_per_deg_lon = FT_PER_DEG_LAT * np.cos(np.deg2rad(mid_lat))
        total_ns_ft = (north - south) * FT_PER_DEG_LAT
        total_ew_ft = (east - west) * feet_per_deg_lon

        # At least one-foot spacing prevents unstable gradients for tiny test DEMs.
        self.row_spacing_ft = max(total_ns_ft / max(rows - 1, 1), 1.0)
        self.col_spacing_ft = max(total_ew_ft / max(cols - 1, 1), 1.0)

        if fly_direction == "south_to_north":
            row_indices = np.arange(rows - 1, -1, -1)
        else:
            row_indices = np.arange(rows)

        ordered_dem = dem[row_indices, :]
        finite = np.isfinite(ordered_dem)
        fill_value = float(np.nanmedian(ordered_dem[finite])) if np.any(finite) else 0.0
        ordered_dem = np.where(finite, ordered_dem, fill_value)
        self.ordered_dem_msl_m = ordered_dem.astype(np.float32)
        surface_ref_m = float(np.min(ordered_dem))
        self.surface_elevation_m = (ordered_dem - surface_ref_m).astype(np.float32)
        self.surface_ref_m = surface_ref_m

        # Track a smooth valley-floor centerline for local heading estimates.
        safe_dem = np.where(np.isfinite(ordered_dem), ordered_dem, np.inf)
        center_col_samples = np.argmin(safe_dem, axis=1).astype(np.int32)
        if smoothing_window > 1:
            window = int(smoothing_window)
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=np.float32) / float(window)
            center_col_samples = np.clip(
                np.rint(np.convolve(center_col_samples.astype(np.float32), kernel, mode="same")),
                0,
                cols - 1,
            ).astype(np.int32)
        self.center_col_samples = center_col_samples

        left_halves = np.zeros(rows, dtype=np.float32)
        right_halves = np.zeros(rows, dtype=np.float32)
        wall_heights = np.zeros(rows, dtype=np.float32)
        last_width = float(min_width_ft)
        last_left = 0.5 * last_width
        last_right = 0.5 * last_width
        last_wall_height = 0.5 * max(min_width_ft, 180.0)
        for i in range(rows):
            width_ft, left_ft, right_ft, wall_height_ft = self._estimate_row_profile(
                z_row=ordered_dem[i],
                valley_rel_elev=valley_rel_elev,
                min_width_ft=min_width_ft,
                max_width_ft=max_width_ft,
                fallback_width_ft=last_width,
                fallback_left_ft=last_left,
                fallback_right_ft=last_right,
                fallback_wall_height_ft=last_wall_height,
            )
            left_halves[i] = left_ft
            right_halves[i] = right_ft
            wall_heights[i] = wall_height_ft
            last_width = width_ft
            last_left = left_ft
            last_right = right_ft
            last_wall_height = wall_height_ft

        if smoothing_window > 1:
            window = int(smoothing_window)
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=np.float32) / float(window)
            left_halves = np.convolve(left_halves, kernel, mode="same")
            right_halves = np.convolve(right_halves, kernel, mode="same")
            wall_heights = np.convolve(wall_heights, kernel, mode="same")

        widths = left_halves + right_halves
        widths_clipped = np.clip(widths, min_width_ft, max_width_ft)
        scaling = np.ones_like(widths_clipped, dtype=np.float32)
        valid_widths = widths > 1e-6
        scaling[valid_widths] = widths_clipped[valid_widths] / widths[valid_widths]
        left_halves = left_halves * scaling
        right_halves = right_halves * scaling
        widths = widths_clipped

        wall_heights = np.clip(wall_heights, 180.0, 3500.0)
        self.north_samples_ft = np.arange(rows, dtype=np.float32) * self.row_spacing_ft
        self.east_samples_ft = (
            (np.arange(cols, dtype=np.float32) - 0.5 * float(cols - 1)) * self.col_spacing_ft
        ).astype(np.float32)
        self.center_east_samples_ft = self.east_samples_ft[self.center_col_samples].astype(np.float32)
        self.width_samples_ft = widths.astype(np.float32)
        self.left_half_samples_ft = left_halves.astype(np.float32)
        self.right_half_samples_ft = right_halves.astype(np.float32)
        self.wall_height_samples_ft = wall_heights.astype(np.float32)
        self.grad_samples = np.gradient(self.width_samples_ft, self.row_spacing_ft).astype(np.float32)

        # Trace a robust centerline using BFS flood-fill if start_pixel is provided.
        # Otherwise, fall back to the simple row-min approach.
        self._compute_centerline(dem_start_pixel, smoothing_window)

        self.total_length_ft = float(self.north_samples_ft[-1]) if rows > 1 else 0.0
        self.anchor_north_ft = 0.0

    def set_anchor_north(self, anchor_north_ft):
        self.anchor_north_ft = float(anchor_north_ft)

    def get_total_length_ft(self):
        return float(self.total_length_ft)

    def get_surface_grid(self):
        return self.surface_elevation_m, self.north_samples_ft, self.east_samples_ft

    def get_pixel_info(self, pixel_x, pixel_y):
        col = int(np.clip(int(round(pixel_x)), 0, self.cols - 1))
        row_original = int(np.clip(int(round(pixel_y)), 0, self.rows - 1))

        if self.fly_direction == "south_to_north":
            row_ordered = self.rows - 1 - row_original
        else:
            row_ordered = row_original

        local_north_ft = float(self.north_samples_ft[row_ordered])
        local_east_ft = float(self.east_samples_ft[col])
        elev_msl_m = float(self.ordered_dem_msl_m[row_ordered, col])

        lat_deg = self.north - ((row_original + 0.5) / float(self.rows)) * (self.north - self.south)
        lon_deg = self.west + ((col + 0.5) / float(self.cols)) * (self.east - self.west)

        return {
            "pixel_x": col,
            "pixel_y": row_original,
            "row_ordered": row_ordered,
            "local_north_ft": local_north_ft,
            "local_east_ft": local_east_ft,
            "lat_deg": float(lat_deg),
            "lon_deg": float(lon_deg),
            "elevation_msl_m": elev_msl_m,
            "elevation_msl_ft": elev_msl_m * M_TO_FT,
        }

    def get_centerline_pixel_info(self, pixel_x, pixel_y):
        base_info = self.get_pixel_info(pixel_x, pixel_y)
        row_ordered = int(base_info["row_ordered"])
        center_east_ft = float(self.center_east_samples_ft[row_ordered])
        center_pixel_x = float(
            np.interp(center_east_ft, self.east_samples_ft, np.arange(self.cols, dtype=np.float32))
        )
        return self.get_pixel_info(center_pixel_x, base_info["pixel_y"])

    def _compute_centerline(self, dem_start_pixel, smoothing_window):
        """Traces the canyon floor using a constrained BFS flood-fill."""
        # Use a cache to avoid recomputing if this DEM and start_pixel have been processed
        cache_key = (self.dem_path, dem_start_pixel)
        if cache_key in _DEM_CENTERLINE_CACHE:
            cached_east, cached_heading = _DEM_CENTERLINE_CACHE[cache_key]
            self.center_east_samples_ft = cached_east.copy()
            self.centerline_heading_samples_rad = cached_heading.copy()
            return

        rows, cols = self.rows, self.cols
        center_east = self.center_east_samples_ft.copy()
        centerline_heading = np.zeros_like(center_east)

        if dem_start_pixel is not None:
            try:
                px, py = dem_start_pixel
                start_info = self.get_pixel_info(px, py)
                start_row = int(start_info["row_ordered"])
                start_col = int(start_info["pixel_x"])

                finite_dem = np.where(np.isfinite(self.ordered_dem_msl_m), self.ordered_dem_msl_m, np.inf)

                # Dynamic valley thresholding similar to run_scenario.py
                row_min = np.nanmin(self.ordered_dem_msl_m, axis=1)
                row_max = np.nanmax(self.ordered_dem_msl_m, axis=1)
                row_span = row_max - row_min
                valley_frac = 0.20
                row_threshold = row_min + valley_frac * np.maximum(row_span, 1.0)

                valley_mask = finite_dem <= row_threshold[:, None]

                visited = np.zeros((rows, cols), dtype=bool)
                best_col = np.full(rows, -1, dtype=np.int32)
                best_elev = np.full(rows, np.inf, dtype=np.float64)

                queue = deque()
                queue.append((start_row, start_col))
                visited[start_row, start_col] = True

                neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                while queue:
                    r, c = queue.popleft()
                    elev = float(finite_dem[r, c])
                    if elev < best_elev[r]:
                        best_elev[r] = elev
                        best_col[r] = c
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and valley_mask[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                # Fill gaps
                tracked_cols = np.full(rows, start_col, dtype=np.int32)
                reached_rows = np.where(best_col >= 0)[0]
                if len(reached_rows) > 0:
                    tracked_cols[best_col >= 0] = best_col[best_col >= 0]
                    for r in range(rows):
                        if best_col[r] < 0:
                            nearest_idx = np.argmin(np.abs(reached_rows - r))
                            tracked_cols[r] = tracked_cols[reached_rows[nearest_idx]]

                center_east = self.east_samples_ft[np.clip(tracked_cols, 0, cols - 1)]
            except Exception:
                pass # Fallback to existing center_east_samples_ft (naive row-min)

        # Smooth and compute headings
        if center_east.size >= 9:
            smooth_win = min(21, center_east.size // 2 * 2 + 1)
            smooth_win = max(9, smooth_win)
            kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
            center_east = np.convolve(center_east, kernel, mode="same").astype(np.float32)

            east_min_ft = float(self.east_samples_ft[0])
            east_max_ft = float(self.east_samples_ft[-1])
            edge_margin_ft = 2.0 * float(self.col_spacing_ft)
            half_width_ft = 0.5 * np.asarray(self.width_samples_ft, dtype=np.float32)
            idx = np.arange(center_east.size, dtype=np.float32)
            for _ in range(8):
                invalid = (
                    (~np.isfinite(center_east))
                    | ((center_east - half_width_ft) <= (east_min_ft + edge_margin_ft))
                    | ((center_east + half_width_ft) >= (east_max_ft - edge_margin_ft))
                )
                if not np.any(invalid):
                    break
                valid = ~invalid
                if np.count_nonzero(valid) < 2:
                    break
                center_east = np.interp(idx, idx[valid], center_east[valid]).astype(np.float32)

            lookahead = min(45, max(12, center_east.size // 18))
            for i in range(center_east.size):
                j = min(i + lookahead, center_east.size - 1)
                if j == i:
                    centerline_heading[i] = centerline_heading[i-1] if i > 0 else 0.0
                else:
                    dn = self.north_samples_ft[j] - self.north_samples_ft[i]
                    de = center_east[j] - center_east[i]
                    centerline_heading[i] = np.arctan2(de, dn)

        self.center_east_samples_ft = center_east
        self.centerline_heading_samples_rad = centerline_heading

        # Cache the result
        _DEM_CENTERLINE_CACHE[cache_key] = (center_east.copy(), centerline_heading.copy())

    def get_local_from_latlon(self, lat_deg, lon_deg):
        row_ordered, col = self._latlon_to_ordered_row_col(lat_deg, lon_deg)

        row_axis = np.arange(self.rows, dtype=np.float32)
        col_axis = np.arange(self.cols, dtype=np.float32)

        local_north_ft = float(np.interp(row_ordered, row_axis, self.north_samples_ft))
        local_east_ft = float(np.interp(col, col_axis, self.east_samples_ft))
        return local_north_ft, local_east_ft

    def get_elevation_msl_ft_from_latlon(self, lat_deg, lon_deg):
        row_ordered, col = self._latlon_to_ordered_row_col(lat_deg, lon_deg)

        r0 = int(np.floor(row_ordered))
        c0 = int(np.floor(col))
        r1 = min(r0 + 1, self.rows - 1)
        c1 = min(c0 + 1, self.cols - 1)

        dr = float(row_ordered - r0)
        dc = float(col - c0)

        z00 = float(self.ordered_dem_msl_m[r0, c0])
        z01 = float(self.ordered_dem_msl_m[r0, c1])
        z10 = float(self.ordered_dem_msl_m[r1, c0])
        z11 = float(self.ordered_dem_msl_m[r1, c1])

        z0 = (1.0 - dc) * z00 + dc * z01
        z1 = (1.0 - dc) * z10 + dc * z11
        z_msl_m = (1.0 - dr) * z0 + dr * z1
        return float(z_msl_m * M_TO_FT)

    def _latlon_to_ordered_row_col(self, lat_deg, lon_deg):
        row_original = ((self.north - float(lat_deg)) / (self.north - self.south)) * float(self.rows) - 0.5
        col = ((float(lon_deg) - self.west) / (self.east - self.west)) * float(self.cols) - 0.5

        row_original = float(np.clip(row_original, 0.0, float(self.rows - 1)))
        col = float(np.clip(col, 0.0, float(self.cols - 1)))

        if self.fly_direction == "south_to_north":
            row_ordered = float(self.rows - 1) - row_original
        else:
            row_ordered = row_original

        return row_ordered, col

    def get_centerline_heading_deg(self, row_ordered, lookahead_rows=40):
        if self.rows <= 1:
            return 0.0

        i0 = int(np.clip(int(round(row_ordered)), 0, self.rows - 1))
        step = max(int(lookahead_rows), 1)
        i1 = min(i0 + step, self.rows - 1)

        if i1 != i0:
            dn = float(self.north_samples_ft[i1] - self.north_samples_ft[i0])
            de = float(self.center_east_samples_ft[i1] - self.center_east_samples_ft[i0])
        else:
            i_prev = max(i0 - step, 0)
            dn = float(self.north_samples_ft[i0] - self.north_samples_ft[i_prev])
            de = float(self.center_east_samples_ft[i0] - self.center_east_samples_ft[i_prev])

        if abs(dn) + abs(de) <= 1e-6:
            return 0.0

        heading_deg = float(np.degrees(np.arctan2(de, dn)))
        return float((heading_deg + 360.0) % 360.0)

    def get_heading_for_pixel(self, pixel_x, pixel_y, lookahead_rows=40, search_radius_px=80):
        pixel_info = self.get_pixel_info(pixel_x, pixel_y)
        row0 = int(pixel_info["row_ordered"])
        col0 = int(pixel_info["pixel_x"])

        step = max(int(lookahead_rows), 1)
        search_radius_px = max(int(search_radius_px), 1)

        row1 = min(row0 + step, self.rows - 1)
        if row1 > row0:
            col = col0
            for row in range(row0 + 1, row1 + 1):
                lo = max(0, col - search_radius_px)
                hi = min(self.cols - 1, col + search_radius_px)
                row_profile = self.ordered_dem_msl_m[row, lo : hi + 1]
                col = lo + int(np.argmin(row_profile))

            dn = float(self.north_samples_ft[row1] - self.north_samples_ft[row0])
            de = float(self.east_samples_ft[col] - self.east_samples_ft[col0])
        else:
            row_prev = max(row0 - step, 0)
            col = col0
            for row in range(row0 - 1, row_prev - 1, -1):
                lo = max(0, col - search_radius_px)
                hi = min(self.cols - 1, col + search_radius_px)
                row_profile = self.ordered_dem_msl_m[row, lo : hi + 1]
                col = lo + int(np.argmin(row_profile))

            dn = float(self.north_samples_ft[row0] - self.north_samples_ft[row_prev])
            de = float(self.east_samples_ft[col0] - self.east_samples_ft[col])

        if abs(dn) + abs(de) <= 1e-6:
            return self.get_centerline_heading_deg(row0, lookahead_rows=lookahead_rows)

        heading_deg = float(np.degrees(np.arctan2(de, dn)))
        return float((heading_deg + 360.0) % 360.0)

    def _to_local_north(self, p_N):
        local_north_ft = float(p_N) - self.anchor_north_ft
        if self.total_length_ft <= 0.0:
            return 0.0
        return float(np.clip(local_north_ft, 0.0, self.total_length_ft))

    def get_geometry(self, p_N):
        local_north_ft = self._to_local_north(p_N)
        if self.total_length_ft <= 0.0:
            return float(self.width_samples_ft[0]), float(self.grad_samples[0])

        width = float(np.interp(local_north_ft, self.north_samples_ft, self.width_samples_ft))
        grad = float(np.interp(local_north_ft, self.north_samples_ft, self.grad_samples))
        return width, grad

    def get_wall_profile(self, p_N):
        local_north_ft = self._to_local_north(p_N)
        left_half_ft = float(np.interp(local_north_ft, self.north_samples_ft, self.left_half_samples_ft))
        right_half_ft = float(np.interp(local_north_ft, self.north_samples_ft, self.right_half_samples_ft))
        wall_height_ft = float(np.interp(local_north_ft, self.north_samples_ft, self.wall_height_samples_ft))
        return left_half_ft, right_half_ft, wall_height_ft

    def _estimate_row_profile(
        self,
        z_row,
        valley_rel_elev,
        min_width_ft,
        max_width_ft,
        fallback_width_ft,
        fallback_left_ft,
        fallback_right_ft,
        fallback_wall_height_ft,
    ):
        finite = np.isfinite(z_row)
        if not np.any(finite):
            return (
                float(fallback_width_ft),
                float(fallback_left_ft),
                float(fallback_right_ft),
                float(fallback_wall_height_ft),
            )

        z_valid = z_row[finite]
        z_min = float(np.min(z_valid))
        z_max = float(np.max(z_valid))
        z_span = z_max - z_min
        if z_span <= 1e-6:
            return (
                float(fallback_width_ft),
                float(fallback_left_ft),
                float(fallback_right_ft),
                float(fallback_wall_height_ft),
            )

        threshold = z_min + valley_rel_elev * z_span
        valley_mask = finite & (z_row <= threshold)

        # If the threshold misses the channel entirely, degrade gracefully.
        if not np.any(valley_mask):
            return (
                float(fallback_width_ft),
                float(fallback_left_ft),
                float(fallback_right_ft),
                float(fallback_wall_height_ft),
            )

        safe_row = np.where(finite, z_row, np.inf)
        center_idx = int(np.argmin(safe_row))
        if not valley_mask[center_idx]:
            candidates = np.flatnonzero(valley_mask)
            nearest = int(np.argmin(np.abs(candidates - center_idx)))
            center_idx = int(candidates[nearest])

        left = center_idx
        right = center_idx
        while left > 0 and valley_mask[left - 1]:
            left -= 1
        while right < valley_mask.shape[0] - 1 and valley_mask[right + 1]:
            right += 1

        left_px = max(center_idx - left + 1, 1)
        right_px = max(right - center_idx + 1, 1)

        left_half_ft = left_px * self.col_spacing_ft
        right_half_ft = right_px * self.col_spacing_ft
        width_ft = left_half_ft + right_half_ft

        width_clipped_ft = float(np.clip(width_ft, min_width_ft, max_width_ft))
        if width_ft > 1e-6:
            scale = width_clipped_ft / width_ft
            left_half_ft *= scale
            right_half_ft *= scale

        wall_height_ft = max(0.7 * z_span * M_TO_FT, 180.0)

        return (
            float(width_clipped_ft),
            float(left_half_ft),
            float(right_half_ft),
            float(wall_height_ft),
        )
