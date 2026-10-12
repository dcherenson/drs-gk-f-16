"""Small, deterministic JSBSim runtime used by the canyon demo.

The original project exposed a Gym environment whose constructor also owned
the JSBSim setup.  The demo needs the same aircraft model and integration
rate, but it must be usable from any working directory and without importing
the renderer.  This module contains that non-visual part of the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class DemoCancelled(RuntimeError):
    """Raised when the user closes the demo window or presses Escape."""


PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
RADIUS_M = 6.3781e6
RADIUS_FT = RADIUS_M * 3.28084

# This is the same property order used by the source JSBSim environment.  The
# first two entries are converted to local metres by ``_get_state``; telemetry
# returned by CanyonFlightEnv is kept in feet and is collected separately.
STATE_FORMAT = (
    "position/lat-gc-rad",
    "position/long-gc-rad",
    "position/h-sl-meters",
    "velocities/mach",
    "aero/alpha-rad",
    "aero/beta-rad",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "attitude/phi-rad",
    "attitude/theta-rad",
    "attitude/psi-rad",
)


def _as_path(root: str | Path | None) -> Path:
    """Resolve a model root without consulting the process working directory."""

    if root is None:
        return ASSETS_ROOT
    return Path(root).expanduser().resolve()


class F16Simulation:
    """JSBSim F-16 setup and the source project's four-tick stepping logic.

    ``simulation`` is intentionally the underlying ``FGFDMExec`` instance so
    existing controllers can continue to use JSBSim property names directly.
    The class itself contains only numerical runtime state and has no Gym or
    rendering dependency.
    """

    down_sample = 4
    dt = 1.0 / 120.0
    wind_theta = 0.5
    default_wind_sigma = 15.0

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        rng: np.random.RandomState | np.random.Generator | None = None,
        wind_sigma: float = default_wind_sigma,
    ) -> None:
        self.model_root = _as_path(root)
        self.aircraft_root = self.model_root / "aircraft" / "f16"
        if not self.aircraft_root.is_dir():
            raise FileNotFoundError(
                "F-16 aircraft assets were not found under "
                f"{self.aircraft_root}. Pass root=<assets root> containing aircraft/f16."
            )

        # Keep importing JSBSim out of module import time.  This lets callers
        # import DemoCancelled and the numerical helpers in lightweight tools.
        try:
            import jsbsim
        except ImportError as exc:  # pragma: no cover - exercised by packaging checks
            raise ImportError("src requires the jsbsim package to run the aircraft") from exc

        self._jsbsim = jsbsim
        self.simulation = self._new_simulation()

        self.down_sample = int(type(self).down_sample)
        self.dt = float(type(self).dt)
        self.wind_theta = float(type(self).wind_theta)
        self.wind_sigma = float(wind_sigma)
        self.rng = rng if rng is not None else np.random.RandomState()
        self.wind_state = np.zeros(3, dtype=np.float64)
        self.time_sec = 0.0
        self.state = np.zeros(len(STATE_FORMAT), dtype=np.float64)
        self._has_initialized = False

        self._set_initial_conditions()
        self.reset_simulation()
        self._has_initialized = True

    def _set_initial_conditions(self) -> None:
        """Set the source model's baseline initial conditions."""

        self.simulation.set_property_value("propulsion/set-running", -1)
        self.simulation.set_property_value("ic/u-fps", 900.0)
        self.simulation.set_property_value("ic/h-sl-ft", 5000.0)

    def _new_simulation(self):
        """Create a clean JSBSim executive for a fresh case."""

        simulation = self._jsbsim.FGFDMExec(str(self.model_root), None)
        simulation.set_debug_level(0)
        simulation.set_dt(type(self).dt)
        simulation.load_model("f16")
        return simulation

    def reset_simulation(
        self,
        initial_conditions: Mapping[str, Any] | None = None,
        *,
        fresh: bool = False,
    ) -> None:
        """Run the JSBSim initial-condition pass and reset numerical wind state."""

        # RunIC updates the property tree, but it does not clear every
        # integrator state after a case has advanced.  Recreate the executive
        # between cases so reset(seed=...) is deterministic and independent of
        # the previous trajectory.  This also preserves the fresh-model state
        # used by the source DataCollectionEnv on its first case.
        if fresh and self._has_initialized:
            close = getattr(self.simulation, "close", None)
            if callable(close):
                close()
            self.simulation = self._new_simulation()
            self._set_initial_conditions()
            # The normal first public reset follows construction's initial
            # RunIC with a second RunIC.  Preserve that sequence after
            # rebuilding so reset(seed=...) has the same hidden integrator
            # state as the first case.
            self.simulation.run_ic()

        if initial_conditions:
            for property_name, value in initial_conditions.items():
                self.simulation.set_property_value(str(property_name), float(value))

        self.simulation.run_ic()
        self.simulation.set_property_value("propulsion/set-running", -1)
        self.wind_state.fill(0.0)
        self.time_sec = 0.0
        for property_name in (
            "atmosphere/wind-north-fps",
            "atmosphere/wind-east-fps",
            "atmosphere/wind-down-fps",
        ):
            self.simulation.set_property_value(property_name, 0.0)
        self._get_state()

    def _get_state(self) -> np.ndarray:
        """Read the compact source-compatible state vector."""

        for index, property_name in enumerate(STATE_FORMAT):
            self.state[index] = self.simulation.get_property_value(property_name)
        self.state[:2] *= RADIUS_M
        return self.state

    def set_rng(self, rng: np.random.RandomState | np.random.Generator) -> None:
        self.rng = rng

    def _north_ft(self) -> float:
        latitude_rad = self.simulation.get_property_value("position/lat-gc-rad")
        return float(latitude_rad * RADIUS_FT)

    def step_simulation(self, action: np.ndarray, canyon: Any | None = None) -> np.ndarray:
        """Advance four JSBSim ticks with source-compatible controls and wind.

        ``canyon`` is queried only for its local width, which keeps the
        geometry-dependent gust scaling identical to DataCollectionEnv while
        avoiding any dataset row creation.
        """

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError("action must contain roll, pitch, yaw and throttle")
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        self.simulation.set_property_value("fcs/aileron-cmd-norm", float(roll_cmd))
        self.simulation.set_property_value("fcs/elevator-cmd-norm", float(pitch_cmd))
        self.simulation.set_property_value("fcs/rudder-cmd-norm", float(yaw_cmd))
        self.simulation.set_property_value("fcs/throttle-cmd-norm", float(throttle))

        for _ in range(self.down_sample):
            # Freeze fuel consumption and hold the landing gear retracted, as
            # in the source DataCollectionEnv.
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000.0)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000.0)
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)

            width_ft = 1000.0
            if canyon is not None and hasattr(canyon, "get_geometry"):
                width_ft = float(canyon.get_geometry(self._north_ft())[0])
            scale = 1000.0 / max(width_ft, 50.0)

            # Ornstein-Uhlenbeck gust process.  The source used a global
            # np.random stream; this per-case RandomState makes reset(seed)
            # reproducible and prevents one environment from perturbing another.
            noise = self.rng.standard_normal(3)
            self.wind_state += (
                -self.wind_theta * self.wind_state * self.dt
                + self.wind_sigma * scale * np.sqrt(self.dt) * noise
            )
            self.simulation.set_property_value(
                "atmosphere/wind-north-fps", float(self.wind_state[0])
            )
            self.simulation.set_property_value(
                "atmosphere/wind-east-fps", float(self.wind_state[1])
            )
            self.simulation.set_property_value(
                "atmosphere/wind-down-fps", float(self.wind_state[2])
            )
            self.simulation.run()
            self.time_sec += self.dt

        return self._get_state()

    def close(self) -> None:
        """Release the JSBSim executive if its binding exposes a close hook."""

        close = getattr(self.simulation, "close", None)
        if callable(close):
            close()


__all__ = [
    "ASSETS_ROOT", "DemoCancelled", "F16Simulation", "RADIUS_FT", "RADIUS_M", "STATE_FORMAT",
]
