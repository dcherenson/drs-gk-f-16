"""Portable empirical uncertainty sampling for the public F-16 demo.

The training-time calibration artifact is intentionally not imported at runtime.
``uncertainty.npz`` contains the learned context rows, the scaler, and the
width-sorted arrays used by the JAX sampler.  The runtime therefore needs only
NumPy and JAX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


CANONICAL_FEATURE_NAMES = (
    "alpha",
    "beta",
    "mach",
    "p",
    "q",
    "r",
    "delta_t",
    "delta_e",
    "delta_a",
    "delta_r",
    "prev_delta_t",
    "prev_delta_e",
    "prev_delta_a",
    "prev_delta_r",
    "qbar",
    "canyon_width",
    "canyon_width_grad",
)
RESIDUAL_NAMES = ("w_C_X", "w_C_Y", "w_C_Z", "w_C_L", "w_C_M", "w_C_N")
LEGACY_DROPPED_FEATURE_NAMES = ("alpha_dot", "wind_u", "wind_v", "wind_w")
CANYON_WIDTH_FEATURE_INDEX = CANONICAL_FEATURE_NAMES.index("canyon_width")


@jax.tree_util.register_pytree_node_class
class JAXEmpiricalData(NamedTuple):
    """Device-ready empirical uncertainty data consumed by the Gatekeeper."""

    feature_mean: jax.Array
    feature_std: jax.Array
    active_feature_indices: jax.Array
    sorted_features_scaled: jax.Array
    sorted_residuals: jax.Array
    sorted_alpha_values: jax.Array
    sorted_canyon_width_values: jax.Array
    canyon_width_feature_index: int
    neighbor_count: int
    max_pool_size: int
    epsilon_w: float
    alpha_threshold: float

    def tree_flatten(self):
        children = (
            self.feature_mean,
            self.feature_std,
            self.active_feature_indices,
            self.sorted_features_scaled,
            self.sorted_residuals,
            self.sorted_alpha_values,
            self.sorted_canyon_width_values,
        )
        static = (
            int(self.canyon_width_feature_index),
            int(self.neighbor_count),
            int(self.max_pool_size),
            float(self.epsilon_w),
            float(self.alpha_threshold),
        )
        return children, static

    @classmethod
    def tree_unflatten(cls, static, children):
        return cls(*children, *static)


def sample_empirical_jax(
    feature_vector: jax.Array,
    gate_id: jax.Array,
    rng_key: jax.Array,
    data: JAXEmpiricalData,
) -> jax.Array:
    """Sample one learned coefficient residual for a packed 17-feature query.

    The feature callback in the public demo already emits the canonical packed
    order in :data:`CANONICAL_FEATURE_NAMES`.  In particular, it is not a
    21-feature vector with holes for the four training-only columns.  Keeping
    this function identity-shaped avoids the old ``[0..14, 19, 20]`` gather
    bug, which indexed beyond the packed callback output.
    """

    del gate_id

    # ``active_feature_indices`` remains in the NamedTuple for compatibility
    # with the gatekeeper API.  It is the identity vector in this portable
    # format and must not be applied to an already-packed callback vector.
    feature_vector = jnp.asarray(feature_vector, dtype=jnp.float32)
    normalized_q = (feature_vector - data.feature_mean) / data.feature_std
    query_width = feature_vector[data.canyon_width_feature_index]

    total_samples = data.sorted_canyon_width_values.shape[0]
    lower = query_width - jnp.float32(data.epsilon_w)
    upper = query_width + jnp.float32(data.epsilon_w)

    lo = jnp.searchsorted(data.sorted_canyon_width_values, lower, side="left")
    hi = jnp.searchsorted(data.sorted_canyon_width_values, upper, side="right")
    width_count = hi - lo

    # If width gating returns no rows, use a centered local pool in sorted
    # width order.  This is the same fallback used by the training artifact.
    center = jnp.searchsorted(data.sorted_canyon_width_values, query_width, side="left")
    half_pool = jnp.int32(data.max_pool_size // 2)
    fallback_lo = jnp.clip(
        center - half_pool,
        0,
        jnp.maximum(total_samples - data.max_pool_size, 0),
    )
    fallback_count = jnp.minimum(jnp.int32(data.max_pool_size), total_samples - fallback_lo)

    pool_start = jnp.where(width_count > 0, lo, fallback_lo)
    pool_count = jnp.where(width_count > 0, width_count, fallback_count)
    pool_count = jnp.minimum(pool_count, jnp.int32(data.max_pool_size))

    offsets = jnp.arange(data.max_pool_size, dtype=jnp.int32)
    pool_indices = pool_start + offsets
    pool_indices_clipped = jnp.clip(pool_indices, 0, total_samples - 1)
    valid_mask = offsets < pool_count

    pool_features = data.sorted_features_scaled[pool_indices_clipped]
    pool_alpha = data.sorted_alpha_values[pool_indices_clipped]
    diffs = pool_features - normalized_q
    dists_sq = jnp.sum(jnp.square(diffs), axis=-1)

    regime_mask = jnp.abs(pool_alpha) < jnp.float32(data.alpha_threshold)
    valid_regime_mask = jnp.logical_and(valid_mask, regime_mask)
    has_valid_regime = jnp.any(valid_regime_mask)
    active_mask = jnp.where(has_valid_regime, valid_regime_mask, valid_mask)

    penalized_dists = jnp.where(active_mask, dists_sq, 1.0e12)
    vals, local_indices = jax.lax.top_k(-penalized_dists, data.neighbor_count)
    distances = -vals

    # Distance-weighted categorical sampling, retained from the calibrated
    # runtime implementation.  The small floor also handles exact matches.
    scale = jnp.mean(distances) + 1.0e-6
    logits = -distances / scale
    choice_local = jax.random.categorical(rng_key, logits)
    return data.sorted_residuals[pool_indices_clipped[local_indices[choice_local]]]


def _as_string_tuple(values: np.ndarray, *, key: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "SU":
        raise ValueError(f"uncertainty artifact field {key!r} must be a 1-D string array")
    return tuple(str(value) for value in array.tolist())


def _finite_array(values: np.ndarray, *, key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if shape is not None and array.shape != shape:
        raise ValueError(f"uncertainty artifact field {key!r} has shape {array.shape}; expected {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"uncertainty artifact field {key!r} must contain finite numeric values")
    return np.array(array, copy=True)


def _setting_value(settings: Mapping[str, object], names: tuple[str, ...], default: object) -> object:
    present = [name for name in names if name in settings]
    if len(present) > 1:
        values = [settings[name] for name in present]
        if any(value != values[0] for value in values[1:]):
            raise TypeError(f"conflicting sampler settings: {', '.join(present)}")
    return settings[present[0]] if present else default


def _normalise_settings(settings: Mapping[str, object]) -> dict[str, float | int]:
    known = {
        "neighbor_count",
        "max_pool_size",
        "epsilon_w",
        "alpha_threshold",
    }
    unknown = sorted(set(settings) - known)
    if unknown:
        raise TypeError(f"unexpected sampler setting(s): {', '.join(unknown)}")

    neighbor_count = int(_setting_value(settings, ("neighbor_count",), 20))
    max_pool_size = int(_setting_value(settings, ("max_pool_size",), 512))
    epsilon_w = float(_setting_value(settings, ("epsilon_w",), 200.0))
    alpha_threshold = float(_setting_value(settings, ("alpha_threshold",), 0.8))
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be >= 1")
    if max_pool_size < 1:
        raise ValueError("max_pool_size must be >= 1")
    if not np.isfinite(epsilon_w) or epsilon_w < 0.0:
        raise ValueError("epsilon_w must be finite and >= 0")
    if not np.isfinite(alpha_threshold) or alpha_threshold <= 0.0:
        raise ValueError("alpha_threshold must be finite and > 0")
    return {
        "neighbor_count": neighbor_count,
        "max_pool_size": max_pool_size,
        "epsilon_w": epsilon_w,
        "alpha_threshold": alpha_threshold,
    }


class RuntimeUncertaintySampler:
    """Load and sample the portable learned uncertainty artifact.

    ``artifact_path=None`` selects the packaged ``assets/models/uncertainty.npz``.
    Settings may be overridden at construction with ``neighbor_count``,
    ``max_pool_size``, ``epsilon_w``, and ``alpha_threshold``.
    """

    def __init__(self, artifact_path: str | Path | None = None, **sampler_settings: object):
        self._settings = _normalise_settings(sampler_settings)
        self.artifact_path = (
            Path(artifact_path).expanduser()
            if artifact_path is not None
            else Path(__file__).resolve().parent / "assets" / "models" / "uncertainty.npz"
        )
        self._load_artifact(self.artifact_path)
        self._active_feature_names: list[str] | None = None

    def _load_artifact(self, artifact_path: Path) -> None:
        required = {
            "format_version",
            "source_feature_names",
            "feature_names",
            "residual_names",
            "removed_feature_names",
            "row_count",
            "feature_mean",
            "feature_std",
            "sorted_features_scaled",
            "sorted_residuals",
            "sorted_alpha",
            "sorted_canyon_width",
        }
        try:
            loaded = np.load(artifact_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"missing portable uncertainty artifact: {artifact_path}") from exc
        with loaded as data:
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"uncertainty artifact is missing required fields: {missing}")

            version = _as_string_tuple(data["format_version"], key="format_version")
            if version != ("drs-f16-uncertainty-v1",):
                raise ValueError(f"unsupported uncertainty artifact format: {version}")

            source_features = _as_string_tuple(data["source_feature_names"], key="source_feature_names")
            feature_names = _as_string_tuple(data["feature_names"], key="feature_names")
            residual_names = _as_string_tuple(data["residual_names"], key="residual_names")
            removed_feature_names = _as_string_tuple(data["removed_feature_names"], key="removed_feature_names")
            expected_source_features = (
                *CANONICAL_FEATURE_NAMES[:15],
                *LEGACY_DROPPED_FEATURE_NAMES,
                *CANONICAL_FEATURE_NAMES[15:],
            )
            if source_features != expected_source_features:
                raise ValueError(
                    "uncertainty artifact source feature order mismatch: "
                    f"expected {expected_source_features}, got {source_features}"
                )
            if feature_names != CANONICAL_FEATURE_NAMES:
                raise ValueError(
                    "uncertainty artifact feature order mismatch: "
                    f"expected {CANONICAL_FEATURE_NAMES}, got {feature_names}"
                )
            if residual_names != RESIDUAL_NAMES:
                raise ValueError(
                    f"uncertainty artifact residual order mismatch: expected {RESIDUAL_NAMES}, got {residual_names}"
                )
            if removed_feature_names != LEGACY_DROPPED_FEATURE_NAMES:
                raise ValueError(
                    "uncertainty artifact removed feature order mismatch: "
                    f"expected {LEGACY_DROPPED_FEATURE_NAMES}, got {removed_feature_names}"
                )
            if not set(LEGACY_DROPPED_FEATURE_NAMES).issubset(set(source_features)):
                raise ValueError("uncertainty artifact source feature metadata is incomplete")

            row_count_array = np.asarray(data["row_count"])
            if row_count_array.shape != (1,) or not np.issubdtype(row_count_array.dtype, np.integer):
                raise ValueError("uncertainty artifact row_count must be a one-element integer array")
            row_count = int(row_count_array[0])
            if row_count < 1:
                raise ValueError("uncertainty artifact row_count must be positive")

            feature_mean = _finite_array(data["feature_mean"], key="feature_mean", shape=(len(feature_names),)).astype(np.float32)
            feature_std = _finite_array(data["feature_std"], key="feature_std", shape=(len(feature_names),)).astype(np.float32)
            if np.any(feature_std <= 0.0):
                raise ValueError("uncertainty artifact feature_std must be strictly positive")

            sorted_features_scaled = _finite_array(
                data["sorted_features_scaled"],
                key="sorted_features_scaled",
                shape=(row_count, len(feature_names)),
            ).astype(np.float32)
            sorted_residuals = _finite_array(
                data["sorted_residuals"],
                key="sorted_residuals",
                shape=(row_count, len(residual_names)),
            ).astype(np.float32)
            sorted_alpha = _finite_array(data["sorted_alpha"], key="sorted_alpha", shape=(row_count,)).astype(np.float32)
            sorted_canyon_width = _finite_array(
                data["sorted_canyon_width"],
                key="sorted_canyon_width",
                shape=(row_count,),
            ).astype(np.float32)
            if np.any(np.diff(sorted_canyon_width) < 0.0):
                raise ValueError("uncertainty artifact sorted_canyon_width must be nondecreasing")

        self.aug_features = list(CANONICAL_FEATURE_NAMES)
        self.residual_columns = list(RESIDUAL_NAMES)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.sorted_features_scaled = sorted_features_scaled
        self.sorted_residuals = sorted_residuals
        self.sorted_alpha_values = sorted_alpha
        self.sorted_canyon_width_values = sorted_canyon_width
        self.row_count = int(row_count)

    @property
    def neighbor_count(self) -> int:
        return int(self._settings["neighbor_count"])

    @property
    def max_pool_size(self) -> int:
        return int(self._settings["max_pool_size"])

    @property
    def epsilon_w(self) -> float:
        return float(self._settings["epsilon_w"])

    @property
    def alpha_threshold(self) -> float:
        return float(self._settings["alpha_threshold"])

    def configure_active_features(
        self,
        drop_feature_names=LEGACY_DROPPED_FEATURE_NAMES,
        drop_constant_features: bool = True,
    ) -> list[str]:
        """Return the canonical packed feature order used by the callback.

        The two parameters are retained for integration compatibility.  The
        four default dropped names are already absent from the portable
        artifact, and all 17 canonical names are retained even when a caller
        asks for constant-feature pruning: the learned model and callback
        have one fixed schema.
        """

        del drop_constant_features
        requested = set(drop_feature_names or ())
        unexpected = sorted(requested - set(LEGACY_DROPPED_FEATURE_NAMES))
        if unexpected:
            raise ValueError(
                "portable uncertainty sampler has a fixed 17-feature schema; "
                f"cannot drop {unexpected}"
            )
        self._active_feature_names = list(CANONICAL_FEATURE_NAMES)
        return list(self._active_feature_names)

    def to_jax(
        self,
        neighbor_count: int | None = None,
        max_pool_size: int | None = None,
        epsilon_w: float | None = None,
        alpha_threshold: float | None = None,
    ) -> JAXEmpiricalData:
        """Return sorted learned arrays and effective sampler settings for JAX."""

        if self._active_feature_names is None:
            self.configure_active_features()
        overrides: dict[str, object] = {}
        if neighbor_count is not None:
            overrides["neighbor_count"] = neighbor_count
        if max_pool_size is not None:
            overrides["max_pool_size"] = max_pool_size
        if epsilon_w is not None:
            overrides["epsilon_w"] = epsilon_w
        if alpha_threshold is not None:
            overrides["alpha_threshold"] = alpha_threshold
        settings = _normalise_settings({**self._settings, **overrides})

        max_neighbors = min(int(settings["neighbor_count"]), self.row_count)
        max_pool = min(max(int(settings["max_pool_size"]), max_neighbors), self.row_count)
        return JAXEmpiricalData(
            feature_mean=jnp.asarray(self.feature_mean, dtype=jnp.float32),
            feature_std=jnp.asarray(self.feature_std, dtype=jnp.float32),
            # The callback vector is already packed in this order.  Keep the
            # field for the gatekeeper NamedTuple contract, but make the
            # identity explicit to catch accidental schema drift in tests.
            active_feature_indices=jnp.arange(len(CANONICAL_FEATURE_NAMES), dtype=jnp.int32),
            sorted_features_scaled=jnp.asarray(self.sorted_features_scaled, dtype=jnp.float32),
            sorted_residuals=jnp.asarray(self.sorted_residuals, dtype=jnp.float32),
            sorted_alpha_values=jnp.asarray(self.sorted_alpha_values, dtype=jnp.float32),
            sorted_canyon_width_values=jnp.asarray(self.sorted_canyon_width_values, dtype=jnp.float32),
            canyon_width_feature_index=CANYON_WIDTH_FEATURE_INDEX,
            neighbor_count=max_neighbors,
            max_pool_size=max_pool,
            epsilon_w=float(settings["epsilon_w"]),
            alpha_threshold=float(settings["alpha_threshold"]),
        )



__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "JAXEmpiricalData",
    "RESIDUAL_NAMES",
    "RuntimeUncertaintySampler",
    "sample_empirical_jax",
]
