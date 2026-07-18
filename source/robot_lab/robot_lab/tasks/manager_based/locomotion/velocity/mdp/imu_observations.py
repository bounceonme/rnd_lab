"""CMP10A-derived policy observations for the opt-in RND STEP task."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import math as math_utils


_REPO_ROOT = Path(__file__).resolve().parents[8]
RND_CMP10A_OBSERVATION_MODEL_PATH = _REPO_ROOT / "scripts" / "tools" / "config" / "rnd_cmp10a_runtime.json"

_MODEL_TYPE = "rnd_cmp10a_policy_observation"
_POLICY_HZ = 50.0
_REQUIRED_QUALITY_GATES = (
    "promotion_pass",
    "static_runtime_gate_pass",
    "static_mount_axis_gate_pass",
    "dynamic_communication_gate_pass",
    "dynamic_consistency_gate_pass",
)
_EXPECTED_RANGES = {
    "gyro delay": (0.0, 0.005),
    "gyro residual bias": (-0.01, 0.01),
    "gyro white-noise sigma": (0.0003, 0.003),
    "gravity delay": (0.0, 0.020),
    "gravity tangent-noise sigma": (0.00005, 0.002),
}
_UNSEEN_STEP = torch.iinfo(torch.int64).min


class RndCmp10aObservationModelError(ValueError):
    """Raised when a CMP10A observation runtime model is incomplete or unsafe."""


@dataclass(frozen=True)
class RndCmp10aObservationModel:
    """Validated simulation envelope used by the policy observation terms."""

    path: Path
    policy_hz: float
    policy_angular_velocity_scale: float
    gyro_delay_range_s: tuple[float, float]
    gyro_bias_range_rad_s: tuple[float, float]
    gyro_noise_sigma_range_rad_s: tuple[float, float]
    gravity_delay_range_s: tuple[float, float]
    gravity_noise_sigma_range_rad: tuple[float, float]
    sensor_to_base_matrix: tuple[tuple[float, float, float], ...]


def _find_named_value(data: Mapping[str, Any], aliases: tuple[str, ...], *, max_depth: int) -> tuple[Any, str] | None:
    for alias in aliases:
        if alias in data:
            return data[alias], alias
    if max_depth <= 0:
        return None
    for key, value in data.items():
        if not isinstance(value, Mapping):
            continue
        found = _find_named_value(value, aliases, max_depth=max_depth - 1)
        if found is not None:
            nested_value, nested_path = found
            return nested_value, f"{key}.{nested_path}"
    return None


def _coerce_range(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        for key in (
            "range",
            "bounds",
            "envelope",
            "range_s",
            "range_rad_s_per_axis",
            "sigma_range_rad_s",
            "sigma_range_rad",
        ):
            if key in value:
                return _coerce_range(value[key], f"{label}.{key}")
        for low_key, high_key in (
            ("minimum", "maximum"),
            ("min", "max"),
            ("low", "high"),
        ):
            if low_key in value and high_key in value:
                value = (value[low_key], value[high_key])
                break
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise RndCmp10aObservationModelError(
            f"{label} must be a two-value range or an object with minimum/maximum keys; got {value!r}."
        )
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise RndCmp10aObservationModelError(f"{label} contains a non-numeric endpoint: {value!r}.") from error
    if not all(math.isfinite(endpoint) for endpoint in result):
        raise RndCmp10aObservationModelError(f"{label} endpoints must be finite; got {result}.")
    if result[0] > result[1]:
        raise RndCmp10aObservationModelError(f"{label} minimum exceeds its maximum: {result}.")
    return result


def _coerce_rotation_matrix(value: Any, label: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise RndCmp10aObservationModelError(f"{label} must be a 3x3 matrix; got {value!r}.")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise RndCmp10aObservationModelError(f"{label}[{row_index}] must contain three values; got {row!r}.")
        try:
            numeric_row = tuple(float(item) for item in row)
        except (TypeError, ValueError) as error:
            raise RndCmp10aObservationModelError(
                f"{label}[{row_index}] contains a non-numeric value: {row!r}."
            ) from error
        if not all(math.isfinite(item) for item in numeric_row):
            raise RndCmp10aObservationModelError(f"{label}[{row_index}] must contain finite values.")
        rows.append(numeric_row)

    matrix = tuple(rows)
    for row_index in range(3):
        for other_index in range(3):
            dot = sum(matrix[row_index][column] * matrix[other_index][column] for column in range(3))
            expected = 1.0 if row_index == other_index else 0.0
            if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=1.0e-6):
                raise RndCmp10aObservationModelError(f"{label} must be orthonormal.")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise RndCmp10aObservationModelError(f"{label} must be a proper rotation with determinant +1.")
    return matrix


def _validate_distribution(value: Any, label: str, expected: tuple[str, ...]) -> None:
    if not isinstance(value, Mapping) or "distribution" not in value:
        return
    distribution = str(value["distribution"]).strip().lower().replace("-", "_").replace(" ", "_")
    if distribution not in expected:
        raise RndCmp10aObservationModelError(
            f"{label}.distribution must be one of {expected!r}; got {value['distribution']!r}."
        )


def _required_range(
    simulation: Mapping[str, Any],
    channel_section: Mapping[str, Any] | None,
    *,
    label: str,
    local_aliases: tuple[str, ...],
    top_level_aliases: tuple[str, ...],
    distributions: tuple[str, ...],
) -> tuple[float, float]:
    found = None
    if channel_section is not None:
        found = _find_named_value(channel_section, local_aliases, max_depth=2)
    if found is None:
        found = _find_named_value(simulation, top_level_aliases, max_depth=2)
    if found is None:
        aliases = ", ".join((*local_aliases, *top_level_aliases))
        raise RndCmp10aObservationModelError(
            f"simulation is missing the {label} range. Expected one of these keys: {aliases}."
        )
    value, key_path = found
    _validate_distribution(value, f"simulation.{key_path}", distributions)
    actual = _coerce_range(value, f"simulation.{key_path}")
    expected = _EXPECTED_RANGES[label]
    if not all(math.isclose(item, target, rel_tol=0.0, abs_tol=1.0e-12) for item, target in zip(actual, expected)):
        raise RndCmp10aObservationModelError(
            f"simulation.{key_path} must be the supported {label} envelope {expected}; got {actual}."
        )
    return actual


def load_rnd_cmp10a_observation_model(
    path: str | Path = RND_CMP10A_OBSERVATION_MODEL_PATH,
) -> RndCmp10aObservationModel:
    """Load and strictly validate the simulation-facing CMP10A model slice."""

    model_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RndCmp10aObservationModelError(f"CMP10A observation model does not exist: {model_path}") from error
    except json.JSONDecodeError as error:
        raise RndCmp10aObservationModelError(
            f"CMP10A observation model is not valid JSON: {model_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RndCmp10aObservationModelError(f"CMP10A observation model root must be an object: {model_path}")
    if payload.get("schema_version") != 1:
        raise RndCmp10aObservationModelError("CMP10A observation model must use schema_version=1.")
    if payload.get("model_type") != _MODEL_TYPE:
        raise RndCmp10aObservationModelError(
            f"CMP10A observation model_type must be {_MODEL_TYPE!r}; got {payload.get('model_type')!r} in {model_path}."
        )
    if payload.get("integration_enabled") is not True:
        raise RndCmp10aObservationModelError("CMP10A observation model must set integration_enabled=true.")
    quality_gates = payload.get("quality_gates")
    if not isinstance(quality_gates, Mapping):
        raise RndCmp10aObservationModelError("CMP10A observation model requires a quality_gates object.")
    failed_gates = [gate for gate in _REQUIRED_QUALITY_GATES if quality_gates.get(gate) is not True]
    if failed_gates:
        raise RndCmp10aObservationModelError(
            "CMP10A observation model cannot be consumed because these quality gates are not true: "
            + ", ".join(failed_gates)
            + "."
        )

    policy_hz_entry = _find_named_value(payload, ("policy_hz",), max_depth=2)
    if policy_hz_entry is None:
        raise RndCmp10aObservationModelError(f"CMP10A observation model is missing policy_hz: {model_path}")
    try:
        policy_hz = float(policy_hz_entry[0])
    except (TypeError, ValueError) as error:
        raise RndCmp10aObservationModelError(
            f"CMP10A observation model policy_hz must be numeric; got {policy_hz_entry[0]!r}."
        ) from error
    if not math.isfinite(policy_hz) or not math.isclose(policy_hz, _POLICY_HZ, rel_tol=0.0, abs_tol=1.0e-6):
        raise RndCmp10aObservationModelError(
            f"CMP10A observation model policy_hz must be {_POLICY_HZ}; got {policy_hz}."
        )
    scale_entry = payload.get("policy_angular_velocity_scale")
    policy_observation = payload.get("policy_observation")
    nested_scale = policy_observation.get("angular_velocity_scale") if isinstance(policy_observation, Mapping) else None
    try:
        policy_scale = float(scale_entry)
        nested_policy_scale = float(nested_scale)
    except (TypeError, ValueError) as error:
        raise RndCmp10aObservationModelError(
            "CMP10A observation model requires numeric policy_angular_velocity_scale and "
            "policy_observation.angular_velocity_scale fields."
        ) from error
    if not math.isfinite(policy_scale) or policy_scale <= 0.0:
        raise RndCmp10aObservationModelError(
            f"CMP10A policy angular-velocity scale must be finite and positive; got {policy_scale}."
        )
    if not math.isclose(policy_scale, nested_policy_scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise RndCmp10aObservationModelError("Duplicated CMP10A policy angular-velocity scales disagree.")

    simulation_entry = _find_named_value(
        payload,
        ("simulation", "simulation_envelope", "assumed_simulation_envelopes", "simulator"),
        max_depth=1,
    )
    if simulation_entry is None or not isinstance(simulation_entry[0], Mapping):
        raise RndCmp10aObservationModelError(
            "CMP10A observation model requires a simulation object (accepted aliases: "
            "simulation, simulation_envelope, assumed_simulation_envelopes, simulator)."
        )
    simulation = simulation_entry[0]
    gyro_entry = _find_named_value(simulation, ("gyro", "angular_velocity", "base_ang_vel"), max_depth=2)
    gravity_entry = _find_named_value(simulation, ("gravity", "projected_gravity", "orientation"), max_depth=2)
    gyro_section = gyro_entry[0] if gyro_entry is not None and isinstance(gyro_entry[0], Mapping) else None
    gravity_section = gravity_entry[0] if gravity_entry is not None and isinstance(gravity_entry[0], Mapping) else None

    gyro_delay = _required_range(
        simulation,
        gyro_section,
        label="gyro delay",
        local_aliases=("delay_s", "delay_range_s", "fractional_delay_s"),
        top_level_aliases=(
            "gyro_sample_age_delay",
            "gyro_delay_s",
            "gyro_delay_range_s",
            "angular_velocity_delay_s",
            "base_ang_vel_delay_s",
        ),
        distributions=("uniform",),
    )
    gyro_bias = _required_range(
        simulation,
        gyro_section,
        label="gyro residual bias",
        local_aliases=("residual_bias_rad_s", "residual_bias_range_rad_s", "bias_rad_s"),
        top_level_aliases=(
            "residual_gyro_episode_bias",
            "gyro_residual_bias_rad_s",
            "gyro_residual_bias_range_rad_s",
            "angular_velocity_residual_bias_rad_s",
        ),
        distributions=("uniform",),
    )
    gyro_noise = _required_range(
        simulation,
        gyro_section,
        label="gyro white-noise sigma",
        local_aliases=(
            "white_noise_sigma_rad_s",
            "white_noise_sigma_range_rad_s",
            "white_noise_std_rad_s",
            "noise_sigma_rad_s",
        ),
        top_level_aliases=(
            "gyro_white_noise",
            "gyro_white_noise_sigma_rad_s",
            "gyro_white_noise_sigma_range_rad_s",
            "gyro_white_noise_std_rad_s",
        ),
        distributions=("log_uniform", "zero_mean_gaussian"),
    )
    gravity_delay = _required_range(
        simulation,
        gravity_section,
        label="gravity delay",
        local_aliases=("delay_s", "delay_range_s", "fractional_delay_s"),
        top_level_aliases=(
            "orientation_delay",
            "gravity_delay_s",
            "gravity_delay_range_s",
            "projected_gravity_delay_s",
            "orientation_delay_s",
        ),
        distributions=("uniform",),
    )
    gravity_noise = _required_range(
        simulation,
        gravity_section,
        label="gravity tangent-noise sigma",
        local_aliases=(
            "tangent_noise_sigma_rad",
            "tangent_noise_sigma_range_rad",
            "tangent_noise_std_rad",
            "angular_noise_sigma_rad",
        ),
        top_level_aliases=(
            "projected_gravity_tangent_angle_noise",
            "gravity_tangent_noise_sigma_rad",
            "gravity_tangent_noise_sigma_range_rad",
            "projected_gravity_tangent_noise_sigma_rad",
            "orientation_noise_sigma_rad",
        ),
        distributions=("log_uniform", "zero_mean_tangent_plane_gaussian"),
    )
    runtime_transform = payload.get("runtime_transform")
    if not isinstance(runtime_transform, Mapping):
        raise RndCmp10aObservationModelError("CMP10A observation model requires a runtime_transform object.")
    if runtime_transform.get("source_frame") != "sensor" or runtime_transform.get("target_frame") != "base_link":
        raise RndCmp10aObservationModelError(
            "CMP10A runtime_transform must map source_frame='sensor' to target_frame='base_link'."
        )
    sensor_to_base_matrix = _coerce_rotation_matrix(
        runtime_transform.get("sensor_to_base_matrix"),
        "runtime_transform.sensor_to_base_matrix",
    )
    canonical_matrix = _coerce_rotation_matrix(payload.get("sensor_to_base_matrix"), "sensor_to_base_matrix")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual_row, expected_row in zip(sensor_to_base_matrix, canonical_matrix, strict=True)
        for actual, expected in zip(actual_row, expected_row, strict=True)
    ):
        raise RndCmp10aObservationModelError(
            "runtime_transform.sensor_to_base_matrix must match the canonical sensor_to_base_matrix."
        )
    return RndCmp10aObservationModel(
        path=model_path,
        policy_hz=policy_hz,
        policy_angular_velocity_scale=policy_scale,
        gyro_delay_range_s=gyro_delay,
        gyro_bias_range_rad_s=gyro_bias,
        gyro_noise_sigma_range_rad_s=gyro_noise,
        gravity_delay_range_s=gravity_delay,
        gravity_noise_sigma_range_rad=gravity_noise,
        sensor_to_base_matrix=sensor_to_base_matrix,
    )


def _fractional_delay_interpolate_unchecked(history: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    maximum_delay = history.shape[1] - 1
    delays = delays.clamp(0.0, float(maximum_delay))
    newer_index = torch.floor(delays).to(dtype=torch.long)
    older_index = (newer_index + 1).clamp(max=maximum_delay)
    batch_index = torch.arange(history.shape[0], device=history.device)
    newer = history[batch_index, newer_index]
    older = history[batch_index, older_index]
    fraction = (delays - newer_index.to(dtype=history.dtype)).unsqueeze(-1)
    return newer + fraction * (older - newer)


def fractional_delay_interpolate(history: torch.Tensor, delay_samples: torch.Tensor | float) -> torch.Tensor:
    """Interpolate current-first batched history at a fractional sample delay."""

    if history.ndim != 3 or history.shape[1] < 1:
        raise ValueError(f"history must have shape [num_envs, history_length, features]; got {history.shape}.")
    if not history.is_floating_point():
        raise TypeError("history must use a floating-point dtype.")
    delays = torch.as_tensor(delay_samples, dtype=history.dtype, device=history.device)
    if delays.ndim == 0:
        delays = delays.expand(history.shape[0])
    if delays.shape != (history.shape[0],):
        raise ValueError(f"delay_samples must have shape [{history.shape[0]}]; got {delays.shape}.")
    if not bool(torch.isfinite(delays).all()):
        raise ValueError("delay_samples must be finite.")
    maximum_delay = history.shape[1] - 1
    if bool(torch.any(delays < 0.0)) or bool(torch.any(delays > maximum_delay + 1.0e-6)):
        raise ValueError(f"delay_samples must lie in [0, {maximum_delay}]; got {delays}.")
    return _fractional_delay_interpolate_unchecked(history, delays)


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not vectors.is_floating_point():
        raise ValueError(f"vectors must be a floating-point tensor with shape [num_envs, 3]; got {vectors.shape}.")
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    epsilon = torch.finfo(vectors.dtype).eps
    normalized = vectors / norms.clamp_min(epsilon)
    fallback = torch.zeros_like(vectors)
    fallback[:, 2] = -1.0
    return torch.where(norms > epsilon, normalized, fallback)


def apply_tangent_plane_angular_noise(vectors: torch.Tensor, angular_noise: torch.Tensor) -> torch.Tensor:
    """Apply tangent-plane rotation vectors and return exactly renormalized vectors."""

    if angular_noise.shape != vectors.shape:
        raise ValueError(f"angular_noise must match vectors shape {vectors.shape}; got {angular_noise.shape}.")
    unit_vectors = _normalize_vectors(vectors)
    tangent = angular_noise - torch.sum(angular_noise * unit_vectors, dim=-1, keepdim=True) * unit_vectors
    angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    epsilon = torch.finfo(vectors.dtype).eps
    sin_over_angle = torch.sin(angle) / angle.clamp_min(epsilon)
    sin_over_angle = torch.where(angle > epsilon, sin_over_angle, torch.ones_like(angle))
    rotated = torch.cos(angle) * unit_vectors + sin_over_angle * tangent
    return _normalize_vectors(rotated)


def transform_sensor_vectors_to_base(vectors: torch.Tensor, sensor_to_base_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate batched sensor-frame vectors into the policy's base frame."""

    if vectors.ndim != 2 or vectors.shape[1] != 3 or not vectors.is_floating_point():
        raise ValueError(f"vectors must be a floating-point tensor with shape [num_envs, 3]; got {vectors.shape}.")
    if sensor_to_base_matrix.shape != (3, 3) or not sensor_to_base_matrix.is_floating_point():
        raise ValueError(
            "sensor_to_base_matrix must be a floating-point tensor with shape [3, 3]; "
            f"got {sensor_to_base_matrix.shape}."
        )
    if sensor_to_base_matrix.device != vectors.device or sensor_to_base_matrix.dtype != vectors.dtype:
        raise ValueError("sensor_to_base_matrix must use the same device and dtype as vectors.")
    return vectors @ sensor_to_base_matrix.transpose(0, 1)


class Cmp10aObservationState:
    """Pure-Torch, per-environment state for one simulated CMP10A channel."""

    def __init__(
        self,
        *,
        num_envs: int,
        channel: str,
        step_dt: float,
        delay_range_s: tuple[float, float],
        noise_sigma_range: tuple[float, float],
        bias_range: tuple[float, float] = (0.0, 0.0),
        sample_randomization: bool,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if channel not in ("gyro", "gravity"):
            raise ValueError(f"channel must be 'gyro' or 'gravity'; got {channel!r}.")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive; got {num_envs}.")
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(f"step_dt must be finite and positive; got {step_dt}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; got {dtype}.")
        self._validate_range(delay_range_s, "delay_range_s", minimum=0.0)
        self._validate_range(bias_range, "bias_range")
        self._validate_range(noise_sigma_range, "noise_sigma_range", minimum=0.0)
        if noise_sigma_range[0] == 0.0 and noise_sigma_range[1] > 0.0:
            raise ValueError("A log-uniform noise sigma range cannot start at zero.")

        self.num_envs = int(num_envs)
        self.channel = channel
        self.step_dt = float(step_dt)
        self.delay_range_s = tuple(float(value) for value in delay_range_s)
        self.bias_range = tuple(float(value) for value in bias_range)
        self.noise_sigma_range = tuple(float(value) for value in noise_sigma_range)
        self.sample_randomization = bool(sample_randomization)
        self.device = torch.device(device)
        self.dtype = dtype

        maximum_delay_samples = self.delay_range_s[1] / self.step_dt
        history_length = max(1, math.ceil(maximum_delay_samples - 1.0e-12) + 1)
        self.history = torch.zeros((self.num_envs, history_length, 3), device=self.device, dtype=self.dtype)
        self.delay_s = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.bias = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self.noise_sigma = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.cached_result = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self.last_step_counter = torch.full((self.num_envs,), _UNSEEN_STEP, device=self.device, dtype=torch.int64)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._has_pending_reset = False
        self._last_observed_step: int | None = None

    @staticmethod
    def _validate_range(values: tuple[float, float], label: str, minimum: float | None = None) -> None:
        if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{label} must contain two finite values; got {values}.")
        if values[0] > values[1]:
            raise ValueError(f"{label} minimum exceeds maximum: {values}.")
        if minimum is not None and values[0] < minimum:
            raise ValueError(f"{label} minimum must be at least {minimum}; got {values}.")

    def _raw_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        raw_tensor = torch.as_tensor(raw, device=self.device, dtype=self.dtype)
        if raw_tensor.shape != (self.num_envs, 3):
            raise ValueError(f"raw observation must have shape [{self.num_envs}, 3]; got {raw_tensor.shape}.")
        return raw_tensor

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        result = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if result.numel() == 0:
            return result
        if bool(torch.any(result < 0)) or bool(torch.any(result >= self.num_envs)):
            raise IndexError(f"env_ids must lie in [0, {self.num_envs - 1}]; got {result}.")
        return torch.unique(result, sorted=False)

    def _uniform(self, count: int, value_range: tuple[float, float], *, components: int | None = None):
        shape = (count,) if components is None else (count, components)
        low, high = value_range
        if low == high:
            return torch.full(shape, low, device=self.device, dtype=self.dtype)
        return torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(low, high)

    def _log_uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        if low == high:
            return torch.full((count,), low, device=self.device, dtype=self.dtype)
        return (
            torch.empty((count,), device=self.device, dtype=self.dtype).uniform_(math.log(low), math.log(high)).exp_()
        )

    def reset(self, raw: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Resample and refill only the selected environments."""

        raw_tensor = self._raw_tensor(raw)
        ids = self._env_ids(env_ids)
        if ids.numel() == 0:
            return
        count = ids.numel()
        if self.sample_randomization:
            self.delay_s[ids] = self._uniform(count, self.delay_range_s)
            self.noise_sigma[ids] = self._log_uniform(count, self.noise_sigma_range)
            if self.channel == "gyro":
                self.bias[ids] = self._uniform(count, self.bias_range, components=3)
            else:
                self.bias[ids] = 0.0
        else:
            self.delay_s[ids] = 0.5 * (self.delay_range_s[0] + self.delay_range_s[1])
            self.bias[ids] = 0.0
            self.noise_sigma[ids] = 0.0

        current = raw_tensor[ids]
        self.history[ids] = current.unsqueeze(1).expand(-1, self.history.shape[1], -1)
        self.cached_result[ids] = current + self.bias[ids] if self.channel == "gyro" else _normalize_vectors(current)
        self.last_step_counter[ids] = _UNSEEN_STEP
        self._pending_reset[ids] = True
        self._has_pending_reset = True

    def observe(self, raw: torch.Tensor, step_counter: int) -> torch.Tensor:
        """Advance each environment at most once for a given policy step."""

        raw_tensor = self._raw_tensor(raw)
        step = int(step_counter)
        if self._last_observed_step != step:
            update_ids = self._all_env_ids
        elif not self._has_pending_reset:
            return self.cached_result.clone()
        else:
            update_ids = torch.nonzero(self._pending_reset, as_tuple=False).flatten()

        if self.history.shape[1] > 1:
            self.history[update_ids, 1:] = self.history[update_ids, :-1].clone()
        self.history[update_ids, 0] = raw_tensor[update_ids]
        delayed = _fractional_delay_interpolate_unchecked(
            self.history[update_ids], self.delay_s[update_ids] / self.step_dt
        )
        if self.channel == "gyro":
            if self.sample_randomization:
                noise = torch.randn_like(delayed) * self.noise_sigma[update_ids].unsqueeze(-1)
            else:
                noise = torch.zeros_like(delayed)
            result = delayed + self.bias[update_ids] + noise
        else:
            delayed = _normalize_vectors(delayed)
            if self.sample_randomization:
                angular_noise = torch.randn_like(delayed) * self.noise_sigma[update_ids].unsqueeze(-1)
            else:
                angular_noise = torch.zeros_like(delayed)
            result = apply_tangent_plane_angular_noise(delayed, angular_noise)

        self.cached_result[update_ids] = result
        self.last_step_counter[update_ids] = step
        self._pending_reset[update_ids] = False
        self._has_pending_reset = False
        self._last_observed_step = step
        return self.cached_result.clone()


class RndCmp10aObservation(ManagerTermBase):
    """Read one IMU link and reproduce the base-frame CMP10A policy channel."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        channel = str(params["channel"])
        if channel not in ("gyro", "gravity"):
            raise ValueError(f"CMP10A observation channel must be 'gyro' or 'gravity'; got {channel!r}.")
        self._channel = channel
        self._sample_randomization = bool(params["sample_randomization"])
        self._model_path_param = str(params["model_path"])
        self._body_name = str(params["body_name"])
        asset = env.scene["robot"]
        body_ids, body_names = asset.find_bodies(self._body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(
                f"CMP10A body_name={self._body_name!r} must resolve to exactly one rigid body; matched {body_names!r}."
            )
        self._body_id = int(body_ids[0])
        self._model = load_rnd_cmp10a_observation_model(params["model_path"])
        body_quat_w = asset.data.body_quat_w
        self._sensor_to_base_matrix = torch.tensor(
            self._model.sensor_to_base_matrix,
            device=body_quat_w.device,
            dtype=body_quat_w.dtype,
        )
        self._validate_mount_alignment(asset)
        expected_scale = self._model.policy_angular_velocity_scale if channel == "gyro" else 1.0
        try:
            configured_scale = float(cfg.scale)
        except (TypeError, ValueError) as error:
            raise ValueError(f"CMP10A {channel} observation scale must be numeric; got {cfg.scale!r}.") from error
        if not math.isclose(configured_scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"CMP10A {channel} observation scale={configured_scale} does not match runtime model "
                f"scale={expected_scale}."
            )

        step_dt = float(env.step_dt)
        environment_policy_hz = 1.0 / step_dt
        if not math.isclose(environment_policy_hz, self._model.policy_hz, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"CMP10A model policy_hz={self._model.policy_hz} does not match 1/env.step_dt={environment_policy_hz}."
            )
        raw = self._raw_sensor(env)
        if channel == "gyro":
            delay_range = self._model.gyro_delay_range_s
            bias_range = self._model.gyro_bias_range_rad_s
            noise_range = self._model.gyro_noise_sigma_range_rad_s
        else:
            delay_range = self._model.gravity_delay_range_s
            bias_range = (0.0, 0.0)
            noise_range = self._model.gravity_noise_sigma_range_rad
        self._state = Cmp10aObservationState(
            num_envs=env.num_envs,
            channel=channel,
            step_dt=step_dt,
            delay_range_s=delay_range,
            bias_range=bias_range,
            noise_sigma_range=noise_range,
            sample_randomization=self._sample_randomization,
            device=env.device,
            dtype=raw.dtype,
        )
        self._state.reset(raw)

    @property
    def state(self) -> Cmp10aObservationState:
        """Expose state for diagnostics and focused unit tests."""

        return self._state

    @property
    def body_name(self) -> str:
        """Name of the rigid body used as the simulated sensor frame."""

        return self._body_name

    @property
    def body_id(self) -> int:
        """Resolved articulation body index used by the simulated sensor."""

        return self._body_id

    def _validate_mount_alignment(self, asset) -> None:
        data = asset.data
        sensor_quat_w = data.body_quat_w[0, self._body_id].unsqueeze(0).expand(3, -1)
        base_quat_w = data.root_link_quat_w[0].unsqueeze(0).expand(3, -1)
        sensor_basis = torch.eye(3, device=sensor_quat_w.device, dtype=sensor_quat_w.dtype)
        basis_w = math_utils.quat_apply(sensor_quat_w, sensor_basis)
        basis_b = math_utils.quat_apply_inverse(base_quat_w, basis_w)
        urdf_sensor_to_base = basis_b.transpose(0, 1)
        maximum_error = torch.max(torch.abs(urdf_sensor_to_base - self._sensor_to_base_matrix))
        if float(maximum_error) > 1.0e-3:
            raise ValueError(
                f"URDF body {self._body_name!r} orientation does not match the measured CMP10A sensor-to-base "
                f"transform (max_abs_error={float(maximum_error):.6f})."
            )

    def _raw_sensor(self, env) -> torch.Tensor:
        asset = env.scene["robot"]
        sensor_quat_w = asset.data.body_quat_w[:, self._body_id]
        if self._channel == "gyro":
            vector_w = asset.data.body_ang_vel_w[:, self._body_id]
        else:
            vector_w = asset.data.GRAVITY_VEC_W
        return math_utils.quat_apply_inverse(sensor_quat_w, vector_w)

    def _to_base(self, sensor_vectors: torch.Tensor) -> torch.Tensor:
        return transform_sensor_vectors_to_base(sensor_vectors, self._sensor_to_base_matrix)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        self._state.reset(self._raw_sensor(self._env), env_ids)

    def __call__(
        self,
        env,
        channel: str,
        model_path: str,
        sample_randomization: bool,
        body_name: str,
    ) -> torch.Tensor:
        if channel != self._channel:
            raise RuntimeError(
                f"CMP10A observation channel changed after initialization: {self._channel!r} -> {channel!r}."
            )
        if str(model_path) != self._model_path_param:
            raise RuntimeError("CMP10A observation model_path changed after term initialization.")
        if bool(sample_randomization) != self._sample_randomization:
            raise RuntimeError("CMP10A sample_randomization changed after term initialization.")
        if str(body_name) != self._body_name:
            raise RuntimeError("CMP10A body_name changed after term initialization.")
        sensor_observation = self._state.observe(self._raw_sensor(env), env.common_step_counter)
        return self._to_base(sensor_observation)
