"""MX-106 current-to-torque calibration and offline friction identification."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


TORQUE_LUT_SCHEMA_VERSION = 1
TORQUE_LUT_MODEL_TYPE = "mx106_performance_graph_current_to_output_torque_lut"


class TorqueIdentificationError(ValueError):
    """Raised when calibration or telemetry cannot support the requested fit."""


@dataclass(frozen=True)
class TorqueConversion:
    """Current-to-torque conversion with domain diagnostics."""

    torque_nm: np.ndarray
    below_observed_curve: np.ndarray
    above_observed_curve: np.ndarray


def _finite_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise TorqueIdentificationError(f"{name} must be a finite one-dimensional array with at least two values.")
    return array


def validate_torque_lut(model: dict[str, Any]) -> None:
    """Validate a digitized manufacturer performance-graph LUT."""

    if model.get("schema_version") != TORQUE_LUT_SCHEMA_VERSION:
        raise TorqueIdentificationError("Unsupported torque LUT schema version.")
    if model.get("model_type") != TORQUE_LUT_MODEL_TYPE:
        raise TorqueIdentificationError("Unsupported torque LUT model type.")
    if model.get("analysis_only") is not True:
        raise TorqueIdentificationError("Manufacturer-graph torque LUT must remain analysis_only=true.")

    curve = model.get("curve")
    if not isinstance(curve, dict):
        raise TorqueIdentificationError("Torque LUT curve block is missing.")
    torque_nm = _finite_array(curve.get("torque_nm"), "curve.torque_nm")
    current_a = _finite_array(curve.get("current_a"), "curve.current_a")
    if torque_nm.shape != current_a.shape:
        raise TorqueIdentificationError("Torque and current LUT arrays must have the same shape.")
    if np.any(torque_nm <= 0.0) or np.any(current_a <= 0.0):
        raise TorqueIdentificationError("Digitized performance-graph points must be positive.")
    if np.any(np.diff(torque_nm) <= 0.0) or np.any(np.diff(current_a) <= 0.0):
        raise TorqueIdentificationError("Torque and current LUT arrays must be strictly increasing.")

    conversion = model.get("conversion")
    if not isinstance(conversion, dict):
        raise TorqueIdentificationError("Torque LUT conversion contract is missing.")
    if conversion.get("sign_convention") != "odd_symmetric_about_zero":
        raise TorqueIdentificationError("Only odd-symmetric signed current conversion is supported.")
    if conversion.get("below_observed_curve") != "linear_origin_to_first_point_unvalidated":
        raise TorqueIdentificationError("Unsupported low-current extrapolation policy.")
    if conversion.get("above_observed_curve") != "clip_to_last_point":
        raise TorqueIdentificationError("Unsupported high-current policy.")


def load_torque_lut(path: str | Path) -> dict[str, Any]:
    """Load and validate a current-to-output-torque LUT."""

    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TorqueIdentificationError(f"Torque LUT does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise TorqueIdentificationError(f"Torque LUT is invalid JSON: {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise TorqueIdentificationError("Torque LUT must be a JSON object.")
    validate_torque_lut(value)
    return value


def current_to_output_torque(current_a: np.ndarray | float, model: dict[str, Any]) -> TorqueConversion:
    """Convert signed MX-106 current to approximate signed output-shaft torque.

    The official graph does not cover the low currents present in much of the
    suspended-leg telemetry. Values below the first digitized point are therefore
    explicitly marked as unvalidated linear extrapolation rather than evidence.
    """

    validate_torque_lut(model)
    current = np.asarray(current_a, dtype=np.float64)
    if not np.all(np.isfinite(current)):
        raise TorqueIdentificationError("Current samples must be finite.")
    curve_current = np.asarray(model["curve"]["current_a"], dtype=np.float64)
    curve_torque = np.asarray(model["curve"]["torque_nm"], dtype=np.float64)
    magnitude = np.abs(current)
    below = (magnitude > 0.0) & (magnitude < curve_current[0])
    above = magnitude > curve_current[-1]

    clipped = np.minimum(magnitude, curve_current[-1])
    torque_magnitude = np.interp(clipped, curve_current, curve_torque)
    low_slope = curve_torque[0] / curve_current[0]
    torque_magnitude = np.where(magnitude < curve_current[0], magnitude * low_slope, torque_magnitude)
    torque = np.copysign(torque_magnitude, current)
    torque = np.where(magnitude == 0.0, 0.0, torque)
    return TorqueConversion(torque_nm=torque, below_observed_curve=below, above_observed_curve=above)


def estimate_joint_kinematics(
    position_rad: np.ndarray,
    sample_hz: float,
    *,
    window_length: int = 11,
    polynomial_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate smooth joint velocity and acceleration from encoder position."""

    position = np.asarray(position_rad, dtype=np.float64)
    if position.ndim != 2 or position.shape[0] < 5 or not np.all(np.isfinite(position)):
        raise TorqueIdentificationError("position_rad must be a finite [samples, joints] matrix.")
    if not math.isfinite(sample_hz) or sample_hz <= 0.0:
        raise TorqueIdentificationError("sample_hz must be finite and positive.")
    if window_length % 2 != 1 or window_length <= polynomial_order or window_length > position.shape[0]:
        raise TorqueIdentificationError("Savitzky-Golay window must be odd, exceed the polynomial order, and fit.")
    if polynomial_order < 2:
        raise TorqueIdentificationError("polynomial_order must be at least two for acceleration estimation.")
    delta = 1.0 / float(sample_hz)
    velocity = savgol_filter(
        position,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=1,
        delta=delta,
        axis=0,
        mode="interp",
    )
    acceleration = savgol_filter(
        position,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=2,
        delta=delta,
        axis=0,
        mode="interp",
    )
    return velocity, acceleration


def fit_friction_residual(
    residual_torque_nm: np.ndarray,
    velocity_rad_s: np.ndarray,
    sample_mask: np.ndarray,
    *,
    transition_velocity_rad_s: float,
    minimum_speed_rad_s: float,
) -> dict[str, Any]:
    """Fit Coulomb, viscous, and constant-bias terms to dynamic residual torque."""

    residual = np.asarray(residual_torque_nm, dtype=np.float64)
    velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    mask = np.asarray(sample_mask, dtype=np.bool_)
    if residual.ndim != 1 or velocity.shape != residual.shape or mask.shape != residual.shape:
        raise TorqueIdentificationError("Residual, velocity, and sample mask must be matching one-dimensional arrays.")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(velocity)):
        raise TorqueIdentificationError("Residual and velocity samples must be finite.")
    if not math.isfinite(transition_velocity_rad_s) or transition_velocity_rad_s <= 0.0:
        raise TorqueIdentificationError("transition_velocity_rad_s must be finite and positive.")
    if not math.isfinite(minimum_speed_rad_s) or minimum_speed_rad_s < 0.0:
        raise TorqueIdentificationError("minimum_speed_rad_s must be finite and non-negative.")

    selected = mask & (np.abs(velocity) >= minimum_speed_rad_s)
    if np.count_nonzero(selected) < 20 or not np.any(velocity[selected] > 0.0) or not np.any(velocity[selected] < 0.0):
        raise TorqueIdentificationError("Friction fit requires at least 20 samples spanning both velocity directions.")
    fit_velocity = velocity[selected]
    fit_residual = residual[selected]
    direction = np.tanh(fit_velocity / transition_velocity_rad_s)

    initial_coulomb = 0.5 * (np.median(fit_residual[fit_velocity > 0.0]) - np.median(fit_residual[fit_velocity < 0.0]))
    initial = np.asarray([max(0.0, initial_coulomb), 0.0, float(np.median(fit_residual))])

    def error(parameters: np.ndarray) -> np.ndarray:
        coulomb, viscous, bias = parameters
        return coulomb * direction + viscous * fit_velocity + bias - fit_residual

    scale = max(float(np.median(np.abs(fit_residual - np.median(fit_residual)))), 1.0e-4)
    result = least_squares(
        error,
        initial,
        bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=scale,
    )
    coulomb, viscous, bias = (float(value) for value in result.x)
    prediction = coulomb * direction + viscous * fit_velocity + bias
    fit_error = fit_residual - prediction
    denominator = float(np.sum(np.square(fit_residual - np.mean(fit_residual))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(fit_error))) / denominator
    return {
        "law": "coulomb_nm * tanh(velocity / transition_velocity) + viscous_nm_per_rad_s * velocity + bias_nm",
        "sample_count": int(fit_residual.size),
        "coulomb_nm": coulomb,
        "viscous_nm_per_rad_s": viscous,
        "bias_nm": bias,
        "transition_velocity_rad_s": float(transition_velocity_rad_s),
        "minimum_speed_rad_s": float(minimum_speed_rad_s),
        "rmse_nm": float(np.sqrt(np.mean(np.square(fit_error)))),
        "r2": r2,
        "optimizer_success": bool(result.success),
    }


def fit_armature_residual(
    residual_torque_nm: np.ndarray,
    velocity_rad_s: np.ndarray,
    acceleration_rad_s2: np.ndarray,
    sample_mask: np.ndarray,
    *,
    transition_velocity_rad_s: float,
    maximum_armature_kg_m2: float = 0.1,
) -> dict[str, Any]:
    """Fit residual joint inertia together with simple friction and bias terms.

    ``residual_torque_nm`` must already have zero-armature URDF inertial,
    Coriolis, and gravity torque removed. The fitted armature is therefore a
    residual reflected inertia, not the complete joint inertia.
    """

    residual = np.asarray(residual_torque_nm, dtype=np.float64)
    velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    acceleration = np.asarray(acceleration_rad_s2, dtype=np.float64)
    mask = np.asarray(sample_mask, dtype=np.bool_)
    if not (
        residual.ndim == 1
        and velocity.shape == residual.shape
        and acceleration.shape == residual.shape
        and mask.shape == residual.shape
    ):
        raise TorqueIdentificationError(
            "Residual, velocity, acceleration, and sample mask must be matching one-dimensional arrays."
        )
    if not all(np.all(np.isfinite(values)) for values in (residual, velocity, acceleration)):
        raise TorqueIdentificationError("Armature-fit residual, velocity, and acceleration must be finite.")
    if not math.isfinite(transition_velocity_rad_s) or transition_velocity_rad_s <= 0.0:
        raise TorqueIdentificationError("transition_velocity_rad_s must be finite and positive.")
    if not math.isfinite(maximum_armature_kg_m2) or maximum_armature_kg_m2 <= 0.0:
        raise TorqueIdentificationError("maximum_armature_kg_m2 must be finite and positive.")

    selected = mask
    if np.count_nonzero(selected) < 100:
        raise TorqueIdentificationError("Armature fit requires at least 100 selected dynamic samples.")
    fit_residual = residual[selected]
    fit_velocity = velocity[selected]
    fit_acceleration = acceleration[selected]
    if not np.any(fit_velocity > 0.0) or not np.any(fit_velocity < 0.0):
        raise TorqueIdentificationError("Armature fit requires both velocity directions.")
    if not np.any(fit_acceleration > 0.0) or not np.any(fit_acceleration < 0.0):
        raise TorqueIdentificationError("Armature fit requires both acceleration directions.")

    direction = np.tanh(fit_velocity / transition_velocity_rad_s)
    design = np.column_stack((fit_acceleration, direction, fit_velocity, np.ones_like(fit_velocity)))
    column_norm = np.linalg.norm(design, axis=0)
    if np.any(column_norm <= 1.0e-12):
        raise TorqueIdentificationError("Armature regression contains a constant or unobservable regressor.")
    normalized_condition_number = float(np.linalg.cond(design / column_norm))

    linear_initial, *_ = np.linalg.lstsq(design, fit_residual, rcond=None)
    initial = np.asarray(
        [
            float(np.clip(linear_initial[0], 0.0, maximum_armature_kg_m2)),
            max(0.0, float(linear_initial[1])),
            max(0.0, float(linear_initial[2])),
            float(linear_initial[3]),
        ],
        dtype=np.float64,
    )

    def full_error(parameters: np.ndarray) -> np.ndarray:
        armature, coulomb, viscous, bias = parameters
        return armature * fit_acceleration + coulomb * direction + viscous * fit_velocity + bias - fit_residual

    scale = max(float(np.median(np.abs(fit_residual - np.median(fit_residual)))), 1.0e-4)
    result = least_squares(
        full_error,
        initial,
        bounds=([0.0, 0.0, 0.0, -np.inf], [maximum_armature_kg_m2, np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=scale,
    )
    armature, coulomb, viscous, bias = (float(value) for value in result.x)
    prediction = design @ result.x
    fit_error = fit_residual - prediction
    denominator = float(np.sum(np.square(fit_residual - np.mean(fit_residual))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(fit_error))) / denominator

    friction_design = design[:, 1:]
    friction_initial = initial[1:]
    friction_result = least_squares(
        lambda parameters: friction_design @ parameters - fit_residual,
        friction_initial,
        bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=scale,
    )
    friction_error = fit_residual - friction_design @ friction_result.x
    rmse = float(np.sqrt(np.mean(np.square(fit_error))))
    friction_only_rmse = float(np.sqrt(np.mean(np.square(friction_error))))
    improvement = max(0.0, (friction_only_rmse - rmse) / max(friction_only_rmse, 1.0e-12))

    return {
        "law": (
            "armature_kg_m2 * acceleration + coulomb_nm * tanh(velocity / transition_velocity) + "
            "viscous_nm_per_rad_s * velocity + bias_nm"
        ),
        "sample_count": int(fit_residual.size),
        "armature_kg_m2": armature,
        "maximum_armature_bound_kg_m2": float(maximum_armature_kg_m2),
        "coulomb_nm": coulomb,
        "viscous_nm_per_rad_s": viscous,
        "bias_nm": bias,
        "transition_velocity_rad_s": float(transition_velocity_rad_s),
        "acceleration_rms_rad_s2": float(np.sqrt(np.mean(np.square(fit_acceleration)))),
        "acceleration_p95_abs_rad_s2": float(np.quantile(np.abs(fit_acceleration), 0.95)),
        "armature_torque_rms_nm": float(abs(armature) * np.sqrt(np.mean(np.square(fit_acceleration)))),
        "normalized_design_condition_number": normalized_condition_number,
        "rmse_nm": rmse,
        "friction_only_rmse_nm": friction_only_rmse,
        "rmse_improvement_over_friction_only": improvement,
        "r2": r2,
        "optimizer_success": bool(result.success),
    }


def fit_harmonic_armature_cycles(
    residual_projection_nm: np.ndarray,
    acceleration_projection_rad_s2: np.ndarray,
    position_projection_rad: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    maximum_armature_kg_m2: float = 0.1,
    minimum_reliable_acceleration_rad_s2: float = 3.0,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit armature from cycle-level fundamental harmonics.

    Velocity-aligned torque is removed before calling this function. The
    remaining fundamental is modeled as residual armature plus a common
    position-dependent torque error. Equal-amplitude, multiple-frequency data
    separates those terms because acceleration scales with frequency squared.
    """

    residual = np.asarray(residual_projection_nm, dtype=np.float64)
    acceleration = np.asarray(acceleration_projection_rad_s2, dtype=np.float64)
    position = np.asarray(position_projection_rad, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if not (
        residual.ndim == 1
        and acceleration.shape == residual.shape
        and position.shape == residual.shape
        and frequency.shape == residual.shape
    ):
        raise TorqueIdentificationError(
            "Harmonic residual, acceleration, position, and frequency must be matching one-dimensional arrays."
        )
    if not all(np.all(np.isfinite(values)) for values in (residual, acceleration, position, frequency)):
        raise TorqueIdentificationError("Harmonic armature inputs must be finite.")
    if residual.size < 9:
        raise TorqueIdentificationError("Harmonic armature fit requires at least nine complete cycles.")
    if np.any(frequency <= 0.0):
        raise TorqueIdentificationError("Harmonic frequencies must be positive.")
    if not math.isfinite(maximum_armature_kg_m2) or maximum_armature_kg_m2 <= 0.0:
        raise TorqueIdentificationError("maximum_armature_kg_m2 must be finite and positive.")
    if not math.isfinite(minimum_reliable_acceleration_rad_s2) or minimum_reliable_acceleration_rad_s2 <= 0.0:
        raise TorqueIdentificationError("minimum_reliable_acceleration_rad_s2 must be finite and positive.")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise TorqueIdentificationError("bootstrap_samples must be at least 100.")

    unique_frequencies = np.unique(frequency)
    if unique_frequencies.size < 3:
        raise TorqueIdentificationError("Harmonic armature fit requires at least three distinct frequencies.")
    frequency_groups = [np.flatnonzero(frequency == value) for value in unique_frequencies]
    if any(group.size < 3 for group in frequency_groups):
        raise TorqueIdentificationError("Each harmonic frequency requires at least three complete cycles.")

    design = np.column_stack((acceleration, position))
    column_norm = np.linalg.norm(design, axis=0)
    if np.any(column_norm <= 1.0e-12):
        raise TorqueIdentificationError("Harmonic armature regression contains an unobservable regressor.")
    normalized_condition_number = float(np.linalg.cond(design / column_norm))
    linear_initial, *_ = np.linalg.lstsq(design, residual, rcond=None)
    unconstrained_armature = float(linear_initial[0])
    initial = np.asarray(
        [float(np.clip(unconstrained_armature, 0.0, maximum_armature_kg_m2)), float(linear_initial[1])]
    )
    scale = max(float(np.median(np.abs(residual - np.median(residual)))), 1.0e-4)
    result = least_squares(
        lambda parameters: design @ parameters - residual,
        initial,
        bounds=([0.0, -np.inf], [maximum_armature_kg_m2, np.inf]),
        loss="soft_l1",
        f_scale=scale,
    )
    armature, position_error = (float(value) for value in result.x)
    prediction = design @ result.x
    fit_error = residual - prediction

    position_only_initial, *_ = np.linalg.lstsq(position[:, None], residual, rcond=None)
    position_only_result = least_squares(
        lambda parameters: position * parameters[0] - residual,
        position_only_initial,
        loss="soft_l1",
        f_scale=scale,
    )
    position_only_error = residual - position * position_only_result.x[0]
    rmse = float(np.sqrt(np.mean(np.square(fit_error))))
    position_only_rmse = float(np.sqrt(np.mean(np.square(position_only_error))))
    improvement = max(0.0, (position_only_rmse - rmse) / max(position_only_rmse, 1.0e-12))
    denominator = float(np.sum(np.square(residual - np.mean(residual))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(fit_error))) / denominator

    rng = np.random.default_rng(seed)
    bootstrap_armatures: list[float] = []
    for _ in range(bootstrap_samples):
        indices = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in frequency_groups])
        sampled_design = design[indices]
        if np.linalg.matrix_rank(sampled_design) < 2:
            continue
        parameters, *_ = np.linalg.lstsq(sampled_design, residual[indices], rcond=None)
        bootstrap_armatures.append(float(parameters[0]))
    if len(bootstrap_armatures) < 0.9 * bootstrap_samples:
        raise TorqueIdentificationError("Too many harmonic bootstrap samples were rank deficient.")
    bootstrap = np.asarray(bootstrap_armatures, dtype=np.float64)
    interval_low, bootstrap_median, interval_high = (
        float(value) for value in np.quantile(bootstrap, (0.05, 0.5, 0.95))
    )
    relative_interval_width = (interval_high - interval_low) / max(abs(bootstrap_median), 1.0e-12)

    frequency_estimates: list[dict[str, Any]] = []
    reliable_estimates: list[float] = []
    for value, group in zip(unique_frequencies, frequency_groups, strict=True):
        valid = group[np.abs(acceleration[group]) > 1.0e-9]
        cycle_estimates = (residual[valid] - position_error * position[valid]) / acceleration[valid]
        estimate_median = float(np.median(cycle_estimates))
        acceleration_median = float(np.median(np.abs(acceleration[group])))
        reliable = acceleration_median >= minimum_reliable_acceleration_rad_s2
        if reliable:
            reliable_estimates.append(estimate_median)
        frequency_estimates.append(
            {
                "frequency_hz": float(value),
                "cycle_count": int(group.size),
                "median_acceleration_projection_rad_s2": acceleration_median,
                "armature_median_kg_m2": estimate_median,
                "armature_mad_kg_m2": float(np.median(np.abs(cycle_estimates - estimate_median))),
                "reliable_for_consistency": reliable,
            }
        )
    reliable_array = np.asarray(reliable_estimates, dtype=np.float64)
    reliable_relative_span = None
    if reliable_array.size >= 2:
        reliable_relative_span = float(
            np.ptp(reliable_array) / max(abs(float(np.median(reliable_array))), 1.0e-12)
        )

    acceleration_rms = float(np.sqrt(np.mean(np.square(acceleration))))
    return {
        "law": "residual_fundamental = armature_kg_m2 * acceleration_fundamental + position_error_nm_per_rad * position_fundamental",
        "sample_unit": "one fundamental-harmonic projection per complete excitation cycle",
        "cycle_count": int(residual.size),
        "frequency_count": int(unique_frequencies.size),
        "armature_kg_m2": armature,
        "unconstrained_armature_kg_m2": unconstrained_armature,
        "maximum_armature_bound_kg_m2": float(maximum_armature_kg_m2),
        "position_error_nm_per_rad": position_error,
        "minimum_reliable_acceleration_rad_s2": float(minimum_reliable_acceleration_rad_s2),
        "acceleration_rms_rad_s2": acceleration_rms,
        "acceleration_p95_abs_rad_s2": float(np.quantile(np.abs(acceleration), 0.95)),
        "armature_torque_rms_nm": float(abs(armature) * acceleration_rms),
        "normalized_design_condition_number": normalized_condition_number,
        "rmse_nm": rmse,
        "position_only_rmse_nm": position_only_rmse,
        "rmse_improvement_over_position_only": improvement,
        "r2": r2,
        "optimizer_success": bool(result.success),
        "bootstrap_samples": int(bootstrap.size),
        "bootstrap_median_kg_m2": bootstrap_median,
        "bootstrap_90pct_kg_m2": [interval_low, interval_high],
        "bootstrap_relative_interval_width": float(relative_interval_width),
        "frequency_estimates": frequency_estimates,
        "reliable_frequency_count": int(reliable_array.size),
        "reliable_frequency_relative_span": reliable_relative_span,
    }


def fit_quasistatic_gravity_calibration(
    position_rad: np.ndarray,
    current_a: np.ndarray,
    velocity_rad_s: np.ndarray,
    gravity_torque_nm: np.ndarray,
    sample_mask: np.ndarray,
    *,
    minimum_speed_rad_s: float,
    bin_count: int = 40,
    minimum_samples_per_direction: int = 8,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Identify a low-current torque constant from matched gravity sweeps.

    Positive- and negative-velocity current samples are paired at the same joint
    position. Their average removes the direction-dependent Coulomb term, while
    half their difference estimates friction current.
    """

    position = np.asarray(position_rad, dtype=np.float64)
    current = np.asarray(current_a, dtype=np.float64)
    velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    gravity = np.asarray(gravity_torque_nm, dtype=np.float64)
    mask = np.asarray(sample_mask, dtype=np.bool_)
    if not (position.ndim == 1 and current.shape == position.shape == velocity.shape == gravity.shape == mask.shape):
        raise TorqueIdentificationError(
            "Position, current, velocity, gravity torque, and mask must be matching one-dimensional arrays."
        )
    if not all(np.all(np.isfinite(values)) for values in (position, current, velocity, gravity)):
        raise TorqueIdentificationError("Quasi-static calibration inputs must be finite.")
    if not math.isfinite(minimum_speed_rad_s) or minimum_speed_rad_s <= 0.0:
        raise TorqueIdentificationError("minimum_speed_rad_s must be finite and positive.")
    if not 8 <= bin_count <= 100:
        raise TorqueIdentificationError("bin_count must be in [8, 100].")
    if minimum_samples_per_direction < 2:
        raise TorqueIdentificationError("minimum_samples_per_direction must be at least two.")
    if bootstrap_samples < 100:
        raise TorqueIdentificationError("bootstrap_samples must be at least 100.")

    selected_position = position[mask]
    if selected_position.size < 2 * bin_count * minimum_samples_per_direction:
        raise TorqueIdentificationError("Quasi-static calibration does not contain enough selected samples.")
    lower, upper = np.quantile(selected_position, [0.01, 0.99])
    if upper <= lower:
        raise TorqueIdentificationError("Quasi-static calibration has no measurable position range.")
    edges = np.linspace(lower, upper, bin_count + 1)
    centers: list[float] = []
    gravity_bins: list[float] = []
    mean_current_bins: list[float] = []
    friction_current_bins: list[float] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    for index in range(bin_count):
        inside = mask & (position >= edges[index]) & (position < edges[index + 1])
        if index == bin_count - 1:
            inside |= mask & (position == edges[index + 1])
        positive = inside & (velocity >= minimum_speed_rad_s)
        negative = inside & (velocity <= -minimum_speed_rad_s)
        positive_count = int(np.count_nonzero(positive))
        negative_count = int(np.count_nonzero(negative))
        if min(positive_count, negative_count) < minimum_samples_per_direction:
            continue
        positive_current = float(np.median(current[positive]))
        negative_current = float(np.median(current[negative]))
        centers.append(float(np.median(position[inside])))
        gravity_bins.append(float(np.median(gravity[inside])))
        mean_current_bins.append(0.5 * (positive_current + negative_current))
        friction_current_bins.append(0.5 * (positive_current - negative_current))
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)

    if len(centers) < 8:
        raise TorqueIdentificationError("Fewer than eight position bins contain both motion directions.")
    gravity_array = np.asarray(gravity_bins)
    mean_current_array = np.asarray(mean_current_bins)
    design = np.column_stack((mean_current_array, np.ones_like(mean_current_array)))
    condition_number = float(np.linalg.cond(design))

    def solve(indices: np.ndarray | None = None) -> tuple[float, float]:
        selected_design = design if indices is None else design[indices]
        selected_gravity = gravity_array if indices is None else gravity_array[indices]
        solution = least_squares(
            lambda parameters: selected_design @ parameters - selected_gravity,
            np.asarray([1.0, 0.0]),
            bounds=([0.1, -1.0], [3.0, 1.0]),
            loss="soft_l1",
            f_scale=max(float(np.ptp(selected_gravity)) * 0.05, 1.0e-4),
        )
        return float(solution.x[0]), float(solution.x[1])

    torque_per_amp, bias = solve()
    prediction = torque_per_amp * mean_current_array + bias
    error = gravity_array - prediction
    denominator = float(np.sum(np.square(gravity_array - np.mean(gravity_array))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(error))) / denominator
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        indices = rng.integers(0, len(centers), size=len(centers))
        bootstrap[index] = solve(indices)[0]
    q05, q50, q95 = (float(value) for value in np.quantile(bootstrap, [0.05, 0.5, 0.95]))
    relative_interval_width = (q95 - q05) / max(abs(q50), 1.0e-9)
    friction_current = np.asarray(friction_current_bins)
    coulomb_current = float(np.median(np.abs(friction_current)))
    gravity_span = float(np.ptp(gravity_array))
    current_span = float(np.ptp(mean_current_array))

    quality_reasons: list[str] = []
    if gravity_span < 0.05:
        quality_reasons.append("Matched gravity torque span is below 0.05 Nm.")
    if current_span < 0.02:
        quality_reasons.append("Matched mean-current span is below 0.02 A.")
    if condition_number > 100.0:
        quality_reasons.append("Calibration design condition number exceeds 100.")
    if r2 is None or r2 < 0.8:
        quality_reasons.append("Gravity/current calibration R2 is below 0.8.")
    if relative_interval_width > 0.3:
        quality_reasons.append("Bootstrap 90% torque-constant interval is wider than 30% of its median.")
    if not 0.2 < torque_per_amp < 2.8:
        quality_reasons.append("Torque constant is too close to the optimizer bounds.")

    return {
        "method": "matched-position bidirectional quasi-static gravity sweep",
        "equation": "gravity_torque_nm = torque_per_amp_nm * mean_direction_current_a + bias_nm",
        "bin_count": len(centers),
        "minimum_speed_rad_s": float(minimum_speed_rad_s),
        "torque_per_amp_nm": torque_per_amp,
        "bias_nm": bias,
        "coulomb_current_a": coulomb_current,
        "coulomb_torque_nm": torque_per_amp * coulomb_current,
        "gravity_torque_span_nm": gravity_span,
        "mean_current_span_a": current_span,
        "condition_number": condition_number,
        "rmse_nm": float(np.sqrt(np.mean(np.square(error)))),
        "r2": r2,
        "bootstrap_90pct_nm_per_a": [q05, q95],
        "bootstrap_median_nm_per_a": q50,
        "bootstrap_relative_interval_width": relative_interval_width,
        "position_rad": centers,
        "gravity_torque_nm": gravity_bins,
        "mean_direction_current_a": mean_current_bins,
        "friction_current_a": friction_current_bins,
        "positive_sample_count": positive_counts,
        "negative_sample_count": negative_counts,
        "quality": {"pass": not quality_reasons, "reasons": quality_reasons},
    }
