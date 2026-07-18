"""Reward-independent NumPy metrics for fixed RND STEP evaluations.

All public metric functions consume synchronized policy-rate samples.  Foot
arrays use the explicit ``(right, left)`` order and root quaternions use
``(w, x, y, z)``.  Results contain only finite JSON scalars, objects, and nulls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FOOT_ORDER = ("right", "left")


class GaitMetricError(ValueError):
    """Raised when a metric input is malformed or its semantics are ambiguous."""


def _positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise GaitMetricError(f"{label} must be numeric, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GaitMetricError(f"{label} must be numeric; got {value!r}.") from error
    if not math.isfinite(result):
        raise GaitMetricError(f"{label} must be finite.")
    if result < 0.0 or (not allow_zero and result == 0.0):
        comparison = "non-negative" if allow_zero else "positive"
        raise GaitMetricError(f"{label} must be {comparison}; got {result}.")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GaitMetricError(f"{label} must be an integer.")
    result = int(value)
    if result < minimum:
        raise GaitMetricError(f"{label} must be >= {minimum}; got {result}.")
    if maximum is not None and result > maximum:
        raise GaitMetricError(f"{label} must be <= {maximum}; got {result}.")
    return result


def _float_array(value: Any, label: str, *, ndim: int, nonempty: bool = True) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GaitMetricError(f"{label} must be numeric: {error}") from error
    if array.ndim != ndim:
        raise GaitMetricError(f"{label} must have {ndim} dimensions; got shape {array.shape}.")
    if nonempty and array.shape[0] == 0:
        raise GaitMetricError(f"{label} must contain at least one sample.")
    if not np.isfinite(array).all():
        raise GaitMetricError(f"{label} contains NaN or infinity.")
    return array


def _bool_array(value: Any, label: str, *, ndim: int, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise GaitMetricError(f"{label} must have {ndim} dimensions; got shape {array.shape}.")
    if nonempty and array.shape[0] == 0:
        raise GaitMetricError(f"{label} must contain at least one sample.")
    if array.dtype.kind != "b":
        raise GaitMetricError(f"{label} must have boolean dtype; got {array.dtype}.")
    return np.asarray(array, dtype=np.bool_)


def _matching_sample_count(labelled_arrays: Sequence[tuple[str, np.ndarray]]) -> int:
    counts = {array.shape[0] for _, array in labelled_arrays}
    if len(counts) != 1:
        shapes = ", ".join(f"{label}={array.shape}" for label, array in labelled_arrays)
        raise GaitMetricError(f"Metric arrays must have the same sample count; got {shapes}.")
    return next(iter(counts))


def _valid_mask(value: Any | None, sample_count: int) -> np.ndarray:
    if value is None:
        return np.ones(sample_count, dtype=np.bool_)
    mask = _bool_array(value, "valid_mask", ndim=1)
    if mask.shape != (sample_count,):
        raise GaitMetricError(f"valid_mask must have shape {(sample_count,)}, got {mask.shape}.")
    return mask


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _relative_difference(right: float | int | None, left: float | int | None) -> tuple[float | None, float | None]:
    if right is None or left is None:
        return None, None
    right_value = float(right)
    left_value = float(left)
    absolute = abs(right_value - left_value)
    denominator = 0.5 * (abs(right_value) + abs(left_value))
    relative = 0.0 if denominator == 0.0 else absolute / denominator
    return float(absolute), float(relative)


def _event_count(mask: np.ndarray) -> int:
    if mask.ndim != 1:
        raise GaitMetricError("Internal event masks must be one-dimensional.")
    previous = np.concatenate((np.asarray([False]), mask[:-1]))
    return int(np.count_nonzero(mask & ~previous))


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for active in mask.tolist():
        if active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _touchdown_mask(contact: np.ndarray) -> np.ndarray:
    touchdown = np.zeros_like(contact, dtype=np.bool_)
    if contact.shape[0] > 1:
        touchdown[1:] = contact[1:] & ~contact[:-1]
    return touchdown


def _single_touchdown_sequence(touchdown: np.ndarray) -> tuple[list[int], int, int]:
    sequence: list[int] = []
    alternating = 0
    same = 0
    previous_side: int | None = None
    for row in touchdown:
        active = np.flatnonzero(row)
        if active.size != 1:
            if active.size > 1:
                previous_side = None
            continue
        side = int(active[0])
        sequence.append(side)
        if previous_side is not None:
            if side == previous_side:
                same += 1
            else:
                alternating += 1
        previous_side = side
    return sequence, alternating, same


def root_yaw_from_wxyz(root_quat_w: Any) -> np.ndarray:
    """Extract world yaw in radians from finite ``(w, x, y, z)`` quaternions."""

    quaternion = _float_array(root_quat_w, "root_quat_w", ndim=2)
    if quaternion.shape[1] != 4:
        raise GaitMetricError(f"root_quat_w must have shape (samples, 4); got {quaternion.shape}.")
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise GaitMetricError("root_quat_w contains a zero-norm quaternion.")
    w, x, y, z = (quaternion / norm[:, None]).T
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(sin_yaw, cos_yaw)


def root_tilt_metrics(
    root_quat_w: Any,
    command: Any,
    *,
    command_speed_threshold_m_s: float,
) -> dict[str, Any]:
    """Measure commanded-translation root tilt along STEP's lateral and sagittal axes.

    STEP's body X axis points left and body Y points backward. Therefore the X
    projected-gravity component measures lateral lean, while Y measures forward/backward
    sagittal lean. Signed sagittal tilt makes a persistent direction bias visible.
    """

    quaternion = _float_array(root_quat_w, "root_quat_w", ndim=2)
    commands = _float_array(command, "command", ndim=2)
    if quaternion.shape[1] != 4 or commands.shape[1] < 2:
        raise GaitMetricError("root_quat_w must have four components and command at least two.")
    _matching_sample_count((("root_quat_w", quaternion), ("command", commands)))
    threshold = _positive_float(
        command_speed_threshold_m_s,
        "command_speed_threshold_m_s",
        allow_zero=True,
    )

    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise GaitMetricError("root_quat_w contains a zero-norm quaternion.")
    w, x, y, z = (quaternion / norm[:, None]).T
    gravity_x = -2.0 * (x * z - w * y)
    gravity_y = -2.0 * (y * z + w * x)
    lateral_deg = np.degrees(np.arcsin(np.clip(gravity_x, -1.0, 1.0)))
    sagittal_deg = np.degrees(np.arcsin(np.clip(gravity_y, -1.0, 1.0)))
    moving = np.linalg.norm(commands[:, :2], axis=1) > threshold
    moving_count = int(np.count_nonzero(moving))
    if moving_count == 0:
        return {
            "moving_sample_count": 0,
            "lateral_abs_mean_deg": None,
            "lateral_abs_p95_deg": None,
            "sagittal_signed_mean_deg": None,
            "sagittal_abs_mean_deg": None,
            "sagittal_abs_p95_deg": None,
        }

    lateral_abs = np.abs(lateral_deg[moving])
    sagittal_moving = sagittal_deg[moving]
    sagittal_abs = np.abs(sagittal_moving)
    return {
        "moving_sample_count": moving_count,
        "lateral_abs_mean_deg": float(np.mean(lateral_abs)),
        "lateral_abs_p95_deg": float(np.percentile(lateral_abs, 95.0)),
        "sagittal_signed_mean_deg": float(np.mean(sagittal_moving)),
        "sagittal_abs_mean_deg": float(np.mean(sagittal_abs)),
        "sagittal_abs_p95_deg": float(np.percentile(sagittal_abs, 95.0)),
    }


def world_xy_to_yaw_frame(vector_xy_w: Any, root_yaw_w: Any) -> np.ndarray:
    """Rotate world-frame XY vectors by inverse root yaw."""

    vectors = _float_array(vector_xy_w, "vector_xy_w", ndim=2)
    if vectors.shape[1] != 2:
        raise GaitMetricError(f"vector_xy_w must have shape (samples, 2); got {vectors.shape}.")
    yaw = _float_array(root_yaw_w, "root_yaw_w", ndim=1)
    _matching_sample_count((("vector_xy_w", vectors), ("root_yaw_w", yaw)))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.column_stack((
        cosine * vectors[:, 0] + sine * vectors[:, 1],
        -sine * vectors[:, 0] + cosine * vectors[:, 1],
    ))


def yaw_frame_linear_velocity_rmse(
    root_lin_vel_w: Any,
    root_quat_w: Any,
    command: Any,
    *,
    valid_mask: Any | None = None,
) -> float | None:
    """Compute ``sqrt(mean(||v_yaw_xy - command_xy||^2))`` in m/s."""

    velocity = _float_array(root_lin_vel_w, "root_lin_vel_w", ndim=2)
    quaternion = _float_array(root_quat_w, "root_quat_w", ndim=2)
    commands = _float_array(command, "command", ndim=2)
    if velocity.shape[1] < 2 or commands.shape[1] < 2:
        raise GaitMetricError("root_lin_vel_w and command must each contain at least two components.")
    sample_count = _matching_sample_count((
        ("root_lin_vel_w", velocity),
        ("root_quat_w", quaternion),
        ("command", commands),
    ))
    mask = _valid_mask(valid_mask, sample_count)
    if not np.any(mask):
        return None
    yaw = root_yaw_from_wxyz(quaternion)
    velocity_yaw = world_xy_to_yaw_frame(velocity[:, :2], yaw)
    squared_norm = np.sum(np.square(velocity_yaw - commands[:, :2]), axis=1)
    return float(np.sqrt(np.mean(squared_norm[mask])))


def yaw_rate_rmse(
    root_ang_vel_w: Any,
    command: Any,
    *,
    valid_mask: Any | None = None,
) -> float | None:
    """Compute world-Z yaw-rate RMSE in rad/s."""

    angular_velocity = _float_array(root_ang_vel_w, "root_ang_vel_w", ndim=2)
    commands = _float_array(command, "command", ndim=2)
    if angular_velocity.shape[1] < 3 or commands.shape[1] < 3:
        raise GaitMetricError("root_ang_vel_w and command must each contain at least three components.")
    sample_count = _matching_sample_count((("root_ang_vel_w", angular_velocity), ("command", commands)))
    mask = _valid_mask(valid_mask, sample_count)
    if not np.any(mask):
        return None
    error = angular_velocity[:, 2] - commands[:, 2]
    return float(np.sqrt(np.mean(np.square(error[mask]))))


def survival_fall_metrics(
    termination: Any,
    *,
    horizon_steps: int,
    step_dt: float,
    timeout: Any | None = None,
) -> dict[str, Any]:
    """Measure survival and early non-timeout fall termination.

    A termination at exactly ``horizon_steps`` is horizon completion, not a
    fall.  Missing samples without termination are reported as censored.
    """

    terminations = _bool_array(termination, "termination", ndim=1)
    horizon = _integer(horizon_steps, "horizon_steps", minimum=1)
    dt = _positive_float(step_dt, "step_dt")
    if terminations.shape[0] > horizon:
        raise GaitMetricError(
            f"termination contains {terminations.shape[0]} samples, exceeding horizon_steps={horizon}."
        )
    if timeout is None:
        timeouts = np.zeros_like(terminations)
    else:
        timeouts = _bool_array(timeout, "timeout", ndim=1)
        if timeouts.shape != terminations.shape:
            raise GaitMetricError(f"timeout must have shape {terminations.shape}, got {timeouts.shape}.")

    boundaries = terminations | timeouts
    boundary_indices = np.flatnonzero(boundaries)
    if boundary_indices.size:
        first_index = int(boundary_indices[0])
        end_step = first_index + 1
        timed_out = bool(timeouts[first_index]) or end_step >= horizon
        fell = bool(terminations[first_index]) and not bool(timeouts[first_index]) and end_step < horizon
    else:
        first_index = None
        end_step = min(terminations.shape[0], horizon)
        timed_out = end_step >= horizon
        fell = False

    completed_horizon = end_step >= horizon and not fell
    censored = not fell and not completed_horizon
    return {
        "fell": bool(fell),
        "fall_step": first_index if fell else None,
        "fall_time_s": float((first_index + 1) * dt) if fell and first_index is not None else None,
        "timed_out": bool(timed_out),
        "completed_horizon": bool(completed_horizon),
        "censored": bool(censored),
        "survived_steps": int(end_step),
        "survival_time_s": float(end_step * dt),
        "survival_fraction": float(end_step / horizon),
    }


def _completed_phase_durations(contact: np.ndarray, *, active_value: bool, step_dt: float) -> list[float]:
    durations: list[float] = []
    start = 0
    sample_count = contact.shape[0]
    while start < sample_count:
        value = bool(contact[start])
        end = start + 1
        while end < sample_count and bool(contact[end]) == value:
            end += 1
        if value == active_value and end < sample_count:
            durations.append(float((end - start) * step_dt))
        start = end
    return durations


def gait_symmetry_metrics(foot_contact: Any, *, step_dt: float) -> dict[str, Any]:
    """Measure observed right/left timing and touchdown-count symmetry.

    Air and stance means use phases completed by an in-episode transition; the
    trailing censored phase is excluded.  Duty factor uses every sample.
    """

    contact = _bool_array(foot_contact, "foot_contact", ndim=2)
    if contact.shape[1] != 2:
        raise GaitMetricError(f"foot_contact must have shape (samples, 2); got {contact.shape}.")
    dt = _positive_float(step_dt, "step_dt")
    touchdown = _touchdown_mask(contact)

    per_foot: dict[str, dict[str, Any]] = {}
    for foot_index, foot_name in enumerate(FOOT_ORDER):
        air_durations = _completed_phase_durations(contact[:, foot_index], active_value=False, step_dt=dt)
        stance_durations = _completed_phase_durations(contact[:, foot_index], active_value=True, step_dt=dt)
        per_foot[foot_name] = {
            "air_time_mean_s": _mean_or_none(air_durations),
            "air_phase_count": int(len(air_durations)),
            "stance_time_mean_s": _mean_or_none(stance_durations),
            "stance_phase_count": int(len(stance_durations)),
            "duty_factor": float(np.mean(contact[:, foot_index])),
            "touchdown_count": int(np.count_nonzero(touchdown[:, foot_index])),
        }

    air_absolute, air_relative = _relative_difference(
        per_foot["right"]["air_time_mean_s"], per_foot["left"]["air_time_mean_s"]
    )
    stance_absolute, stance_relative = _relative_difference(
        per_foot["right"]["stance_time_mean_s"], per_foot["left"]["stance_time_mean_s"]
    )
    duty_absolute, duty_relative = _relative_difference(
        per_foot["right"]["duty_factor"], per_foot["left"]["duty_factor"]
    )
    count_absolute, count_relative = _relative_difference(
        per_foot["right"]["touchdown_count"], per_foot["left"]["touchdown_count"]
    )
    sequence, alternating, same = _single_touchdown_sequence(touchdown)
    transition_count = alternating + same

    return {
        "right": per_foot["right"],
        "left": per_foot["left"],
        "air_time_abs_difference_s": air_absolute,
        "air_time_relative_difference": air_relative,
        "stance_time_abs_difference_s": stance_absolute,
        "stance_time_relative_difference": stance_relative,
        "duty_factor_abs_difference": duty_absolute,
        "duty_factor_relative_difference": duty_relative,
        "touchdown_count_abs_difference": int(count_absolute) if count_absolute is not None else None,
        "touchdown_count_relative_difference": count_relative,
        "single_touchdown_count": int(len(sequence)),
        "alternating_transition_count": int(alternating),
        "consecutive_same_foot_count": int(same),
        "consecutive_same_foot_fraction": (float(same / transition_count) if transition_count > 0 else None),
    }


def _preceding_air_samples(contact: np.ndarray, touchdown_step: int) -> int:
    index = touchdown_step - 1
    count = 0
    while index >= 0 and not bool(contact[index]):
        count += 1
        index -= 1
    return count


def touchdown_metrics(
    foot_contact: Any,
    foot_pos_w: Any,
    root_quat_w: Any,
    command: Any,
    *,
    step_dt: float,
    minimum_progress_m: float,
    tap_max_air_time_s: float,
    command_speed_threshold_m_s: float,
    touchdown_event: Any | None = None,
    preceding_air_time_s: Any | None = None,
    preimpact_speed_m_s: Any | None = None,
) -> dict[str, Any]:
    """Measure touchdown progress, alternation, taps, and simultaneous landings.

    Progress is the landing foot's lead over the other foot projected onto the
    commanded yaw-frame XY direction at the touchdown sample.  Simultaneous
    touchdowns and commands below the speed threshold are excluded from progress.
    """

    contact = _bool_array(foot_contact, "foot_contact", ndim=2)
    positions = _float_array(foot_pos_w, "foot_pos_w", ndim=3)
    quaternion = _float_array(root_quat_w, "root_quat_w", ndim=2)
    commands = _float_array(command, "command", ndim=2)
    if contact.shape[1] != 2 or positions.shape[1:] != (2, 3):
        raise GaitMetricError(
            f"foot_contact and foot_pos_w must have shapes (samples, 2) and (samples, 2, 3); "
            f"got {contact.shape} and {positions.shape}."
        )
    if quaternion.shape[1] != 4 or commands.shape[1] < 2:
        raise GaitMetricError("root_quat_w must have four components and command at least two.")
    sample_count = _matching_sample_count((
        ("foot_contact", contact),
        ("foot_pos_w", positions),
        ("root_quat_w", quaternion),
        ("command", commands),
    ))
    dt = _positive_float(step_dt, "step_dt")
    minimum_progress = _positive_float(minimum_progress_m, "minimum_progress_m")
    tap_threshold = _positive_float(tap_max_air_time_s, "tap_max_air_time_s", allow_zero=True)
    command_threshold = _positive_float(command_speed_threshold_m_s, "command_speed_threshold_m_s", allow_zero=True)

    if (touchdown_event is None) != (preceding_air_time_s is None):
        raise GaitMetricError("touchdown_event and preceding_air_time_s must be provided together.")
    if touchdown_event is None:
        touchdown = _touchdown_mask(contact)
        event_air_time = None
        event_preimpact = None
    else:
        touchdown = _bool_array(touchdown_event, "touchdown_event", ndim=2)
        event_air_time = _float_array(preceding_air_time_s, "preceding_air_time_s", ndim=2)
        if touchdown.shape != contact.shape or event_air_time.shape != contact.shape:
            raise GaitMetricError(
                "Physics touchdown events and preceding air time must match foot_contact shape "
                f"{contact.shape}; got {touchdown.shape} and {event_air_time.shape}."
            )
        if bool(np.any(event_air_time < 0.0)):
            raise GaitMetricError("preceding_air_time_s must be non-negative.")
        if preimpact_speed_m_s is None:
            event_preimpact = None
        else:
            event_preimpact = _float_array(preimpact_speed_m_s, "preimpact_speed_m_s", ndim=2)
            if event_preimpact.shape != contact.shape or bool(np.any(event_preimpact < 0.0)):
                raise GaitMetricError(
                    f"preimpact_speed_m_s must be non-negative with shape {contact.shape}."
                )
    yaw = root_yaw_from_wxyz(quaternion)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    positions_yaw = np.empty_like(positions)
    positions_yaw[..., 0] = cosine[:, None] * positions[..., 0] + sine[:, None] * positions[..., 1]
    positions_yaw[..., 1] = -sine[:, None] * positions[..., 0] + cosine[:, None] * positions[..., 1]
    positions_yaw[..., 2] = positions[..., 2]

    progress_values: list[float] = []
    progress_by_foot: dict[str, list[float]] = {name: [] for name in FOOT_ORDER}
    tap_count = 0
    single_count = 0
    simultaneous_count = 0
    touchdown_frame_count = 0
    touchdown_event_count = 0
    below_minimum_count = 0
    preimpact_values: list[float] = []

    for step in range(sample_count):
        active = np.flatnonzero(touchdown[step])
        if active.size == 0:
            continue
        touchdown_frame_count += 1
        touchdown_event_count += int(active.size)
        if event_preimpact is not None:
            preimpact_values.extend(float(event_preimpact[step, foot_index]) for foot_index in active)
        if active.size == 2:
            simultaneous_count += 1
            continue

        single_count += 1
        foot_index = int(active[0])
        air_time = (
            _preceding_air_samples(contact[:, foot_index], step) * dt
            if event_air_time is None
            else float(event_air_time[step, foot_index])
        )
        is_tap = air_time < tap_threshold
        if is_tap:
            tap_count += 1

        command_xy = commands[step, :2]
        command_speed = float(np.linalg.norm(command_xy))
        if is_tap or command_speed <= command_threshold:
            continue
        direction = command_xy / command_speed
        other_index = 1 - foot_index
        lead = float(np.dot(positions_yaw[step, foot_index, :2] - positions_yaw[step, other_index, :2], direction))
        progress_values.append(lead)
        progress_by_foot[FOOT_ORDER[foot_index]].append(lead)
        if lead < minimum_progress:
            below_minimum_count += 1

    sequence, alternating, same = _single_touchdown_sequence(touchdown)
    alternation_denominator = alternating + same
    progress_count = len(progress_values)
    return {
        "touchdown_frame_count": int(touchdown_frame_count),
        "touchdown_event_count": int(touchdown_event_count),
        "single_touchdown_count": int(single_count),
        "simultaneous_touchdown_count": int(simultaneous_count),
        "simultaneous_touchdown_fraction": (
            float(simultaneous_count / touchdown_frame_count) if touchdown_frame_count > 0 else None
        ),
        "tap_count": int(tap_count),
        "tap_fraction": float(tap_count / single_count) if single_count > 0 else None,
        "alternating_transition_count": int(alternating),
        "consecutive_same_foot_count": int(same),
        "alternation_fraction": (float(alternating / alternation_denominator) if alternation_denominator > 0 else None),
        "progress_event_count": int(progress_count),
        "progress_mean_m": _mean_or_none(progress_values),
        "progress_minimum_m": float(min(progress_values)) if progress_values else None,
        "progress_below_minimum_count": int(below_minimum_count),
        "progress_below_minimum_fraction": (
            float(below_minimum_count / progress_count) if progress_count > 0 else None
        ),
        "right_progress_mean_m": _mean_or_none(progress_by_foot["right"]),
        "left_progress_mean_m": _mean_or_none(progress_by_foot["left"]),
        "preimpact_speed_mean_m_s": _mean_or_none(preimpact_values),
        "preimpact_speed_p95_m_s": (
            float(np.percentile(np.asarray(preimpact_values, dtype=np.float64), 95.0))
            if preimpact_values
            else None
        ),
    }


def torque_saturation_metrics(
    applied_torque: Any,
    effort_limits: Any,
    *,
    step_dt: float,
    threshold_fraction: float,
    joint_names: Sequence[str] | None = None,
    valid_mask: Any | None = None,
) -> dict[str, Any]:
    """Measure saturation occupancy, contiguous events, and longest dwell.

    A sample is saturated when ``abs(torque) >= threshold_fraction * limit``.
    System events operate on ``any(joint saturated)``; joint events are also
    reported individually and summed.
    """

    torque = _float_array(applied_torque, "applied_torque", ndim=2)
    sample_count, joint_count = torque.shape
    if joint_count == 0:
        raise GaitMetricError("applied_torque must contain at least one joint.")
    dt = _positive_float(step_dt, "step_dt")
    fraction = _positive_float(threshold_fraction, "threshold_fraction")
    if fraction > 1.0:
        raise GaitMetricError("threshold_fraction must be <= 1.0.")
    mask = _valid_mask(valid_mask, sample_count)
    if not np.any(mask):
        raise GaitMetricError("valid_mask must select at least one torque sample.")

    limits = np.asarray(effort_limits, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full((sample_count, joint_count), float(limits), dtype=np.float64)
    elif limits.shape == (joint_count,):
        limits = np.broadcast_to(limits[None, :], (sample_count, joint_count))
    elif limits.shape != (sample_count, joint_count):
        raise GaitMetricError(
            f"effort_limits must be scalar, shape (joints,), or shape (samples, joints); got {limits.shape}."
        )
    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise GaitMetricError("effort_limits must be finite and strictly positive.")

    if joint_names is None:
        names = tuple(f"joint_{index}" for index in range(joint_count))
    else:
        if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
            raise GaitMetricError("joint_names must be a sequence of strings.")
        names = tuple(joint_names)
        if len(names) != joint_count or any(not isinstance(name, str) or not name for name in names):
            raise GaitMetricError(f"joint_names must contain {joint_count} non-empty strings.")
        if len(names) != len(set(names)):
            raise GaitMetricError("joint_names must not contain duplicates.")

    saturation = np.abs(torque) >= fraction * limits
    saturation &= mask[:, None]
    valid_joint_samples = int(np.count_nonzero(mask) * joint_count)
    any_joint = np.any(saturation, axis=1)
    system_event_count = _event_count(any_joint)
    system_longest = _longest_true_run(any_joint)

    per_joint: dict[str, dict[str, Any]] = {}
    total_joint_events = 0
    longest_joint_dwell = 0
    for joint_index, name in enumerate(names):
        joint_mask = saturation[:, joint_index]
        event_count = _event_count(joint_mask)
        longest = _longest_true_run(joint_mask)
        total_joint_events += event_count
        longest_joint_dwell = max(longest_joint_dwell, longest)
        per_joint[name] = {
            "fraction": float(np.count_nonzero(joint_mask) / np.count_nonzero(mask)),
            "event_count": int(event_count),
            "longest_dwell_steps": int(longest),
            "longest_dwell_s": float(longest * dt),
        }

    return {
        "threshold_fraction": float(fraction),
        "sample_joint_fraction": float(np.count_nonzero(saturation) / valid_joint_samples),
        "any_joint_step_fraction": float(np.count_nonzero(any_joint) / np.count_nonzero(mask)),
        "system_event_count": int(system_event_count),
        "system_longest_dwell_steps": int(system_longest),
        "system_longest_dwell_s": float(system_longest * dt),
        "joint_event_count": int(total_joint_events),
        "longest_joint_dwell_steps": int(longest_joint_dwell),
        "longest_joint_dwell_s": float(longest_joint_dwell * dt),
        "per_joint": per_joint,
    }


def push_recovery_metrics(
    linear_velocity_error: Any,
    yaw_rate_error: Any,
    *,
    push_end_step: int,
    step_dt: float,
    linear_velocity_threshold_m_s: float,
    yaw_rate_threshold_rad_s: float,
    dwell_s: float,
) -> dict[str, Any]:
    """Find the first post-push threshold interval sustained for ``dwell_s``.

    Linear error may be a scalar magnitude per sample or XY error vectors.  The
    recovery time is measured from the first post-pulse sample to the start of
    the first qualifying dwell.  No qualifying dwell yields right censoring.
    """

    linear = np.asarray(linear_velocity_error, dtype=np.float64)
    if linear.ndim == 1:
        linear_magnitude = np.abs(linear)
    elif linear.ndim == 2 and linear.shape[1] == 2:
        linear_magnitude = np.linalg.norm(linear, axis=1)
    else:
        raise GaitMetricError(f"linear_velocity_error must have shape (samples,) or (samples, 2); got {linear.shape}.")
    yaw = _float_array(yaw_rate_error, "yaw_rate_error", ndim=1, nonempty=False)
    if linear_magnitude.shape[0] != yaw.shape[0]:
        raise GaitMetricError("linear_velocity_error and yaw_rate_error must have the same sample count.")
    if not np.isfinite(linear_magnitude).all():
        raise GaitMetricError("linear_velocity_error contains NaN or infinity.")

    sample_count = yaw.shape[0]
    push_end = _integer(push_end_step, "push_end_step", minimum=0, maximum=sample_count)
    dt = _positive_float(step_dt, "step_dt")
    linear_threshold = _positive_float(linear_velocity_threshold_m_s, "linear_velocity_threshold_m_s")
    yaw_threshold = _positive_float(yaw_rate_threshold_rad_s, "yaw_rate_threshold_rad_s")
    dwell = _positive_float(dwell_s, "dwell_s")
    dwell_steps = max(1, int(math.ceil(dwell / dt - 1.0e-12)))

    qualifying = (linear_magnitude <= linear_threshold) & (np.abs(yaw) <= yaw_threshold)
    post_push = qualifying[push_end:]
    recovery_start: int | None = None
    current_start = 0
    current_length = 0
    for relative_step, is_qualifying in enumerate(post_push.tolist()):
        if is_qualifying:
            if current_length == 0:
                current_start = relative_step
            current_length += 1
            if current_length >= dwell_steps:
                recovery_start = push_end + current_start
                break
        else:
            current_length = 0

    observation_steps = sample_count - push_end
    qualifying_fraction = float(np.mean(post_push)) if observation_steps > 0 else None
    if recovery_start is None:
        censor_time = observation_steps * dt
        return {
            "recovered": False,
            "censored": True,
            "recovery_step": None,
            "confirmation_step": None,
            "recovery_time_s": None,
            "confirmation_time_s": None,
            "censor_time_s": float(censor_time),
            "recovery_time_or_censor_s": float(censor_time),
            "dwell_steps": int(dwell_steps),
            "dwell_s": float(dwell_steps * dt),
            "post_push_qualifying_fraction": qualifying_fraction,
        }

    confirmation_step = recovery_start + dwell_steps - 1
    recovery_time = (recovery_start - push_end) * dt
    confirmation_time = (confirmation_step - push_end + 1) * dt
    return {
        "recovered": True,
        "censored": False,
        "recovery_step": int(recovery_start),
        "confirmation_step": int(confirmation_step),
        "recovery_time_s": float(recovery_time),
        "confirmation_time_s": float(confirmation_time),
        "censor_time_s": None,
        "recovery_time_or_censor_s": float(recovery_time),
        "dwell_steps": int(dwell_steps),
        "dwell_s": float(dwell_steps * dt),
        "post_push_qualifying_fraction": qualifying_fraction,
    }


def _push_recovery_not_applicable() -> dict[str, Any]:
    """Return the stable metric-tree shape used by cases without a pulse."""

    return {
        "applicable": False,
        "delivery_complete": None,
        "recovered": None,
        "censored": None,
        "recovery_step": None,
        "confirmation_step": None,
        "recovery_time_s": None,
        "confirmation_time_s": None,
        "censor_time_s": None,
        "recovery_time_or_censor_s": None,
        "dwell_steps": None,
        "dwell_s": None,
        "post_push_qualifying_fraction": None,
    }


def _push_recovery_delivery_censored(
    *,
    sample_count: int,
    push_end_step: int,
    step_dt: float,
    dwell_s: float,
) -> dict[str, Any]:
    """Censor recovery when an episode ended before the full pulse was delivered."""

    push_end = _integer(push_end_step, "push_end_step", minimum=0, maximum=sample_count)
    dt = _positive_float(step_dt, "step_dt")
    dwell = _positive_float(dwell_s, "dwell_s")
    dwell_steps = max(1, int(math.ceil(dwell / dt - 1.0e-12)))
    censor_time = (sample_count - push_end) * dt
    return {
        "applicable": True,
        "delivery_complete": False,
        "recovered": False,
        "censored": True,
        "recovery_step": None,
        "confirmation_step": None,
        "recovery_time_s": None,
        "confirmation_time_s": None,
        "censor_time_s": float(censor_time),
        "recovery_time_or_censor_s": float(censor_time),
        "dwell_steps": int(dwell_steps),
        "dwell_s": float(dwell_steps * dt),
        "post_push_qualifying_fraction": None,
    }


def to_jsonable(value: Any, path: str = "result") -> Any:
    """Convert NumPy scalars/arrays to finite JSON-native values."""

    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), path)
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GaitMetricError(f"{path} contains NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GaitMetricError(f"{path} contains non-string key {key!r}.")
            result[key] = to_jsonable(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise GaitMetricError(f"{path} contains unsupported type {type(value).__name__}.")


def _mean_metric_trees(values: Sequence[Any], path: str = "metrics") -> Any:
    if not values:
        raise GaitMetricError(f"{path} must contain at least one value.")
    if all(value is None for value in values):
        return None
    non_null = [value for value in values if value is not None]
    exemplar = non_null[0]

    if isinstance(exemplar, Mapping):
        expected_keys = set(exemplar)
        for value in non_null:
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise GaitMetricError(f"{path} metric objects must have identical keys.")
        return {
            key: _mean_metric_trees([value[key] if value is not None else None for value in values], f"{path}.{key}")
            for key in sorted(expected_keys)
        }
    if isinstance(exemplar, bool):
        if any(not isinstance(value, bool) for value in non_null):
            raise GaitMetricError(f"{path} mixes boolean and non-boolean values.")
        return float(sum(bool(value) for value in non_null) / len(non_null))
    if isinstance(exemplar, (int, float)) and not isinstance(exemplar, bool):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in non_null):
            raise GaitMetricError(f"{path} mixes numeric and non-numeric values.")
        numeric = np.asarray(non_null, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise GaitMetricError(f"{path} contains NaN or infinity.")
        return float(np.mean(numeric))
    if isinstance(exemplar, str):
        if any(value != exemplar for value in non_null):
            raise GaitMetricError(f"{path} contains inconsistent string metadata.")
        return exemplar
    raise GaitMetricError(f"{path} contains unsupported aggregate leaf {type(exemplar).__name__}.")


def aggregate_episode_metrics(episode_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Give every episode equal weight when producing one case metric tree.

    Conditional null leaves (for example uncensored recovery time) are averaged
    over episodes where they are defined.  Their unconditional rate/censor
    leaves still include every episode.
    """

    if not isinstance(episode_metrics, Sequence) or isinstance(episode_metrics, (str, bytes)):
        raise GaitMetricError("episode_metrics must be a sequence of metric objects.")
    converted = [to_jsonable(metrics, f"episode_metrics[{index}]") for index, metrics in enumerate(episode_metrics)]
    if not converted or any(not isinstance(metrics, dict) for metrics in converted):
        raise GaitMetricError("episode_metrics must contain at least one metric object.")
    result = _mean_metric_trees(converted)
    if not isinstance(result, dict):
        raise GaitMetricError("Aggregated episode metrics must form an object.")
    return result


def aggregate_evaluation_results(episode_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate equal-weight episodes into cases, then equal-weight cases.

    Each record must contain exactly ``case_id``, ``split``, and ``metrics``.
    The top-level result is case-balanced even when cases have different episode
    counts; split results use the same case-first rule.
    """

    if not isinstance(episode_records, Sequence) or isinstance(episode_records, (str, bytes)):
        raise GaitMetricError("episode_records must be a sequence.")
    grouped: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(episode_records):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"case_id", "split", "metrics"}:
            raise GaitMetricError(f"episode_records[{index}] must contain exactly case_id, split, and metrics.")
        case_id = raw_record["case_id"]
        split = raw_record["split"]
        metrics = raw_record["metrics"]
        if not isinstance(case_id, str) or not case_id:
            raise GaitMetricError(f"episode_records[{index}].case_id must be a non-empty string.")
        if split not in ("validation", "test"):
            raise GaitMetricError(f"episode_records[{index}].split must be 'validation' or 'test'.")
        if not isinstance(metrics, Mapping):
            raise GaitMetricError(f"episode_records[{index}].metrics must be an object.")
        entry = grouped.setdefault(case_id, {"split": split, "episodes": []})
        if entry["split"] != split:
            raise GaitMetricError(f"case_id {case_id!r} appears in multiple splits.")
        entry["episodes"].append(metrics)
    if not grouped:
        raise GaitMetricError("episode_records must not be empty.")

    cases: dict[str, dict[str, Any]] = {}
    case_metrics_by_split: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    all_case_metrics: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        entry = grouped[case_id]
        metrics = aggregate_episode_metrics(entry["episodes"])
        cases[case_id] = {
            "split": entry["split"],
            "episode_count": int(len(entry["episodes"])),
            "metrics": metrics,
        }
        case_metrics_by_split[entry["split"]].append(metrics)
        all_case_metrics.append(metrics)

    splits: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        split_case_ids = [case_id for case_id, case in cases.items() if case["split"] == split]
        metrics = _mean_metric_trees(case_metrics_by_split[split]) if case_metrics_by_split[split] else None
        splits[split] = {
            "case_count": int(len(split_case_ids)),
            "episode_count": int(sum(cases[case_id]["episode_count"] for case_id in split_case_ids)),
            "metrics": metrics,
        }

    return to_jsonable({
        "aggregation": "equal_episode_then_equal_case",
        "case_count": int(len(cases)),
        "episode_count": int(len(episode_records)),
        "metrics": _mean_metric_trees(all_case_metrics),
        "splits": splits,
        "cases": cases,
    })


def evaluate_episode_metrics(
    telemetry: Mapping[str, Any],
    *,
    horizon_steps: int,
    step_dt: float,
    effort_limits: Any,
    minimum_touchdown_progress_m: float,
    tap_max_air_time_s: float,
    command_speed_threshold_m_s: float,
    torque_saturation_threshold_fraction: float,
    joint_names: Sequence[str] | None = None,
    timeout: Any | None = None,
    touchdown_step_dt: float | None = None,
    push_end_step: int | None = None,
    push_delivery_complete: bool = True,
    linear_velocity_error_threshold_m_s: float | None = None,
    yaw_rate_error_threshold_rad_s: float | None = None,
    recovery_dwell_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate one synchronized telemetry episode under an explicit contract.

    Required telemetry keys are ``termination``, ``command``, root velocity and
    quaternion arrays, ordered foot contact/position arrays, and
    ``applied_torque``.  When the complete ``physics_*`` touchdown group is
    supplied, gait and touchdown metrics consume those physics-rate samples and
    exact monitor events instead of reconstructing contact edges at policy rate.
    """

    required = {
        "termination",
        "command",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "root_quat_w",
        "foot_contact",
        "foot_pos_w",
        "applied_torque",
    }
    if not isinstance(telemetry, Mapping):
        raise GaitMetricError("telemetry must be a mapping.")
    missing = sorted(required - set(telemetry))
    if missing:
        raise GaitMetricError(f"telemetry is missing required keys: {missing}.")

    physics_required = {
        "physics_command",
        "physics_root_quat_w",
        "physics_foot_contact",
        "physics_foot_pos_w",
        "physics_touchdown_event",
        "physics_touchdown_air_time_s",
        "physics_touchdown_preimpact_speed_m_s",
    }
    physics_present = physics_required & set(telemetry)
    if physics_present and physics_present != physics_required:
        missing_physics = sorted(physics_required - physics_present)
        raise GaitMetricError(f"Physics touchdown telemetry is incomplete; missing keys: {missing_physics}.")
    if physics_present:
        if touchdown_step_dt is None:
            raise GaitMetricError("touchdown_step_dt is required with physics touchdown telemetry.")
        gait_step_dt = _positive_float(touchdown_step_dt, "touchdown_step_dt")
        gait_contact = telemetry["physics_foot_contact"]
        touchdown_contact = telemetry["physics_foot_contact"]
        touchdown_position = telemetry["physics_foot_pos_w"]
        touchdown_quaternion = telemetry["physics_root_quat_w"]
        touchdown_command = telemetry["physics_command"]
        touchdown_event = telemetry["physics_touchdown_event"]
        touchdown_air_time = telemetry["physics_touchdown_air_time_s"]
        touchdown_preimpact = telemetry["physics_touchdown_preimpact_speed_m_s"]
    else:
        if touchdown_step_dt is not None:
            raise GaitMetricError("touchdown_step_dt was supplied without the complete physics touchdown group.")
        gait_step_dt = step_dt
        gait_contact = telemetry["foot_contact"]
        touchdown_contact = telemetry["foot_contact"]
        touchdown_position = telemetry["foot_pos_w"]
        touchdown_quaternion = telemetry["root_quat_w"]
        touchdown_command = telemetry["command"]
        touchdown_event = None
        touchdown_air_time = None
        touchdown_preimpact = None

    command = _float_array(telemetry["command"], "telemetry.command", ndim=2)
    root_lin_vel = _float_array(telemetry["root_lin_vel_w"], "telemetry.root_lin_vel_w", ndim=2)
    root_ang_vel = _float_array(telemetry["root_ang_vel_w"], "telemetry.root_ang_vel_w", ndim=2)
    sample_count = _matching_sample_count((
        ("command", command),
        ("root_lin_vel_w", root_lin_vel),
        ("root_ang_vel_w", root_ang_vel),
    ))

    result: dict[str, Any] = {
        "survival": survival_fall_metrics(
            telemetry["termination"], horizon_steps=horizon_steps, step_dt=step_dt, timeout=timeout
        ),
        "tracking": {
            "yaw_frame_linear_velocity_rmse_m_s": yaw_frame_linear_velocity_rmse(
                root_lin_vel,
                telemetry["root_quat_w"],
                command,
            ),
            "yaw_rate_rmse_rad_s": yaw_rate_rmse(root_ang_vel, command),
        },
        "posture": root_tilt_metrics(
            telemetry["root_quat_w"],
            command,
            command_speed_threshold_m_s=command_speed_threshold_m_s,
        ),
        "gait_symmetry": gait_symmetry_metrics(gait_contact, step_dt=gait_step_dt),
        "touchdown": touchdown_metrics(
            touchdown_contact,
            touchdown_position,
            touchdown_quaternion,
            touchdown_command,
            step_dt=gait_step_dt,
            minimum_progress_m=minimum_touchdown_progress_m,
            tap_max_air_time_s=tap_max_air_time_s,
            command_speed_threshold_m_s=command_speed_threshold_m_s,
            touchdown_event=touchdown_event,
            preceding_air_time_s=touchdown_air_time,
            preimpact_speed_m_s=touchdown_preimpact,
        ),
        "torque_saturation": torque_saturation_metrics(
            telemetry["applied_torque"],
            effort_limits,
            step_dt=step_dt,
            threshold_fraction=torque_saturation_threshold_fraction,
            joint_names=joint_names,
        ),
    }

    result["push_recovery"] = _push_recovery_not_applicable()
    if push_end_step is not None:
        if (
            linear_velocity_error_threshold_m_s is None
            or yaw_rate_error_threshold_rad_s is None
            or recovery_dwell_s is None
        ):
            raise GaitMetricError("Push recovery requires both thresholds and recovery_dwell_s.")
        yaw = root_yaw_from_wxyz(telemetry["root_quat_w"])
        actual_yaw_velocity = world_xy_to_yaw_frame(root_lin_vel[:, :2], yaw)
        linear_error = actual_yaw_velocity - command[:, :2]
        yaw_error = root_ang_vel[:, 2] - command[:, 2]
        if linear_error.shape[0] != sample_count:
            raise GaitMetricError("Push recovery sample counts are inconsistent.")
        if push_delivery_complete:
            recovery_metrics = push_recovery_metrics(
                linear_error,
                yaw_error,
                push_end_step=push_end_step,
                step_dt=step_dt,
                linear_velocity_threshold_m_s=linear_velocity_error_threshold_m_s,
                yaw_rate_threshold_rad_s=yaw_rate_error_threshold_rad_s,
                dwell_s=recovery_dwell_s,
            )
            result["push_recovery"] = {
                "applicable": True,
                "delivery_complete": True,
                **recovery_metrics,
            }
        else:
            result["push_recovery"] = _push_recovery_delivery_censored(
                sample_count=sample_count,
                push_end_step=push_end_step,
                step_dt=step_dt,
                dwell_s=recovery_dwell_s,
            )

    return to_jsonable(result)


__all__ = [
    "FOOT_ORDER",
    "GaitMetricError",
    "aggregate_episode_metrics",
    "aggregate_evaluation_results",
    "evaluate_episode_metrics",
    "gait_symmetry_metrics",
    "push_recovery_metrics",
    "root_tilt_metrics",
    "root_yaw_from_wxyz",
    "survival_fall_metrics",
    "to_jsonable",
    "torque_saturation_metrics",
    "touchdown_metrics",
    "world_xy_to_yaw_frame",
    "yaw_frame_linear_velocity_rmse",
    "yaw_rate_rmse",
]
