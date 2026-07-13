"""Cycle-domain identification for RND STEP encoder/current telemetry.

The collector schedules commands on a fixed control index, while USB status
packet completion times contain deterministic jitter. Identification therefore
uses the configured control period for command-domain dynamics and retains the
host/device timing only as diagnostics.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import IdentificationConfig
from .dataset import Real2SimDataset


MODEL_SCHEMA_VERSION = 2


class IdentificationError(RuntimeError):
    """Raised when a dataset cannot support a defensible fit."""


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    residual = target - prediction
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(target - np.mean(target))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(residual))) / denominator
    return {"rmse": rmse, "r2": r2}


def _quantiles(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise IdentificationError("Cannot summarize an empty or non-finite parameter sample.")
    return float(np.median(array)), float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))


def _expanded_range(q25: float, q75: float, margin: float, additive: float = 0.0) -> list[float]:
    return [max(0.0, q25 * (1.0 - margin) - additive), q75 * (1.0 + margin) + additive]


def _unwrap_ticks(raw_ticks: np.ndarray) -> np.ndarray:
    increments = np.diff(raw_ticks.astype(np.int64), axis=0)
    increments[increments < 0] += 32768
    return increments


def _timing_diagnostics(dataset: Real2SimDataset) -> dict[str, Any]:
    sample_hz = float(dataset.metadata.get("sample_hz", 0.0))
    if not math.isfinite(sample_hz) or sample_hz <= 0.0:
        raise IdentificationError("Dataset metadata.sample_hz must be positive.")
    nominal_dt = 1.0 / sample_hz
    host_dt = np.diff(dataset.arrays["time_s"])
    if host_dt.size == 0:
        raise IdentificationError("Dataset contains too few samples for timing diagnostics.")
    device_dt_ms = _unwrap_ticks(dataset.arrays["device_tick_ms"])
    valid_device_dt = device_dt_ms[(device_dt_ms > 0) & (device_dt_ms < 1000)]
    host_elapsed = float(dataset.arrays["time_s"][-1] - dataset.arrays["time_s"][0])
    nominal_elapsed = (dataset.sample_count - 1) * nominal_dt
    elapsed_error_fraction = abs(host_elapsed - nominal_elapsed) / max(nominal_elapsed, nominal_dt)
    return {
        "time_base_used": "fixed command index from metadata.sample_hz",
        "nominal_sample_hz": sample_hz,
        "nominal_sample_period_s": nominal_dt,
        "host_elapsed_s": host_elapsed,
        "nominal_elapsed_s": nominal_elapsed,
        "elapsed_error_fraction": elapsed_error_fraction,
        "host_read_completion_dt_s": {
            "mean": float(np.mean(host_dt)),
            "median": float(np.median(host_dt)),
            "q25": float(np.quantile(host_dt, 0.25)),
            "q75": float(np.quantile(host_dt, 0.75)),
            "minimum": float(np.min(host_dt)),
            "maximum": float(np.max(host_dt)),
        },
        "device_tick_dt_s": {
            "mean": float(np.mean(valid_device_dt) / 1000.0),
            "median": float(np.median(valid_device_dt) / 1000.0),
        },
        "quality_pass": elapsed_error_fraction <= 0.02,
    }


def _phase_metadata(dataset: Real2SimDataset, phase_id: int) -> dict[str, Any]:
    phase_table = dataset.metadata.get("phase_metadata")
    if not isinstance(phase_table, dict):
        raise IdentificationError("Dataset metadata.phase_metadata is missing.")
    metadata = phase_table.get(str(phase_id))
    if not isinstance(metadata, dict):
        raise IdentificationError(f"Metadata for phase {phase_id} is missing.")
    return metadata


def _joint_phases(dataset: Real2SimDataset, joint_index: int) -> list[tuple[int, np.ndarray, dict[str, Any]]]:
    phase_ids = dataset.arrays["phase_id"]
    excitation_ids = dataset.arrays["excitation_joint_id"]
    phases = []
    for phase_id in sorted(int(value) for value in np.unique(phase_ids) if value >= 0):
        indices = np.flatnonzero((phase_ids == phase_id) & (excitation_ids == joint_index))
        if indices.size:
            phases.append((phase_id, indices, _phase_metadata(dataset, phase_id)))
    if not phases:
        raise IdentificationError(f"No excitation phase exists for joint index {joint_index}.")
    return phases


def _expected_phase_shape(metadata: dict[str, Any], sample_hz: float) -> tuple[int, int]:
    frequency_hz = float(metadata["frequency_hz"])
    cycles = int(metadata["cycles"])
    if frequency_hz <= 0.0 or cycles <= 0:
        raise IdentificationError(f"Invalid phase frequency/cycles: {metadata}")
    samples_per_cycle_float = sample_hz / frequency_hz
    samples_per_cycle = round(samples_per_cycle_float)
    if abs(samples_per_cycle_float - samples_per_cycle) > 1.0e-6:
        raise IdentificationError(
            f"Profile {metadata.get('profile_name')} does not contain an integer number of samples per cycle."
        )
    return samples_per_cycle, samples_per_cycle * cycles


def _harmonic_fit(signal: np.ndarray, frequency_hz: float, sample_hz: float) -> dict[str, Any]:
    sample_time = np.arange(signal.size, dtype=np.float64) / sample_hz
    phase = 2.0 * math.pi * frequency_hz * sample_time
    design = np.column_stack((np.ones(signal.size), np.sin(phase), np.cos(phase)))
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    prediction = design @ coefficients
    return {
        "offset": float(coefficients[0]),
        "amplitude": float(math.hypot(coefficients[1], coefficients[2])),
        "phase_rad": float(math.atan2(coefficients[2], coefficients[1])),
        "prediction": prediction,
        "metrics": _metrics(signal, prediction),
    }


def _phase_lag(input_phase: float, output_phase: float) -> float:
    lag = (input_phase - output_phase) % (2.0 * math.pi)
    return lag - 2.0 * math.pi if lag > math.pi else lag


def _frequency_response(
    dataset: Real2SimDataset,
    joint_index: int,
    phase_id: int,
    indices: np.ndarray,
    metadata: dict[str, Any],
    config: IdentificationConfig,
) -> dict[str, Any]:
    sample_hz = float(dataset.metadata["sample_hz"])
    frequency_hz = float(metadata["frequency_hz"])
    samples_per_cycle, expected_samples = _expected_phase_shape(metadata, sample_hz)
    if indices.size != expected_samples:
        raise IdentificationError(f"Phase {phase_id} has {indices.size} samples; expected {expected_samples}.")
    goal = dataset.arrays["goal_position_rad"][indices, joint_index]
    position = dataset.arrays["position_rad"][indices, joint_index]
    cycle_results = []
    for cycle in range(int(metadata["cycles"])):
        start = cycle * samples_per_cycle
        stop = start + samples_per_cycle
        goal_fit = _harmonic_fit(goal[start:stop], frequency_hz, sample_hz)
        position_fit = _harmonic_fit(position[start:stop], frequency_hz, sample_hz)
        if goal_fit["amplitude"] <= 1.0e-9:
            raise IdentificationError(f"Phase {phase_id} has no measurable command amplitude.")
        lag_rad = _phase_lag(goal_fit["phase_rad"], position_fit["phase_rad"])
        cycle_results.append({
            "cycle": cycle,
            "gain": position_fit["amplitude"] / goal_fit["amplitude"],
            "phase_lag_rad": lag_rad,
            "equivalent_delay_s": lag_rad / (2.0 * math.pi * frequency_hz),
            "output_fit_r2": position_fit["metrics"]["r2"],
            "output_fit_rmse_rad": position_fit["metrics"]["rmse"],
        })

    delays = [result["equivalent_delay_s"] for result in cycle_results]
    gains = [result["gain"] for result in cycle_results]
    delay_median, delay_q25, delay_q75 = _quantiles(delays)
    gain_median, gain_q25, gain_q75 = _quantiles(gains)
    full_output_fit = _harmonic_fit(position, frequency_hz, sample_hz)
    minimum_r2 = min(result["output_fit_r2"] for result in cycle_results if result["output_fit_r2"] is not None)
    delay_iqr = delay_q75 - delay_q25
    quality_pass = (
        len(cycle_results) >= 3
        and 0.5 <= gain_median <= 1.2
        and 0.0 <= delay_median <= config.max_delay_s
        and delay_iqr <= 1.0 / sample_hz
        and minimum_r2 >= 0.95
    )
    amplitude_rad = float(metadata["amplitude_rad"])
    return {
        "phase_id": phase_id,
        "profile_name": metadata["profile_name"],
        "frequency_hz": frequency_hz,
        "amplitude_rad": amplitude_rad,
        "peak_command_velocity_rad_s": 2.0 * math.pi * frequency_hz * amplitude_rad,
        "cycle_count": len(cycle_results),
        "cycles": cycle_results,
        "gain": {"median": gain_median, "q25": gain_q25, "q75": gain_q75},
        "equivalent_delay_s": {"median": delay_median, "q25": delay_q25, "q75": delay_q75},
        "full_output_fit": {
            "r2": full_output_fit["metrics"]["r2"],
            "rmse_rad": full_output_fit["metrics"]["rmse"],
        },
        "quality_pass": quality_pass,
    }


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    sample_index = np.arange(signal.size, dtype=np.float64)
    return np.interp(sample_index - delay_samples, sample_index, signal, left=signal[0], right=signal[-1])


def _play_transform(command: np.ndarray, half_width_rad: float) -> np.ndarray:
    output = np.empty_like(command)
    output[0] = command[0]
    for index in range(1, command.size):
        if command[index] > output[index - 1] + half_width_rad:
            output[index] = command[index] - half_width_rad
        elif command[index] < output[index - 1] - half_width_rad:
            output[index] = command[index] + half_width_rad
        else:
            output[index] = output[index - 1]
    return output


def _backlash_from_triangle(
    dataset: Real2SimDataset,
    joint_index: int,
    phase_id: int,
    indices: np.ndarray,
    metadata: dict[str, Any],
    reference_delay_s: float,
    config: IdentificationConfig,
) -> dict[str, Any]:
    sample_hz = float(dataset.metadata["sample_hz"])
    samples_per_cycle, expected_samples = _expected_phase_shape(metadata, sample_hz)
    if indices.size != expected_samples:
        raise IdentificationError(f"Triangle phase {phase_id} has {indices.size} samples; expected {expected_samples}.")
    goal = dataset.arrays["goal_position_rad"][indices, joint_index]
    position = dataset.arrays["position_rad"][indices, joint_index]
    delayed_goal = _fractional_delay(goal, reference_delay_s * sample_hz)
    cycle_results = []
    for cycle in range(int(metadata["cycles"])):
        start = cycle * samples_per_cycle
        stop = start + samples_per_cycle
        cycle_goal = delayed_goal[start:stop]
        cycle_position = position[start:stop]
        direction = np.gradient(cycle_goal)
        center = 0.5 * (float(np.min(cycle_goal)) + float(np.max(cycle_goal)))
        amplitude = 0.5 * (float(np.max(cycle_goal)) - float(np.min(cycle_goal)))
        residual = cycle_goal - cycle_position
        mid_stroke = np.abs(cycle_goal - center) <= 0.6 * amplitude
        rising = residual[(direction > 0.0) & mid_stroke]
        falling = residual[(direction < 0.0) & mid_stroke]
        if rising.size < 10 or falling.size < 10:
            continue
        rising_offset = float(np.median(rising))
        falling_offset = float(np.median(falling))
        cycle_results.append({
            "cycle": cycle,
            "dead_travel_rad": rising_offset - falling_offset,
            "center_bias_rad": 0.5 * (rising_offset + falling_offset),
            "rising_sample_count": int(rising.size),
            "falling_sample_count": int(falling.size),
        })
    if not cycle_results:
        raise IdentificationError(f"Triangle phase {phase_id} produced no branch-paired backlash cycles.")

    dead_travel = [result["dead_travel_rad"] for result in cycle_results]
    median, q25, q75 = _quantiles(dead_travel)
    positive_fraction = float(np.mean(np.asarray(dead_travel) > 0.0))
    relative_iqr = (q75 - q25) / max(abs(median), 1.0e-9)
    quality_pass = (
        len(cycle_results) >= config.min_reversal_events
        and positive_fraction >= 0.9
        and 0.0 < median <= math.radians(5.0)
        and relative_iqr <= 0.35
    )

    effective_goal = _play_transform(delayed_goal, 0.5 * max(0.0, median))
    train_cycles = max(1, int(len(cycle_results) * (1.0 - config.validation_fraction)))
    split = min(train_cycles * samples_per_cycle, effective_goal.size - samples_per_cycle)
    train_design = np.column_stack((np.ones(split), effective_goal[:split]))
    affine = np.linalg.lstsq(train_design, position[:split], rcond=None)[0]
    validation_prediction = affine[0] + affine[1] * effective_goal[split:]
    validation_metrics = _metrics(position[split:], validation_prediction)
    if validation_metrics["r2"] is None or validation_metrics["r2"] < 0.95:
        quality_pass = False
    return {
        "phase_id": phase_id,
        "profile_name": metadata["profile_name"],
        "definition": "rising/falling command-position hysteresis after fractional delay compensation",
        "reference_delay_s": reference_delay_s,
        "median_rad": median,
        "q25_rad": q25,
        "q75_rad": q75,
        "play_half_width_rad": 0.5 * median,
        "branch_pair_count": len(cycle_results),
        "positive_fraction": positive_fraction,
        "relative_iqr": relative_iqr,
        "cycles": cycle_results,
        "play_model": {
            "affine_offset_rad": float(affine[0]),
            "gain": float(affine[1]),
            "validation": validation_metrics,
        },
        "quality_pass": quality_pass,
    }


def _coulomb_from_matched_branches(
    dataset: Real2SimDataset,
    joint_index: int,
    phase_id: int,
    indices: np.ndarray,
    metadata: dict[str, Any],
    config: IdentificationConfig,
) -> dict[str, Any]:
    sample_hz = float(dataset.metadata["sample_hz"])
    samples_per_cycle, expected_samples = _expected_phase_shape(metadata, sample_hz)
    if indices.size != expected_samples:
        raise IdentificationError(f"Friction phase {phase_id} has {indices.size} samples; expected {expected_samples}.")
    position = dataset.arrays["position_rad"][indices, joint_index]
    velocity = dataset.arrays["velocity_rad_s"][indices, joint_index]
    current = dataset.arrays["current_a"][indices, joint_index]
    cycle_results = []
    all_bin_estimates = []
    for cycle in range(int(metadata["cycles"])):
        start = cycle * samples_per_cycle
        stop = start + samples_per_cycle
        cycle_position = position[start:stop]
        cycle_velocity = velocity[start:stop]
        cycle_current = current[start:stop]
        edges = np.linspace(float(np.min(cycle_position)), float(np.max(cycle_position)), 11)
        bin_estimates = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            in_bin = (cycle_position >= lower) & (cycle_position < upper)
            positive = in_bin & (cycle_velocity >= config.velocity_threshold_rad_s)
            negative = in_bin & (cycle_velocity <= -config.velocity_threshold_rad_s)
            if np.count_nonzero(positive) < 2 or np.count_nonzero(negative) < 2:
                continue
            positive_current = float(np.median(cycle_current[positive]))
            negative_current = float(np.median(cycle_current[negative]))
            bin_estimates.append(0.5 * (positive_current - negative_current))
        if len(bin_estimates) < 5:
            continue
        estimate = float(np.median(bin_estimates))
        cycle_results.append({
            "cycle": cycle,
            "coulomb_current_a": estimate,
            "bin_count": len(bin_estimates),
            "positive_bin_fraction": float(np.mean(np.asarray(bin_estimates) > 0.0)),
        })
        all_bin_estimates.extend(bin_estimates)
    if not cycle_results:
        raise IdentificationError(f"Sine phase {phase_id} produced no matched-direction current cycles.")

    cycle_estimates = [result["coulomb_current_a"] for result in cycle_results]
    median, q25, q75 = _quantiles(cycle_estimates)
    positive_bin_fraction = float(np.mean(np.asarray(all_bin_estimates) > 0.0))
    relative_iqr = (q75 - q25) / max(abs(median), 1.0e-9)
    quality_pass = len(cycle_results) >= 3 and median > 0.0 and positive_bin_fraction >= 0.7 and relative_iqr <= 0.75
    return {
        "phase_id": phase_id,
        "profile_name": metadata["profile_name"],
        "domain": "joint-coordinate motor current; not a direct output-torque measurement",
        "method": "half current difference between positive/negative motion at matched joint positions",
        "coulomb_current_a": median,
        "q25_a": q25,
        "q75_a": q75,
        "cycle_count": len(cycle_results),
        "positive_bin_fraction": positive_bin_fraction,
        "relative_iqr": relative_iqr,
        "cycles": cycle_results,
        "viscous_identified": False,
        "viscous_a_per_rad_s": None,
        "quality_pass": quality_pass,
    }


def _identify_joint(
    dataset: Real2SimDataset,
    joint_index: int,
    config: IdentificationConfig,
) -> dict[str, Any]:
    phases = _joint_phases(dataset, joint_index)
    sine_phases = [phase for phase in phases if phase[2].get("waveform") == "sine"]
    triangle_phases = [phase for phase in phases if phase[2].get("waveform") == "triangle"]
    if not sine_phases or not triangle_phases:
        raise IdentificationError("Identification requires at least one sine and one triangle phase.")

    responses = [
        _frequency_response(dataset, joint_index, phase_id, indices, metadata, config)
        for phase_id, indices, metadata in sine_phases
    ]
    reference_response = max(responses, key=lambda item: item["peak_command_velocity_rad_s"])
    reference_delay = float(reference_response["equivalent_delay_s"]["median"])
    triangle_phase = min(
        triangle_phases,
        key=lambda item: 4.0 * float(item[2]["frequency_hz"]) * float(item[2]["amplitude_rad"]),
    )
    backlash = _backlash_from_triangle(
        dataset,
        joint_index,
        triangle_phase[0],
        triangle_phase[1],
        triangle_phase[2],
        reference_delay,
        config,
    )
    friction_phase = min(
        sine_phases,
        key=lambda item: (2.0 * math.pi * float(item[2]["frequency_hz"])) ** 2 * float(item[2]["amplitude_rad"]),
    )
    friction = _coulomb_from_matched_branches(
        dataset,
        joint_index,
        friction_phase[0],
        friction_phase[1],
        friction_phase[2],
        config,
    )

    warnings = []
    if not reference_response["quality_pass"]:
        warnings.append("Reference sine frequency response failed its repeatability/fit quality gate.")
    if not backlash["quality_pass"]:
        warnings.append("Triangle hysteresis estimate failed its cycle consistency or play-model quality gate.")
    if not friction["quality_pass"]:
        warnings.append("Matched-direction Coulomb-current estimate failed its consistency gate.")
    response_delays = [item["equivalent_delay_s"]["median"] for item in responses]
    if max(response_delays) - min(response_delays) > 1.0 / float(dataset.metadata["sample_hz"]):
        warnings.append(
            "Equivalent phase delay is rate-dependent; use the fastest profile for target delay and the slow profile only as a diagnostic."
        )
    warnings.append("Viscous friction is not separately identifiable from this suspended encoder/current dataset.")

    target_usable = bool(reference_response["quality_pass"] and backlash["quality_pass"])
    coulomb_usable = bool(friction["quality_pass"])
    if target_usable and coulomb_usable:
        status = "usable_target_and_coulomb"
    elif target_usable:
        status = "usable_target_only"
    else:
        status = "rejected"

    margin = config.randomization_margin
    dt = 1.0 / float(dataset.metadata["sample_hz"])
    delay_range = None
    backlash_range = None
    if target_usable:
        delay_range = _expanded_range(
            reference_response["equivalent_delay_s"]["q25"],
            reference_response["equivalent_delay_s"]["q75"],
            margin=0.0,
            additive=dt,
        )
        backlash_range = _expanded_range(backlash["q25_rad"], backlash["q75_rad"], margin=margin)
    coulomb_range = None
    if coulomb_usable:
        coulomb_range = _expanded_range(friction["q25_a"], friction["q75_a"], margin=margin)

    nominal_ratio = config.nominal_torque_per_amp_nm
    return {
        "joint_index": joint_index,
        "command_delay": {
            "definition": "phase-equivalent command-to-position delay from the fastest sine profile",
            "reference_profile": reference_response["profile_name"],
            "seconds": reference_delay,
            "control_samples_equivalent": reference_delay / dt,
            "q25_s": reference_response["equivalent_delay_s"]["q25"],
            "q75_s": reference_response["equivalent_delay_s"]["q75"],
        },
        "frequency_response": responses,
        "effective_backlash": backlash,
        "friction_current_model": friction,
        "nominal_torque_proxy": {
            "warning": "Uses a nominal torque/current ratio; it is not measured joint torque.",
            "torque_per_amp_nm": nominal_ratio,
            "coulomb_nm": friction["coulomb_current_a"] * nominal_ratio,
            "viscous_nm_per_rad_s": None,
        },
        "randomization": {
            "command_delay_s": delay_range,
            "effective_backlash_rad": backlash_range,
            "coulomb_positive_a": coulomb_range,
            "coulomb_negative_a": coulomb_range,
            "viscous_a_per_rad_s": None,
        },
        "quality": {
            "status": status,
            "target_randomization_usable": target_usable,
            "coulomb_randomization_usable": coulomb_usable,
            "viscous_randomization_usable": False,
        },
        "warnings": warnings,
    }


def identify_dataset(dataset: Real2SimDataset, config: IdentificationConfig) -> dict[str, Any]:
    excitation_names = dataset.metadata.get("excitation_joint_names")
    if not isinstance(excitation_names, list) or not excitation_names:
        raise IdentificationError("Dataset metadata contains no excitation_joint_names.")
    name_to_index = {name: index for index, name in enumerate(dataset.joint_names)}
    unknown = sorted(set(excitation_names) - set(name_to_index))
    if unknown:
        raise IdentificationError(f"Dataset excitation joints are not in joint_names: {unknown}")
    timing = _timing_diagnostics(dataset)
    if not timing["quality_pass"]:
        raise IdentificationError(
            "Dataset elapsed time differs from the configured control index by more than 2%; recollect before fitting."
        )

    joints = {}
    failures = {}
    for name in excitation_names:
        try:
            joints[name] = _identify_joint(dataset, name_to_index[name], config)
        except (IdentificationError, KeyError, ValueError) as error:
            failures[name] = str(error)
    if not joints:
        raise IdentificationError(f"No joint could be identified. Failures={failures}")
    status_counts: dict[str, int] = {}
    for result in joints.values():
        status = result["quality"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "rnd_encoder_domain_equivalent_actuator",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset.path),
        "source_dataset_sha256": dataset.sha256,
        "source_dataset_dry_run": bool(dataset.metadata.get("dry_run")),
        "sample_hz": float(dataset.metadata["sample_hz"]),
        "timing": timing,
        "reference_pose": dataset.metadata.get("reference_pose"),
        "imu_used": False,
        "assumption": "Upper body was rigidly constrained and did not swing during suspended-leg excitation.",
        "limitations": [
            "Encoder/current data identifies equivalent closed-loop behavior, not unique physical gear parameters.",
            "The reported delay is phase-equivalent at the reference sine profile, not pure bus transport latency.",
            "Backlash is low-speed command-position hysteresis after delay compensation and includes residual compliance/static friction.",
            "Only Coulomb-like directional current is retained; viscous friction is not separately identifiable in this experiment.",
            "Nominal torque proxies are not substitutes for an external torque sensor or calibrated test bench.",
        ],
        "identification_config": {
            "max_delay_s": config.max_delay_s,
            "velocity_threshold_rad_s": config.velocity_threshold_rad_s,
            "min_reversal_events": config.min_reversal_events,
            "validation_fraction": config.validation_fraction,
            "nominal_torque_per_amp_nm": config.nominal_torque_per_amp_nm,
            "randomization_margin": config.randomization_margin,
        },
        "quality_summary": status_counts,
        "joints": joints,
        "failed_joints": failures,
    }


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(model, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def model_report(model: dict[str, Any]) -> str:
    lines = [
        "RND STEP cycle-domain actuator identification",
        f"source: {model['source_dataset']}",
        "",
        "joint                     delay(ms)  backlash(deg)  Coulomb(A)  ref gain  fit R2   status",
    ]
    for name, result in model["joints"].items():
        reference_name = result["command_delay"]["reference_profile"]
        reference = next(item for item in result["frequency_response"] if item["profile_name"] == reference_name)
        lines.append(
            f"{name:25s} {1000.0 * result['command_delay']['seconds']:9.2f}  "
            f"{math.degrees(result['effective_backlash']['median_rad']):13.3f}  "
            f"{result['friction_current_model']['coulomb_current_a']:10.4f}  "
            f"{reference['gain']['median']:8.4f}  "
            f"{reference['full_output_fit']['r2']:7.4f}  {result['quality']['status']}"
        )
        for warning in result["warnings"]:
            lines.append(f"  warning: {warning}")
    if model["failed_joints"]:
        lines.extend(("", f"failed joints: {model['failed_joints']}"))
    lines.extend((
        "",
        "Only parameters whose quality flags are true are exported as randomization ranges.",
        "Torque values remain nominal proxies, not measurements.",
    ))
    return "\n".join(lines)
