#!/usr/bin/env python3
"""Replay one RND hardware excitation in fixed-base Isaac simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


_TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = _TOOL_DIR / "config" / "rnd_actuator_model.json"

parser = argparse.ArgumentParser(
    description=(
        "Drive a fixed-base STEP joint with a real2sim goal trace and compare the explicit-PD simulator response "
        "with the hardware encoder response. This tool never edits the actuator model or training config."
    )
)
parser.add_argument("--dataset", help="Complete single-joint rnd_real2sim .npz dataset.")
parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Integration-gated actuator seed JSON.")
parser.add_argument("--joint", help="Excited joint; inferred when the dataset contains exactly one.")
parser.add_argument("--output-prefix", help="Output path prefix for .npz and .json reports.")
parser.add_argument("--settle-s", type=float, default=3.0, help="Initial fixed-pose settle duration.")
parser.add_argument("--stiffness", type=float, help="Override the model seed stiffness for this replay only.")
parser.add_argument("--damping", type=float, help="Override the model seed damping for this replay only.")
parser.add_argument(
    "--residual-delay-s",
    type=float,
    help=(
        "Override the command-path residual delay for this replay only. Fractional physics-step delays are "
        "supported; the model JSON is not edited."
    ),
)
parser.add_argument(
    "--position-bias-rad",
    type=float,
    help=(
        "Override the additive residual target-position bias for this replay only. With --sweep-pd and no override, "
        "a bias is calibrated only when a constant shift can make every phase pass its shape gate."
    ),
)
parser.add_argument(
    "--sweep-pd",
    action="store_true",
    help=(
        "Sweep stiffness/damping on the fastest sine phase, select a residual-delay-compensable candidate, then "
        "run the selected gains over the complete trace. The model JSON is never edited."
    ),
)
parser.add_argument(
    "--stiffness-scales",
    default="1.0,1.25,1.5,1.75,2.0",
    help="Comma-separated stiffness multipliers relative to the model seed for --sweep-pd.",
)
parser.add_argument(
    "--damping-scales",
    default="0.6,0.8,1.0,1.2,1.4",
    help="Comma-separated damping multipliers relative to the model seed for --sweep-pd.",
)
parser.add_argument(
    "--allow-unresolved",
    action="store_true",
    help="Permit the unresolved identity placeholder for diagnostics; it can never pass the integration gate.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import copy
import json
import math
import os
import tempfile
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_lab.actuators.rnd_isaac import RndEquivalentActuatorCfg
from robot_lab.actuators.rnd_stateful import RndActuatorModelError, load_rnd_actuator_model
from robot_lab.assets.rnd import STEP_CFG
from rnd_actuator_sweep import (
    PDSweepError,
    build_pd_candidates,
    parse_positive_scales,
    select_pd_candidate,
)


class SimReplayError(ValueError):
    """Raised when a trace or simulator configuration cannot support replay."""


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as error:
        raise SimReplayError(f"Unable to load dataset {path}: {error}") from error
    if metadata.get("status") != "complete":
        raise SimReplayError(f"Dataset status must be complete, got {metadata.get('status')!r}.")
    for name in ("goal_position_rad", "position_rad", "phase_id"):
        if name not in arrays:
            raise SimReplayError(f"Dataset is missing {name!r}.")
    return arrays, metadata


def _select_joint(metadata: dict[str, Any], requested: str | None) -> str:
    joint_names = metadata.get("joint_names")
    excited = metadata.get("excitation_joint_names")
    if not isinstance(joint_names, list) or not isinstance(excited, list):
        raise SimReplayError("Dataset metadata is missing joint_names or excitation_joint_names.")
    if requested is not None:
        if requested not in excited:
            raise SimReplayError(f"Requested joint {requested!r} was not excited; available={excited}.")
        return requested
    if len(excited) != 1:
        raise SimReplayError(f"Dataset excites {excited}; select one with --joint.")
    return str(excited[0])


def _configure_robot(model: dict[str, Any], joint_name: str, model_path: Path) -> Articulation:
    robot_cfg = copy.deepcopy(STEP_CFG)
    robot_cfg.prim_path = "/World/RND_STEP"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
    robot_cfg.spawn.fix_base = True
    robot_cfg.spawn.activate_contact_sensors = False
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False

    seed = model["joints"][joint_name]["controller_seed"]
    source_group = seed["source_group"]
    if source_group not in robot_cfg.actuators:
        raise SimReplayError(f"Controller seed source group {source_group!r} is absent from STEP_CFG.")
    hold_cfg = robot_cfg.actuators.pop(source_group)
    remaining = [name for name in hold_cfg.joint_names_expr if name != joint_name]
    if len(remaining) == len(hold_cfg.joint_names_expr):
        raise SimReplayError(f"Joint {joint_name} is absent from STEP_CFG actuator group {source_group}.")
    if remaining:
        hold_cfg.joint_names_expr = remaining
        robot_cfg.actuators[f"{source_group}_hold"] = hold_cfg

    robot_cfg.actuators[f"{joint_name}_replay"] = RndEquivalentActuatorCfg(
        joint_names_expr=[joint_name],
        model_path=str(model_path),
        physics_hz=float(model["physics_hz"]),
        sample_randomization=False,
        allow_unvalidated_model=True,
        allow_unresolved_joints=args_cli.allow_unresolved,
        stiffness=float(seed["stiffness"]),
        damping=float(seed["damping"]),
        effort_limit=float(seed["effort_limit_nm"]),
        velocity_limit=float(seed["velocity_limit_rad_s"]),
        armature=float(seed["armature"]),
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    return Articulation(robot_cfg)


def _set_replay_pd(robot: Articulation, joint_name: str, stiffness: float, damping: float) -> None:
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise SimReplayError(f"Stiffness must be finite and positive, got {stiffness!r}.")
    if not math.isfinite(damping) or damping < 0.0:
        raise SimReplayError(f"Damping must be finite and non-negative, got {damping!r}.")
    actuator_name = f"{joint_name}_replay"
    if actuator_name not in robot.actuators:
        raise SimReplayError(f"Replay actuator {actuator_name!r} is absent from the articulation.")
    actuator = robot.actuators[actuator_name]
    actuator.stiffness.fill_(stiffness)
    actuator.damping.fill_(damping)


def _set_replay_position_bias(robot: Articulation, joint_name: str, position_bias_rad: float) -> None:
    if not math.isfinite(position_bias_rad):
        raise SimReplayError(f"Position bias must be finite, got {position_bias_rad!r}.")
    actuator_name = f"{joint_name}_replay"
    if actuator_name not in robot.actuators:
        raise SimReplayError(f"Replay actuator {actuator_name!r} is absent from the articulation.")
    actuator = robot.actuators[actuator_name]
    actuator.command_path.set_position_bias_override(position_bias_rad)


def _set_replay_delay(robot: Articulation, joint_name: str, residual_delay_s: float) -> None:
    if not math.isfinite(residual_delay_s) or residual_delay_s < 0.0:
        raise SimReplayError(f"Residual delay must be finite and non-negative, got {residual_delay_s!r}.")
    actuator_name = f"{joint_name}_replay"
    if actuator_name not in robot.actuators:
        raise SimReplayError(f"Replay actuator {actuator_name!r} is absent from the articulation.")
    actuator = robot.actuators[actuator_name]
    actuator.command_path.set_delay_override(residual_delay_s)


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum(np.square(reference - np.mean(reference))))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(error))) / denominator
    span = float(np.ptp(reference))
    return {
        "rmse_rad": rmse,
        "mae_rad": mae,
        "normalized_rmse": None if span <= 1.0e-12 else rmse / span,
        "r2": r2,
    }


def _position_bias_calibration(
    trace: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: dict[str, Any],
    initial_report: dict[str, Any],
    current_position_bias_rad: float,
) -> dict[str, Any]:
    phase_metadata = metadata.get("phase_metadata", {})
    triangle_phase_ids = sorted(
        int(phase_id)
        for phase_id, phase in phase_metadata.items()
        if isinstance(phase, dict) and phase.get("waveform") == "triangle"
    )
    if not triangle_phase_ids:
        return {
            "attempted": False,
            "eligible": False,
            "reason": "Dataset has no triangle phase for low-amplitude center-bias calibration.",
        }

    source_phase_id = triangle_phase_ids[0]
    source_mask = trace["phase_id"] == source_phase_id
    position_bias_delta = float(
        np.mean(trace["hardware_position_rad"][source_mask] - trace["simulated_position_rad"][source_mask])
    )
    thresholds = _gate_thresholds(model)
    corrected_phase_metrics: list[dict[str, Any]] = []
    valid = trace["phase_id"] >= 0
    for phase_id in sorted(int(value) for value in np.unique(trace["phase_id"][valid])):
        mask = trace["phase_id"] == phase_id
        metrics = _metrics(
            trace["hardware_position_rad"][mask],
            trace["simulated_position_rad"][mask] + position_bias_delta,
        )
        corrected_phase_metrics.append({
            "phase_id": phase_id,
            "profile_name": phase_metadata[str(phase_id)]["profile_name"],
            "hardware_vs_shifted_simulation": metrics,
        })

    phase_gate = all(
        phase["hardware_vs_shifted_simulation"]["r2"] is not None
        and phase["hardware_vs_shifted_simulation"]["r2"] >= thresholds["minimum_phase_r2"]
        and phase["hardware_vs_shifted_simulation"]["normalized_rmse"] is not None
        and phase["hardware_vs_shifted_simulation"]["normalized_rmse"] <= thresholds["maximum_phase_normalized_rmse"]
        for phase in corrected_phase_metrics
    )
    reference_phase = next(
        phase for phase in initial_report["phases"] if phase["profile_name"] == initial_report["reference_profile"]
    )
    reference_gate = (
        abs(float(reference_phase["delay_error_s"])) <= thresholds["maximum_reference_delay_error_s"]
        and float(reference_phase["gain_relative_error"]) <= thresholds["maximum_reference_gain_relative_error"]
    )
    within_safety_limit = abs(position_bias_delta) <= math.radians(2.0)
    eligible = bool(phase_gate and reference_gate and within_safety_limit)
    reason = None
    if not phase_gate:
        reason = "A constant position shift cannot make every phase pass the trajectory-shape gate."
    elif not reference_gate:
        reason = "The selected PD candidate still fails the reference sine delay or gain gate."
    elif not within_safety_limit:
        reason = "Required position-bias correction exceeds the 2 deg diagnostic safety limit."
    return {
        "attempted": True,
        "eligible": eligible,
        "reason": reason,
        "source_phase_id": source_phase_id,
        "source_profile": phase_metadata[str(source_phase_id)]["profile_name"],
        "initial_position_bias_rad": current_position_bias_rad,
        "recommended_delta_rad": position_bias_delta,
        "selected_position_bias_rad": current_position_bias_rad + position_bias_delta,
        "offline_shift_phase_gate_satisfied": phase_gate,
        "reference_gate_satisfied": reference_gate,
        "corrected_phase_metrics": corrected_phase_metrics,
    }


def _gate_thresholds(model: dict[str, Any]) -> dict[str, float]:
    return {
        "minimum_phase_r2": 0.95,
        "maximum_phase_normalized_rmse": 0.10,
        "maximum_reference_delay_error_s": 1.0 / float(model["physics_hz"]),
        "maximum_reference_gain_relative_error": 0.10,
    }


def _harmonic_fit(signal: np.ndarray, frequency_hz: float, sample_hz: float) -> tuple[float, float, float | None]:
    time_s = np.arange(signal.size, dtype=np.float64) / sample_hz
    phase = 2.0 * math.pi * frequency_hz * time_s
    design = np.column_stack((np.ones(signal.size), np.sin(phase), np.cos(phase)))
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    prediction = design @ coefficients
    amplitude = float(math.hypot(coefficients[1], coefficients[2]))
    phase_rad = float(math.atan2(coefficients[2], coefficients[1]))
    return amplitude, phase_rad, _metrics(signal, prediction)["r2"]


def _phase_lag(input_phase: float, output_phase: float) -> float:
    lag = (input_phase - output_phase) % (2.0 * math.pi)
    return lag - 2.0 * math.pi if lag > math.pi else lag


def _frequency_response(
    goal: np.ndarray, response: np.ndarray, *, frequency_hz: float, sample_hz: float
) -> dict[str, float | None]:
    goal_amplitude, goal_phase, goal_fit_r2 = _harmonic_fit(goal, frequency_hz, sample_hz)
    output_amplitude, output_phase, output_fit_r2 = _harmonic_fit(response, frequency_hz, sample_hz)
    if goal_amplitude <= 1.0e-12:
        raise SimReplayError("Sine phase has no measurable goal amplitude.")
    lag = _phase_lag(goal_phase, output_phase)
    return {
        "gain": output_amplitude / goal_amplitude,
        "phase_lag_rad": lag,
        "equivalent_delay_s": lag / (2.0 * math.pi * frequency_hz),
        "goal_fit_r2": goal_fit_r2,
        "output_fit_r2": output_fit_r2,
    }


def _reference_phase_id(metadata: dict[str, Any]) -> int:
    phase_metadata = metadata.get("phase_metadata")
    if not isinstance(phase_metadata, dict):
        raise SimReplayError("Dataset metadata is missing phase_metadata.")
    sine_phases: list[tuple[float, int]] = []
    for raw_phase_id, phase in phase_metadata.items():
        if isinstance(phase, dict) and phase.get("waveform") == "sine":
            sine_phases.append((float(phase["frequency_hz"]), int(raw_phase_id)))
    if not sine_phases:
        raise SimReplayError("Dataset contains no sine phase for PD selection.")
    return max(sine_phases)[1]


def _phase_segment(arrays: dict[str, np.ndarray], phase_id: int) -> dict[str, np.ndarray]:
    phase_ids = arrays["phase_id"]
    indices = np.flatnonzero(phase_ids == phase_id)
    if indices.size == 0:
        raise SimReplayError(f"Dataset contains no samples for phase {phase_id}.")
    start = int(indices[0])
    while start > 0 and int(phase_ids[start - 1]) < 0:
        start -= 1
    stop = int(indices[-1]) + 1
    sample_count = phase_ids.shape[0]
    return {
        name: values[start:stop]
        for name, values in arrays.items()
        if isinstance(values, np.ndarray) and values.ndim > 0 and values.shape[0] == sample_count
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replay(
    sim: SimulationContext,
    robot: Articulation,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    joint_name: str,
    settle_s: float,
) -> dict[str, np.ndarray]:
    dataset_names = list(metadata["joint_names"])
    dataset_indices = {name: index for index, name in enumerate(dataset_names)}
    missing = sorted(set(dataset_names) - set(robot.joint_names))
    if missing:
        raise SimReplayError(f"Dataset joints are absent from the simulated articulation: {missing}.")
    robot_indices = torch.tensor(
        [robot.joint_names.index(name) for name in dataset_names], dtype=torch.long, device=robot.device
    )
    selected_robot_index = robot.joint_names.index(joint_name)
    physics_hz = 1.0 / sim.get_physics_dt()
    sample_hz = float(metadata["sample_hz"])
    decimation_float = physics_hz / sample_hz
    decimation = round(decimation_float)
    if abs(decimation_float - decimation) > 1.0e-6:
        raise SimReplayError(f"physics_hz/sample_hz must be integral, got {decimation_float}.")

    target = robot.data.default_joint_pos.clone()
    initial = torch.as_tensor(arrays["goal_position_rad"][0], dtype=target.dtype, device=robot.device)
    target[:, robot_indices] = initial
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.reset()

    def step_target(target_value: torch.Tensor) -> None:
        robot.set_joint_position_target(target_value)
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())

    settle_steps = round(settle_s * physics_hz)
    for _ in range(settle_steps):
        step_target(target)

    sample_count = arrays["goal_position_rad"].shape[0]
    simulated_position = np.empty(sample_count, dtype=np.float64)
    simulated_velocity = np.empty(sample_count, dtype=np.float64)
    simulated_effort = np.empty(sample_count, dtype=np.float64)
    for sample_index in range(sample_count):
        dataset_target = torch.as_tensor(
            arrays["goal_position_rad"][sample_index], dtype=target.dtype, device=robot.device
        )
        target[:, robot_indices] = dataset_target
        for _ in range(decimation):
            step_target(target)
        position = float(robot.data.joint_pos[0, selected_robot_index].item())
        velocity = float(robot.data.joint_vel[0, selected_robot_index].item())
        effort = float(robot.data.applied_torque[0, selected_robot_index].item())
        if not all(math.isfinite(value) for value in (position, velocity, effort)):
            raise SimReplayError(f"Simulation became non-finite at replay sample {sample_index}.")
        simulated_position[sample_index] = position
        simulated_velocity[sample_index] = velocity
        simulated_effort[sample_index] = effort

    selected_dataset_index = dataset_indices[joint_name]
    return {
        "goal_position_rad": arrays["goal_position_rad"][:, selected_dataset_index],
        "hardware_position_rad": arrays["position_rad"][:, selected_dataset_index],
        "simulated_position_rad": simulated_position,
        "simulated_velocity_rad_s": simulated_velocity,
        "simulated_effort_nm": simulated_effort,
        "phase_id": arrays["phase_id"],
        "time_s": np.arange(sample_count, dtype=np.float64) / sample_hz,
    }


def _pd_candidate_summary(
    trace: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: dict[str, Any],
    phase_id: int,
    stiffness: float,
    damping: float,
    effort_limit_nm: float,
) -> dict[str, Any]:
    mask = trace["phase_id"] == phase_id
    if not np.any(mask):
        raise SimReplayError(f"Sweep trace is missing reference phase {phase_id}.")
    phase_info = metadata["phase_metadata"][str(phase_id)]
    frequency_hz = float(phase_info["frequency_hz"])
    sample_hz = float(metadata["sample_hz"])
    comparison = _metrics(trace["hardware_position_rad"][mask], trace["simulated_position_rad"][mask])
    hardware_response = _frequency_response(
        trace["goal_position_rad"][mask],
        trace["hardware_position_rad"][mask],
        frequency_hz=frequency_hz,
        sample_hz=sample_hz,
    )
    simulation_response = _frequency_response(
        trace["goal_position_rad"][mask],
        trace["simulated_position_rad"][mask],
        frequency_hz=frequency_hz,
        sample_hz=sample_hz,
    )
    hardware_delay = float(hardware_response["equivalent_delay_s"])
    simulation_delay = float(simulation_response["equivalent_delay_s"])
    gain_relative_error = abs(float(simulation_response["gain"]) - float(hardware_response["gain"])) / max(
        abs(float(hardware_response["gain"])), 1.0e-9
    )
    max_abs_effort = float(np.max(np.abs(trace["simulated_effort_nm"][mask])))
    max_abs_velocity = float(np.max(np.abs(trace["simulated_velocity_rad_s"][mask])))
    effort_saturated = max_abs_effort >= effort_limit_nm * (1.0 - 1.0e-6)
    thresholds = _gate_thresholds(model)
    rejection_reasons: list[str] = []
    if comparison["r2"] is None or comparison["r2"] < thresholds["minimum_phase_r2"]:
        rejection_reasons.append("response_r2")
    if (
        comparison["normalized_rmse"] is None
        or comparison["normalized_rmse"] > thresholds["maximum_phase_normalized_rmse"]
    ):
        rejection_reasons.append("normalized_rmse")
    if gain_relative_error > thresholds["maximum_reference_gain_relative_error"]:
        rejection_reasons.append("gain_relative_error")
    if effort_saturated:
        rejection_reasons.append("effort_saturation")
    delay_error_s = simulation_delay - hardware_delay
    within_delay_gate = abs(delay_error_s) <= thresholds["maximum_reference_delay_error_s"]

    return {
        "stiffness": stiffness,
        "damping": damping,
        "valid": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "response_r2": comparison["r2"],
        "normalized_rmse": comparison["normalized_rmse"],
        "hardware_delay_s": hardware_delay,
        "simulation_delay_s": simulation_delay,
        "delay_error_s": delay_error_s,
        "within_delay_gate": within_delay_gate,
        "full_gate_passed": not rejection_reasons and within_delay_gate,
        "hardware_gain": hardware_response["gain"],
        "simulation_gain": simulation_response["gain"],
        "gain_relative_error": gain_relative_error,
        "max_abs_effort_nm": max_abs_effort,
        "effort_limit_nm": effort_limit_nm,
        "effort_saturated": effort_saturated,
        "max_abs_velocity_rad_s": max_abs_velocity,
    }


def _run_pd_sweep(
    sim: SimulationContext,
    robot: Articulation,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: dict[str, Any],
    joint_name: str,
    settle_s: float,
    stiffness_scales: list[float],
    damping_scales: list[float],
) -> dict[str, Any]:
    seed = model["joints"][joint_name]["controller_seed"]
    seed_stiffness = float(seed["stiffness"])
    seed_damping = float(seed["damping"])
    effort_limit_nm = float(seed["effort_limit_nm"])
    candidates = build_pd_candidates(seed_stiffness, seed_damping, stiffness_scales, damping_scales)
    reference_phase_id = _reference_phase_id(metadata)
    reference_phase = metadata["phase_metadata"][str(reference_phase_id)]
    segment = _phase_segment(arrays, reference_phase_id)
    results: list[dict[str, Any]] = []

    print(
        f"[INFO] PD sweep: joint={joint_name}, reference={reference_phase['profile_name']}, "
        f"candidates={len(candidates)}, seed=Kp {seed_stiffness:.3f}/Kd {seed_damping:.3f}"
    )
    for candidate_index, (stiffness, damping) in enumerate(candidates, start=1):
        _set_replay_pd(robot, joint_name, stiffness, damping)
        try:
            trace = _replay(sim, robot, segment, metadata, joint_name, settle_s)
            result = _pd_candidate_summary(
                trace,
                metadata,
                model,
                reference_phase_id,
                stiffness,
                damping,
                effort_limit_nm,
            )
            print(
                f"[INFO] PD {candidate_index:02d}/{len(candidates):02d}: Kp={stiffness:.3f}, Kd={damping:.3f}, "
                f"delay_error={1000.0 * result['delay_error_s']:+.2f} ms, "
                f"gain_error={100.0 * result['gain_relative_error']:.2f}%, "
                f"response_valid={result['valid']}, delay_gate={result['within_delay_gate']}"
            )
        except (SimReplayError, RuntimeError, ValueError) as error:
            result = {
                "stiffness": stiffness,
                "damping": damping,
                "valid": False,
                "rejection_reasons": ["simulation_error"],
                "error": str(error),
            }
            print(
                f"[WARN] PD {candidate_index:02d}/{len(candidates):02d}: "
                f"Kp={stiffness:.3f}, Kd={damping:.3f} failed: {error}"
            )
        results.append(result)

    selection = select_pd_candidate(
        results,
        seed_stiffness=seed_stiffness,
        seed_damping=seed_damping,
        maximum_delay_error_s=_gate_thresholds(model)["maximum_reference_delay_error_s"],
    )
    selected = selection["selected"]
    return {
        "reference_phase_id": reference_phase_id,
        "reference_profile": reference_phase["profile_name"],
        "selection_policy": (
            "First select candidates that pass response, gain, saturation, and delay gates. Within that set, "
            "minimize normalized distance from the existing PD seed, then effort and gain magnitude. Consider "
            "non-negative residual-delay compensation only when no candidate already passes the delay gate."
        ),
        "selection_mode": selection["selection_mode"],
        "positive_residual_compensation_available": selection["positive_residual_compensation_available"],
        "selected_within_delay_gate": selection["selected_within_delay_gate"],
        "within_delay_gate_candidate_count": selection["within_delay_gate_candidate_count"],
        "residual_compensable_candidate_count": selection["residual_compensable_candidate_count"],
        "seed_stiffness": seed_stiffness,
        "seed_damping": seed_damping,
        "stiffness_scales": stiffness_scales,
        "damping_scales": damping_scales,
        "candidate_count": len(results),
        "valid_candidate_count": selection["valid_candidate_count"],
        "selected_stiffness": float(selected["stiffness"]),
        "selected_damping": float(selected["damping"]),
        "selected_reference_result": selected,
        "candidates": results,
    }


def _report(
    trace: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: dict[str, Any],
    dataset_path: Path,
    model_path: Path,
    joint_name: str,
    stiffness: float,
    damping: float,
    applied_residual_delay_s: float,
    position_bias_rad: float,
    pd_sweep: dict[str, Any] | None,
    position_bias_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_hz = float(metadata["sample_hz"])
    phase_metadata = metadata["phase_metadata"]
    phases: list[dict[str, Any]] = []
    sine_phases: list[dict[str, Any]] = []
    valid = trace["phase_id"] >= 0
    for phase_id in sorted(int(value) for value in np.unique(trace["phase_id"][valid])):
        mask = trace["phase_id"] == phase_id
        phase_info = phase_metadata[str(phase_id)]
        comparison = _metrics(trace["hardware_position_rad"][mask], trace["simulated_position_rad"][mask])
        phase_result: dict[str, Any] = {
            "phase_id": phase_id,
            "profile_name": phase_info["profile_name"],
            "waveform": phase_info["waveform"],
            "frequency_hz": float(phase_info["frequency_hz"]),
            "sample_count": int(np.count_nonzero(mask)),
            "hardware_vs_simulation": comparison,
            "raw_goal_vs_hardware": _metrics(trace["hardware_position_rad"][mask], trace["goal_position_rad"][mask]),
        }
        if phase_info["waveform"] == "sine":
            frequency = float(phase_info["frequency_hz"])
            hardware_response = _frequency_response(
                trace["goal_position_rad"][mask],
                trace["hardware_position_rad"][mask],
                frequency_hz=frequency,
                sample_hz=sample_hz,
            )
            simulation_response = _frequency_response(
                trace["goal_position_rad"][mask],
                trace["simulated_position_rad"][mask],
                frequency_hz=frequency,
                sample_hz=sample_hz,
            )
            phase_result["hardware_frequency_response"] = hardware_response
            phase_result["simulation_frequency_response"] = simulation_response
            phase_result["delay_error_s"] = (
                simulation_response["equivalent_delay_s"] - hardware_response["equivalent_delay_s"]
            )
            phase_result["gain_relative_error"] = abs(simulation_response["gain"] - hardware_response["gain"]) / max(
                abs(hardware_response["gain"]), 1.0e-9
            )
            sine_phases.append(phase_result)
        phases.append(phase_result)

    if not sine_phases:
        raise SimReplayError("Dataset contains no sine phase for residual-delay comparison.")
    reference = max(sine_phases, key=lambda value: value["frequency_hz"])
    hardware_delay = float(reference["hardware_frequency_response"]["equivalent_delay_s"])
    simulation_delay = float(reference["simulation_frequency_response"]["equivalent_delay_s"])
    recommended_residual_delay = max(0.0, hardware_delay - simulation_delay)
    recommended_total_residual_delay = max(
        0.0,
        applied_residual_delay_s + hardware_delay - simulation_delay,
    )
    seed_usable = bool(model["joints"][joint_name]["quality"]["command_path_seed_usable"])
    gate_thresholds = _gate_thresholds(model)
    phase_gate = all(
        phase["hardware_vs_simulation"]["r2"] is not None
        and phase["hardware_vs_simulation"]["r2"] >= gate_thresholds["minimum_phase_r2"]
        and phase["hardware_vs_simulation"]["normalized_rmse"] is not None
        and phase["hardware_vs_simulation"]["normalized_rmse"] <= gate_thresholds["maximum_phase_normalized_rmse"]
        for phase in phases
    )
    reference_gate = (
        abs(float(reference["delay_error_s"])) <= gate_thresholds["maximum_reference_delay_error_s"]
        and float(reference["gain_relative_error"]) <= gate_thresholds["maximum_reference_gain_relative_error"]
    )
    gate_passed = bool(seed_usable and phase_gate and reference_gate)

    residual_compensable = simulation_delay <= hardware_delay
    if gate_passed:
        integration_action = (
            "Candidate passed this isolated replay; review the report before editing model quality gates."
        )
    elif residual_compensable:
        integration_action = (
            "The selected PD response can be slowed with the reported non-negative residual delay. Apply it only to "
            "a diagnostic model copy and replay before editing any quality gate."
        )
    else:
        integration_action = (
            "Do not enable this joint in training. The simulator remains slower than hardware, so positive residual "
            "delay cannot correct it; expand or refine the PD sweep and replay again."
        )

    report = {
        "schema_version": 1,
        "validation_type": "fixed_base_isaac_explicit_pd_replay",
        "dataset": str(dataset_path),
        "model": str(model_path),
        "joint": joint_name,
        "physics_hz": float(model["physics_hz"]),
        "sample_hz": sample_hz,
        "command_path_seed_usable": seed_usable,
        "sim_replay_gate_satisfied": gate_passed,
        "gate_thresholds": gate_thresholds,
        "reference_profile": reference["profile_name"],
        "reference_hardware_delay_s": hardware_delay,
        "reference_simulation_delay_s": simulation_delay,
        "reference_residual_compensable": residual_compensable,
        "applied_residual_delay_s": applied_residual_delay_s,
        "recommended_residual_delay_s": recommended_residual_delay,
        "recommended_total_residual_delay_s": recommended_total_residual_delay,
        "controller_settings": {
            "stiffness": stiffness,
            "damping": damping,
            "residual_position_bias_rad": position_bias_rad,
            "seed_stiffness": float(model["joints"][joint_name]["controller_seed"]["stiffness"]),
            "seed_damping": float(model["joints"][joint_name]["controller_seed"]["damping"]),
        },
        "phases": phases,
        "integration_action": integration_action,
        "automatic_model_update_performed": False,
    }
    if pd_sweep is not None:
        report["pd_sweep"] = pd_sweep
    if position_bias_calibration is not None:
        report["position_bias_calibration"] = position_bias_calibration
    return report


def main() -> int:
    try:
        if not args_cli.dataset:
            raise SimReplayError("--dataset is required.")
        if args_cli.sweep_pd and (args_cli.stiffness is not None or args_cli.damping is not None):
            raise SimReplayError("--sweep-pd cannot be combined with --stiffness or --damping.")
        dataset_path = Path(args_cli.dataset).expanduser().resolve()
        model_path = Path(args_cli.model).expanduser().resolve()
        output_prefix = (
            Path(args_cli.output_prefix).expanduser().resolve()
            if args_cli.output_prefix
            else dataset_path.with_name(
                f"{dataset_path.stem}_{'sim_replay_pd_sweep' if args_cli.sweep_pd else 'sim_replay'}"
            )
        )
        if args_cli.settle_s < 0.0:
            raise SimReplayError("--settle-s must be non-negative.")
        arrays, metadata = _load_dataset(dataset_path)
        joint_name = _select_joint(metadata, args_cli.joint)
        model = load_rnd_actuator_model(
            model_path,
            (joint_name,),
            require_command_path_seed=not args_cli.allow_unresolved,
        )
        if not model["joints"][joint_name]["quality"]["command_path_seed_usable"] and not args_cli.allow_unresolved:
            raise SimReplayError(f"{joint_name} has no accepted command-path seed.")
        seed = model["joints"][joint_name]["controller_seed"]
        stiffness = float(seed["stiffness"] if args_cli.stiffness is None else args_cli.stiffness)
        damping = float(seed["damping"] if args_cli.damping is None else args_cli.damping)
        configured_bias_range = model["joints"][joint_name]["command_path"].get(
            "residual_position_bias_rad_range", [0.0, 0.0]
        )
        configured_delay_range = model["joints"][joint_name]["command_path"]["residual_delay_s_range"]
        residual_delay_s = (
            0.5 * (float(configured_delay_range[0]) + float(configured_delay_range[1]))
            if args_cli.residual_delay_s is None
            else float(args_cli.residual_delay_s)
        )
        if not math.isfinite(residual_delay_s) or residual_delay_s < 0.0:
            raise SimReplayError("--residual-delay-s must be finite and non-negative.")
        position_bias_rad = (
            0.5 * (float(configured_bias_range[0]) + float(configured_bias_range[1]))
            if args_cli.position_bias_rad is None
            else float(args_cli.position_bias_rad)
        )
        if not math.isfinite(position_bias_rad):
            raise SimReplayError("--position-bias-rad must be finite.")

        physics_hz = float(model["physics_hz"])
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / physics_hz,
            render_interval=int(model["policy_decimation"]),
            device=args_cli.device,
        )
        sim = SimulationContext(sim_cfg)
        robot = _configure_robot(model, joint_name, model_path)
        sim.reset()
        robot.update(sim.get_physics_dt())
        _set_replay_delay(robot, joint_name, residual_delay_s)
        _set_replay_position_bias(robot, joint_name, position_bias_rad)
        pd_sweep: dict[str, Any] | None = None
        if args_cli.sweep_pd:
            stiffness_scales = parse_positive_scales(args_cli.stiffness_scales, "--stiffness-scales")
            damping_scales = parse_positive_scales(args_cli.damping_scales, "--damping-scales")
            pd_sweep = _run_pd_sweep(
                sim,
                robot,
                arrays,
                metadata,
                model,
                joint_name,
                args_cli.settle_s,
                stiffness_scales,
                damping_scales,
            )
            stiffness = float(pd_sweep["selected_stiffness"])
            damping = float(pd_sweep["selected_damping"])
            print(
                f"[INFO] Selected PD candidate: Kp={stiffness:.3f}, Kd={damping:.3f}, "
                f"mode={pd_sweep['selection_mode']}. Replaying the complete trace."
            )
        _set_replay_pd(robot, joint_name, stiffness, damping)
        trace = _replay(sim, robot, arrays, metadata, joint_name, args_cli.settle_s)
        initial_report = _report(
            trace,
            metadata,
            model,
            dataset_path,
            model_path,
            joint_name,
            stiffness,
            damping,
            residual_delay_s,
            position_bias_rad,
            pd_sweep,
            None,
        )
        position_bias_calibration: dict[str, Any] | None = None
        if args_cli.sweep_pd and args_cli.position_bias_rad is None and not initial_report["sim_replay_gate_satisfied"]:
            position_bias_calibration = _position_bias_calibration(
                trace,
                metadata,
                model,
                initial_report,
                position_bias_rad,
            )
            if position_bias_calibration["eligible"]:
                position_bias_rad = float(position_bias_calibration["selected_position_bias_rad"])
                print(
                    "[INFO] Calibrated residual position bias from "
                    f"{position_bias_calibration['source_profile']}: {position_bias_rad:+.6f} rad "
                    f"({math.degrees(position_bias_rad):+.3f} deg). Replaying the complete trace."
                )
                _set_replay_position_bias(robot, joint_name, position_bias_rad)
                trace = _replay(sim, robot, arrays, metadata, joint_name, args_cli.settle_s)
        elif args_cli.position_bias_rad is not None:
            position_bias_calibration = {
                "attempted": False,
                "eligible": True,
                "reason": "Explicit --position-bias-rad override used; no automatic calibration performed.",
                "selected_position_bias_rad": position_bias_rad,
            }

        report = _report(
            trace,
            metadata,
            model,
            dataset_path,
            model_path,
            joint_name,
            stiffness,
            damping,
            residual_delay_s,
            position_bias_rad,
            pd_sweep,
            position_bias_calibration,
        )

        npz_path = output_prefix.with_suffix(".npz")
        json_path = output_prefix.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **trace, metadata_json=np.asarray(json.dumps(report, sort_keys=True)))
        _atomic_write_json(json_path, report)
        print(f"Saved simulator trace: {npz_path}")
        print(f"Saved simulator replay report: {json_path}")
        print(
            f"Kp={stiffness:.3f}, Kd={damping:.3f}, position_bias={position_bias_rad:+.6f} rad, "
            f"reference={report['reference_profile']}, "
            f"hardware_delay={1000.0 * report['reference_hardware_delay_s']:.2f} ms, "
            f"simulation_delay={1000.0 * report['reference_simulation_delay_s']:.2f} ms, "
            f"applied_residual={1000.0 * report['applied_residual_delay_s']:.2f} ms, "
            f"recommended_total_residual={1000.0 * report['recommended_total_residual_delay_s']:.2f} ms"
        )
        print(f"sim_replay_gate_satisfied={report['sim_replay_gate_satisfied']}")
        return 0
    except (PDSweepError, SimReplayError, RndActuatorModelError, OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
