#!/usr/bin/env python3
"""Estimate RND joint friction from MX-106 current and fixed-base URDF dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


_TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_LUT = _TOOL_DIR / "config" / "mx106_performance_lut.json"

parser = argparse.ArgumentParser(
    description=(
        "Convert one complete RND real2sim trace from Present Current to approximate output torque, query "
        "zero-armature fixed-base STEP dynamics from PhysX, and fit an analysis-only friction residual."
    )
)
parser.add_argument("--dataset", required=True, help="Complete single-joint rnd_real2sim NPZ dataset.")
parser.add_argument("--joint", help="Excited joint; inferred when the dataset contains exactly one.")
parser.add_argument("--lut", default=str(DEFAULT_LUT), help="Digitized MX-106 current-to-torque LUT JSON.")
parser.add_argument("--output-prefix", help="Output prefix for _torque_friction.npz and .json.")
parser.add_argument("--batch-size", type=int, default=8, help="Number of fixed-base PhysX clones used per query.")
parser.add_argument("--filter-window", type=int, default=11, help="Odd Savitzky-Golay derivative window.")
parser.add_argument("--filter-order", type=int, default=3, help="Savitzky-Golay polynomial order.")
parser.add_argument(
    "--transition-velocity-deg-s",
    type=float,
    default=4.0,
    help="Smooth Coulomb transition velocity used by the residual fit.",
)
parser.add_argument(
    "--minimum-fit-speed-deg-s",
    type=float,
    default=1.0,
    help="Exclude lower-speed samples from the friction regression.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from robot_lab.assets.rnd import STEP_CFG
from rnd_real2sim.dataset import Real2SimDataset, load_dataset
from rnd_real2sim.torque_identification import (
    TorqueIdentificationError,
    current_to_output_torque,
    estimate_joint_kinematics,
    fit_friction_residual,
    fit_quasistatic_gravity_calibration,
    load_torque_lut,
)


class FrictionAnalysisError(ValueError):
    """Raised when the dataset or simulator dynamics are incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_joint(dataset: Real2SimDataset, requested: str | None) -> str:
    excited = dataset.metadata.get("excitation_joint_names")
    if not isinstance(excited, list) or not excited:
        raise FrictionAnalysisError("Dataset metadata contains no excitation_joint_names.")
    if requested is not None:
        if requested not in excited:
            raise FrictionAnalysisError(f"Requested joint {requested!r} was not excited; available={excited}.")
        return requested
    if len(excited) != 1:
        raise FrictionAnalysisError(f"Dataset excites {excited}; select one with --joint.")
    return str(excited[0])


def _zero_dynamics_actuator_extras(robot_cfg) -> None:
    for actuator in robot_cfg.actuators.values():
        actuator.stiffness = 0.0
        actuator.damping = 0.0
        actuator.armature = 0.0
        for field in ("friction", "dynamic_friction", "viscous_friction"):
            if hasattr(actuator, field):
                setattr(actuator, field, 0.0)


def _build_scene(batch_size: int) -> tuple[SimulationContext, InteractiveScene]:
    if batch_size <= 0:
        raise FrictionAnalysisError("batch-size must be positive.")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    robot_cfg = copy.deepcopy(STEP_CFG)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.fix_base = True
    robot_cfg.spawn.activate_contact_sensors = False
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
    _zero_dynamics_actuator_extras(robot_cfg)

    @configclass
    class DynamicsSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    scene = InteractiveScene(DynamicsSceneCfg(num_envs=batch_size, env_spacing=1.5))
    sim.reset()
    return sim, scene


def _phase_interior_mask(dataset: Real2SimDataset, margin: int) -> np.ndarray:
    phase_id = dataset.arrays["phase_id"]
    excitation_joint_id = dataset.arrays["excitation_joint_id"]
    mask = np.zeros(dataset.sample_count, dtype=np.bool_)
    for phase in np.unique(phase_id):
        if phase < 0:
            continue
        indices = np.flatnonzero(phase_id == phase)
        if indices.size <= 2 * margin:
            continue
        interior = indices[margin : indices.size - margin]
        mask[interior] = excitation_joint_id[interior] >= 0
    return mask


def _dynamic_friction_mask(dataset: Real2SimDataset, margin: int, selected_joint_name: str) -> np.ndarray:
    mask = _phase_interior_mask(dataset, margin)
    phase_metadata = dataset.metadata.get("phase_metadata", {})
    if not isinstance(phase_metadata, dict):
        raise FrictionAnalysisError("Dataset metadata contains no phase_metadata mapping.")
    supported_phase_ids = {
        int(phase_id)
        for phase_id, phase_info in phase_metadata.items()
        if isinstance(phase_info, dict)
        and phase_info.get("joint_name") == selected_joint_name
        and phase_info.get("waveform") == "sine"
    }
    if not supported_phase_ids:
        raise FrictionAnalysisError("Dynamic friction fit requires at least one sine excitation phase.")
    return mask & np.isin(dataset.arrays["phase_id"], list(supported_phase_ids))


def _query_dynamics(
    dataset: Real2SimDataset,
    velocity_rad_s: np.ndarray,
    acceleration_rad_s2: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    sim, scene = _build_scene(batch_size)
    robot = scene["robot"]
    robot_names = list(robot.joint_names)
    if set(robot_names) != set(dataset.joint_names):
        raise FrictionAnalysisError(
            f"Dataset/PhysX joint mismatch: dataset={dataset.joint_names}, PhysX={tuple(robot_names)}."
        )
    dataset_indices = [dataset.joint_names.index(name) for name in robot_names]
    position = dataset.arrays["position_rad"][:, dataset_indices]
    velocity = velocity_rad_s[:, dataset_indices]
    acceleration = acceleration_rad_s2[:, dataset_indices]

    inertial = np.empty_like(position)
    coriolis = np.empty_like(position)
    gravity = np.empty_like(position)
    mass_diagonal = np.empty_like(position)
    device = robot.device
    batch_count = math.ceil(dataset.sample_count / batch_size)
    progress_interval = max(1, batch_count // 10)
    for batch_index, start in enumerate(range(0, dataset.sample_count, batch_size)):
        stop = min(start + batch_size, dataset.sample_count)
        count = stop - start
        q = np.repeat(position[stop - 1 : stop], batch_size, axis=0)
        qd = np.repeat(velocity[stop - 1 : stop], batch_size, axis=0)
        qdd = np.repeat(acceleration[stop - 1 : stop], batch_size, axis=0)
        q[:count] = position[start:stop]
        qd[:count] = velocity[start:stop]
        qdd[:count] = acceleration[start:stop]
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=device)
        qd_tensor = torch.as_tensor(qd, dtype=torch.float32, device=device)
        qdd_tensor = torch.as_tensor(qdd, dtype=torch.float32, device=device)
        robot.write_joint_state_to_sim(q_tensor, qd_tensor)
        sim.forward()
        mass = robot.root_physx_view.get_generalized_mass_matrices()
        coriolis_batch = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        gravity_batch = robot.root_physx_view.get_gravity_compensation_forces()
        inertial_batch = torch.bmm(mass, qdd_tensor.unsqueeze(-1)).squeeze(-1)
        inertial[start:stop] = inertial_batch[:count].cpu().numpy()
        coriolis[start:stop] = coriolis_batch[:count].cpu().numpy()
        gravity[start:stop] = gravity_batch[:count].cpu().numpy()
        mass_diagonal[start:stop] = torch.diagonal(mass[:count], dim1=-2, dim2=-1).cpu().numpy()
        if batch_index % progress_interval == 0 or stop == dataset.sample_count:
            print(f"[INFO] URDF dynamics {stop:5d}/{dataset.sample_count}")
    return {
        "inertial_torque_nm": inertial,
        "coriolis_torque_nm": coriolis,
        "gravity_torque_nm": gravity,
        "mass_matrix_diagonal_kg_m2": mass_diagonal,
    }, robot_names


def _fit(
    residual: np.ndarray,
    velocity: np.ndarray,
    selected_joint_index: int,
    mask: np.ndarray,
) -> dict[str, Any]:
    return fit_friction_residual(
        residual[:, selected_joint_index],
        velocity[:, selected_joint_index],
        mask,
        transition_velocity_rad_s=math.radians(args_cli.transition_velocity_deg_s),
        minimum_speed_rad_s=math.radians(args_cli.minimum_fit_speed_deg_s),
    )


def _fit_by_phase(
    dataset: Real2SimDataset,
    residual: np.ndarray,
    velocity: np.ndarray,
    selected_joint_index: int,
    selected_joint_name: str,
    base_mask: np.ndarray,
    conversion,
) -> dict[str, Any]:
    phase_metadata = dataset.metadata.get("phase_metadata", {})
    result: dict[str, Any] = {}
    for phase in np.unique(dataset.arrays["phase_id"]):
        if phase < 0:
            continue
        phase_key = str(int(phase))
        phase_info = phase_metadata.get(phase_key, {}) if isinstance(phase_metadata, dict) else {}
        if phase_info.get("joint_name") != selected_joint_name:
            continue
        profile_name = str(phase_info.get("profile_name", f"phase_{phase_key}"))
        phase_mask = base_mask & (dataset.arrays["phase_id"] == phase)
        selected_below = conversion.below_observed_curve[phase_mask, selected_joint_index]
        selected_above = conversion.above_observed_curve[phase_mask, selected_joint_index]
        profile_result = {
            "phase_id": int(phase),
            "amplitude_rad": phase_info.get("amplitude_rad"),
            "frequency_hz": phase_info.get("frequency_hz"),
            "waveform": phase_info.get("waveform"),
            "observed_curve_fraction": float(1.0 - np.mean(selected_below) - np.mean(selected_above)),
            "low_current_extrapolation_fraction": float(np.mean(selected_below)),
            "high_current_clipping_fraction": float(np.mean(selected_above)),
        }
        if phase_info.get("waveform") != "sine":
            profile_result.update({
                "fit_supported": False,
                "reason": "Non-sine micro excitation is reserved for backlash/stiction diagnostics.",
            })
        else:
            profile_result.update({
                "fit_supported": True,
                **_fit(residual, velocity, selected_joint_index, phase_mask),
            })
        result[profile_name] = profile_result
    return result


def _quasistatic_calibration(
    dataset: Real2SimDataset,
    velocity: np.ndarray,
    gravity_torque: np.ndarray,
    joint_index: int,
    phase_interior_mask: np.ndarray,
) -> dict[str, Any]:
    joint_name = dataset.joint_names[joint_index]
    phase_metadata = dataset.metadata.get("phase_metadata", {})
    if not isinstance(phase_metadata, dict):
        return {"available": False, "reason": "Dataset metadata contains no phase_metadata mapping."}
    calibration_phases = []
    for phase_id, phase_info in phase_metadata.items():
        if not isinstance(phase_info, dict):
            continue
        if (
            phase_info.get("joint_name") == joint_name
            and phase_info.get("waveform") == "sine"
            and float(phase_info.get("frequency_hz", math.inf)) <= 0.03
            and float(phase_info.get("amplitude_rad", 0.0)) >= math.radians(15.0)
        ):
            calibration_phases.append(int(phase_id))
    if not calibration_phases:
        return {
            "available": False,
            "reason": "No sine phase combines frequency <= 0.03 Hz with amplitude >= 15 deg.",
        }
    mask = phase_interior_mask & np.isin(dataset.arrays["phase_id"], calibration_phases)
    result = fit_quasistatic_gravity_calibration(
        dataset.arrays["position_rad"][:, joint_index],
        dataset.arrays["current_a"][:, joint_index],
        velocity[:, joint_index],
        gravity_torque[:, joint_index],
        mask,
        minimum_speed_rad_s=math.radians(0.2),
    )
    return {"available": True, "phase_ids": calibration_phases, **result}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    dataset = load_dataset(args_cli.dataset)
    joint_name = _select_joint(dataset, args_cli.joint)
    lut_path = Path(args_cli.lut).expanduser().resolve()
    lut = load_torque_lut(lut_path)
    sample_hz = float(dataset.metadata["sample_hz"])
    velocity, acceleration = estimate_joint_kinematics(
        dataset.arrays["position_rad"],
        sample_hz,
        window_length=args_cli.filter_window,
        polynomial_order=args_cli.filter_order,
    )
    conversion = current_to_output_torque(dataset.arrays["current_a"], lut)
    dynamics_robot_order, robot_names = _query_dynamics(dataset, velocity, acceleration, args_cli.batch_size)
    robot_to_dataset = [robot_names.index(name) for name in dataset.joint_names]
    dynamics = {name: values[:, robot_to_dataset] for name, values in dynamics_robot_order.items()}
    modeled_torque = dynamics["inertial_torque_nm"] + dynamics["coriolis_torque_nm"] + dynamics["gravity_torque_nm"]
    residual = conversion.torque_nm - modeled_torque
    joint_index = dataset.joint_names.index(joint_name)
    phase_interior_mask = _phase_interior_mask(dataset, args_cli.filter_window // 2)
    mask = _dynamic_friction_mask(dataset, args_cli.filter_window // 2, joint_name)
    fit = _fit(residual, velocity, joint_index, mask)
    profile_fits = _fit_by_phase(
        dataset,
        residual,
        velocity,
        joint_index,
        joint_name,
        phase_interior_mask,
        conversion,
    )
    low_current_calibration = _quasistatic_calibration(
        dataset,
        velocity,
        dynamics["gravity_torque_nm"],
        joint_index,
        phase_interior_mask,
    )

    stall_ratio = float(lut["reference_specification"]["stall_torque_per_amp_nm"])
    stall_proxy_torque = dataset.arrays["current_a"] * stall_ratio
    stall_proxy_residual = stall_proxy_torque - modeled_torque
    stall_proxy_fit = _fit(stall_proxy_residual, velocity, joint_index, mask)

    selected = mask
    selected_below = conversion.below_observed_curve[selected, joint_index]
    selected_above = conversion.above_observed_curve[selected, joint_index]
    low_fraction = float(np.mean(selected_below))
    high_fraction = float(np.mean(selected_above))
    observed_fraction = max(0.0, 1.0 - low_fraction - high_fraction)
    fit_r2 = fit["r2"]
    quality_pass = bool(observed_fraction >= 0.8 and fit["optimizer_success"] and fit_r2 is not None and fit_r2 >= 0.5)
    quality_reasons: list[str] = []
    if observed_fraction < 0.8:
        quality_reasons.append(
            "Less than 80% of selected current samples are inside the manufacturer graph's observed current range."
        )
    if not fit["optimizer_success"]:
        quality_reasons.append("Robust residual optimizer did not converge.")
    if fit["r2"] is None or fit["r2"] < 0.5:
        quality_reasons.append("Friction residual fit R2 is below 0.5.")

    dataset_path = dataset.path
    prefix = (
        Path(args_cli.output_prefix).expanduser().resolve()
        if args_cli.output_prefix
        else dataset_path.with_suffix("").with_name(f"{dataset_path.stem}_torque_friction")
    )
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    report = {
        "schema_version": 1,
        "model_type": "rnd_real2sim_dynamic_friction_analysis",
        "analysis_only": True,
        "integration_enabled": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": dataset.sha256,
        "source_lut": str(lut_path),
        "source_lut_sha256": _sha256(lut_path),
        "joint": joint_name,
        "sample_hz": sample_hz,
        "dynamics": {
            "source": "PhysX fixed-base STEP URDF generalized mass matrix, Coriolis compensation, and gravity compensation",
            "armature_kg_m2": 0.0,
            "joint_friction_nm": 0.0,
            "root_fixed": True,
            "position_source": "measured encoder position",
            "velocity_and_acceleration_source": "Savitzky-Golay derivatives of measured encoder position",
            "filter_window": args_cli.filter_window,
            "filter_order": args_cli.filter_order,
            "included_waveforms": ["sine"],
            "excluded_waveforms": {
                "triangle": "Reserved for backlash/stiction diagnostics; not used in dynamic friction regression."
            },
            "warning": (
                "Reflected rotor inertia/armature has not been measured and is set to zero; its acceleration-dependent "
                "effect can be absorbed into the reported residual."
            ),
        },
        "torque_conversion": {
            "method": "digitized ROBOTIS MX-106 performance graph with odd symmetry",
            "observed_curve_fraction": observed_fraction,
            "low_current_extrapolation_fraction": low_fraction,
            "high_current_clipping_fraction": high_fraction,
            "warning": lut["conversion"]["warning"],
        },
        "friction_fit": fit,
        "profile_fits": profile_fits,
        "low_current_torque_calibration": low_current_calibration,
        "stall_spec_sensitivity_fit": {
            "torque_per_amp_nm": stall_ratio,
            "purpose": "sensitivity only; not selected as the performance-graph calibration",
            **stall_proxy_fit,
        },
        "quality": {
            "pass": quality_pass,
            "reasons": quality_reasons,
            "automatic_integration_allowed": False,
        },
    }
    arrays = {
        "time_s": dataset.arrays["time_s"],
        "phase_id": dataset.arrays["phase_id"],
        "fit_sample_mask": mask,
        "smoothed_velocity_rad_s": velocity,
        "smoothed_acceleration_rad_s2": acceleration,
        "estimated_output_torque_nm": conversion.torque_nm,
        "low_current_extrapolation_mask": conversion.below_observed_curve,
        "high_current_clipping_mask": conversion.above_observed_curve,
        **dynamics,
        "modeled_urdf_torque_nm": modeled_torque,
        "friction_residual_torque_nm": residual,
        "stall_spec_proxy_residual_torque_nm": stall_proxy_residual,
        "metadata_json": np.asarray(json.dumps(report, sort_keys=True), dtype=np.str_),
    }
    _atomic_npz(npz_path, arrays)
    _atomic_json(json_path, report)
    print(f"Saved torque/friction trace: {npz_path}")
    print(f"Saved torque/friction report: {json_path}")
    print(
        f"joint={joint_name}, coulomb={fit['coulomb_nm']:.4f} Nm, viscous={fit['viscous_nm_per_rad_s']:.4f} "
        f"Nm/(rad/s), R2={fit['r2']}, observed_graph_fraction={observed_fraction:.3f}, quality_pass={quality_pass}"
    )
    if low_current_calibration.get("available"):
        interval = low_current_calibration["bootstrap_90pct_nm_per_a"]
        print(
            "low_current_torque_calibration: "
            f"torque_per_amp={low_current_calibration['torque_per_amp_nm']:.4f} Nm/A, "
            f"coulomb_current={low_current_calibration['coulomb_current_a']:.4f} A, "
            f"bootstrap_90pct=[{interval[0]:.4f}, {interval[1]:.4f}] Nm/A, "
            f"quality_pass={low_current_calibration['quality']['pass']}"
        )
    if not quality_pass:
        print("[WARNING] Result remains analysis-only: " + " ".join(quality_reasons))


if __name__ == "__main__":
    try:
        main()
    except (FrictionAnalysisError, TorqueIdentificationError, ValueError) as error:
        print(f"[ERROR] {error}")
        raise SystemExit(1) from error
    finally:
        simulation_app.close()
