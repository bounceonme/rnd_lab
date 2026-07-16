#!/usr/bin/env python3
"""Build a gated RND STEP runtime actuator seed from the accepted baseline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import RND_ACTUATOR_MODEL_SCHEMA_VERSION, RND_ACTUATOR_MODEL_TYPE
from rnd_real2sim.config import RND_LEG_JOINT_NAMES


DEFAULT_BASELINE = _TOOL_DIR / "config" / "rnd_real2sim_baseline.json"
DEFAULT_ASSET_CFG = _ROBOT_PACKAGE_DIR / "assets" / "rnd.py"
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_actuator_model.json"


class ActuatorBuildError(ValueError):
    """Raised when baseline evidence cannot be converted without guessing."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stateful RND actuator command-path seed. The result remains integration-blocked until "
            "its residual delay and explicit-PD response pass simulator replay."
        )
    )
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Accepted analysis baseline JSON.")
    parser.add_argument("--asset-cfg", default=str(DEFAULT_ASSET_CFG), help="RND STEP articulation config source.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Runtime actuator seed JSON.")
    parser.add_argument("--physics-hz", type=float, default=200.0, help="Isaac physics/actuator update frequency.")
    parser.add_argument("--policy-hz", type=float, default=50.0, help="RL policy command frequency.")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ActuatorBuildError(f"Input does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ActuatorBuildError(f"Input is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ActuatorBuildError(f"Expected a JSON object in {path}.")
    return value


def _call_keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise ActuatorBuildError(f"STEP_CFG actuator call is missing literal {name!r}.")


def _literal_keyword(call: ast.Call, name: str) -> Any:
    try:
        return ast.literal_eval(_call_keyword(call, name))
    except (ValueError, TypeError) as error:
        raise ActuatorBuildError(f"STEP_CFG actuator {name!r} must remain a Python literal.") from error


def _load_controller_seeds(path: Path) -> dict[str, dict[str, Any]]:
    """Read STEP_CFG actuator values without importing Isaac or robot_lab."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ActuatorBuildError(f"Unable to parse STEP asset config {path}: {error}") from error

    step_cfg_call = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "STEP_CFG"
            and isinstance(node.value, ast.Call)
        ):
            step_cfg_call = node.value
            break
    if step_cfg_call is None:
        raise ActuatorBuildError("STEP_CFG assignment was not found in the asset config.")

    actuator_node = _call_keyword(step_cfg_call, "actuators")
    if not isinstance(actuator_node, ast.Dict):
        raise ActuatorBuildError("STEP_CFG.actuators must be a literal dictionary for provenance extraction.")

    seeds: dict[str, dict[str, Any]] = {}
    for group_key, group_value in zip(actuator_node.keys, actuator_node.values, strict=True):
        if not isinstance(group_value, ast.Call):
            raise ActuatorBuildError("Every STEP_CFG actuator group must be a constructor call.")
        group_name = str(ast.literal_eval(group_key))
        joint_names = list(_literal_keyword(group_value, "joint_names_expr"))
        group_values = {
            "source_group": group_name,
            "stiffness": float(_literal_keyword(group_value, "stiffness")),
            "damping": float(_literal_keyword(group_value, "damping")),
            "effort_limit_nm": float(_literal_keyword(group_value, "effort_limit_sim")),
            "velocity_limit_rad_s": float(_literal_keyword(group_value, "velocity_limit_sim")),
            "armature": float(_literal_keyword(group_value, "armature")),
        }
        for joint_name in joint_names:
            if joint_name in seeds:
                raise ActuatorBuildError(f"Joint {joint_name} appears in more than one STEP_CFG actuator group.")
            seeds[joint_name] = dict(group_values)

    expected = set(RND_LEG_JOINT_NAMES)
    if set(seeds) != expected:
        raise ActuatorBuildError(
            f"STEP_CFG actuator joints do not match RND legs: missing={sorted(expected - set(seeds))}, "
            f"extra={sorted(set(seeds) - expected)}"
        )
    return seeds


def _measured_summary(component: dict[str, Any], field: str) -> dict[str, Any] | None:
    value = component.get(field)
    if not isinstance(value, dict):
        return None
    return {
        "count": int(value["count"]),
        "median": float(value["median"]),
        "minimum": float(value["minimum"]),
        "maximum": float(value["maximum"]),
    }


def _play_seed(target: dict[str, Any]) -> tuple[list[float], list[float], float, list[float]]:
    backlash = _measured_summary(target, "effective_backlash_rad")
    if backlash is None or backlash["median"] <= 0.0:
        raise ActuatorBuildError("A usable target entry is missing positive effective_backlash_rad evidence.")
    median = float(backlash["median"])
    scale_low = max(0.0, float(backlash["minimum"]) / median)
    scale_high = max(scale_low, float(backlash["maximum"]) / median)
    return [0.5 * median], [1.0], 0.0, [scale_low, scale_high]


def _explicit_command_path_seed(component: dict[str, Any]) -> dict[str, Any]:
    command_path = component.get("command_path")
    if not isinstance(command_path, dict):
        raise ActuatorBuildError("A usable amplitude-dependent entry is missing command_path parameters.")
    return {
        "residual_delay_s_range": [float(value) for value in command_path["residual_delay_s_range"]],
        "residual_position_bias_rad_range": [
            float(value) for value in command_path["residual_position_bias_rad_range"]
        ],
        "play_thresholds_rad": [float(value) for value in command_path["play_thresholds_rad"]],
        "play_weights": [float(value) for value in command_path["play_weights"]],
        "linear_weight": float(command_path["linear_weight"]),
        "play_threshold_scale_range": [float(value) for value in command_path["play_threshold_scale_range"]],
    }


def build_actuator_model(
    baseline_path: str | Path,
    asset_cfg_path: str | Path,
    *,
    physics_hz: float = 200.0,
    policy_hz: float = 50.0,
) -> dict[str, Any]:
    """Build an integration-gated runtime seed from accepted evidence."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    asset_cfg_path = Path(asset_cfg_path).expanduser().resolve()
    if physics_hz <= 0.0 or policy_hz <= 0.0 or physics_hz < policy_hz:
        raise ActuatorBuildError("Require positive physics_hz >= policy_hz.")
    ratio = physics_hz / policy_hz
    if abs(ratio - round(ratio)) > 1.0e-9:
        raise ActuatorBuildError("physics_hz / policy_hz must be an integer decimation ratio.")

    baseline = _load_json(baseline_path)
    if baseline.get("schema_version") != 1 or baseline.get("analysis_only") is not True:
        raise ActuatorBuildError("Baseline must be schema_version=1 and analysis_only=true.")
    baseline_joints = baseline.get("joints")
    if not isinstance(baseline_joints, dict) or set(baseline_joints) != set(RND_LEG_JOINT_NAMES):
        raise ActuatorBuildError("Baseline joint set does not match the 12 RND leg joints.")
    controller_seeds = _load_controller_seeds(asset_cfg_path)

    joints: dict[str, Any] = {}
    unresolved: list[str] = []
    replay_validated: list[str] = []
    for joint_name in RND_LEG_JOINT_NAMES:
        evidence = baseline_joints[joint_name]
        target = evidence["target"]
        amplitude_dependent = evidence.get("command_path_model", {"usable": False})
        coulomb = evidence["coulomb_current"]
        target_usable = target.get("usable") is True
        amplitude_dependent_usable = amplitude_dependent.get("usable") is True
        if target_usable and amplitude_dependent_usable:
            raise ActuatorBuildError(
                f"Joint {joint_name} has both constant-play and amplitude-dependent command-path seeds."
            )
        if amplitude_dependent_usable:
            command_path = _explicit_command_path_seed(amplitude_dependent)
            measured_source = amplitude_dependent.get("measured", {})
            seed_source = "multi_amplitude_generalized_play"
        elif target_usable:
            thresholds, weights, linear_weight, scale_range = _play_seed(target)
            command_path = {
                "residual_delay_s_range": [0.0, 0.0],
                "residual_position_bias_rad_range": [0.0, 0.0],
                "play_thresholds_rad": thresholds,
                "play_weights": weights,
                "linear_weight": linear_weight,
                "play_threshold_scale_range": scale_range,
            }
            measured_source = target
            seed_source = "constant_play_aggregate"
        else:
            command_path = {
                "residual_delay_s_range": [0.0, 0.0],
                "residual_position_bias_rad_range": [0.0, 0.0],
                "play_thresholds_rad": [],
                "play_weights": [],
                "linear_weight": 1.0,
                "play_threshold_scale_range": [1.0, 1.0],
            }
            measured_source = target
            seed_source = "unresolved_identity_placeholder"
            unresolved.append(joint_name)

        controller_seed = dict(controller_seeds[joint_name])
        sim_replay = amplitude_dependent.get("sim_replay") if amplitude_dependent_usable else None
        if sim_replay is not None:
            selected_controller = sim_replay.get("selected_controller")
            if not isinstance(selected_controller, dict):
                raise ActuatorBuildError(f"Joint {joint_name} has malformed multi-amplitude replay controller data.")
            controller_seed["pre_replay_stiffness"] = float(controller_seed["stiffness"])
            controller_seed["pre_replay_damping"] = float(controller_seed["damping"])
            controller_seed["stiffness"] = float(selected_controller["stiffness"])
            controller_seed["damping"] = float(selected_controller["damping"])

        measured_coulomb = _measured_summary(coulomb, "coulomb_current_a")
        seed_usable = target_usable or amplitude_dependent_usable
        joint_replay_validated = sim_replay is not None
        if joint_replay_validated:
            replay_validated.append(joint_name)
        joints[joint_name] = {
            "controller_seed": controller_seed,
            "measured": {
                "closed_loop_delay_s": _measured_summary(measured_source, "command_delay_s"),
                "effective_backlash_rad": _measured_summary(target, "effective_backlash_rad"),
                "effective_hysteresis_rad": _measured_summary(measured_source, "effective_hysteresis_rad"),
                "command_minus_position_center_bias_rad": _measured_summary(
                    target, "command_minus_position_center_bias_rad"
                ),
                "reference_sine_gain": _measured_summary(measured_source, "reference_sine_gain"),
                "coulomb_current_a": measured_coulomb,
                "coulomb_current_domain": coulomb.get(
                    "domain", "joint-coordinate motor current; not measured joint torque"
                ),
                "command_path_seed_source": seed_source,
                "multi_amplitude_model": (
                    {
                        "model": amplitude_dependent["model"],
                        "model_sha256": amplitude_dependent["model_sha256"],
                        "amplitudes_rad": amplitude_dependent["amplitudes_rad"],
                        "validation": amplitude_dependent["validation"],
                        "sim_replay": sim_replay,
                    }
                    if amplitude_dependent_usable
                    else None
                ),
                "notes": list(evidence.get("notes", [])),
            },
            "command_path": command_path,
            "friction": {
                "enabled": False,
                "coulomb_torque_nm": None,
                "reason": (
                    "Motor current is retained as evidence only; no measured current-to-joint-torque calibration exists."
                ),
            },
            "quality": {
                "command_path_seed_usable": seed_usable,
                "sim_replay_validated": joint_replay_validated,
                "integration_allowed": False,
                "status": (
                    "sim_replay_validated_not_enabled"
                    if joint_replay_validated
                    else "requires_sim_replay_validation"
                    if seed_usable
                    else "unresolved_command_path"
                ),
            },
        }

    return {
        "schema_version": RND_ACTUATOR_MODEL_SCHEMA_VERSION,
        "model_type": RND_ACTUATOR_MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "application_status": "requires_sim_replay_validation",
        "integration_enabled": False,
        "physics_hz": float(physics_hz),
        "policy_hz": float(policy_hz),
        "policy_decimation": int(round(ratio)),
        "joint_order": list(RND_LEG_JOINT_NAMES),
        "source_baseline": _relative(baseline_path),
        "source_baseline_sha256": _sha256(baseline_path),
        "controller_seed_source": _relative(asset_cfg_path),
        "controller_seed_source_sha256": _sha256(asset_cfg_path),
        "measurement_domain": baseline.get("measurement_domain"),
        "delay_semantics": {
            "measured": "phase-equivalent closed-loop command-to-encoder response delay",
            "runtime": "residual delay to add only after measuring the explicit-PD simulator response",
            "initial_residual_delay_s_range": [0.0, 0.0],
        },
        "position_bias_semantics": {
            "measured": ("command minus encoder position at the suspended reference pose after delay compensation"),
            "runtime": (
                "additive target-position bias remaining after the explicit-PD simulator response; "
                "initialize at zero and identify only through simulator replay"
            ),
            "initial_residual_position_bias_rad_range": [0.0, 0.0],
        },
        "gain_semantics": (
            "Measured reference-sine gain is diagnostic and is not multiplied into policy position commands."
        ),
        "torque_calibration": {
            "available": False,
            "current_to_joint_torque_nm_per_a": None,
            "reason": "No external joint-torque calibration or measured transmission efficiency is available.",
        },
        "quality_summary": {
            "joint_count": len(joints),
            "command_path_seed_usable_joint_count": len(joints) - len(unresolved),
            "sim_replay_validated_joint_count": len(replay_validated),
            "sim_replay_validated_joints": replay_validated,
            "integration_ready": False,
            "unresolved_joints": unresolved,
        },
        "limitations": list(baseline.get("limitations", []))
        + [
            "Controller gains remain existing simulator seeds unless their command-path artifact includes passing "
            "fixed-base replay calibration.",
            "No measured friction torque is applied by this model.",
            "Residual position bias is local to the suspended reference pose and is not a global encoder-zero fit.",
            "Cross-amplitude trace validation does not replace simulator replay; finalized artifacts must include "
            "passing replay provenance for every source dataset.",
            "Every joint must pass fixed-base simulator replay before this file may be enabled in training.",
        ],
        "joints": joints,
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


def main() -> int:
    args = _parser().parse_args()
    try:
        model = build_actuator_model(
            args.baseline, args.asset_cfg, physics_hz=args.physics_hz, policy_hz=args.policy_hz
        )
        output = Path(args.output).expanduser().resolve()
        _atomic_write_json(output, model)
        summary = model["quality_summary"]
        print(f"Saved integration-gated actuator seed: {output}")
        print(
            f"joints={summary['joint_count']}, "
            f"command_path_seed_usable={summary['command_path_seed_usable_joint_count']}, "
            f"sim_replay_validated={summary['sim_replay_validated_joint_count']}"
        )
        if summary["unresolved_joints"]:
            print(f"Unresolved command paths: {', '.join(summary['unresolved_joints'])}")
        print("Training integration remains disabled.")
        return 0
    except (ActuatorBuildError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
