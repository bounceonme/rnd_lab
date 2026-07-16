#!/usr/bin/env python3
"""Aggregate explicitly selected RND actuator fits without applying them to simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
sys.path.insert(0, str(_TOOL_DIR))

from rnd_real2sim.config import RND_LEG_JOINT_NAMES


DEFAULT_MANIFEST = _TOOL_DIR / "config" / "rnd_real2sim_baseline_manifest.toml"
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_real2sim_baseline.json"


class AggregationError(ValueError):
    """Raised when a selected model is missing or violates the manifest contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate validated per-joint actuator fits into an analysis-only baseline JSON."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Explicit model-selection TOML.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output analysis-only baseline JSON.")
    return parser


def _resolve_model_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _load_model(path_value: str, expected_joint: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = _resolve_model_path(path_value)
    try:
        model = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise AggregationError(f"Selected model does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise AggregationError(f"Selected model is not valid JSON: {path}: {error}") from error

    if model.get("schema_version") != 2:
        raise AggregationError(f"Unsupported model schema in {path}: {model.get('schema_version')!r}")
    if model.get("source_dataset_dry_run"):
        raise AggregationError(f"Synthetic model cannot enter a hardware baseline: {path}")
    joints = model.get("joints", {})
    if set(joints) != {expected_joint}:
        raise AggregationError(f"Expected only {expected_joint} in {path}, found {sorted(joints)}")
    return path, model, joints[expected_joint]


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _reference_response(joint_model: dict[str, Any]) -> dict[str, Any]:
    reference_name = joint_model["command_delay"]["reference_profile"]
    for response in joint_model["frequency_response"]:
        if response["profile_name"] == reference_name:
            return response
    raise AggregationError(f"Reference response {reference_name!r} is missing from a selected model.")


def _command_minus_position_center_bias(joint_model: dict[str, Any]) -> float:
    cycles = joint_model.get("effective_backlash", {}).get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise AggregationError("Selected target model is missing branch-paired center-bias cycles.")
    values = [float(cycle["center_bias_rad"]) for cycle in cycles]
    if not all(math.isfinite(value) for value in values):
        raise AggregationError("Selected target model contains a non-finite center bias.")
    return float(statistics.median(values))


def _provenance(path: Path, model: dict[str, Any], joint_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": _relative_path(path),
        "model_sha256": _sha256(path),
        "source_dataset": Path(model["source_dataset"]).name,
        "source_dataset_sha256": model["source_dataset_sha256"],
        "status": joint_model["quality"]["status"],
    }


def _aggregate_target(paths: list[str], joint_name: str) -> dict[str, Any]:
    if not paths:
        return {"usable": False, "run_count": 0}

    delays: list[float] = []
    backlashes: list[float] = []
    gains: list[float] = []
    fit_r2: list[float] = []
    center_biases: list[float] = []
    provenance: list[dict[str, Any]] = []
    for path_value in paths:
        path, model, joint_model = _load_model(path_value, joint_name)
        if not joint_model["quality"].get("target_randomization_usable", False):
            raise AggregationError(f"Target quality gate failed for selected model: {path}")
        response = _reference_response(joint_model)
        delays.append(float(joint_model["command_delay"]["seconds"]))
        backlashes.append(float(joint_model["effective_backlash"]["median_rad"]))
        gains.append(float(response["gain"]["median"]))
        fit_r2.append(float(response["full_output_fit"]["r2"]))
        center_biases.append(_command_minus_position_center_bias(joint_model))
        provenance.append(_provenance(path, model, joint_model))

    return {
        "usable": True,
        "run_count": len(paths),
        "command_delay_s": _summary(delays),
        "effective_backlash_rad": _summary(backlashes),
        "command_minus_position_center_bias_rad": _summary(center_biases),
        "reference_sine_gain": _summary(gains),
        "reference_sine_fit_r2": _summary(fit_r2),
        "provenance": provenance,
    }


def _aggregate_coulomb(paths: list[str], joint_name: str) -> dict[str, Any]:
    if not paths:
        return {
            "usable": False,
            "run_count": 0,
            "domain": "joint-coordinate motor current; not measured joint torque",
        }

    values: list[float] = []
    provenance: list[dict[str, Any]] = []
    for path_value in paths:
        path, model, joint_model = _load_model(path_value, joint_name)
        if not joint_model["quality"].get("coulomb_randomization_usable", False):
            raise AggregationError(f"Coulomb quality gate failed for selected model: {path}")
        values.append(float(joint_model["friction_current_model"]["coulomb_current_a"]))
        provenance.append(_provenance(path, model, joint_model))

    return {
        "usable": True,
        "run_count": len(paths),
        "coulomb_current_a": _summary(values),
        "domain": "joint-coordinate motor current; not measured joint torque",
        "provenance": provenance,
    }


def _validate_command_path_sim_replay(
    model: dict[str, Any],
    path: Path,
    quality: dict[str, Any],
    command_path: dict[str, Any],
    sources: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    sim_replay = model.get("sim_replay")
    replay_validated = quality.get("sim_replay_validated") is True
    residual_delay = command_path["residual_delay_s_range"]
    residual_bias = command_path["residual_position_bias_rad_range"]
    if not replay_validated:
        if residual_delay != [0.0, 0.0] or residual_bias != [0.0, 0.0] or sim_replay is not None:
            raise AggregationError(f"Unvalidated command-path model must keep residual calibration at zero: {path}")
        return None, False

    if model.get("application_status") != "sim_replay_validated_not_integrated":
        raise AggregationError(f"Validated command-path model has an unexpected application status: {path}")
    if quality.get("integration_allowed") is not False or not isinstance(sim_replay, dict):
        raise AggregationError(f"Validated command-path model must remain integration-disabled: {path}")
    controller = sim_replay.get("selected_controller")
    replay_reports = sim_replay.get("reports")
    if (
        not isinstance(controller, dict)
        or not _is_finite_number(controller.get("stiffness"))
        or float(controller["stiffness"]) <= 0.0
        or not _is_finite_number(controller.get("damping"))
        or float(controller["damping"]) < 0.0
        or sim_replay.get("residual_delay_s_range") != residual_delay
        or sim_replay.get("residual_position_bias_rad_range") != residual_bias
        or not isinstance(replay_reports, list)
        or len(replay_reports) != len(sources)
    ):
        raise AggregationError(f"Validated command-path replay calibration is malformed: {path}")
    source_paths = {_relative_path(_resolve_model_path(source["dataset"])) for source in sources}
    replay_source_paths = set()
    for report in replay_reports:
        if not isinstance(report, dict) or not isinstance(report.get("report"), str):
            raise AggregationError(f"Validated command-path replay provenance is malformed: {path}")
        report_path = _resolve_model_path(report["report"])
        if _sha256(report_path) != report.get("report_sha256"):
            raise AggregationError(f"Command-path simulator replay report changed: {report_path}")
        replay_source_paths.add(str(report.get("dataset")))
    if replay_source_paths != source_paths:
        raise AggregationError(f"Command-path replay reports do not cover every source dataset: {path}")
    return sim_replay, True


def _aggregate_command_path(path_value: str | None, joint_name: str) -> dict[str, Any]:
    """Validate an amplitude-dependent command-path artifact without treating it as a constant-play fit."""

    if path_value is None:
        return {"usable": False}
    path = _resolve_model_path(path_value)
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AggregationError(f"Selected command-path model does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise AggregationError(f"Selected command-path model is not valid JSON: {path}: {error}") from error

    if model.get("schema_version") != 1 or model.get("model_type") != "rnd_multi_amplitude_generalized_play":
        raise AggregationError(f"Unsupported command-path model schema/type: {path}")
    if model.get("analysis_only") is not True:
        raise AggregationError(f"Command-path model must remain analysis_only: {path}")
    if model.get("joint") != joint_name:
        raise AggregationError(
            f"Command-path model joint mismatch: expected {joint_name}, found {model.get('joint')!r}: {path}"
        )
    quality = model.get("quality")
    fit = model.get("fit")
    if not isinstance(quality, dict) or quality.get("cross_amplitude_usable") is not True:
        raise AggregationError(f"Cross-amplitude quality gate failed: {path}")
    if not isinstance(fit, dict) or fit.get("all_validation_gates_pass") is not True:
        raise AggregationError(f"Not every selected amplitude passed validation: {path}")
    dataset_reports = fit.get("datasets")
    if not isinstance(dataset_reports, list) or len(dataset_reports) < 2:
        raise AggregationError(f"Command-path model requires at least two dataset reports: {path}")
    if any(not isinstance(report, dict) or report.get("validation_pass") is not True for report in dataset_reports):
        raise AggregationError(f"Command-path model contains a failed dataset report: {path}")
    amplitudes = model.get("amplitudes_rad")
    if not isinstance(amplitudes, list) or len({round(float(value), 12) for value in amplitudes}) < 2:
        raise AggregationError(f"Command-path model requires at least two distinct amplitudes: {path}")
    amplitudes_deg = model.get("amplitudes_deg")
    if (
        not isinstance(amplitudes_deg, list)
        or len(amplitudes_deg) != len(amplitudes)
        or any(not math.isfinite(float(value)) for value in amplitudes_deg)
    ):
        raise AggregationError(f"Command-path model amplitudes_deg are invalid: {path}")

    command_path = model.get("command_path")
    if not isinstance(command_path, dict):
        raise AggregationError(f"Command-path parameters are missing: {path}")
    thresholds = command_path.get("play_thresholds_rad")
    weights = command_path.get("play_weights")
    linear_weight = command_path.get("linear_weight")
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or not isinstance(weights, list)
        or len(weights) != len(thresholds)
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in thresholds)
        or [float(value) for value in thresholds] != sorted(float(value) for value in thresholds)
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights)
        or not isinstance(linear_weight, (int, float))
        or not math.isfinite(float(linear_weight))
        or float(linear_weight) < 0.0
        or not math.isclose(float(linear_weight) + sum(float(value) for value in weights), 1.0, abs_tol=1.0e-6)
    ):
        raise AggregationError(f"Command-path weights or thresholds are invalid: {path}")
    residual_delay = command_path.get("residual_delay_s_range")
    residual_bias = command_path.get("residual_position_bias_rad_range")
    if (
        not isinstance(residual_delay, list)
        or len(residual_delay) != 2
        or any(not _is_finite_number(value) or float(value) < 0.0 for value in residual_delay)
        or float(residual_delay[0]) > float(residual_delay[1])
        or not isinstance(residual_bias, list)
        or len(residual_bias) != 2
        or any(not _is_finite_number(value) for value in residual_bias)
        or float(residual_bias[0]) > float(residual_bias[1])
    ):
        raise AggregationError(f"Command-path residual ranges are invalid: {path}")
    if command_path.get("play_threshold_scale_range") != [1.0, 1.0]:
        raise AggregationError(f"Cross-amplitude threshold randomization is not yet supported: {path}")

    sources = model.get("source_datasets")
    if not isinstance(sources, list) or len(sources) != len(dataset_reports):
        raise AggregationError(f"Command-path source provenance is incomplete: {path}")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("dataset"), str)
            or not isinstance(source.get("identification_model"), str)
        ):
            raise AggregationError(f"Command-path source provenance is malformed: {path}")
        dataset_path = _resolve_model_path(source["dataset"])
        identification_path = _resolve_model_path(source["identification_model"])
        if _sha256(dataset_path) != source.get("dataset_sha256"):
            raise AggregationError(f"Command-path source dataset changed after fitting: {dataset_path}")
        if _sha256(identification_path) != source.get("identification_model_sha256"):
            raise AggregationError(f"Command-path identification model changed after fitting: {identification_path}")

    sim_replay, replay_validated = _validate_command_path_sim_replay(
        model,
        path,
        quality,
        command_path,
        sources,
    )

    return {
        "usable": True,
        "model_type": model["model_type"],
        "model": _relative_path(path),
        "model_sha256": _sha256(path),
        "amplitudes_rad": [float(value) for value in amplitudes],
        "amplitudes_deg": [float(value) for value in amplitudes_deg],
        "command_path": command_path,
        "measured": model.get("measured"),
        "validation": {
            "minimum_validation_r2_required": quality["minimum_validation_r2_required"],
            "maximum_normalized_rmse_allowed": quality["maximum_normalized_rmse_allowed"],
            "minimum_validation_r2": fit["minimum_validation_r2"],
            "maximum_validation_normalized_rmse": fit["maximum_validation_normalized_rmse"],
            "dataset_count": len(dataset_reports),
            "sim_replay_validated": replay_validated,
        },
        "source_datasets": sources,
        "sim_replay": sim_replay,
    }


def _diagnostic_record(entry: dict[str, Any]) -> dict[str, Any]:
    joint_name = str(entry["joint"])
    path, model, joint_model = _load_model(str(entry["model"]), joint_name)
    response = _reference_response(joint_model)
    backlash = joint_model.get("effective_backlash", {})
    play_model = backlash.get("play_model", {})
    friction = joint_model.get("friction_current_model")
    return {
        "joint": joint_name,
        "purpose": str(entry["purpose"]),
        "excluded_from_simple_baseline": True,
        "micro_triangle_amplitude_deg": float(entry["micro_triangle_amplitude_deg"]),
        "command_delay_s": joint_model["command_delay"].get("seconds"),
        "effective_backlash_rad": backlash.get("median_rad"),
        "play_model_gain": play_model.get("gain"),
        "play_model_validation_r2": play_model.get("validation", {}).get("r2"),
        "reference_sine_gain": response["gain"]["median"],
        "coulomb_current_a": friction.get("coulomb_current_a") if friction else None,
        "provenance": _provenance(path, model, joint_model),
    }


def aggregate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    try:
        manifest = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AggregationError(f"Unable to load manifest {path}: {error}") from error

    if manifest.get("schema_version") != 1:
        raise AggregationError("Manifest schema_version must be 1.")
    if manifest.get("analysis_only") is not True:
        raise AggregationError("Manifest must set analysis_only=true.")
    joint_entries = manifest.get("joints")
    if not isinstance(joint_entries, list):
        raise AggregationError("Manifest requires [[joints]] entries.")
    names = [entry.get("name") for entry in joint_entries]
    if len(names) != len(set(names)):
        raise AggregationError("Manifest joint names must be unique.")
    if set(names) != set(RND_LEG_JOINT_NAMES):
        missing = sorted(set(RND_LEG_JOINT_NAMES) - set(names))
        extra = sorted(set(names) - set(RND_LEG_JOINT_NAMES))
        raise AggregationError(f"Manifest joint mismatch: missing={missing}, extra={extra}")

    joints: dict[str, Any] = {}
    for entry in joint_entries:
        name = str(entry["name"])
        target = _aggregate_target(list(entry.get("target_models", [])), name)
        coulomb = _aggregate_coulomb(list(entry.get("coulomb_models", [])), name)
        command_path = _aggregate_command_path(entry.get("command_path_model"), name)
        if target["usable"] and command_path["usable"]:
            raise AggregationError(
                f"Joint {name} selects both a constant-play target and an amplitude-dependent command path."
            )
        joints[name] = {
            "target": target,
            "command_path_model": command_path,
            "coulomb_current": coulomb,
            "notes": list(entry.get("notes", [])),
        }

    target_count = sum(joint["target"]["usable"] for joint in joints.values())
    command_path_count = sum(
        joint["target"]["usable"] or joint["command_path_model"]["usable"] for joint in joints.values()
    )
    coulomb_count = sum(joint["coulomb_current"]["usable"] for joint in joints.values())
    fully_usable_count = sum(
        (joint["target"]["usable"] or joint["command_path_model"]["usable"]) and joint["coulomb_current"]["usable"]
        for joint in joints.values()
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "application_status": "not_integrated_with_rl_or_simulation",
        "purpose": manifest["purpose"],
        "manifest": _relative_path(path),
        "measurement_domain": "rigid-upper-body suspended robot at 50 Hz using encoder/current telemetry",
        "limitations": [
            "Delay is phase-equivalent closed-loop response delay, not pure transport latency.",
            "Effective backlash includes compliance and static-friction hysteresis.",
            "Coulomb current is not joint torque and must not be converted without calibration.",
            "Ground-contact load dependence was not measured.",
        ],
        "quality_summary": {
            "joint_count": len(joints),
            "target_usable_joint_count": target_count,
            "command_path_usable_joint_count": command_path_count,
            "coulomb_usable_joint_count": coulomb_count,
            "fully_usable_joint_count": fully_usable_count,
        },
        "joints": joints,
        "diagnostics": [_diagnostic_record(entry) for entry in manifest.get("diagnostics", [])],
        "exclusions": list(manifest.get("exclusions", [])),
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        baseline = aggregate_manifest(args.manifest)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        summary = baseline["quality_summary"]
        print(f"Saved analysis-only baseline: {output.resolve()}")
        print(
            "joints={joint_count}, target_usable={target_usable_joint_count}, "
            "command_path_usable={command_path_usable_joint_count}, coulomb_usable={coulomb_usable_joint_count}, "
            "fully_usable={fully_usable_joint_count}".format(**summary)
        )
        return 0
    except (AggregationError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
