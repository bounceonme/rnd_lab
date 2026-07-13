#!/usr/bin/env python3
"""Aggregate explicitly selected RND actuator fits without applying them to simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        provenance.append(_provenance(path, model, joint_model))

    return {
        "usable": True,
        "run_count": len(paths),
        "command_delay_s": _summary(delays),
        "effective_backlash_rad": _summary(backlashes),
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
        joints[name] = {
            "target": target,
            "coulomb_current": coulomb,
            "notes": list(entry.get("notes", [])),
        }

    target_count = sum(joint["target"]["usable"] for joint in joints.values())
    coulomb_count = sum(joint["coulomb_current"]["usable"] for joint in joints.values())
    fully_usable_count = sum(
        joint["target"]["usable"] and joint["coulomb_current"]["usable"] for joint in joints.values()
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
            "coulomb_usable={coulomb_usable_joint_count}, fully_usable={fully_usable_joint_count}".format(**summary)
        )
        return 0
    except (AggregationError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
