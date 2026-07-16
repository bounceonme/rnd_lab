#!/usr/bin/env python3
"""Identify all RND residual joint armatures from one dynamic collection."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rnd_real2sim.armature_batch import (
    ArmatureBatchError,
    analyze_armature_joints,
    load_armature_dynamics_trace,
    load_torque_calibration_report,
)
from rnd_real2sim.dataset import DatasetError, load_dataset


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
DEFAULT_TORQUE_CALIBRATION = (
    _REPO_ROOT
    / "logs"
    / "rnd_real2sim"
    / "all_joints_torque_calibration_01_all_joint_torque_calibration.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit residual armature for every dynamically excited RND joint using a zero-armature PhysX trace and a "
            "separately quality-gated current-to-torque calibration report."
        )
    )
    parser.add_argument("--dataset", required=True, help="Complete multi-joint armature experiment NPZ dataset.")
    parser.add_argument(
        "--dynamics-trace",
        help="Cached *_torque_friction.npz. Defaults to the conventional path next to the dataset.",
    )
    parser.add_argument(
        "--torque-calibration",
        default=str(DEFAULT_TORQUE_CALIBRATION),
        help="Quality-gated all-joint low-current torque calibration JSON.",
    )
    parser.add_argument("--joint", action="append", help="Optional joint subset; omitted means every excited joint.")
    parser.add_argument("--output", help="Aggregate armature-analysis JSON output path.")
    return parser


def _atomic_json(path: Path, value: dict) -> None:
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


def main() -> int:
    args = _parser().parse_args()
    try:
        dataset = load_dataset(args.dataset)
        default_trace = dataset.path.with_suffix("").with_name(f"{dataset.path.stem}_torque_friction.npz")
        trace = load_armature_dynamics_trace(args.dynamics_trace or default_trace, dataset)
        calibration = load_torque_calibration_report(args.torque_calibration)
        excited = dataset.metadata.get("excitation_joint_names")
        if not isinstance(excited, list) or not excited:
            raise ArmatureBatchError("Dataset metadata contains no excitation_joint_names.")
        selected = tuple(args.joint or excited)
        results, summary = analyze_armature_joints(dataset, trace, calibration, selected)
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else dataset.path.with_suffix("").with_name(f"{dataset.path.stem}_all_joint_armature.json")
        )
        report = {
            "schema_version": 2,
            "model_type": "rnd_real2sim_all_joint_armature_analysis",
            "analysis_only": True,
            "integration_enabled": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset.path),
            "source_dataset_sha256": dataset.sha256,
            "source_dynamics_trace": str(trace.path),
            "source_dynamics_trace_sha256": trace.sha256,
            "source_torque_calibration": str(calibration.path),
            "source_torque_calibration_sha256": calibration.sha256,
            "definition": (
                "Residual reflected joint inertia after zero-armature URDF inertial, Coriolis, and gravity torque "
                "are removed; this is not total link or joint inertia. Selection uses cycle-level fundamental "
                "harmonics projected away from velocity, with a shared position-error term across frequencies."
            ),
            "joints": results,
            "summary": summary,
        }
        _atomic_json(output, report)
        print(f"Saved all-joint armature report: {output}")
        for joint_name, result in results.items():
            if result["quality"]["pass"]:
                fit = result["fit"]
                print(
                    f"[PASS] {joint_name:27s} Jarm={fit['armature_kg_m2']:.6f} kg*m^2 "
                    f"harmonic_R2={fit['r2']:.3f} "
                    f"improvement={100.0 * fit['rmse_improvement_over_position_only']:.1f}%"
                )
            else:
                print(f"[FAIL] {joint_name:27s} {' '.join(result['quality']['reasons'])}")
        print(
            f"armature_passed={len(summary['armature_passed_joints'])}/{len(selected)}, "
            "automatic_integration_allowed=False"
        )
        return 0
    except (DatasetError, ArmatureBatchError, OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
