#!/usr/bin/env python3
"""Analyze every recorded RND joint from one cached PhysX dynamics trace."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rnd_real2sim.dataset import DatasetError, load_dataset
from rnd_real2sim.torque_batch import TorqueBatchError, analyze_joints, load_dynamics_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse one cached PhysX trace and analyze all excited joints without starting Isaac Sim again."
    )
    parser.add_argument("--dataset", required=True, help="Complete multi-joint rnd_real2sim NPZ dataset.")
    parser.add_argument(
        "--dynamics-trace",
        help="Cached *_torque_friction.npz. Defaults to the conventional path next to the dataset.",
    )
    parser.add_argument("--joint", action="append", help="Optional joint subset; omitted means every excited joint.")
    parser.add_argument("--output", help="Aggregate JSON output path.")
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
        trace = load_dynamics_trace(args.dynamics_trace or default_trace, dataset)
        excited = dataset.metadata.get("excitation_joint_names")
        if not isinstance(excited, list) or not excited:
            raise TorqueBatchError("Dataset metadata contains no excitation_joint_names.")
        selected = tuple(args.joint or excited)
        results, summary = analyze_joints(dataset, trace, selected)
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else dataset.path.with_suffix("").with_name(f"{dataset.path.stem}_all_joint_torque_calibration.json")
        )
        report = {
            "schema_version": 1,
            "model_type": "rnd_real2sim_all_joint_torque_calibration",
            "analysis_only": True,
            "integration_enabled": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset.path),
            "source_dataset_sha256": dataset.sha256,
            "source_dynamics_trace": str(trace.path),
            "source_dynamics_trace_sha256": trace.sha256,
            "joints": results,
            "summary": summary,
        }
        _atomic_json(output, report)
        print(f"Saved all-joint torque calibration report: {output}")
        for joint_name, result in results.items():
            calibration = result["low_current_torque_calibration"]
            if calibration.get("quality", {}).get("pass"):
                interval = calibration["bootstrap_90pct_nm_per_a"]
                print(
                    f"[PASS] {joint_name:27s} Kt={calibration['torque_per_amp_nm']:.4f} Nm/A "
                    f"CI90=[{interval[0]:.4f}, {interval[1]:.4f}] "
                    f"Ic={calibration['coulomb_current_a']:.4f} A"
                )
            else:
                reasons = calibration.get("quality", {}).get("reasons") or [calibration.get("reason", "unknown")]
                print(f"[FAIL] {joint_name:27s} {' '.join(str(reason) for reason in reasons)}")
        print(
            f"calibration_passed={len(summary['calibration_passed_joints'])}/{len(selected)}, "
            "automatic_integration_allowed=False"
        )
        return 0
    except (DatasetError, TorqueBatchError, OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
