#!/usr/bin/env python3
"""Guided, read-only dynamic consistency test for the RND STEP CMP10A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_HARDWARE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_HARDWARE_DIR))

from cmp10a import CMP10AParser, CMP10ASerialError, CMP10ASerialReader
from rnd_imu.collector import (
    ImuCollectionError,
    build_oscillation_cues,
    collect_cued_stage,
    collect_stage,
    parser_stats,
    resolve_serial_port,
)
from rnd_imu.dynamic import analyze_dynamic_imu_dataset
from rnd_imu.dynamic_config import DynamicImuConfigError, load_dynamic_imu_config
from rnd_imu.identification import load_imu_dataset, save_imu_dataset, write_identification_report


DEFAULT_CONFIG = _TOOL_DIR / "config" / "rnd_cmp10a_dynamic.toml"
DYNAMIC_STAGES = (
    (
        "dynamic_axis_x",
        "X_B AXIS: keep the robot rigid and rock the complete body forward/backward. "
        "Target about 10-15 degrees on each side.",
        "MOVE TOP FORWARD",
        "MOVE TOP BACKWARD",
    ),
    (
        "dynamic_axis_y",
        "Y_B AXIS: keep the robot rigid and rock the complete body left/right. "
        "Directions are from the robot's point of view; target about 10-15 degrees.",
        "MOVE TOP LEFT",
        "MOVE TOP RIGHT",
    ),
    (
        "dynamic_axis_z",
        "Z_B AXIS: keep the robot upright and rotate the complete body left/right about the vertical axis. "
        "Target about 15-20 degrees on each side.",
        "TURN ROBOT LEFT",
        "TURN ROBOT RIGHT",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect guided CMP10A X/Y/Z oscillations and compare Euler-derived angular velocity with gyro output. "
            "The tool is read-only and never enables motor torque."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true", help="Run the guided hardware collection and analyze it.")
    mode.add_argument("--analyze", metavar="DATASET", help="Re-analyze an existing dynamic CMP10A NPZ dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Dynamic CMP10A test TOML.")
    parser.add_argument("--port", help="Serial device; defaults to the config value.")
    parser.add_argument("--baud", type=int, help="Known CMP10A baud rate; defaults to the config value.")
    parser.add_argument(
        "--identification-report",
        help="Passing static/axis report used as provenance. Required for --collect.",
    )
    parser.add_argument("--output", help="Output NPZ path for --collect.")
    parser.add_argument("--report", help="Output JSON report path.")
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="Required before --collect may open a serial device.",
    )
    return parser


def _confirm(prompt: str):
    answer = input(f"{prompt}\nPress Enter when ready, or type q to abort: ").strip().lower()
    if answer in {"q", "quit", "abort"}:
        raise KeyboardInterrupt


def _countdown(seconds: int):
    for remaining in range(seconds, 0, -1):
        print(f"  capture starts in {remaining}...", flush=True)
        time.sleep(1.0)


def _load_identification_provenance(path: str | Path) -> tuple[Path, dict, str]:
    report_path = Path(path).expanduser().resolve()
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read identification report {report_path}: {error}") from error
    if not report.get("runtime_gate", {}).get("quality_pass"):
        raise ValueError(f"Identification report did not pass its runtime gate: {report_path}")
    signed_mapping = report.get("mount_axis_identification", {}).get("signed_axis_approximation")
    expected_mapping = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    if signed_mapping != expected_mapping:
        raise ValueError(
            "The accepted CMP10A mapping must be diag(-1, -1, +1) for the current aligned STEP installation, "
            f"got {signed_mapping}."
        )
    return report_path, report, hashlib.sha256(raw).hexdigest()


def _communication_summary(data: dict, maximum_checksum_error_fraction: float) -> dict:
    stats = dict(data.get("metadata", {}).get("parser_stats", {}))
    valid = int(stats.get("valid_frames", data["frame_type"].shape[0]))
    failed = int(stats.get("checksum_failures", 0))
    error_fraction = failed / max(valid + failed, 1)
    return {
        "valid_frames": valid,
        "checksum_failures": failed,
        "garbage_bytes": int(stats.get("garbage_bytes", 0)),
        "checksum_error_fraction": error_fraction,
        "maximum_checksum_error_fraction": maximum_checksum_error_fraction,
        "quality_pass": error_fraction <= maximum_checksum_error_fraction,
    }


def _analyze(data: dict, config) -> dict:
    report = analyze_dynamic_imu_dataset(
        data,
        max_lag_ms=config.quality.maximum_relative_lag_ms,
        minimum_correlation=config.quality.minimum_correlation,
        minimum_axis_rms_rad_s=config.quality.minimum_axis_rms_rad_s,
        minimum_dominance_ratio=config.quality.minimum_dominance_ratio,
    )
    communication = _communication_summary(data, config.quality.maximum_checksum_error_fraction)
    metadata = dict(data.get("metadata", {}))
    report["communication"] = communication
    report["policy_hz"] = config.experiment.policy_hz
    report["source"] = {
        "experiment": metadata.get("experiment"),
        "device": metadata.get("device"),
        "baudrate": metadata.get("baudrate"),
        "identification_report": metadata.get("identification_report"),
        "identification_report_sha256": metadata.get("identification_report_sha256"),
        "sensor_to_base_signed_mapping": metadata.get("sensor_to_base_signed_mapping"),
        "mount_location": metadata.get("mount_location"),
    }
    report["quality_pass"] = bool(report.get("quality_pass") and communication["quality_pass"])
    return report


def _collect(device: str, baudrate: int, config, identification_path: Path, identification: dict, digest: str):
    parser = CMP10AParser()
    records: list[dict] = []
    experiment = config.experiment
    print(
        "\nSecurely support the complete robot, stop every walking/controller process, and keep motor torque OFF.\n"
        "Follow each target cue smoothly. Do not move the CMP10A relative to Upper_Body."
    )
    with CMP10ASerialReader(device, baudrate=baudrate, timeout=config.serial.read_timeout_s) as reader:
        _confirm(
            "BASELINE: hold the complete robot still in its current pose for "
            f"{experiment.baseline_duration_s:.0f}s. The pose does not need to be world-upright."
        )
        _countdown(experiment.countdown_s)
        print("[INFO] Capturing dynamic_baseline...")
        records.extend(collect_stage(reader, parser, "dynamic_baseline", experiment.baseline_duration_s))

        for stage, instruction, positive_label, negative_label in DYNAMIC_STAGES:
            _confirm(
                f"RESET: return to the center pose and hold it still.\n{instruction}\n"
                f"After MOVE NOW, hold center for {experiment.neutral_duration_s:.0f}s, then follow the cues. "
                f"Use the full {experiment.half_cycle_s:.1f}s to move smoothly from one side to the other "
                f"for {experiment.cycles} cycles; do not snap to the target and wait."
            )
            _countdown(experiment.countdown_s)
            cues = build_oscillation_cues(
                neutral_duration_s=experiment.neutral_duration_s,
                half_cycle_s=experiment.half_cycle_s,
                cycles=experiment.cycles,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            print(f"[INFO] Capturing {stage}; MOVE NOW and watch the cues...")
            stage_records = collect_cued_stage(
                reader,
                parser,
                stage,
                experiment.stage_duration_s,
                cues,
                cue_callback=lambda label: print(f"\a[CUE] {label}", flush=True),
            )
            records.extend(stage_records)
            print(f"[INFO] {stage} complete; decoded frames={len(stage_records)}")

    metadata = {
        "schema_version": 1,
        "experiment": "cmp10a_dynamic_consistency",
        "sensor": "Yahboom CMP10A",
        "device": device,
        "baudrate": baudrate,
        "host_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "parser_stats": parser_stats(parser),
        "identification_report": str(identification_path),
        "identification_report_sha256": digest,
        "sensor_to_base_signed_mapping": identification["mount_axis_identification"]["signed_axis_approximation"],
        "mount_location": "top_of_Upper_Body",
        "base_link_to_upper_body_joint": "fixed",
        "stage_duration_s": experiment.stage_duration_s,
        "half_cycle_s": experiment.half_cycle_s,
        "cycles": experiment.cycles,
    }
    return records, metadata


def _print_summary(report: dict):
    for stage, metrics in report.get("stages", {}).items():
        print(
            f"{stage}: pass={metrics.get('quality_pass')}, delay={metrics.get('delay_ms')} ms, "
            f"correlation={metrics.get('correlation')}, "
            f"dominance={metrics.get('dominant_axis_rms_ratio')}"
        )
    print(
        f"median_relative_delay_ms={report.get('median_relative_delay_ms')}, "
        f"communication_pass={report.get('communication', {}).get('quality_pass')}, "
        f"quality_pass={report.get('quality_pass')}"
    )


def main() -> int:
    args = _parser().parse_args()
    if args.collect and not args.enable_hardware:
        print("[ERROR] --collect requires --enable-hardware.", file=sys.stderr)
        return 2
    try:
        config = load_dynamic_imu_config(args.config)
        if args.analyze:
            data = load_imu_dataset(args.analyze)
            report = _analyze(data, config)
            report_path = (
                Path(args.report).expanduser().resolve()
                if args.report
                else Path(args.analyze)
                .expanduser()
                .resolve()
                .with_name(f"{Path(args.analyze).stem}_dynamic_report.json")
            )
            report_path = write_identification_report(report_path, report)
            _print_summary(report)
            print(f"Saved dynamic report: {report_path}")
            return 0 if report["quality_pass"] else 1

        if not args.identification_report:
            raise ValueError("--collect requires --identification-report from the passing static/axis run.")
        identification_path, identification, digest = _load_identification_provenance(args.identification_report)
        device = resolve_serial_port(args.port or config.serial.port)
        baudrate = args.baud or config.serial.baudrate
        if baudrate <= 0:
            raise ValueError("baudrate must be positive.")
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else _REPO_ROOT / "logs" / "rnd_imu" / f"rnd_cmp10a_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        records, metadata = _collect(device, baudrate, config, identification_path, identification, digest)
        output = save_imu_dataset(output, records, metadata)
        report = _analyze(load_imu_dataset(output), config)
        report_path = (
            Path(args.report).expanduser().resolve() if args.report else output.with_name(f"{output.stem}_report.json")
        )
        report_path = write_identification_report(report_path, report)
        _print_summary(report)
        print(f"Saved dynamic dataset: {output}")
        print(f"Saved dynamic report:  {report_path}")
        return 0 if report["quality_pass"] else 1
    except KeyboardInterrupt:
        print("\n[ERROR] Dynamic IMU test interrupted; no sensor settings were changed.", file=sys.stderr)
        return 130
    except (
        CMP10ASerialError,
        DynamicImuConfigError,
        ImuCollectionError,
        OSError,
        ValueError,
    ) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
