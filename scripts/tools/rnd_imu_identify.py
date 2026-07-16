#!/usr/bin/env python3
"""Read-only CMP10A probe and mount-axis identification for RND STEP."""

from __future__ import annotations

import argparse
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
    collect_stage,
    discover_serial_ports,
    parser_stats,
    probe_baudrates,
    resolve_serial_port,
    select_probe_result,
)
from rnd_imu.config import ImuIdentificationConfigError, load_imu_identification_config
from rnd_imu.identification import (
    FRAME_NAMES,
    identify_imu_dataset,
    load_imu_dataset,
    save_imu_dataset,
    write_identification_report,
)


DEFAULT_CONFIG = _TOOL_DIR / "config" / "rnd_cmp10a.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify the Yahboom CMP10A stream needed by the RND STEP 50 Hz policy. "
            "The tool is strictly read-only: it never sends configuration or calibration commands to the sensor."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-ports", action="store_true", help="List candidate serial devices without opening them.")
    mode.add_argument("--probe", action="store_true", help="Read-only baud and packet-type detection.")
    mode.add_argument(
        "--identify",
        action="store_true",
        help="Collect upright static data and three guided positive base-axis rotations, then analyze them.",
    )
    mode.add_argument("--analyze", metavar="DATASET", help="Re-analyze an existing CMP10A .npz dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="CMP10A identification TOML.")
    parser.add_argument("--port", help="Serial device. Required if auto-discovery finds multiple devices.")
    parser.add_argument("--baud", type=int, help="Known baud rate; omitted means read-only auto-probe.")
    parser.add_argument("--output", help="Output .npz path for --identify.")
    parser.add_argument("--report", help="Output .json path for --identify or --analyze.")
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="Required before --probe or --identify may open a serial device.",
    )
    return parser


def _print_ports() -> int:
    ports = discover_serial_ports()
    if not ports:
        print("No /dev/serial/by-id, ttyUSB, or ttyACM devices found.")
        return 1
    print("Candidate serial devices:")
    for port in ports:
        print(f"  {port}")
    return 0


def _probe(device: str, baudrates: tuple[int, ...], config):
    results = probe_baudrates(
        device,
        baudrates,
        config.serial.probe_duration_s,
        timeout_s=config.serial.read_timeout_s,
    )
    print(f"Read-only CMP10A probe: device={device}")
    for result in results:
        types = ", ".join(
            f"{FRAME_NAMES.get(frame_type, hex(frame_type))}={count}"
            for frame_type, count in result.frame_type_counts.items()
        )
        print(
            f"  baud={result.baudrate:6d} valid={result.valid_frames:5d} "
            f"checksum_failures={result.checksum_failures:4d} garbage={result.garbage_bytes:6d} "
            f"types=[{types}]"
        )
    selected = select_probe_result(results, config.serial.minimum_valid_probe_frames)
    print(f"[PASS] Selected baud={selected.baudrate}; no bytes were written to the sensor.")
    return selected


def _countdown(seconds: int):
    for remaining in range(seconds, 0, -1):
        print(f"  capture starts in {remaining}...", flush=True)
        time.sleep(1.0)


def _confirm(prompt: str):
    answer = input(f"{prompt}\nPress Enter when ready, or type q to abort: ").strip().lower()
    if answer in {"q", "quit", "abort"}:
        raise KeyboardInterrupt


def _collect_identification(device: str, baudrate: int, config, output: Path) -> tuple[Path, Path, dict]:
    parser = CMP10AParser()
    records: list[dict] = []
    print(
        "\nThe robot must be securely supported, with no walking policy running. "
        "This program reads only the IMU and never enables motor torque."
    )
    with CMP10ASerialReader(device, baudrate=baudrate, timeout=config.serial.read_timeout_s) as reader:
        _confirm(
            "STATIC: place STEP upright and keep it completely still. "
            f"The capture lasts {config.experiment.static_duration_s:.0f}s."
        )
        _countdown(config.experiment.countdown_s)
        print("[INFO] Capturing static_upright...")
        records.extend(collect_stage(reader, parser, "static_upright", config.experiment.static_duration_s))
        print(f"[INFO] Static capture complete; decoded frames={len(records)}")

        static_types = {record["frame_type"] for record in records}
        if 0x52 not in static_types:
            raise ImuCollectionError("The stream has no angular-velocity (0x52) frames.")
        if 0x53 not in static_types and 0x59 not in static_types:
            raise ImuCollectionError("The stream has neither Euler-angle (0x53) nor quaternion (0x59) frames.")

        axis_instructions = (
            (
                "axis_pos_x",
                "+X_B points to the robot's LEFT. During capture, make one smooth positive +X rotation: "
                "tilt the top of the robot FORWARD, then hold it there until capture ends.",
            ),
            (
                "axis_pos_y",
                "+Y_B points BACKWARD. During capture, make one smooth positive +Y rotation: "
                "tilt the top of the robot to its LEFT, then hold it there until capture ends.",
            ),
            (
                "axis_pos_z",
                "+Z_B points UP. During capture, make one smooth positive +Z rotation: "
                "turn the robot LEFT when viewed from above, then hold it there until capture ends.",
            ),
        )
        for stage, instruction in axis_instructions:
            _confirm(
                f"RESET: return the robot to the upright starting pose.\n{instruction}\n"
                f"Move only after the countdown; capture lasts {config.experiment.axis_duration_s:.0f}s."
            )
            _countdown(config.experiment.countdown_s)
            print(f"[INFO] Capturing {stage}; MOVE NOW...")
            stage_records = collect_stage(reader, parser, stage, config.experiment.axis_duration_s)
            records.extend(stage_records)
            print(f"[INFO] {stage} complete; decoded frames={len(stage_records)}")

    metadata = {
        "schema_version": 1,
        "sensor": "Yahboom CMP10A",
        "device": device,
        "baudrate": baudrate,
        "host_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "parser_stats": parser_stats(parser),
        "upper_body_mount": True,
        "mount_location": "top_of_Upper_Body",
        "mount_translation_m": None,
        "base_link_to_upper_body_joint": "fixed",
    }
    output = save_imu_dataset(output, records, metadata)
    data = load_imu_dataset(output)
    report = identify_imu_dataset(data, config)
    report_path = output.with_name(f"{output.stem}_report.json")
    report_path = write_identification_report(report_path, report)
    return output, report_path, report


def _print_report_summary(report: dict):
    gyro = report["static_gyro"]
    accel = report["static_accelerometer"]
    mount = report["mount_axis_identification"]
    gate = report["runtime_gate"]
    print(f"static gyro bias [rad/s]: {gyro['bias_rad_s']}")
    print(f"static gyro std  [rad/s]: {gyro['std_rad_s']}")
    print(f"accel norm: mean={accel['norm_mean_mps2']:.4f} m/s^2, std={accel['norm_std_mps2']:.4f} m/s^2")
    print(f"sensor_to_base_matrix: {mount.get('sensor_to_base_matrix')}")
    print(
        f"runtime gate: pass={gate['quality_pass']}, gyro={gate['gyro_rate_hz']} Hz, "
        f"orientation={gate['orientation_source']} at {gate['orientation_rate_hz']} Hz"
    )


def main() -> int:
    args = _parser().parse_args()
    if args.list_ports:
        return _print_ports()
    if (args.probe or args.identify) and not args.enable_hardware:
        print("[ERROR] --probe and --identify require --enable-hardware.", file=sys.stderr)
        return 2

    try:
        config = load_imu_identification_config(args.config)
        if args.analyze:
            data = load_imu_dataset(args.analyze)
            report = identify_imu_dataset(data, config)
            report_path = (
                Path(args.report)
                if args.report
                else Path(args.analyze).with_name(f"{Path(args.analyze).stem}_report.json")
            )
            write_identification_report(report_path, report)
            _print_report_summary(report)
            print(f"Saved report: {report_path.resolve()}")
            return 0 if report["runtime_gate"]["quality_pass"] else 1

        requested_port = args.port or config.serial.port
        device = resolve_serial_port(requested_port)
        baudrates = (args.baud,) if args.baud else config.serial.baud_candidates
        selected = _probe(device, baudrates, config)
        if args.probe:
            return 0

        output = (
            Path(args.output)
            if args.output
            else _REPO_ROOT / "logs" / "rnd_imu" / f"rnd_cmp10a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        output, default_report_path, report = _collect_identification(device, selected.baudrate, config, output)
        if args.report:
            requested_report = Path(args.report)
            write_identification_report(requested_report, report)
            report_path = requested_report
            if default_report_path != requested_report.resolve():
                default_report_path.unlink(missing_ok=True)
        else:
            report_path = default_report_path
        _print_report_summary(report)
        print(f"Saved dataset: {output}")
        print(f"Saved report:  {report_path.resolve()}")
        return 0 if report["runtime_gate"]["quality_pass"] else 1
    except KeyboardInterrupt:
        print("\n[ERROR] IMU identification interrupted; no sensor settings were changed.", file=sys.stderr)
        return 130
    except (
        ImuCollectionError,
        ImuIdentificationConfigError,
        CMP10ASerialError,
        OSError,
        ValueError,
    ) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
