#!/usr/bin/env python3
"""Collect standalone MX-106 telemetry for RND STEP actuator identification."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
sys.path.insert(0, str(_TOOL_DIR))

from rnd_real2sim.bus import DynamixelReal2SimError, Mx2TelemetryBus
from rnd_real2sim.collector import SafetyTrip, collect_dataset
from rnd_real2sim.config import (
    RND_LEG_JOINT_NAMES,
    Real2SimConfigError,
    load_experiment_config,
    load_mapping_config,
)
from rnd_real2sim.synthetic import SyntheticMx2Bus


DEFAULT_MAPPING = _TOOL_DIR / "config" / "rnd_dynamixel.toml"
DEFAULT_EXPERIMENT = _TOOL_DIR / "config" / "rnd_real2sim.toml"
DEFAULT_URDF = _REPO_ROOT / "source" / "robot_lab" / "data" / "Robots" / "rnd" / "step" / "urdf" / "step.urdf"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Excite a rigidly suspended RND STEP robot and synchronously record MX-106(2.0) encoder/current/PWM "
            "telemetry. This tool does not start Omniverse and is independent of rnd_joint_coordinate_test.py."
        )
    )
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="Verified motor ID/zero/direction TOML.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT), help="Real-to-sim experiment TOML.")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="URDF used only to cross-check joint limits.")
    parser.add_argument(
        "--joint",
        action="append",
        choices=RND_LEG_JOINT_NAMES,
        help="Joint to excite. Repeat for multiple joints; omitted means all 12 leg joints.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help="Profile name from rnd_real2sim.toml. Repeat as needed; omitted means every profile.",
    )
    parser.add_argument("--output", help="Output .npz path. Defaults to logs/rnd_real2sim/<timestamp>.npz.")
    parser.add_argument("--dry-run", action="store_true", help="Run immediately with a deterministic synthetic bus.")
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="Required before the serial port can be opened.",
    )
    parser.add_argument(
        "--confirm-rigidly-fixed",
        action="store_true",
        help="Confirm the upper body is rigidly fixed, not merely hanging from a flexible rope.",
    )
    parser.add_argument(
        "--confirm-clearance",
        action="store_true",
        help="Confirm every leg link has clearance during the automatic reference-pose move and excitation.",
    )
    parser.add_argument(
        "--inspect-runtime-only",
        action="store_true",
        help="Torque OFF all motors, print controller settings, and exit without moving or collecting data.",
    )
    return parser


def _select_profiles(experiment, requested_names: list[str] | None):
    if not requested_names:
        return experiment.profiles
    by_name = {profile.name: profile for profile in experiment.profiles}
    unknown = sorted(set(requested_names) - set(by_name))
    if unknown:
        raise Real2SimConfigError(f"Unknown profiles {unknown}; available={sorted(by_name)}")
    if len(requested_names) != len(set(requested_names)):
        raise Real2SimConfigError("--profile values must not be repeated.")
    return tuple(by_name[name] for name in requested_names)


def _describe_runtime_settings(mapping, runtime_info) -> str:
    lines = [
        "Runtime controller settings (raw control-table values):",
        "joint                       ID     P     I     D   FF1   FF2   PWM  Current  VelLimit  HomeOffset",
    ]
    for joint in mapping.joints:
        info = runtime_info[joint.name]
        lines.append(
            f"{joint.name:27s} {joint.motor_id:3d}  {info.position_p_gain:4d}  {info.position_i_gain:4d}  "
            f"{info.position_d_gain:4d}  {info.feedforward_1st_gain:4d}  {info.feedforward_2nd_gain:4d}  "
            f"{info.pwm_limit_raw:4d}  {info.current_limit_raw:7d}  {info.velocity_limit_raw:8d}  "
            f"{info.homing_offset_raw:+10d}"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    if args.dry_run and args.enable_hardware:
        print("[ERROR] --dry-run and --enable-hardware are mutually exclusive.", file=sys.stderr)
        return 2
    if not args.dry_run and not (args.enable_hardware and args.confirm_rigidly_fixed and args.confirm_clearance):
        print(
            "[ERROR] Hardware collection requires --enable-hardware --confirm-rigidly-fixed "
            "--confirm-clearance. Use --dry-run to test without opening the port.",
            file=sys.stderr,
        )
        return 2

    try:
        mapping = load_mapping_config(args.mapping)
        experiment = load_experiment_config(args.config)
        excitation_joints = tuple(args.joint or RND_LEG_JOINT_NAMES)
        if len(excitation_joints) != len(set(excitation_joints)):
            raise Real2SimConfigError("--joint values must not be repeated.")
        profiles = _select_profiles(experiment, args.profile)
        output = (
            Path(args.output)
            if args.output
            else _REPO_ROOT / "logs" / "rnd_real2sim" / f"rnd_real2sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        bus = SyntheticMx2Bus(mapping, experiment.sample_hz) if args.dry_run else Mx2TelemetryBus(mapping)

        if args.inspect_runtime_only:
            bus.open()
            try:
                print(_describe_runtime_settings(mapping, bus.runtime_info))
            finally:
                bus.close()
            return 0

        collect_dataset(
            bus=bus,
            mapping=mapping,
            experiment=experiment,
            urdf_path=args.urdf,
            excitation_joint_names=excitation_joints,
            profiles=profiles,
            output_path=output,
            dry_run=args.dry_run,
            confirm_arm=lambda _prompt: True,
        )
        return 0
    except KeyboardInterrupt:
        print("\n[ERROR] Collection interrupted; torque OFF was requested and partial data was saved.", file=sys.stderr)
        return 130
    except (Real2SimConfigError, DynamixelReal2SimError, SafetyTrip, OSError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
