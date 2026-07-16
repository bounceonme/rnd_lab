#!/usr/bin/env python3
"""Promote passing CMP10A identification evidence into a tracked runtime JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
sys.path.insert(0, str(_TOOL_DIR))

from rnd_imu.runtime_config import Cmp10aRuntimeConfigError, build_cmp10a_runtime_config_from_files


DEFAULT_STATIC_REPORT = _REPO_ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_20260716_035739_report.json"
DEFAULT_DYNAMIC_REPORT = _REPO_ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_dynamic_20260716_042138_report.json"
DEFAULT_DYNAMIC_DATASET = _REPO_ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_dynamic_20260716_042138.npz"
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_cmp10a_runtime.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote passing CMP10A static and dynamic evidence into the policy-observation runtime model."
    )
    parser.add_argument("--static-report", default=str(DEFAULT_STATIC_REPORT), help="Passing static/axis report JSON.")
    parser.add_argument("--dynamic-report", default=str(DEFAULT_DYNAMIC_REPORT), help="Passing dynamic report JSON.")
    parser.add_argument("--dynamic-dataset", default=str(DEFAULT_DYNAMIC_DATASET), help="Dynamic source NPZ dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output runtime JSON.")
    return parser


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
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
        model = build_cmp10a_runtime_config_from_files(
            args.static_report,
            args.dynamic_report,
            args.dynamic_dataset,
        )
        output = Path(args.output).expanduser().resolve()
        _atomic_write_json(output, model)
        print(f"Saved CMP10A runtime model: {output}")
        print(
            "integration_enabled=true, transform=diag(-1,-1,+1), "
            f"held_baseline_gyro_samples={model['measured']['held_baseline']['gyro']['samples']}"
        )
        return 0
    except (Cmp10aRuntimeConfigError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
