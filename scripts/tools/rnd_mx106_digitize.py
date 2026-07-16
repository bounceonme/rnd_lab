#!/usr/bin/env python3
"""Digitize the official ROBOTIS MX-106 current/torque performance curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SOURCE_URL = "https://emanual.robotis.com/assets/images/dxl/mx/mx-106_ntgraph_2.jpg"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "config" / "mx106_performance_lut.json"


class DigitizationError(ValueError):
    """Raised when the official graph cannot be digitized reproducibly."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Locally downloaded official mx-106_ntgraph_2.jpg.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output LUT JSON.")
    parser.add_argument(
        "--torque-step",
        type=float,
        default=0.05,
        help="Interpolated LUT torque spacing in Nm; 0.05 Nm is close to two source-image pixels.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digitize_graph(image_path: str | Path, *, torque_step: float = 0.05) -> dict:
    """Extract the magenta current curve using calibrated graph-axis pixels."""

    path = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DigitizationError(f"Could not read graph image: {path}")
    if image.shape[:2] != (381, 389):
        raise DigitizationError(f"Unexpected official graph dimensions {image.shape[:2]}; expected (381, 389).")
    if not np.isfinite(torque_step) or torque_step <= 0.0:
        raise DigitizationError("torque_step must be finite and positive.")

    axis = {
        "x_min_px": 78,
        "x_max_px": 329,
        "torque_min_nm": 0.0,
        "torque_max_nm": 6.0,
        "y_min_px": 311,
        "y_max_px": 58,
        "current_min_a": 0.0,
        "current_max_a": 5.0,
    }
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 160) & (hsv[:, :, 0] <= 175) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 70)
    crop = np.zeros_like(mask)
    crop[axis["y_max_px"] : axis["y_min_px"] + 1, axis["x_min_px"] : axis["x_max_px"] + 1] = True
    mask &= crop

    curve_x: list[float] = []
    curve_y: list[float] = []
    for x_px in range(axis["x_min_px"], axis["x_max_px"] + 1):
        y_pixels = np.flatnonzero(mask[:, x_px])
        if y_pixels.size:
            curve_x.append(float(x_px))
            curve_y.append(float(np.median(y_pixels)))
    if len(curve_x) < 150:
        raise DigitizationError(f"Only {len(curve_x)} curve columns were found; expected at least 150.")

    curve_x_array = np.asarray(curve_x)
    curve_y_array = np.asarray(curve_y)
    observed_torque = (curve_x_array - axis["x_min_px"]) * axis["torque_max_nm"] / (axis["x_max_px"] - axis["x_min_px"])
    observed_current = (
        (axis["y_min_px"] - curve_y_array) * axis["current_max_a"] / (axis["y_min_px"] - axis["y_max_px"])
    )
    first_torque = np.ceil(observed_torque.min() / torque_step) * torque_step
    last_torque = np.floor(observed_torque.max() / torque_step) * torque_step
    torque_nm = np.arange(first_torque, last_torque + 0.5 * torque_step, torque_step)
    current_a = np.interp(torque_nm, observed_torque, observed_current)
    current_a = np.maximum.accumulate(current_a)
    if np.any(np.diff(current_a) <= 0.0):
        raise DigitizationError("Digitized current samples are not strictly increasing.")

    return {
        "schema_version": 1,
        "model_type": "mx106_performance_graph_current_to_output_torque_lut",
        "analysis_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "manufacturer": "ROBOTIS",
            "product": "DYNAMIXEL MX-106T/R",
            "document_url": "https://emanual.robotis.com/docs/kr/dxl/mx/mx-106/",
            "image_url": SOURCE_URL,
            "image_sha256": _sha256(path),
            "image_width_px": int(image.shape[1]),
            "image_height_px": int(image.shape[0]),
            "note": "Official graph image was digitized at native resolution; no upscaling adds source information.",
        },
        "digitization": {
            "axis_calibration": axis,
            "curve_color": "magenta current curve",
            "hsv_threshold_opencv": {"hue": [160, 175], "saturation_min": 60, "value_min": 70},
            "column_reduction": "median curve pixel per x column",
            "observed_column_count": len(curve_x),
            "observed_torque_domain_nm": [float(observed_torque.min()), float(observed_torque.max())],
            "observed_current_domain_a": [float(observed_current.min()), float(observed_current.max())],
            "output_torque_step_nm": float(torque_step),
            "estimated_curve_pick_uncertainty_px": 1.5,
            "native_curve": {
                "torque_nm": [round(float(value), 6) for value in observed_torque],
                "current_a": [round(float(value), 6) for value in observed_current],
            },
        },
        "curve": {
            "torque_nm": [round(float(value), 6) for value in torque_nm],
            "current_a": [round(float(value), 6) for value in current_a],
        },
        "conversion": {
            "input": "signed Present Current in joint coordinates",
            "output": "approximate signed servo output-shaft torque",
            "sign_convention": "odd_symmetric_about_zero",
            "below_observed_curve": "linear_origin_to_first_point_unvalidated",
            "above_observed_curve": "clip_to_last_point",
            "warning": (
                "The manufacturer graph is a gradually loaded steady-state test, not a torque-sensor calibration. "
                "Low-current suspended-leg samples fall outside its observed domain."
            ),
        },
        "reference_specification": {
            "recommended_voltage_v": 12.0,
            "stall_torque_nm": 8.4,
            "stall_current_a": 5.2,
            "stall_torque_per_amp_nm": 1.615,
            "use": "sensitivity_reference_only_not_part_of_performance_curve_lut",
        },
    }


def _atomic_write(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(model, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    args = _parser().parse_args()
    model = digitize_graph(args.image, torque_step=args.torque_step)
    destination = Path(args.output).expanduser().resolve()
    _atomic_write(destination, model)
    print(f"Saved MX-106 performance LUT: {destination}")
    print(
        f"Observed graph domain: current={model['digitization']['observed_current_domain_a']} A, "
        f"torque={model['digitization']['observed_torque_domain_nm']} Nm"
    )


if __name__ == "__main__":
    main()
