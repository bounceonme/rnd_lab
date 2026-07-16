#!/usr/bin/env python3
"""Replay an RND hardware trace through the stateful command-path seed."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import RndActuatorModelError, StatefulCommandPath, load_rnd_actuator_model


DEFAULT_MODEL = _TOOL_DIR / "config" / "rnd_actuator_model.json"


class TraceReplayError(ValueError):
    """Raised when a hardware trace cannot be compared with the model."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay position targets through the pure command-path model and compare them with encoder data. "
            "This is a trace diagnostic, not the required Isaac simulator replay gate."
        )
    )
    parser.add_argument("dataset", help="Complete rnd_real2sim .npz dataset.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Runtime actuator seed JSON.")
    parser.add_argument("--joint", help="Excited joint; inferred when the dataset contains exactly one.")
    parser.add_argument("--output", help="Output JSON; defaults beside the dataset.")
    parser.add_argument("--device", default="cpu", help="Torch device for the stateful replay.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used only with --sample-randomization.")
    parser.add_argument(
        "--sample-randomization",
        action="store_true",
        help="Sample configured ranges instead of replaying their nominal center/scale.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Permit an unresolved joint's explicit identity placeholder for diagnostics.",
    )
    return parser


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as error:
        raise TraceReplayError(f"Unable to load real2sim dataset {path}: {error}") from error
    if metadata.get("status") != "complete":
        raise TraceReplayError(f"Dataset status must be complete, got {metadata.get('status')!r}.")
    required = {"goal_position_rad", "position_rad", "phase_id"}
    missing = sorted(required - set(arrays))
    if missing:
        raise TraceReplayError(f"Dataset is missing arrays: {missing}.")
    return arrays, metadata


def _select_joint(metadata: dict[str, Any], requested: str | None) -> str:
    joint_names = metadata.get("joint_names")
    excited = metadata.get("excitation_joint_names")
    if not isinstance(joint_names, list) or not isinstance(excited, list):
        raise TraceReplayError("Dataset metadata is missing joint_names or excitation_joint_names.")
    if requested is not None:
        if requested not in joint_names:
            raise TraceReplayError(f"Requested joint {requested!r} is absent from the dataset.")
        if requested not in excited:
            raise TraceReplayError(f"Requested joint {requested!r} was not excited in this dataset.")
        return requested
    if len(excited) != 1:
        raise TraceReplayError(f"Dataset excites {excited}; select one with --joint.")
    return str(excited[0])


def _nominal_model(model: dict[str, Any], joint_name: str) -> dict[str, Any]:
    result = copy.deepcopy(model)
    command_path = result["joints"][joint_name]["command_path"]
    delay_range = command_path["residual_delay_s_range"]
    delay_midpoint = 0.5 * (float(delay_range[0]) + float(delay_range[1]))
    command_path["residual_delay_s_range"] = [delay_midpoint, delay_midpoint]
    bias_range = command_path.get("residual_position_bias_rad_range", [0.0, 0.0])
    bias_midpoint = 0.5 * (float(bias_range[0]) + float(bias_range[1]))
    command_path["residual_position_bias_rad_range"] = [bias_midpoint, bias_midpoint]
    command_path["play_threshold_scale_range"] = [1.0, 1.0]
    return result


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    centered = reference - float(np.mean(reference))
    denominator = float(np.sum(np.square(centered)))
    r2 = None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(error))) / denominator
    span = float(np.ptp(reference))
    return {
        "rmse_rad": rmse,
        "mae_rad": mae,
        "normalized_rmse": None if span <= 1.0e-12 else rmse / span,
        "r2": r2,
    }


def replay_trace(
    dataset_path: str | Path,
    model_path: str | Path,
    *,
    joint_name: str | None = None,
    device: str = "cpu",
    seed: int = 0,
    sample_randomization: bool = False,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    """Run a deterministic or sampled command-path replay over one hardware trace."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    arrays, metadata = _load_dataset(dataset_path)
    selected_joint = _select_joint(metadata, joint_name)
    model = load_rnd_actuator_model(model_path, (selected_joint,))
    joint_quality = model["joints"][selected_joint]["quality"]
    if not joint_quality["command_path_seed_usable"] and not allow_unresolved:
        raise TraceReplayError(
            f"{selected_joint} has no accepted constant-play command path. "
            "Use --allow-unresolved only to replay its identity placeholder."
        )
    replay_model = model if sample_randomization else _nominal_model(model, selected_joint)

    sample_hz = float(metadata["sample_hz"])
    if sample_hz <= 0.0 or not math.isfinite(sample_hz):
        raise TraceReplayError(f"Invalid dataset sample_hz={sample_hz!r}.")
    names = list(metadata["joint_names"])
    joint_index = names.index(selected_joint)
    raw_target = arrays["goal_position_rad"][:, joint_index]
    measured = arrays["position_rad"][:, joint_index]
    if raw_target.shape != measured.shape or raw_target.ndim != 1:
        raise TraceReplayError("Goal and position arrays must have matching one-dimensional joint traces.")

    kernel = StatefulCommandPath(
        replay_model,
        (selected_joint,),
        num_envs=1,
        device=device,
        step_hz=sample_hz,
        dtype=torch.float64,
        seed=seed,
        sample_randomization=sample_randomization,
    )
    initial = torch.tensor([[raw_target[0]]], dtype=torch.float64, device=device)
    kernel.reset(initial)
    transformed = np.empty_like(raw_target)
    with torch.inference_mode():
        for sample_index, value in enumerate(raw_target):
            command = torch.tensor([[value]], dtype=torch.float64, device=device)
            transformed[sample_index] = float(kernel.transform(command)[0, 0].item())

    phase_metadata = metadata.get("phase_metadata", {})
    phase_results: list[dict[str, Any]] = []
    valid_phase = arrays["phase_id"] >= 0
    for phase_id in sorted(int(value) for value in np.unique(arrays["phase_id"][valid_phase])):
        mask = arrays["phase_id"] == phase_id
        phase_info = phase_metadata.get(str(phase_id), {})
        raw_metrics = _metrics(measured[mask], raw_target[mask])
        transformed_metrics = _metrics(measured[mask], transformed[mask])
        phase_results.append({
            "phase_id": phase_id,
            "profile_name": phase_info.get("profile_name"),
            "waveform": phase_info.get("waveform"),
            "sample_count": int(np.count_nonzero(mask)),
            "raw_target": raw_metrics,
            "transformed_target": transformed_metrics,
            "rmse_change_fraction": (
                None
                if raw_metrics["rmse_rad"] == 0.0
                else (transformed_metrics["rmse_rad"] - raw_metrics["rmse_rad"]) / raw_metrics["rmse_rad"]
            ),
        })

    overall_raw = _metrics(measured[valid_phase], raw_target[valid_phase])
    overall_transformed = _metrics(measured[valid_phase], transformed[valid_phase])
    return {
        "schema_version": 1,
        "validation_type": "hardware_trace_command_path_diagnostic",
        "sim_replay_gate_satisfied": False,
        "dataset": str(dataset_path),
        "model": str(Path(model_path).expanduser().resolve()),
        "joint": selected_joint,
        "sample_hz": sample_hz,
        "sample_randomization": sample_randomization,
        "command_path_seed_usable": bool(joint_quality["command_path_seed_usable"]),
        "identity_placeholder_used": not bool(joint_quality["command_path_seed_usable"]),
        "sampled_residual_delay_s": float(kernel.sampled_delay_s[0, 0].item()),
        "sampled_residual_position_bias_rad": float(kernel.sampled_position_bias_rad[0, 0].item()),
        "sampled_play_thresholds_rad": [
            float(value) for value in kernel.sampled_play_thresholds_rad[0, 0].detach().cpu().tolist()
        ],
        "overall": {
            "raw_target": overall_raw,
            "transformed_target": overall_transformed,
        },
        "phases": phase_results,
        "interpretation": (
            "This replay checks the stateful command transformation against encoder traces. It does not include "
            "PhysX, explicit PD dynamics, load dependence, or contact and therefore cannot enable training integration."
        ),
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
    dataset_path = Path(args.dataset).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dataset_path.with_name(f"{dataset_path.stem}_actuator_replay.json")
    )
    try:
        report = replay_trace(
            dataset_path,
            args.model,
            joint_name=args.joint,
            device=args.device,
            seed=args.seed,
            sample_randomization=args.sample_randomization,
            allow_unresolved=args.allow_unresolved,
        )
        _atomic_write_json(output, report)
        print(f"Saved command-path trace diagnostic: {output}")
        print("phase  profile              raw RMSE(rad)  transformed RMSE(rad)  change")
        for phase in report["phases"]:
            raw_rmse = phase["raw_target"]["rmse_rad"]
            transformed_rmse = phase["transformed_target"]["rmse_rad"]
            change = phase["rmse_change_fraction"]
            change_text = "n/a" if change is None else f"{100.0 * change:+.1f}%"
            print(
                f"{phase['phase_id']:5d}  {str(phase['profile_name']):19s}  "
                f"{raw_rmse:13.6f}  {transformed_rmse:21.6f}  {change_text:>7s}"
            )
        print("Simulator replay gate remains unsatisfied.")
        return 0
    except (TraceReplayError, RndActuatorModelError, OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
