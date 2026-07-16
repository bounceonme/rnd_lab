# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


class CheckpointExperimentError(RuntimeError):
    """Raised when a checkpoint cannot be verified against the selected task experiment."""


def validate_checkpoint_experiment(
    checkpoint_path: str | Path,
    current_experiment_name: str,
    *,
    allow_task_mismatch: bool = False,
) -> str | None:
    """Fail closed unless a checkpoint's saved experiment matches the current task.

    Args:
        checkpoint_path: Path to the selected model checkpoint in an RSL-RL run directory.
        current_experiment_name: Experiment name from the current task's agent configuration.
        allow_task_mismatch: Explicitly bypass the metadata check when set.

    Returns:
        The checkpoint's saved experiment name, or ``None`` when the check is bypassed.

    Raises:
        CheckpointExperimentError: If metadata is unavailable, invalid, or does not match.
    """
    if allow_task_mismatch:
        return None

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    agent_config_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not agent_config_path.is_file():
        raise CheckpointExperimentError(
            "Cannot verify the checkpoint task experiment because its saved agent configuration is missing: "
            f"{agent_config_path}. Pass --allow_task_mismatch only if this checkpoint is intentionally being used "
            "with a different or unverifiable task configuration."
        )

    try:
        saved_agent_cfg = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointExperimentError(
            f"Cannot read checkpoint agent configuration {agent_config_path}: {exc}"
        ) from exc

    if not isinstance(saved_agent_cfg, Mapping):
        raise CheckpointExperimentError(
            f"Checkpoint agent configuration must be a mapping: {agent_config_path}"
        )

    checkpoint_experiment_name = saved_agent_cfg.get("experiment_name")
    if not isinstance(checkpoint_experiment_name, str) or not checkpoint_experiment_name:
        raise CheckpointExperimentError(
            f"Checkpoint agent configuration has no valid experiment_name: {agent_config_path}"
        )

    if checkpoint_experiment_name != current_experiment_name:
        raise CheckpointExperimentError(
            "Checkpoint/task experiment mismatch: checkpoint "
            f"{checkpoint_path} was trained for {checkpoint_experiment_name!r}, but the selected task uses "
            f"{current_experiment_name!r}. Select the matching task/checkpoint, or pass --allow_task_mismatch "
            "only for an intentional cross-task experiment."
        )

    return checkpoint_experiment_name
