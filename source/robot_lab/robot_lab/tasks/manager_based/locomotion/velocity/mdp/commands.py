# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vel_command_target_b = torch.zeros_like(self.vel_command_b)
        self.command_ramp_rates = (
            torch.tensor(cfg.command_ramp_rates, device=self.device).unsqueeze(0)
            if cfg.command_ramp_rates is not None
            else None
        )

    def _sample_command_target(self, env_ids: Sequence[int]):
        """Sample the target command while the exposed command can ramp toward it."""
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_target_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_target_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_target_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        # set small commands to zero
        threshold = getattr(self.cfg, "zero_velocity_threshold", 0.2)
        self.vel_command_target_b[env_ids, :2] *= (
            torch.norm(self.vel_command_target_b[env_ids, :2], dim=1) > threshold
        ).unsqueeze(1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample target velocity commands with threshold."""
        previous_command = self.vel_command_b[env_ids].clone()
        self._sample_command_target(env_ids)
        if self.command_ramp_rates is None:
            self.vel_command_b[env_ids] = self.vel_command_target_b[env_ids]
        else:
            self.vel_command_b[env_ids] = previous_command

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        previous_command = self.vel_command_b.clone()

        # First, call parent's update command on the sampled target.
        self.vel_command_b[:] = self.vel_command_target_b
        super()._update_command()

        # Isaac Lab resolves heading from the body x-axis by default.
        # Some robots in this workspace use a different physical forward axis.
        if self.cfg.heading_command and self.cfg.heading_offset != 0.0:
            heading_env_ids = torch.logical_and(self.is_heading_env, ~self.is_standing_env).nonzero(
                as_tuple=False
            ).flatten()
            if len(heading_env_ids) > 0:
                forward_heading = self.robot.data.heading_w[heading_env_ids] + self.cfg.heading_offset
                heading_error = math_utils.wrap_to_pi(self.heading_target[heading_env_ids] - forward_heading)
                self.vel_command_b[heading_env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._sample_command_target(left_pit_env_ids)
            self.vel_command_b[left_pit_env_ids] = self.vel_command_target_b[left_pit_env_ids]

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0
            self.vel_command_target_b[pit_env_ids] = self.vel_command_b[pit_env_ids]

        if self.command_ramp_rates is not None:
            max_delta = self.command_ramp_rates * self._env.step_dt
            self.vel_command_b[:] = previous_command + torch.clamp(
                self.vel_command_b - previous_command,
                min=-max_delta,
                max=max_delta,
            )

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand
    zero_velocity_threshold: float = 0.2
    heading_offset: float = 0.0
    command_ramp_rates: tuple[float, float, float] | None = None


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
