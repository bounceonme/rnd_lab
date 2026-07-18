# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _action_excess_l2(actions: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a hinge penalty for raw actions outside the nominal policy range."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    excess = torch.clamp(torch.abs(actions) - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def action_excess_l2(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize only excessive raw actions while leaving ordinary gait actions untouched."""
    return _action_excess_l2(env.action_manager.action, threshold)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_xy_underspeed_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only the planar velocity deficit along the commanded direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / command_speed.unsqueeze(1).clamp_min(command_threshold)

    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    achieved_along_command = torch.sum(vel_yaw[:, :2] * command_direction, dim=1)
    underspeed = torch.clamp(command_speed - achieved_along_command, min=0.0)

    penalty = torch.square(underspeed)
    penalty *= command_speed > command_threshold
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_heading_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    lin_vel_threshold: float = 0.1,
    heading_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward facing the commanded heading direction.

    This encourages the robot's forward direction to rotate toward the command term's heading target,
    especially while moving.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        return torch.zeros(env.num_envs, device=env.device)

    forward_heading = asset.data.heading_w + heading_offset
    heading_error = math_utils.wrap_to_pi(command_term.heading_target - forward_heading)
    reward = torch.exp(-torch.square(heading_error) / std**2)

    cmd_lin_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= (cmd_lin_speed > lin_vel_threshold).float()
    if hasattr(command_term, "is_heading_env"):
        reward *= command_term.is_heading_env.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_velocity_heading_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
    heading_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for facing its actual planar velocity direction while moving."""
    asset: RigidObject = env.scene[asset_cfg.name]

    vel_xy_w = asset.data.root_lin_vel_w[:, :2]
    vel_speed = torch.linalg.norm(vel_xy_w, dim=1)
    vel_heading = torch.atan2(vel_xy_w[:, 1], vel_xy_w[:, 0])
    forward_heading = asset.data.heading_w + heading_offset
    heading_error = math_utils.wrap_to_pi(vel_heading - forward_heading)

    reward = torch.exp(-torch.square(heading_error) / std**2)
    reward *= (torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold).float()
    reward *= (vel_speed > velocity_threshold).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_planar_motion_l2_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.08,
    yaw_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar translation and yaw motion while the velocity command is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    motion += yaw_weight * torch.square(asset.data.root_ang_vel_b[:, 2])
    motion *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    motion *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return motion


def _straight_yaw_command_gate(
    env: ManagerBasedRLEnv, command_name: str, yaw_threshold: float, yaw_scale: float
) -> torch.Tensor:
    yaw_command = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    if yaw_scale <= yaw_threshold:
        return (yaw_command <= yaw_threshold).float()
    return 1.0 - torch.clamp((yaw_command - yaw_threshold) / (yaw_scale - yaw_threshold), min=0.0, max=1.0)


def joint_deviation_l1_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float,
    yaw_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation mainly for straight walking, while allowing yaw joints during turns."""
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def signed_joint_pair_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_pairs: list[list[str]],
) -> torch.Tensor:
    """Penalize mirrored joint pairs that should keep opposite signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    cache_key = tuple(tuple(joint_pair) for joint_pair in joint_pairs)
    if (
        not hasattr(env, "signed_joint_pair_cache")
        or env.signed_joint_pair_cache is None
        or not isinstance(env.signed_joint_pair_cache, dict)
    ):
        env.signed_joint_pair_cache = {}
    if cache_key not in env.signed_joint_pair_cache:
        env.signed_joint_pair_cache[cache_key] = [
            [asset.find_joints(joint_name)[0] for joint_name in joint_pair] for joint_pair in joint_pairs
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for left_ids, right_ids in env.signed_joint_pair_cache[cache_key]:
        signed_pair_error = asset.data.joint_pos[:, left_ids] + asset.data.joint_pos[:, right_ids]
        reward += torch.mean(torch.square(signed_pair_error), dim=1)
    reward *= 1 / len(joint_pairs) if len(joint_pairs) > 0 else 0.0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def signed_joint_pair_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float,
    yaw_scale: float,
    asset_cfg: SceneEntityCfg,
    joint_pairs: list[list[str]],
) -> torch.Tensor:
    """Penalize mirrored joint-pair error mainly during straight walking."""
    reward = signed_joint_pair_l2(env, asset_cfg=asset_cfg, joint_pairs=joint_pairs)
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Reward completed swing phases while penalizing rapid foot taps.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = torch.clamp(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], max=max_time)
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _biped_touchdown_progress_l2(
    landing_progress: torch.Tensor,
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    min_progress: float,
    min_air_time: float,
) -> torch.Tensor:
    """Penalize a landing foot that has not passed the stance foot far enough."""
    single_touchdown = torch.sum(first_contact.int(), dim=1) == 1
    valid_touchdown = torch.logical_and(first_contact, last_air_time >= min_air_time)
    valid_touchdown &= single_touchdown.unsqueeze(1)
    normalized_deficit = torch.clamp((min_progress - landing_progress) / min_progress, min=0.0)
    return torch.sum(torch.square(normalized_deficit) * valid_touchdown, dim=1)


def biped_touchdown_progress_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_progress: float,
    min_air_time: float,
    yaw_threshold: float,
    yaw_scale: float,
    command_threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Require each landing foot to pass the stance foot along the commanded direction.

    Timing-only gait rewards accept a policy that alternates contacts while one side takes a
    much larger step. This term checks spatial progress at touchdown and is symmetric in the
    explicitly ordered right/left feet.
    """
    if min_progress <= 0.0:
        raise ValueError("min_progress must be positive.")
    if min_air_time < 0.0:
        raise ValueError("min_air_time must be non-negative.")

    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(asset_cfg.body_ids) != 2 or len(sensor_cfg.body_ids) != 2:
        raise ValueError("biped_touchdown_progress_l2 expects exactly two ordered feet.")

    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    foot_pos_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1),
        foot_pos_w,
    )
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / command_speed.unsqueeze(1).clamp_min(command_threshold)
    right_minus_left = torch.sum((foot_pos_yaw[:, 0, :2] - foot_pos_yaw[:, 1, :2]) * command_direction, dim=1)
    landing_progress = torch.stack((right_minus_left, -right_minus_left), dim=1)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = _biped_touchdown_progress_l2(
        landing_progress,
        first_contact,
        last_air_time,
        min_progress,
        min_air_time,
    )
    reward *= command_speed > command_threshold
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _update_biped_touchdown_progress_balance(
    progress_ema: torch.Tensor,
    initialized: torch.Tensor,
    landing_progress: torch.Tensor,
    valid_touchdown: torch.Tensor,
    active: torch.Tensor,
    ema_alpha: float,
    progress_scale: float,
) -> torch.Tensor:
    """Update per-foot touchdown progress and return a bounded left/right imbalance cost."""
    progress_ema[~active] = 0.0
    initialized[~active] = False
    update = valid_touchdown & active.unsqueeze(1)
    first_update = update & ~initialized
    continuing_update = update & initialized
    progress_ema[first_update] = landing_progress[first_update]
    progress_ema[continuing_update] += ema_alpha * (
        landing_progress[continuing_update] - progress_ema[continuing_update]
    )
    initialized[update] = True
    comparable = active & torch.all(initialized, dim=1)
    normalized_difference = torch.clamp(
        (progress_ema[:, 0] - progress_ema[:, 1]) / progress_scale,
        min=-1.0,
        max=1.0,
    )
    return torch.square(normalized_difference) * comparable.float()


class BipedTouchdownProgressBalanceL2(ManagerTermBase):
    """Penalize persistent right/left step-length mismatch during straight translation."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        if len(asset_cfg.body_ids) != 2 or len(sensor_cfg.body_ids) != 2:
            raise ValueError("BipedTouchdownProgressBalanceL2 expects exactly two ordered feet.")
        if cfg.params["min_air_time"] < 0.0:
            raise ValueError("min_air_time must be non-negative.")
        if not 0.0 < cfg.params["ema_alpha"] <= 1.0:
            raise ValueError("ema_alpha must lie in (0, 1].")
        if cfg.params["progress_scale"] <= 0.0:
            raise ValueError("progress_scale must be positive.")
        self._progress_ema = torch.zeros(env.num_envs, 2, device=env.device)
        self._progress_initialized = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._progress_ema[env_ids] = 0.0
        self._progress_initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_air_time: float,
        ema_alpha: float,
        progress_scale: float,
        yaw_threshold: float,
        command_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
        foot_pos_yaw = quat_apply_inverse(
            yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1),
            foot_pos_w,
        )
        command = env.command_manager.get_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.vector_norm(command_xy, dim=1)
        command_direction = command_xy / command_speed.unsqueeze(1).clamp_min(command_threshold)
        right_minus_left = torch.sum(
            (foot_pos_yaw[:, 0, :2] - foot_pos_yaw[:, 1, :2]) * command_direction,
            dim=1,
        )
        landing_progress = torch.stack((right_minus_left, -right_minus_left), dim=1)

        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        single_touchdown = torch.sum(first_contact.int(), dim=1) == 1
        valid_touchdown = first_contact & (contact_sensor.data.last_air_time[:, sensor_cfg.body_ids] >= min_air_time)
        valid_touchdown &= single_touchdown.unsqueeze(1)
        active = (command_speed > command_threshold) & (torch.abs(command[:, 2]) <= yaw_threshold)
        reward = _update_biped_touchdown_progress_balance(
            self._progress_ema,
            self._progress_initialized,
            landing_progress,
            valid_touchdown,
            active,
            ema_alpha,
            progress_scale,
        )
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float,
    yaw_scale: float,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize left/right foot timing variance during commanded walking."""
    reward = feet_air_time_variance_penalty(env, sensor_cfg=sensor_cfg)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    return reward


def biped_gait_phase_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Penalize limp biped timing by matching each stance phase to the opposite swing phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = torch.clamp(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], max=max_time)
    air_time = torch.clamp(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], max=max_time)

    if contact_time.shape[1] != 2:
        raise ValueError("biped_gait_phase_l2 expects exactly two foot bodies.")

    left_right_phase_error = torch.square(contact_time[:, 0] - air_time[:, 1])
    right_left_phase_error = torch.square(contact_time[:, 1] - air_time[:, 0])
    reward = (left_right_phase_error + right_left_phase_error) / (max_time**2)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def biped_gait_phase_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    yaw_threshold: float,
    yaw_scale: float,
    command_threshold: float = 0.1,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Penalize limp biped timing strongly for straight commands and weakly during turns."""
    reward = biped_gait_phase_l2(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        command_threshold=command_threshold,
        max_time=max_time,
    )
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    return reward


def _bounded_biped_phase_duration_l2(
    contact_time: torch.Tensor,
    air_time: torch.Tensor,
    max_time: float,
) -> torch.Tensor:
    """Return a finite phase-duration penalty for two-foot contact timers."""
    contact_excess = torch.clamp(contact_time - max_time, min=0.0, max=max_time)
    air_excess = torch.clamp(air_time - max_time, min=0.0, max=max_time)
    return torch.sum(torch.square(contact_excess) + torch.square(air_excess), dim=1) / (max_time**2)


def biped_phase_duration_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    max_time: float = 0.4,
) -> torch.Tensor:
    """Penalize one foot staying in stance or swing for too long during biped walking."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    if contact_time.shape[1] != 2:
        raise ValueError("biped_phase_duration_l2 expects exactly two foot bodies.")

    # Contact timers continue accumulating while an environment has a standing command.  Without
    # an upper bound, the first walking step after a 10 s standing command can receive a raw penalty
    # above 1,000 and force an abrupt, consistently one-sided first step.  Keep this safeguard finite
    # while still penalizing phases that remain longer than ``max_time`` after walking begins.
    reward = _bounded_biped_phase_duration_l2(contact_time, air_time, max_time)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def biped_phase_duration_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    yaw_threshold: float,
    yaw_scale: float,
    command_threshold: float = 0.1,
    max_time: float = 0.4,
) -> torch.Tensor:
    """Limit excessively long stance/swing phases mainly for straight commands."""
    reward = biped_phase_duration_l2(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        command_threshold=command_threshold,
        max_time=max_time,
    )
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    return reward


def feet_flight_penalty(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Penalize phases where both feet are simultaneously off the ground during commanded motion."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    no_foot_contact = torch.sum(in_contact.int(), dim=1) == 0
    reward = no_foot_contact.float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stance_contact_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward sustained foot contact while the command is near zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    reward = torch.mean(in_contact.float(), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_min_lateral_distance_x_l2(
    env: ManagerBasedRLEnv,
    min_width: float,
    lateral_signs: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a narrow stance, optionally preserving the expected left/right foot order."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), foot_pos_w[:, i, :]
        )

    if lateral_signs is None:
        lateral_width = torch.max(foot_pos_b[:, :, 0], dim=1)[0] - torch.min(foot_pos_b[:, :, 0], dim=1)[0]
    else:
        if n_feet != len(lateral_signs):
            raise ValueError("lateral_signs must contain one sign for each configured foot.")
        signs = torch.tensor(lateral_signs, device=env.device, dtype=foot_pos_b.dtype)
        lateral_width = torch.sum(foot_pos_b[:, :, 0] * signs.unsqueeze(0), dim=1)
    width_deficit = torch.clamp(min_width - lateral_width, min=0.0) / min_width
    reward = torch.square(width_deficit)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_min_lateral_distance_x_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_width: float,
    yaw_threshold: float,
    yaw_scale: float,
    lateral_signs: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a narrow nominal stance while releasing the constraint during large turns."""
    reward = feet_min_lateral_distance_x_l2(
        env,
        min_width=min_width,
        lateral_signs=lateral_signs,
        asset_cfg=asset_cfg,
    )
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    return reward


def feet_max_lateral_distance_x_l2(
    env: ManagerBasedRLEnv, max_width: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize biped feet when their body-frame x-axis separation is wider than the maximum width."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), foot_pos_w[:, i, :]
        )

    lateral_width = torch.max(foot_pos_b[:, :, 0], dim=1)[0] - torch.min(foot_pos_b[:, :, 0], dim=1)[0]
    width_excess = torch.clamp(lateral_width - max_width, min=0.0) / max_width
    reward = torch.square(width_excess)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_heading_error_exp_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float,
    yaw_scale: float,
    std: float,
    parallel_std: float,
    parallel_scale: float,
    foot_forward_axis: tuple[float, float, float],
    body_forward_axis: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-relative foot yaw and opposing foot headings without blocking commanded turns."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    n_feet = foot_quat_w.shape[1]
    if n_feet != 2:
        raise ValueError("feet_heading_error_exp_straight_yaw_command expects exactly two feet.")

    foot_forward_local = torch.tensor(foot_forward_axis, device=env.device, dtype=foot_quat_w.dtype)
    foot_forward_local = foot_forward_local.view(1, 1, 3).expand(env.num_envs, n_feet, -1)
    foot_forward_w = math_utils.quat_apply(foot_quat_w.reshape(-1, 4), foot_forward_local.reshape(-1, 3)).view(
        env.num_envs, n_feet, 3
    )

    root_yaw_w = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, n_feet, -1)
    foot_forward_b = math_utils.quat_apply_inverse(root_yaw_w.reshape(-1, 4), foot_forward_w.reshape(-1, 3)).view(
        env.num_envs, n_feet, 3
    )

    foot_heading_b = foot_forward_b[..., :2]
    foot_heading_b = foot_heading_b / torch.linalg.vector_norm(foot_heading_b, dim=2, keepdim=True).clamp_min(1e-6)
    desired_heading_b = torch.tensor(body_forward_axis[:2], device=env.device, dtype=foot_quat_w.dtype)
    desired_heading_b = desired_heading_b / torch.linalg.vector_norm(desired_heading_b).clamp_min(1e-6)

    heading_dot = torch.sum(foot_heading_b * desired_heading_b.view(1, 1, 2), dim=2)
    heading_cross = desired_heading_b[0] * foot_heading_b[..., 1] - desired_heading_b[1] * foot_heading_b[..., 0]
    heading_error = torch.atan2(heading_cross, heading_dot)
    alignment_error = torch.mean(1.0 - torch.exp(-torch.square(heading_error) / std**2), dim=1)
    alignment_error *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)

    heading_difference = heading_error[:, 0] - heading_error[:, 1]
    heading_difference = torch.atan2(torch.sin(heading_difference), torch.cos(heading_difference))
    parallel_error = 1.0 - torch.exp(-torch.square(heading_difference) / parallel_std**2)

    reward = alignment_error + parallel_scale * parallel_error
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_lateral_position_x_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_width: float,
    yaw_threshold: float,
    yaw_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asymmetric foot lateral placement during straight walking without blocking commanded turns."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), foot_pos_w[:, i, :]
        )

    if n_feet == 2:
        # STEP URDF places the right foot on negative body-frame x and the left foot on positive x.
        desired_x = torch.tensor([-0.5, 0.5], device=env.device).unsqueeze(0) * stance_width
    else:
        desired_x = torch.linspace(-0.5, 0.5, n_feet, device=env.device).unsqueeze(0) * stance_width
    reward = torch.sum(torch.square(foot_pos_b[:, :, 0] - desired_x), dim=1) / (stance_width**2)
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_lateral_center_x_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_width: float,
    yaw_threshold: float,
    yaw_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize both biped feet drifting to the same lateral side of the base."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), foot_pos_w[:, i, :]
        )

    if n_feet != 2:
        raise ValueError("feet_lateral_center_x_l2_straight_yaw_command expects exactly two feet.")

    center_x = torch.mean(foot_pos_b[:, :, 0], dim=1) / stance_width
    reward = torch.square(center_x)
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_forward_position_y_l2_straight_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_length: float,
    yaw_threshold: float,
    yaw_scale: float,
    target_center_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize fore-aft foot-center error relative to the nominal STEP stance."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), foot_pos_w[:, i, :]
        )

    if n_feet != 2:
        raise ValueError("feet_forward_position_y_l2_straight_yaw_command expects exactly two feet.")

    fore_aft_bias = torch.sum(foot_pos_b[:, :, 1] - target_center_y, dim=1) / stance_length
    reward = torch.square(fore_aft_bias)
    reward *= _straight_yaw_command_gate(env, command_name, yaw_threshold, yaw_scale)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _update_biped_fore_aft_bias_ema(
    ema: torch.Tensor,
    normalized_difference: torch.Tensor,
    active: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Update the gait-cycle bias filter and return a new penalty tensor."""
    ema[~active] = 0.0
    ema[active] += alpha * (normalized_difference[active] - ema[active])
    return torch.square(ema)


class BipedFeetForeAftBalanceL2(ManagerTermBase):
    """Penalize persistent one-foot-leading bias without suppressing alternating strides.

    The instantaneous right-minus-left fore-aft distance is low-pass filtered over multiple
    steps.  A balanced alternating gait cancels in the filter, whereas keeping the same foot
    ahead converges to a non-zero penalty.  The filter is cleared outside straight translation
    so standing and turning cannot leak stale state into the next walking segment.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if len(asset_cfg.body_ids) != 2:
            raise ValueError("BipedFeetForeAftBalanceL2 expects exactly two ordered foot bodies.")
        if cfg.params["stance_length"] <= 0.0:
            raise ValueError("stance_length must be positive.")
        if cfg.params["time_constant"] <= 0.0:
            raise ValueError("time_constant must be positive.")
        self._fore_aft_bias_ema = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._fore_aft_bias_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        stance_length: float,
        time_constant: float,
        yaw_threshold: float,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w.unsqueeze(1)
        foot_pos_yaw = quat_apply_inverse(
            yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1),
            foot_pos_w,
        )
        normalized_difference = torch.clamp(
            (foot_pos_yaw[:, 0, 1] - foot_pos_yaw[:, 1, 1]) / stance_length,
            min=-1.0,
            max=1.0,
        )

        command = env.command_manager.get_command(command_name)
        active = torch.logical_and(
            torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold,
            torch.abs(command[:, 2]) <= yaw_threshold,
        )
        alpha = -math.expm1(-float(env.step_dt) / time_constant)
        reward = _update_biped_fore_aft_bias_ema(
            self._fore_aft_bias_ema,
            normalized_difference,
            active,
            alpha,
        )
        reward *= active.float()
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lateral_lin_vel_x_yaw_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize lateral body sway using yaw-aligned x velocity.

    For STEP, the walking direction is along the body -y axis, so the yaw-aligned x axis is the lateral sway axis.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    reward = torch.square(vel_yaw[:, 0])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lateral_tilt_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize persistent side lean along STEP's lateral body x-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.projected_gravity_b[:, 0])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lateral_tilt_x_l2_with_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral lean while STEP is commanded to translate."""
    reward = lateral_tilt_x_l2(env, asset_cfg=asset_cfg)
    planar_command = env.command_manager.get_command(command_name)[:, :2]
    reward *= torch.linalg.norm(planar_command, dim=1) > command_threshold
    return reward


def _update_lateral_tilt_bias_ema(
    ema: torch.Tensor,
    lateral_tilt: torch.Tensor,
    active: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Low-pass lateral tilt so alternating sway cancels while a sustained lean remains."""
    ema[~active] = 0.0
    ema[active] += alpha * (lateral_tilt[active] - ema[active])
    return torch.square(ema)


class PersistentLateralTiltBiasL2(ManagerTermBase):
    """Penalize only the low-frequency lateral lean component during straight walking."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.params["time_constant"] <= 0.0:
            raise ValueError("time_constant must be positive.")
        self._lateral_tilt_ema = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lateral_tilt_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        time_constant: float,
        yaw_threshold: float,
        command_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        active = (torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold) & (
            torch.abs(command[:, 2]) <= yaw_threshold
        )
        alpha = -math.expm1(-float(env.step_dt) / time_constant)
        reward = _update_lateral_tilt_bias_ema(
            self._lateral_tilt_ema,
            torch.clamp(asset.data.projected_gravity_b[:, 0], min=-0.35, max=0.35),
            active,
            alpha,
        )
        reward *= active.float()
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


def _update_planar_tilt_bias_ema(
    ema: torch.Tensor,
    planar_tilt: torch.Tensor,
    active: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Low-pass roll/pitch tilt together so a policy cannot exchange one axis for the other."""
    ema[~active] = 0.0
    ema[active] += alpha * (planar_tilt[active] - ema[active])
    return torch.sum(torch.square(ema), dim=1)


class PersistentPlanarTiltBiasL2(ManagerTermBase):
    """Penalize low-frequency roll and pitch bias during straight translation.

    The temporal filter leaves brief acceleration lean and disturbance recovery largely
    untouched. The straight-command gate releases the term for deliberate turning.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.params["time_constant"] <= 0.0:
            raise ValueError("time_constant must be positive.")
        self._planar_tilt_ema = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._planar_tilt_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        time_constant: float,
        yaw_threshold: float,
        command_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        active = (torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold) & (
            torch.abs(command[:, 2]) <= yaw_threshold
        )
        alpha = -math.expm1(-float(env.step_dt) / time_constant)
        reward = _update_planar_tilt_bias_ema(
            self._planar_tilt_ema,
            torch.clamp(asset.data.projected_gravity_b[:, :2], min=-0.35, max=0.35),
            active,
            alpha,
        )
        reward *= active.float()
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def body_flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize roll/pitch tilt of selected bodies relative to gravity.

    This uses the body link orientation in world-frame and ignores yaw by only penalizing the xy-components
    of the gravity vector expressed in the body frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, body_quat_w.shape[1], -1)
    gravity_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).view(
        env.num_envs, body_quat_w.shape[1], 3
    )
    reward = torch.mean(torch.sum(torch.square(gravity_b[..., :2]), dim=2), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def body_flat_orientation_l2_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected body roll/pitch strongly when the command is near zero."""
    reward = body_flat_orientation_l2(env, asset_cfg=asset_cfg)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return reward
