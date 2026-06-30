# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets.rnd import STEP_BASE_HEIGHT_TARGET


def dampen_arm_actions(action_scale: dict) -> dict:
    tuned_scale = dict(action_scale)
    for joint_name, scale in tuned_scale.items():
        if any(token in joint_name for token in ("Arm", "arm_hand", "Neck")):
            tuned_scale[joint_name] = 0.2 * scale
    return tuned_scale


def apply_step_flat_stable_walk_rewards(env_cfg) -> None:
    """Apply reward terms for stable STEP flat walking."""

    env_cfg.rewards.is_terminated.weight = -500.0
    env_cfg.rewards.flat_orientation_l2.weight = -7.0
    env_cfg.rewards.ang_vel_xy_l2.weight = -1.0
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 7.0
    env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
    env_cfg.rewards.track_ang_vel_z_exp.weight = 2.0
    env_cfg.rewards.track_heading_command_exp.weight = 1.0
    env_cfg.rewards.lin_vel_z_l2.weight = -0.2
    env_cfg.rewards.lateral_lin_vel_x_yaw_l2 = RewTerm(func=mdp.lateral_lin_vel_x_yaw_l2, weight=-0.6)
    env_cfg.rewards.lateral_tilt_x_l2 = RewTerm(func=mdp.lateral_tilt_x_l2, weight=-5.0)
    env_cfg.rewards.base_height_l2.weight = -3.0
    env_cfg.rewards.base_height_l2.params["target_height"] = STEP_BASE_HEIGHT_TARGET

    env_cfg.rewards.body_lin_acc_l2.weight = -0.03
    env_cfg.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]
    env_cfg.rewards.upper_body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=env_cfg.base_link_name)},
    )
    env_cfg.rewards.upper_body_flat_orientation_l2 = RewTerm(
        func=mdp.body_flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=env_cfg.base_link_name)},
    )

    env_cfg.rewards.action_rate_l2.weight = -0.01
    env_cfg.rewards.stand_still.weight = -0.2
    env_cfg.rewards.joint_pos_penalty.weight = -0.25
    env_cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 3.0
    env_cfg.rewards.joint_deviation_hip_l1.weight = -0.03
    env_cfg.rewards.joint_deviation_hip_yaw_l1 = RewTerm(
        func=mdp.joint_deviation_l1_straight_yaw_command,
        weight=-0.12,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.15,
            "yaw_scale": 0.65,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*"]),
        },
    )
    env_cfg.rewards.lateral_roll_joint_symmetry_l2 = RewTerm(
        func=mdp.signed_joint_pair_l2,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pairs": [
                ["R_Leg_hip_roll", "L_Leg_hip_roll"],
                ["R_Leg_ankle_roll", "L_Leg_ankle_roll"],
            ],
        },
    )

    env_cfg.rewards.upward.weight = 1.5
    env_cfg.rewards.undesired_contacts.weight = -2.0
    env_cfg.rewards.joint_acc_l2.weight = -2.0e-7
    env_cfg.rewards.joint_torques_l2.weight = -1.0e-5
    env_cfg.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*hip.*", ".*knee.*"]
    env_cfg.rewards.joint_pos_limits.weight = -0.2
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{env_cfg.foot_link_name}).*"]

    env_cfg.rewards.feet_air_time.weight = 0.25
    env_cfg.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
    env_cfg.rewards.feet_air_time.params["threshold"] = 0.25
    env_cfg.rewards.feet_air_time_variance.weight = -4.5
    env_cfg.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    env_cfg.rewards.biped_gait_phase_l2 = RewTerm(
        func=mdp.biped_gait_phase_l2,
        weight=-3.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "max_time": 0.40,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.biped_phase_duration_l2 = RewTerm(
        func=mdp.biped_phase_duration_l2,
        weight=-1.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "max_time": 0.40,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_flight_penalty.weight = -0.1
    env_cfg.rewards.feet_flight_penalty.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    env_cfg.rewards.feet_slide.weight = -0.15
    env_cfg.rewards.feet_height.weight = 0.0
    env_cfg.rewards.feet_min_lateral_distance_x_l2 = RewTerm(
        func=mdp.feet_min_lateral_distance_x_l2,
        weight=-4.0,
        params={
            # Foot mesh lateral width is about 0.095 m, so 0.185 m leaves roughly 9 cm of inner clearance.
            "min_width": 0.185,
            "asset_cfg": SceneEntityCfg("robot", body_names=[env_cfg.foot_link_name]),
        },
    )
    env_cfg.rewards.feet_lateral_position_x_l2 = RewTerm(
        func=mdp.feet_lateral_position_x_l2_straight_yaw_command,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "stance_width": 0.22,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.65,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_lateral_center_x_l2 = RewTerm(
        func=mdp.feet_lateral_center_x_l2_straight_yaw_command,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "stance_width": 0.22,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.65,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_forward_position_y_l2 = RewTerm(
        func=mdp.feet_forward_position_y_l2_straight_yaw_command,
        weight=-0.35,
        params={
            "command_name": "base_velocity",
            "stance_length": 0.18,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.65,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.joint_mirror.weight = 0.0
    env_cfg.rewards.action_mirror.weight = 0.0


def apply_step_flat_terminations(env_cfg) -> None:
    env_cfg.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        env_cfg.base_link_name,
    ]


def apply_step_flat_commands(env_cfg) -> None:
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.8, -0.2)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
    env_cfg.commands.base_velocity.zero_velocity_threshold = 0.05
    env_cfg.commands.base_velocity.rel_standing_envs = 0.1
    env_cfg.commands.base_velocity.rel_heading_envs = 1.0
    env_cfg.commands.base_velocity.heading_command = True
