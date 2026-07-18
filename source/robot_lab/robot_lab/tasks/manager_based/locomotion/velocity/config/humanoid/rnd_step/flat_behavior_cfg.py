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
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 8.0
    env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
    env_cfg.rewards.lin_vel_xy_underspeed_l2 = RewTerm(
        func=mdp.lin_vel_xy_underspeed_l2,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.rewards.track_ang_vel_z_exp.weight = 5.5
    env_cfg.rewards.track_ang_vel_z_exp.params["std"] = 0.35
    env_cfg.rewards.track_heading_command_exp.weight = 0.5
    env_cfg.rewards.lin_vel_z_l2.weight = -0.2
    env_cfg.rewards.lateral_lin_vel_x_yaw_l2 = RewTerm(func=mdp.lateral_lin_vel_x_yaw_l2, weight=-1.0)
    env_cfg.rewards.lateral_tilt_x_l2 = RewTerm(func=mdp.lateral_tilt_x_l2, weight=-6.0)
    env_cfg.rewards.lateral_tilt_x_with_cmd_l2 = RewTerm(
        func=mdp.lateral_tilt_x_l2_with_cmd,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.rewards.persistent_planar_tilt_bias_l2 = RewTerm(
        func=mdp.PersistentPlanarTiltBiasL2,
        weight=-15.0,
        params={
            "command_name": "base_velocity",
            "time_constant": 1.5,
            "yaw_threshold": 0.15,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.rewards.base_height_l2.weight = -4.5
    env_cfg.rewards.base_height_l2.params["target_height"] = STEP_BASE_HEIGHT_TARGET

    env_cfg.rewards.body_lin_acc_l2.weight = -0.05
    env_cfg.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [env_cfg.base_link_name]
    env_cfg.rewards.upper_body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=env_cfg.base_link_name)},
    )
    env_cfg.rewards.upper_body_flat_orientation_l2 = RewTerm(
        func=mdp.body_flat_orientation_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=env_cfg.base_link_name)},
    )
    env_cfg.rewards.upper_body_flat_orientation_without_cmd_l2 = RewTerm(
        func=mdp.body_flat_orientation_l2_without_cmd,
        weight=-8.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=env_cfg.base_link_name),
        },
    )

    env_cfg.rewards.action_rate_l2.weight = -0.02
    env_cfg.rewards.stand_still.weight = -0.5
    env_cfg.rewards.stand_still.params["command_threshold"] = 0.08
    env_cfg.rewards.base_planar_motion_without_cmd_l2 = RewTerm(
        func=mdp.base_planar_motion_l2_without_cmd,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_weight": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.rewards.joint_pos_penalty.weight = -0.25
    env_cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 4.0
    env_cfg.rewards.joint_pos_penalty.params["command_threshold"] = 0.08
    env_cfg.rewards.joint_deviation_hip_l1.weight = -0.03
    env_cfg.rewards.joint_deviation_hip_yaw_l1 = RewTerm(
        func=mdp.joint_deviation_l1_straight_yaw_command,
        weight=-0.30,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.10,
            "yaw_scale": 0.45,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*"]),
        },
    )
    env_cfg.rewards.lateral_roll_joint_symmetry_l2 = RewTerm(
        func=mdp.signed_joint_pair_l2_straight_yaw_command,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.15,
            "yaw_scale": 0.75,
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pairs": [
                ["R_Leg_hip_yaw", "L_Leg_hip_yaw"],
                ["R_Leg_hip_roll", "L_Leg_hip_roll"],
                ["R_Leg_ankle_roll", "L_Leg_ankle_roll"],
            ],
        },
    )

    env_cfg.rewards.upward.weight = 1.5
    env_cfg.rewards.undesired_contacts.weight = -2.0
    env_cfg.rewards.joint_acc_l2.weight = -4.0e-7
    env_cfg.rewards.joint_torques_l2.weight = -1.0e-5
    env_cfg.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*hip.*", ".*knee.*"]
    env_cfg.rewards.joint_pos_limits.weight = -0.2
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{env_cfg.foot_link_name}).*"]

    # Reward completed swing phases instead of every single-support frame.  The previous dense
    # reward combined with the phase terms made a hard contact swap every 0.4 s locally optimal.
    env_cfg.rewards.feet_air_time.weight = 3.0
    env_cfg.rewards.feet_air_time.func = mdp.feet_air_time
    env_cfg.rewards.feet_air_time.params["threshold"] = 0.20
    env_cfg.rewards.feet_air_time.params["max_time"] = 0.50
    env_cfg.rewards.feet_air_time_variance.weight = -6.0
    env_cfg.rewards.feet_air_time_variance.func = mdp.feet_air_time_variance_penalty_straight_yaw_command
    env_cfg.rewards.feet_air_time_variance.params["command_name"] = "base_velocity"
    env_cfg.rewards.feet_air_time_variance.params["yaw_threshold"] = 0.15
    env_cfg.rewards.feet_air_time_variance.params["yaw_scale"] = 0.75
    env_cfg.rewards.feet_air_time_variance.params["command_threshold"] = 0.1
    env_cfg.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    env_cfg.rewards.biped_gait_phase_l2 = RewTerm(
        func=mdp.biped_gait_phase_l2_straight_yaw_command,
        # Retain an alternating-gait preference without prohibiting a brief, stable double support.
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.15,
            "yaw_scale": 0.75,
            "command_threshold": 0.1,
            "max_time": 0.40,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.biped_phase_duration_l2 = RewTerm(
        func=mdp.biped_phase_duration_l2_straight_yaw_command,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.15,
            "yaw_scale": 0.75,
            "command_threshold": 0.1,
            "max_time": 0.40,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.biped_touchdown_progress_l2 = RewTerm(
        func=mdp.biped_touchdown_progress_l2_straight_yaw_command,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "min_progress": 0.04,
            "min_air_time": 0.06,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.75,
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["R_Leg_foot", "L_Leg_foot"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["R_Leg_foot", "L_Leg_foot"],
                preserve_order=True,
            ),
        },
    )
    env_cfg.rewards.biped_touchdown_progress_balance_l2 = RewTerm(
        func=mdp.BipedTouchdownProgressBalanceL2,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "min_air_time": 0.15,
            "ema_alpha": 0.25,
            "progress_scale": 0.04,
            "yaw_threshold": 0.15,
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["R_Leg_foot", "L_Leg_foot"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["R_Leg_foot", "L_Leg_foot"],
                preserve_order=True,
            ),
        },
    )
    env_cfg.rewards.feet_flight_penalty.weight = -0.5
    env_cfg.rewards.feet_flight_penalty.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    env_cfg.rewards.feet_stance_contact_without_cmd = RewTerm(
        func=mdp.feet_stance_contact_without_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[env_cfg.foot_link_name]),
        },
    )
    env_cfg.rewards.feet_slide.weight = -0.15
    env_cfg.rewards.feet_height.weight = 0.0
    env_cfg.rewards.feet_min_lateral_distance_x_l2 = RewTerm(
        func=mdp.feet_min_lateral_distance_x_l2_straight_yaw_command,
        weight=-6.0,
        params={
            "command_name": "base_velocity",
            "min_width": 0.16,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.65,
            # Body-frame x is negative on STEP's right side and positive on its left side.
            "lateral_signs": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"], preserve_order=True),
        },
    )
    env_cfg.rewards.feet_hard_min_lateral_distance_x_l2 = RewTerm(
        func=mdp.feet_min_lateral_distance_x_l2,
        weight=-8.0,
        params={
            "min_width": 0.11,
            "lateral_signs": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"], preserve_order=True),
        },
    )
    env_cfg.rewards.feet_max_lateral_distance_x_l2 = RewTerm(
        func=mdp.feet_max_lateral_distance_x_l2,
        weight=-1.5,
        params={
            "max_width": 0.23,
            "asset_cfg": SceneEntityCfg("robot", body_names=[env_cfg.foot_link_name]),
        },
    )
    env_cfg.rewards.feet_heading_error_exp = RewTerm(
        func=mdp.feet_heading_error_exp_straight_yaw_command,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "yaw_threshold": 0.10,
            "yaw_scale": 0.45,
            "std": 0.25,
            "parallel_std": 0.30,
            "parallel_scale": 0.5,
            # STEP foot local -Z points toward the robot's physical body -Y forward direction.
            "foot_forward_axis": (0.0, 0.0, -1.0),
            "body_forward_axis": (0.0, -1.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_lateral_position_x_l2 = RewTerm(
        func=mdp.feet_lateral_position_x_l2_straight_yaw_command,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            # FK of STEP_DEFAULT_JOINT_POS gives about 0.193 m between foot link origins.
            "stance_width": 0.19,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.55,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"], preserve_order=True),
        },
    )
    env_cfg.rewards.feet_lateral_center_x_l2 = RewTerm(
        func=mdp.feet_lateral_center_x_l2_straight_yaw_command,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "stance_width": 0.19,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.55,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_forward_position_y_l2 = RewTerm(
        func=mdp.feet_forward_position_y_l2_straight_yaw_command,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "stance_length": 0.18,
            # Default-pose FK places both foot-link origins at body-frame y ~= +0.0516 m.
            # Target that nominal support geometry instead of pulling both feet 5 cm forward.
            "target_center_y": 0.052,
            "yaw_threshold": 0.15,
            "yaw_scale": 0.45,
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_Leg_foot", "L_Leg_foot"]),
        },
    )
    env_cfg.rewards.feet_fore_aft_balance_l2 = RewTerm(
        func=mdp.BipedFeetForeAftBalanceL2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "stance_length": 0.18,
            "time_constant": 1.5,
            "yaw_threshold": 0.15,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["R_Leg_foot", "L_Leg_foot"],
                preserve_order=True,
            ),
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
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.65, -0.05)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    env_cfg.commands.base_velocity.zero_velocity_threshold = 0.05
    env_cfg.commands.base_velocity.rel_standing_envs = 0.25
    env_cfg.commands.base_velocity.rel_pure_yaw_envs = 0.25
    env_cfg.commands.base_velocity.rel_straight_envs = 0.35
    # Preserve most of the original i.i.d. command distribution while dedicating a bounded
    # subset to realistic start/stop and straight/turn transitions. All targets remain inside
    # the same configured velocity ranges and pass through the existing rate limiter.
    env_cfg.commands.base_velocity.transition_sequence_probabilities = (0.15, 0.15)
    env_cfg.commands.base_velocity.stand_start_stop_times_s = (2.0, 8.0)
    env_cfg.commands.base_velocity.straight_turn_times_s = (2.0, 6.0, 10.0, 14.0)
    env_cfg.commands.base_velocity.transition_min_abs_yaw = 0.35
    env_cfg.commands.base_velocity.rel_heading_envs = 0.4
    env_cfg.commands.base_velocity.heading_command = True
    env_cfg.commands.base_velocity.command_ramp_rates = (0.4, 0.8, 1.2)
