# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for in-house RND robots."""

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


STEP_INIT_POS = (0.0, 0.0, 0.44)
STEP_BASE_HEIGHT_TARGET = 0.45
STEP_DEFAULT_JOINT_POS = {
    "R_Leg_hip_yaw": 0.0,
    "R_Leg_hip_roll": 0.10,
    "R_Leg_hip_pitch": -0.63,
    "R_Leg_knee": 0.15,
    "R_Leg_ankle_pitch": -0.91,
    "R_Leg_ankle_roll": -0.10,
    "L_Leg_hip_yaw": 0.0,
    "L_Leg_hip_roll": -0.10,
    "L_Leg_hip_pitch": 0.63,
    "L_Leg_knee": -0.15,
    "L_Leg_ankle_pitch": 0.91,
    "L_Leg_ankle_roll": 0.10,
}


STEP_4BAR_INIT_POS = (0.0, 0.0, 0.46)
STEP_4BAR_BASE_HEIGHT_TARGET = 0.46
STEP_4BAR_DEFAULT_JOINT_POS = {
    "R_Leg_hip_yaw": 0.0,
    "R_Leg_hip_roll": 0.10,
    "R_Leg_hip_pitch": -0.24,
    "R_Leg_knee": 0.67,
    "R_Leg_ankle_pitch": -0.41,
    "L_Leg_hip_yaw": 0.0,
    "L_Leg_hip_roll": -0.10,
    "L_Leg_hip_pitch": 0.24,
    "L_Leg_knee": -0.67,
    "L_Leg_ankle_pitch": 0.41,
    "Waist": 0.0,
    "R_Arm_roll": 0.0,
    "R_Arm_pitch": 1.20,
    "R_Arm_elbow": 1.56,
    "L_Arm_roll": 0.0,
    "L_Arm_pitch": -1.20,
    "L_Arm_elbow": -1.56,
}


STEP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/rnd/step/urdf/step.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=STEP_INIT_POS,
        joint_pos=dict(STEP_DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Leg_hip_yaw",
                "R_Leg_hip_roll",
                "R_Leg_hip_pitch",
                "R_Leg_knee",
                "L_Leg_hip_yaw",
                "L_Leg_hip_roll",
                "L_Leg_hip_pitch",
                "L_Leg_knee",
            ],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=24.0,
            damping=1.8,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Leg_ankle_pitch",
                "R_Leg_ankle_roll",
                "L_Leg_ankle_pitch",
                "L_Leg_ankle_roll",
            ],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=21.0,
            damping=1.8,
            armature=0.005,
        ),
    },
)


STEP_ACTION_SCALE = {
    "R_Leg_hip_yaw": 0.20,
    "R_Leg_hip_roll": 0.20,
    "R_Leg_hip_pitch": 0.18,
    "R_Leg_knee": 0.18,
    "R_Leg_ankle_pitch": 0.24,
    "R_Leg_ankle_roll": 0.18,
    "L_Leg_hip_yaw": 0.20,
    "L_Leg_hip_roll": 0.20,
    "L_Leg_hip_pitch": 0.18,
    "L_Leg_knee": 0.18,
    "L_Leg_ankle_pitch": 0.24,
    "L_Leg_ankle_roll": 0.18,
}


STEP_4BAR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/rnd/step_4bar/urdf/STEP_4Bar_RL_fixed.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=STEP_4BAR_INIT_POS,
        joint_pos=dict(STEP_4BAR_DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Leg_hip_yaw",
                "R_Leg_hip_roll",
                "R_Leg_hip_pitch",
                "R_Leg_knee",
                "L_Leg_hip_yaw",
                "L_Leg_hip_roll",
                "L_Leg_hip_pitch",
                "L_Leg_knee",
            ],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=24.0,
            damping=1.8,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Leg_ankle_pitch",
                "L_Leg_ankle_pitch",
            ],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=21.0,
            damping=1.8,
            armature=0.005,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=8.0,
            damping=0.8,
            armature=0.003,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_Arm_roll",
                "R_Arm_pitch",
                "R_Arm_elbow",
                "L_Arm_roll",
                "L_Arm_pitch",
                "L_Arm_elbow",
            ],
            effort_limit_sim=5.4,
            velocity_limit_sim=4.18,
            stiffness=8.0,
            damping=0.8,
            armature=0.003,
        ),
    },
)


STEP_4BAR_ACTION_SCALE = {
    "R_Leg_hip_yaw": 0.20,
    "R_Leg_hip_roll": 0.20,
    "R_Leg_hip_pitch": 0.18,
    "R_Leg_knee": 0.18,
    "R_Leg_ankle_pitch": 0.24,
    "L_Leg_hip_yaw": 0.20,
    "L_Leg_hip_roll": 0.20,
    "L_Leg_hip_pitch": 0.18,
    "L_Leg_knee": 0.18,
    "L_Leg_ankle_pitch": 0.24,
    "Waist": 0.12,
    "R_Arm_roll": 0.0,
    "R_Arm_pitch": 0.0,
    "R_Arm_elbow": 0.0,
    "L_Arm_roll": 0.0,
    "L_Arm_pitch": 0.0,
    "L_Arm_elbow": 0.0,
}
