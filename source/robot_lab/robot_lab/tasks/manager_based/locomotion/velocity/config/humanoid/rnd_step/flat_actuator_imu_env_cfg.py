"""Opt-in RND STEP actuator task with simulated CMP10A policy observations."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_actuator_env_cfg import RndStepFlatActuatorEnvCfg


@configclass
class RndStepFlatActuatorImuEnvCfg(RndStepFlatActuatorEnvCfg):
    """Apply the CMP10A model only to the actor's IMU-derived observation slice."""

    def __post_init__(self):
        super().__post_init__()
        imu_model = mdp.load_rnd_cmp10a_observation_model(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH)

        self.observations.policy.base_ang_vel.func = mdp.RndCmp10aObservation
        self.observations.policy.base_ang_vel.params = {
            "channel": "gyro",
            "model_path": str(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH),
            "sample_randomization": True,
            "body_name": "imu",
        }
        self.observations.policy.base_ang_vel.noise = None
        self.observations.policy.base_ang_vel.scale = imu_model.policy_angular_velocity_scale

        self.observations.policy.projected_gravity.func = mdp.RndCmp10aObservation
        self.observations.policy.projected_gravity.params = {
            "channel": "gravity",
            "model_path": str(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH),
            "sample_randomization": True,
            "body_name": "imu",
        }
        self.observations.policy.projected_gravity.noise = None
        self.observations.policy.projected_gravity.scale = 1.0

        encoder_model = mdp.load_rnd_dynamixel_encoder_observation_model(
            mdp.RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH
        )
        self.observations.policy.joint_pos.func = mdp.RndDynamixelEncoderObservation
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(encoder_model.joint_order),
                preserve_order=True,
            ),
            "model_path": str(mdp.RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH),
            "sample_randomization": True,
        }
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_pos.scale = 1.0
        # Position and velocity now arrive as one coherent 24-D encoder sample.
        self.observations.policy.joint_vel = None

        # Isaac Lab's CircularBuffer fills every history slot with the first post-reset value.
        # This gives a current-value prefill instead of an artificial all-zero transient.
        for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "actions"):
            term = getattr(self.observations.policy, term_name)
            term.history_length = 4
            term.flatten_history_dim = True
        self.observations.policy.velocity_commands.history_length = 0

        # Keep transition coverage without letting long standing phases dominate the larger
        # history actor's first training run.
        self.commands.base_velocity.transition_sequence_probabilities = (0.05, 0.05)

        # The 171-D history actor can otherwise amplify its own previous Gaussian samples.
        # Keep the useful +/-1.5 action range free and penalize only targets that produced the
        # measured action-growth failure in the 2026-07-17 11:54 run.
        self.rewards.action_excess_l2 = RewTerm(
            func=mdp.action_excess_l2,
            weight=-0.10,
            params={"threshold": 1.5},
        )

        # Penalize excessive landing speed and 20-60 ms tapping at 200 Hz. The lower
        # bound excludes sub-policy-step contact chatter, and the command gate leaves
        # the stationary stance objective untouched.
        self.rewards.vertical_touchdown_impact = RewTerm(
            func=mdp.PhysicsTouchdownImpactCost,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                "command_xy_threshold": 0.10,
                "command_yaw_threshold": 0.15,
                "impact_speed_offset": 0.25,
                "impact_speed_range": 0.50,
                "foot_body_names": ["R_Leg_foot", "L_Leg_foot"],
                "asset_name": "robot",
                "sensor_name": "contact_forces",
                "min_air_time": 0.06,
                "short_air_time_floor": 0.02,
                "short_air_time_penalty_scale": 3.0,
            },
        )
