"""Opt-in RND STEP actuator task with simulated CMP10A policy observations."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.utils import configclass

from .flat_actuator_env_cfg import RndStepFlatActuatorEnvCfg


@configclass
class RndStepFlatActuatorImuEnvCfg(RndStepFlatActuatorEnvCfg):
    """Apply the CMP10A model only to the actor's IMU-derived observation slice."""

    def __post_init__(self):
        super().__post_init__()
        model = mdp.load_rnd_cmp10a_observation_model(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH)

        self.observations.policy.base_ang_vel.func = mdp.RndCmp10aObservation
        self.observations.policy.base_ang_vel.params = {
            "channel": "gyro",
            "model_path": str(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH),
            "sample_randomization": True,
        }
        self.observations.policy.base_ang_vel.noise = None
        self.observations.policy.base_ang_vel.scale = model.policy_angular_velocity_scale

        self.observations.policy.projected_gravity.func = mdp.RndCmp10aObservation
        self.observations.policy.projected_gravity.params = {
            "channel": "gravity",
            "model_path": str(mdp.RND_CMP10A_OBSERVATION_MODEL_PATH),
            "sample_randomization": True,
        }
        self.observations.policy.projected_gravity.noise = None
        self.observations.policy.projected_gravity.scale = 1.0
