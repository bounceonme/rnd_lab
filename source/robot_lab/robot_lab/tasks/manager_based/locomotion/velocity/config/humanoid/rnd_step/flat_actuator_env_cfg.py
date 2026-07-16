"""Flat RND STEP task with actuator physics and balance-specific safeguards."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.actuators.rnd_stateful import load_rnd_actuator_model
from robot_lab.assets.rnd_actuator import (
    RND_ACTUATOR_RUNTIME_MODEL_PATH,
    RND_ARMATURE_RANDOMIZATION_PATH,
    STEP_ACTUATOR_CFG,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from .flat_env_cfg import RndStepFlatEnvCfg


_RND_LEG_JOINT_NAMES = list(load_rnd_actuator_model(RND_ACTUATOR_RUNTIME_MODEL_PATH)["joint_order"])


@configclass
class RndStepFlatActuatorEventCfg(EventCfg):
    """Add fixed-per-environment armature uncertainty to the actuator task."""

    randomize_joint_armature = EventTerm(
        func=mdp.randomize_rnd_joint_armature,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_RND_LEG_JOINT_NAMES, preserve_order=True),
            "model_path": str(RND_ARMATURE_RANDOMIZATION_PATH),
            "sample_randomization": True,
            "seed_offset": 2_000_003,
        },
    )


@configclass
class RndStepFlatActuatorEnvCfg(RndStepFlatEnvCfg):
    """Add the measured actuator path while protecting straight-foot alignment."""

    events: RndStepFlatActuatorEventCfg = RndStepFlatActuatorEventCfg()

    def __post_init__(self):
        super().__post_init__()
        model = load_rnd_actuator_model(RND_ACTUATOR_RUNTIME_MODEL_PATH)
        physics_hz = 1.0 / float(self.sim.dt)
        policy_hz = physics_hz / int(self.decimation)
        if abs(physics_hz - float(model["physics_hz"])) > 1.0e-6:
            raise ValueError(
                f"Flat actuator task physics_hz={physics_hz} does not match model physics_hz={model['physics_hz']}."
            )
        if abs(policy_hz - float(model["policy_hz"])) > 1.0e-6:
            raise ValueError(
                f"Flat actuator task policy_hz={policy_hz} does not match model policy_hz={model['policy_hz']}."
            )
        self.scene.robot = STEP_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The completed 10k run traded foot heading and hip-yaw neutrality for the
        # fore-aft centering term late in training. Keep the stability terms intact,
        # but make that crooked-foot solution unprofitable for the actuator task.
        self.rewards.feet_heading_error_exp.weight = -1.5
        self.rewards.joint_deviation_hip_yaw_l1.weight = -0.45
        self.disable_zero_weight_rewards()
