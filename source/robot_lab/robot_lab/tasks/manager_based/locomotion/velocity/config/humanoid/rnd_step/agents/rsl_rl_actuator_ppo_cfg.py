"""RSL-RL runner configuration for the opt-in RND actuator task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import RndStepFlatPPORunnerCfg


@configclass
class RndStepFlatActuatorPPORunnerCfg(RndStepFlatPPORunnerCfg):
    experiment_name = "rnd_step/flat_actuator"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rnd_step/flat_actuator"
        # The completed actuator run peaked near 6.2k and regressed when continued
        # to 10k. Preserve enough training headroom without defaulting into that drift.
        self.max_iterations = 6500
