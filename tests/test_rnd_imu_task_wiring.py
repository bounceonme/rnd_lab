from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source" / "robot_lab" / "robot_lab"
RND_STEP_INIT = (
    PACKAGE / "tasks" / "manager_based" / "locomotion" / "velocity" / "config" / "humanoid" / "rnd_step" / "__init__.py"
)
MDP_INIT = PACKAGE / "tasks" / "manager_based" / "locomotion" / "velocity" / "mdp" / "__init__.py"
HARDWARE_INIT = PACKAGE / "hardware" / "__init__.py"
LIST_ENVS = ROOT / "scripts" / "tools" / "list_envs.py"


class RndImuTaskWiringTest(unittest.TestCase):
    def test_opt_in_task_registration_points_to_imu_configs(self):
        source = RND_STEP_INIT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        registrations = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "register"
        ]
        imu_registration = next(
            call.value
            for call in registrations
            if any(
                keyword.arg == "id"
                and ast.literal_eval(keyword.value) == "RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0"
                for keyword in call.value.keywords
            )
        )
        kwargs_node = next(keyword.value for keyword in imu_registration.keywords if keyword.arg == "kwargs")
        rendered = ast.unparse(kwargs_node)
        self.assertIn("flat_actuator_imu_env_cfg:RndStepFlatActuatorImuEnvCfg", rendered)
        self.assertIn("rsl_rl_actuator_imu_ppo_cfg:RndStepFlatActuatorImuPPORunnerCfg", rendered)

    def test_mdp_and_hardware_packages_export_runtime_symbols(self):
        mdp_source = MDP_INIT.read_text(encoding="utf-8")
        hardware_source = HARDWARE_INIT.read_text(encoding="utf-8")

        self.assertIn("from .imu_observations import *", mdp_source)
        for symbol in (
            "CMP10ARuntimeAdapter",
            "CMP10ARuntimeSource",
            "CMP10ARuntimeSnapshotError",
            "load_cmp10a_runtime_model",
        ):
            self.assertIn(symbol, hardware_source)

    def test_environment_listing_accepts_rndlab_prefix(self):
        source = LIST_ENVS.read_text(encoding="utf-8")

        self.assertIn('task_spec.id.startswith(("RobotLab-", "RNDLab-"))', source)


if __name__ == "__main__":
    unittest.main()
