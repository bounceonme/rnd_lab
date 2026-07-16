import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tools" / "rnd_pd_test.py"
RUNTIME_MODEL_PATH = ROOT / "scripts" / "tools" / "config" / "rnd_actuator_model_runtime.json"
TORQUE_RANDOMIZATION_PATH = ROOT / "scripts" / "tools" / "config" / "rnd_torque_randomization.json"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class TestRndPdTestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.runtime_model = json.loads(RUNTIME_MODEL_PATH.read_text(encoding="utf-8"))
        cls.torque_randomization = json.loads(TORQUE_RANDOMIZATION_PATH.read_text(encoding="utf-8"))

    def test_default_task_uses_replay_validated_actuator_environment(self):
        task_argument = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--task"
        )
        default = next(keyword.value for keyword in task_argument.keywords if keyword.arg == "default")
        self.assertIsInstance(default, ast.Constant)
        self.assertEqual(default.value, "RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0")

    def test_runtime_model_is_required_and_fully_validated(self):
        function = _find_function(self.tree, "_validate_pd_test_env")
        source = ast.get_source_segment(self.source, function)
        self.assertIsNotNone(source)
        self.assertIn("RND_ACTUATOR_RUNTIME_MODEL_PATH", source)
        self.assertIn("require_sim_replay_validation=True", source)
        self.assertIn("require_command_path_seed=True", source)
        self.assertIn("integration_joint_names", source)

        model = self.runtime_model
        self.assertTrue(model["integration_enabled"])
        self.assertEqual(model["application_status"], "sim_replay_validated")
        self.assertEqual(set(model["integration_joint_names"]), set(model["joint_order"]))
        self.assertEqual(model["fallback_joint_names"], [])
        for joint in model["joints"].values():
            self.assertTrue(joint["quality"]["sim_replay_validated"])
            self.assertTrue(joint["quality"]["command_path_seed_usable"])
            self.assertTrue(joint["quality"]["integration_allowed"])

    def test_pd_defaults_come_from_instantiated_actuators(self):
        function = _find_function(self.tree, "_collect_joint_defaults")
        source = ast.get_source_segment(self.source, function)
        self.assertIsNotNone(source)
        self.assertIn("robot.actuators.items()", source)
        self.assertIn("actuator.stiffness", source)
        self.assertIn("actuator.damping", source)
        self.assertIn("actuator.effort_limit", source)
        self.assertIn("actuator.velocity_limit", source)
        self.assertIn("robot.data.joint_armature", source)
        self.assertIn("actuator.armature", source)
        self.assertNotIn("STEP_CFG.actuators", self.source)

    def test_explicit_pd_override_does_not_write_physx_drive_gains(self):
        window = next(node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == "PDGainWindow")
        function = next(
            node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_gain"
        )
        source = ast.get_source_segment(self.source, function)
        self.assertIsNotNone(source)
        self.assertIn("actuator.stiffness[:, actuator_joint_id]", source)
        self.assertIn("actuator.damping[:, actuator_joint_id]", source)
        self.assertEqual(source.count("if actuator.is_implicit_model"), 2)

    def test_command_path_is_initialized_before_reporting_samples(self):
        function = _find_function(self.tree, "main")
        source = ast.get_source_segment(self.source, function)
        self.assertIsNotNone(source)
        self.assertLess(source.index("env.step(zero_actions)"), source.index("_collect_joint_defaults"))
        self.assertLess(source.index("_collect_joint_defaults"), source.index("_print_actuator_summary"))

    def test_torque_randomization_is_reported_separately_from_command_model(self):
        self.assertFalse(self.runtime_model["torque_calibration"]["available"])
        self.assertTrue(all(not joint["friction"]["enabled"] for joint in self.runtime_model["joints"].values()))
        self.assertTrue(self.torque_randomization["integration_enabled"])
        self.assertEqual(self.torque_randomization["quality_summary"]["measured_joint_count"], 4)
        self.assertIn("Torque/friction domain randomization is ON", self.source)
        self.assertIn("sampled_coulomb_torque_nm", self.source)
        self.assertIn("sampled_motor_strength_scale", self.source)

    def test_fixed_actuator_sample_also_uses_midpoint_armature(self):
        function = _find_function(self.tree, "_configure_pd_test_env")
        source = ast.get_source_segment(self.source, function)
        self.assertIsNotNone(source)
        self.assertIn('hasattr(env_cfg.events, "randomize_joint_armature")', source)
        self.assertIn(
            'randomize_joint_armature.params["sample_randomization"] = not args_cli.fixed_actuator_sample',
            source,
        )


if __name__ == "__main__":
    unittest.main()
