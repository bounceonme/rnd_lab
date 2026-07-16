from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VELOCITY_ROOT = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity"
VELOCITY_CFG_PATH = VELOCITY_ROOT / "velocity_env_cfg.py"
ROUGH_CFG_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "rough_env_cfg.py"
IMU_CFG_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_actuator_imu_env_cfg.py"
IMU_RUNNER_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "agents" / "rsl_rl_actuator_imu_ppo_cfg.py"


def _attribute_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} was not found.")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {name!r} was not found in {class_node.name}.")


def _assignments(node: ast.AST) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            path = _attribute_path(child.targets[0])
            if path is not None:
                result[path] = child.value
    return result


def _dict_items(node: ast.expr) -> dict[str, ast.expr]:
    if not isinstance(node, ast.Dict):
        raise AssertionError(f"Expected a dictionary expression, got {ast.dump(node)}")
    return {ast.literal_eval(key): value for key, value in zip(node.keys, node.values, strict=True) if key is not None}


class RndImuEnvConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.velocity_tree = ast.parse(VELOCITY_CFG_PATH.read_text(encoding="utf-8"))
        cls.rough_tree = ast.parse(ROUGH_CFG_PATH.read_text(encoding="utf-8"))
        cls.imu_source = IMU_CFG_PATH.read_text(encoding="utf-8")
        cls.imu_tree = ast.parse(cls.imu_source)
        cls.runner_source = IMU_RUNNER_PATH.read_text(encoding="utf-8")
        cls.runner_tree = ast.parse(cls.runner_source)

    def test_opt_in_env_inherits_actuator_env_and_changes_only_actor_imu_terms(self):
        env_cfg = _class(self.imu_tree, "RndStepFlatActuatorImuEnvCfg")
        self.assertEqual([_attribute_path(base) for base in env_cfg.bases], ["RndStepFlatActuatorEnvCfg"])
        post_init = _method(env_cfg, "__post_init__")
        source = ast.get_source_segment(self.imu_source, post_init)
        self.assertIsNotNone(source)
        self.assertIn("super().__post_init__()", source)
        self.assertNotIn("observations.critic", source)

        assignments = _assignments(post_init)
        expected_targets = {
            "model",
            "self.observations.policy.base_ang_vel.func",
            "self.observations.policy.base_ang_vel.params",
            "self.observations.policy.base_ang_vel.noise",
            "self.observations.policy.base_ang_vel.scale",
            "self.observations.policy.projected_gravity.func",
            "self.observations.policy.projected_gravity.params",
            "self.observations.policy.projected_gravity.noise",
            "self.observations.policy.projected_gravity.scale",
        }
        self.assertEqual(set(assignments), expected_targets)
        self.assertIsInstance(assignments["model"], ast.Call)
        self.assertEqual(
            _attribute_path(assignments["model"].func),
            "mdp.load_rnd_cmp10a_observation_model",
        )
        self.assertEqual(
            _attribute_path(assignments["self.observations.policy.base_ang_vel.func"]),
            "mdp.RndCmp10aObservation",
        )
        self.assertEqual(
            _attribute_path(assignments["self.observations.policy.projected_gravity.func"]),
            "mdp.RndCmp10aObservation",
        )
        self.assertIsNone(ast.literal_eval(assignments["self.observations.policy.base_ang_vel.noise"]))
        self.assertIsNone(ast.literal_eval(assignments["self.observations.policy.projected_gravity.noise"]))
        self.assertEqual(
            _attribute_path(assignments["self.observations.policy.base_ang_vel.scale"]),
            "model.policy_angular_velocity_scale",
        )
        self.assertEqual(ast.literal_eval(assignments["self.observations.policy.projected_gravity.scale"]), 1.0)

        gyro_params = _dict_items(assignments["self.observations.policy.base_ang_vel.params"])
        gravity_params = _dict_items(assignments["self.observations.policy.projected_gravity.params"])
        self.assertEqual(set(gyro_params), {"channel", "model_path", "sample_randomization"})
        self.assertEqual(set(gravity_params), {"channel", "model_path", "sample_randomization"})
        self.assertEqual(ast.literal_eval(gyro_params["channel"]), "gyro")
        self.assertEqual(ast.literal_eval(gravity_params["channel"]), "gravity")
        self.assertTrue(ast.literal_eval(gyro_params["sample_randomization"]))
        self.assertTrue(ast.literal_eval(gravity_params["sample_randomization"]))

    def test_base_actor_order_and_existing_scales_are_preserved(self):
        policy_cfg = _class(self.velocity_tree, "PolicyCfg")
        term_names = [
            target.id
            for node in policy_cfg.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        self.assertEqual(term_names[:4], ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands"])

        projected_gravity = next(
            node.value
            for node in policy_cfg.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "projected_gravity" for target in node.targets)
        )
        self.assertIsInstance(projected_gravity, ast.Call)
        gravity_scale = next(keyword.value for keyword in projected_gravity.keywords if keyword.arg == "scale")
        self.assertEqual(ast.literal_eval(gravity_scale), 1.0)

        rough_cfg = _class(self.rough_tree, "RndStepRoughEnvCfg")
        rough_assignments = _assignments(_method(rough_cfg, "__post_init__"))
        self.assertEqual(
            ast.literal_eval(rough_assignments["self.observations.policy.base_ang_vel.scale"]),
            0.25,
        )

    def test_runner_only_changes_experiment_name_from_actuator_runner(self):
        runner = _class(self.runner_tree, "RndStepFlatActuatorImuPPORunnerCfg")
        self.assertEqual([_attribute_path(base) for base in runner.bases], ["RndStepFlatActuatorPPORunnerCfg"])
        class_assignments = {
            target.id: node.value
            for node in runner.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(set(class_assignments), {"experiment_name"})
        self.assertEqual(ast.literal_eval(class_assignments["experiment_name"]), "rnd_step/flat_actuator_imu")

        post_init = _method(runner, "__post_init__")
        assignments = _assignments(post_init)
        self.assertEqual(set(assignments), {"self.experiment_name"})
        self.assertEqual(ast.literal_eval(assignments["self.experiment_name"]), "rnd_step/flat_actuator_imu")
        source = ast.get_source_segment(self.runner_source, post_init)
        self.assertIsNotNone(source)
        self.assertIn("super().__post_init__()", source)


if __name__ == "__main__":
    unittest.main()
