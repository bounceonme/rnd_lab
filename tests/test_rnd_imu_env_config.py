from __future__ import annotations

import ast
import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VELOCITY_ROOT = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity"
VELOCITY_CFG_PATH = VELOCITY_ROOT / "velocity_env_cfg.py"
ROUGH_CFG_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "rough_env_cfg.py"
IMU_CFG_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_actuator_imu_env_cfg.py"
IMU_RUNNER_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "agents" / "rsl_rl_actuator_imu_ppo_cfg.py"
IMU_MODEL_PATH = REPO_ROOT / "scripts" / "tools" / "config" / "rnd_cmp10a_runtime.json"
STEP_URDF_PATH = REPO_ROOT / "source" / "robot_lab" / "data" / "Robots" / "rnd" / "step" / "urdf" / "step.urdf"


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

    def test_opt_in_env_inherits_actuator_env_and_changes_only_actor_sensor_terms(self):
        env_cfg = _class(self.imu_tree, "RndStepFlatActuatorImuEnvCfg")
        self.assertEqual([_attribute_path(base) for base in env_cfg.bases], ["RndStepFlatActuatorEnvCfg"])
        post_init = _method(env_cfg, "__post_init__")
        source = ast.get_source_segment(self.imu_source, post_init)
        self.assertIsNotNone(source)
        self.assertIn("super().__post_init__()", source)
        self.assertNotIn("observations.critic", source)

        assignments = _assignments(post_init)
        expected_targets = {
            "imu_model",
            "self.observations.policy.base_ang_vel.func",
            "self.observations.policy.base_ang_vel.params",
            "self.observations.policy.base_ang_vel.noise",
            "self.observations.policy.base_ang_vel.scale",
            "self.observations.policy.projected_gravity.func",
            "self.observations.policy.projected_gravity.params",
            "self.observations.policy.projected_gravity.noise",
            "self.observations.policy.projected_gravity.scale",
            "encoder_model",
            "self.observations.policy.joint_pos.func",
            "self.observations.policy.joint_pos.params",
            "self.observations.policy.joint_pos.noise",
            "self.observations.policy.joint_pos.scale",
            "self.observations.policy.joint_vel",
            "self.observations.policy.velocity_commands.history_length",
            "self.commands.base_velocity.transition_sequence_probabilities",
            "self.rewards.action_excess_l2",
            "self.rewards.vertical_touchdown_impact",
            "term",
            "term.history_length",
            "term.flatten_history_dim",
        }
        self.assertEqual(set(assignments), expected_targets)
        self.assertIsInstance(assignments["imu_model"], ast.Call)
        self.assertEqual(
            _attribute_path(assignments["imu_model"].func),
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
            "imu_model.policy_angular_velocity_scale",
        )
        self.assertEqual(ast.literal_eval(assignments["self.observations.policy.projected_gravity.scale"]), 1.0)

        gyro_params = _dict_items(assignments["self.observations.policy.base_ang_vel.params"])
        gravity_params = _dict_items(assignments["self.observations.policy.projected_gravity.params"])
        self.assertEqual(set(gyro_params), {"channel", "model_path", "sample_randomization", "body_name"})
        self.assertEqual(set(gravity_params), {"channel", "model_path", "sample_randomization", "body_name"})
        self.assertEqual(ast.literal_eval(gyro_params["channel"]), "gyro")
        self.assertEqual(ast.literal_eval(gravity_params["channel"]), "gravity")
        self.assertEqual(ast.literal_eval(gyro_params["body_name"]), "imu")
        self.assertEqual(ast.literal_eval(gravity_params["body_name"]), "imu")
        self.assertTrue(ast.literal_eval(gyro_params["sample_randomization"]))
        self.assertTrue(ast.literal_eval(gravity_params["sample_randomization"]))

        self.assertEqual(
            _attribute_path(assignments["encoder_model"].func),
            "mdp.load_rnd_dynamixel_encoder_observation_model",
        )
        self.assertEqual(
            _attribute_path(assignments["self.observations.policy.joint_pos.func"]),
            "mdp.RndDynamixelEncoderObservation",
        )
        encoder_params = _dict_items(assignments["self.observations.policy.joint_pos.params"])
        self.assertEqual(set(encoder_params), {"asset_cfg", "model_path", "sample_randomization"})
        self.assertTrue(ast.literal_eval(encoder_params["sample_randomization"]))
        self.assertIsNone(ast.literal_eval(assignments["self.observations.policy.joint_pos.noise"]))
        self.assertIsNone(ast.literal_eval(assignments["self.observations.policy.joint_vel"]))

        self.assertEqual(ast.literal_eval(assignments["term.history_length"]), 4)
        self.assertTrue(ast.literal_eval(assignments["term.flatten_history_dim"]))
        self.assertEqual(
            ast.literal_eval(assignments["self.observations.policy.velocity_commands.history_length"]),
            0,
        )
        self.assertEqual(
            ast.literal_eval(assignments["self.commands.base_velocity.transition_sequence_probabilities"]),
            (0.05, 0.05),
        )
        self.assertIn(
            '("base_ang_vel", "projected_gravity", "joint_pos", "actions")',
            source,
        )
        # 4 * (gyro 3 + gravity 3 + coherent encoder 24 + last action 12) + command 3.
        self.assertEqual(4 * (3 + 3 + 24 + 12) + 3, 171)

        action_excess_term = assignments["self.rewards.action_excess_l2"]
        self.assertIsInstance(action_excess_term, ast.Call)
        action_excess_keywords = {keyword.arg: keyword.value for keyword in action_excess_term.keywords}
        self.assertEqual(_attribute_path(action_excess_keywords["func"]), "mdp.action_excess_l2")
        self.assertEqual(ast.literal_eval(action_excess_keywords["weight"]), -0.10)
        action_excess_params = _dict_items(action_excess_keywords["params"])
        self.assertEqual(ast.literal_eval(action_excess_params["threshold"]), 1.5)

        touchdown_term = assignments["self.rewards.vertical_touchdown_impact"]
        self.assertIsInstance(touchdown_term, ast.Call)
        self.assertEqual(_attribute_path(touchdown_term.func), "RewTerm")
        touchdown_keywords = {keyword.arg: keyword.value for keyword in touchdown_term.keywords}
        self.assertEqual(
            _attribute_path(touchdown_keywords["func"]),
            "mdp.PhysicsTouchdownImpactCost",
        )
        self.assertEqual(ast.literal_eval(touchdown_keywords["weight"]), -0.5)
        touchdown_params = _dict_items(touchdown_keywords["params"])
        self.assertEqual(ast.literal_eval(touchdown_params["impact_speed_offset"]), 0.25)
        self.assertEqual(ast.literal_eval(touchdown_params["impact_speed_range"]), 0.50)
        self.assertEqual(ast.literal_eval(touchdown_params["min_air_time"]), 0.06)
        self.assertEqual(ast.literal_eval(touchdown_params["short_air_time_floor"]), 0.02)
        self.assertEqual(ast.literal_eval(touchdown_params["short_air_time_penalty_scale"]), 3.0)
        self.assertEqual(
            ast.literal_eval(touchdown_params["foot_body_names"]),
            ["R_Leg_foot", "L_Leg_foot"],
        )

    def test_active_imu_mount_matches_promoted_sensor_to_base_transform(self):
        root = ET.parse(STEP_URDF_PATH).getroot()
        joint = next(node for node in root.findall("joint") if node.get("name") == "imu_body")
        self.assertEqual(joint.get("type"), "fixed")
        self.assertEqual(joint.get("dont_collapse"), "true")
        self.assertEqual(joint.find("parent").get("link"), "Upper_Body")
        self.assertEqual(joint.find("child").get("link"), "imu")

        roll, pitch, yaw = (float(value) for value in joint.find("origin").get("rpy").split())
        self.assertAlmostEqual(roll, 0.0, places=12)
        self.assertAlmostEqual(pitch, 0.0, places=12)
        self.assertAlmostEqual(yaw, math.pi, places=12)
        urdf_sensor_to_base = [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
        runtime = json.loads(IMU_MODEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(runtime["runtime_transform"]["source_frame"], "sensor")
        self.assertEqual(runtime["runtime_transform"]["target_frame"], "base_link")
        runtime_matrix = runtime["runtime_transform"]["sensor_to_base_matrix"]
        for actual_row, expected_row in zip(urdf_sensor_to_base, runtime_matrix, strict=True):
            for actual, expected in zip(actual_row, expected_row, strict=True):
                self.assertAlmostEqual(actual, expected, places=12)

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

    def test_runner_stabilizes_the_history_actor_without_changing_the_critic(self):
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
        self.assertEqual(
            set(assignments),
            {
                "self.experiment_name",
                "self.policy.actor_obs_normalization",
                "self.policy.init_noise_std",
                "self.algorithm.entropy_coef",
            },
        )
        self.assertEqual(ast.literal_eval(assignments["self.experiment_name"]), "rnd_step/flat_actuator_imu")
        self.assertTrue(ast.literal_eval(assignments["self.policy.actor_obs_normalization"]))
        self.assertEqual(ast.literal_eval(assignments["self.policy.init_noise_std"]), 0.5)
        self.assertEqual(ast.literal_eval(assignments["self.algorithm.entropy_coef"]), 0.004)
        self.assertNotIn("critic_obs_normalization", assignments)
        source = ast.get_source_segment(self.runner_source, post_init)
        self.assertIsNotNone(source)
        self.assertIn("super().__post_init__()", source)


if __name__ == "__main__":
    unittest.main()
