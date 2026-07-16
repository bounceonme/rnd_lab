from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VELOCITY_ROOT = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity"
COMMANDS_PATH = VELOCITY_ROOT / "mdp" / "commands.py"
REWARDS_PATH = VELOCITY_ROOT / "mdp" / "rewards.py"
EVENTS_PATH = VELOCITY_ROOT / "mdp" / "events.py"
FLAT_BEHAVIOR_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_behavior_cfg.py"
FLAT_ENV_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_env_cfg.py"
FLAT_ACTUATOR_ENV_PATH = VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_actuator_env_cfg.py"
FLAT_DOMAIN_RANDOMIZATION_PATH = (
    VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "flat_domain_randomization.py"
)
ACTUATOR_AGENT_PATH = (
    VELOCITY_ROOT / "config" / "humanoid" / "rnd_step" / "agents" / "rsl_rl_actuator_ppo_cfg.py"
)


def _attribute_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found.")


def _find_literal_assignment(node: ast.AST, target_path: str):
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            value = child.value
        elif isinstance(child, ast.AnnAssign) and child.value is not None:
            target = child.target
            value = child.value
        else:
            continue
        if _attribute_path(target) == target_path:
            return ast.literal_eval(value)
    raise AssertionError(f"Assignment to {target_path!r} was not found.")


def _find_call_keyword(node: ast.AST, target_path: str, keyword_name: str):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        if _attribute_path(child.targets[0]) != target_path or not isinstance(child.value, ast.Call):
            continue
        for keyword in child.value.keywords:
            if keyword.arg == keyword_name:
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"Keyword {keyword_name!r} for assignment {target_path!r} was not found.")


def _find_call_dict_value(node: ast.AST, target_path: str, keyword_name: str, key_name: str):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        if _attribute_path(child.targets[0]) != target_path or not isinstance(child.value, ast.Call):
            continue
        keyword = next((item for item in child.value.keywords if item.arg == keyword_name), None)
        if keyword is None or not isinstance(keyword.value, ast.Dict):
            continue
        for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == key_name:
                return ast.literal_eval(value)
    raise AssertionError(f"Dictionary key {key_name!r} for assignment {target_path!r} was not found.")


def _compile_isolated_function(tree: ast.Module, name: str):
    function = _find_function(tree, name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, REWARDS_PATH, "exec"), namespace)
    return namespace[name]


class RndStepTurnTrainingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands_source = COMMANDS_PATH.read_text(encoding="utf-8")
        cls.commands_tree = ast.parse(cls.commands_source)
        cls.rewards_source = REWARDS_PATH.read_text(encoding="utf-8")
        cls.rewards_tree = ast.parse(cls.rewards_source)
        cls.events_source = EVENTS_PATH.read_text(encoding="utf-8")
        cls.events_tree = ast.parse(cls.events_source)
        cls.behavior_tree = ast.parse(FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"))
        cls.flat_env_source = FLAT_ENV_PATH.read_text(encoding="utf-8")
        cls.flat_env_tree = ast.parse(cls.flat_env_source)
        cls.flat_actuator_source = FLAT_ACTUATOR_ENV_PATH.read_text(encoding="utf-8")
        cls.flat_actuator_tree = ast.parse(cls.flat_actuator_source)
        cls.domain_randomization_source = FLAT_DOMAIN_RANDOMIZATION_PATH.read_text(encoding="utf-8")
        cls.domain_randomization_tree = ast.parse(cls.domain_randomization_source)
        cls.actuator_agent_source = ACTUATOR_AGENT_PATH.read_text(encoding="utf-8")
        cls.actuator_agent_tree = ast.parse(cls.actuator_agent_source)

    def test_flat_and_actuator_tasks_share_the_turn_training_configuration(self):
        flat_env = next(
            node
            for node in self.flat_env_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepFlatEnvCfg"
        )
        post_init = next(
            node for node in flat_env.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        source = ast.get_source_segment(self.flat_env_source, post_init)
        self.assertIsNotNone(source)
        self.assertIn("apply_step_flat_stable_walk_rewards(self)", source)
        self.assertIn("apply_step_flat_commands(self)", source)

        actuator_env = next(
            node
            for node in self.flat_actuator_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepFlatActuatorEnvCfg"
        )
        self.assertEqual([_attribute_path(base) for base in actuator_env.bases], ["RndStepFlatEnvCfg"])

    def test_pure_yaw_sampling_is_opt_in_for_shared_command_generator(self):
        command_cfg = next(
            node
            for node in self.commands_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "UniformThresholdVelocityCommandCfg"
        )
        self.assertEqual(_find_literal_assignment(command_cfg, "rel_pure_yaw_envs"), 0.0)

    def test_rnd_flat_enables_turn_training_without_reducing_yaw_range(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_commands")
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.commands.base_velocity.rel_standing_envs"), 0.25)
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.commands.base_velocity.rel_pure_yaw_envs"), 0.25)
        self.assertEqual(
            _find_literal_assignment(configure, "env_cfg.commands.base_velocity.ranges.ang_vel_z"), (-1.0, 1.0)
        )
        self.assertEqual(
            _find_literal_assignment(configure, "env_cfg.commands.base_velocity.command_ramp_rates"),
            (0.4, 0.8, 1.2),
        )

    def test_rnd_flat_uses_tighter_velocity_tracking_rewards(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.rewards.track_lin_vel_xy_exp.weight"), 8.0)
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.rewards.track_ang_vel_z_exp.weight"), 5.5)
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.rewards.feet_flight_penalty.weight"), -0.5)
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.lin_vel_xy_underspeed_l2", "weight"), -4.0
        )

        source = ast.get_source_segment(FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"), configure)
        self.assertIsNotNone(source)
        self.assertIn('track_lin_vel_xy_exp.params["std"] = 0.25', source)
        self.assertIn('track_ang_vel_z_exp.params["std"] = 0.35', source)

    def test_contact_shaping_does_not_force_hard_single_support_swaps(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        self.assertEqual(_find_literal_assignment(configure, "env_cfg.rewards.feet_air_time.weight"), 1.0)
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.biped_gait_phase_l2", "weight"),
            -1.5,
        )
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.biped_phase_duration_l2", "weight"),
            -0.5,
        )

        source = ast.get_source_segment(FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"), configure)
        self.assertIsNotNone(source)
        self.assertIn("env_cfg.rewards.feet_air_time.func = mdp.feet_air_time", source)
        self.assertIn('feet_air_time.params["threshold"] = 0.20', source)
        self.assertIn('feet_air_time.params["max_time"] = 0.50', source)

    def test_phase_duration_penalty_is_bounded_after_a_standing_command(self):
        penalty_fn = _compile_isolated_function(self.rewards_tree, "_bounded_biped_phase_duration_l2")
        contact_time = torch.tensor([[10.0, 10.0], [0.4, 0.4]])
        air_time = torch.zeros_like(contact_time)

        penalty = penalty_fn(contact_time, air_time, 0.4)

        self.assertAlmostEqual(float(penalty[0]), 2.0, places=6)
        self.assertEqual(float(penalty[1]), 0.0)
        self.assertTrue(torch.isfinite(penalty).all())

    def test_fore_aft_balance_ema_rejects_fixed_lead_but_not_alternating_steps(self):
        update_fn = _compile_isolated_function(self.rewards_tree, "_update_biped_fore_aft_bias_ema")

        def run(sequence: list[float], step_dt: float) -> tuple[float, float]:
            ema = torch.zeros(1)
            active = torch.ones(1, dtype=torch.bool)
            alpha = -math.expm1(-step_dt / 1.5)
            penalties = []
            for value in sequence:
                penalty = update_fn(ema, torch.tensor([value]), active, alpha)
                penalties.append(float(penalty.item()))
            return float(ema.item()), sum(penalties) / len(penalties)

        fixed_positive, _ = run([1.0] * 225, 0.02)
        fixed_negative, _ = run([-1.0] * 225, 0.02)
        alternating_sequence = [1.0 if (step // 20) % 2 == 0 else -1.0 for step in range(3000)]
        _, alternating_mean_penalty = run(alternating_sequence, 0.02)

        self.assertGreater(fixed_positive**2, 0.85)
        self.assertAlmostEqual(fixed_positive**2, fixed_negative**2, places=6)
        self.assertLess(alternating_mean_penalty, 0.01)

        ema = torch.tensor([0.75, -0.5])
        active = torch.tensor([False, True])
        update_fn(ema, torch.tensor([1.0, -1.0]), active, 0.1)
        self.assertEqual(float(ema[0]), 0.0)

    def test_fore_aft_balance_uses_ordered_feet_and_yaw_only_frame(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_fore_aft_balance_l2", "weight"),
            -0.5,
        )
        self.assertEqual(
            _find_call_dict_value(
                configure,
                "env_cfg.rewards.feet_fore_aft_balance_l2",
                "params",
                "time_constant",
            ),
            1.5,
        )

        reward_class = next(
            node
            for node in self.rewards_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BipedFeetForeAftBalanceL2"
        )
        source = ast.get_source_segment(self.rewards_source, reward_class)
        self.assertIsNotNone(source)
        self.assertIn("yaw_quat(asset.data.root_link_quat_w)", source)
        self.assertIn("self._fore_aft_bias_ema[env_ids] = 0.0", source)

        configure_source = ast.get_source_segment(
            FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"),
            configure,
        )
        self.assertIsNotNone(configure_source)
        self.assertIn('body_names=["R_Leg_foot", "L_Leg_foot"]', configure_source)
        self.assertIn("preserve_order=True", configure_source)

    def test_underspeed_penalty_uses_command_direction_in_yaw_frame(self):
        function = _find_function(self.rewards_tree, "lin_vel_xy_underspeed_l2")
        source = ast.get_source_segment(self.rewards_source, function)
        self.assertIsNotNone(source)
        self.assertIn("quat_apply_inverse(yaw_quat", source)
        self.assertIn("command_speed - achieved_along_command", source)
        self.assertIn("min=0.0", source)

    def test_straight_gait_uses_actual_foot_placement_symmetry(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_forward_position_y_l2", "weight"), -1.5
        )
        source = ast.get_source_segment(FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"), configure)
        self.assertIsNotNone(source)
        self.assertIn('"yaw_scale": 0.45', source)

    def test_forward_center_target_matches_the_default_step_urdf_stance(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        target_center_y = _find_call_dict_value(
            configure,
            "env_cfg.rewards.feet_forward_position_y_l2",
            "params",
            "target_center_y",
        )
        self.assertEqual(target_center_y, 0.052)

        function = _find_function(self.rewards_tree, "feet_forward_position_y_l2_straight_yaw_command")
        source = ast.get_source_segment(self.rewards_source, function)
        self.assertIsNotNone(source)
        self.assertIn("foot_pos_b[:, :, 1] - target_center_y", source)

        # FK at STEP_DEFAULT_JOINT_POS gives these foot-link origins in the base-link frame.
        right_foot_y = 0.051527978977
        left_foot_y = 0.051601795288
        corrected_penalty = ((right_foot_y - target_center_y + left_foot_y - target_center_y) / 0.18) ** 2
        self.assertLess(corrected_penalty, 3.0e-5)

    def test_actuator_task_strengthens_heading_without_weakening_shared_stability(self):
        actuator_env = next(
            node
            for node in self.flat_actuator_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepFlatActuatorEnvCfg"
        )
        post_init = next(
            node for node in actuator_env.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertEqual(_find_literal_assignment(post_init, "self.rewards.feet_heading_error_exp.weight"), -1.5)
        self.assertEqual(
            _find_literal_assignment(post_init, "self.rewards.joint_deviation_hip_yaw_l1.weight"), -0.45
        )
        source = ast.get_source_segment(self.flat_actuator_source, post_init)
        self.assertIsNotNone(source)
        self.assertNotIn("is_terminated.weight", source)
        self.assertNotIn("flat_orientation_l2.weight", source)

    def test_armature_randomization_is_startup_only_and_actuator_task_only(self):
        events_cfg = next(
            node
            for node in self.flat_actuator_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepFlatActuatorEventCfg"
        )
        self.assertEqual([_attribute_path(base) for base in events_cfg.bases], ["EventCfg"])
        assignment = next(
            node
            for node in events_cfg.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "randomize_joint_armature"
        )
        self.assertIsInstance(assignment.value, ast.Call)
        self.assertEqual(_attribute_path(assignment.value.func), "EventTerm")
        keywords = {keyword.arg: keyword.value for keyword in assignment.value.keywords}
        self.assertEqual(ast.literal_eval(keywords["mode"]), "startup")
        self.assertIsInstance(keywords["params"], ast.Dict)
        params = {
            key.value: value
            for key, value in zip(keywords["params"].keys, keywords["params"].values, strict=True)
            if isinstance(key, ast.Constant)
        }
        self.assertTrue(ast.literal_eval(params["sample_randomization"]))
        self.assertEqual(ast.literal_eval(params["seed_offset"]), 2_000_003)
        self.assertNotIn("randomize_joint_armature", self.domain_randomization_source)

    def test_armature_event_writes_the_sampled_values_to_physx(self):
        function = _find_function(self.events_tree, "randomize_rnd_joint_armature")
        source = ast.get_source_segment(self.events_source, function)
        self.assertIsNotNone(source)
        self.assertIn("load_rnd_armature_randomization", source)
        self.assertIn("sample_rnd_armatures", source)
        self.assertIn("asset.write_joint_armature_to_sim", source)
        self.assertIn('getattr(env.cfg, "seed", 0)', source)

    def test_actuator_runner_stops_before_the_observed_late_drift_window(self):
        runner = next(
            node
            for node in self.actuator_agent_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepFlatActuatorPPORunnerCfg"
        )
        post_init = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
        self.assertEqual(_find_literal_assignment(post_init, "self.max_iterations"), 6500)

    def test_external_force_and_torque_randomization_remain_enabled(self):
        configure = _find_function(self.domain_randomization_tree, "apply_step_flat_domain_randomization")
        source = ast.get_source_segment(self.domain_randomization_source, configure)
        self.assertIsNotNone(source)
        self.assertIn('["force_range"] = (-1.0, 1.0)', source)
        self.assertIn('["torque_range"] = (-0.25, 0.25)', source)
        self.assertIn("randomize_push_robot.interval_range_s = (10.0, 15.0)", source)

    def test_data_driven_foot_placement_balance_preserves_turning(self):
        configure = _find_function(self.behavior_tree, "apply_step_flat_stable_walk_rewards")
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_min_lateral_distance_x_l2", "weight"), -6.0
        )
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_hard_min_lateral_distance_x_l2", "weight"),
            -8.0,
        )
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_lateral_position_x_l2", "weight"), -2.0
        )
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.feet_lateral_center_x_l2", "weight"), -4.0
        )
        self.assertEqual(_find_call_keyword(configure, "env_cfg.rewards.biped_gait_phase_l2", "weight"), -1.5)
        self.assertEqual(
            _find_call_keyword(configure, "env_cfg.rewards.lateral_tilt_x_with_cmd_l2", "weight"), -10.0
        )

        source = ast.get_source_segment(FLAT_BEHAVIOR_PATH.read_text(encoding="utf-8"), configure)
        self.assertIsNotNone(source)
        self.assertIn("func=mdp.feet_min_lateral_distance_x_l2_straight_yaw_command", source)
        self.assertIn('"min_width": 0.11', source)
        self.assertIn('"yaw_scale": 0.55', source)

    def test_nominal_minimum_width_reward_is_released_for_large_turns(self):
        function = _find_function(self.rewards_tree, "feet_min_lateral_distance_x_l2_straight_yaw_command")
        source = ast.get_source_segment(self.rewards_source, function)
        self.assertIsNotNone(source)
        self.assertIn("feet_min_lateral_distance_x_l2(", source)
        self.assertIn("_straight_yaw_command_gate", source)

    def test_pure_yaw_sampler_zeros_translation_but_not_yaw(self):
        command_class = next(
            node
            for node in self.commands_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "UniformThresholdVelocityCommand"
        )
        sampler = next(
            node
            for node in command_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sample_command_target"
        )
        source = ast.get_source_segment(self.commands_source, sampler)
        self.assertIsNotNone(source)
        self.assertIn("~self.is_standing_env[env_ids]", source)
        self.assertIn("~self.is_pure_yaw_env[env_ids]", source)
        self.assertNotIn("self.vel_command_target_b[env_ids, 2] *=", source)

    def test_flight_penalty_gates_on_linear_and_angular_commands(self):
        function = _find_function(self.rewards_tree, "feet_flight_penalty")
        source = ast.get_source_segment(self.rewards_source, function)
        self.assertIsNotNone(source)
        self.assertIn("torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)", source)
        self.assertNotIn("get_command(command_name)[:, :2]", source)


if __name__ == "__main__":
    unittest.main()
