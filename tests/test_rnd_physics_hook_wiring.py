from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RND_STEP_ROOT = (
    REPO_ROOT
    / "source"
    / "robot_lab"
    / "robot_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "humanoid"
    / "rnd_step"
)
ENV_PATH = RND_STEP_ROOT / "rnd_step_env.py"
REGISTRATION_PATH = RND_STEP_ROOT / "__init__.py"


class RndPhysicsHookWiringTest(unittest.TestCase):
    def test_physics_observer_reads_current_scene_data_before_termination(self):
        source = ENV_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        env_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RndStepManagerBasedRLEnv"
        )
        step = next(node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        step_source = ast.get_source_segment(source, step)
        self.assertIsNotNone(step_source)

        scene_update = step_source.index("self.scene.update(dt=self.physics_dt)")
        observer_update = step_source.index("observer.on_post_scene_update")
        termination = step_source.index("self.termination_manager.compute()")
        self.assertLess(scene_update, observer_update)
        self.assertLess(observer_update, termination)

    def test_terminal_sample_is_marked_before_environment_reset(self):
        source = ENV_PATH.read_text(encoding="utf-8")
        pre_reset = source.index("observer.on_pre_reset")
        reset = source.index("self._reset_idx(reset_env_ids)")
        post_reset = source.index("observer.on_post_reset")
        self.assertLess(pre_reset, reset)
        self.assertLess(reset, post_reset)

    def test_only_actuator_imu_task_uses_the_project_hook(self):
        source = REGISTRATION_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'id="RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0"',
            source,
        )
        self.assertEqual(source.count('entry_point=f"{__name__}.rnd_step_env:RndStepManagerBasedRLEnv"'), 1)
        self.assertEqual(source.count('entry_point="isaaclab.envs:ManagerBasedRLEnv"'), 3)


if __name__ == "__main__":
    unittest.main()
