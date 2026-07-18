from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOUCHDOWN_PATH = (
    REPO_ROOT
    / "source"
    / "robot_lab"
    / "robot_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "touchdown.py"
)
TELEMETRY_PATH = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "physics_touchdown_telemetry.py"


def _load_touchdown_without_isaac_app():
    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    isaaclab_stub = types.ModuleType("isaaclab")
    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.ManagerTermBase = ManagerTermBase
    isaaclab_stub.managers = managers_stub
    module_names = ("isaaclab", "isaaclab.managers")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules["isaaclab"] = isaaclab_stub
    sys.modules["isaaclab.managers"] = managers_stub
    try:
        module_name = "_test_rnd_touchdown"
        spec = importlib.util.spec_from_file_location(module_name, TOUCHDOWN_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {TOUCHDOWN_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_telemetry_module():
    module_name = "_test_rnd_touchdown_telemetry"
    spec = importlib.util.spec_from_file_location(module_name, TELEMETRY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TELEMETRY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


touchdown = _load_touchdown_without_isaac_app()
telemetry = _load_telemetry_module()


def _forces(contact: torch.Tensor, magnitude: float = 2.0) -> torch.Tensor:
    result = torch.zeros((*contact.shape, 3), dtype=torch.float32, device=contact.device)
    result[..., 2] = contact.to(dtype=torch.float32) * magnitude
    return result


def _velocities(vz: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((*vz.shape, 3), dtype=torch.float32, device=vz.device)
    result[..., 2] = vz
    return result


def _monitor(num_envs: int = 1):
    return touchdown.PhysicsTouchdownMonitor(
        num_envs=num_envs,
        device="cpu",
        physics_dt=0.005,
        samples_per_policy=4,
        force_threshold=1.0,
        min_air_time=0.06,
    )


def _airborne_samples(monitor, count: int, vz: torch.Tensor | None = None) -> None:
    if vz is None:
        vz = torch.full((monitor.num_envs, 2), -0.5, dtype=torch.float32)
    no_contact = torch.zeros((monitor.num_envs, 2), dtype=torch.bool)
    for _ in range(count):
        monitor.update(_forces(no_contact), _velocities(vz))


class _FakeEntity:
    def __init__(self, body_names: list[str], data: SimpleNamespace, cfg: SimpleNamespace | None = None):
        self.body_names = body_names
        self.data = data
        self.cfg = cfg

    def find_bodies(self, requested, preserve_order=False):
        names = [str(name) for name in requested]
        if not preserve_order:
            names = sorted(names, key=self.body_names.index)
        return [self.body_names.index(name) for name in names], names


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, name: str) -> torch.Tensor:
        if name != "base_velocity":
            raise KeyError(name)
        return self.command


class _FakeObserverEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = "cpu"
        self.physics_dt = 0.005
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(decimation=4)
        self.observers = []

        robot_body_names = ["base", "L_Leg_foot", "R_Leg_foot"]
        sensor_body_names = ["base", "R_Leg_foot", "L_Leg_foot"]
        joint_names = ["left_joint", "right_joint"]
        robot_data = SimpleNamespace(
            root_link_pos_w=torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3),
            root_link_quat_w=torch.arange(num_envs * 4, dtype=torch.float32).reshape(num_envs, 4),
            root_link_lin_vel_w=torch.full((num_envs, 3), 0.3, dtype=torch.float32),
            root_link_ang_vel_w=torch.full((num_envs, 3), 0.4, dtype=torch.float32),
            body_link_pos_w=torch.arange(num_envs * 3 * 3, dtype=torch.float32).reshape(num_envs, 3, 3),
            body_link_lin_vel_w=torch.zeros((num_envs, 3, 3), dtype=torch.float32),
            body_link_ang_vel_w=torch.full((num_envs, 3, 3), 0.2, dtype=torch.float32),
            joint_pos=torch.full((num_envs, 2), 0.1, dtype=torch.float32),
            joint_vel=torch.full((num_envs, 2), 0.2, dtype=torch.float32),
            applied_torque=torch.full((num_envs, 2), 0.3, dtype=torch.float32),
            computed_torque=torch.full((num_envs, 2), 0.4, dtype=torch.float32),
        )
        robot = _FakeEntity(robot_body_names, robot_data)
        robot.joint_names = joint_names
        sensor_data = SimpleNamespace(net_forces_w=torch.zeros((num_envs, 3, 3), dtype=torch.float32))
        sensor = _FakeEntity(
            sensor_body_names,
            sensor_data,
            cfg=SimpleNamespace(force_threshold=1.0),
        )
        self.scene = {"robot": robot, "contact_forces": sensor}
        self.command_manager = _FakeCommandManager(torch.zeros((num_envs, 3), dtype=torch.float32))

    def add_rnd_physics_observer(self, observer) -> None:
        if observer not in self.observers:
            self.observers.append(observer)

    def post_scene_update(self, action: torch.Tensor, policy_step: int, substep_index: int) -> None:
        for observer in tuple(self.observers):
            observer.on_post_scene_update(self, action, policy_step, substep_index)

    def pre_reset(self, env_ids: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        for observer in tuple(self.observers):
            observer.on_pre_reset(self, env_ids, terminated, truncated)

    def post_reset(self, env_ids: torch.Tensor) -> None:
        for observer in tuple(self.observers):
            observer.on_post_reset(self, env_ids)


class PhysicsTouchdownMonitorTest(unittest.TestCase):
    def test_four_samples_per_policy_at_five_milliseconds(self):
        monitor = _monitor()
        no_contact = torch.zeros((1, 2), dtype=torch.bool)
        samples = [monitor.update(_forces(no_contact), _velocities(torch.zeros((1, 2)))) for _ in range(8)]

        self.assertEqual([sample.physics_step.item() for sample in samples], list(range(8)))
        self.assertEqual([sample.policy_step.item() for sample in samples], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual([sample.substep.item() for sample in samples], [0, 1, 2, 3, 0, 1, 2, 3])
        np.testing.assert_allclose(
            [sample.physics_time_s.item() for sample in samples],
            np.arange(1, 9) * 0.005,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_preimpact_uses_previous_link_velocity_and_persists_to_policy_reward(self):
        monitor = _monitor()
        airborne_vz = torch.tensor([[-0.9, -0.4]], dtype=torch.float32)
        _airborne_samples(monitor, 12, airborne_vz)

        contact = torch.tensor([[True, False]])
        touchdown_sample = monitor.update(
            _forces(contact),
            _velocities(torch.tensor([[0.2, -0.4]], dtype=torch.float32)),
        )
        self.assertTrue(torch.equal(touchdown_sample.first, torch.tensor([[True, False]])))
        self.assertTrue(torch.equal(touchdown_sample.valid, torch.tensor([[True, False]])))
        torch.testing.assert_close(touchdown_sample.preimpact, torch.tensor([[0.9, 0.0]]))

        for _ in range(2):
            later = monitor.update(_forces(contact), _velocities(torch.zeros((1, 2))))
            self.assertFalse(bool(later.first.any()))
        pending = monitor.peek_pending()
        self.assertTrue(torch.equal(pending.valid, torch.tensor([[True, False]])))
        torch.testing.assert_close(pending.preimpact_speed, torch.tensor([[0.9, 0.0]]))

        consumed = monitor.consume_pending()
        self.assertTrue(torch.equal(consumed.valid, torch.tensor([[True, False]])))
        self.assertFalse(bool(monitor.consume_pending().valid.any()))

    def test_short_touchdown_is_a_tap_but_not_a_valid_impact(self):
        monitor = _monitor()
        initial_contact = torch.tensor([[True, False]])
        monitor.update(_forces(initial_contact), _velocities(torch.zeros((1, 2))))
        _airborne_samples(monitor, 11, torch.tensor([[-1.0, -0.5]], dtype=torch.float32))

        chatter = monitor.update(_forces(initial_contact), _velocities(torch.zeros((1, 2))))
        self.assertTrue(chatter.first[0, 0])
        self.assertAlmostEqual(chatter.preceding_air_time_s[0, 0].item(), 0.055, places=6)
        self.assertFalse(chatter.valid[0, 0])
        pending = monitor.peek_pending()
        self.assertFalse(pending.valid[0, 0])
        self.assertAlmostEqual(pending.short_air_time_cost[0, 0].item(), 0.125, places=5)

    def test_sub_twenty_millisecond_contact_chatter_is_ignored(self):
        monitor = _monitor()
        initial_contact = torch.tensor([[True, False]])
        monitor.update(_forces(initial_contact), _velocities(torch.zeros((1, 2))))
        _airborne_samples(monitor, 3, torch.tensor([[-1.0, -0.5]], dtype=torch.float32))

        monitor.update(_forces(initial_contact), _velocities(torch.zeros((1, 2))))

        pending = monitor.peek_pending()
        self.assertFalse(pending.valid[0, 0])
        self.assertEqual(pending.short_air_time_cost[0, 0].item(), 0.0)

    def test_simultaneous_touchdowns_are_preserved_and_upward_precontact_is_zero(self):
        monitor = _monitor()
        _airborne_samples(monitor, 12, torch.tensor([[-0.8, 0.5]], dtype=torch.float32))

        simultaneous = monitor.update(
            _forces(torch.tensor([[True, True]])),
            _velocities(torch.zeros((1, 2))),
        )
        self.assertTrue(torch.equal(simultaneous.first, torch.tensor([[True, True]])))
        self.assertTrue(torch.equal(simultaneous.valid, torch.tensor([[True, True]])))
        torch.testing.assert_close(simultaneous.preimpact, torch.tensor([[0.8, 0.0]]))

    def test_reset_isolation_and_terminal_sample_survive_post_reset(self):
        monitor = _monitor(num_envs=2)
        _airborne_samples(monitor, 12, torch.full((2, 2), -1.0))
        contact = torch.tensor([[True, False], [False, False]])
        terminal_sample = monitor.update(_forces(contact), _velocities(torch.full((2, 2), -1.0)))
        self.assertTrue(terminal_sample.valid[0, 0])
        monitor.consume_pending()

        monitor.preserve_terminal_before_reset(
            terminated=torch.tensor([True, False]),
            truncated=torch.tensor([False, False]),
            reset_after_sample=torch.tensor([True, False]),
        )
        monitor.reset(torch.tensor([0]))

        self.assertTrue(torch.equal(monitor.episode_id, torch.tensor([1, 0])))
        torch.testing.assert_close(monitor.airborne_duration[0], torch.zeros(2))
        self.assertGreater(monitor.airborne_duration[1, 0].item(), 0.06)
        preserved = monitor.take_preserved_terminal(torch.tensor([0]))
        self.assertTrue(preserved.sample.terminated.item())
        self.assertTrue(preserved.sample.reset_after_sample.item())
        self.assertTrue(preserved.sample.valid[0, 0])
        torch.testing.assert_close(preserved.sample.preimpact[0], torch.tensor([1.0, 0.0]))

        next_contact = torch.tensor([[False, False], [True, True]])
        isolated = monitor.update(_forces(next_contact), _velocities(torch.zeros((2, 2))))
        self.assertFalse(bool(isolated.valid[0].any()))
        self.assertTrue(bool(isolated.valid[1].all()))

    def test_reward_gate_saturation_two_foot_sum_and_one_time_consumption(self):
        monitor = _monitor(num_envs=4)
        precontact_vz = torch.tensor(
            [
                [-1.0, -1.0],
                [-0.5, -0.75],
                [-1.2, -0.25],
                [-1.0, -1.0],
            ],
            dtype=torch.float32,
        )
        _airborne_samples(monitor, 12, precontact_vz)
        monitor.update(_forces(torch.ones((4, 2), dtype=torch.bool)), _velocities(torch.zeros((4, 2))))
        command = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0],
                [0.0, 0.0, 0.16],
                [0.10, 0.0, 0.15],
            ],
            dtype=torch.float32,
        )

        cost = touchdown.consume_touchdown_impact_cost(monitor, command)
        torch.testing.assert_close(cost, torch.tensor([0.0, 1.5, 1.0, 0.0]))
        torch.testing.assert_close(touchdown.consume_touchdown_impact_cost(monitor, command), torch.zeros(4))

    def test_short_air_time_cost_is_command_gated_and_consumed_once(self):
        monitor = _monitor(num_envs=2)
        initial_contact = torch.tensor([[True, False], [True, False]])
        monitor.update(_forces(initial_contact), _velocities(torch.zeros((2, 2))))
        _airborne_samples(monitor, 8, torch.full((2, 2), -0.5))
        monitor.update(_forces(initial_contact), _velocities(torch.zeros((2, 2))))
        command = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])

        cost = touchdown.consume_touchdown_impact_cost(
            monitor,
            command,
            short_air_time_penalty_scale=3.0,
        )

        torch.testing.assert_close(cost, torch.tensor([1.5, 0.0]))
        torch.testing.assert_close(
            touchdown.consume_touchdown_impact_cost(
                monitor,
                command,
                short_air_time_penalty_scale=3.0,
            ),
            torch.zeros(2),
        )


class PhysicsTouchdownObserverAndTelemetryTest(unittest.TestCase):
    def _attach_term(self, env: _FakeObserverEnv):
        cfg = SimpleNamespace(params={"command_name": "base_velocity"})
        term = touchdown.PhysicsTouchdownImpactCost(cfg, env)
        self.assertIs(env.physics_touchdown_monitor, term.monitor)
        self.assertEqual(env.observers, [term.monitor])
        return term

    def test_manager_term_registers_protocol_observer_and_consumes_once(self):
        env = _FakeObserverEnv(num_envs=1)
        term = self._attach_term(env)
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        action = torch.zeros((1, 2))
        foot_robot_ids = term.monitor.asset_foot_ids
        foot_sensor_ids = term.monitor.sensor_foot_ids

        for physics_step in range(12):
            robot.data.body_link_lin_vel_w[:, foot_robot_ids, 2] = -0.75
            sensor.data.net_forces_w[:, foot_sensor_ids, :] = 0.0
            env.post_scene_update(action, physics_step // 4, physics_step % 4)
        sensor.data.net_forces_w[:, foot_sensor_ids, 2] = 2.0
        env.post_scene_update(action, 3, 0)
        env.command_manager.command[:, 0] = 0.2

        torch.testing.assert_close(term(env, command_name="base_velocity"), torch.tensor([2.0]))
        torch.testing.assert_close(term(env, command_name="base_velocity"), torch.tensor([0.0]))

        env.pre_reset(torch.tensor([0]), torch.tensor([True]), torch.tensor([False]))
        env.post_reset(torch.tensor([0]))
        preserved = term.monitor.take_preserved_terminal(torch.tensor([0]))
        self.assertTrue(preserved.sample.terminated.item())
        self.assertEqual(term.monitor.episode_id.item(), 1)

    def test_duplicate_term_reuses_only_an_identical_observer_configuration(self):
        env = _FakeObserverEnv(num_envs=1)
        first = self._attach_term(env)
        second = touchdown.PhysicsTouchdownImpactCost(
            SimpleNamespace(params={"command_name": "base_velocity"}),
            env,
        )
        self.assertIs(second.monitor, first.monitor)
        self.assertEqual(env.observers, [first.monitor])

        with self.assertRaisesRegex(touchdown.PhysicsTouchdownError, "conflicting"):
            touchdown.PhysicsTouchdownImpactCost(
                SimpleNamespace(params={"command_name": "base_velocity", "min_air_time": 0.08}),
                env,
            )

    def test_chunked_adapter_writes_pickle_free_npz_v2_with_terminal_boundary(self):
        env = _FakeObserverEnv(num_envs=2)
        term = self._attach_term(env)
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        action = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "touchdown.telemetry"
            logger = telemetry.PhysicsTouchdownTelemetryLogger.from_attached_env(
                env,
                env_ids=[1],
                chunk_size=2,
                task="RND-Step",
                checkpoint="model.pt",
            )
            adapter = telemetry.PhysicsTouchdownTelemetryAdapter(logger, output_path=output_path).attach(env)
            self.assertEqual(env.observers, [term.monitor, adapter])

            for substep_index in range(3):
                sensor.data.net_forces_w.zero_()
                robot.data.body_link_lin_vel_w[:, term.monitor.asset_foot_ids, 2] = -0.4
                env.post_scene_update(action, 0, substep_index)
            self.assertEqual(logger.num_samples, 3)
            self.assertEqual(logger.num_cpu_flushes, 1)

            env.pre_reset(
                torch.tensor([1]),
                terminated=torch.tensor([False, True]),
                truncated=torch.tensor([False, False]),
            )
            env.post_reset(torch.tensor([1]))
            saved_path = adapter.save()

            self.assertEqual(saved_path, output_path.resolve())
            with np.load(saved_path, allow_pickle=False) as archive:
                self.assertEqual(archive["schema_version"].item(), 2)
                self.assertEqual(archive["schema_name"].item(), "robot_lab.physics_touchdown_telemetry")
                self.assertEqual(archive["task"].item(), "RND-Step")
                self.assertEqual(archive["num_samples"].item(), 3)
                np.testing.assert_array_equal(archive["env_ids"], [1])
                np.testing.assert_array_equal(archive["foot_body_names"], ["R_Leg_foot", "L_Leg_foot"])
                self.assertEqual(archive["command"].shape, (3, 1, 3))
                self.assertEqual(archive["actions"].shape, (3, 1, 2))
                self.assertEqual(archive["foot_pos_w"].shape, (3, 1, 2, 3))
                self.assertEqual(archive["joint_pos"].shape, (3, 1, 2))
                self.assertTrue(archive["terminated"][-1, 0])
                self.assertFalse(archive["truncated"][-1, 0])
                self.assertTrue(archive["reset_after_sample"][-1, 0])
                required_series = {
                    "episode_id",
                    "physics_step",
                    "policy_step",
                    "substep",
                    "root_pos_w",
                    "root_lin_vel_w",
                    "root_ang_vel_w",
                    "foot_pos_w",
                    "foot_lin_vel_w",
                    "foot_ang_vel_w",
                    "foot_force_w",
                    "foot_contact",
                    "foot_first",
                    "foot_valid",
                    "foot_preimpact_speed",
                    "joint_pos",
                    "joint_vel",
                    "applied_torque",
                    "computed_torque",
                }
                self.assertTrue(required_series.issubset(archive.files))
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))


if __name__ == "__main__":
    unittest.main()
