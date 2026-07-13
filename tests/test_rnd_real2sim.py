from __future__ import annotations

import ast
import dataclasses
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_real2sim.collector import SafetyTrip, collect_dataset
from rnd_real2sim.config import RND_LEG_JOINT_NAMES, load_experiment_config, load_mapping_config
from rnd_real2sim.dataset import load_dataset
from rnd_real2sim.identification import identify_dataset
from rnd_real2sim.model import EncoderDomainActuatorRandomizer
from rnd_real2sim.synthetic import SyntheticMx2Bus


MAPPING_PATH = TOOLS_DIR / "config" / "rnd_dynamixel.toml"
EXPERIMENT_PATH = TOOLS_DIR / "config" / "rnd_real2sim.toml"
URDF_PATH = REPO_ROOT / "source" / "robot_lab" / "data" / "Robots" / "rnd" / "step" / "urdf" / "step.urdf"
ASSET_CFG_PATH = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "assets" / "rnd.py"


def _load_step_default_joint_positions() -> dict[str, float]:
    tree = ast.parse(ASSET_CFG_PATH.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "STEP_DEFAULT_JOINT_POS"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("STEP_DEFAULT_JOINT_POS was not found.")


class RndReal2SimTest(unittest.TestCase):
    def test_mapping_and_experiment_are_self_consistent(self):
        mapping = load_mapping_config(MAPPING_PATH)
        experiment = load_experiment_config(EXPERIMENT_PATH)
        self.assertEqual(len(mapping.joints), 12)
        self.assertEqual({joint.motor_id for joint in mapping.joints}, {10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
        self.assertEqual(mapping.protocol, 2.0)
        self.assertEqual(mapping.control_table, "mx_2")
        self.assertEqual(experiment.sample_hz, 50.0)
        self.assertEqual(set(experiment.reference_pose.positions_rad), set(RND_LEG_JOINT_NAMES))
        asset_default_positions = _load_step_default_joint_positions()
        self.assertEqual(set(asset_default_positions), set(RND_LEG_JOINT_NAMES))
        for name in RND_LEG_JOINT_NAMES:
            self.assertAlmostEqual(
                experiment.reference_pose.positions_rad[name], asset_default_positions[name], places=7
            )
        self.assertAlmostEqual(math.degrees(experiment.reference_pose.move_speed_rad_s), 8.0)
        self.assertAlmostEqual(experiment.reference_pose.settle_s, 3.0)
        self.assertAlmostEqual(math.degrees(experiment.reference_pose.tolerance_rad), 1.5)
        micro_triangle = next(profile for profile in experiment.profiles if profile.name == "micro_triangle")
        self.assertEqual(micro_triangle.precondition_cycles, 1)
        self.assertLessEqual(
            max(profile.amplitude_rad for profile in experiment.profiles), experiment.safety.max_excursion_rad
        )

    def test_dry_collection_fit_and_target_randomizer(self):
        mapping = load_mapping_config(MAPPING_PATH)
        base_experiment = load_experiment_config(EXPERIMENT_PATH)
        profiles = (
            dataclasses.replace(base_experiment.profiles[0], cycles=3),
            dataclasses.replace(base_experiment.profiles[-1], cycles=3),
        )
        identification = dataclasses.replace(base_experiment.identification, min_reversal_events=2)
        experiment = dataclasses.replace(
            base_experiment,
            initial_settle_s=0.1,
            inter_profile_settle_s=0.1,
            return_duration_s=0.1,
            profiles=profiles,
            identification=identification,
        )
        joint_name = "R_Leg_knee"
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "synthetic.npz"
            bus = SyntheticMx2Bus(mapping, experiment.sample_hz)
            collect_dataset(
                bus=bus,
                mapping=mapping,
                experiment=experiment,
                urdf_path=URDF_PATH,
                excitation_joint_names=(joint_name,),
                profiles=profiles,
                output_path=dataset_path,
                dry_run=True,
                confirm_arm=lambda _: True,
            )
            dataset = load_dataset(dataset_path)
            self.assertEqual(dataset.metadata["status"], "complete")
            self.assertGreater(dataset.sample_count, 1000)
            runtime = dataset.metadata["motor_runtime"][joint_name]
            self.assertEqual(runtime["position_p_gain"], 850)
            self.assertEqual(runtime["pwm_limit_raw"], 885)
            self.assertEqual(runtime["current_limit_raw"], 2047)
            reference_metadata = dataset.metadata["reference_pose"]
            transition = reference_metadata["transition"]
            self.assertGreater(transition["move_steps"], 0)
            self.assertLessEqual(
                max(abs(error) for error in transition["final_errors_rad"].values()),
                experiment.reference_pose.tolerance_rad,
            )
            for index, name in enumerate(dataset.joint_names):
                self.assertAlmostEqual(
                    dataset.arrays["goal_position_rad"][0, index],
                    experiment.reference_pose.positions_rad[name],
                )
            joint_index = dataset.joint_names.index(joint_name)
            precondition = (dataset.arrays["phase_id"] == -1) & (
                dataset.arrays["excitation_joint_id"] == joint_index
            )
            precondition_motion = np.abs(
                dataset.arrays["goal_position_rad"][:, joint_index]
                - experiment.reference_pose.positions_rad[joint_name]
            )
            self.assertTrue(np.any(precondition & (precondition_motion > 0.0)))
            self.assertFalse(any(bus.torque_enabled.values()))

            model = identify_dataset(dataset, identification)
            joint_model = model["joints"][joint_name]
            self.assertEqual(model["schema_version"], 2)
            self.assertEqual(model["reference_pose"], dataset.metadata["reference_pose"])
            self.assertAlmostEqual(model["timing"]["nominal_sample_period_s"], 0.02)
            self.assertTrue(joint_model["quality"]["target_randomization_usable"])
            self.assertGreaterEqual(joint_model["command_delay"]["seconds"], 0.0)
            self.assertLessEqual(joint_model["command_delay"]["seconds"], identification.max_delay_s)
            self.assertGreater(joint_model["effective_backlash"]["median_rad"], 0.0)
            self.assertLess(joint_model["effective_backlash"]["median_rad"], math.radians(5.0))

            # USB read-completion jitter must not replace the fixed 50 Hz command time base.
            jittered_arrays = dict(dataset.arrays)
            intervals = np.resize(np.asarray([0.016, 0.016, 0.016, 0.032]), dataset.sample_count - 1)
            jittered_arrays["time_s"] = np.concatenate((np.zeros(1), np.cumsum(intervals)))
            jittered_dataset = dataclasses.replace(dataset, arrays=jittered_arrays)
            jittered_model = identify_dataset(jittered_dataset, identification)
            jittered_joint = jittered_model["joints"][joint_name]
            self.assertAlmostEqual(jittered_model["timing"]["host_read_completion_dt_s"]["median"], 0.016)
            self.assertAlmostEqual(jittered_joint["command_delay"]["seconds"], joint_model["command_delay"]["seconds"])
            self.assertAlmostEqual(
                jittered_joint["effective_backlash"]["median_rad"],
                joint_model["effective_backlash"]["median_rad"],
            )

            randomizer = EncoderDomainActuatorRandomizer(model, (joint_name,), control_hz=50.0, seed=7)
            randomizer.reset(np.zeros(1))
            first = randomizer.transform(np.asarray([0.1]))
            self.assertEqual(first.shape, (1,))
            self.assertTrue(np.all(np.isfinite(first)))
            friction = randomizer.friction_torque_proxy(np.asarray([0.2]))
            self.assertLessEqual(float(friction[0]), 0.0)

    def test_reference_pose_failure_disables_all_torque(self):
        mapping = load_mapping_config(MAPPING_PATH)
        base_experiment = load_experiment_config(EXPERIMENT_PATH)
        reference_pose = dataclasses.replace(base_experiment.reference_pose, tolerance_rad=1.0e-8)
        experiment = dataclasses.replace(base_experiment, reference_pose=reference_pose)
        bus = SyntheticMx2Bus(mapping, experiment.sample_hz)
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "must_not_exist.npz"
            with self.assertRaisesRegex(SafetyTrip, "Reference pose did not settle"):
                collect_dataset(
                    bus=bus,
                    mapping=mapping,
                    experiment=experiment,
                    urdf_path=URDF_PATH,
                    excitation_joint_names=("R_Leg_knee",),
                    profiles=experiment.profiles,
                    output_path=dataset_path,
                    dry_run=True,
                    confirm_arm=lambda _: True,
                )
            self.assertFalse(any(bus.torque_enabled.values()))
            self.assertFalse(dataset_path.exists())


if __name__ == "__main__":
    unittest.main()
