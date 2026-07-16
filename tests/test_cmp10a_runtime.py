from __future__ import annotations

import copy
import json
import math
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from cmp10a import CMP10AFrame, compute_checksum, decode_frame
from cmp10a_runtime import (
    CMP10ARuntimeAdapter,
    CMP10ARuntimeFrameError,
    CMP10ARuntimeModelError,
    CMP10ARuntimeSnapshotError,
    CMP10ARuntimeSource,
    CMP10ARuntimeSourceError,
    load_cmp10a_runtime_model,
    validate_cmp10a_runtime_model,
)


MS = 1_000_000


def _model(**overrides) -> dict:
    model = {
        "schema_version": 1,
        "model_type": "rnd_cmp10a_policy_observation",
        "integration_enabled": True,
        "sensor_to_base_matrix": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        "gyro_bias_rad_s": [0.1, -0.2, 0.3],
        "policy_angular_velocity_scale": 0.25,
        "sensor_baudrate": 921600,
        "sensor_rates_hz": {"gyro": 200.0, "euler": 200.0},
        "policy_hz": 50.0,
        "quality_gates": {
            "promotion_pass": True,
            "static_runtime_gate_pass": True,
            "static_mount_axis_gate_pass": True,
            "dynamic_communication_gate_pass": True,
            "dynamic_consistency_gate_pass": True,
        },
        "runtime": {"max_frame_age_ms": 30.0, "max_pair_skew_ms": 6.0},
        "provenance": {"source_report": "fixture", "ignored_extra": [1, 2, 3]},
    }
    model.update(overrides)
    return model


def _raw_frame(frame_type: int, payload: bytes) -> bytes:
    if len(payload) != 8:
        raise ValueError("payload must be eight bytes")
    body = bytes((0x55, frame_type)) + payload
    return body + bytes((compute_checksum(body),))


def _gyro(timestamp_ns: int, raw_xyz: tuple[int, int, int]) -> CMP10AFrame:
    return decode_frame(_raw_frame(0x52, struct.pack("<hhhh", *raw_xyz, 0)), timestamp_ns)


def _euler(timestamp_ns: int, raw_rpy: tuple[int, int, int]) -> CMP10AFrame:
    return decode_frame(_raw_frame(0x53, struct.pack("<hhhh", *raw_rpy, 0)), timestamp_ns)


class CMP10ARuntimeModelTest(unittest.TestCase):
    def test_mapping_and_path_models_validate_with_extra_provenance(self):
        validate_cmp10a_runtime_model(_model())
        nested = _model()
        nested.pop("sensor_to_base_matrix")
        nested.pop("gyro_bias_rad_s")
        nested.pop("policy_angular_velocity_scale")
        nested.pop("sensor_baudrate")
        nested.pop("sensor_rates_hz")
        nested.pop("policy_hz")
        nested["sensor"] = {
            "sensor_to_base_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "gyro_bias_rad_s": [0.0, 0.0, 0.0],
            "baudrate": 921600,
            "gyro_rate_hz": 200.0,
            "euler_rate_hz": 200.0,
        }
        nested["policy"] = {"rate_hz": 50.0, "angular_velocity_scale": 0.25}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runtime.json"
            path.write_text(json.dumps(nested), encoding="utf-8")
            loaded = load_cmp10a_runtime_model(path)
        self.assertEqual(loaded["provenance"]["source_report"], "fixture")

    def test_promoted_runtime_contract_nesting_is_consumed(self):
        compact = _model()
        promoted = {
            "schema_version": compact["schema_version"],
            "model_type": compact["model_type"],
            "integration_enabled": compact["integration_enabled"],
            "quality_gates": compact["quality_gates"],
            "sensor_baudrate": compact["sensor_baudrate"],
            "runtime_transform": {"sensor_to_base_matrix": compact["sensor_to_base_matrix"]},
            "policy_observation": {
                "policy_hz": compact["policy_hz"],
                "angular_velocity_scale": compact["policy_angular_velocity_scale"],
            },
            "runtime_limits": {
                "maximum_sample_age_s": 0.03,
                "maximum_gyro_orientation_pair_skew_s": 0.006,
            },
            "measured": {
                "packet_rates": {
                    "gyro_rate_hz": compact["sensor_rates_hz"]["gyro"],
                    "orientation_rate_hz": compact["sensor_rates_hz"]["euler"],
                },
                "held_baseline": {"gyro": {"mean_rad_s": compact["gyro_bias_rad_s"]}},
            },
            "provenance_chain": {"ignored_by_runtime": True},
        }

        adapter = CMP10ARuntimeAdapter(promoted)

        self.assertEqual(adapter.config.sensor_baudrate, 921600)
        self.assertEqual(adapter.config.max_frame_age_ns, 30 * MS)
        self.assertEqual(adapter.config.max_pair_skew_ns, 6 * MS)

    def test_safety_critical_model_errors_are_clear(self):
        cases = []
        wrong_schema = _model(schema_version=True)
        cases.append(("schema", wrong_schema, "schema_version"))
        disabled = _model(integration_enabled=False)
        cases.append(("integration", disabled, "integration_enabled=true"))
        reflection = _model(sensor_to_base_matrix=[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cases.append(("reflection", reflection, r"determinant \+1"))
        slow = _model(sensor_rates_hz={"gyro": 49.0, "euler": 200.0})
        cases.append(("rates", slow, "at least policy_hz"))
        missing_bias = _model()
        missing_bias.pop("gyro_bias_rad_s")
        cases.append(("bias", missing_bias, "gyro bias"))
        no_age = _model(runtime={"max_pair_skew_ms": 6.0})
        cases.append(("age", no_age, "maximum frame age"))
        no_skew = _model(runtime={"max_frame_age_ms": 30.0})
        cases.append(("skew", no_skew, "maximum gyro/Euler skew"))
        failed_gate = _model()
        failed_gate["quality_gates"]["dynamic_consistency_gate_pass"] = False
        cases.append(("gate", failed_gate, "dynamic_consistency_gate_pass"))
        duplicate_scale = _model()
        duplicate_scale["policy_observation"] = {"angular_velocity_scale": 0.5, "policy_hz": 50.0}
        cases.append(("duplicate scale", duplicate_scale, "scale fields disagree"))

        for name, model, pattern in cases:
            with self.subTest(name=name), self.assertRaisesRegex(CMP10ARuntimeModelError, pattern):
                validate_cmp10a_runtime_model(model)


class CMP10ARuntimeAdapterTest(unittest.TestCase):
    def test_known_bias_sign_and_zyx_orientation_transform(self):
        adapter = CMP10ARuntimeAdapter(_model())
        gyro = _gyro(100 * MS, (1200, -700, 350))
        euler = _euler(98 * MS, (5461, -3641, 12345))
        adapter.ingest(gyro)
        adapter.ingest(euler)

        sample = adapter.snapshot(105 * MS)

        omega_sensor = (
            gyro.values["x_rad_s"],
            gyro.values["y_rad_s"],
            gyro.values["z_rad_s"],
        )
        expected_omega_base = (
            -(omega_sensor[0] - 0.1),
            -(omega_sensor[1] + 0.2),
            omega_sensor[2] - 0.3,
        )
        for actual, expected in zip(sample.base_angular_velocity_rad_s, expected_omega_base):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(sample.policy_angular_velocity, expected_omega_base):
            self.assertAlmostEqual(actual, 0.25 * expected, places=12)

        roll = euler.values["roll_rad"]
        pitch = euler.values["pitch_rad"]
        gravity_sensor = (
            math.sin(pitch),
            -math.sin(roll) * math.cos(pitch),
            -math.cos(roll) * math.cos(pitch),
        )
        expected_gravity_base = (-gravity_sensor[0], -gravity_sensor[1], gravity_sensor[2])
        for actual, expected in zip(sample.projected_gravity_b, expected_gravity_base):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in sample.projected_gravity_b)), 1.0, places=12)
        self.assertEqual(sample.gyro_age_ns, 5 * MS)
        self.assertEqual(sample.euler_age_ns, 7 * MS)
        self.assertEqual(sample.pair_skew_ns, 2 * MS)

    def test_each_50_hz_snapshot_uses_latest_200_hz_pair_without_averaging(self):
        model = _model(gyro_bias_rad_s=[0.0, 0.0, 0.0])
        adapter = CMP10ARuntimeAdapter(model)
        latest_gyro = None
        for policy_phase in (20, 40):
            for timestamp_ms in range(policy_phase - 20, policy_phase + 1, 5):
                latest_gyro = _gyro(timestamp_ms * MS, (timestamp_ms * 20, 0, 0))
                adapter.ingest(latest_gyro)
                adapter.ingest(_euler(timestamp_ms * MS, (timestamp_ms, -timestamp_ms, 0)))
            sample = adapter.snapshot(policy_phase * MS)
            self.assertEqual(sample.gyro_timestamp_ns, policy_phase * MS)
            self.assertEqual(sample.euler_timestamp_ns, policy_phase * MS)
            self.assertAlmostEqual(sample.base_angular_velocity_rad_s[0], -latest_gyro.values["x_rad_s"])
        self.assertEqual(sample.counters.gyro_frames, 10)
        self.assertEqual(sample.counters.euler_frames, 10)

    def test_snapshot_fails_closed_for_missing_future_stale_and_skewed_pairs(self):
        with self.assertRaisesRegex(CMP10ARuntimeSnapshotError, "missing latest"):
            CMP10ARuntimeAdapter(_model()).snapshot(100 * MS)

        cases = (
            ("future", 101 * MS, 100 * MS, 100 * MS, "future"),
            ("stale", 60 * MS, 60 * MS, 100 * MS, "stale"),
            ("skew", 100 * MS, 90 * MS, 100 * MS, "skew"),
        )
        for name, gyro_ns, euler_ns, now_ns, pattern in cases:
            adapter = CMP10ARuntimeAdapter(_model())
            adapter.ingest(_gyro(gyro_ns, (1, 2, 3)))
            adapter.ingest(_euler(euler_ns, (1, 2, 3)))
            with self.subTest(name=name), self.assertRaisesRegex(CMP10ARuntimeSnapshotError, pattern):
                adapter.snapshot(now_ns)

    def test_direct_ingest_revalidates_checksum_and_ignores_older_frames(self):
        adapter = CMP10ARuntimeAdapter(_model())
        latest = _gyro(10 * MS, (100, 0, 0))
        adapter.ingest(latest)
        self.assertFalse(adapter.ingest(_gyro(5 * MS, (200, 0, 0))))
        adapter.ingest(_euler(10 * MS, (0, 0, 0)))
        sample = adapter.snapshot(10 * MS)
        self.assertAlmostEqual(sample.base_angular_velocity_rad_s[0], -(latest.values["x_rad_s"] - 0.1))
        self.assertEqual(sample.counters.out_of_order_frames, 1)

        damaged = bytearray(latest.raw)
        damaged[-1] ^= 1
        forged = CMP10AFrame(latest.frame_type, latest.timestamp_ns, bytes(damaged), copy.deepcopy(latest.values))
        with self.assertRaisesRegex(CMP10ARuntimeFrameError, "checksum"):
            adapter.ingest(forged)


class _FakeReader:
    def __init__(self, chunks: list[bytes], drained: threading.Event, **kwargs):
        self.chunks = list(chunks)
        self.drained = drained
        self.kwargs = kwargs
        self.entered = False
        self.closed = False
        self.read_sizes: list[int] = []
        self.write_calls = 0
        self.reset_input_buffer_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self.chunks:
            return self.chunks.pop(0)
        self.drained.set()
        time.sleep(0.001)
        return b""

    def write(self, data: bytes) -> int:
        self.write_calls += 1
        raise AssertionError(f"runtime source attempted a write: {data!r}")

    def reset_input_buffer(self) -> None:
        self.reset_input_buffer_calls += 1


class CMP10ARuntimeSourceTest(unittest.TestCase):
    def test_fake_reader_lifecycle_parser_counters_and_read_only_snapshot(self):
        drained = threading.Event()
        now_ns = 100 * MS
        chunk = _gyro(now_ns, (10, 20, 30)).raw + _euler(now_ns, (0, 0, 0)).raw
        readers: list[_FakeReader] = []

        def reader_factory(device: str, **kwargs) -> _FakeReader:
            reader = _FakeReader([chunk], drained, device=device, **kwargs)
            readers.append(reader)
            return reader

        source = CMP10ARuntimeSource(
            "/dev/fake-cmp10a",
            _model(),
            monotonic_ns=lambda: now_ns,
            reader_factory=reader_factory,
        )
        with source as opened:
            self.assertIs(opened, source)
            self.assertTrue(source.thread_is_daemon)
            self.assertTrue(drained.wait(1.0))
            sample = source.snapshot()
            self.assertEqual(sample.parser_counters.valid_frames, 2)
            self.assertEqual(source.parser_counters.valid_frames, 2)
            self.assertEqual(source.counters.gyro_frames, 1)
            self.assertEqual(source.counters.euler_frames, 1)
        reader = readers[0]
        self.assertEqual(
            reader.kwargs,
            {"device": "/dev/fake-cmp10a", "baudrate": 921600, "timeout": 0.01},
        )
        self.assertTrue(reader.entered)
        self.assertTrue(reader.closed)
        self.assertEqual(reader.write_calls, 0)
        self.assertEqual(reader.reset_input_buffer_calls, 1)
        self.assertTrue(reader.read_sizes)
        self.assertFalse(source.is_running)

    def test_background_reader_error_is_exposed_by_snapshot_and_close(self):
        release = threading.Event()
        failed = threading.Event()

        class FailingReader:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def read(self, size: int) -> bytes:
                release.wait(1.0)
                failed.set()
                raise OSError("synthetic read failure")

        source = CMP10ARuntimeSource("/dev/failing", _model(), reader_factory=FailingReader)
        source.start()
        release.set()
        self.assertTrue(failed.wait(1.0))
        deadline = time.monotonic() + 1.0
        while source.thread_error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        with self.assertRaisesRegex(CMP10ARuntimeSourceError, "synthetic read failure"):
            source.snapshot(100 * MS)
        with self.assertRaisesRegex(CMP10ARuntimeSourceError, "synthetic read failure"):
            source.close()


if __name__ == "__main__":
    unittest.main()
