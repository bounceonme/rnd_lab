from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
HARDWARE_DIR = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(HARDWARE_DIR))

from cmp10a import CMP10AParser, compute_checksum
from rnd_imu.collector import build_oscillation_cues, collect_cued_stage
from rnd_imu.dynamic_config import load_dynamic_imu_config


CONFIG_PATH = TOOLS_DIR / "config" / "rnd_cmp10a_dynamic.toml"


def _frame(frame_type: int, payload: bytes | None = None) -> bytes:
    payload = bytes(8) if payload is None else payload
    body = bytes((0x55, frame_type)) + payload
    return body + bytes((compute_checksum(body),))


class _FakeClock:
    def __init__(self, step_s: float = 0.01):
        self.now_s = 0.0
        self.step_s = step_s

    def monotonic(self) -> float:
        return self.now_s

    def monotonic_ns(self) -> int:
        return round(self.now_s * 1.0e9)

    def advance(self):
        self.now_s += self.step_s


class _ReadOnlyFakeReader:
    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.reset_input_buffer_calls = 0
        self.read_calls = 0

    def reset_input_buffer(self):
        self.reset_input_buffer_calls += 1

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        self.clock.advance()
        return _frame(0x52, struct.pack("<hhhh", 100, 0, 0, 0))[:size]

    def write(self, data: bytes):
        raise AssertionError(f"dynamic collection must never write to the sensor: {data!r}")


class RndImuDynamicCollectionTest(unittest.TestCase):
    def test_default_config_matches_guided_capture_timing(self):
        config = load_dynamic_imu_config(CONFIG_PATH)

        self.assertEqual(config.serial.baudrate, 921600)
        self.assertEqual(config.experiment.policy_hz, 50.0)
        self.assertEqual(config.experiment.cycles, 6)
        self.assertEqual(config.experiment.stage_duration_s, 16.0)

    def test_cues_cover_centered_bidirectional_oscillations(self):
        cues = build_oscillation_cues(
            neutral_duration_s=2.0,
            half_cycle_s=1.0,
            cycles=2,
            positive_label="POSITIVE",
            negative_label="NEGATIVE",
        )

        self.assertEqual(
            cues,
            (
                (0.0, "HOLD CENTER"),
                (2.0, "POSITIVE"),
                (3.0, "NEGATIVE"),
                (4.0, "POSITIVE"),
                (5.0, "NEGATIVE"),
                (6.0, "RETURN TO CENTER"),
            ),
        )

    def test_cued_collection_discards_stale_bytes_and_only_reads(self):
        clock = _FakeClock(step_s=0.01)
        reader = _ReadOnlyFakeReader(clock)
        parser = CMP10AParser()
        parser.feed(_frame(0x52)[:5], timestamp_ns=1)
        emitted_cues: list[str] = []

        records = collect_cued_stage(
            reader,
            parser,
            "dynamic_axis_x",
            0.06,
            ((0.0, "CENTER"), (0.02, "FORWARD"), (0.04, "BACKWARD")),
            cue_callback=emitted_cues.append,
            monotonic=clock.monotonic,
            monotonic_ns=clock.monotonic_ns,
        )

        self.assertEqual(reader.reset_input_buffer_calls, 1)
        self.assertGreater(reader.read_calls, 0)
        self.assertEqual(emitted_cues, ["CENTER", "FORWARD", "BACKWARD"])
        self.assertTrue(records)
        self.assertTrue(all(record["stage"] == "dynamic_axis_x" for record in records))
        self.assertTrue(all(record["frame_type"] == 0x52 for record in records))
        self.assertGreaterEqual(parser.garbage_bytes, 5)


if __name__ == "__main__":
    unittest.main()
