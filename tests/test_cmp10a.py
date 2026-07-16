from __future__ import annotations

import math
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from cmp10a import (
    CMP10AFrame,
    CMP10AParser,
    CMP10AParserCounters,
    CMP10AProtocolError,
    CMP10ASerialError,
    CMP10ASerialReader,
    compute_checksum,
    decode_frame,
)


def _frame(frame_type: int, payload: bytes | None = None) -> bytes:
    payload = bytes(8) if payload is None else payload
    if len(payload) != 8:
        raise ValueError("test payload must be eight bytes")
    body = bytes((0x55, frame_type)) + payload
    return body + bytes((compute_checksum(body),))


class CMP10ADecodeTest(unittest.TestCase):
    def test_acceleration_is_decoded_to_meters_per_second_squared(self):
        raw = _frame(0x51, struct.pack("<hhhh", 16384, -16384, 8192, 0))
        frame = decode_frame(raw, timestamp_ns=123)

        self.assertIsInstance(frame, CMP10AFrame)
        self.assertEqual(frame.frame_type, 0x51)
        self.assertEqual(frame.timestamp_ns, 123)
        self.assertEqual(frame.raw, raw)
        self.assertAlmostEqual(frame.values["x_m_s2"], 8.0 * 9.80665)
        self.assertAlmostEqual(frame.values["y_m_s2"], -8.0 * 9.80665)
        self.assertAlmostEqual(frame.values["z_m_s2"], 4.0 * 9.80665)

    def test_gyro_is_decoded_to_radians_per_second(self):
        frame = decode_frame(_frame(0x52, struct.pack("<hhhh", 16384, -8192, 0, 0)), timestamp_ns=1)

        self.assertAlmostEqual(frame.values["x_rad_s"], math.radians(1000.0))
        self.assertAlmostEqual(frame.values["y_rad_s"], math.radians(-500.0))
        self.assertAlmostEqual(frame.values["z_rad_s"], 0.0)

    def test_euler_angles_are_decoded_to_radians(self):
        frame = decode_frame(_frame(0x53, struct.pack("<hhhh", 16384, -16384, 8192, 0)), timestamp_ns=1)

        self.assertAlmostEqual(frame.values["roll_rad"], math.pi / 2.0)
        self.assertAlmostEqual(frame.values["pitch_rad"], -math.pi / 2.0)
        self.assertAlmostEqual(frame.values["yaw_rad"], math.pi / 4.0)

    def test_magnetometer_preserves_signed_raw_values(self):
        frame = decode_frame(_frame(0x54, struct.pack("<hhhh", -123, 456, -32768, 0)), timestamp_ns=1)

        self.assertEqual(frame.values, {"x_raw": -123, "y_raw": 456, "z_raw": -32768})

    def test_quaternion_is_decoded_in_q0_through_q3_order(self):
        frame = decode_frame(_frame(0x59, struct.pack("<hhhh", 16384, -16384, 8192, -8192)), timestamp_ns=1)

        self.assertEqual(frame.values, {"q0": 0.5, "q1": -0.5, "q2": 0.25, "q3": -0.25})

    def test_other_protocol_frame_types_are_retained_without_interpretation(self):
        for frame_type in (*range(0x50, 0x5B), 0x5F):
            with self.subTest(frame_type=frame_type):
                frame = decode_frame(_frame(frame_type), timestamp_ns=1)
                self.assertEqual(frame.frame_type, frame_type)
                if frame_type not in (0x51, 0x52, 0x53, 0x54, 0x59):
                    self.assertEqual(frame.values, {})

    def test_invalid_frames_are_rejected(self):
        valid = _frame(0x51)
        cases = {
            "length": valid[:-1],
            "header": bytes((0x54,)) + valid[1:],
            "type": bytes((0x55, 0x4F)) + valid[2:],
            "checksum": valid[:-1] + bytes(((valid[-1] + 1) & 0xFF,)),
        }
        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(CMP10AProtocolError):
                decode_frame(raw, timestamp_ns=1)


class CMP10AParserTest(unittest.TestCase):
    def test_feed_timestamps_frames_when_their_final_bytes_arrive(self):
        acceleration = _frame(0x51, struct.pack("<hhhh", 1, 2, 3, 0))
        quaternion = _frame(0x59, struct.pack("<hhhh", 4, 5, 6, 7))
        parser = CMP10AParser()

        self.assertEqual(parser.feed(acceleration[:4], timestamp_ns=100), [])
        frames = parser.feed(acceleration[4:] + quaternion, timestamp_ns=200)

        self.assertEqual([frame.timestamp_ns for frame in frames], [200, 200])
        self.assertEqual([frame.frame_type for frame in frames], [0x51, 0x59])
        self.assertEqual(parser.buffered_bytes, 0)
        self.assertEqual(parser.valid_frames, 2)

    def test_parser_recovers_after_garbage_and_checksum_failure(self):
        damaged = bytearray(_frame(0x51))
        damaged[-1] ^= 0x01
        valid = _frame(0x53, struct.pack("<hhhh", 10, 20, 30, 0))
        parser = CMP10AParser()

        frames = parser.feed(b"\x00\xfe" + bytes(damaged) + valid, timestamp_ns=500)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_type, 0x53)
        self.assertEqual(parser.valid_frames, 1)
        self.assertEqual(parser.checksum_failures, 1)
        self.assertEqual(parser.garbage_bytes, 13)
        self.assertEqual(parser.counters, CMP10AParserCounters(1, 1, 13))

    def test_parser_recovers_from_a_header_with_an_invalid_type(self):
        parser = CMP10AParser()
        valid = _frame(0x54, struct.pack("<hhhh", 1, 2, 3, 0))

        frames = parser.feed(b"\x55\x4f" + valid, timestamp_ns=9)

        self.assertEqual([frame.frame_type for frame in frames], [0x54])
        self.assertEqual(parser.counters, CMP10AParserCounters(valid_frames=1, checksum_failures=0, garbage_bytes=2))

    def test_checksum_resynchronization_preserves_an_embedded_later_header(self):
        valid = _frame(0x59, struct.pack("<hhhh", 1, 2, 3, 4))
        parser = CMP10AParser()

        frames = parser.feed(b"\x55\x51\x00" + valid, timestamp_ns=77)

        self.assertEqual([frame.raw for frame in frames], [valid])
        self.assertEqual(parser.checksum_failures, 1)
        self.assertEqual(parser.garbage_bytes, 3)

    def test_buffered_partial_frame_can_be_discarded_between_capture_stages(self):
        parser = CMP10AParser()
        parser.feed(_frame(0x52)[:7], timestamp_ns=1)

        self.assertEqual(parser.discard_buffered_bytes(), 7)
        self.assertEqual(parser.buffered_bytes, 0)
        self.assertEqual(parser.garbage_bytes, 7)
        self.assertEqual(parser.discard_buffered_bytes(), 0)


class CMP10ASerialReaderTest(unittest.TestCase):
    def test_context_manager_opens_lazily_and_only_reads(self):
        class FakeSerialPort:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.read_sizes = []
                self.reset_input_buffer_calls = 0
                self.write_calls = 0
                self.closed = False
                self.__class__.instances.append(self)

            def read(self, size):
                self.read_sizes.append(size)
                return b"sample"[:size]

            def write(self, data):
                self.write_calls += 1
                raise AssertionError(f"unexpected serial write: {data!r}")

            def reset_input_buffer(self):
                self.reset_input_buffer_calls += 1

            def close(self):
                self.closed = True

        serial_module = types.ModuleType("serial")
        serial_module.Serial = FakeSerialPort
        reader = CMP10ASerialReader("/dev/ttyUSB9", baudrate=115200, timeout=0.25)
        self.assertFalse(reader.is_open)
        self.assertEqual(FakeSerialPort.instances, [])

        with mock.patch.dict(sys.modules, {"serial": serial_module}):
            with reader as opened:
                self.assertIs(opened, reader)
                self.assertTrue(reader.is_open)
                self.assertEqual(reader.read(3), b"sam")
                reader.reset_input_buffer()

        port = FakeSerialPort.instances[0]
        self.assertEqual(port.kwargs, {"port": "/dev/ttyUSB9", "baudrate": 115200, "timeout": 0.25})
        self.assertEqual(port.read_sizes, [3])
        self.assertEqual(port.reset_input_buffer_calls, 1)
        self.assertEqual(port.write_calls, 0)
        self.assertTrue(port.closed)
        self.assertFalse(reader.is_open)

    def test_read_requires_an_open_source(self):
        with self.assertRaisesRegex(CMP10ASerialError, "not open"):
            CMP10ASerialReader("/dev/ttyUSB9").read(1)

    def test_reset_input_buffer_requires_an_open_source(self):
        with self.assertRaisesRegex(CMP10ASerialError, "not open"):
            CMP10ASerialReader("/dev/ttyUSB9").reset_input_buffer()


if __name__ == "__main__":
    unittest.main()
