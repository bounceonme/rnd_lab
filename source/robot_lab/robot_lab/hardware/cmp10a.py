"""Pure CMP10A frame decoding, stream parsing, and read-only serial input."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


CMP10A_HEADER = 0x55
CMP10A_FRAME_SIZE = 11
CMP10A_FRAME_TYPES = frozenset((*range(0x50, 0x5B), 0x5F))

_HEADER_BYTES = bytes((CMP10A_HEADER,))


class CMP10AProtocolError(ValueError):
    """Raised when a complete byte sequence is not a valid CMP10A frame."""


class CMP10ASerialError(RuntimeError):
    """Raised when the optional read-only serial source cannot be used."""


@dataclass(frozen=True)
class CMP10AFrame:
    """A decoded CMP10A frame.

    ``values`` uses these keys for the decoded frame types:

    - ``0x51``: ``x_m_s2``, ``y_m_s2``, ``z_m_s2``
    - ``0x52``: ``x_rad_s``, ``y_rad_s``, ``z_rad_s``
    - ``0x53``: ``roll_rad``, ``pitch_rad``, ``yaw_rad``
    - ``0x54``: ``x_raw``, ``y_raw``, ``z_raw``
    - ``0x59``: ``q0``, ``q1``, ``q2``, ``q3``

    Other valid frame types retain their complete bytes in ``raw`` and have an
    empty ``values`` dictionary.
    """

    frame_type: int
    timestamp_ns: int
    raw: bytes
    values: dict[str, float | int]


@dataclass(frozen=True)
class CMP10AParserCounters:
    """Cumulative stream parser counters."""

    valid_frames: int
    checksum_failures: int
    garbage_bytes: int


def compute_checksum(data: bytes) -> int:
    """Return the CMP10A modulo-256 checksum for ``data``."""
    return sum(data) & 0xFF


def _signed_int16_values(payload: bytes, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}h", payload)


def _decode_values(frame_type: int, payload: bytes) -> dict[str, float | int]:
    if frame_type == 0x51:
        x_raw, y_raw, z_raw = _signed_int16_values(payload, 3)
        scale = 16.0 * 9.80665 / 32768.0
        return {"x_m_s2": x_raw * scale, "y_m_s2": y_raw * scale, "z_m_s2": z_raw * scale}

    if frame_type == 0x52:
        x_raw, y_raw, z_raw = _signed_int16_values(payload, 3)
        scale = 2000.0 * math.pi / (180.0 * 32768.0)
        return {"x_rad_s": x_raw * scale, "y_rad_s": y_raw * scale, "z_rad_s": z_raw * scale}

    if frame_type == 0x53:
        roll_raw, pitch_raw, yaw_raw = _signed_int16_values(payload, 3)
        scale = math.pi / 32768.0
        return {
            "roll_rad": roll_raw * scale,
            "pitch_rad": pitch_raw * scale,
            "yaw_rad": yaw_raw * scale,
        }

    if frame_type == 0x54:
        x_raw, y_raw, z_raw = _signed_int16_values(payload, 3)
        return {"x_raw": x_raw, "y_raw": y_raw, "z_raw": z_raw}

    if frame_type == 0x59:
        q0_raw, q1_raw, q2_raw, q3_raw = _signed_int16_values(payload, 4)
        scale = 1.0 / 32768.0
        return {
            "q0": q0_raw * scale,
            "q1": q1_raw * scale,
            "q2": q2_raw * scale,
            "q3": q3_raw * scale,
        }

    return {}


def decode_frame(raw: bytes, timestamp_ns: int) -> CMP10AFrame:
    """Validate and decode one complete CMP10A frame."""
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes.")
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
        raise TypeError("timestamp_ns must be an integer.")
    if len(raw) != CMP10A_FRAME_SIZE:
        raise CMP10AProtocolError(f"CMP10A frames must be {CMP10A_FRAME_SIZE} bytes, got {len(raw)}.")
    if raw[0] != CMP10A_HEADER:
        raise CMP10AProtocolError(f"Invalid CMP10A header 0x{raw[0]:02X}; expected 0x{CMP10A_HEADER:02X}.")
    if raw[1] not in CMP10A_FRAME_TYPES:
        raise CMP10AProtocolError(f"Unsupported CMP10A frame type 0x{raw[1]:02X}.")

    expected_checksum = compute_checksum(raw[: CMP10A_FRAME_SIZE - 1])
    if raw[-1] != expected_checksum:
        raise CMP10AProtocolError(
            f"CMP10A checksum mismatch: received 0x{raw[-1]:02X}, expected 0x{expected_checksum:02X}."
        )

    return CMP10AFrame(
        frame_type=raw[1],
        timestamp_ns=timestamp_ns,
        raw=raw,
        values=_decode_values(raw[1], raw[2:10]),
    )


class CMP10AParser:
    """Incrementally parse fixed-size CMP10A frames from arbitrary byte chunks."""

    def __init__(self):
        self._buffer = bytearray()
        self._valid_frames = 0
        self._checksum_failures = 0
        self._garbage_bytes = 0

    @property
    def valid_frames(self) -> int:
        return self._valid_frames

    @property
    def checksum_failures(self) -> int:
        return self._checksum_failures

    @property
    def garbage_bytes(self) -> int:
        return self._garbage_bytes

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    @property
    def counters(self) -> CMP10AParserCounters:
        return CMP10AParserCounters(
            valid_frames=self._valid_frames,
            checksum_failures=self._checksum_failures,
            garbage_bytes=self._garbage_bytes,
        )

    def _discard(self, count: int):
        del self._buffer[:count]
        self._garbage_bytes += count

    def discard_buffered_bytes(self) -> int:
        """Discard an incomplete frame before a new labeled capture stage."""
        count = len(self._buffer)
        self._discard(count)
        return count

    def feed(self, data: bytes, timestamp_ns: int) -> list[CMP10AFrame]:
        """Feed bytes and timestamp every frame completed by this call.

        ``timestamp_ns`` should be the host monotonic timestamp captured when
        ``data`` was supplied. A frame split across calls receives the timestamp
        from the call that supplies its final bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes.")
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise TypeError("timestamp_ns must be an integer.")

        self._buffer.extend(data)
        frames = []

        while self._buffer:
            header_index = self._buffer.find(_HEADER_BYTES)
            if header_index < 0:
                self._discard(len(self._buffer))
                break
            if header_index > 0:
                self._discard(header_index)

            if len(self._buffer) < 2:
                break
            if self._buffer[1] not in CMP10A_FRAME_TYPES:
                self._discard(1)
                continue
            if len(self._buffer) < CMP10A_FRAME_SIZE:
                break

            candidate = bytes(self._buffer[:CMP10A_FRAME_SIZE])
            if candidate[-1] != compute_checksum(candidate[:-1]):
                self._checksum_failures += 1
                self._discard(1)
                continue

            del self._buffer[:CMP10A_FRAME_SIZE]
            frames.append(decode_frame(candidate, timestamp_ns))
            self._valid_frames += 1

        return frames


class CMP10ASerialReader:
    """Lazy pyserial byte source with a deliberately read-only public API.

    Opening the port configures only the local serial line parameters required
    for reading. This class never writes protocol bytes or sends sensor setting
    commands.
    """

    def __init__(self, device: str, *, baudrate: int = 9600, timeout: float | None = 0.1):
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty serial-device path.")
        if not isinstance(baudrate, int) or isinstance(baudrate, bool) or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer.")
        if timeout is not None and (not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout < 0):
            raise ValueError("timeout must be None or a non-negative number of seconds.")

        self.device = device
        self.baudrate = baudrate
        self.timeout = None if timeout is None else float(timeout)
        self._port: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._port is not None

    def open(self) -> CMP10ASerialReader:
        if self._port is not None:
            return self

        try:
            import serial
        except ImportError as error:
            raise CMP10ASerialError("pyserial is required to open a CMP10A serial source.") from error

        self._port = serial.Serial(port=self.device, baudrate=self.baudrate, timeout=self.timeout)
        return self

    def read(self, size: int) -> bytes:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("size must be a non-negative integer.")
        if self._port is None:
            raise CMP10ASerialError("CMP10A serial source is not open.")
        return bytes(self._port.read(size))

    def reset_input_buffer(self):
        """Discard bytes queued in pyserial's local input buffer without writing."""
        if self._port is None:
            raise CMP10ASerialError("CMP10A serial source is not open.")
        self._port.reset_input_buffer()

    def close(self):
        port = self._port
        self._port = None
        if port is not None:
            port.close()

    def __enter__(self) -> CMP10ASerialReader:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "CMP10A_FRAME_SIZE",
    "CMP10A_FRAME_TYPES",
    "CMP10A_HEADER",
    "CMP10AFrame",
    "CMP10AParser",
    "CMP10AParserCounters",
    "CMP10AProtocolError",
    "CMP10ASerialError",
    "CMP10ASerialReader",
    "compute_checksum",
    "decode_frame",
]
