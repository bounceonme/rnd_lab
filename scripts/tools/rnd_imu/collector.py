"""Read-only serial probing and staged CMP10A frame collection."""

from __future__ import annotations

import collections
import dataclasses
import glob
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HARDWARE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(_HARDWARE_DIR))

from cmp10a import CMP10AFrame, CMP10AParser, CMP10ASerialReader


class ImuCollectionError(RuntimeError):
    """Raised when a read-only CMP10A collection cannot proceed."""


@dataclasses.dataclass(frozen=True)
class BaudProbeResult:
    baudrate: int
    valid_frames: int
    checksum_failures: int
    garbage_bytes: int
    frame_type_counts: dict[int, int]


def discover_serial_ports() -> tuple[str, ...]:
    """Return stable serial paths first, without opening any device."""

    candidates = []
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        candidates.extend(sorted(glob.glob(pattern)))

    unique: list[str] = []
    targets: set[Path] = set()
    for candidate in candidates:
        path = Path(candidate)
        try:
            target = path.resolve(strict=True)
        except OSError:
            continue
        if target in targets:
            continue
        targets.add(target)
        unique.append(str(path))
    return tuple(unique)


def resolve_serial_port(requested: str) -> str:
    """Resolve ``auto`` only when exactly one serial device exists."""

    if requested != "auto":
        return requested
    ports = discover_serial_ports()
    if not ports:
        raise ImuCollectionError("No /dev/serial/by-id, ttyUSB, or ttyACM device was found.")
    if len(ports) > 1:
        listed = "\n  ".join(ports)
        raise ImuCollectionError(f"Multiple serial devices were found; select the CMP10A with --port:\n  {listed}")
    return ports[0]


def _read_for_duration(reader, parser: CMP10AParser, duration_s: float) -> list[CMP10AFrame]:
    frames: list[CMP10AFrame] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        data = reader.read(11)
        timestamp_ns = time.monotonic_ns()
        if data:
            frames.extend(parser.feed(data, timestamp_ns))
    return frames


def probe_baudrates(
    device: str,
    baudrates: Iterable[int],
    duration_s: float,
    *,
    timeout_s: float,
    reader_factory: Callable[..., object] = CMP10ASerialReader,
) -> tuple[BaudProbeResult, ...]:
    """Try baud rates without writing commands to the sensor."""

    results = []
    for baudrate in baudrates:
        parser = CMP10AParser()
        try:
            with reader_factory(device, baudrate=baudrate, timeout=timeout_s) as reader:
                if hasattr(reader, "reset_input_buffer"):
                    reader.reset_input_buffer()
                frames = _read_for_duration(reader, parser, duration_s)
        except OSError as error:
            raise ImuCollectionError(f"Unable to read {device} at {baudrate} baud: {error}") from error
        type_counts = collections.Counter(frame.frame_type for frame in frames)
        results.append(
            BaudProbeResult(
                baudrate=int(baudrate),
                valid_frames=parser.valid_frames,
                checksum_failures=parser.checksum_failures,
                garbage_bytes=parser.garbage_bytes,
                frame_type_counts=dict(sorted(type_counts.items())),
            )
        )
    return tuple(results)


def select_probe_result(results: Iterable[BaudProbeResult], minimum_valid_frames: int) -> BaudProbeResult:
    """Select the unambiguous baud result with the most valid frames."""

    eligible = [result for result in results if result.valid_frames >= minimum_valid_frames]
    if not eligible:
        raise ImuCollectionError(
            "No baud rate produced enough checksum-valid CMP10A frames. Check the port, power, and interface mode."
        )
    eligible.sort(key=lambda result: (result.valid_frames, -result.checksum_failures), reverse=True)
    best = eligible[0]
    if len(eligible) > 1 and eligible[1].valid_frames >= 0.8 * best.valid_frames:
        raise ImuCollectionError(
            f"Baud detection is ambiguous between {best.baudrate} and {eligible[1].baudrate}; pass --baud explicitly."
        )
    return best


def frame_to_record(frame: CMP10AFrame, stage: str) -> dict:
    """Convert a protocol frame to the stable NPZ row schema."""

    values = frame.values
    record = {
        "timestamp_ns": frame.timestamp_ns,
        "stage": stage,
        "frame_type": frame.frame_type,
        "raw": frame.raw,
    }
    if frame.frame_type == 0x51:
        record["accel_mps2"] = [values["x_m_s2"], values["y_m_s2"], values["z_m_s2"]]
    elif frame.frame_type == 0x52:
        record["gyro_rad_s"] = [values["x_rad_s"], values["y_rad_s"], values["z_rad_s"]]
    elif frame.frame_type == 0x53:
        record["euler_rad"] = [values["roll_rad"], values["pitch_rad"], values["yaw_rad"]]
    elif frame.frame_type == 0x54:
        record["mag_raw"] = [values["x_raw"], values["y_raw"], values["z_raw"]]
    elif frame.frame_type == 0x59:
        record["quat_wxyz"] = [values["q0"], values["q1"], values["q2"], values["q3"]]
    return record


def collect_stage(reader, parser: CMP10AParser, stage: str, duration_s: float) -> list[dict]:
    """Discard stale input and collect one uninterrupted labeled stage."""

    if hasattr(reader, "reset_input_buffer"):
        reader.reset_input_buffer()
    parser.discard_buffered_bytes()
    start = time.monotonic()
    frames = _read_for_duration(reader, parser, duration_s)
    elapsed = time.monotonic() - start
    if not frames:
        raise ImuCollectionError(f"No valid CMP10A frames arrived during {stage} ({elapsed:.1f}s).")
    return [frame_to_record(frame, stage) for frame in frames]


def build_oscillation_cues(
    *,
    neutral_duration_s: float,
    half_cycle_s: float,
    cycles: int,
    positive_label: str,
    negative_label: str,
) -> tuple[tuple[float, str], ...]:
    """Build target-change cues for a centered bidirectional oscillation."""

    if neutral_duration_s < 0.0 or half_cycle_s <= 0.0 or cycles < 1:
        raise ValueError("Oscillation timing requires neutral>=0, half_cycle>0, and cycles>=1.")
    cues: list[tuple[float, str]] = [(0.0, "HOLD CENTER")]
    for index in range(2 * cycles):
        label = positive_label if index % 2 == 0 else negative_label
        cues.append((neutral_duration_s + index * half_cycle_s, label))
    cues.append((neutral_duration_s + 2 * cycles * half_cycle_s, "RETURN TO CENTER"))
    return tuple(cues)


def collect_cued_stage(
    reader,
    parser: CMP10AParser,
    stage: str,
    duration_s: float,
    cues: Iterable[tuple[float, str]],
    *,
    cue_callback: Callable[[str], None] = print,
    monotonic: Callable[[], float] = time.monotonic,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> list[dict]:
    """Collect one stage while emitting non-blocking target-change cues."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    ordered_cues = tuple(sorted((float(offset), str(label)) for offset, label in cues))
    if any(offset < 0.0 or offset > duration_s for offset, _ in ordered_cues):
        raise ValueError("Every cue offset must lie inside the capture duration.")

    if hasattr(reader, "reset_input_buffer"):
        reader.reset_input_buffer()
    parser.discard_buffered_bytes()
    frames: list[CMP10AFrame] = []
    start = monotonic()
    deadline = start + duration_s
    cue_index = 0
    while monotonic() < deadline:
        elapsed = monotonic() - start
        while cue_index < len(ordered_cues) and elapsed >= ordered_cues[cue_index][0]:
            cue_callback(ordered_cues[cue_index][1])
            cue_index += 1
        data = reader.read(11)
        timestamp_ns = monotonic_ns()
        if data:
            frames.extend(parser.feed(data, timestamp_ns))
    if not frames:
        raise ImuCollectionError(f"No valid CMP10A frames arrived during {stage} ({duration_s:.1f}s).")
    return [frame_to_record(frame, stage) for frame in frames]


def parser_stats(parser: CMP10AParser) -> dict[str, int]:
    """Return JSON-safe cumulative parser counters."""

    return {
        "valid_frames": parser.valid_frames,
        "checksum_failures": parser.checksum_failures,
        "garbage_bytes": parser.garbage_bytes,
    }
