"""Parsing helpers for RobotLab / RSL-RL logs and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os
import re
from typing import Iterable, Iterator

DEFAULT_RSL_RL_LOG_ROOT = Path("/home/chae/robot_lab/logs/rsl_rl")

_ITERATION_RE = re.compile(r"^Learning iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\s*$")
_TIMESTEP_RE = re.compile(r"^Total timesteps:\s*(?P<timesteps>\d+)\s*$")
_ITERATION_TIME_RE = re.compile(r"^Iteration time:\s*(?P<seconds>\d+(?:\.\d+)?)s\s*$")
_TIME_ELAPSED_RE = re.compile(r"^Time elapsed:\s*(?P<value>[0-9:]+)\s*$")
_ETA_RE = re.compile(r"^ETA:\s*(?P<value>[0-9:]+)\s*$")
_KEY_VALUE_RE = re.compile(r"^(?P<label>[A-Za-z0-9_.:/ -]+):\s*(?P<value>.+?)\s*$")
_NUMERIC_VALUE_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
_CHECKPOINT_RE = re.compile(r"^model_(?P<step>\d+)\.pt$", re.IGNORECASE)


class ArtifactKind(str, Enum):
    """High-level file classification."""

    CHECKPOINT = "checkpoint"
    VIDEO = "video"
    TENSORBOARD_EVENT = "tensorboard_event"
    EXPORT = "export"
    PARAMS = "params"
    LOG = "log"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RslRlIterationStatus:
    """Structured view of the latest RSL-RL status block."""

    iteration: int
    total_iterations: int | None = None
    total_timesteps: int | None = None
    iteration_time_seconds: float | None = None
    time_elapsed: str | None = None
    eta: str | None = None
    metrics: dict[str, str | float] = field(default_factory=dict)
    raw_lines: tuple[str, ...] = ()
    line_number: int | None = None

    @property
    def progress_fraction(self) -> float | None:
        if not self.total_iterations:
            return None
        return min(max(self.iteration / self.total_iterations, 0.0), 1.0)

    def summary(self) -> str:
        parts = [f"iter {self.iteration}"]
        if self.total_iterations is not None:
            parts[0] += f"/{self.total_iterations}"
        if self.total_timesteps is not None:
            parts.append(f"timesteps {self.total_timesteps}")
        if self.iteration_time_seconds is not None:
            parts.append(f"iter_time {self.iteration_time_seconds:.3f}s")
        if self.time_elapsed:
            parts.append(f"elapsed {self.time_elapsed}")
        if self.eta:
            parts.append(f"eta {self.eta}")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class RobotLabLogSnapshot:
    """Snapshot of a parsed log tail."""

    source_path: Path
    status: RslRlIterationStatus | None
    tail_text: str
    bytes_read: int

    def summary(self) -> str:
        if self.status is None:
            return f"{self.source_path}: no RSL-RL status block found"
        return f"{self.source_path}: {self.status.summary()}"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Classification result for a file under the RobotLab log tree."""

    path: Path
    kind: ArtifactKind
    run_dir: Path | None = None
    checkpoint_step: int | None = None
    modified_time: float | None = None

    @property
    def is_checkpoint(self) -> bool:
        return self.kind == ArtifactKind.CHECKPOINT

    @property
    def is_video(self) -> bool:
        return self.kind == ArtifactKind.VIDEO


@dataclass(frozen=True, slots=True)
class DiscoveredArtifacts:
    """Latest artifacts found within a search scope."""

    scope: Path
    latest_checkpoint: ArtifactRecord | None = None
    latest_video: ArtifactRecord | None = None
    latest_export: ArtifactRecord | None = None
    latest_log: ArtifactRecord | None = None


def _parse_scalar(value: str) -> str | float:
    value = value.strip()
    if _NUMERIC_VALUE_RE.match(value):
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    return value


def parse_rsl_rl_line(line: str, *, line_number: int | None = None) -> RslRlIterationStatus | None:
    """Parse a single RSL-RL iteration header line."""

    match = _ITERATION_RE.match(line.strip())
    if match is None:
        return None
    return RslRlIterationStatus(
        iteration=int(match.group("iteration")),
        total_iterations=int(match.group("total")),
        raw_lines=(line.rstrip("\n"),),
        line_number=line_number,
    )


def parse_rsl_rl_iteration_line(line: str, *, line_number: int | None = None) -> RslRlIterationStatus | None:
    """Backward-compatible alias for older internal naming."""

    return parse_rsl_rl_line(line, line_number=line_number)


def _finalize_status(
    status: RslRlIterationStatus | None,
    metrics: dict[str, str | float],
    raw_lines: list[str],
) -> RslRlIterationStatus | None:
    if status is None:
        return None
    return RslRlIterationStatus(
        iteration=status.iteration,
        total_iterations=status.total_iterations,
        total_timesteps=status.total_timesteps,
        iteration_time_seconds=status.iteration_time_seconds,
        time_elapsed=status.time_elapsed,
        eta=status.eta,
        metrics=dict(metrics),
        raw_lines=tuple(raw_lines),
        line_number=status.line_number,
    )


def parse_rsl_rl_log(text: str) -> RslRlIterationStatus | None:
    """Parse a full RSL-RL log string and return the latest status block."""

    latest: RslRlIterationStatus | None = None
    current: RslRlIterationStatus | None = None
    current_metrics: dict[str, str | float] = {}
    current_lines: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        start = parse_rsl_rl_line(line, line_number=line_number)
        if start is not None:
            latest = _finalize_status(current, current_metrics, current_lines) or latest
            current = start
            current_metrics = {}
            current_lines = [line]
            continue

        if current is None:
            continue

        current_lines.append(line)

        if match := _TIMESTEP_RE.match(line):
            current = RslRlIterationStatus(
                iteration=current.iteration,
                total_iterations=current.total_iterations,
                total_timesteps=int(match.group("timesteps")),
                iteration_time_seconds=current.iteration_time_seconds,
                time_elapsed=current.time_elapsed,
                eta=current.eta,
                metrics=dict(current_metrics),
                raw_lines=tuple(current_lines),
                line_number=current.line_number,
            )
            continue

        if match := _ITERATION_TIME_RE.match(line):
            current = RslRlIterationStatus(
                iteration=current.iteration,
                total_iterations=current.total_iterations,
                total_timesteps=current.total_timesteps,
                iteration_time_seconds=float(match.group("seconds")),
                time_elapsed=current.time_elapsed,
                eta=current.eta,
                metrics=dict(current_metrics),
                raw_lines=tuple(current_lines),
                line_number=current.line_number,
            )
            continue

        if match := _TIME_ELAPSED_RE.match(line):
            current = RslRlIterationStatus(
                iteration=current.iteration,
                total_iterations=current.total_iterations,
                total_timesteps=current.total_timesteps,
                iteration_time_seconds=current.iteration_time_seconds,
                time_elapsed=match.group("value"),
                eta=current.eta,
                metrics=dict(current_metrics),
                raw_lines=tuple(current_lines),
                line_number=current.line_number,
            )
            continue

        if match := _ETA_RE.match(line):
            current = RslRlIterationStatus(
                iteration=current.iteration,
                total_iterations=current.total_iterations,
                total_timesteps=current.total_timesteps,
                iteration_time_seconds=current.iteration_time_seconds,
                time_elapsed=current.time_elapsed,
                eta=match.group("value"),
                metrics=dict(current_metrics),
                raw_lines=tuple(current_lines),
                line_number=current.line_number,
            )
            continue

        if match := _KEY_VALUE_RE.match(line):
            current_metrics[match.group("label").strip()] = _parse_scalar(match.group("value"))

    return _finalize_status(current, current_metrics, current_lines) or latest


def parse_rsl_rl_log_text(text: str) -> RslRlIterationStatus | None:
    """Backward-compatible alias for older internal naming."""

    return parse_rsl_rl_log(text)


def read_tail_text(path: Path | str, *, max_bytes: int = 128 * 1024) -> tuple[str, int]:
    """Read the tail of a text file safely."""

    resolved = Path(path)
    if not resolved.exists():
        return "", 0

    size = resolved.stat().st_size
    if size <= max_bytes:
        return resolved.read_text(encoding="utf-8", errors="replace"), size

    with resolved.open("rb") as handle:
        handle.seek(-max_bytes, os.SEEK_END)
        data = handle.read()
    return data.decode("utf-8", errors="replace"), len(data)


def snapshot_rsl_rl_log(path: Path | str, *, max_bytes: int = 128 * 1024) -> RobotLabLogSnapshot:
    """Create a structured snapshot from the tail of a log file."""

    resolved = Path(path)
    tail_text, bytes_read = read_tail_text(resolved, max_bytes=max_bytes)
    return RobotLabLogSnapshot(
        source_path=resolved,
        status=parse_rsl_rl_log(tail_text),
        tail_text=tail_text,
        bytes_read=bytes_read,
    )


def classify_artifact(path: Path | str) -> ArtifactKind:
    """Classify a path inside the RobotLab log tree."""

    resolved = Path(path)
    name = resolved.name
    lower_name = name.lower()

    if _CHECKPOINT_RE.match(name):
        return ArtifactKind.CHECKPOINT
    if lower_name.startswith("events.out.tfevents"):
        return ArtifactKind.TENSORBOARD_EVENT
    if resolved.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} and "videos" in resolved.parts:
        return ArtifactKind.VIDEO
    if resolved.suffix.lower() in {".pt", ".onnx", ".jit"} and "exported" in resolved.parts:
        return ArtifactKind.EXPORT
    if lower_name in {"env.yaml", "agent.yaml"} or "params" in resolved.parts:
        return ArtifactKind.PARAMS
    if resolved.suffix.lower() in {".log", ".txt", ".out"}:
        return ArtifactKind.LOG
    return ArtifactKind.UNKNOWN


def infer_run_dir(path: Path | str) -> Path | None:
    """Infer the run directory from a path within a RobotLab log tree."""

    resolved = Path(path)
    for candidate in (resolved, *resolved.parents):
        if candidate == candidate.parent:
            break
        if (candidate / "params").exists() or (candidate / "git").exists():
            return candidate
    return None


def _stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("step"))


def build_artifact_record(path: Path | str) -> ArtifactRecord:
    """Build a classified artifact record from a file path."""

    resolved = Path(path)
    return ArtifactRecord(
        path=resolved,
        kind=classify_artifact(resolved),
        run_dir=infer_run_dir(resolved),
        checkpoint_step=_checkpoint_step(resolved),
        modified_time=_stat_mtime(resolved),
    )


def iter_run_directories(scope: Path | str = DEFAULT_RSL_RL_LOG_ROOT) -> Iterator[Path]:
    """Yield directories that look like RSL-RL run directories."""

    root = Path(scope)
    if not root.exists():
        return iter(())

    candidates = [candidate for candidate in root.rglob("*") if candidate.is_dir() and (candidate / "params").is_dir()]
    candidates.sort(key=lambda item: (_stat_mtime(item), item.as_posix()))
    return iter(candidates)


def find_latest_run_dir(scope: Path | str = DEFAULT_RSL_RL_LOG_ROOT) -> Path | None:
    """Return the newest run directory in the scope."""

    latest: Path | None = None
    latest_key: tuple[float, str] | None = None
    for candidate in iter_run_directories(scope):
        key = (_stat_mtime(candidate), candidate.as_posix())
        if latest is None or latest_key is None or key > latest_key:
            latest = candidate
            latest_key = key
    return latest


def _best_artifact(scope: Path | str, kinds: set[ArtifactKind]) -> ArtifactRecord | None:
    root = Path(scope)
    if not root.exists():
        return None

    best: ArtifactRecord | None = None
    best_key: tuple[float, int, str] | None = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        record = build_artifact_record(path)
        if record.kind not in kinds:
            continue
        key = (record.modified_time or 0.0, record.checkpoint_step or -1, record.path.as_posix())
        if best is None or best_key is None or key > best_key:
            best = record
            best_key = key
    return best


def discover_artifacts(scope: Path | str = DEFAULT_RSL_RL_LOG_ROOT) -> DiscoveredArtifacts:
    """Discover the most recent artifacts inside a search scope."""

    root = Path(scope)
    return DiscoveredArtifacts(
        scope=root,
        latest_checkpoint=_best_artifact(root, {ArtifactKind.CHECKPOINT}),
        latest_video=_best_artifact(root, {ArtifactKind.VIDEO}),
        latest_export=_best_artifact(root, {ArtifactKind.EXPORT}),
        latest_log=_best_artifact(root, {ArtifactKind.LOG, ArtifactKind.TENSORBOARD_EVENT}),
    )


def find_latest_checkpoint(scope: Path | str = DEFAULT_RSL_RL_LOG_ROOT) -> Path | None:
    """Return the newest checkpoint in the scope."""

    record = _best_artifact(scope, {ArtifactKind.CHECKPOINT})
    return None if record is None else record.path


def find_latest_video(scope: Path | str = DEFAULT_RSL_RL_LOG_ROOT) -> Path | None:
    """Return the newest recorded video in the scope."""

    record = _best_artifact(scope, {ArtifactKind.VIDEO})
    return None if record is None else record.path

