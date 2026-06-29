from __future__ import annotations

import atexit
import codecs
import json
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


class _LatestOutputMirror:
    def __init__(self, latest_path: Path, *, max_lines: int = 64, max_chars: int = 4000):
        self.latest_path = latest_path
        self._max_lines = max_lines
        self._max_chars = max_chars
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._current_line = ""
        self._handle = latest_path.open("w", encoding="utf-8")
        self._closed = False

    def consume_text(self, text: str) -> None:
        for char in text:
            if char == "\r":
                self._current_line = ""
            elif char == "\n":
                self._lines.append(self._current_line)
                self._current_line = ""
            else:
                self._current_line += char
        self._rewrite()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._rewrite()
        self._handle.close()

    def _rewrite(self) -> None:
        lines = list(self._lines)
        if self._current_line:
            lines.append(self._current_line)
        snapshot = "\n".join(lines)
        if len(snapshot) > self._max_chars:
            snapshot = snapshot[-self._max_chars :]
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(snapshot)
        self._handle.flush()


def _pump_fd_to_terminal_and_log(
    *,
    read_fd: int,
    target_fd: int,
    log_handle: BinaryIO,
    latest_output_mirror: _LatestOutputMirror,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            try:
                os.write(target_fd, chunk)
            except OSError:
                pass
            log_handle.write(chunk)
            latest_output_mirror.consume_text(decoder.decode(chunk))
        tail = decoder.decode(b"", final=True)
        if tail:
            latest_output_mirror.consume_text(tail)
    finally:
        latest_output_mirror.close()
        os.close(read_fd)


@dataclass(slots=True)
class FixedTrainLogger:
    script_name: str
    log_dir: Path
    log_path: Path
    status_path: Path
    latest_path: Path
    _log_handle: BinaryIO
    _stdout_fd: int
    _stderr_fd: int
    _original_stdout_fd: int
    _original_stderr_fd: int
    _pump_thread: threading.Thread
    _status: dict[str, Any] = field(default_factory=dict)
    _closed: bool = False

    def update_status(self, **payload: Any) -> None:
        self._status.update(payload)
        self._status["updated_at"] = _now_iso()
        self.status_path.write_text(
            json.dumps(_clean_status_payload(self._status), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass

        os.dup2(self._original_stdout_fd, self._stdout_fd)
        os.dup2(self._original_stderr_fd, self._stderr_fd)

        self._pump_thread.join(timeout=2.0)
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        self._log_handle.flush()
        self._log_handle.close()


def initialize_fixed_train_logging(
    *,
    script_name: str,
    args: Any,
    raw_argv: list[str] | None = None,
    extra_status: dict[str, Any] | None = None,
) -> FixedTrainLogger:
    train_log_dir = _project_root() / "train_log"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = train_log_dir / "current_train.log"
    status_path = train_log_dir / "current_train_status.json"
    latest_path = train_log_dir / "current_train_latest.txt"
    log_handle = log_path.open("wb", buffering=0)
    latest_output_mirror = _LatestOutputMirror(latest_path)

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    original_stdout_fd = os.dup(stdout_fd)
    original_stderr_fd = os.dup(stderr_fd)
    pipe_read_fd, pipe_write_fd = os.pipe()

    os.dup2(pipe_write_fd, stdout_fd)
    os.dup2(pipe_write_fd, stderr_fd)
    os.close(pipe_write_fd)

    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

    pump_thread = threading.Thread(
        target=_pump_fd_to_terminal_and_log,
        kwargs={
            "read_fd": pipe_read_fd,
            "target_fd": original_stdout_fd,
            "log_handle": log_handle,
            "latest_output_mirror": latest_output_mirror,
        },
        name="fixed-train-log-pump",
        daemon=True,
    )
    pump_thread.start()

    logger = FixedTrainLogger(
        script_name=script_name,
        log_dir=train_log_dir,
        log_path=log_path,
        status_path=status_path,
        latest_path=latest_path,
        _log_handle=log_handle,
        _stdout_fd=stdout_fd,
        _stderr_fd=stderr_fd,
        _original_stdout_fd=original_stdout_fd,
        _original_stderr_fd=original_stderr_fd,
        _pump_thread=pump_thread,
    )
    logger.update_status(
        script_name=script_name,
        status="starting",
        pid=os.getpid(),
        task=getattr(args, "task", None),
        agent=getattr(args, "agent", None),
        started_at=_now_iso(),
        log_path=log_path,
        latest_output_path=latest_path,
        raw_argv=raw_argv or list(sys.argv),
        **(extra_status or {}),
    )
    atexit.register(logger.close)
    return logger


def latest_checkpoint_path(run_dir: str | Path | None) -> str | None:
    if run_dir is None:
        return None
    candidate_dir = Path(run_dir).expanduser()
    if not candidate_dir.is_dir():
        return None
    checkpoints = [path for path in candidate_dir.rglob("*.pt") if path.is_file()]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size))
    return str(latest)
