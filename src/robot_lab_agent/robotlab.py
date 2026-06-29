"""RobotLab operator primitives for the Discord-facing agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import os
import shlex
from typing import Mapping, Sequence

from .log_parser import (
    ArtifactKind,
    ArtifactRecord,
    DEFAULT_RSL_RL_LOG_ROOT,
    DiscoveredArtifacts,
    RobotLabLogSnapshot,
    build_artifact_record,
    classify_artifact,
    discover_artifacts,
    find_latest_checkpoint,
    find_latest_run_dir,
    find_latest_video,
    infer_run_dir,
    snapshot_rsl_rl_log,
)

DEFAULT_ROBOTLAB_ROOT = Path("/home/chae/robot_lab")


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    """Subprocess-ready command invocation."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True, slots=True)
class RobotLabStatus:
    """Snapshot of the current RobotLab operator state."""

    log_snapshot: RobotLabLogSnapshot | None = None
    artifacts: DiscoveredArtifacts | None = None
    latest_checkpoint: Path | None = None
    latest_video: Path | None = None
    latest_run_dir: Path | None = None


def _append_cli_flag(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _append_cli_bool(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _append_extra_args(argv: list[str], extra_args: Sequence[str] | None) -> None:
    if extra_args:
        argv.extend(extra_args)


class RobotLabOperator:
    """Async operator for RobotLab training and playback workflows."""

    def __init__(
        self,
        *,
        robotlab_root: Path | str = DEFAULT_ROBOTLAB_ROOT,
        logs_root: Path | str = DEFAULT_RSL_RL_LOG_ROOT,
        conda_env: str = "isaac",
        conda_executable: str = "conda",
        python_executable: str = "python",
    ) -> None:
        self.robotlab_root = Path(robotlab_root)
        self.logs_root = Path(logs_root)
        self.conda_env = conda_env
        self.conda_executable = conda_executable
        self.python_executable = python_executable

        self.train_script = self.robotlab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
        self.play_script = self.robotlab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"

    def _conda_run_prefix(self) -> list[str]:
        return [
            self.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            self.python_executable,
        ]

    def build_train_command(
        self,
        *,
        task: str,
        agent: str = "rsl_rl_cfg_entry_point",
        num_envs: int | None = None,
        seed: int | None = None,
        max_iterations: int | None = None,
        video: bool = False,
        video_length: int | None = None,
        video_interval: int | None = None,
        distributed: bool = False,
        export_io_descriptors: bool = False,
        ray_proc_id: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> CommandInvocation:
        argv = self._conda_run_prefix() + [str(self.train_script)]
        _append_cli_flag(argv, "--task", task)
        _append_cli_flag(argv, "--agent", agent)
        _append_cli_flag(argv, "--num_envs", num_envs)
        _append_cli_flag(argv, "--seed", seed)
        _append_cli_flag(argv, "--max_iterations", max_iterations)
        _append_cli_bool(argv, "--video", video)
        _append_cli_flag(argv, "--video_length", video_length)
        _append_cli_flag(argv, "--video_interval", video_interval)
        _append_cli_bool(argv, "--distributed", distributed)
        _append_cli_bool(argv, "--export_io_descriptors", export_io_descriptors)
        _append_cli_flag(argv, "--ray-proc-id", ray_proc_id)
        _append_extra_args(argv, extra_args)
        return CommandInvocation(argv=tuple(argv), cwd=self.robotlab_root)

    def build_play_command(
        self,
        *,
        task: str,
        agent: str = "rsl_rl_cfg_entry_point",
        num_envs: int | None = None,
        seed: int | None = None,
        video: bool = True,
        video_length: int | None = None,
        use_pretrained_checkpoint: bool = False,
        checkpoint: str | None = None,
        load_run: str | None = None,
        load_checkpoint: str | None = None,
        real_time: bool = False,
        keyboard: bool = False,
        disable_torque_plot: bool = True,
        torque_plot_interval: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> CommandInvocation:
        argv = self._conda_run_prefix() + [str(self.play_script)]
        _append_cli_flag(argv, "--task", task)
        _append_cli_flag(argv, "--agent", agent)
        _append_cli_flag(argv, "--num_envs", num_envs)
        _append_cli_flag(argv, "--seed", seed)
        _append_cli_bool(argv, "--video", video)
        _append_cli_flag(argv, "--video_length", video_length)
        _append_cli_bool(argv, "--use_pretrained_checkpoint", use_pretrained_checkpoint)
        _append_cli_flag(argv, "--checkpoint", checkpoint)
        _append_cli_flag(argv, "--load_run", load_run)
        _append_cli_flag(argv, "--load_checkpoint", load_checkpoint)
        _append_cli_bool(argv, "--real-time", real_time)
        _append_cli_bool(argv, "--keyboard", keyboard)
        _append_cli_bool(argv, "--disable_torque_plot", disable_torque_plot)
        _append_cli_flag(argv, "--torque_plot_interval", torque_plot_interval)
        _append_extra_args(argv, extra_args)
        return CommandInvocation(argv=tuple(argv), cwd=self.robotlab_root)

    async def start_subprocess(
        self,
        invocation: CommandInvocation,
        *,
        extra_env: Mapping[str, str] | None = None,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> asyncio.subprocess.Process:
        env = os.environ.copy()
        env.update(invocation.env)
        if extra_env:
            env.update(extra_env)
        return await asyncio.create_subprocess_exec(
            *invocation.argv,
            cwd=str(invocation.cwd),
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )

    async def snapshot_from_log(self, log_path: Path | str, *, max_bytes: int = 128 * 1024) -> RobotLabLogSnapshot:
        return await asyncio.to_thread(snapshot_rsl_rl_log, log_path, max_bytes=max_bytes)

    async def latest_checkpoint(self, scope: Path | str | None = None) -> Path | None:
        return await asyncio.to_thread(find_latest_checkpoint, scope or self.logs_root)

    async def latest_video(self, scope: Path | str | None = None) -> Path | None:
        return await asyncio.to_thread(find_latest_video, scope or self.logs_root)

    async def latest_run_dir(self, scope: Path | str | None = None) -> Path | None:
        return await asyncio.to_thread(find_latest_run_dir, scope or self.logs_root)

    async def latest_artifacts(self, scope: Path | str | None = None) -> DiscoveredArtifacts:
        return await asyncio.to_thread(discover_artifacts, scope or self.logs_root)

    async def latest_status(self, log_path: Path | str | None = None) -> RobotLabStatus:
        log_snapshot = None
        if log_path is not None:
            log_snapshot = await self.snapshot_from_log(log_path)

        artifacts = await self.latest_artifacts()
        return RobotLabStatus(
            log_snapshot=log_snapshot,
            artifacts=artifacts,
            latest_checkpoint=None if artifacts.latest_checkpoint is None else artifacts.latest_checkpoint.path,
            latest_video=None if artifacts.latest_video is None else artifacts.latest_video.path,
            latest_run_dir=await self.latest_run_dir(),
        )

    def classify_artifact(self, path: Path | str) -> ArtifactKind:
        return classify_artifact(path)

    def build_artifact_record(self, path: Path | str) -> ArtifactRecord:
        return build_artifact_record(path)

    def infer_run_dir(self, path: Path | str) -> Path | None:
        return infer_run_dir(path)

