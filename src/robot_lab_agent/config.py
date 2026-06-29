"""Runtime settings for the robot_lab agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser()


class DiscordSettings(BaseModel):
    """Discord-specific runtime settings."""

    bot_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN", "discord_bot_token"),
    )
    application_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("DISCORD_APPLICATION_ID", "discord_application_id"),
    )
    guild_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("DISCORD_GUILD_ID", "discord_guild_id"),
    )
    command_prefix: str = Field(
        default="/",
        validation_alias=AliasChoices("DISCORD_COMMAND_PREFIX", "discord_command_prefix"),
    )
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("DISCORD_ALLOWED_USER_IDS", "discord_allowed_user_ids"),
    )
    allowed_role_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("DISCORD_ALLOWED_ROLE_IDS", "discord_allowed_role_ids"),
    )
    webhook_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DISCORD_WEBHOOK_URL", "discord_webhook_url"),
    )

    @field_validator("allowed_user_ids", "allowed_role_ids", mode="before")
    @classmethod
    def _parse_id_list(cls, value: object) -> list[int]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        raise TypeError(f"Unsupported Discord id list value: {value!r}")


class RobotLabSettings(BaseModel):
    """Paths and execution settings for the robot_lab workspace."""

    root: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab"),
        validation_alias=AliasChoices("ROBOTLAB_ROOT", "robotlab_root"),
    )
    agent_root: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab_agent"),
        validation_alias=AliasChoices("ROBOTLAB_AGENT_ROOT", "robotlab_agent_root"),
    )
    conda_env: str = Field(
        default="isaac",
        validation_alias=AliasChoices("ROBOTLAB_CONDA_ENV", "robotlab_conda_env"),
    )
    python_executable: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("ROBOTLAB_PYTHON_EXECUTABLE", "robotlab_python_executable"),
    )
    train_script: Path = Field(
        default=Path("scripts/reinforcement_learning/rsl_rl/train.py"),
        validation_alias=AliasChoices("ROBOTLAB_TRAIN_SCRIPT", "robotlab_train_script"),
    )
    play_script: Path = Field(
        default=Path("scripts/reinforcement_learning/rsl_rl/play.py"),
        validation_alias=AliasChoices("ROBOTLAB_PLAY_SCRIPT", "robotlab_play_script"),
    )
    log_root: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab_agent/logs"),
        validation_alias=AliasChoices("ROBOTLAB_LOG_ROOT", "robotlab_log_root"),
    )
    artifact_root: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab_agent/artifacts"),
        validation_alias=AliasChoices("ROBOTLAB_ARTIFACT_ROOT", "robotlab_artifact_root"),
    )
    checkpoint_root: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab_agent/checkpoints"),
        validation_alias=AliasChoices("ROBOTLAB_CHECKPOINT_ROOT", "robotlab_checkpoint_root"),
    )
    default_video_length: int = Field(
        default=200,
        ge=1,
        validation_alias=AliasChoices("ROBOTLAB_DEFAULT_VIDEO_LENGTH", "robotlab_default_video_length"),
    )

    @field_validator("root", "agent_root", "python_executable", "train_script", "play_script", "log_root", "artifact_root", "checkpoint_root", mode="before")
    @classmethod
    def _expand_robotlab_paths(cls, value: object) -> Path | None:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            return _expand_path(value)
        raise TypeError(f"Unsupported path value: {value!r}")

    def resolve_relative(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @property
    def resolved_train_script(self) -> Path:
        return self.resolve_relative(self.train_script)

    @property
    def resolved_play_script(self) -> Path:
        return self.resolve_relative(self.play_script)


class PolicySettings(BaseModel):
    """Operational policy toggles."""

    allow_shell_execution: bool = Field(
        default=False,
        validation_alias=AliasChoices("ALLOW_SHELL_EXECUTION", "allow_shell_execution"),
    )
    allow_process_control: bool = Field(
        default=False,
        validation_alias=AliasChoices("ALLOW_PROCESS_CONTROL", "allow_process_control"),
    )
    allow_discord_commands: bool = Field(
        default=True,
        validation_alias=AliasChoices("ALLOW_DISCORD_COMMANDS", "allow_discord_commands"),
    )
    require_approval_for_destructive_actions: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "REQUIRE_APPROVAL_FOR_DESTRUCTIVE_ACTIONS",
            "require_approval_for_destructive_actions",
        ),
    )
    require_approval_for_code_changes: bool = Field(
        default=False,
        validation_alias=AliasChoices("REQUIRE_APPROVAL_FOR_CODE_CHANGES", "require_approval_for_code_changes"),
    )
    require_approval_for_external_network: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "REQUIRE_APPROVAL_FOR_EXTERNAL_NETWORK",
            "require_approval_for_external_network",
        ),
    )
    max_message_snippet_chars: int = Field(
        default=2000,
        ge=128,
        validation_alias=AliasChoices("MAX_MESSAGE_SNIPPET_CHARS", "max_message_snippet_chars"),
    )


class AppSettings(BaseSettings):
    """Top-level settings container loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    discord: DiscordSettings = Field(default_factory=DiscordSettings)
    robotlab: RobotLabSettings = Field(default_factory=RobotLabSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    database_path: Path = Field(
        default_factory=lambda: Path("/home/chae/robot_lab_agent/data/state.sqlite3"),
        validation_alias=AliasChoices("DATABASE_PATH", "database_path"),
    )

    @field_validator("database_path", mode="before")
    @classmethod
    def _expand_database_path(cls, value: object) -> Path:
        if isinstance(value, (str, Path)):
            return _expand_path(value)
        raise TypeError(f"Unsupported database path value: {value!r}")


def load_settings() -> AgentSettings:
    """Load settings from the current process environment."""

    return AppSettings()


AgentSettings = AppSettings
