"""Data models used by the agent state store."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


class BaseRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, use_enum_values=True)

    id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class SessionStatus(str, StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class SessionMode(str, StrEnum):
    CHAT = "chat"
    OPS = "ops"
    TRAINING = "training"
    REVIEW = "review"


class JobStatus(str, StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalStatus(str, StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ArtifactKind(str, StrEnum):
    CHECKPOINT = "checkpoint"
    VIDEO = "video"
    LOG = "log"
    SCREENSHOT = "screenshot"
    REPORT = "report"
    OTHER = "other"


class MessageSource(str, StrEnum):
    DISCORD = "discord"
    TERMINAL = "terminal"
    SYSTEM = "system"


class SessionRecord(BaseRecord):
    session_key: str
    guild_id: int | None = None
    channel_id: int | None = None
    channel_name: str | None = None
    user_id: int | None = None
    peer_id: str | None = None
    name: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    mode: SessionMode = SessionMode.CHAT
    summary: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SessionCreate(BaseModel):
    session_key: str
    guild_id: int | None = None
    channel_id: int | None = None
    channel_name: str | None = None
    user_id: int | None = None
    peer_id: str | None = None
    name: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    mode: SessionMode = SessionMode.CHAT
    summary: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    session_key: str | None = None
    guild_id: int | None = None
    channel_id: int | None = None
    channel_name: str | None = None
    user_id: int | None = None
    peer_id: str | None = None
    name: str | None = None
    status: SessionStatus | None = None
    mode: SessionMode | None = None
    summary: str | None = None
    metadata: dict[str, object] | None = None


class JobRecord(BaseRecord):
    session_key: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    command: str | None = None
    working_dir: str | None = None
    summary: str | None = None
    result: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobCreate(BaseModel):
    session_key: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    command: str | None = None
    working_dir: str | None = None
    summary: str | None = None
    result: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobUpdate(BaseModel):
    kind: str | None = None
    status: JobStatus | None = None
    command: str | None = None
    working_dir: str | None = None
    summary: str | None = None
    result: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ApprovalRecord(BaseRecord):
    session_key: str
    job_id: str | None = None
    scope: str
    requested_by: str | None = None
    requested_action: str | None = None
    reason: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ApprovalCreate(BaseModel):
    session_key: str
    job_id: str | None = None
    scope: str
    requested_by: str | None = None
    requested_action: str | None = None
    reason: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ApprovalUpdate(BaseModel):
    scope: str | None = None
    requested_by: str | None = None
    requested_action: str | None = None
    reason: str | None = None
    status: ApprovalStatus | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    metadata: dict[str, object] | None = None


class ArtifactRecord(BaseRecord):
    session_key: str
    job_id: str | None = None
    kind: ArtifactKind = ArtifactKind.OTHER
    name: str | None = None
    path: str
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    note: str | None = None


class ArtifactCreate(BaseModel):
    session_key: str
    job_id: str | None = None
    kind: ArtifactKind = ArtifactKind.OTHER
    name: str | None = None
    path: str
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    note: str | None = None


class ArtifactUpdate(BaseModel):
    kind: ArtifactKind | None = None
    name: str | None = None
    path: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, object] | None = None
    note: str | None = None


class MessageRecord(BaseRecord):
    session_key: str
    source: MessageSource
    message_id: str | None = None
    channel_id: str | None = None
    author_id: str | None = None
    content: str
    snippet: str
    metadata: dict[str, object] = Field(default_factory=dict)


class MessageSnippetCreate(BaseModel):
    session_key: str
    source: MessageSource
    message_id: str | None = None
    channel_id: str | None = None
    author_id: str | None = None
    content: str
    snippet: str
    metadata: dict[str, object] = Field(default_factory=dict)


class MessageSnippetUpdate(BaseModel):
    session_key: str | None = None
    source: MessageSource | None = None
    message_id: str | None = None
    channel_id: str | None = None
    author_id: str | None = None
    content: str | None = None
    snippet: str | None = None
    metadata: dict[str, object] | None = None
