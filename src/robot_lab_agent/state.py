"""SQLite-backed state store for sessions, jobs, approvals, and artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import aiosqlite

from robot_lab_agent.models import (
    ApprovalCreate,
    ApprovalRecord,
    ApprovalStatus,
    ApprovalUpdate,
    ArtifactCreate,
    ArtifactKind,
    ArtifactRecord,
    ArtifactUpdate,
    JobCreate,
    JobRecord,
    JobStatus,
    JobUpdate,
    MessageSnippetCreate,
    MessageSnippetRecord,
    MessageSnippetUpdate,
    SessionCreate,
    SessionMode,
    SessionRecord,
    SessionStatus,
    SessionUpdate,
    utcnow,
)

T = TypeVar("T")


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    guild_id INTEGER,
    channel_id INTEGER,
    user_id INTEGER,
    name TEXT,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_channel_id ON sessions(channel_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    command TEXT,
    working_dir TEXT,
    summary TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    requested_by TEXT,
    requested_action TEXT,
    reason TEXT,
    status TEXT NOT NULL,
    decided_by TEXT,
    decision_reason TEXT,
    decided_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_approvals_session_id ON approvals(session_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT,
    path TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts(job_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);

CREATE TABLE IF NOT EXISTS message_snippets (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    message_id TEXT,
    channel_id TEXT,
    author_id TEXT,
    content TEXT NOT NULL,
    snippet TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_message_snippets_session_id ON message_snippets(session_id);
CREATE INDEX IF NOT EXISTS idx_message_snippets_source ON message_snippets(source);
"""


def _utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _dump_json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _apply_filters(base_sql: str, filters: list[tuple[str, Any]], order_by: str, limit: int, offset: int) -> tuple[str, list[Any]]:
    clauses = [base_sql]
    params: list[Any] = []
    if filters:
        clauses.append("WHERE " + " AND ".join(f"{column} = ?" for column, _ in filters))
        params.extend(value for _, value in filters)
    clauses.append(f"ORDER BY {order_by}")
    clauses.append("LIMIT ? OFFSET ?")
    params.extend([limit, offset])
    return " ".join(clauses), params


class StateStore:
    """aiosqlite-backed persistence for agent state."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "StateStore":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> None:
        if self._connection is not None:
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA journal_mode = WAL")
        await connection.execute("PRAGMA busy_timeout = 5000")
        self._connection = connection

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("StateStore is not open")
        return self._connection

    async def initialize(self) -> None:
        await self.open()
        await self.connection.executescript(SCHEMA)
        await self.connection.commit()

    async def healthcheck(self) -> bool:
        await self.open()
        async with self.connection.execute("SELECT 1") as cursor:
            row = await cursor.fetchone()
        return bool(row)

    @staticmethod
    def _session_from_row(row: Mapping[str, Any]) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            user_id=row["user_id"],
            name=row["name"],
            status=SessionStatus(row["status"]),
            mode=SessionMode(row["mode"]),
            summary=row["summary"],
            metadata=_load_json(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or utcnow(),
            updated_at=_parse_datetime(row["updated_at"]) or utcnow(),
        )

    @staticmethod
    def _job_from_row(row: Mapping[str, Any]) -> JobRecord:
        return JobRecord(
            id=row["id"],
            session_id=row["session_id"],
            kind=row["kind"],
            status=JobStatus(row["status"]),
            command=row["command"],
            working_dir=row["working_dir"],
            summary=row["summary"],
            result=_load_json(row["result_json"]),
            metadata=_load_json(row["metadata_json"]),
            started_at=_parse_datetime(row["started_at"]),
            finished_at=_parse_datetime(row["finished_at"]),
            created_at=_parse_datetime(row["created_at"]) or utcnow(),
            updated_at=_parse_datetime(row["updated_at"]) or utcnow(),
        )

    @staticmethod
    def _approval_from_row(row: Mapping[str, Any]) -> ApprovalRecord:
        return ApprovalRecord(
            id=row["id"],
            session_id=row["session_id"],
            job_id=row["job_id"],
            scope=row["scope"],
            requested_by=row["requested_by"],
            requested_action=row["requested_action"],
            reason=row["reason"],
            status=ApprovalStatus(row["status"]),
            decided_by=row["decided_by"],
            decision_reason=row["decision_reason"],
            decided_at=_parse_datetime(row["decided_at"]),
            metadata=_load_json(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or utcnow(),
            updated_at=_parse_datetime(row["updated_at"]) or utcnow(),
        )

    @staticmethod
    def _artifact_from_row(row: Mapping[str, Any]) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            session_id=row["session_id"],
            job_id=row["job_id"],
            kind=ArtifactKind(row["kind"]),
            name=row["name"],
            path=row["path"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            metadata=_load_json(row["metadata_json"]),
            note=row["note"],
            created_at=_parse_datetime(row["created_at"]) or utcnow(),
            updated_at=_parse_datetime(row["updated_at"]) or utcnow(),
        )

    @staticmethod
    def _message_snippet_from_row(row: Mapping[str, Any]) -> MessageSnippetRecord:
        return MessageSnippetRecord(
            id=row["id"],
            session_id=row["session_id"],
            source=row["source"],
            message_id=row["message_id"],
            channel_id=row["channel_id"],
            author_id=row["author_id"],
            content=row["content"],
            snippet=row["snippet"],
            metadata=_load_json(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or utcnow(),
            updated_at=_parse_datetime(row["updated_at"]) or utcnow(),
        )

    async def create_session(self, data: SessionCreate) -> SessionRecord:
        await self.open()
        record = SessionRecord(**data.model_dump())
        await self.connection.execute(
            """
            INSERT INTO sessions (
                id, guild_id, channel_id, user_id, name, status, mode, summary,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.guild_id,
                record.channel_id,
                record.user_id,
                record.name,
                record.status,
                record.mode,
                record.summary,
                _dump_json(record.metadata),
                _utc(record.created_at),
                _utc(record.updated_at),
            ),
        )
        await self.connection.commit()
        return record

    async def get_session(self, session_id: str) -> SessionRecord | None:
        await self.open()
        async with self.connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        return self._session_from_row(row) if row else None

    async def list_sessions(
        self,
        *,
        status: SessionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionRecord]:
        await self.open()
        filters: list[tuple[str, Any]] = []
        if status is not None:
            filters.append(("status", status))
        sql, params = _apply_filters("SELECT * FROM sessions", filters, "updated_at DESC", limit, offset)
        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._session_from_row(row) for row in rows]

    async def update_session(self, session_id: str, data: SessionUpdate) -> SessionRecord | None:
        await self.open()
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return await self.get_session(session_id)
        changes["updated_at"] = utcnow()
        columns = []
        values: list[Any] = []
        for key, value in changes.items():
            if key == "metadata":
                columns.append("metadata_json = ?")
                values.append(_dump_json(value))
            elif key in {"status", "mode"}:
                columns.append(f"{key} = ?")
                values.append(value)
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(_utc(changes["updated_at"]))
        values.append(session_id)
        await self.connection.execute(f"UPDATE sessions SET {', '.join(columns)} WHERE id = ?", values)
        await self.connection.commit()
        return await self.get_session(session_id)

    async def delete_session(self, session_id: str) -> bool:
        await self.open()
        cursor = await self.connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self.connection.commit()
        return cursor.rowcount > 0

    async def create_job(self, data: JobCreate) -> JobRecord:
        await self.open()
        record = JobRecord(**data.model_dump())
        await self.connection.execute(
            """
            INSERT INTO jobs (
                id, session_id, kind, status, command, working_dir, summary,
                result_json, metadata_json, started_at, finished_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.kind,
                record.status,
                record.command,
                record.working_dir,
                record.summary,
                _dump_json(record.result),
                _dump_json(record.metadata),
                _utc(record.started_at),
                _utc(record.finished_at),
                _utc(record.created_at),
                _utc(record.updated_at),
            ),
        )
        await self.connection.commit()
        return record

    async def get_job(self, job_id: str) -> JobRecord | None:
        await self.open()
        async with self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        return self._job_from_row(row) if row else None

    async def list_jobs(
        self,
        *,
        session_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        await self.open()
        filters: list[tuple[str, Any]] = []
        if session_id is not None:
            filters.append(("session_id", session_id))
        if status is not None:
            filters.append(("status", status))
        sql, params = _apply_filters("SELECT * FROM jobs", filters, "updated_at DESC", limit, offset)
        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._job_from_row(row) for row in rows]

    async def update_job(self, job_id: str, data: JobUpdate) -> JobRecord | None:
        await self.open()
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return await self.get_job(job_id)
        changes["updated_at"] = utcnow()
        columns = []
        values: list[Any] = []
        for key, value in changes.items():
            if key in {"result", "metadata"}:
                columns.append(f"{key}_json = ?")
                values.append(_dump_json(value))
            elif key in {"status"}:
                columns.append(f"{key} = ?")
                values.append(value)
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(_utc(changes["updated_at"]))
        values.append(job_id)
        await self.connection.execute(f"UPDATE jobs SET {', '.join(columns)} WHERE id = ?", values)
        await self.connection.commit()
        return await self.get_job(job_id)

    async def delete_job(self, job_id: str) -> bool:
        await self.open()
        cursor = await self.connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await self.connection.commit()
        return cursor.rowcount > 0

    async def create_approval(self, data: ApprovalCreate) -> ApprovalRecord:
        await self.open()
        record = ApprovalRecord(**data.model_dump())
        await self.connection.execute(
            """
            INSERT INTO approvals (
                id, session_id, job_id, scope, requested_by, requested_action,
                reason, status, decided_by, decision_reason, decided_at,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.job_id,
                record.scope,
                record.requested_by,
                record.requested_action,
                record.reason,
                record.status,
                record.decided_by,
                record.decision_reason,
                _utc(record.decided_at),
                _dump_json(record.metadata),
                _utc(record.created_at),
                _utc(record.updated_at),
            ),
        )
        await self.connection.commit()
        return record

    async def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        await self.open()
        async with self.connection.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)) as cursor:
            row = await cursor.fetchone()
        return self._approval_from_row(row) if row else None

    async def list_approvals(
        self,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ApprovalRecord]:
        await self.open()
        filters: list[tuple[str, Any]] = []
        if session_id is not None:
            filters.append(("session_id", session_id))
        if job_id is not None:
            filters.append(("job_id", job_id))
        if status is not None:
            filters.append(("status", status))
        sql, params = _apply_filters("SELECT * FROM approvals", filters, "updated_at DESC", limit, offset)
        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._approval_from_row(row) for row in rows]

    async def update_approval(self, approval_id: str, data: ApprovalUpdate) -> ApprovalRecord | None:
        await self.open()
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return await self.get_approval(approval_id)
        changes["updated_at"] = utcnow()
        columns = []
        values: list[Any] = []
        for key, value in changes.items():
            if key == "metadata":
                columns.append("metadata_json = ?")
                values.append(_dump_json(value))
            elif key == "decided_at":
                columns.append("decided_at = ?")
                values.append(_utc(value))
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(_utc(changes["updated_at"]))
        values.append(approval_id)
        await self.connection.execute(f"UPDATE approvals SET {', '.join(columns)} WHERE id = ?", values)
        await self.connection.commit()
        return await self.get_approval(approval_id)

    async def delete_approval(self, approval_id: str) -> bool:
        await self.open()
        cursor = await self.connection.execute("DELETE FROM approvals WHERE id = ?", (approval_id,))
        await self.connection.commit()
        return cursor.rowcount > 0

    async def create_artifact(self, data: ArtifactCreate) -> ArtifactRecord:
        await self.open()
        record = ArtifactRecord(**data.model_dump())
        await self.connection.execute(
            """
            INSERT INTO artifacts (
                id, session_id, job_id, kind, name, path, mime_type, size_bytes,
                sha256, metadata_json, note, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.job_id,
                record.kind,
                record.name,
                record.path,
                record.mime_type,
                record.size_bytes,
                record.sha256,
                _dump_json(record.metadata),
                record.note,
                _utc(record.created_at),
                _utc(record.updated_at),
            ),
        )
        await self.connection.commit()
        return record

    async def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        await self.open()
        async with self.connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)) as cursor:
            row = await cursor.fetchone()
        return self._artifact_from_row(row) if row else None

    async def list_artifacts(
        self,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
        kind: ArtifactKind | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ArtifactRecord]:
        await self.open()
        filters: list[tuple[str, Any]] = []
        if session_id is not None:
            filters.append(("session_id", session_id))
        if job_id is not None:
            filters.append(("job_id", job_id))
        if kind is not None:
            filters.append(("kind", kind))
        sql, params = _apply_filters("SELECT * FROM artifacts", filters, "updated_at DESC", limit, offset)
        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._artifact_from_row(row) for row in rows]

    async def update_artifact(self, artifact_id: str, data: ArtifactUpdate) -> ArtifactRecord | None:
        await self.open()
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return await self.get_artifact(artifact_id)
        changes["updated_at"] = utcnow()
        columns = []
        values: list[Any] = []
        for key, value in changes.items():
            if key == "metadata":
                columns.append("metadata_json = ?")
                values.append(_dump_json(value))
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(_utc(changes["updated_at"]))
        values.append(artifact_id)
        await self.connection.execute(f"UPDATE artifacts SET {', '.join(columns)} WHERE id = ?", values)
        await self.connection.commit()
        return await self.get_artifact(artifact_id)

    async def delete_artifact(self, artifact_id: str) -> bool:
        await self.open()
        cursor = await self.connection.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        await self.connection.commit()
        return cursor.rowcount > 0

    async def create_message_snippet(self, data: MessageSnippetCreate) -> MessageSnippetRecord:
        await self.open()
        record = MessageSnippetRecord(**data.model_dump())
        await self.connection.execute(
            """
            INSERT INTO message_snippets (
                id, session_id, source, message_id, channel_id, author_id, content,
                snippet, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.source,
                record.message_id,
                record.channel_id,
                record.author_id,
                record.content,
                record.snippet,
                _dump_json(record.metadata),
                _utc(record.created_at),
                _utc(record.updated_at),
            ),
        )
        await self.connection.commit()
        return record

    async def get_message_snippet(self, snippet_id: str) -> MessageSnippetRecord | None:
        await self.open()
        async with self.connection.execute("SELECT * FROM message_snippets WHERE id = ?", (snippet_id,)) as cursor:
            row = await cursor.fetchone()
        return self._message_snippet_from_row(row) if row else None

    async def list_message_snippets(
        self,
        *,
        session_id: str | None = None,
        source: MessageSource | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MessageSnippetRecord]:
        await self.open()
        filters: list[tuple[str, Any]] = []
        if session_id is not None:
            filters.append(("session_id", session_id))
        if source is not None:
            filters.append(("source", source))
        sql, params = _apply_filters("SELECT * FROM message_snippets", filters, "created_at DESC", limit, offset)
        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._message_snippet_from_row(row) for row in rows]

    async def update_message_snippet(self, snippet_id: str, data: MessageSnippetUpdate) -> MessageSnippetRecord | None:
        await self.open()
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return await self.get_message_snippet(snippet_id)
        changes["updated_at"] = utcnow()
        columns = []
        values: list[Any] = []
        for key, value in changes.items():
            if key == "metadata":
                columns.append("metadata_json = ?")
                values.append(_dump_json(value))
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(_utc(changes["updated_at"]))
        values.append(snippet_id)
        await self.connection.execute(f"UPDATE message_snippets SET {', '.join(columns)} WHERE id = ?", values)
        await self.connection.commit()
        return await self.get_message_snippet(snippet_id)

    async def delete_message_snippet(self, snippet_id: str) -> bool:
        await self.open()
        cursor = await self.connection.execute("DELETE FROM message_snippets WHERE id = ?", (snippet_id,))
        await self.connection.commit()
        return cursor.rowcount > 0
