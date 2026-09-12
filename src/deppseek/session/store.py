"""Session persistence, resume, and branching.

v1 kept one `.deepseek_agent_history.json` in the workspace root. That file was
inside the tree the agent searched, so the agent could read its own history as if
it were project content; it held exactly one conversation, so starting a second
task meant destroying the first; and it was loaded wholesale at startup with no
way to inspect or pick between runs.

Sessions here live under `.deppseek/sessions/`, which is both gitignored and
excluded from search. Each is a separate file with metadata, so `--resume` can
list them, and branching forks a session at its current point rather than
overwriting it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SESSIONS_DIRNAME = "sessions"
SCHEMA_VERSION = 2


@dataclass
class SessionMeta:
    id: str
    created_at: float
    updated_at: float
    title: str = ""
    model: str = ""
    workspace: str = ""
    step_count: int = 0
    message_count: int = 0
    estimated_cost_usd: float = 0.0
    total_tokens: int = 0
    parent_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def age(self) -> str:
        seconds = max(0, time.time() - self.updated_at)
        if seconds < 90:
            return f"{int(seconds)}s ago"
        if seconds < 5400:
            return f"{int(seconds / 60)}m ago"
        if seconds < 172800:
            return f"{int(seconds / 3600)}h ago"
        return f"{int(seconds / 86400)}d ago"


@dataclass
class Session:
    meta: SessionMeta
    messages: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"meta": asdict(self.meta), "messages": self.messages, "todos": self.todos},
            ensure_ascii=False,
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> Session:
        raw = json.loads(text)
        meta_raw = raw.get("meta", {})
        known = set(SessionMeta.__dataclass_fields__)
        meta = SessionMeta(**{k: v for k, v in meta_raw.items() if k in known})
        return cls(meta=meta, messages=raw.get("messages", []), todos=raw.get("todos", []))


class SessionStore:
    def __init__(self, state_dir: Path) -> None:
        self.root = state_dir / SESSIONS_DIRNAME

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def new(self, *, workspace: Path, model: str, title: str = "") -> Session:
        now = time.time()
        session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        meta = SessionMeta(
            id=session_id,
            created_at=now,
            updated_at=now,
            title=title,
            model=model,
            workspace=str(workspace),
        )
        return Session(meta=meta)

    def save(self, session: Session) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        session.meta.updated_at = time.time()
        session.meta.message_count = len(session.messages)
        path = self._path(session.meta.id)
        # Write then replace: an interrupted save must not destroy the session
        # it was about to update.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(session.to_json(), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        try:
            return Session.from_json(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def list(self, limit: int = 25) -> list[SessionMeta]:
        if not self.root.is_dir():
            return []
        metas: list[SessionMeta] = []
        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                meta_raw = raw.get("meta", {})
                known = set(SessionMeta.__dataclass_fields__)
                metas.append(SessionMeta(**{k: v for k, v in meta_raw.items() if k in known}))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas[:limit]

    def latest(self) -> Session | None:
        metas = self.list(limit=1)
        return self.load(metas[0].id) if metas else None

    def branch(self, session: Session, title: str = "") -> Session:
        """Fork a session, so an experiment does not overwrite the original."""
        forked = self.new(
            workspace=Path(session.meta.workspace),
            model=session.meta.model,
            title=title or f"branch of {session.meta.title or session.meta.id}",
        )
        forked.meta.parent_id = session.meta.id
        forked.messages = [dict(m) for m in session.messages]
        forked.todos = [dict(t) for t in session.todos]
        return forked

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def describe(self, limit: int = 15) -> str:
        metas = self.list(limit)
        if not metas:
            return "No saved sessions."
        lines = [f"{'id':<24} {'when':<10} {'msgs':>5} {'cost':>9}  title"]
        for meta in metas:
            lines.append(
                f"{meta.id:<24} {meta.age:<10} {meta.message_count:>5} "
                f"${meta.estimated_cost_usd:>8.4f}  {meta.title[:44]}"
            )
        return "\n".join(lines)
