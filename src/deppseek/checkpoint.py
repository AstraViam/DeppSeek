"""Content-addressed checkpoints and undo.

Running mostly-autonomously means writes land without a prompt, so the recovery
path has to be better than "hope it was committed". v1 had none: a bad
`write_file` destroyed the previous contents with no way back.

Design:

* Before any mutation, the affected files are snapshotted into an object store
  keyed by SHA-256 of their contents. Identical content is stored once, so
  re-checkpointing an unchanged 5 MB mesh file costs nothing after the first
  time.
* Each checkpoint appends one JSON line to a manifest. Append-only means a crash
  mid-write cannot corrupt earlier history.
* Undo restores the recorded *before* state, including recreating files that
  were deleted and removing files that were created.

This is deliberately independent of git. The workspace may not be a repository,
may have unrelated uncommitted work, or may be mid-rebase; none of that should
affect whether the agent's own last edit can be taken back.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import DeppSeekError

MANIFEST_NAME = "checkpoints.jsonl"
OBJECTS_DIRNAME = "objects"
# Files larger than this are not snapshotted; the entry records the omission so
# undo can report honestly rather than silently restoring nothing.
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


@dataclass
class FileState:
    """One file's before/after content hashes within a checkpoint."""

    path: str  # workspace-relative POSIX
    before: str | None  # sha256, or None when the file did not exist
    after: str | None = None  # filled in after the mutation completes
    before_size: int = 0
    skipped_reason: str | None = None

    @property
    def created(self) -> bool:
        return self.before is None

    @property
    def deleted(self) -> bool:
        return self.after is None and self.before is not None


@dataclass
class Checkpoint:
    id: str
    created_at: float
    session_id: str
    step: int
    tool: str
    description: str
    files: list[FileState] = field(default_factory=list)
    undone: bool = False

    @property
    def when(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.created_at))

    def summary(self) -> str:
        created = sum(1 for f in self.files if f.created)
        modified = sum(1 for f in self.files if not f.created and not f.deleted)
        deleted = sum(1 for f in self.files if f.deleted)
        bits = []
        if created:
            bits.append(f"{created} created")
        if modified:
            bits.append(f"{modified} modified")
        if deleted:
            bits.append(f"{deleted} deleted")
        return ", ".join(bits) or "no file changes"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Checkpoint:
        raw = json.loads(line)
        files = [FileState(**f) for f in raw.pop("files", [])]
        return cls(files=files, **raw)


class CheckpointStore:
    def __init__(self, state_dir: Path, workspace: Path, session_id: str) -> None:
        self.workspace = workspace.resolve()
        self.root = state_dir / "checkpoints"
        self.objects = state_dir / OBJECTS_DIRNAME
        self.manifest = self.root / MANIFEST_NAME
        self.session_id = session_id
        self._counter = 0

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------
    def _object_path(self, digest: str) -> Path:
        # Two-level fan-out keeps any single directory well under the point
        # where NTFS directory enumeration slows down.
        return self.objects / digest[:2] / digest[2:]

    def _store_blob(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self._object_path(digest)
        if target.exists():
            return digest  # content-addressed: identical content is stored once
        self._ensure_dirs()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name then rename, so a crash never leaves a truncated
        # object under a hash that claims to describe complete content.
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return digest

    def _load_blob(self, digest: str) -> bytes:
        path = self._object_path(digest)
        if not path.is_file():
            raise DeppSeekError(f"Checkpoint object {digest[:12]} is missing from the store")
        return path.read_bytes()

    # ------------------------------------------------------------------
    # Snapshot / commit
    # ------------------------------------------------------------------
    def snapshot(
        self,
        paths: Iterable[Path],
        *,
        tool: str,
        description: str,
        step: int = 0,
    ) -> Checkpoint:
        """Record the current state of `paths` before they are mutated."""
        self._counter += 1
        checkpoint = Checkpoint(
            id=f"cp{self._counter:04d}-{int(time.time() * 1000) % 1_000_000:06d}",
            created_at=time.time(),
            session_id=self.session_id,
            step=step,
            tool=tool,
            description=description,
        )

        for path in paths:
            rel = self._relative(path)
            if not path.exists():
                checkpoint.files.append(FileState(path=rel, before=None))
                continue
            if path.is_dir():
                # Directories carry no content; recreating them on undo is
                # implied by restoring the files inside.
                continue
            size = path.stat().st_size
            if size > MAX_SNAPSHOT_BYTES:
                checkpoint.files.append(
                    FileState(
                        path=rel,
                        before=None,
                        before_size=size,
                        skipped_reason=f"file is {size:,} bytes, above the {MAX_SNAPSHOT_BYTES:,} snapshot limit",
                    )
                )
                continue
            digest = self._store_blob(path.read_bytes())
            checkpoint.files.append(FileState(path=rel, before=digest, before_size=size))

        return checkpoint

    def commit(self, checkpoint: Checkpoint) -> Checkpoint:
        """Record post-mutation state and append the checkpoint to the manifest."""
        for entry in checkpoint.files:
            target = self.workspace / entry.path
            if target.is_file():
                entry.after = self._store_blob(target.read_bytes())
            else:
                entry.after = None

        # A checkpoint where nothing actually changed is noise in the undo stack.
        if all(f.before == f.after for f in checkpoint.files):
            checkpoint.undone = True  # mark inert so undo skips it
        self._append(checkpoint)
        return checkpoint

    def _append(self, checkpoint: Checkpoint) -> None:
        self._ensure_dirs()
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(checkpoint.to_json() + "\n")

    # ------------------------------------------------------------------
    # History / undo
    # ------------------------------------------------------------------
    def history(self, limit: int = 20) -> list[Checkpoint]:
        if not self.manifest.is_file():
            return []
        entries: list[Checkpoint] = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(Checkpoint.from_json(line))
            except (json.JSONDecodeError, TypeError):
                continue  # a torn final line must not hide the rest of the history
        return entries[-limit:]

    def last_undoable(self) -> Checkpoint | None:
        for checkpoint in reversed(self.history(limit=10_000)):
            if not checkpoint.undone:
                return checkpoint
        return None

    def undo(self, checkpoint_id: str | None = None) -> str:
        """Restore the recorded before-state of one checkpoint."""
        entries = self.history(limit=10_000)
        if checkpoint_id:
            target = next((c for c in entries if c.id == checkpoint_id), None)
            if target is None:
                return f"No checkpoint with id {checkpoint_id!r}. Use /checkpoints to list them."
            if target.undone:
                return f"Checkpoint {checkpoint_id} was already undone."
        else:
            # Must be selected from `entries`, not via last_undoable(), which
            # re-reads the manifest and returns a *different* object. Flipping
            # `undone` on that copy would be discarded by the rewrite below,
            # letting the same checkpoint be undone repeatedly.
            target = next((c for c in reversed(entries) if not c.undone), None)
            if target is None:
                return "Nothing to undo."

        restored: list[str] = []
        removed: list[str] = []
        failed: list[str] = []

        for entry in target.files:
            path = self.workspace / entry.path
            try:
                if entry.before is None:
                    if entry.skipped_reason:
                        failed.append(f"{entry.path} ({entry.skipped_reason})")
                        continue
                    # The file did not exist before, so undo removes it.
                    if path.is_file():
                        path.unlink()
                        removed.append(entry.path)
                else:
                    data = self._load_blob(entry.before)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    restored.append(entry.path)
            except (OSError, DeppSeekError) as exc:
                failed.append(f"{entry.path} ({exc})")

        target.undone = True
        self._rewrite_manifest(entries)

        lines = [f"Undid checkpoint {target.id} ({target.tool}: {target.description})"]
        if restored:
            lines.append(f"  restored: {', '.join(restored[:20])}")
        if removed:
            lines.append(f"  removed:  {', '.join(removed[:20])}")
        if failed:
            lines.append(f"  FAILED:   {'; '.join(failed[:20])}")
        return "\n".join(lines)

    def _rewrite_manifest(self, entries: list[Checkpoint]) -> None:
        """Rewrite the manifest atomically after flipping an `undone` flag."""
        self._ensure_dirs()
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(
            "".join(c.to_json() + "\n" for c in entries), encoding="utf-8"
        )
        os.replace(tmp, self.manifest)

    def describe_history(self, limit: int = 20) -> str:
        entries = self.history(limit)
        if not entries:
            return "No checkpoints recorded yet."
        lines = [f"{'id':<20} {'time':<10} {'tool':<14} changes"]
        for checkpoint in entries:
            mark = " (undone)" if checkpoint.undone else ""
            lines.append(
                f"{checkpoint.id:<20} {checkpoint.when:<10} {checkpoint.tool:<14} "
                f"{checkpoint.summary()}{mark}"
            )
        return "\n".join(lines)

    def prune(self, keep: int = 200) -> str:
        """Trim old checkpoints and delete unreferenced objects."""
        entries = self.history(limit=10_000)
        if len(entries) <= keep:
            kept = entries
        else:
            kept = entries[-keep:]
        self._rewrite_manifest(kept)

        referenced: set[str] = set()
        for checkpoint in kept:
            for entry in checkpoint.files:
                if entry.before:
                    referenced.add(entry.before)
                if entry.after:
                    referenced.add(entry.after)

        freed = 0
        if self.objects.is_dir():
            for shard in self.objects.iterdir():
                if not shard.is_dir():
                    continue
                for blob in shard.iterdir():
                    digest = shard.name + blob.name
                    if digest not in referenced:
                        freed += blob.stat().st_size
                        blob.unlink()
                if not any(shard.iterdir()):
                    shard.rmdir()

        return (
            f"Kept {len(kept)} checkpoints, dropped {len(entries) - len(kept)}, "
            f"freed {freed:,} bytes of objects."
        )

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return path.as_posix()

    def clear(self) -> None:
        """Remove all checkpoint state. Used by tests and /reset."""
        if self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)
        if self.objects.is_dir():
            shutil.rmtree(self.objects, ignore_errors=True)


class _NullCheckpointStore:
    """No-op store, for read-only sessions and dry runs."""

    def snapshot(self, paths: Iterable[Path], **_: Any) -> Checkpoint:
        return Checkpoint("cp-null", time.time(), "none", 0, "none", "checkpointing disabled")

    def commit(self, checkpoint: Checkpoint) -> Checkpoint:
        return checkpoint

    def undo(self, checkpoint_id: str | None = None) -> str:
        return "Checkpointing is disabled for this session."

    def describe_history(self, limit: int = 20) -> str:
        return "Checkpointing is disabled for this session."
