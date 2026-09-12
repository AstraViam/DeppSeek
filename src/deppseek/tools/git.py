"""Git tools.

The v1 `git_commit` ran `git add -- .` and committed everything dirty in the
tree. Under autonomous operation that is actively dangerous: it sweeps up
unrelated work in progress, editor backups, and large result files, and it
produces a commit nobody can review. Here staging is explicit, and a commit
without a path list stages only files the agent itself modified this session.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool
from .shell import run_process

GIT_TIMEOUT = 60


def git_available() -> bool:
    return shutil.which("git") is not None


def _git(ctx: ToolContext, *args: str, timeout: int = GIT_TIMEOUT) -> tuple[int, str, str]:
    if not git_available():
        raise ToolError("git is not installed or not on PATH.")
    code, stdout, stderr, _, timed_out = run_process(
        ["git", *args], ctx.workspace, timeout
    )
    if timed_out:
        raise ToolError(f"git {' '.join(args)} timed out after {timeout}s")
    return code, stdout, stderr


def _require_repo(ctx: ToolContext) -> None:
    code, stdout, _ = _git(ctx, "rev-parse", "--is-inside-work-tree")
    if code != 0 or stdout.strip() != "true":
        raise ToolError(
            f"{ctx.workspace} is not a git repository. Run `git init` first if you "
            f"want version control here."
        )


@tool()
def git_status(ctx: ToolContext) -> ToolResult:
    """Show the current branch and the state of the working tree."""
    _require_repo(ctx)
    _, branch, _ = _git(ctx, "rev-parse", "--abbrev-ref", "HEAD")
    code, stdout, stderr = _git(ctx, "status", "--short", "--branch", "--untracked-files=normal")
    if code != 0:
        raise ToolError(f"git status failed: {stderr.strip()}")

    lines = [line for line in stdout.splitlines() if line.strip()]
    body = "\n".join(lines) if lines else "working tree clean"
    return ToolResult(
        content=f"Branch: {branch.strip()}\n{body}",
        display=f"git status: {branch.strip()}, {max(0, len(lines) - 1)} change(s)",
    )


@tool()
def git_diff(ctx: ToolContext, path: str = "", staged: bool = False) -> ToolResult:
    """Show the diff of uncommitted changes.

    Args:
        path: Optional workspace-relative path to limit the diff to.
        staged: Show staged changes instead of unstaged ones.
    """
    _require_repo(ctx)
    args = ["diff", "--no-ext-diff", "--no-color"]
    if staged:
        args.append("--cached")
    args.append("--")
    if path:
        _, relative, outside = ctx.resolve(path)
        if outside:
            raise ToolError(f"{path} resolves outside the workspace.")
        args.append(relative)
    else:
        args.append(".")

    code, stdout, stderr = _git(ctx, *args)
    if code != 0:
        raise ToolError(f"git diff failed: {stderr.strip()}")
    if not stdout.strip():
        scope = "staged" if staged else "unstaged"
        return ToolResult(content=f"No {scope} changes.", display=f"git diff: no {scope} changes")
    changed = sum(1 for line in stdout.splitlines() if line.startswith("diff --git"))
    return ToolResult(content=stdout, display=f"git diff: {changed} file(s)")


@tool()
def git_log(ctx: ToolContext, count: int = 15, path: str = "") -> ToolResult:
    """Show recent commits, most recent first.

    Args:
        count: How many commits to show, 1 to 100.
        path: Optional path to limit history to.
    """
    _require_repo(ctx)
    count = max(1, min(count, 100))
    args = ["log", f"-{count}", "--pretty=format:%h %ad %an: %s", "--date=short"]
    if path:
        _, relative, outside = ctx.resolve(path)
        if outside:
            raise ToolError(f"{path} resolves outside the workspace.")
        args += ["--", relative]

    code, stdout, stderr = _git(ctx, *args)
    if code != 0:
        raise ToolError(f"git log failed: {stderr.strip()}")
    return ToolResult(
        content=stdout or "(no commits yet)",
        display=f"git log: {len(stdout.splitlines())} commit(s)",
    )


@tool(mutates=False)
def git_stage(ctx: ToolContext, paths: list[str]) -> ToolResult:
    """Stage specific files for the next commit.

    Args:
        paths: Workspace-relative file paths to stage. Wildcards are not expanded;
            use glob_files first if you need them.
    """
    _require_repo(ctx)
    if not paths:
        raise ToolError("No paths given. Staging requires an explicit file list.")

    relatives: list[str] = []
    for raw in paths:
        _, relative, outside = ctx.resolve(raw)
        if outside:
            raise ToolError(f"{raw} resolves outside the workspace; refusing to stage it.")
        relatives.append(relative)

    code, _, stderr = _git(ctx, "add", "--", *relatives)
    if code != 0:
        raise ToolError(f"git add failed: {stderr.strip()}")
    return ToolResult(
        content=f"Staged {len(relatives)} path(s):\n" + "\n".join(f"  {r}" for r in relatives),
        display=f"git stage: {len(relatives)} path(s)",
    )


@tool()
def git_commit(ctx: ToolContext, message: str, paths: list[str] | None = None) -> ToolResult:
    """Create a commit from specific files.

    Staging is always explicit. When `paths` is omitted, only files this session
    actually modified are staged, never the whole working tree. This keeps
    unrelated work in progress out of the agent's commits.

    Args:
        message: Commit message. First line should be a concise summary.
        paths: Workspace-relative paths to commit. Omit to use the files this
            session modified.
    """
    _require_repo(ctx)
    if not message.strip():
        raise ToolError("A commit message is required.")

    if paths:
        targets: list[str] = []
        for raw in paths:
            _, relative, outside = ctx.resolve(raw)
            if outside:
                raise ToolError(f"{raw} resolves outside the workspace.")
            targets.append(relative)
    else:
        targets = sorted(ctx.state.get("modified_paths", set()))
        if not targets:
            raise ToolError(
                "No paths given and this session has not modified any files yet. "
                "Pass an explicit `paths` list if you intend to commit pre-existing "
                "changes; this tool will not stage the whole working tree."
            )

    existing = [t for t in targets if (ctx.workspace / t).exists()]
    deleted = [t for t in targets if not (ctx.workspace / t).exists()]

    code, _, stderr = _git(ctx, "add", "--", *targets)
    if code != 0:
        raise ToolError(f"git add failed: {stderr.strip()}")

    code, stdout, stderr = _git(ctx, "commit", "-m", message)
    if code != 0:
        combined = f"{stdout}\n{stderr}".strip()
        if "nothing to commit" in combined.lower():
            return ToolResult(
                content="Nothing to commit: the staged content matches HEAD.",
                display="git commit: nothing to commit",
            )
        raise ToolError(f"git commit failed:\n{combined}")

    _, head, _ = _git(ctx, "rev-parse", "--short", "HEAD")
    detail = (
        f"Committed {head.strip()}: {message.splitlines()[0]}\n"
        f"  {len(existing)} file(s) changed"
        + (f", {len(deleted)} deleted" if deleted else "")
        + "\n" + "\n".join(f"  {t}" for t in targets[:30])
    )
    return ToolResult(content=detail, display=f"git commit {head.strip()}")
