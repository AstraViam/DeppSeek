"""Workspace search.

v1 walked `rglob` in pure Python, read every candidate file, and filtered on a
hardcoded directory set. On a repository with a `build/` tree or a conda
environment checked in, that is both slow and wrong: it misses the project's own
`.gitignore` and burns seconds on files nobody wants searched.

This version respects `.gitignore` via pathspec and hands off to ripgrep when it
is installed, falling back to the Python walk otherwise. The fallback is kept
because a chemical-engineering laptop is unlikely to have ripgrep, and the tool
must not simply stop working there.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from ..errors import ToolError
from .fs import MAX_FILE_BYTES, TEXT_EXTENSIONS, is_ignored, looks_binary, read_text
from .registry import ToolContext, ToolResult, tool

MAX_MATCHES = 300
MAX_LINE_CHARS = 400


def load_gitignore(workspace: Path):
    """Build a pathspec matcher from .gitignore files, if pathspec is available."""
    try:
        import pathspec
    except ImportError:
        return None

    patterns: list[str] = []
    for name in (".gitignore", ".git/info/exclude"):
        candidate = workspace / name
        if candidate.is_file():
            try:
                patterns.extend(candidate.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue
    if not patterns:
        return None
    try:
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except Exception:  # noqa: BLE001 - a malformed .gitignore must not break search
        return None


def ripgrep_path() -> str | None:
    return shutil.which("rg")


@tool()
def search_workspace(
    ctx: ToolContext,
    pattern: str,
    glob: str = "",
    case_sensitive: bool = False,
    max_results: int = 120,
    context_lines: int = 0,
) -> ToolResult:
    """Search file contents with a regular expression.

    This is the fastest way to locate a symbol, a physical constant, a TODO, or
    the place a variable is defined. Prefer it over reading files one by one.

    Args:
        pattern: Regular expression to search for.
        glob: Optional file filter, e.g. "*.m" or "src/**/*.py". Empty means all
            text files.
        case_sensitive: Whether the match respects case.
        max_results: Maximum matching lines to return.
        context_lines: Lines of surrounding context to include, 0 to 5.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ToolError(
            f"Invalid regular expression {pattern!r}: {exc}. "
            f"Escape regex metacharacters if you meant a literal string."
        ) from exc

    max_results = max(1, min(max_results, MAX_MATCHES))
    context_lines = max(0, min(context_lines, 5))

    rg = ripgrep_path()
    if rg:
        result = _search_with_ripgrep(
            ctx, rg, pattern, glob, case_sensitive, max_results, context_lines
        )
        if result is not None:
            return result

    return _search_in_python(ctx, pattern, glob, case_sensitive, max_results, context_lines)


def _search_with_ripgrep(
    ctx: ToolContext,
    rg: str,
    pattern: str,
    glob: str,
    case_sensitive: bool,
    max_results: int,
    context_lines: int,
) -> ToolResult | None:
    command = [
        rg, "--line-number", "--no-heading", "--color", "never",
        "--max-columns", str(MAX_LINE_CHARS), "--max-count", str(max_results),
        "--sortr", "modified",
    ]
    if not case_sensitive:
        command.append("--ignore-case")
    if context_lines:
        command += ["--context", str(context_lines)]
    for ignored in sorted(set(str(d) for d in ("node_modules", ".venv", ".deppseek", "slprj"))):
        command += ["--glob", f"!{ignored}/**"]
    if glob:
        command += ["--glob", glob]
    command += ["--regexp", pattern, "."]

    try:
        completed = subprocess.run(
            command,
            cwd=ctx.workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None  # fall back to the Python walk

    # ripgrep exits 1 for "no matches", which is not an error here.
    if completed.returncode not in (0, 1):
        return None

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return ToolResult(
            content=f"No matches for {pattern!r}" + (f" in {glob}" if glob else "") + ".",
            display=f"search: 0 matches",
        )
    trimmed = lines[:max_results]
    note = (
        f"\n... {len(lines) - len(trimmed)} more matching lines omitted; narrow the pattern or glob."
        if len(lines) > len(trimmed)
        else ""
    )
    return ToolResult(
        content=f"{len(trimmed)} match(es) for {pattern!r} (ripgrep):\n" + "\n".join(trimmed) + note,
        display=f"search {pattern!r}: {len(trimmed)} matches",
    )


def _search_in_python(
    ctx: ToolContext,
    pattern: str,
    glob: str,
    case_sensitive: bool,
    max_results: int,
    context_lines: int,
) -> ToolResult:
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    spec = load_gitignore(ctx.workspace)

    results: list[str] = []
    scanned = 0
    candidates = ctx.workspace.rglob(glob) if glob else ctx.workspace.rglob("*")

    for path in candidates:
        if len(results) >= max_results:
            break
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(ctx.workspace).as_posix()
        except ValueError:
            continue
        if is_ignored(relative):
            continue
        if spec is not None and spec.match_file(relative):
            continue
        if not glob and path.suffix.lower() not in TEXT_EXTENSIONS:
            if path.name.lower() not in {"readme", "license", "makefile", "dockerfile"}:
                continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if looks_binary(path):
            continue

        scanned += 1
        try:
            lines = read_text(path).splitlines()
        except OSError:
            continue

        for number, line in enumerate(lines, start=1):
            if not regex.search(line):
                continue
            if context_lines:
                lo = max(0, number - 1 - context_lines)
                hi = min(len(lines), number + context_lines)
                for offset, ctx_line in enumerate(lines[lo:hi], start=lo + 1):
                    marker = ":" if offset == number else "-"
                    results.append(f"{relative}:{offset}{marker} {ctx_line[:MAX_LINE_CHARS]}")
                results.append("--")
            else:
                results.append(f"{relative}:{number}: {line[:MAX_LINE_CHARS]}")
            if len(results) >= max_results:
                break

    if not results:
        return ToolResult(
            content=(
                f"No matches for {pattern!r}" + (f" in {glob}" if glob else "")
                + f". Scanned {scanned} file(s)."
            ),
            display="search: 0 matches",
        )
    return ToolResult(
        content=f"{len(results)} match line(s) for {pattern!r} across {scanned} file(s):\n"
        + "\n".join(results),
        display=f"search {pattern!r}: {len(results)} matches",
    )
