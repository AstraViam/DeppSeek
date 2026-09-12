"""Filesystem tools.

The important departure from v1 is `edit_file`: an exact-string replacement that
must match exactly once. v1 could only overwrite a whole file, which means every
one-line change round-trips the entire file through the model. On a 2000-line
CFD solver that is both expensive and a reliable way to lose code the model did
not bother to reproduce.

Two guardrails that matter under autonomous operation:

* A file must be read before it is overwritten. Blind whole-file writes to a file
  the model has never seen are the single most common way an agent destroys work.
* Edits verify their match count before touching disk, so an ambiguous edit fails
  loudly instead of silently changing the wrong occurrence.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..errors import ToolError
from ..permissions.prompt import diff_stats, unified_diff
from .registry import ToolContext, ToolResult, tool

# Directories that are noise in almost every project. Matched against the
# workspace-*relative* path, not the absolute one: v1 tested every component of
# the absolute path, so a workspace living under any directory named "env" or
# "node_modules" reported its entire tree as ignored.
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".idea", ".vscode", ".ipynb_checkpoints", "dist", "build", ".eggs",
    "site-packages", ".deppseek", ".gradle", "target", "bin", "obj",
    "slprj", "codegen",  # Simulink/MATLAB Coder build output
})

TEXT_EXTENSIONS = frozenset({
    ".py", ".pyw", ".pyi", ".ipynb",
    ".m", ".mlx", ".slx", ".mdl", ".mat",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cu", ".f", ".f90", ".f95",
    ".java", ".js", ".ts", ".tsx", ".jsx", ".cs", ".rs", ".go", ".rb", ".php",
    ".sql", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1", ".psm1",
    ".md", ".markdown", ".rst", ".txt", ".org",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv", ".dat", ".xml", ".html", ".htm", ".css", ".scss",
    ".tex", ".bib", ".cls", ".sty",
    ".gitignore", ".dockerignore", ".editorconfig",
})

BINARY_SNIFF_BYTES = 8192
MAX_FILE_BYTES = 8_000_000
MAX_READ_LINES = 2000
MAX_WRITE_CHARS = 1_200_000


def is_ignored(relative_path: str) -> bool:
    """True when a workspace-relative path lies inside an ignored directory."""
    parts = Path(relative_path).parts
    return any(part in IGNORE_DIRS for part in parts)


def looks_binary(path: Path) -> bool:
    """Sniff for binary content rather than trusting the extension alone."""
    try:
        chunk = path.open("rb").read(BINARY_SNIFF_BYTES)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    # A high proportion of bytes outside printable ASCII and common whitespace
    # indicates binary even without a NUL, e.g. some .mat and image formats.
    if not chunk:
        return False
    printable = sum(1 for b in chunk if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(chunk) < 0.75


def read_text(path: Path) -> str:
    """Read a text file, tolerating the encodings a mixed Windows tree carries."""
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def detect_newline(text: str) -> str:
    """Preserve a file's existing line ending instead of imposing one.

    MATLAB and Visual Studio projects on Windows commonly use CRLF; rewriting a
    file with LF produces a diff touching every line, which buries the real
    change and breaks blame.
    """
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def write_text(path: Path, content: str, newline: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        normalised = normalised.replace("\n", newline)
    # Write then replace, so an interrupted write cannot truncate the original.
    tmp = path.with_name(path.name + ".deppseek-tmp")
    tmp.write_bytes(normalised.encode("utf-8"))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@tool(path_arg="path")
def list_workspace(ctx: ToolContext, path: str = ".") -> ToolResult:
    """List the files and directories directly inside one workspace directory.

    Use this to orient yourself before reading. For a recursive overview use
    `tree`; to find files by name use `glob_files`.

    Args:
        path: Workspace-relative directory. Defaults to the workspace root.
    """
    target, relative, _ = ctx.resolve(path)
    if not target.exists():
        raise ToolError(f"Directory does not exist: {path}")
    if not target.is_dir():
        raise ToolError(f"Not a directory: {path}. Use read_file for files.")

    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        child_rel = child.relative_to(ctx.workspace).as_posix()
        if is_ignored(child_rel):
            continue
        if child.is_dir():
            entries.append(f"  {child.name}/")
        else:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append(f"  {child.name:<48} {size:>12,} bytes")

    body = "\n".join(entries[:800]) or "  (empty)"
    header = f"{relative or '.'}/  ({len(entries)} entries)"
    return ToolResult(content=f"{header}\n{body}", display=header)


@tool(path_arg="path")
def tree(ctx: ToolContext, path: str = ".", max_depth: int = 3, max_entries: int = 400) -> ToolResult:
    """Show a recursive directory tree of the workspace.

    Args:
        path: Subdirectory to start from. Defaults to the workspace root.
        max_depth: How many levels deep to descend, 1 to 10.
        max_entries: Stop after this many entries to keep the output bounded.
    """
    root, relative, _ = ctx.resolve(path)
    if not root.is_dir():
        raise ToolError(f"Not a directory: {path}")

    max_depth = max(1, min(max_depth, 10))
    lines = [f"{relative or ctx.workspace.name}/"]
    truncated = False

    def walk(directory: Path, prefix: str, depth: int) -> None:
        nonlocal truncated
        if depth > max_depth or truncated:
            return
        try:
            children = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except OSError as exc:
            lines.append(f"{prefix}[unreadable: {exc}]")
            return
        children = [
            c for c in children
            if not is_ignored(c.relative_to(ctx.workspace).as_posix())
        ]
        for index, child in enumerate(children):
            if len(lines) >= max_entries:
                truncated = True
                return
            last = index == len(children) - 1
            branch = "`-- " if last else "|-- "
            if child.is_dir():
                lines.append(f"{prefix}{branch}{child.name}/")
                walk(child, prefix + ("    " if last else "|   "), depth + 1)
            else:
                lines.append(f"{prefix}{branch}{child.name}")

    walk(root, "", 1)
    if truncated:
        lines.append(f"... truncated at {max_entries} entries; narrow `path` or lower `max_depth`")
    return ToolResult(content="\n".join(lines), display=f"tree {relative or '.'} ({len(lines)} lines)")


@tool(path_arg="path")
def read_file(
    ctx: ToolContext, path: str, start_line: int = 1, line_count: int = 400
) -> ToolResult:
    """Read a text file with line numbers. Read before you edit.

    Reading a file marks it as seen, which is what allows `write_file` to
    overwrite it later. Line numbers in the output are 1-based and match what
    `edit_file` and stack traces refer to.

    Args:
        path: Workspace-relative file path.
        start_line: First line to return, 1-based.
        line_count: How many lines to return, up to 2000.
    """
    target, relative, _ = ctx.resolve(path)
    if not target.exists():
        raise ToolError(
            f"File does not exist: {path}. Use glob_files or list_workspace to find it."
        )
    if target.is_dir():
        raise ToolError(f"{path} is a directory. Use list_workspace or tree.")

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ToolError(
            f"{path} is {size:,} bytes, above the {MAX_FILE_BYTES:,} byte read limit. "
            f"Use search_workspace to find the relevant lines, then read a range."
        )
    if looks_binary(target):
        raise ToolError(
            f"{path} appears to be binary ({size:,} bytes). Reading it would produce "
            f"noise. If it is a MATLAB .mat file, load it in MATLAB and print a summary."
        )

    text = read_text(target)
    lines = text.splitlines()
    total = len(lines)

    start = max(1, start_line)
    count = max(1, min(line_count, MAX_READ_LINES))
    end = min(total, start + count - 1)
    if start > total:
        return ToolResult(
            content=f"{relative}: requested line {start} but the file has {total} lines.",
            display=f"{relative}: out of range",
        )

    width = len(str(end))
    body = "\n".join(
        f"{number:>{width}}| {line}"
        for number, line in enumerate(lines[start - 1 : end], start=start)
    )

    ctx.read_files.add(relative)

    header = f"{relative} (lines {start}-{end} of {total})"
    footer = ""
    if end < total:
        footer = f"\n... {total - end} more lines. Continue with start_line={end + 1}."
    return ToolResult(
        content=f"{header}\n{body}{footer}",
        display=header,
        touched=(target,),
    )


@tool()
def glob_files(ctx: ToolContext, pattern: str, max_results: int = 200) -> ToolResult:
    """Find files by path pattern, newest first.

    Args:
        pattern: Glob relative to the workspace, e.g. "src/**/*.py" or "**/*.m".
        max_results: Cap on returned paths.
    """
    max_results = max(1, min(max_results, 1000))
    matches: list[tuple[float, str]] = []
    try:
        for found in ctx.workspace.glob(pattern):
            if not found.is_file():
                continue
            relative = found.relative_to(ctx.workspace).as_posix()
            if is_ignored(relative):
                continue
            try:
                mtime = found.stat().st_mtime
            except OSError:
                mtime = 0.0
            matches.append((mtime, relative))
    except (OSError, ValueError) as exc:
        raise ToolError(f"Invalid glob pattern {pattern!r}: {exc}") from exc

    matches.sort(reverse=True)
    trimmed = matches[:max_results]
    if not trimmed:
        return ToolResult(
            content=f"No files match {pattern!r}.", display=f"glob {pattern}: 0"
        )
    body = "\n".join(relative for _, relative in trimmed)
    note = (
        f"\n... {len(matches) - len(trimmed)} more matches omitted"
        if len(matches) > len(trimmed)
        else ""
    )
    return ToolResult(
        content=f"{len(trimmed)} file(s) matching {pattern!r}, newest first:\n{body}{note}",
        display=f"glob {pattern}: {len(trimmed)}",
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

@tool(path_arg="path", mutates=True)
def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    """Create a file, or replace an existing file's entire contents.

    Prefer `edit_file` for changes to an existing file: it is cheaper, it cannot
    accidentally drop code you did not reproduce, and it produces a reviewable
    diff. Overwriting a file you have not read in this session is refused.

    Args:
        path: Workspace-relative file path.
        content: Complete new contents of the file.
    """
    target, relative, _ = ctx.resolve(path)

    if len(content) > MAX_WRITE_CHARS:
        raise ToolError(
            f"Refusing to write {len(content):,} characters in one call "
            f"(limit {MAX_WRITE_CHARS:,}). Split the file or use edit_file."
        )

    existed = target.exists()
    if existed and relative not in ctx.read_files:
        raise ToolError(
            f"{relative} already exists and has not been read in this session. "
            f"Read it first, then either edit_file the specific region or "
            f"write_file the full replacement. This guard exists because blind "
            f"whole-file overwrites silently discard code."
        )

    before = read_text(target) if existed and target.is_file() else ""
    newline = detect_newline(before) if before else os.linesep if os.name == "nt" else "\n"
    write_text(target, content, newline)
    ctx.read_files.add(relative)

    added, removed = diff_stats(before, content)
    verb = "Updated" if existed else "Created"
    detail = f"{verb} {relative} ({len(content):,} chars, +{added}/-{removed} lines)"
    return ToolResult(content=detail, display=detail, touched=(target,))


@tool(path_arg="path", mutates=True)
def edit_file(
    ctx: ToolContext,
    path: str,
    old_text: str,
    new_text: str,
    expected_matches: int = 1,
) -> ToolResult:
    """Replace an exact block of text in a file. The preferred way to change code.

    `old_text` must appear in the file exactly `expected_matches` times. If it
    appears a different number of times the edit is refused and nothing is
    written, so an ambiguous match can never silently change the wrong line.
    Include enough surrounding context to make the match unique.

    Args:
        path: Workspace-relative file path.
        old_text: Exact text to replace, including indentation.
        new_text: Replacement text. Use an empty string to delete.
        expected_matches: How many occurrences you expect to replace.
    """
    target, relative, _ = ctx.resolve(path)
    if not target.is_file():
        raise ToolError(f"File does not exist: {path}")

    if old_text == new_text:
        raise ToolError("old_text and new_text are identical; nothing to do.")

    before = read_text(target)
    found = before.count(old_text)

    if found == 0:
        hint = _near_miss_hint(before, old_text)
        raise ToolError(
            f"old_text was not found in {relative}. Nothing was written.{hint}"
        )
    if found != expected_matches:
        raise ToolError(
            f"old_text appears {found} time(s) in {relative} but expected_matches "
            f"is {expected_matches}. Nothing was written. Either add surrounding "
            f"context to make the match unique, or set expected_matches={found} "
            f"if you intend to replace all of them."
        )

    after = before.replace(old_text, new_text)
    write_text(target, after, detect_newline(before))
    ctx.read_files.add(relative)

    added, removed = diff_stats(before, after)
    diff = unified_diff(before, after, relative, max_lines=ctx.config.ui.max_diff_lines)
    summary = f"Edited {relative}: {found} replacement(s), +{added}/-{removed} lines"
    return ToolResult(content=f"{summary}\n\n{diff}", display=summary, touched=(target,))


@tool(path_arg="path", mutates=True)
def multi_edit(ctx: ToolContext, path: str, edits: list[dict]) -> ToolResult:
    """Apply several exact-text replacements to one file atomically.

    Edits are applied in order against the running text. If any edit fails to
    match, none are written. Use this for a set of related changes to one file
    so the file is never left half-edited.

    Args:
        path: Workspace-relative file path.
        edits: List of objects with keys "old_text", "new_text", and optional
            "expected_matches" (default 1).
    """
    target, relative, _ = ctx.resolve(path)
    if not target.is_file():
        raise ToolError(f"File does not exist: {path}")
    if not edits:
        raise ToolError("edits is empty; nothing to do.")

    before = read_text(target)
    working = before
    applied = 0

    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise ToolError(f"edits[{index}] must be an object with old_text/new_text.")
        old_text = edit.get("old_text")
        new_text = edit.get("new_text")
        if old_text is None or new_text is None:
            raise ToolError(f"edits[{index}] needs both old_text and new_text.")
        expected = int(edit.get("expected_matches", 1))

        found = working.count(old_text)
        if found != expected:
            raise ToolError(
                f"edits[{index}] matched {found} time(s), expected {expected}. "
                f"No edits were written to {relative}. Note that earlier edits in "
                f"this list had already been applied to the working text, so add "
                f"context that accounts for them."
            )
        working = working.replace(old_text, new_text)
        applied += found

    write_text(target, working, detect_newline(before))
    ctx.read_files.add(relative)

    added, removed = diff_stats(before, working)
    diff = unified_diff(before, working, relative, max_lines=ctx.config.ui.max_diff_lines)
    summary = (
        f"Applied {len(edits)} edit(s) to {relative}: {applied} replacement(s), "
        f"+{added}/-{removed} lines"
    )
    return ToolResult(content=f"{summary}\n\n{diff}", display=summary, touched=(target,))


@tool(path_arg="path", mutates=True)
def make_directory(ctx: ToolContext, path: str) -> ToolResult:
    """Create a directory, including any missing parents.

    Args:
        path: Workspace-relative directory path.
    """
    target, relative, _ = ctx.resolve(path)
    if target.exists():
        return ToolResult(content=f"{relative} already exists.", display="already exists")
    target.mkdir(parents=True, exist_ok=True)
    return ToolResult(content=f"Created directory {relative}", display=f"mkdir {relative}")


@tool(path_arg="path", mutates=True)
def delete_path(ctx: ToolContext, path: str, recursive: bool = False) -> ToolResult:
    """Delete a file, or a directory when recursive is true.

    Args:
        path: Workspace-relative path to delete.
        recursive: Required to delete a directory and its contents.
    """
    target, relative, _ = ctx.resolve(path)
    if not target.exists():
        raise ToolError(f"Nothing to delete at {path}.")

    if target.is_dir():
        if not recursive:
            raise ToolError(
                f"{relative} is a directory. Pass recursive=true to delete it and "
                f"everything inside."
            )
        count = sum(1 for _ in target.rglob("*"))
        shutil.rmtree(target)
        detail = f"Deleted directory {relative} and {count} entries inside it"
    else:
        target.unlink()
        detail = f"Deleted file {relative}"
    ctx.read_files.discard(relative)
    return ToolResult(content=detail, display=detail)


@tool(path_arg="source", mutates=True)
def move_path(ctx: ToolContext, source: str, destination: str) -> ToolResult:
    """Move or rename a file or directory inside the workspace.

    Args:
        source: Current workspace-relative path.
        destination: New workspace-relative path.
    """
    src, src_rel, _ = ctx.resolve(source)
    dst, dst_rel, dst_outside = ctx.resolve(destination)

    if dst_outside:
        raise ToolError(f"Destination {destination} resolves outside the workspace.")
    if not src.exists():
        raise ToolError(f"Source does not exist: {source}")
    if dst.exists():
        raise ToolError(f"Destination already exists: {dst_rel}. Delete it first if intended.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    ctx.read_files.discard(src_rel)
    detail = f"Moved {src_rel} -> {dst_rel}"
    return ToolResult(content=detail, display=detail)


def _near_miss_hint(haystack: str, needle: str) -> str:
    """Explain a failed exact match, which is usually whitespace or line endings."""
    if needle.strip() and needle.strip() in haystack:
        return (
            " The text is present but the surrounding whitespace differs. "
            "Re-read the file and copy the indentation exactly."
        )
    collapsed_needle = " ".join(needle.split())
    collapsed_hay = " ".join(haystack.split())
    if collapsed_needle and collapsed_needle in collapsed_hay:
        return (
            " A whitespace-insensitive match exists, so line endings or indentation "
            "differ. Re-read the exact lines before editing."
        )
    first_line = needle.splitlines()[0].strip() if needle.splitlines() else ""
    if first_line and first_line in haystack:
        return f" The first line ({first_line[:60]!r}) is present, so the later lines differ."
    return " Re-read the file; it may have changed since you last saw it."
