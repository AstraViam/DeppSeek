"""PowerShell and process execution.

Notes specific to Windows, all of which v1 got wrong or ignored:

* **PowerShell 7 first.** `pwsh` is preferred over `powershell.exe` when present:
  Windows PowerShell 5.1 defaults to a non-UTF-8 output encoding, which mangles
  any output containing units, Greek letters, or box-drawing characters. That is
  most scientific output.
* **Script files, not -Command.** Passing a multi-line command through
  `-Command` means fighting two layers of quoting. Writing the command to a
  temporary `.ps1` and invoking it with `-File` removes the quoting problem
  entirely and makes the executed text auditable.
* **Interleaved output is separated.** stdout and stderr are captured
  separately, and a non-zero exit is reported as a fact rather than buried, so
  the model cannot claim a run succeeded when it did not.
* **Output is bounded from both ends.** A run that prints 200k lines keeps its
  head and its tail; the middle is dropped. The tail is where the traceback is.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool

MAX_CAPTURE_CHARS = 40_000
IS_WINDOWS = os.name == "nt"

# Preamble prepended to every PowerShell script. Stopping on error is what makes
# a failed step visible instead of letting the script run on to a misleading
# "success" at the end.
PS_PREAMBLE = """\
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
"""


def powershell_executable() -> str | None:
    """Locate a PowerShell, preferring 7+ for its UTF-8 default."""
    for candidate in ("pwsh", "powershell"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def format_process_output(
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    duration_s: float,
    *,
    timed_out: bool = False,
) -> str:
    """Render a process result so success and failure are unambiguous."""
    status = (
        f"TIMED OUT after {duration_s:.1f}s"
        if timed_out
        else ("SUCCESS (exit 0)" if returncode == 0 else f"FAILED (exit {returncode})")
    )
    parts = [f"{status}  [{duration_s:.1f}s]", f"Command: {command}"]
    if stdout.strip():
        parts.append("--- stdout ---")
        parts.append(_clamp(stdout))
    if stderr.strip():
        parts.append("--- stderr ---")
        parts.append(_clamp(stderr))
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    if returncode != 0 and not timed_out:
        parts.append(
            "The command failed. Diagnose from the output above before retrying; "
            "do not report this step as successful."
        )
    return "\n".join(parts)


def _clamp(text: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    """Keep the head and tail of long output; the tail holds the traceback."""
    if len(text) <= limit:
        return text.rstrip()
    head = text[: limit // 3]
    tail = text[-(2 * limit) // 3 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [{omitted:,} characters omitted from the middle] ...\n{tail}".rstrip()


def run_process(
    command: list[str],
    cwd: Path,
    timeout: int,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, float, bool]:
    """Run a subprocess, returning `(code, stdout, stderr, seconds, timed_out)`."""
    import time

    started = time.monotonic()
    merged_env = {**os.environ, **(env or {})}
    # Force UTF-8 in child Python processes so tracebacks with non-ASCII survive.
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    merged_env.setdefault("PYTHONUTF8", "1")

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=merged_env,
            # Never inherit the terminal's stdin: a command that prompts would
            # otherwise hang the agent forever with no visible cause.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return (
            -1,
            _decode(exc.stdout),
            _decode(exc.stderr),
            elapsed,
            True,
        )
    except FileNotFoundError as exc:
        raise ToolError(
            f"Executable not found: {command[0]}. Check that it is installed and on PATH. ({exc})"
        ) from exc

    return (
        completed.returncode,
        completed.stdout or "",
        completed.stderr or "",
        time.monotonic() - started,
        False,
    )


def _decode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@tool(command_arg="command", slow=True)
def run_powershell(ctx: ToolContext, command: str, timeout: int = 120) -> ToolResult:
    """Run a PowerShell command in the workspace directory.

    Use this for builds, package management, file utilities, and anything with no
    dedicated tool. Prefer `run_python`, `run_tests`, and the git tools where
    they apply: they give better-structured output.

    The command runs with $ErrorActionPreference = 'Stop', so a failing cmdlet
    stops the script rather than continuing to a misleading success.

    Args:
        command: PowerShell to execute. Multi-line is fine.
        timeout: Seconds before the process is killed, 1 to 1800.
    """
    shell = powershell_executable()
    if shell is None:
        raise ToolError(
            "No PowerShell found on PATH. This build targets Windows PowerShell "
            f"(detected platform: {platform.system()}). Install PowerShell 7 "
            "(`winget install Microsoft.PowerShell`) or run on Windows."
        )

    timeout = max(1, min(timeout, ctx.config.budget.max_timeout_s))

    # Writing the script to a file sidesteps nested-quoting problems entirely and
    # leaves an auditable record of exactly what ran.
    script_dir = ctx.workspace / ".deppseek" / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    handle, script_path = tempfile.mkstemp(suffix=".ps1", dir=str(script_dir), text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8-sig") as fh:
            fh.write(PS_PREAMBLE)
            fh.write(command)
            fh.write("\nexit $LASTEXITCODE\n" if "exit" not in command else "\n")

        code, stdout, stderr, elapsed, timed_out = run_process(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", script_path],
            ctx.workspace,
            timeout,
        )
    finally:
        # Best-effort cleanup: a locked or already-removed script must not mask
        # the process result we came here for.
        with suppress(OSError):
            os.unlink(script_path)

    body = format_process_output(
        command.splitlines()[0][:100], code, stdout, stderr, elapsed, timed_out=timed_out
    )
    status = "timeout" if timed_out else ("ok" if code == 0 else f"exit {code}")
    return ToolResult(
        content=body,
        is_error=timed_out or code != 0,
        display=f"powershell: {status} ({elapsed:.1f}s)",
    )


@tool(path_arg="script", slow=True)
def run_python(
    ctx: ToolContext,
    script: str,
    args: list[str] | None = None,
    timeout: int = 300,
) -> ToolResult:
    """Run a Python script from the workspace with the agent's interpreter.

    Args:
        script: Workspace-relative path to a .py file.
        args: Command-line arguments to pass to the script.
        timeout: Seconds before the process is killed.
    """
    target, relative, _ = ctx.resolve(script)
    if not target.is_file():
        raise ToolError(f"Python script not found: {script}")
    if target.suffix.lower() not in {".py", ".pyw"}:
        raise ToolError(f"{script} is not a Python file.")

    timeout = max(1, min(timeout, ctx.config.budget.max_timeout_s))
    command = [sys.executable, str(target), *(str(a) for a in (args or []))]

    code, stdout, stderr, elapsed, timed_out = run_process(command, ctx.workspace, timeout)
    body = format_process_output(
        f"python {relative} {' '.join(args or [])}".strip(),
        code, stdout, stderr, elapsed, timed_out=timed_out,
    )
    status = "timeout" if timed_out else ("ok" if code == 0 else f"exit {code}")
    return ToolResult(
        content=body,
        is_error=timed_out or code != 0,
        display=f"python {relative}: {status} ({elapsed:.1f}s)",
    )


@tool(slow=True)
def run_python_snippet(ctx: ToolContext, code: str, timeout: int = 120) -> ToolResult:
    """Run a short Python snippet in the workspace without creating a file.

    Use this for quick numerical checks, unit conversions, and sanity tests on a
    result. Anything worth keeping should be written to a real file instead.

    Args:
        code: Python source to execute.
        timeout: Seconds before the process is killed.
    """
    timeout = max(1, min(timeout, ctx.config.budget.max_timeout_s))
    script_dir = ctx.workspace / ".deppseek" / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    handle, script_path = tempfile.mkstemp(suffix=".py", dir=str(script_dir), text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(code)
        result = run_process([sys.executable, script_path], ctx.workspace, timeout)
    finally:
        with suppress(OSError):
            os.unlink(script_path)

    code_rc, stdout, stderr, elapsed, timed_out = result
    body = format_process_output(
        "python snippet", code_rc, stdout, stderr, elapsed, timed_out=timed_out
    )
    return ToolResult(
        content=body,
        is_error=timed_out or code_rc != 0,
        display=f"snippet: {'ok' if code_rc == 0 else 'failed'} ({elapsed:.1f}s)",
    )


@tool(slow=True)
def run_tests(
    ctx: ToolContext,
    target: str = "",
    extra_args: str = "",
    timeout: int = 600,
) -> ToolResult:
    """Run the project's pytest suite and report failures.

    Args:
        target: Optional test path or node id, e.g. "tests/test_solver.py::test_cfl".
        extra_args: Extra pytest arguments, e.g. "-k reynolds -x".
        timeout: Seconds before the run is killed.
    """
    timeout = max(1, min(timeout, ctx.config.budget.max_timeout_s))
    command = [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short"]
    if target:
        _, _, outside = ctx.resolve(target.split("::")[0])
        if outside:
            raise ToolError(f"Test target {target} resolves outside the workspace.")
        command.append(target)
    if extra_args:
        import shlex

        command.extend(shlex.split(extra_args))

    code, stdout, stderr, elapsed, timed_out = run_process(command, ctx.workspace, timeout)
    combined = f"{stdout}\n{stderr}".strip()

    if code == 5:
        return ToolResult(
            content=f"pytest collected no tests ({elapsed:.1f}s).\n{_clamp(combined)}",
            display="pytest: no tests collected",
        )
    body = format_process_output(
        " ".join(command[2:]), code, stdout, stderr, elapsed, timed_out=timed_out
    )
    summary = _extract_pytest_summary(combined)
    return ToolResult(
        content=body,
        is_error=timed_out or code != 0,
        display=f"pytest: {summary}",
    )


def _extract_pytest_summary(output: str) -> str:
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip("= ").strip()[:100]
    return "no summary line"
