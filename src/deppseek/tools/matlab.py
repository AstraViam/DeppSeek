"""MATLAB integration, with two execution paths.

v1 shelled out to `matlab -batch` once per call. Every call therefore paid the
full MATLAB startup cost (10-40 s on a typical laptop) and started from an empty
workspace, so a variable computed in one step was gone by the next. For
interactive engineering work that is close to unusable: you cannot load a mesh,
inspect it, and then operate on it.

Two paths are provided and selected automatically:

**Warm engine** -- when `matlab.engine` imports, one MATLAB session is started
and kept. Startup is paid once, variables persist across tool calls, and output
is captured per call. This is the good path.

**Batch with workspace persistence** -- when the engine is unavailable, each
call runs `matlab -batch`, but the base workspace is saved to a `.mat` file
afterwards and reloaded before the next call. Variables still persist; only
startup cost and figure-handle continuity are lost. Replaying a command journal
was considered and rejected: it is quadratic in session length and re-executes
side effects.

MathWorks pins each engine release to one MATLAB release *and* caps the
interpreter version. For MATLAB R2026a the engine is `matlabengine==26.1.x`,
which declares `python_requires >=3.9,<3.14`. On a newer interpreter the import
simply fails and the batch path is used, which `matlab_status` explains.
"""

from __future__ import annotations

import io
import os
import shutil
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool
from .shell import format_process_output, run_process

ENGINE_STATE_KEY = "matlab_engine"
FIGURE_DIRNAME = "figures"
WORKSPACE_MAT = "workspace.mat"
JOURNAL_NAME = "journal.m"


@dataclass
class MatlabProbe:
    """What is actually available on this machine."""

    engine_importable: bool
    engine_error: str
    executable: str | None
    python_version: str
    engine_supported_here: bool

    def describe(self) -> str:
        lines = [f"MATLAB executable: {self.executable or 'NOT FOUND on PATH'}"]
        if self.engine_importable:
            lines.append("MATLAB Engine API: available (warm workspace, variables persist)")
        else:
            lines.append(f"MATLAB Engine API: unavailable ({self.engine_error})")
            if not self.engine_supported_here:
                lines.append(
                    f"  Python {self.python_version} is outside the range the current "
                    f"MATLAB engine supports "
                    f"(>={ENGINE_MIN_PYTHON[0]}.{ENGINE_MIN_PYTHON[1]}, "
                    f"<{ENGINE_MAX_PYTHON_EXCLUSIVE[0]}.{ENGINE_MAX_PYTHON_EXCLUSIVE[1]}). "
                    f"The engine cannot be installed on this interpreter."
                )
                lines.append(
                    f"  To get a warm MATLAB workspace, create the agent's virtual "
                    f"environment on Python "
                    f"{ENGINE_MAX_PYTHON_EXCLUSIVE[0]}."
                    f"{ENGINE_MAX_PYTHON_EXCLUSIVE[1] - 1} or earlier and run:"
                )
                lines.append("    python -m pip install matlabengine")
            else:
                lines.append("  Install it with:  python -m pip install matlabengine")
            lines.append(
                "  Without it, each MATLAB call pays full startup cost; variables "
                "still persist via a saved workspace .mat file."
            )
        return "\n".join(lines)


# The interpreter range the current MATLAB engine release binds to. For MATLAB
# R2026a this is matlabengine 26.1.x, whose setup.py declares
# python_requires=">=3.9, <3.14". Outside this range the engine cannot be
# installed at all, so suggesting `pip install matlabengine` would be useless
# advice; probe_matlab says what to do instead.
ENGINE_MIN_PYTHON = (3, 9)
ENGINE_MAX_PYTHON_EXCLUSIVE = (3, 14)


def probe_matlab(
    executable: str = "matlab",
    version_info: tuple[int, int] | None = None,
) -> MatlabProbe:
    """Report what MATLAB support is actually available.

    `version_info` is injectable so the unsupported-interpreter path can be
    tested without monkeypatching sys.
    """
    import sys

    major_minor = version_info or sys.version_info[:2]
    version = f"{major_minor[0]}.{major_minor[1]}"
    supported = ENGINE_MIN_PYTHON <= tuple(major_minor) < ENGINE_MAX_PYTHON_EXCLUSIVE

    engine_importable = False
    engine_error = ""
    try:
        import matlab.engine  # noqa: F401

        engine_importable = True
    except ImportError as exc:
        engine_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a broken install must not crash startup
        engine_error = f"{type(exc).__name__}: {exc}"

    found = shutil.which(executable)
    if not found and os.name == "nt":
        found = _find_matlab_on_windows()

    return MatlabProbe(
        engine_importable=engine_importable,
        engine_error=engine_error or "not installed",
        executable=found,
        python_version=version,
        engine_supported_here=supported,
    )


def _find_matlab_on_windows() -> str | None:
    """Look in the default install location when MATLAB is not on PATH.

    A default MATLAB install does not add itself to PATH, so `where matlab`
    failing is the common case rather than the exceptional one.
    """
    roots = [Path(r"C:\Program Files\MATLAB"), Path(r"C:\Program Files (x86)\MATLAB")]
    candidates: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for release in root.iterdir():
                exe = release / "bin" / "matlab.exe"
                if exe.is_file():
                    candidates.append((release.name, exe))
        except OSError:
            continue
    if not candidates:
        return None
    # Newest release wins: R2026a sorts after R2025b under a plain string sort.
    candidates.sort(reverse=True)
    return str(candidates[0][1])


def matlab_dir(ctx: ToolContext) -> Path:
    path = ctx.workspace / ".deppseek" / "matlab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(ctx: ToolContext) -> Path:
    path = ctx.workspace / ".deppseek" / FIGURE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Warm engine
# ---------------------------------------------------------------------------

class WarmEngine:
    """A long-lived MATLAB session."""

    def __init__(self, workspace: Path) -> None:
        import matlab.engine

        self.workspace = workspace
        started = time.monotonic()
        self.engine = matlab.engine.start_matlab("-nodesktop")
        self.engine.cd(str(workspace), nargout=0)
        self.startup_s = time.monotonic() - started

    def run(self, command: str, capture_figures_to: Path | None) -> tuple[str, str, list[str]]:
        """Execute `command`, returning `(stdout, stderr, figure_paths)`."""
        out, err = io.StringIO(), io.StringIO()
        try:
            self.engine.eval(command, nargout=0, stdout=out, stderr=err)
        except Exception as exc:  # noqa: BLE001 - MATLABExecutionError and friends
            err.write(f"\n{type(exc).__name__}: {exc}")

        figures: list[str] = []
        if capture_figures_to is not None:
            figures = self._export_figures(capture_figures_to)
        return out.getvalue(), err.getvalue(), figures

    def _export_figures(self, target: Path) -> list[str]:
        """Save every open figure to PNG and close it.

        Figures are the actual output of most CFD and battery work, so leaving
        them open and invisible inside a headless MATLAB session wastes them.
        """
        stamp = time.strftime("%H%M%S")
        script = textwrap.dedent(f"""
            figs = findall(groot, 'Type', 'figure');
            saved = strings(0,1);
            for k = numel(figs):-1:1
                f = figs(k);
                name = fullfile('{_escape(str(target))}', sprintf('fig_%s_%02d.png', '{stamp}', k));
                try
                    exportgraphics(f, name, 'Resolution', 150);
                catch
                    saveas(f, name);
                end
                saved(end+1) = string(name);
                close(f);
            end
            deppseek_saved_figures = saved;
        """).strip()
        out, err = io.StringIO(), io.StringIO()
        try:
            self.engine.eval(script, nargout=0, stdout=out, stderr=err)
            values = self.engine.workspace["deppseek_saved_figures"]
        except Exception:  # noqa: BLE001 - figure export is best-effort
            return []
        if isinstance(values, str):
            return [values] if values else []
        try:
            return [str(v) for v in values if str(v)]
        except TypeError:
            return []

    def workspace_variables(self) -> str:
        out, err = io.StringIO(), io.StringIO()
        try:
            self.engine.eval("whos", nargout=0, stdout=out, stderr=err)
        except Exception as exc:  # noqa: BLE001
            return f"Could not list variables: {exc}"
        return out.getvalue() or "(workspace is empty)"

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:  # noqa: BLE001 - shutting down must never raise
            pass


def _escape(path: str) -> str:
    """Escape a Windows path for embedding in a MATLAB single-quoted string."""
    return path.replace("'", "''")


def get_engine(ctx: ToolContext) -> WarmEngine | None:
    """Return the warm engine, starting it on first use."""
    if not ctx.config.matlab.prefer_engine:
        return None
    cached = ctx.state.get(ENGINE_STATE_KEY)
    if cached is not None:
        return cached if isinstance(cached, WarmEngine) else None

    probe = probe_matlab(ctx.config.matlab.exe)
    if not probe.engine_importable:
        ctx.state[ENGINE_STATE_KEY] = False  # do not retry the import every call
        return None
    try:
        engine = WarmEngine(ctx.workspace)
    except Exception as exc:  # noqa: BLE001 - a failed start falls back to batch
        ctx.state[ENGINE_STATE_KEY] = False
        ctx.state["matlab_engine_error"] = str(exc)
        return None
    ctx.state[ENGINE_STATE_KEY] = engine
    return engine


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool()
def matlab_status(ctx: ToolContext) -> ToolResult:
    """Report which MATLAB execution path is in use and why.

    Call this once at the start of MATLAB work so you know whether variables
    persist cheaply (warm engine) or through a saved workspace file (batch).
    """
    probe = probe_matlab(ctx.config.matlab.exe)
    lines = [probe.describe()]

    cached = ctx.state.get(ENGINE_STATE_KEY)
    if isinstance(cached, WarmEngine):
        lines.append(f"\nWarm session is running (started in {cached.startup_s:.1f}s).")
    elif cached is False:
        error = ctx.state.get("matlab_engine_error")
        lines.append(f"\nUsing the batch path{f' after a failed engine start: {error}' if error else '.'}")
        lines.append(
            "Variables persist between calls via a saved workspace file, so you can "
            "still build state across steps. Each call pays MATLAB startup cost."
        )
    return ToolResult(content="\n".join(lines), display="matlab status")


@tool(command_arg="command", slow=True)
def run_matlab(ctx: ToolContext, command: str, timeout: int = 300) -> ToolResult:
    """Run MATLAB code in a workspace that persists across calls.

    Variables you create stay available to later calls. Open figures are exported
    to PNG under .deppseek/figures/ and closed, and their paths are reported.

    Write plain MATLAB statements. Do not wrap them in a function definition.

    Args:
        command: MATLAB code to execute.
        timeout: Seconds before the call is abandoned.
    """
    timeout = max(1, min(timeout, ctx.config.budget.max_timeout_s))
    capture = figure_dir(ctx) if ctx.config.matlab.capture_figures else None

    engine = get_engine(ctx)
    if engine is not None:
        return _run_with_engine(ctx, engine, command, capture)
    return _run_with_batch(ctx, command, timeout, capture)


def _run_with_engine(
    ctx: ToolContext, engine: WarmEngine, command: str, capture: Path | None
) -> ToolResult:
    started = time.monotonic()
    stdout, stderr, figures = engine.run(command, capture)
    elapsed = time.monotonic() - started

    _append_journal(ctx, command, failed=bool(stderr.strip()))

    failed = bool(stderr.strip())
    parts = [
        ("FAILED" if failed else "OK") + f"  [{elapsed:.1f}s, warm engine]",
        f"Command: {command.splitlines()[0][:100]}",
    ]
    if stdout.strip():
        parts += ["--- output ---", stdout.rstrip()]
    if stderr.strip():
        parts += ["--- errors ---", stderr.rstrip()]
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output; assign without a semicolon to print a value)")
    if figures:
        relative = [_relative_to(ctx, f) for f in figures]
        parts.append("--- figures saved ---")
        parts += [f"  {r}" for r in relative]
        parts.append("Use inspect_figure to look at one, if a vision model is configured.")
    if failed:
        parts.append("MATLAB reported an error. Fix it before continuing; do not "
                     "report this step as successful.")

    return ToolResult(
        content="\n".join(parts),
        is_error=failed,
        display=f"matlab: {'failed' if failed else 'ok'} ({elapsed:.1f}s, warm)",
    )


def _run_with_batch(
    ctx: ToolContext, command: str, timeout: int, capture: Path | None
) -> ToolResult:
    probe = probe_matlab(ctx.config.matlab.exe)
    if not probe.executable:
        raise ToolError(
            "MATLAB was not found. Set matlab.exe in .deppseek/config.toml or the "
            "MATLAB_EXE environment variable to the full path of matlab.exe, "
            r"for example C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
        )

    work = matlab_dir(ctx)
    mat_file = work / WORKSPACE_MAT
    script_path = work / "step.m"

    # Load the previous workspace, run the command, save the workspace back.
    # `evalin('base', ...)` is avoided: -batch already runs in the base workspace,
    # and wrapping adds a layer that breaks `clear` and `who`.
    preamble = (
        f"if isfile('{_escape(str(mat_file))}')\n"
        f"    load('{_escape(str(mat_file))}');\n"
        f"end\n"
    )
    figure_block = ""
    if capture is not None:
        stamp = time.strftime("%H%M%S")
        figure_block = textwrap.dedent(f"""
            figs = findall(groot, 'Type', 'figure');
            for k = numel(figs):-1:1
                nm = fullfile('{_escape(str(capture))}', sprintf('fig_%s_%02d.png', '{stamp}', k));
                try
                    exportgraphics(figs(k), nm, 'Resolution', 150);
                catch
                    saveas(figs(k), nm);
                end
                fprintf('[figure saved] %s\\n', nm);
                close(figs(k));
            end
        """)
    postamble = (
        f"\nclear figs k nm ans;\n"
        f"save('{_escape(str(mat_file))}');\n"
    )

    script_path.write_text(preamble + command + "\n" + figure_block + postamble, encoding="utf-8")

    code, stdout, stderr, elapsed, timed_out = run_process(
        [probe.executable, "-batch", f"run('{_escape(str(script_path))}')"],
        ctx.workspace,
        timeout,
    )

    _append_journal(ctx, command, failed=code != 0)

    figures = [
        line.split("] ", 1)[1].strip()
        for line in stdout.splitlines()
        if line.startswith("[figure saved] ")
    ]
    body = format_process_output(
        command.splitlines()[0][:100], code, stdout, stderr, elapsed, timed_out=timed_out
    )
    extra = [
        "",
        f"(batch path: MATLAB started fresh and paid {elapsed:.0f}s of startup. "
        f"Variables were saved to {WORKSPACE_MAT} and reload automatically next call. "
        f"Run matlab_status to see how to enable the warm engine.)",
    ]
    if figures:
        extra.append("Figures saved: " + ", ".join(_relative_to(ctx, f) for f in figures))

    return ToolResult(
        content=body + "\n".join(extra),
        is_error=timed_out or code != 0,
        display=f"matlab: {'failed' if code else 'ok'} ({elapsed:.1f}s, batch)",
    )


@tool()
def matlab_workspace(ctx: ToolContext) -> ToolResult:
    """List the variables currently in the MATLAB workspace with sizes and classes.

    Use this to check what state exists before writing code that depends on it,
    rather than recomputing something that is already loaded.
    """
    engine = get_engine(ctx)
    if engine is not None:
        return ToolResult(content=engine.workspace_variables(), display="matlab whos")

    mat_file = matlab_dir(ctx) / WORKSPACE_MAT
    if not mat_file.is_file():
        return ToolResult(
            content="The MATLAB workspace is empty; nothing has been run yet this session.",
            display="matlab workspace: empty",
        )
    return run_matlab(ctx, "whos", timeout=ctx.config.matlab.default_timeout_s)


@tool()
def matlab_reset(ctx: ToolContext, restart_engine: bool = False) -> ToolResult:
    """Clear the persistent MATLAB workspace.

    Args:
        restart_engine: Also shut down and restart the warm MATLAB session.
    """
    messages: list[str] = []
    mat_file = matlab_dir(ctx) / WORKSPACE_MAT
    if mat_file.is_file():
        mat_file.unlink()
        messages.append(f"Deleted {WORKSPACE_MAT}.")

    engine = ctx.state.get(ENGINE_STATE_KEY)
    if isinstance(engine, WarmEngine):
        if restart_engine:
            engine.close()
            ctx.state.pop(ENGINE_STATE_KEY, None)
            messages.append("Shut down the warm MATLAB session; it restarts on next use.")
        else:
            engine.run("clear all; close all;", None)
            messages.append("Cleared all variables and closed all figures in the warm session.")

    return ToolResult(
        content="\n".join(messages) or "Nothing to reset.", display="matlab reset"
    )


def _append_journal(ctx: ToolContext, command: str, *, failed: bool) -> None:
    """Record every command run, as an auditable record of the session.

    This is a reproducibility artefact, not a replay mechanism: research code
    that cannot be traced back to the commands that produced its numbers is not
    worth much.
    """
    journal = matlab_dir(ctx) / JOURNAL_NAME
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    marker = "  (FAILED)" if failed else ""
    try:
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(f"\n%% {stamp}{marker}\n{command.rstrip()}\n")
    except OSError:
        pass


def _relative_to(ctx: ToolContext, path: str) -> str:
    try:
        return Path(path).resolve().relative_to(ctx.workspace).as_posix()
    except (ValueError, OSError):
        return path


def shutdown_matlab(ctx: ToolContext) -> None:
    """Close the warm session at exit, so MATLAB does not linger."""
    engine = ctx.state.get(ENGINE_STATE_KEY)
    if isinstance(engine, WarmEngine):
        engine.close()
        ctx.state.pop(ENGINE_STATE_KEY, None)
