"""Command-line entry point.

Holds the wiring: build the config, provider, permission engine, checkpoint
store, toolbox, and conversation buffer, then either run one task or start the
interactive shell.

v1's shell hardcoded `max_steps=30` for `/agent` while also accepting a
`--max-steps` flag that only applied to the non-interactive path, so the flag
silently did nothing in the mode people actually used. Here there is one budget,
built once, and every path reads it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .agent import AgentLoop
from .checkpoint import CheckpointStore
from .config import AUTONOMY_TIERS, Config, load_config, project_config_path, user_config_path
from .errors import ConfigError, DeppSeekError
from .mcp_client import McpManager
from .permissions import PermissionEngine, rule_from_dict
from .permissions.prompt import Approver
from .prompts import build_system_prompt, load_project_notes
from .providers import DeepSeekProvider, Usage
from .providers.pricing import MODELS
from .session import ConversationBuffer, SessionStore
from .session.tokens import TokenEstimator
from .tools import Toolbox, ToolContext
from .tools.matlab import probe_matlab, shutdown_matlab
from .tools.todo import render_todos
from .ui import InlineUI, build_session, read_input

COMMANDS: dict[str, str] = {
    "help": "show this list",
    "exit": "leave the shell",
    "clear": "start a fresh conversation, keeping the session file",
    "compact": "summarise and drop older turns now",
    "context": "show context usage and the token estimator's state",
    "cost": "show token usage and estimated spend",
    "usage": "alias for /cost",
    "tools": "list available tools",
    "permissions": "show the active permission rules",
    "autonomy": "show or set the autonomy tier",
    "undo": "revert the last file change the agent made",
    "checkpoints": "list recent checkpoints",
    "plan": "show the current plan",
    "sessions": "list saved sessions",
    "resume": "resume a saved session by id",
    "branch": "fork the current session",
    "matlab": "show MATLAB integration status",
    "mcp": "show configured MCP servers and their tools",
    "doctor": "check the environment and report problems",
    "config": "show where config is loaded from",
    "tree": "show the workspace tree",
    "model": "show or set the model",
    "workspace": "show the workspace root",
}


class Controller:
    """Owns one session's objects and the commands that act on them."""

    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.ui = InlineUI(config)
        self.usage = Usage()

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ConfigError(
                f"{config.api_key_env} is not set.\n"
                f"In PowerShell, for this window:\n"
                f'    $env:{config.api_key_env} = "sk-your-key"\n'
                f"To persist it:\n"
                f'    [Environment]::SetEnvironmentVariable("{config.api_key_env}", "sk-your-key", "User")'
            )

        self.provider = DeepSeekProvider(
            api_key=api_key,
            model=config.model,
            base_url=config.base_url,
            thinking=config.thinking,
            reasoning_effort=config.reasoning_effort,
            usage=self.usage,
            on_warning=self.ui.warn,
        )

        rules = tuple(rule_from_dict(r) for r in config.permission_rules)
        self.engine = PermissionEngine(
            autonomy=config.autonomy, user_rules=rules, secret_paths=config.secret_paths
        )
        self.approver = Approver(
            self.engine,
            ask_fn=self.ui.ask_approval,
            max_diff_lines=config.ui.max_diff_lines,
            non_interactive=not args.interactive and bool(args.prompt),
        )

        self.sessions = SessionStore(config.state_dir)
        self.session = self.sessions.new(
            workspace=config.workspace, model=self.provider.model
        )
        self.checkpoints = CheckpointStore(
            config.state_dir, config.workspace, self.session.meta.id
        )

        self.ctx = ToolContext(
            workspace=config.workspace,
            config=config,
            approver=self.approver,
            checkpoints=self.checkpoints,
            ui=self.ui,
        )
        self.ctx.state["provider"] = self.provider

        self.toolbox = Toolbox.build(self.ctx)
        self.mcp = McpManager()

        notes = load_project_notes(config.workspace)
        self.system_prompt = build_system_prompt(
            config.workspace, autonomy=config.autonomy, project_notes=notes
        )
        self.buffer = ConversationBuffer(
            self.system_prompt,
            TokenEstimator(),
            soft_limit=config.context.soft_limit,
            hard_limit=config.context.hard_limit,
            keep_recent_turns=config.context.keep_recent_turns,
            enable_compaction=config.context.enable_compaction,
        )
        self._project_notes = bool(notes)
        self._interrupt = False

    # ------------------------------------------------------------------
    def start_mcp(self) -> None:
        if not self.config.mcp_servers:
            return
        messages = self.mcp.start_all(
            self.config.mcp_servers, cwd=str(self.config.workspace)
        )
        registered = self.mcp.register_with(self.toolbox, self.ctx)
        for message in messages:
            (self.ui.warn if "failed" in message else self.ui.info)(message)
        if registered:
            self.ui.info(f"{registered} MCP tool(s) available.")

    def banner(self) -> None:
        probe = probe_matlab(self.config.matlab.exe)
        matlab_line = (
            f"MATLAB: {'warm engine' if probe.engine_importable else 'batch path'}"
            if probe.executable
            else "MATLAB: not found"
        )
        extra = [
            matlab_line,
            f"tools: {len(self.toolbox.all_specs())}",
            f"session: {self.session.meta.id}",
        ]
        if self._project_notes:
            extra.append("notes: DEPPSEEK.md loaded")
        if self.config.budget.max_cost_usd:
            extra.append(f"budget: ${self.config.budget.max_cost_usd:.2f} per run")
        self.ui.banner(
            model=self.provider.model,
            workspace=self.config.workspace,
            autonomy=self.config.autonomy,
            extra_lines=extra,
        )

    # ------------------------------------------------------------------
    def run_task(self, task: str) -> None:
        loop = AgentLoop(
            provider=self.provider,
            toolbox=self.toolbox,
            buffer=self.buffer,
            config=self.config,
            sink=self.ui,
            on_event=self.ui.on_event,
        )
        try:
            result = loop.run(task, max_steps=self.args.max_steps)
        except KeyboardInterrupt:
            self.ui.warn("Interrupted.")
            return
        except DeppSeekError as exc:
            self.ui.error(str(exc))
            return

        if result.stopped_reason not in ("completed",):
            self.ui.warn(result.stopped_reason)

        self.ui.status_line(
            step=len(result.steps),
            tokens=self.usage.total_tokens,
            cost=self.usage.estimated_cost_usd,
            cache_rate=self.usage.cache_hit_rate,
        )
        self.save()

    def save(self) -> None:
        self.session.messages = self.buffer.messages
        self.session.todos = self.ctx.state.get("todos", [])
        self.session.meta.estimated_cost_usd = self.usage.estimated_cost_usd
        self.session.meta.total_tokens = self.usage.total_tokens
        if not self.session.meta.title:
            first = next(
                (m for m in self.buffer.messages if m.get("role") == "user"), None
            )
            if first:
                self.session.meta.title = str(first.get("content", ""))[:70]
        self.sessions.save(self.session)

    def shutdown(self) -> None:
        self.save()
        self.mcp.stop_all()
        shutdown_matlab(self.ctx)

    # ------------------------------------------------------------------
    def handle_command(self, raw: str) -> bool:
        """Handle a slash command. Returns False to exit the shell."""
        parts = raw[1:].split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        argument = parts[1].strip() if len(parts) > 1 else ""

        if name in ("exit", "quit", "q"):
            return False

        if name == "help":
            width = max(len(c) for c in COMMANDS)
            self.ui.plain(
                "\n".join(f"  /{c:<{width}}  {d}" for c, d in sorted(COMMANDS.items()))
                + "\n\n  Anything not starting with / is sent to the agent."
                + "\n  Type @ to complete a file path. Alt+Enter for a new line."
            )
        elif name == "clear":
            self.buffer.messages.clear()
            self.ctx.read_files.clear()
            self.session = self.sessions.new(
                workspace=self.config.workspace, model=self.provider.model
            )
            self.ui.info(f"Fresh conversation. New session {self.session.meta.id}.")
        elif name == "compact":
            self.ui.info(self.buffer.compact(tools=self.toolbox.schemas()))
        elif name == "context":
            self.ui.plain(self.buffer.stats(self.toolbox.schemas()))
        elif name in ("cost", "usage"):
            self.ui.plain(self.usage.summary(self.provider.model))
        elif name == "tools":
            specs = self.toolbox.all_specs()
            self.ui.plain(
                "\n".join(
                    f"  {n:<24} {s.description.splitlines()[0][:80]}"
                    for n, s in sorted(specs.items())
                )
                + f"\n\n  {len(specs)} tools"
            )
        elif name == "permissions":
            self.ui.plain(self.engine.describe())
        elif name == "autonomy":
            if argument:
                if argument not in AUTONOMY_TIERS:
                    self.ui.error(f"Unknown tier. Choose one of: {', '.join(AUTONOMY_TIERS)}")
                else:
                    self.engine.set_autonomy(argument)
                    self.config = Config(**{**self.config.__dict__, "autonomy": argument})
                    self.ui.info(f"Autonomy tier is now {argument}.")
            else:
                self.ui.plain(f"Autonomy tier: {self.engine.autonomy}")
        elif name == "undo":
            self.ui.plain(self.checkpoints.undo(argument or None))
        elif name == "checkpoints":
            self.ui.plain(self.checkpoints.describe_history())
        elif name == "plan":
            self.ui.plain(render_todos(self.ctx.state.get("todos", [])))
        elif name == "sessions":
            self.ui.plain(self.sessions.describe())
        elif name == "resume":
            self._resume(argument)
        elif name == "branch":
            self.session = self.sessions.branch(self.session, argument)
            self.save()
            self.ui.info(f"Branched to session {self.session.meta.id}.")
        elif name == "matlab":
            self.ui.plain(self.toolbox.execute("matlab_status", {}).content)
        elif name == "mcp":
            self.ui.plain(self.mcp.describe())
        elif name == "doctor":
            self.ui.plain(self.doctor())
        elif name == "config":
            self.ui.plain(self.describe_config())
        elif name == "tree":
            self.ui.plain(self.toolbox.execute("tree", {"max_depth": 3}).content)
        elif name == "model":
            if argument:
                self.provider.model = argument
                self.ui.info(f"Model is now {argument}.")
            else:
                self.ui.plain(
                    f"{self.provider.model}\nKnown models: {', '.join(sorted(MODELS))}"
                )
        elif name in ("workspace", "pwd"):
            self.ui.plain(str(self.config.workspace))
        else:
            self.ui.error(f"Unknown command /{name}. Try /help.")
        return True

    def _resume(self, session_id: str) -> None:
        session = self.sessions.load(session_id) if session_id else self.sessions.latest()
        if session is None:
            self.ui.error(
                f"No session {session_id!r}." if session_id else "No saved sessions."
            )
            return
        self.session = session
        self.buffer.messages = list(session.messages)
        self.ctx.state["todos"] = list(session.todos)
        problems = self.buffer.validate()
        if problems:
            self.ui.warn(
                f"Resumed session has {len(problems)} malformed tool pairing(s); "
                f"the affected turns were dropped."
            )
            self.buffer.messages = [
                m for m in self.buffer.messages if m.get("role") != "tool"
            ]
        self.ui.info(
            f"Resumed {session.meta.id}: {len(session.messages)} message(s), "
            f"${session.meta.estimated_cost_usd:.4f} spent previously."
        )

    # ------------------------------------------------------------------
    def doctor(self) -> str:
        """Check the environment and report anything that will bite."""
        lines: list[str] = []
        ok, warn, bad = "ok  ", "warn", "FAIL"

        lines.append(f"{ok} Python {sys.version.split()[0]} at {sys.executable}")
        lines.append(f"{ok} deppseek {__version__}")

        key = os.getenv(self.config.api_key_env)
        lines.append(
            f"{ok} {self.config.api_key_env} is set ({key[:6]}...)"
            if key
            else f"{bad} {self.config.api_key_env} is not set"
        )

        if self.provider.model in MODELS:
            lines.append(f"{ok} model {self.provider.model} is current")
        else:
            lines.append(
                f"{warn} model {self.provider.model} is not in the pricing table; "
                f"cost estimates may be wrong"
            )

        probe = probe_matlab(self.config.matlab.exe)
        if probe.executable and probe.engine_importable:
            lines.append(f"{ok} MATLAB with warm engine: {probe.executable}")
        elif probe.executable:
            lines.append(f"{warn} MATLAB found at {probe.executable}, batch path only")
            lines.append(f"       {probe.describe().splitlines()[2].strip()}")
        else:
            lines.append(f"{warn} MATLAB not found; run_matlab will fail")

        import shutil as _shutil

        for name, why in (
            ("git", "git tools"),
            ("rg", "fast search (a slower Python walk is used without it)"),
            ("pwsh", "PowerShell 7 (falls back to powershell.exe)"),
        ):
            found = _shutil.which(name)
            lines.append(
                f"{ok} {name} at {found}" if found else f"{warn} {name} not found: {why}"
            )

        for module, why in (
            ("rich", "styled output"),
            ("prompt_toolkit", "history and completion"),
            ("textual", "the --tui dashboard"),
            ("pint", "unit checking"),
            ("pathspec", ".gitignore-aware search"),
        ):
            try:
                __import__(module)
                lines.append(f"{ok} {module}")
            except ImportError:
                lines.append(f"{warn} {module} not installed: {why} unavailable")

        if self.config.vision.enabled:
            lines.append(f"{ok} vision model {self.config.vision.model} configured")
        else:
            lines.append(f"{warn} no vision model: figures cannot be examined")

        try:
            probe_file = self.config.state_dir / ".write-probe"
            probe_file.parent.mkdir(parents=True, exist_ok=True)
            probe_file.write_text("x", encoding="utf-8")
            probe_file.unlink()
            lines.append(f"{ok} workspace state directory is writable")
        except OSError as exc:
            lines.append(f"{bad} cannot write to {self.config.state_dir}: {exc}")

        return "\n".join(lines)

    def describe_config(self) -> str:
        user_path = user_config_path()
        project_path = project_config_path(self.config.workspace)
        lines = [
            f"user config:     {user_path} {'(found)' if user_path.is_file() else '(none)'}",
            f"project config:  {project_path} {'(found)' if project_path.is_file() else '(none)'}",
            "",
            f"model            {self.provider.model}",
            f"thinking         {self.config.thinking} (effort {self.config.reasoning_effort})",
            f"autonomy         {self.config.autonomy}",
            f"max steps        {self.config.budget.max_steps}",
            "cost ceiling     "
            + (f"${self.config.budget.max_cost_usd:.2f}" if self.config.budget.max_cost_usd else "none"),
            f"context soft/hard {self.config.context.soft_limit:,} / {self.config.context.hard_limit:,}",
            f"parallel tools   {self.config.max_parallel_tools}",
            f"state directory  {self.config.state_dir}",
        ]
        return "\n".join(lines)

    def interrupt(self) -> None:
        self._interrupt = True

    def handle_input(self, text: str) -> None:
        """Entry point shared by the shell and the TUI."""
        if text.startswith("/"):
            self.handle_command(text)
        else:
            self.run_task(text)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def interactive_shell(controller: Controller) -> int:
    controller.banner()
    session = build_session(
        controller.config.workspace,
        COMMANDS,
        controller.config.state_dir / "input-history.txt",
    )

    while True:
        try:
            raw = read_input(session, "deppseek> ").strip()
        except (EOFError, KeyboardInterrupt):
            controller.ui.info("\nBye.")
            return 0
        if not raw:
            continue
        if raw.startswith("/"):
            if not controller.handle_command(raw):
                controller.ui.info("Bye.")
                return 0
            continue
        try:
            controller.run_task(raw)
        except KeyboardInterrupt:
            controller.ui.warn("Interrupted; the conversation is saved.")


def run_tui(controller: Controller) -> int:
    from .ui import tui as tui_module

    if not tui_module.available():
        controller.ui.error(
            "The dashboard needs textual, which is not installed.\n"
            "  python -m pip install textual\n"
            "Running the inline interface instead."
        )
        return interactive_shell(controller)

    app = tui_module.DeppSeekTUI(controller)
    sink = tui_module.TuiSink(app)
    controller.ui_sink = sink  # type: ignore[attr-defined]

    original_run_task = controller.run_task

    def run_task_in_tui(task: str) -> None:
        loop = AgentLoop(
            provider=controller.provider,
            toolbox=controller.toolbox,
            buffer=controller.buffer,
            config=controller.config,
            sink=sink,
            on_event=sink.on_event,
        )
        result = loop.run(task, max_steps=controller.args.max_steps)
        sink.on_event(
            "meters",
            {
                "step": len(result.steps),
                "tokens": controller.usage.total_tokens,
                "cost": controller.usage.estimated_cost_usd,
                "cache_rate": controller.usage.cache_hit_rate,
                "context_pct": min(
                    1.0,
                    controller.buffer.estimate(controller.toolbox.schemas())
                    / controller.config.context.hard_limit,
                ),
            },
        )
        sink.on_event("plan", {"text": render_todos(controller.ctx.state.get("todos", []))})
        controller.save()

    controller.run_task = run_task_in_tui  # type: ignore[method-assign]
    try:
        app.run()
    finally:
        controller.run_task = original_run_task  # type: ignore[method-assign]
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deppseek",
        description="An approval-gated DeepSeek engineering agent for Windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  deppseek                                  start the interactive shell\n"
            '  deppseek "why does case3 diverge?"        run one task and exit\n'
            "  deppseek --tui                            full-screen dashboard\n"
            "  deppseek --resume                         continue the last session\n"
            "  deppseek --autonomy ask                   confirm every side effect\n"
            "  deppseek --doctor                         check the environment\n"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="task to run; omit for the shell")
    parser.add_argument("--workspace", default=".", help="project root (default: cwd)")
    parser.add_argument("--model", default=None, help=f"one of: {', '.join(sorted(MODELS))}")
    parser.add_argument(
        "--autonomy", choices=AUTONOMY_TIERS, default=None, help="permission tier"
    )
    parser.add_argument("--reasoning", default=None, help="low, high, max, or 1-100")
    parser.add_argument(
        "--no-thinking", action="store_true", help="disable extended reasoning"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="tool/step ceiling for one task"
    )
    parser.add_argument(
        "--max-cost", type=float, default=None, help="cost ceiling in USD for one run"
    )
    parser.add_argument("--tui", action="store_true", help="full-screen dashboard")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="shell, even with a prompt"
    )
    parser.add_argument(
        "--resume", nargs="?", const="", default=None, metavar="ID",
        help="resume a session (latest if no id given)",
    )
    parser.add_argument("--doctor", action="store_true", help="check the environment and exit")
    parser.add_argument("--no-mcp", action="store_true", help="skip starting MCP servers")
    parser.add_argument("--version", action="version", version=f"deppseek {__version__}")
    return parser


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI flags onto the config tree, omitting anything not supplied."""
    overrides: dict[str, Any] = {
        "model": args.model,
        "autonomy": args.autonomy,
        "reasoning_effort": args.reasoning,
    }
    if args.no_thinking:
        overrides["thinking"] = False
    if args.tui:
        overrides["ui"] = {"mode": "tui"}
    budget: dict[str, Any] = {}
    if args.max_steps is not None:
        budget["max_steps"] = args.max_steps
    if args.max_cost is not None:
        budget["max_cost_usd"] = args.max_cost
    if budget:
        overrides["budget"] = budget
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"Workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    try:
        config = load_config(workspace, cli_overrides(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # A step ceiling must exist for both paths; the shell used to ignore the flag.
    if args.max_steps is None:
        args.max_steps = config.budget.max_steps
    args.interactive = args.interactive or not args.prompt

    try:
        controller = Controller(config, args)
    except ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except DeppSeekError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    if args.doctor:
        print(controller.doctor())
        return 0

    try:
        if not args.no_mcp:
            controller.start_mcp()
        if args.resume is not None:
            controller._resume(args.resume)

        if config.ui.mode == "tui" or args.tui:
            return run_tui(controller)
        if args.interactive:
            return interactive_shell(controller)

        controller.run_task(" ".join(args.prompt))
        return 0
    finally:
        controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
