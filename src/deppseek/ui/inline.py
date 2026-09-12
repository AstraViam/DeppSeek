"""Inline terminal interface.

A scrolling transcript, the way a CLI normally behaves, rather than a full-screen
application. Chosen as the default because it keeps native scrollback, copy and
paste, and output redirection working, all of which a full-screen app breaks.

Streaming is rendered into a live region so markdown formats as it arrives; when
the turn ends the region is finalised and becomes ordinary scrollback. Reasoning
is streamed into a separate collapsed region that is replaced by the answer, so a
long chain of thought does not bury the result.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from ..permissions.prompt import Approval, ask_plain, format_request
from .theme import build_theme, glyphs_for

try:
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH = True
except ImportError:  # pragma: no cover - exercised only on a broken install
    RICH = False


class InlineUI:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.glyphs = glyphs_for(config.ui.unicode)
        self.console = (
            Console(theme=build_theme(), highlight=False, soft_wrap=False) if RICH else None
        )
        self._live: Any = None
        self._buffer: list[str] = []
        self._reasoning: list[str] = []
        self._mode = "idle"  # idle | reasoning | content
        self._tools_this_step: list[str] = []

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------
    def _print(self, *args: Any, **kwargs: Any) -> None:
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*[a for a in args if isinstance(a, str)])

    def rule(self, label: str = "") -> None:
        if self.console:
            self.console.rule(f"[ds.rule]{label}[/]" if label else "", style="ds.rule")
        else:
            width = shutil.get_terminal_size((80, 24)).columns
            print(f" {label} ".center(width, "-") if label else "-" * width)

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    def banner(self, *, model: str, workspace, autonomy: str, extra_lines: list[str]) -> None:
        g = self.glyphs
        if not self.console:
            print(f"DeppSeek  model={model}  workspace={workspace}  autonomy={autonomy}")
            for line in extra_lines:
                print(f"  {line}")
            return

        table = Table.grid(padding=(0, 2))
        table.add_column(style="ds.muted", justify="right")
        table.add_column(style="ds.assistant")
        table.add_row("model", f"[ds.accent]{model}[/]")
        table.add_row("workspace", str(workspace))
        table.add_row("autonomy", _autonomy_markup(autonomy))
        for line in extra_lines:
            if ":" in line:
                key, _, value = line.partition(":")
                table.add_row(key.strip(), value.strip())
            else:
                table.add_row("", line)

        self.console.print(
            Panel(
                table,
                title=f"[ds.accent]{g.think} DeppSeek[/]",
                subtitle="[ds.muted]/help for commands[/]",
                border_style="ds.accent_dim",
                padding=(1, 2),
            )
        )

    # ------------------------------------------------------------------
    # Streaming sink
    # ------------------------------------------------------------------
    def on_reasoning(self, delta: str) -> None:
        if not self.config.ui.show_reasoning:
            return
        self._reasoning.append(delta)
        if self._mode != "reasoning":
            self._mode = "reasoning"
            self._start_live()
        self._refresh()

    def on_content(self, delta: str) -> None:
        self._buffer.append(delta)
        if self._mode != "content":
            self._mode = "content"
            if self._live is None:
                self._start_live()
        self._refresh()

    def on_tool_call_start(self, name: str) -> None:
        if name not in self._tools_this_step:
            self._tools_this_step.append(name)
        self._refresh()

    def on_done(self) -> None:
        self._stop_live()
        self._reasoning.clear()
        self._buffer.clear()
        self._tools_this_step.clear()
        self._mode = "idle"

    def _start_live(self) -> None:
        if not self.console or self._live is not None:
            return
        self._live = Live(
            "",
            console=self.console,
            refresh_per_second=8,
            # Long answers must keep scrolling rather than being clipped to the
            # window height, which is what a full-screen Live region would do.
            vertical_overflow="visible",
            transient=False,
        )
        self._live.start()

    def _refresh(self) -> None:
        if self._live is None:
            # No rich: stream raw text so the user still sees progress.
            if self._mode == "content" and self._buffer:
                sys.stdout.write(self._buffer[-1])
                sys.stdout.flush()
            return
        try:
            self._live.update(self._render())
        except Exception:  # noqa: BLE001 - a render failure must not kill the run
            pass

    def _render(self):
        g = self.glyphs
        parts = []

        if self._reasoning and self._mode == "reasoning":
            text = "".join(self._reasoning)
            lines = text.splitlines() or [text]
            visible = lines[-self.config.ui.reasoning_lines :]
            parts.append(
                Panel(
                    Text("\n".join(visible), style="ds.reasoning"),
                    title=f"[ds.muted]{g.think} thinking[/]",
                    border_style="ds.rule",
                    padding=(0, 1),
                )
            )

        if self._buffer:
            body = "".join(self._buffer)
            try:
                parts.append(Markdown(body, code_theme=self.config.ui.syntax_theme))
            except Exception:  # noqa: BLE001 - malformed markdown mid-stream
                parts.append(Text(body, style="ds.assistant"))

        if self._tools_this_step:
            parts.append(
                Text(
                    f"  {g.tool} " + "  ".join(self._tools_this_step),
                    style="ds.tool",
                )
            )

        return Group(*parts) if parts else Text("")

    def _stop_live(self) -> None:
        if self._live is None:
            if self._mode == "content":
                sys.stdout.write("\n")
                sys.stdout.flush()
            return
        try:
            self._live.update(self._render_final())
            self._live.stop()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._live = None

    def _render_final(self):
        """Final frame: drop the thinking panel, keep the answer."""
        if not self._buffer:
            return Text("")
        body = "".join(self._buffer)
        try:
            return Markdown(body, code_theme=self.config.ui.syntax_theme)
        except Exception:  # noqa: BLE001
            return Text(body, style="ds.assistant")

    # ------------------------------------------------------------------
    # Agent events
    # ------------------------------------------------------------------
    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        g = self.glyphs
        if event == "step_start":
            if self.console:
                self.console.print(
                    f"[ds.muted]step {payload['index']}/{payload['max_steps']}[/]",
                    justify="right",
                )
        elif event == "tool_start":
            self._print(f"[ds.tool]{g.tool} {payload['name']}[/] [ds.muted]{_brief(payload.get('arguments'))}[/]")
        elif event == "tool_end":
            mark = f"[ds.tool_error]{g.cross}[/]" if payload.get("is_error") else f"[ds.tool_ok]{g.check}[/]"
            duration = payload.get("duration_s") or 0.0
            timing = f" [ds.muted]{duration:.1f}s[/]" if duration > 0.5 else ""
            self._print(f"  {mark} [ds.muted]{payload.get('display', '')}[/]{timing}")
        elif event == "parallel_tools":
            self._print(
                f"[ds.muted]  running {payload['count']} tools in parallel "
                f"({payload['workers']} workers)[/]"
            )
        elif event == "compacted":
            self._print(f"[ds.warn]{g.bullet} {payload['message']}[/]")
        elif event == "budget":
            self._print(f"[ds.warn]{g.bullet} {payload['message']}[/]")
        elif event == "error":
            self._print(f"[ds.error]{g.cross} {payload['message']}[/]")
        elif event == "interrupted":
            self._print(f"[ds.warn]{g.bullet} {payload['message']}[/]")

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._print(f"[ds.muted]{message}[/]")

    def warn(self, message: str) -> None:
        self._print(f"[ds.warn]{self.glyphs.bullet} {message}[/]")

    def error(self, message: str) -> None:
        self._print(f"[ds.error]{self.glyphs.cross} {message}[/]")

    def markdown(self, text: str) -> None:
        if self.console:
            try:
                self.console.print(Markdown(text, code_theme=self.config.ui.syntax_theme))
                return
            except Exception:  # noqa: BLE001
                pass
        print(text)

    def plain(self, text: str) -> None:
        if self.console:
            self.console.print(Text(text))
        else:
            print(text)

    def diff(self, text: str) -> None:
        """Render a unified diff with per-line colouring."""
        if not self.console:
            print(text)
            return
        rendered = Text()
        for line in text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                rendered.append(line + "\n", style="ds.muted")
            elif line.startswith("@@"):
                rendered.append(line + "\n", style="ds.accent_dim")
            elif line.startswith("+"):
                rendered.append(line + "\n", style="ds.added")
            elif line.startswith("-"):
                rendered.append(line + "\n", style="ds.removed")
            else:
                rendered.append(line + "\n", style="ds.muted")
        self.console.print(rendered)

    def code(self, text: str, language: str = "python") -> None:
        if self.console:
            self.console.print(
                Syntax(text, language, theme=self.config.ui.syntax_theme, word_wrap=True)
            )
        else:
            print(text)

    def status_line(self, *, step: int, tokens: int, cost: float, cache_rate: float) -> None:
        g = self.glyphs
        if not self.console:
            print(f"[{step} steps | {tokens:,} tokens | ${cost:.4f}]")
            return
        self.console.print(
            f"[ds.muted]{g.corner} {step} step(s)  {tokens:,} tokens  "
            f"cache {cache_rate * 100:.0f}%  [/][ds.cost]${cost:.4f}[/]",
            justify="right",
        )

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------
    def ask_approval(self, request, verdict) -> Approval:
        """Prompt for approval, showing the diff or command that is pending."""
        if not self.console:
            return ask_plain(request, verdict, max_diff_lines=self.config.ui.max_diff_lines)

        g = self.glyphs
        body = format_request(request, verdict, max_diff_lines=self.config.ui.max_diff_lines)

        renderables = []
        if request.diff:
            head, _, diff_part = body.partition(request.diff)
            renderables.append(Text(head.rstrip(), style="ds.assistant"))
            diff_text = Text()
            for line in request.diff.splitlines():
                style = (
                    "ds.added" if line.startswith("+") and not line.startswith("+++")
                    else "ds.removed" if line.startswith("-") and not line.startswith("---")
                    else "ds.accent_dim" if line.startswith("@@")
                    else "ds.muted"
                )
                diff_text.append(line + "\n", style=style)
            renderables.append(diff_text)
            renderables.append(Text(diff_part.strip(), style="ds.muted"))
        else:
            renderables.append(Text(body, style="ds.assistant"))

        self.console.print(
            Panel(
                Group(*renderables),
                title=f"[ds.warn]{g.bullet} approval required[/]",
                border_style="ds.warn",
                padding=(1, 2),
            )
        )

        rememberable = verdict.rule is None or verdict.rule.remberable
        options = "[bold]y[/]es  [bold]n[/]o" + (
            "  [bold]a[/]lways  [bold]d[/]eny for session" if rememberable else ""
        )
        while True:
            try:
                answer = self.console.input(f"  {options} > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return Approval(approved=False, note="interrupted")
            if answer in {"y", "yes"}:
                return Approval(approved=True)
            if answer in {"n", "no", ""}:
                return Approval(approved=False, note="denied by user")
            if rememberable and answer in {"a", "always"}:
                return Approval(approved=True, remember=True)
            if rememberable and answer in {"d", "deny", "never"}:
                return Approval(
                    approved=False, remember=True, deny_for_session=True,
                    note="denied for session",
                )
            self.console.print("[ds.muted]  answer y, n, a, or d[/]")


def _brief(arguments: Any, limit: int = 88) -> str:
    """One-line preview of tool arguments."""
    if not arguments:
        return ""
    text = arguments if isinstance(arguments, str) else str(arguments)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _autonomy_markup(autonomy: str) -> str:
    colour = {
        "readonly": "ds.tool_ok",
        "ask": "ds.tool_ok",
        "standard": "ds.warn",
        "autonomous": "ds.error",
    }.get(autonomy, "ds.muted")
    note = {
        "readonly": "no side effects",
        "ask": "confirm every side effect",
        "standard": "writes run, execution confirmed",
        "autonomous": "writes and execution run unattended",
    }.get(autonomy, "")
    return f"[{colour}]{autonomy}[/] [ds.muted]({note})[/]"
