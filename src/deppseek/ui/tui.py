"""Full-screen dashboard, behind --tui.

Not the default. A full-screen application takes over the alternate screen
buffer, which costs native scrollback, breaks terminal copy-paste selection, and
makes output redirection meaningless. Those are real losses for a CLI tool.

What it buys, and why it is worth offering for long unattended runs: everything
is visible at once. The plan, the running token and cost meters, the tool log,
and the transcript do not scroll past each other. On a forty-step refactor that
is the difference between watching the work and hunting through scrollback.
"""

from __future__ import annotations

from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, RichLog, Static

    TEXTUAL = True
except ImportError:  # pragma: no cover - optional extra
    TEXTUAL = False
    App = object  # type: ignore[assignment,misc]


CSS = """
Screen { layout: vertical; }
#body { height: 1fr; }
#transcript {
    width: 3fr;
    border: round $primary-darken-2;
    padding: 0 1;
}
#side { width: 1fr; layout: vertical; }
#meters {
    height: auto;
    border: round $primary-darken-2;
    padding: 0 1;
}
#plan {
    height: 1fr;
    border: round $primary-darken-2;
    padding: 0 1;
}
#toollog {
    height: 1fr;
    border: round $primary-darken-2;
    padding: 0 1;
}
#prompt { dock: bottom; height: 3; }
.title { text-style: bold; color: $accent; }
"""


class Meters(Static):
    """Live token, cost, and step counters."""

    step = reactive(0)
    tokens = reactive(0)
    cost = reactive(0.0)
    cache_rate = reactive(0.0)
    context_pct = reactive(0.0)

    def render(self) -> str:
        bar_width = 18
        filled = int(self.context_pct * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        return (
            f"[b]step[/b]     {self.step}\n"
            f"[b]tokens[/b]   {self.tokens:,}\n"
            f"[b]cache[/b]    {self.cache_rate * 100:.0f}%\n"
            f"[b]cost[/b]     ${self.cost:.4f}\n"
            f"[b]context[/b]  {bar} {self.context_pct * 100:.0f}%"
        )


class DeppSeekTUI(App):  # type: ignore[misc]
    """Dashboard driven by the same agent events as the inline UI."""

    CSS = CSS
    BINDINGS = [
        ("ctrl+c", "interrupt", "Interrupt"),
        ("ctrl+d", "quit", "Quit"),
        ("ctrl+l", "clear_transcript", "Clear"),
    ]

    def __init__(self, controller: Any) -> None:
        super().__init__()
        self.controller = controller
        self.title = "DeppSeek"
        self.sub_title = f"{controller.config.model}  {controller.config.workspace}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield RichLog(id="transcript", wrap=True, markup=True, highlight=False)
            with Vertical(id="side"):
                yield Meters(id="meters")
                yield RichLog(id="plan", wrap=True, markup=True)
                yield RichLog(id="toollog", wrap=True, markup=True)
        yield Input(placeholder="Ask, or /help", id="prompt")
        yield Footer()

    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.query_one("#transcript", RichLog).write(
            "[b cyan]DeppSeek[/b cyan] dashboard. Type a task and press Enter.\n"
            "Ctrl+C interrupts a running task; Ctrl+D quits."
        )
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.query_one("#transcript", RichLog).write(f"\n[b white]> {text}[/b white]")
        # The agent runs on a worker thread so the UI keeps repainting; Textual
        # marshals the event callbacks below back onto its own thread.
        self.run_worker(
            lambda: self.controller.handle_input(text), thread=True, exclusive=True
        )

    def action_interrupt(self) -> None:
        self.controller.interrupt()
        self.query_one("#transcript", RichLog).write("[yellow]interrupt requested[/yellow]")

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", RichLog).clear()

    # ------------------------------------------------------------------
    # Called from the agent, via call_from_thread.
    # ------------------------------------------------------------------
    def write_transcript(self, markup: str) -> None:
        self.query_one("#transcript", RichLog).write(markup)

    def write_tool(self, markup: str) -> None:
        self.query_one("#toollog", RichLog).write(markup)

    def set_plan(self, text: str) -> None:
        log = self.query_one("#plan", RichLog)
        log.clear()
        log.write(text)

    def update_meters(self, **values: Any) -> None:
        meters = self.query_one("#meters", Meters)
        for key, value in values.items():
            if hasattr(meters, key):
                setattr(meters, key, value)


class TuiSink:
    """Adapts the streaming and event interfaces onto the Textual app.

    Every call crosses a thread boundary, so each one goes through
    `call_from_thread`; touching a widget directly from the worker thread is the
    usual way a Textual app corrupts its own display.
    """

    def __init__(self, app: DeppSeekTUI) -> None:
        self.app = app
        self._content: list[str] = []
        self._reasoning_shown = False

    def _safe(self, func, *args: Any, **kwargs: Any) -> None:
        try:
            self.app.call_from_thread(func, *args, **kwargs)
        except Exception:  # noqa: BLE001 - a shutting-down app must not crash the run
            pass

    def on_reasoning(self, delta: str) -> None:
        if not self._reasoning_shown:
            self._reasoning_shown = True
            self._safe(self.app.write_transcript, "[dim]thinking...[/dim]")

    def on_content(self, delta: str) -> None:
        self._content.append(delta)
        # Flush on line boundaries: a per-token repaint of a full-screen app is
        # visibly slower than the model produces tokens.
        if "\n" in delta:
            text = "".join(self._content)
            self._content.clear()
            self._safe(self.app.write_transcript, text.rstrip("\n"))

    def on_tool_call_start(self, name: str) -> None:
        self._safe(self.app.write_tool, f"[blue]> {name}[/blue]")

    def on_done(self) -> None:
        if self._content:
            self._safe(self.app.write_transcript, "".join(self._content))
            self._content.clear()
        self._reasoning_shown = False

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "tool_start":
            self._safe(self.app.write_tool, f"[blue]> {payload['name']}[/blue]")
        elif event == "tool_end":
            colour = "red" if payload.get("is_error") else "green"
            self._safe(
                self.app.write_tool,
                f"  [{colour}]{payload.get('display', '')}[/{colour}]",
            )
        elif event in ("compacted", "budget", "interrupted"):
            self._safe(self.app.write_transcript, f"[yellow]{payload.get('message', '')}[/yellow]")
        elif event == "error":
            self._safe(self.app.write_transcript, f"[red]{payload.get('message', '')}[/red]")
        elif event == "meters":
            self._safe(self.app.update_meters, **payload)
        elif event == "plan":
            self._safe(self.app.set_plan, payload.get("text", ""))


def available() -> bool:
    return TEXTUAL
