"""Interactive input.

Uses prompt_toolkit when available, for the things a bare `input()` cannot do:
persistent history across sessions, multi-line editing, and completion of slash
commands and workspace paths. Falls back to `input()` so the agent still runs on
a machine where prompt_toolkit failed to install.

Enter submits. Alt+Enter (or a trailing backslash) inserts a newline, which
matters when pasting a MATLAB snippet or a stack trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    PTK = True
except ImportError:  # pragma: no cover
    PTK = False
    Completer = object  # type: ignore[assignment,misc]


class WorkspaceCompleter(Completer):  # type: ignore[misc]
    """Completes slash commands, and file paths after an @ sign."""

    def __init__(self, workspace: Path, commands: dict[str, str]) -> None:
        self.workspace = workspace
        self.commands = commands
        self._path_cache: list[str] = []
        self._cache_stamp = 0.0

    def _paths(self) -> list[str]:
        import time

        # Rebuilding the file list on every keystroke is unusable on a large
        # repository, so it is cached briefly.
        if time.monotonic() - self._cache_stamp < 5.0 and self._path_cache:
            return self._path_cache
        from ..tools.fs import is_ignored

        found: list[str] = []
        for path in self.workspace.rglob("*"):
            if len(found) >= 4000:
                break
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.workspace).as_posix()
            except ValueError:
                continue
            if is_ignored(relative):
                continue
            found.append(relative)
        self._path_cache = found
        self._cache_stamp = time.monotonic()
        return found

    def get_completions(self, document, complete_event):  # noqa: ANN001
        text = document.text_before_cursor

        if text.startswith("/") and " " not in text:
            prefix = text[1:]
            for name, description in sorted(self.commands.items()):
                if name.startswith(prefix):
                    yield Completion(
                        name, start_position=-len(prefix), display_meta=description
                    )
            return

        at_index = text.rfind("@")
        if at_index >= 0 and " " not in text[at_index:]:
            prefix = text[at_index + 1 :]
            matches = 0
            for relative in self._paths():
                if prefix.lower() in relative.lower():
                    yield Completion(relative, start_position=-len(prefix))
                    matches += 1
                    if matches >= 40:
                        return


def build_session(
    workspace: Path, commands: dict[str, str], history_path: Path
) -> Any | None:
    """Create a prompt_toolkit session, or None when it is unavailable."""
    if not PTK:
        return None
    history_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):  # noqa: ANN001
        """Alt+Enter inserts a literal newline rather than submitting."""
        event.current_buffer.insert_text("\n")

    style = Style.from_dict(
        {
            "prompt": "bold ansicyan",
            "continuation": "ansibrightblack",
            "completion-menu.completion": "bg:#1c2733 #cdd6e0",
            "completion-menu.completion.current": "bg:#2a4a63 #ffffff",
            "completion-menu.meta.completion": "bg:#1c2733 #7d8a99",
        }
    )

    return PromptSession(
        history=FileHistory(str(history_path)),
        completer=WorkspaceCompleter(workspace, commands),
        key_bindings=bindings,
        style=style,
        multiline=False,
        complete_while_typing=True,
        enable_history_search=True,
    )


def read_input(session: Any | None, prompt: str = "> ") -> str:
    """Read one line, supporting backslash continuation for multi-line paste."""
    if session is None:
        return input(prompt)

    from prompt_toolkit.formatted_text import HTML

    text = session.prompt(HTML(f"<prompt>{prompt}</prompt>"))
    # A trailing backslash continues onto the next line, the shell convention,
    # for terminals where Alt+Enter is intercepted by the emulator.
    while text.rstrip().endswith("\\"):
        text = text.rstrip()[:-1] + "\n" + session.prompt(HTML("<continuation>... </continuation>"))
    return text
