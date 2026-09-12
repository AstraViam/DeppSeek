"""Provider-agnostic message and response shapes.

The agent loop talks only to these types, so swapping or adding a backend means
writing one adapter rather than touching the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, parsed and validated by the registry

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Completion:
    """One assistant turn, normalised across providers."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw_usage: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> dict[str, Any]:
        """Build the assistant message to append to history.

        DeepSeek thinking mode requires `reasoning_content` to be carried back on
        assistant messages in a multi-turn conversation, unlike the older R1
        behaviour where it had to be stripped. Dropping it degrades tool-calling
        quality across steps, so it is preserved here.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = [tc.to_wire() for tc in self.tool_calls]
        return message


class StreamSink(Protocol):
    """Receives incremental output while a completion is in flight."""

    def on_reasoning(self, delta: str) -> None: ...
    def on_content(self, delta: str) -> None: ...
    def on_tool_call_start(self, name: str) -> None: ...
    def on_done(self) -> None: ...


class NullSink:
    """Discards streamed output. Used for subagents and non-interactive runs."""

    def on_reasoning(self, delta: str) -> None:
        return None

    def on_content(self, delta: str) -> None:
        return None

    def on_tool_call_start(self, name: str) -> None:
        return None

    def on_done(self) -> None:
        return None


class Provider(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        sink: StreamSink | None = None,
        stream: bool = True,
    ) -> Completion: ...
