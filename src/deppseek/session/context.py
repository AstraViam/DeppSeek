"""Conversation buffer with token budgeting and compaction.

v1 appended every message and every full tool result to one list, loaded the
whole thing from disk at startup, and never trimmed it. Two consequences:

* A single 300 kB file read stayed in context for the rest of the session, and
  was re-sent and re-billed on every subsequent step.
* A long session eventually exceeded the context window and started failing, with
  no warning and nothing the user could do except delete the history file.

The buffer here holds a strict ordering invariant that the OpenAI-compatible
wire format requires: every `tool` message must follow an `assistant` message
whose `tool_calls` includes its `tool_call_id`. Compaction that breaks that pairing
produces a 400 from the API, so the compactor only ever cuts at a boundary
between complete assistant/tool groups.

Compaction summarises the oldest groups into a single system note and drops
them. It invalidates the provider's prefix cache for one request, which is why
the soft limit is set well below the hard limit: compaction should be rare
enough that the cache saving dwarfs its cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tokens import TokenEstimator


@dataclass
class TurnGroup:
    """One assistant turn plus the tool results it triggered.

    Grouping matters because these messages cannot be separated: dropping an
    assistant message while keeping its tool replies produces an API error.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tokens_hint(self) -> int:
        return sum(len(str(m.get("content") or "")) for m in self.messages)


class ConversationBuffer:
    def __init__(
        self,
        system_prompt: str,
        estimator: TokenEstimator | None = None,
        *,
        soft_limit: int = 420_000,
        hard_limit: int = 960_000,
        keep_recent_turns: int = 8,
        enable_compaction: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        # Built once and reused. Constructing a fresh dict per render gave the
        # system message a new identity every step, so it missed the estimator's
        # memo on every call -- and the system prompt is the largest single
        # message in a short conversation.
        self._system_message: dict[str, Any] = {"role": "system", "content": system_prompt}
        self.estimator = estimator or TokenEstimator()
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.keep_recent_turns = keep_recent_turns
        self.enable_compaction = enable_compaction
        # Everything after the system message.
        self.messages: list[dict[str, Any]] = []
        self.compactions = 0
        self.dropped_messages = 0
        self._last_estimate = 0

    # ------------------------------------------------------------------
    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def append_user(self, content: str) -> None:
        self.append({"role": "user", "content": content})

    def append_tool_result(self, tool_call_id: str, content: str) -> None:
        self.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def render(self) -> list[dict[str, Any]]:
        """The full message list to send, system prompt first.

        The system message object is reused rather than rebuilt, both so the
        token memo can hit and so the serialised prefix is identical between
        steps, which is what lets the provider's context cache hit.
        """
        return [self._system_message, *self.messages]

    def estimate(self, tools: list[dict[str, Any]] | None = None) -> int:
        total = self.estimator.estimate_messages(self.render())
        if tools:
            total += self.estimator.estimate_tools(tools)
        self._last_estimate = total
        return total

    def calibrate(self, actual_prompt_tokens: int) -> None:
        self.estimator.calibrate(self._last_estimate, actual_prompt_tokens)

    # ------------------------------------------------------------------
    def _group_messages(self) -> list[TurnGroup]:
        """Split the history into units that must not be separated."""
        groups: list[TurnGroup] = []
        current = TurnGroup()

        for message in self.messages:
            role = message.get("role")
            if role == "tool":
                # A tool result belongs to the group that opened it.
                if current.messages:
                    current.messages.append(message)
                else:
                    # Orphaned tool message: keep it attached to whatever
                    # precedes it rather than starting a group with it, since a
                    # leading tool message is invalid on the wire.
                    if groups:
                        groups[-1].messages.append(message)
                    else:
                        current.messages.append(message)
                continue

            if role == "assistant" and message.get("tool_calls"):
                if current.messages:
                    groups.append(current)
                current = TurnGroup([message])
                continue

            if current.messages:
                groups.append(current)
            current = TurnGroup([message])

        if current.messages:
            groups.append(current)
        return groups

    def needs_compaction(self, tools: list[dict[str, Any]] | None = None) -> bool:
        return self.enable_compaction and self.estimate(tools) > self.soft_limit

    def over_hard_limit(self, tools: list[dict[str, Any]] | None = None) -> bool:
        return self.estimate(tools) > self.hard_limit

    def compact(
        self,
        summarise: Callable[[list[dict[str, Any]]], str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Summarise and drop the oldest turn groups.

        `summarise` is normally a cheap model call. When it is None or fails, a
        deterministic structural summary is used instead, so compaction never
        depends on the network being up.
        """
        groups = self._group_messages()
        if len(groups) <= self.keep_recent_turns + 1:
            return "Nothing to compact: too few turns."

        keep_from = len(groups) - self.keep_recent_turns
        old_groups = groups[:keep_from]
        recent_groups = groups[keep_from:]

        old_messages = [m for group in old_groups for m in group.messages]
        before_tokens = self.estimate(tools)

        summary_text = ""
        if summarise is not None:
            try:
                summary_text = summarise(old_messages)
            except Exception:  # noqa: BLE001 - compaction must never fail the run
                summary_text = ""
        if not summary_text:
            summary_text = structural_summary(old_messages)

        self.messages = [
            {
                "role": "user",
                "content": (
                    "[Earlier conversation was compacted to stay within the context "
                    "budget. Summary of what happened before this point:]\n\n"
                    f"{summary_text}\n\n"
                    "[End of summary. Treat the above as established history. "
                    "Re-read any file you need to quote exactly.]"
                ),
            },
            *[m for group in recent_groups for m in group.messages],
        ]
        self.compactions += 1
        self.dropped_messages += len(old_messages)
        after_tokens = self.estimate(tools)

        return (
            f"Compacted {len(old_messages)} message(s) across {len(old_groups)} turn(s): "
            f"~{before_tokens:,} -> ~{after_tokens:,} tokens."
        )

    # ------------------------------------------------------------------
    def stats(self, tools: list[dict[str, Any]] | None = None) -> str:
        estimate = self.estimate(tools)
        pct = estimate / self.hard_limit * 100
        return (
            f"Messages: {len(self.messages)}  "
            f"Estimated context: ~{estimate:,} tokens ({pct:.1f}% of limit)  "
            f"Compactions: {self.compactions}  "
            f"Estimator: {self.estimator.describe()}"
        )

    def validate(self) -> list[str]:
        """Check the tool-call pairing invariant. Used in tests and by /doctor."""
        problems: list[str] = []
        open_ids: set[str] = set()
        for index, message in enumerate(self.messages):
            role = message.get("role")
            if role == "assistant":
                open_ids = {
                    call.get("id") for call in (message.get("tool_calls") or ())
                }
            elif role == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in open_ids:
                    problems.append(
                        f"message {index}: tool result {call_id!r} has no preceding "
                        f"assistant tool_call with that id"
                    )
        return problems


def structural_summary(messages: list[dict[str, Any]]) -> str:
    """Deterministic fallback summary.

    Records what was asked, which tools ran against which paths, and what the
    assistant concluded. Loses nuance, but never lies about what happened and
    costs nothing.
    """
    import json

    user_asks: list[str] = []
    tool_uses: list[str] = []
    conclusions: list[str] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "user" and isinstance(content, str):
            user_asks.append(content.strip()[:300])
        elif role == "assistant":
            for call in message.get("tool_calls") or ():
                function = call.get("function", {})
                name = function.get("name", "?")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                target = args.get("path") or args.get("script") or args.get("pattern") or ""
                tool_uses.append(f"{name}({target})" if target else name)
            if isinstance(content, str) and content.strip():
                conclusions.append(content.strip()[:400])

    parts: list[str] = []
    if user_asks:
        parts.append("Requests made:\n" + "\n".join(f"  - {a}" for a in user_asks[-6:]))
    if tool_uses:
        counts: dict[str, int] = {}
        for use in tool_uses:
            counts[use] = counts.get(use, 0) + 1
        listed = sorted(counts.items(), key=lambda kv: -kv[1])[:25]
        parts.append(
            "Tools used:\n" + "\n".join(f"  - {name} x{n}" if n > 1 else f"  - {name}" for name, n in listed)
        )
    if conclusions:
        parts.append("Assistant conclusions:\n" + "\n".join(f"  - {c}" for c in conclusions[-5:]))
    return "\n\n".join(parts) or "(no recoverable detail)"
