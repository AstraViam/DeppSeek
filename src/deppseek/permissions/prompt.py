"""Approval prompt.

v1 printed a wall of exclamation marks and asked `y/N`, with the file's new
content nowhere in sight. Approving a write meant approving content you could
not see, which makes the gate theatre rather than control.

This module shows the actual change: a unified diff for edits, the command text
for shell calls, the resolved absolute path for anything touching the filesystem.
It also offers a fourth answer beyond yes and no -- "always, this session" --
which is what makes a gated tier usable across a long task.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass

from .rules import Decision, Request, Verdict

# Answers accepted at the prompt.
_YES = {"y", "yes"}
_NO = {"n", "no", ""}
_ALWAYS = {"a", "always"}
_NEVER = {"d", "deny", "never"}


@dataclass
class Approval:
    approved: bool
    remember: bool = False
    # Set when the user chose "never": the tool is denied for the session.
    deny_for_session: bool = False
    note: str = ""


def unified_diff(
    before: str,
    after: str,
    path: str,
    *,
    max_lines: int = 160,
    context: int = 3,
) -> str:
    """Unified diff, truncated in the middle rather than at the end.

    Truncating at the end hides the tail of a large refactor, which is usually
    where the damage is. Keeping both ends shows what the change starts and
    finishes with.
    """
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=False),
            after.splitlines(keepends=False),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=context,
        )
    )
    if not diff:
        return "(no textual change)"
    if len(diff) <= max_lines:
        return "\n".join(diff)

    head = max_lines // 2
    tail = max_lines - head
    omitted = len(diff) - head - tail
    return "\n".join(
        [*diff[:head], f"... {omitted} more diff lines omitted ...", *diff[-tail:]]
    )


def diff_stats(before: str, after: str) -> tuple[int, int]:
    """Count added and removed lines. Cheap summary for a one-line status."""
    added = removed = 0
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0
    ):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def format_request(request: Request, verdict: Verdict, *, max_diff_lines: int = 160) -> str:
    """Render the pending action as text, for any front end."""
    lines = [f"Tool:   {request.tool}"]
    if request.summary:
        lines.append(f"Action: {request.summary}")
    if request.path:
        lines.append(f"Path:   {request.path}")
    if request.command:
        lines.append("Command:")
        for line in request.command.splitlines()[:20]:
            lines.append(f"  {line}")
    if request.details:
        lines.append(request.details)
    if request.diff:
        lines.append("")
        diff_lines = request.diff.splitlines()
        if len(diff_lines) > max_diff_lines:
            head = max_diff_lines // 2
            tail = max_diff_lines - head
            diff_lines = (
                [*diff_lines[:head], f"... {len(diff_lines) - head - tail} more diff lines omitted ...", *diff_lines[-tail:]]
            )
        lines.extend(diff_lines)
    lines.append("")
    lines.append(f"Reason for asking: {verdict.reason}")
    return "\n".join(lines)


def ask_plain(
    request: Request,
    verdict: Verdict,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_diff_lines: int = 160,
    allow_remember: bool = True,
) -> Approval:
    """Plain-text approval prompt. Used when rich is unavailable or piped."""
    output_fn("")
    output_fn("-" * 72)
    output_fn("APPROVAL REQUIRED")
    output_fn("-" * 72)
    output_fn(format_request(request, verdict, max_diff_lines=max_diff_lines))
    output_fn("-" * 72)

    options = "[y]es / [n]o"
    if allow_remember and verdict.rule is not None and verdict.rule.remberable:
        options += " / [a]lways this session / [d]eny for session"

    while True:
        try:
            answer = input_fn(f"Approve? {options}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return Approval(approved=False, note="interrupted")

        if answer in _YES:
            return Approval(approved=True)
        if answer in _NO:
            return Approval(approved=False, note="denied by user")
        if allow_remember and answer in _ALWAYS:
            return Approval(approved=True, remember=True)
        if allow_remember and answer in _NEVER:
            return Approval(
                approved=False, remember=True, deny_for_session=True, note="denied for session"
            )
        output_fn(f"Please answer one of: {options}")


class Approver:
    """Couples the rule engine to whatever front end is asking the questions.

    The agent loop calls `check()` and gets back a simple allow/deny, with every
    prompting, remembering, and logging detail handled here.
    """

    def __init__(
        self,
        engine,
        *,
        ask_fn: Callable[[Request, Verdict], Approval] | None = None,
        max_diff_lines: int = 160,
        non_interactive: bool = False,
    ) -> None:
        self.engine = engine
        self.max_diff_lines = max_diff_lines
        self.non_interactive = non_interactive
        self._ask = ask_fn or (
            lambda req, verdict: ask_plain(
                req, verdict, max_diff_lines=self.max_diff_lines
            )
        )
        self.log: list[tuple[Request, Decision, str]] = []

    def check(self, request: Request) -> tuple[bool, str]:
        """Return `(allowed, explanation)` for a pending tool call."""
        verdict = self.engine.evaluate(request)

        if verdict.decision is Decision.ALLOW:
            self.log.append((request, Decision.ALLOW, verdict.reason))
            return True, verdict.reason

        if verdict.decision is Decision.DENY:
            self.log.append((request, Decision.DENY, verdict.reason))
            return False, f"Denied: {verdict.reason}"

        # ASK, with nobody to ask.
        if self.non_interactive:
            self.log.append((request, Decision.DENY, "non-interactive"))
            return False, (
                f"Denied: this action needs approval ({verdict.reason}) but the "
                f"session is non-interactive. Re-run interactively, or set a "
                f"permission rule in .deppseek/config.toml to allow it."
            )

        approval = self._ask(request, verdict)
        if approval.remember:
            self.engine.grant_for_session(
                request, Decision.DENY if approval.deny_for_session else Decision.ALLOW
            )
        decision = Decision.ALLOW if approval.approved else Decision.DENY
        self.log.append((request, decision, approval.note or "user decision"))

        if approval.approved:
            return True, "approved by user" + (" (remembered)" if approval.remember else "")
        return False, (
            f"Denied by user{' for this session' if approval.deny_for_session else ''}. "
            f"Do not retry this action; choose a different approach or ask what to do."
        )

    def summary(self) -> str:
        if not self.log:
            return "No permission decisions recorded."
        allowed = sum(1 for _, d, _ in self.log if d is Decision.ALLOW)
        denied = len(self.log) - allowed
        lines = [f"{allowed} allowed, {denied} denied", ""]
        for request, decision, reason in self.log[-25:]:
            target = request.path or (request.command or "")[:48] or "-"
            lines.append(f"  {decision.value:5} {request.tool:16} {target:40} {reason[:40]}")
        return "\n".join(lines)
