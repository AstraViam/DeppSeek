"""The agent loop.

Differences from v1 that change how the thing feels to use:

* **It streams.** v1 ran agent mode with `stream=False`, so a step that took
  forty seconds showed nothing at all until it finished.
* **Independent reads run in parallel.** When the model asks for four file reads
  in one turn, they execute concurrently. Mutating tools are always serialised,
  because two concurrent edits to one file is a race with no upside.
* **It has a budget.** Cost, step, and token ceilings are checked before each
  request, and hitting one stops with a clear statement rather than silently
  continuing to spend.
* **It compacts.** When the conversation approaches the soft context limit, the
  oldest turns are summarised and dropped instead of the run failing.
* **Interrupts are graceful.** Ctrl-C stops after the current tool call and
  leaves the session resumable, rather than losing the conversation.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .errors import BudgetExceeded, ProviderError
from .prompts import SUMMARISER_PROMPT
from .providers.base import Completion, NullSink, StreamSink, ToolCall
from .session.context import ConversationBuffer
from .tools.registry import ToolResult


@dataclass
class StepRecord:
    """One iteration of the loop, for reporting and debugging."""

    index: int
    tool_calls: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    had_errors: bool = False
    compacted: bool = False


@dataclass
class RunResult:
    answer: str
    steps: list[StepRecord] = field(default_factory=list)
    stopped_reason: str = "completed"
    interrupted: bool = False

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)


class AgentLoop:
    def __init__(
        self,
        *,
        provider: Any,
        toolbox: Any,
        buffer: ConversationBuffer,
        config: Any,
        sink: StreamSink | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.toolbox = toolbox
        self.buffer = buffer
        self.config = config
        self.sink = sink or NullSink()
        self.on_event = on_event or (lambda name, payload: None)
        self.interrupted = False
        self._start_cost = 0.0

    # ------------------------------------------------------------------
    def _emit(self, event: str, /, **payload: Any) -> None:
        """Emit a UI event.

        `event` is positional-only: several payloads carry their own "name" key
        (the tool being run), which would otherwise collide with the parameter.
        """
        with contextlib.suppress(Exception):
            self.on_event(event, payload)

    def _check_budget(self, step: int, max_steps: int) -> None:
        budget = self.config.budget
        if step > max_steps:
            raise BudgetExceeded(
                f"Reached the step limit of {max_steps}. The work so far is saved; "
                f"continue with another instruction, or raise --max-steps."
            )
        spent = self.provider.usage.estimated_cost_usd - self._start_cost
        if budget.max_cost_usd is not None and spent >= budget.max_cost_usd:
            raise BudgetExceeded(
                f"Reached the cost ceiling of ${budget.max_cost_usd:.2f} "
                f"(estimated ${spent:.4f} spent this run). Raise it with "
                f"--max-cost or in .deppseek/config.toml."
            )
        if (
            budget.max_total_tokens is not None
            and self.provider.usage.total_tokens >= budget.max_total_tokens
        ):
            raise BudgetExceeded(
                f"Reached the token ceiling of {budget.max_total_tokens:,}."
            )

    def _maybe_compact(self, tools: list[dict[str, Any]]) -> bool:
        if not self.buffer.needs_compaction(tools):
            return False
        message = self.buffer.compact(summarise=self._summarise, tools=tools)
        self._emit("compacted", message=message)
        return True

    def _summarise(self, messages: list[dict[str, Any]]) -> str:
        """Summarise dropped history with a cheap non-thinking call.

        Thinking is disabled for this call: the summary is a mechanical
        transformation and reasoning tokens on it are pure cost.
        """
        from .providers.deepseek import DeepSeekProvider

        transcript_parts: list[str] = []
        for message in messages:
            role = message.get("role", "?")
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                transcript_parts.append(f"[{role}] {content[:2000]}")
            for call in message.get("tool_calls") or ():
                function = call.get("function", {})
                transcript_parts.append(
                    f"[tool call] {function.get('name')}({str(function.get('arguments'))[:300]})"
                )
        transcript = "\n".join(transcript_parts)[:120_000]

        if not isinstance(self.provider, DeepSeekProvider):
            return ""

        saved_thinking = self.provider.thinking
        try:
            self.provider.thinking = False
            completion = self.provider.complete(
                [
                    {"role": "system", "content": SUMMARISER_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                tools=None,
                sink=None,
                stream=False,
            )
            return completion.content.strip()
        except ProviderError:
            return ""  # structural summary is used instead
        finally:
            self.provider.thinking = saved_thinking

    # ------------------------------------------------------------------
    def run(self, task: str, max_steps: int | None = None) -> RunResult:
        """Run one task to completion, or until a limit stops it."""
        max_steps = max_steps or self.config.budget.max_steps
        self._start_cost = self.provider.usage.estimated_cost_usd
        self.buffer.append_user(task)

        steps: list[StepRecord] = []
        answer = ""
        stopped = "completed"

        for step_index in range(1, max_steps + 2):
            record = StepRecord(index=step_index)
            started = time.monotonic()

            try:
                self._check_budget(step_index, max_steps)
            except BudgetExceeded as exc:
                stopped = str(exc)
                answer = answer or stopped
                self._emit("budget", message=stopped)
                break

            tools = self.toolbox.schemas()
            record.compacted = self._maybe_compact(tools)

            if self.buffer.over_hard_limit(tools):
                stopped = (
                    "Context is at its hard limit and could not be compacted further. "
                    "Start a fresh session, or narrow the task."
                )
                answer = answer or stopped
                break

            self.toolbox.ctx.step = step_index
            self._emit("step_start", index=step_index, max_steps=max_steps)

            try:
                completion = self.provider.complete(
                    self.buffer.render(), tools, sink=self.sink, stream=True
                )
            except KeyboardInterrupt:
                self.interrupted = True
                stopped = "interrupted by user during model response"
                break
            except ProviderError as exc:
                stopped = f"Provider error: {exc}"
                answer = answer or stopped
                self._emit("error", message=stopped)
                break

            self._calibrate(completion)
            self.buffer.append(completion.to_message())

            if not completion.tool_calls:
                answer = completion.content
                record.duration_s = time.monotonic() - started
                steps.append(record)
                self._emit("final", content=answer)
                break

            record.tool_calls = [call.name for call in completion.tool_calls]
            try:
                results = self._execute_tool_calls(completion.tool_calls)
            except KeyboardInterrupt:
                self.interrupted = True
                stopped = "interrupted by user during tool execution"
                # Every issued tool call must still get a reply, or the next
                # request is malformed on the wire.
                self._backfill_tool_results(completion.tool_calls)
                break

            record.had_errors = any(result.is_error for _, result in results)
            for call, result in results:
                self.buffer.append_tool_result(call.id, result.content)

            record.duration_s = time.monotonic() - started
            steps.append(record)
        else:
            stopped = (
                f"Stopped after {max_steps} steps without a final answer. "
                f"The conversation is saved; continue with another instruction."
            )
            answer = answer or stopped

        if self.interrupted:
            self._emit("interrupted", message=stopped)

        return RunResult(
            answer=answer, steps=steps, stopped_reason=stopped, interrupted=self.interrupted
        )

    # ------------------------------------------------------------------
    def _calibrate(self, completion: Completion) -> None:
        usage = completion.raw_usage
        if usage is None:
            return
        actual = getattr(usage, "prompt_tokens", 0) or 0
        if actual:
            self.buffer.calibrate(actual)

    def _execute_tool_calls(
        self, calls: list[ToolCall]
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Run a turn's tool calls, in parallel where that is safe.

        Safety rule: a call is parallelisable only if its tool does not mutate
        and is not interactive. Anything that writes, deletes, or may prompt for
        approval runs sequentially in the order the model asked for, because
        both ordering and the user's attention are single-threaded.
        """
        specs = self.toolbox.all_specs()

        def is_parallel_safe(call: ToolCall) -> bool:
            spec = specs.get(call.name)
            return bool(spec and not spec.mutates and not spec.slow)

        parallel = [c for c in calls if is_parallel_safe(c)]
        serial = [c for c in calls if not is_parallel_safe(c)]
        results: dict[str, ToolResult] = {}

        if len(parallel) > 1:
            workers = min(self.config.max_parallel_tools, len(parallel))
            self._emit("parallel_tools", count=len(parallel), workers=workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_one, call): call for call in parallel
                }
                for future, call in futures.items():
                    results[call.id] = future.result()
        else:
            for call in parallel:
                results[call.id] = self._run_one(call)

        for call in serial:
            results[call.id] = self._run_one(call)

        # Return in the model's original order so the transcript reads sensibly.
        return [(call, results[call.id]) for call in calls]

    def _run_one(self, call: ToolCall) -> ToolResult:
        self._emit("tool_start", name=call.name, arguments=call.arguments)
        result = self.toolbox.execute(call.name, call.arguments)
        self._emit(
            "tool_end",
            name=call.name,
            display=result.display,
            is_error=result.is_error,
            duration_s=result.duration_s,
        )
        return result

    def _backfill_tool_results(self, calls: list[ToolCall]) -> None:
        """Reply to every outstanding tool call after an interrupt.

        The wire format requires one tool message per issued tool_call_id.
        Leaving them unanswered makes the saved session unresumable.
        """
        answered = {
            message.get("tool_call_id")
            for message in self.buffer.messages
            if message.get("role") == "tool"
        }
        for call in calls:
            if call.id not in answered:
                self.buffer.append_tool_result(
                    call.id, "Cancelled: the user interrupted the run before this tool ran."
                )
