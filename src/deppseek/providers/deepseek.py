"""DeepSeek provider.

Notable behaviour, all of it load-bearing:

* **Streaming with tool calls.** Tool-call arguments arrive as fragments indexed
  by position and must be reassembled; v1 disabled streaming in agent mode
  entirely, so a long step showed nothing until it completed. Here the loop
  streams reasoning, prose, and the *names* of tools as soon as they are known.
* **Retry with backoff.** Rate limits and 5xx responses are retried with
  exponential backoff plus jitter, honouring `Retry-After` when present. A bare
  exception used to end the whole task.
* **Cache-stable prefixes.** The request is assembled so that the system prompt
  and tool schemas are byte-identical across steps, which is what lets DeepSeek's
  context cache hit and cuts input cost by roughly 50x on the cached portion.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from typing import Any

from openai import OpenAI

from ..errors import ProviderError, RetryableProviderError
from .base import Completion, StreamSink, ToolCall
from .pricing import Usage, pricing_for, resolve_model

# Substrings that mark an error as worth retrying. Matching on text is ugly but
# the OpenAI SDK wraps transport failures in ways that vary by version.
RETRYABLE_MARKERS = (
    "rate limit", "rate_limit", "429",
    "500", "502", "503", "504",
    "overloaded", "timeout", "timed out",
    "connection", "temporarily unavailable",
    "server_error", "service unavailable",
)


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.deepseek.com",
        thinking: bool = True,
        reasoning_effort: str | int = "high",
        max_retries: int = 5,
        request_timeout_s: float = 600.0,
        usage: Usage | None = None,
        on_warning: Any = None,
    ) -> None:
        resolved, warning = resolve_model(model)
        self.model = resolved
        if warning and on_warning:
            on_warning(warning)

        self.thinking = thinking
        self.reasoning_effort = self._normalise_effort(reasoning_effort, on_warning)
        self.max_retries = max(0, max_retries)
        self.usage = usage or Usage()
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout_s,
            # The SDK's own retry would bypass our backoff accounting and logging.
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Request assembly
    # ------------------------------------------------------------------
    def _normalise_effort(self, effort: str | int, on_warning: Any) -> str | int:
        """Clamp numeric effort, and downgrade it when the model cannot take it."""
        pricing = pricing_for(self.model)
        if isinstance(effort, str) and effort.isdigit():
            effort = int(effort)
        if isinstance(effort, int):
            if not pricing.supports_numeric_effort:
                named = "max" if effort >= 80 else "high" if effort >= 40 else "low"
                if on_warning:
                    on_warning(
                        f"{self.model} does not accept numeric reasoning effort; "
                        f"using {named!r} instead of {effort}."
                    )
                return named
            return max(1, min(100, effort))
        return effort

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if stream:
            # Usage is omitted from streamed responses unless asked for, which is
            # how v1's streaming path lost all cost accounting.
            request["stream_options"] = {"include_usage": True}
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        return request

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        sink: StreamSink | None = None,
        stream: bool = True,
    ) -> Completion:
        request = self._build_request(messages, tools, stream)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    return self._complete_streaming(request, sink)
                return self._complete_blocking(request)
            except RetryableProviderError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = exc.retry_after if exc.retry_after else _backoff(attempt)
                if sink is not None:
                    sink.on_content(
                        f"\n[retrying in {delay:.1f}s after: {exc}]\n"
                    )
                time.sleep(delay)
            except ProviderError:
                raise
            except Exception as exc:
                classified = _classify(exc)
                if isinstance(classified, RetryableProviderError) and attempt < self.max_retries:
                    last_error = classified
                    delay = classified.retry_after or _backoff(attempt)
                    if sink is not None:
                        sink.on_content(f"\n[retrying in {delay:.1f}s after: {exc}]\n")
                    time.sleep(delay)
                    continue
                raise classified from exc

        raise ProviderError(
            f"Giving up after {self.max_retries + 1} attempts: {last_error}"
        )

    def _complete_blocking(self, request: dict[str, Any]) -> Completion:
        response = self._client.chat.completions.create(**request)
        self.usage.add_response(response, self.model)

        if not getattr(response, "choices", None):
            raise ProviderError("Provider returned no choices")
        message = response.choices[0].message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        return Completion(
            content=getattr(message, "content", None) or "",
            reasoning=getattr(message, "reasoning_content", None) or "",
            tool_calls=tool_calls,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
            raw_usage=getattr(response, "usage", None),
        )

    def _complete_streaming(
        self, request: dict[str, Any], sink: StreamSink | None
    ) -> Completion:
        stream = self._client.chat.completions.create(**request)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Tool calls arrive fragmented and out of order; key by the delta index.
        partial: dict[int, dict[str, str]] = {}
        announced: set[int] = set()
        finish_reason: str | None = None
        final_usage: Any = None

        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    final_usage = usage

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if sink is not None:
                        sink.on_reasoning(reasoning)

                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    if sink is not None:
                        sink.on_content(content)

                for tc_delta in getattr(delta, "tool_calls", None) or []:
                    index = getattr(tc_delta, "index", 0) or 0
                    slot = partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc_delta, "id", None):
                        slot["id"] = tc_delta.id
                    fn = getattr(tc_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments
                    # Announce the tool as soon as its name is complete enough to
                    # show, so the UI is not silent during a long argument stream.
                    if slot["name"] and index not in announced:
                        announced.add(index)
                        if sink is not None:
                            sink.on_tool_call_start(slot["name"])
        except Exception as exc:
            raise _classify(exc) from exc
        finally:
            if sink is not None:
                sink.on_done()
            close = getattr(stream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        if final_usage is not None:
            self.usage.add_response(_UsageCarrier(final_usage), self.model)

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for index, slot in sorted(partial.items())
            if slot["name"]
        ]

        return Completion(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_usage=final_usage,
        )


class _UsageCarrier:
    """Adapts a bare usage object to the `.usage` attribute Usage.add_response wants."""

    def __init__(self, usage: Any) -> None:
        self.usage = usage


def _backoff(attempt: int, base: float = 1.5, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter.

    Full jitter rather than fixed delay so that several concurrent subagents
    retrying after the same rate limit do not synchronise into another burst.
    """
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0.0, ceiling)


def _classify(exc: Exception) -> ProviderError:
    """Decide whether an SDK exception is worth retrying."""
    retry_after = _extract_retry_after(exc)
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        if status == 429 or 500 <= status < 600:
            return RetryableProviderError(f"HTTP {status}: {exc}", retry_after)
        if status == 401:
            return ProviderError(
                "Authentication failed (HTTP 401). Check DEEPSEEK_API_KEY."
            )
        if status == 402:
            return ProviderError(
                "Insufficient balance (HTTP 402). Top up your DeepSeek account."
            )
        if status == 400:
            return ProviderError(f"Bad request (HTTP 400): {exc}")

    text = str(exc).lower()
    if any(marker in text for marker in RETRYABLE_MARKERS):
        return RetryableProviderError(str(exc), retry_after)
    return ProviderError(str(exc))


def _extract_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(60.0, float(raw)))
    except (TypeError, ValueError):
        return None


def tool_schema_digest(tools: list[dict[str, Any]]) -> str:
    """Stable digest of the tool schemas.

    Used to detect that the tool set changed between steps, which invalidates the
    provider-side prefix cache and is worth knowing about when cache hit rates
    drop unexpectedly.
    """
    import hashlib

    payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
