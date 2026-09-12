"""Token estimation.

DeepSeek does not publish a tokeniser package, so exact local counting is not
available. Rather than guess blindly, this module estimates from character
classes and then *calibrates* against the exact `prompt_tokens` the API reports
after every call. After a few steps the estimate tracks reality closely, which is
what the compaction trigger needs: an estimate that drifts high wastes money on
unnecessary compaction, and one that drifts low overruns the window.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Starting ratios, characters per token, before any calibration. English prose
# runs near 4; code and identifier-dense text run lower; CJK runs far lower.
PROSE_CHARS_PER_TOKEN = 3.9
CODE_CHARS_PER_TOKEN = 3.2
CJK_CHARS_PER_TOKEN = 1.5

_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")
_CODEY = re.compile(r"[{}\[\]()<>=/\\|_;:#$%^&*+`~]")

# Per-message overhead for role markers and delimiters, and per-tool-call
# overhead for the function-calling envelope.
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 12


@dataclass
class TokenEstimator:
    chars_per_token: float = PROSE_CHARS_PER_TOKEN
    samples: int = 0
    # Exponential smoothing factor for calibration. Low enough that one odd
    # request does not swing the ratio, high enough to converge in a few steps.
    alpha: float = 0.35

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK.findall(text))
        remaining = len(text) - cjk
        codey = len(_CODEY.findall(text))
        density = codey / max(1, remaining)
        # Blend the prose and code ratios by how punctuation-dense the text is.
        ratio = self.chars_per_token
        if density > 0.04:
            blend = min(1.0, (density - 0.04) / 0.12)
            ratio = ratio * (1 - blend) + CODE_CHARS_PER_TOKEN * blend
        return int(cjk / CJK_CHARS_PER_TOKEN + remaining / max(1.0, ratio)) + 1

    def estimate_message(self, message: dict[str, Any]) -> int:
        total = MESSAGE_OVERHEAD_TOKENS
        content = message.get("content")
        if isinstance(content, str):
            total += self.estimate_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += self.estimate_text(str(part.get("text", "")))

        # reasoning_content is resent in thinking-mode multi-turn conversations
        # and is billed like any other input, so it must be counted.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            total += self.estimate_text(reasoning)

        for call in message.get("tool_calls", ()) or ():
            total += TOOL_CALL_OVERHEAD_TOKENS
            function = call.get("function", {}) if isinstance(call, dict) else {}
            total += self.estimate_text(str(function.get("name", "")))
            total += self.estimate_text(str(function.get("arguments", "")))
        return total

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.estimate_message(m) for m in messages)

    def estimate_tools(self, tools: list[dict[str, Any]]) -> int:
        if not tools:
            return 0
        return self.estimate_text(json.dumps(tools, separators=(",", ":")))

    def calibrate(self, estimated: int, actual: int) -> None:
        """Pull the ratio toward whatever the API actually charged."""
        if estimated <= 0 or actual <= 0:
            return
        implied = self.chars_per_token * (estimated / actual)
        # Clamp to a sane band so one anomalous response cannot wreck the model.
        implied = max(1.5, min(8.0, implied))
        self.chars_per_token = (1 - self.alpha) * self.chars_per_token + self.alpha * implied
        self.samples += 1

    def describe(self) -> str:
        state = "calibrated" if self.samples else "uncalibrated"
        return f"{self.chars_per_token:.2f} chars/token ({state}, {self.samples} sample(s))"
