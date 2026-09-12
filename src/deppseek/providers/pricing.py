"""Token accounting and cost estimation.

Three things the v1 script got wrong and this module fixes:

1. **Cache tiers.** DeepSeek bills prompt tokens that hit its context cache at
   roughly 2% of the cache-miss rate. Ignoring that overstates the cost of an
   agent loop by an order of magnitude, because an agent loop resends a nearly
   identical prefix every step and therefore hits cache constantly.
2. **Peak vs off-peak.** Rates differ by 2x depending on the hour, so a single
   flat rate is wrong half the time.
3. **Retired model IDs.** `deepseek-v4-flash` was retired 2026-09-10 and the
   legacy `deepseek-chat` / `deepseek-reasoner` aliases were retired
   2026-07-24. Calls with those IDs fail rather than silently downgrade.

These rates are reference values for estimation. The invoice on your DeepSeek
account is authoritative; `usage` totals from the API are exact token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any

# Off-peak discount window, UTC. DeepSeek discounts requests that start inside
# this window; it wraps past midnight, which the comparison below accounts for.
OFF_PEAK_START = time(16, 30)
OFF_PEAK_END = time(0, 30)


@dataclass(frozen=True)
class Rate:
    """USD per 1M tokens, for one billing tier."""

    cache_hit_input: float
    cache_miss_input: float
    output: float


@dataclass(frozen=True)
class ModelPricing:
    peak: Rate
    off_peak: Rate
    context_window: int
    max_output_tokens: int
    supports_thinking: bool = True
    supports_tools: bool = True
    # V4.1 Flash accepts a continuous 1-100 effort in addition to named levels.
    supports_numeric_effort: bool = False


# Live model IDs as of 2026-09-12.
MODELS: dict[str, ModelPricing] = {
    "deepseek-flash": ModelPricing(
        peak=Rate(cache_hit_input=0.006, cache_miss_input=0.30, output=1.20),
        off_peak=Rate(cache_hit_input=0.003, cache_miss_input=0.15, output=0.60),
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_numeric_effort=True,
    ),
    "deepseek-v4-pro": ModelPricing(
        peak=Rate(cache_hit_input=0.044, cache_miss_input=1.32, output=3.96),
        off_peak=Rate(cache_hit_input=0.022, cache_miss_input=0.66, output=1.98),
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_numeric_effort=False,
    ),
}

# Model IDs DeepSeek has retired, mapped to the current replacement. Calling a
# retired ID returns an API error, so we rewrite it and say so loudly.
RETIRED_MODELS: dict[str, tuple[str, str]] = {
    "deepseek-v4-flash": ("deepseek-flash", "retired 2026-09-10"),
    "deepseek-chat": ("deepseek-flash", "alias retired 2026-07-24"),
    "deepseek-reasoner": ("deepseek-flash", "alias retired 2026-07-24"),
    "deepseek-v3": ("deepseek-flash", "superseded"),
    "deepseek-r1": ("deepseek-flash", "superseded"),
}

DEFAULT_MODEL = "deepseek-flash"


def resolve_model(model: str) -> tuple[str, str | None]:
    """Map a possibly-retired model ID to a live one.

    Returns `(model_id, warning)`. `warning` is None when the ID was already
    current.
    """
    if model in MODELS:
        return model, None
    if model in RETIRED_MODELS:
        replacement, reason = RETIRED_MODELS[model]
        return replacement, (
            f"Model {model!r} is {reason}; using {replacement!r} instead. "
            f"Update your config to silence this."
        )
    # Unknown IDs are passed through untouched: DeepSeek may ship a model newer
    # than this table, and refusing it would be worse than estimating its cost
    # with the default rates.
    return model, (
        f"Model {model!r} is not in the pricing table; cost estimates will use "
        f"{DEFAULT_MODEL} rates and may be wrong."
    )


def pricing_for(model: str) -> ModelPricing:
    return MODELS.get(model) or MODELS[DEFAULT_MODEL]


def is_off_peak(moment: datetime | None = None) -> bool:
    """True when `moment` falls inside the off-peak discount window."""
    moment = moment or datetime.now(timezone.utc)
    now = moment.astimezone(timezone.utc).time()
    # The window wraps midnight, so it is a union of two ranges.
    return now >= OFF_PEAK_START or now < OFF_PEAK_END


def rate_for(model: str, moment: datetime | None = None) -> Rate:
    pricing = pricing_for(model)
    return pricing.off_peak if is_off_peak(moment) else pricing.peak


@dataclass
class Usage:
    """Accumulated token counts and estimated spend for a session."""

    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    api_calls: int = 0

    @property
    def input_tokens(self) -> int:
        return self.cache_hit_input_tokens + self.cache_miss_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens
        return self.cache_hit_input_tokens / total if total else 0.0

    def add_response(self, response: Any, model: str, moment: datetime | None = None) -> None:
        """Fold one API response's usage metadata into the running total."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = _int_attr(usage, "prompt_tokens")
        hit = _int_attr(usage, "prompt_cache_hit_tokens")
        miss = _int_attr(usage, "prompt_cache_miss_tokens")

        # Some gateways report only prompt_tokens. Attribute everything to the
        # cache-miss tier in that case, which is the conservative direction: it
        # overestimates rather than understates cost.
        if hit == 0 and miss == 0 and prompt_tokens:
            miss = prompt_tokens
        elif hit + miss != prompt_tokens and prompt_tokens:
            # Trust the breakdown for the split but the total for the magnitude.
            miss = max(0, prompt_tokens - hit)

        output = _int_attr(usage, "completion_tokens")
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = _int_attr(details, "reasoning_tokens") if details else 0

        self.cache_hit_input_tokens += hit
        self.cache_miss_input_tokens += miss
        self.output_tokens += output
        self.reasoning_tokens += reasoning
        self.api_calls += 1
        self.estimated_cost_usd += estimate_cost(
            model, hit_tokens=hit, miss_tokens=miss, output_tokens=output, moment=moment
        )

    def summary(self, model: str) -> str:
        off_peak = is_off_peak()
        rate = rate_for(model)
        lines = [
            f"Model                {model}",
            f"API calls            {self.api_calls:,}",
            f"Input  (cache hit)   {self.cache_hit_input_tokens:>12,}  @ ${rate.cache_hit_input}/1M",
            f"Input  (cache miss)  {self.cache_miss_input_tokens:>12,}  @ ${rate.cache_miss_input}/1M",
            f"Output               {self.output_tokens:>12,}  @ ${rate.output}/1M",
        ]
        if self.reasoning_tokens:
            lines.append(
                f"  of which reasoning {self.reasoning_tokens:>12,}  (billed as output)"
            )
        lines += [
            f"Total tokens         {self.total_tokens:>12,}",
            f"Cache hit rate       {self.cache_hit_rate * 100:>11.1f}%",
            f"Estimated cost       ${self.estimated_cost_usd:.6f}  "
            f"({'off-peak' if off_peak else 'peak'} rates)",
            "Estimate only; your DeepSeek invoice is authoritative.",
        ]
        return "\n".join(lines)


def estimate_cost(
    model: str,
    *,
    hit_tokens: int,
    miss_tokens: int,
    output_tokens: int,
    moment: datetime | None = None,
) -> float:
    rate = rate_for(model, moment)
    return (
        hit_tokens / 1_000_000 * rate.cache_hit_input
        + miss_tokens / 1_000_000 * rate.cache_miss_input
        + output_tokens / 1_000_000 * rate.output
    )


def _int_attr(obj: Any, name: str) -> int:
    """Read an int attribute or dict key, tolerating absence and None."""
    if obj is None:
        return 0
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
