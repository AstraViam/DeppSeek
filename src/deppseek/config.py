"""Layered configuration.

Resolution order, lowest precedence first:

  1. Built-in defaults in this module.
  2. User config:    %USERPROFILE%\\.deppseek\\config.toml
  3. Project config: <workspace>\\.deppseek\\config.toml
  4. Environment variables (DEEPSEEK_* / DEPPSEEK_*).
  5. Explicit CLI flags.

Project config beats user config so a repository can tighten permissions or pin a
model without the user editing their global file. CLI flags always win so a
one-off override never requires touching a file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError

CONFIG_DIRNAME = ".deppseek"
CONFIG_FILENAME = "config.toml"

# Autonomy tiers. These select a *preset* of permission rules; a project config
# can still override individual rules on top of the preset.
AUTONOMY_TIERS = ("readonly", "ask", "standard", "autonomous")

REASONING_LEVELS = ("low", "high", "max")


@dataclass(frozen=True)
class BudgetConfig:
    """Hard ceilings for a single run. Hitting one aborts rather than truncates."""

    max_cost_usd: float | None = 5.0
    max_total_tokens: int | None = None
    max_steps: int = 40
    # Per-tool wall-clock ceiling, seconds. Individual tools may ask for less.
    default_timeout_s: int = 120
    max_timeout_s: int = 1800


@dataclass(frozen=True)
class ContextConfig:
    """Context-window budgeting.

    DeepSeek models expose a 1M-token window, but cost scales with every resent
    token, so we budget well below the hard limit and compact early. `soft_limit`
    is where compaction triggers; `hard_limit` is where we refuse to send.
    """

    hard_limit: int = 960_000
    soft_limit: int = 420_000
    # Tool results larger than this are spilled to disk and replaced with a
    # pointer plus a head/tail excerpt, so one big file read cannot dominate.
    max_tool_result_chars: int = 24_000
    # Number of most-recent turns that compaction will never touch.
    keep_recent_turns: int = 8
    # Emitted when compaction runs, so the model knows history was summarised.
    enable_compaction: bool = True


@dataclass(frozen=True)
class MatlabConfig:
    exe: str = "matlab"
    # Prefer the MATLAB Engine API for Python (warm workspace) when importable.
    prefer_engine: bool = True
    # Fall back to `matlab -batch` with journal replay when the engine is absent.
    allow_batch_fallback: bool = True
    startup_timeout_s: int = 180
    default_timeout_s: int = 300
    # Capture open figures to PNG after every MATLAB call.
    capture_figures: bool = True


@dataclass(frozen=True)
class VisionConfig:
    """Optional multimodal endpoint used to actually look at generated figures.

    DeepSeek's text models are not multimodal, so figure inspection requires a
    separate OpenAI-compatible vision endpoint. When unset, figure tools still
    work but report metadata and numeric summaries instead of visual reading.
    """

    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "DEPPSEEK_VISION_API_KEY"
    max_image_bytes: int = 4_000_000

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model and os.getenv(self.api_key_env))


@dataclass(frozen=True)
class SearchConfig:
    """Research tooling.

    arXiv and Crossref are keyless public APIs, so paper lookup always works.
    General web search needs a provider key; without one the search tool reports
    that it is unconfigured rather than silently returning nothing.
    """

    provider: str = "none"  # none | brave | tavily | serper
    api_key_env: str = "DEPPSEEK_SEARCH_API_KEY"
    max_results: int = 8
    timeout_s: int = 30
    user_agent: str = "deppseek/2.0 (+https://github.com/AstraViam/DeppSeek)"

    @property
    def web_enabled(self) -> bool:
        return self.provider != "none" and bool(os.getenv(self.api_key_env))


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    enabled: bool = True
    startup_timeout_s: int = 30


@dataclass(frozen=True)
class UIConfig:
    mode: str = "inline"  # inline | tui
    show_reasoning: bool = True
    # Collapse streamed reasoning to this many visible lines in inline mode.
    reasoning_lines: int = 6
    syntax_theme: str = "monokai"
    max_diff_lines: int = 160
    unicode: bool = True


@dataclass(frozen=True)
class Config:
    workspace: Path
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    thinking: bool = True
    reasoning_effort: str | int = "high"
    autonomy: str = "autonomous"
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    matlab: MatlabConfig = field(default_factory=MatlabConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    mcp_servers: tuple[McpServerConfig, ...] = ()
    # Extra permission rules layered on top of the autonomy preset.
    permission_rules: tuple[dict[str, Any], ...] = ()
    # Glob patterns that are never readable, to keep credentials out of the API.
    secret_paths: tuple[str, ...] = (
        ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12",
        "id_rsa", "id_ed25519", "*.keystore", ".netrc", "credentials",
        "*secrets*.json", "*secrets*.yaml", "*secrets*.yml",
    )
    session_dir: Path | None = None
    # Number of tool calls the loop may execute concurrently in one step.
    max_parallel_tools: int = 4

    @property
    def state_dir(self) -> Path:
        """Per-workspace state: sessions, checkpoints, spilled tool output."""
        return self.workspace / CONFIG_DIRNAME

    def validate(self) -> None:
        if self.autonomy not in AUTONOMY_TIERS:
            raise ConfigError(
                f"autonomy must be one of {AUTONOMY_TIERS}, got {self.autonomy!r}"
            )
        effort = self.reasoning_effort
        # V4.1 Flash also accepts a numeric effort, so a digit string is valid
        # even though it is not one of the named levels.
        if (
            isinstance(effort, str)
            and effort not in REASONING_LEVELS
            and not effort.isdigit()
        ):
            raise ConfigError(
                f"reasoning_effort must be one of {REASONING_LEVELS} or 1-100, "
                f"got {effort!r}"
            )
        if isinstance(effort, int) and not 1 <= effort <= 100:
            raise ConfigError(f"numeric reasoning_effort must be 1-100, got {effort}")
        if self.context.soft_limit >= self.context.hard_limit:
            raise ConfigError("context.soft_limit must be below context.hard_limit")
        if self.ui.mode not in ("inline", "tui"):
            raise ConfigError(f"ui.mode must be 'inline' or 'tui', got {self.ui.mode!r}")
        if self.max_parallel_tools < 1:
            raise ConfigError("max_parallel_tools must be >= 1")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Could not read config {path}: {exc}") from exc


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge `overlay` into `base`, recursing into nested tables.

    Lists replace rather than concatenate: a project that declares MCP servers
    means "these servers", not "these in addition to the user's".
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overlay() -> dict[str, Any]:
    """Environment variables, mapped onto the config tree.

    DEEPSEEK_* names are kept for backward compatibility with the v1 script;
    DEPPSEEK_* names cover everything new.
    """
    overlay: dict[str, Any] = {}

    def put(path: str, value: Any) -> None:
        node = overlay
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    simple: dict[str, str] = {
        "DEEPSEEK_MODEL": "model",
        "DEPPSEEK_MODEL": "model",
        "DEEPSEEK_BASE_URL": "base_url",
        "DEPPSEEK_AUTONOMY": "autonomy",
        "DEPPSEEK_UI_MODE": "ui.mode",
        "DEPPSEEK_SYNTAX_THEME": "ui.syntax_theme",
        "MATLAB_EXE": "matlab.exe",
        "DEPPSEEK_VISION_BASE_URL": "vision.base_url",
        "DEPPSEEK_VISION_MODEL": "vision.model",
        "DEPPSEEK_SEARCH_PROVIDER": "search.provider",
    }
    for env_name, dotted in simple.items():
        raw = os.getenv(env_name)
        if raw:
            put(dotted, raw)

    effort = os.getenv("DEEPSEEK_REASONING") or os.getenv("DEPPSEEK_REASONING")
    if effort:
        put("reasoning_effort", int(effort) if effort.isdigit() else effort)

    thinking = os.getenv("DEPPSEEK_THINKING")
    if thinking is not None:
        put("thinking", thinking.strip().lower() in {"1", "true", "on", "yes"})

    budget = os.getenv("DEPPSEEK_MAX_COST_USD")
    if budget:
        try:
            put("budget.max_cost_usd", float(budget))
        except ValueError as exc:
            raise ConfigError(f"DEPPSEEK_MAX_COST_USD must be a number: {exc}") from exc

    steps = os.getenv("DEPPSEEK_MAX_STEPS")
    if steps:
        if not steps.isdigit():
            raise ConfigError("DEPPSEEK_MAX_STEPS must be a positive integer")
        put("budget.max_steps", int(steps))

    return overlay


def _build_mcp_servers(raw: Any) -> tuple[McpServerConfig, ...]:
    if not raw:
        return ()
    if not isinstance(raw, dict):
        raise ConfigError("[mcp.servers] must be a table of server definitions")
    servers: list[McpServerConfig] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"MCP server {name!r} must be a table")
        command = spec.get("command")
        if not command:
            raise ConfigError(f"MCP server {name!r} is missing 'command'")
        env_items = spec.get("env", {}) or {}
        if not isinstance(env_items, dict):
            raise ConfigError(f"MCP server {name!r}: 'env' must be a table")
        servers.append(
            McpServerConfig(
                name=name,
                command=str(command),
                args=tuple(str(a) for a in spec.get("args", ()) or ()),
                env=tuple((str(k), str(v)) for k, v in env_items.items()),
                enabled=bool(spec.get("enabled", True)),
                startup_timeout_s=int(spec.get("startup_timeout_s", 30)),
            )
        )
    return tuple(servers)


def _subconfig(cls: type, raw: Any, label: str) -> Any:
    """Instantiate a frozen dataclass from a TOML table, rejecting unknown keys."""
    if not raw:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"[{label}] must be a table")
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"[{label}] has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    return cls(**raw)


def user_config_path() -> Path:
    return Path.home() / CONFIG_DIRNAME / CONFIG_FILENAME


def project_config_path(workspace: Path) -> Path:
    return workspace / CONFIG_DIRNAME / CONFIG_FILENAME


def load_config(workspace: Path, cli_overrides: dict[str, Any] | None = None) -> Config:
    """Assemble the effective config for a workspace."""
    workspace = workspace.expanduser().resolve()

    merged: dict[str, Any] = {}
    for path in (user_config_path(), project_config_path(workspace)):
        merged = _deep_merge(merged, _read_toml(path))
    merged = _deep_merge(merged, _env_overlay())
    merged = _deep_merge(merged, _strip_none(cli_overrides or {}))

    mcp_raw = (merged.pop("mcp", {}) or {}).get("servers", {})
    permissions_raw = merged.pop("permissions", {}) or {}
    rules = tuple(permissions_raw.get("rules", ()) or ())
    if "autonomy" in permissions_raw and "autonomy" not in merged:
        merged["autonomy"] = permissions_raw["autonomy"]

    secret_paths = merged.pop("secret_paths", None)

    cfg_kwargs: dict[str, Any] = {
        "workspace": workspace,
        "budget": _subconfig(BudgetConfig, merged.pop("budget", None), "budget"),
        "context": _subconfig(ContextConfig, merged.pop("context", None), "context"),
        "matlab": _subconfig(MatlabConfig, merged.pop("matlab", None), "matlab"),
        "vision": _subconfig(VisionConfig, merged.pop("vision", None), "vision"),
        "search": _subconfig(SearchConfig, merged.pop("search", None), "search"),
        "ui": _subconfig(UIConfig, merged.pop("ui", None), "ui"),
        "mcp_servers": _build_mcp_servers(mcp_raw),
        "permission_rules": rules,
    }
    if secret_paths is not None:
        cfg_kwargs["secret_paths"] = tuple(str(p) for p in secret_paths)

    known_top = set(Config.__dataclass_fields__)
    unknown = set(merged) - known_top
    if unknown:
        raise ConfigError(
            f"Unknown top-level config key(s): {', '.join(sorted(unknown))}"
        )
    cfg_kwargs.update(merged)

    config = Config(**cfg_kwargs)
    config.validate()
    return config


def _strip_none(data: dict[str, Any]) -> dict[str, Any]:
    """Drop None values so unset CLI flags do not clobber file config."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _strip_none(value)
            if nested:
                out[key] = nested
        else:
            out[key] = value
    return out


def with_overrides(config: Config, **kwargs: Any) -> Config:
    """Return a copy of `config` with top-level fields replaced, then revalidated."""
    updated = replace(config, **kwargs)
    updated.validate()
    return updated
