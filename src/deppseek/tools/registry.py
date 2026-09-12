"""Tool registry.

v1 declared each tool three times: a Python method, a hand-written JSON schema,
and a branch in a long `if name == ...` dispatch chain. The three drifted, and
argument errors surfaced as raw `TypeError` text.

Here a tool is declared once. The JSON schema is generated from the function's
type hints and docstring, dispatch is a dict lookup, and four cross-cutting
concerns are handled centrally so no individual tool can forget them:

  * **Permission** -- every call is evaluated before the body runs.
  * **Checkpointing** -- mutating tools are snapshotted before and committed
    after, so undo works even for a tool whose author never thought about it.
  * **Redaction** -- every result is scrubbed before it can reach the API.
  * **Truncation** -- oversized results are spilled to disk and replaced with a
    pointer plus head and tail excerpts, so one large read cannot swamp the
    context window. v1 truncated only the *display* and fed the full text to the
    model, which is backwards.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from ..errors import ToolError
from ..permissions import Request
from ..secrets import redact

JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class ToolResult:
    """What a tool hands back to the loop."""

    content: str
    is_error: bool = False
    # Short line for the UI; the full content still goes to the model.
    display: str = ""
    # Paths the tool touched, for checkpoint commit and UI reporting.
    touched: tuple[Path, ...] = ()
    # Set when the content was spilled to disk rather than kept whole.
    spilled_to: Path | None = None
    duration_s: float = 0.0

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, is_error=True, display=message.splitlines()[0][:120])


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any]
    # Argument names the permission engine reads to build its Request.
    path_arg: str | None = None
    command_arg: str | None = None
    # Mutating tools are checkpointed. Extra paths a tool will touch beyond
    # `path_arg` are reported by the tool itself via `extra_paths`.
    mutates: bool = False
    # Tools excluded from the model's schema list but callable internally.
    internal: bool = False
    # Rough hint for the UI about how long this may take.
    slow: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolContext:
    """Everything a tool needs, assembled once per session."""

    def __init__(
        self,
        *,
        workspace: Path,
        config: Any,
        approver: Any,
        checkpoints: Any,
        ui: Any = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.config = config
        self.approver = approver
        self.checkpoints = checkpoints
        self.ui = ui
        self.step = 0
        # Files the model has read this session. Used to refuse blind overwrites
        # of files it has never looked at, which is the single most common way an
        # autonomous agent destroys work.
        self.read_files: set[str] = set()
        # Scratch space shared between tools, e.g. the warm MATLAB engine.
        self.state: dict[str, Any] = {}

    def resolve(self, raw_path: str) -> tuple[Path, str, bool]:
        """Resolve a user- or model-supplied path against the workspace.

        Returns `(absolute, workspace_relative_posix, escaped)`. Resolution
        happens before the containment check so that symlinks, `..`, and Windows
        short names cannot step outside the root.
        """
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        try:
            relative = resolved.relative_to(self.workspace).as_posix()
            return resolved, relative, False
        except ValueError:
            return resolved, resolved.as_posix(), True

    def spill_dir(self) -> Path:
        path = self.workspace / ".deppseek" / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    name: str | None = None,
    path_arg: str | None = None,
    command_arg: str | None = None,
    mutates: bool = False,
    internal: bool = False,
    slow: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a tool, deriving its schema from its signature.

    The docstring's first paragraph becomes the tool description the model sees,
    so it must be written for the model, not for a maintainer. Per-parameter
    descriptions come from a `Args:` block.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        description, arg_docs = _split_docstring(func.__doc__ or "")
        parameters = _build_schema(func, arg_docs)
        _REGISTRY[tool_name] = ToolSpec(
            name=tool_name,
            description=description,
            func=func,
            parameters=parameters,
            path_arg=path_arg,
            command_arg=command_arg,
            mutates=mutates,
            internal=internal,
            slow=slow,
        )
        return func

    return decorator


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into a description and per-argument descriptions."""
    lines = inspect.cleandoc(doc).splitlines()
    description_lines: list[str] = []
    arg_docs: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args:
            match = re.match(r"^(\w+)\s*:\s*(.*)$", stripped)
            if match:
                current = match.group(1)
                arg_docs[current] = match.group(2).strip()
            elif current and stripped:
                arg_docs[current] += " " + stripped
            elif not stripped:
                current = None
        else:
            description_lines.append(line)

    return "\n".join(description_lines).strip(), arg_docs


def _build_schema(func: Callable[..., Any], arg_docs: dict[str, str]) -> dict[str, Any]:
    """Generate a JSON Schema object from type hints and defaults."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - a bad annotation must not break registration
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param_name in {"ctx", "self"}:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        annotation = hints.get(param_name, str)
        schema = _type_to_schema(annotation)
        if param_name in arg_docs:
            schema["description"] = arg_docs[param_name]
        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                schema["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        # Rejecting unknown keys makes a hallucinated argument a clear schema
        # error the model can correct, rather than a silent TypeError.
        "additionalProperties": False,
    }


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)

    if origin is Literal:
        options = list(get_args(annotation))
        return {"type": _json_type_of(type(options[0])), "enum": options}

    if origin in (list, Sequence):
        args = get_args(annotation)
        item = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    # Optional[X] / X | None: describe X, since None is expressed by omission.
    if origin is not None and type(None) in get_args(annotation):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _type_to_schema(inner[0])

    return {"type": JSON_TYPES.get(annotation, "string")}


def _json_type_of(python_type: type) -> str:
    return JSON_TYPES.get(python_type, "string")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@dataclass
class Toolbox:
    """The set of tools available to one session."""

    ctx: ToolContext
    enabled: dict[str, ToolSpec] = field(default_factory=dict)
    # Tools contributed at runtime, e.g. from MCP servers.
    dynamic: dict[str, ToolSpec] = field(default_factory=dict)
    _specs_cache: dict[str, ToolSpec] | None = field(default=None, repr=False)
    _schemas_cache: list[dict[str, Any]] | None = field(default=None, repr=False)

    @classmethod
    def build(cls, ctx: ToolContext, *, exclude: Sequence[str] = ()) -> Toolbox:
        enabled = {
            name: spec
            for name, spec in _REGISTRY.items()
            if name not in exclude and not spec.internal
        }
        return cls(ctx=ctx, enabled=enabled)

    def register_dynamic(self, spec: ToolSpec) -> None:
        self.dynamic[spec.name] = spec
        self._invalidate()

    def _invalidate(self) -> None:
        self._specs_cache = None
        self._schemas_cache = None

    def all_specs(self) -> dict[str, ToolSpec]:
        """The merged tool table.

        Cached because it is rebuilt on every tool call and every schema render,
        and only changes when an MCP server contributes tools at startup.
        """
        if self._specs_cache is None:
            self._specs_cache = {**self.enabled, **self.dynamic}
        return self._specs_cache

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the API request.

        Sorted by name so the serialised list is byte-identical between steps.
        The provider's prefix cache keys on exact bytes, and an unstable ordering
        would miss the cache on every request, multiplying input cost by roughly
        50x on the cached portion.
        """
        if self._schemas_cache is None:
            self._schemas_cache = [
                spec.schema() for _, spec in sorted(self.all_specs().items())
            ]
        # Returned as the same list object every step, so the serialised bytes
        # are identical and the estimator's tool memo can hit.
        return self._schemas_cache

    # ------------------------------------------------------------------
    def execute(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        spec = self.all_specs().get(name)
        if spec is None:
            available = ", ".join(sorted(self.all_specs()))
            return ToolResult.error(
                f"Unknown tool {name!r}. Available tools: {available}"
            )

        try:
            arguments = _parse_arguments(raw_arguments)
        except ToolError as exc:
            return ToolResult.error(str(exc))

        try:
            _validate_arguments(spec, arguments)
        except ToolError as exc:
            return ToolResult.error(str(exc))

        request = self._build_request(spec, arguments)
        allowed, reason = self.ctx.approver.check(request)
        if not allowed:
            return ToolResult.error(reason)

        checkpoint = None
        if spec.mutates and request.path is not None:
            targets = [self.ctx.workspace / request.path]
            targets += [Path(p) for p in arguments.get("_extra_paths", [])]
            checkpoint = self.ctx.checkpoints.snapshot(
                targets,
                tool=name,
                description=request.summary or str(arguments)[:80],
                step=self.ctx.step,
            )

        try:
            result = spec.func(self.ctx, **arguments)
        except ToolError as exc:
            result = ToolResult.error(f"{name}: {exc}")
        except PermissionError as exc:
            result = ToolResult.error(f"{name}: permission error from the OS: {exc}")
        except FileNotFoundError as exc:
            result = ToolResult.error(f"{name}: not found: {exc}")
        except TypeError as exc:
            result = ToolResult.error(
                f"{name}: argument error: {exc}. Expected parameters: "
                f"{', '.join(spec.parameters['properties'])}"
            )
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
            result = ToolResult.error(f"{name}: unexpected {type(exc).__name__}: {exc}")

        if isinstance(result, str):
            result = ToolResult(content=result)

        if checkpoint is not None:
            self.ctx.checkpoints.commit(checkpoint)

        # Record what changed so git_commit can stage precisely these paths.
        if spec.mutates and not result.is_error:
            modified: set[str] = self.ctx.state.setdefault("modified_paths", set())
            for touched in result.touched or ():
                try:
                    modified.add(touched.resolve().relative_to(self.ctx.workspace).as_posix())
                except ValueError:
                    continue
            if not result.touched and request.path:
                modified.add(request.path)

        result.duration_s = time.monotonic() - started
        return self._post_process(result)

    def _build_request(self, spec: ToolSpec, arguments: dict[str, Any]) -> Request:
        path_value = arguments.get(spec.path_arg) if spec.path_arg else None
        command_value = arguments.get(spec.command_arg) if spec.command_arg else None

        relative: str | None = None
        outside = False
        if path_value:
            _, relative, outside = self.ctx.resolve(str(path_value))

        summary_bits = [f"{spec.name}"]
        if relative:
            summary_bits.append(relative)
        return Request(
            tool=spec.name,
            path=relative,
            command=str(command_value) if command_value else None,
            outside_workspace=outside,
            summary=" ".join(summary_bits),
        )

    def _post_process(self, result: ToolResult) -> ToolResult:
        """Redact, then bound the size of what reaches the conversation."""
        report = redact(result.content)
        content = report.text
        if report.redacted:
            content += "\n\n" + report.describe()

        limit = self.ctx.config.context.max_tool_result_chars
        if len(content) > limit:
            spill = self.ctx.spill_dir() / f"tool-{int(time.time() * 1000)}.txt"
            try:
                spill.write_text(content, encoding="utf-8")
                pointer = f"full output saved to {spill.relative_to(self.ctx.workspace).as_posix()}"
            except OSError:
                pointer = "full output could not be saved"
            head = content[: limit // 2]
            tail = content[-limit // 2 :]
            omitted = len(content) - len(head) - len(tail)
            content = (
                f"{head}\n\n"
                f"... [{omitted:,} characters omitted; {pointer}. "
                f"Read specific line ranges from that file, or narrow the query, "
                f"rather than asking for the whole thing again.] ...\n\n"
                f"{tail}"
            )
            result.spilled_to = spill

        result.content = content
        if not result.display:
            first = content.splitlines()[0] if content else ""
            result.display = first[:120]
        return result


def _parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Models occasionally emit trailing commas or single quotes under load.
        # One repair attempt beats burning a whole step on a re-ask.
        repaired = _repair_json(text)
        if repaired is not None:
            parsed = repaired
        else:
            raise ToolError(
                f"Arguments were not valid JSON ({exc}). Re-send this tool call "
                f"with a valid JSON object."
            ) from exc
    if not isinstance(parsed, dict):
        raise ToolError("Tool arguments must be a JSON object, not a bare value.")
    return parsed


def _repair_json(text: str) -> dict[str, Any] | None:
    candidate = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
    """Check required keys and reject unknown ones with an actionable message."""
    properties = spec.parameters["properties"]
    required = spec.parameters.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolError(
            f"{spec.name} is missing required argument(s): {', '.join(missing)}. "
            f"Expected: {', '.join(properties)}"
        )

    unknown = [k for k in arguments if k not in properties and not k.startswith("_")]
    if unknown:
        raise ToolError(
            f"{spec.name} got unknown argument(s): {', '.join(unknown)}. "
            f"Valid arguments: {', '.join(properties)}"
        )


def registry() -> dict[str, ToolSpec]:
    return dict(_REGISTRY)
