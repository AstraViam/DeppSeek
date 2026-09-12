"""Terminal interfaces: an inline transcript by default, a dashboard behind --tui."""

from __future__ import annotations

from typing import Any

from .inline import InlineUI
from .theme import glyphs_for

__all__ = ["InlineUI", "build_session", "glyphs_for", "read_input"]

# `input` pulls in prompt_toolkit, which costs about 75 ms and is only needed
# once someone is actually going to type. Deferring it keeps --version,
# --doctor, and one-shot runs fast, and the cost lands where the user is already
# waiting at a prompt. PEP 562 module-level __getattr__ makes the deferral
# invisible to callers.
def __getattr__(name: str) -> Any:
    if name in ("build_session", "read_input"):
        from . import input as _input

        return getattr(_input, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
