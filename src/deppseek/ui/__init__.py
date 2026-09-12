"""Terminal interfaces: an inline transcript by default, a dashboard behind --tui."""

from .inline import InlineUI
from .input import build_session, read_input
from .theme import glyphs_for

__all__ = ["InlineUI", "build_session", "glyphs_for", "read_input"]
