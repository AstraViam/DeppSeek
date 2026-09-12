"""Colour and glyph definitions.

Two constraints shape this. Windows Terminal handles truecolor and Unicode well;
the legacy conhost that still opens for `powershell.exe` from a shortcut handles
neither reliably. So every glyph has an ASCII fallback, and colours are chosen to
stay legible on both the default blue conhost background and Windows Terminal's
dark grey.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Semantic colours. Named by role rather than hue so a future theme swap does not
# require touching call sites.
THEME_COLOURS = {
    "accent": "bright_cyan",
    "accent_dim": "cyan",
    "user": "bright_white",
    "assistant": "white",
    "reasoning": "grey54",
    "tool": "bright_blue",
    "tool_ok": "green",
    "tool_error": "bright_red",
    "warn": "yellow",
    "error": "bright_red",
    "muted": "grey50",
    "cost": "magenta",
    "added": "green",
    "removed": "red",
    "rule": "grey35",
}


@dataclass(frozen=True)
class Glyphs:
    bullet: str
    arrow: str
    check: str
    cross: str
    think: str
    tool: str
    spinner: str
    bar_full: str
    bar_empty: str
    corner: str

    @classmethod
    def unicode(cls) -> Glyphs:
        return cls(
            bullet="•", arrow="→", check="✓", cross="✗", think="◈", tool="▸",
            spinner="dots", bar_full="█", bar_empty="░", corner="└",
        )

    @classmethod
    def ascii(cls) -> Glyphs:
        return cls(
            bullet="*", arrow="->", check="ok", cross="x", think="~", tool=">",
            spinner="line", bar_full="#", bar_empty=".", corner="`",
        )


def supports_unicode() -> bool:
    """Decide whether box-drawing and symbol glyphs will render.

    Windows Terminal sets WT_SESSION. Legacy conhost does not, and its default
    code page mangles anything outside the active OEM page, so ASCII is used
    there even though the font might technically have the glyph.
    """
    if os.name != "nt":
        return True
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
        return True
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


def glyphs_for(prefer_unicode: bool = True) -> Glyphs:
    return Glyphs.unicode() if (prefer_unicode and supports_unicode()) else Glyphs.ascii()


def build_theme():
    """Build a rich Theme, or None when rich is unavailable."""
    try:
        from rich.theme import Theme
    except ImportError:
        return None
    return Theme({f"ds.{name}": colour for name, colour in THEME_COLOURS.items()})
