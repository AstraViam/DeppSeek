"""Secret containment.

Two separate jobs, often conflated:

1. **Access control** -- refuse to *read* files that exist to hold credentials.
   A path-glob denylist handles this, enforced in the permission engine.
2. **Egress redaction** -- scrub credential-shaped strings out of every tool
   result *before* it is appended to the conversation. This is the backstop:
   a key pasted into a source file, printed by a script, or echoed in an error
   message never reaches the API even though the file itself was legitimate to
   read.

v1 had neither. It would happily read `.env` and post it to DeepSeek.

Redaction is deliberately lossy and irreversible. It preserves a short prefix so
a human can still tell *which* key was involved when debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern captures the secret in group "secret" so the replacement can keep
# a recognisable prefix. Ordered most-specific first: a PEM block should be
# matched as a block, not shredded by the generic assigned-secret rule.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private-key-block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
            r"(?P<secret>.*?)"
            r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("deepseek-key", re.compile(r"(?P<secret>sk-[0-9a-f]{32}(?![0-9a-zA-Z]))")),
    ("openai-key", re.compile(r"(?P<secret>sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,})")),
    ("github-token", re.compile(r"(?P<secret>gh[pousr]_[A-Za-z0-9]{36,})")),
    ("github-pat", re.compile(r"(?P<secret>github_pat_[A-Za-z0-9_]{50,})")),
    ("aws-access-key", re.compile(r"(?P<secret>(?:AKIA|ASIA)[0-9A-Z]{16})")),
    ("google-api-key", re.compile(r"(?P<secret>AIza[0-9A-Za-z_\-]{35})")),
    ("slack-token", re.compile(r"(?P<secret>xox[abprs]-[0-9A-Za-z\-]{10,})")),
    ("anthropic-key", re.compile(r"(?P<secret>sk-ant-api\d{2}-[A-Za-z0-9_\-]{80,})")),
    ("jwt", re.compile(r"(?P<secret>eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})")),
    (
        "bearer-header",
        re.compile(r"(?i:bearer)\s+(?P<secret>[A-Za-z0-9_\-\.=]{24,})"),
    ),
    (
        "connection-string-password",
        re.compile(
            r"(?i:(?:password|passwd|pwd)\s*=\s*)(?P<secret>[^\s;'\"]{6,})"
        ),
    ),
    (
        "url-credentials",
        re.compile(r"://[^\s:/@]+:(?P<secret>[^\s@/]{4,})@"),
    ),
    (
        "assigned-secret",
        # KEY = "value" where the key name advertises itself as a secret.
        re.compile(
            r"(?i:(?:api[_\-]?key|apikey|secret|token|passwd|password|access[_\-]?key"
            r"|private[_\-]?key|client[_\-]?secret))"
            r"\s*[:=]\s*['\"]?(?P<secret>[A-Za-z0-9_\-\.+/=]{12,})['\"]?"
        ),
    ),
]

# Strings that match a secret pattern but are obviously placeholders. Redacting
# these is harmless but noisy, and it hides the fact that a config is unfilled.
# A repeated single character for the whole string ("xxxxxxxx", "00000000") is a
# masked or stubbed value, not a live credential.
_PLACEHOLDER_FILLER = re.compile(r"^(.)\1{5,}$")

# Leading words that mark a value as unfilled template text. These must be
# checked as whole words so that a real secret merely *starting* with one of
# these letters is not waved through -- the bug that previously let every
# Slack token (xoxb-...) past, because "x+" matched its first character.
_PLACEHOLDER_PREFIX = re.compile(
    r"^(?:your|my|the|insert|replace|changeme|change[_\-]me|placeholder|example|sample"
    r"|dummy|fake|test|todo|tbd|xxx+|none|null|true|false|undefined|redacted)"
    r"(?:[_\-\s]|$)",
    re.IGNORECASE,
)

# Template interpolation of any common flavour: ${VAR}, {{var}}, <VAR>, %VAR%.
_PLACEHOLDER_TEMPLATE = re.compile(r"^(?:\$\{|\{\{|<|%[A-Za-z_])")


def _is_placeholder(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    return bool(
        _PLACEHOLDER_FILLER.match(value)
        or _PLACEHOLDER_PREFIX.match(value)
        or _PLACEHOLDER_TEMPLATE.match(value)
    )


@dataclass
class RedactionReport:
    text: str
    hits: dict[str, int]

    @property
    def redacted(self) -> bool:
        return bool(self.hits)

    def describe(self) -> str:
        if not self.hits:
            return ""
        parts = ", ".join(f"{name} x{count}" for name, count in sorted(self.hits.items()))
        return f"[redacted before leaving this machine: {parts}]"


def redact(text: str, *, min_length: int = 12) -> RedactionReport:
    """Replace credential-shaped substrings with an irreversible marker."""
    if not text:
        return RedactionReport(text, {})

    hits: dict[str, int] = {}

    def make_sub(label: str):
        def _sub(match: re.Match[str]) -> str:
            secret = match.group("secret")
            if len(secret) < min_length or _is_placeholder(secret):
                return match.group(0)
            hits[label] = hits.get(label, 0) + 1
            keep = secret[:4] if len(secret) > 12 else ""
            marker = f"{keep}...[REDACTED:{label}:{len(secret)}ch]"
            # Rebuild the full match with only the secret group swapped, so
            # surrounding syntax (quotes, "Bearer ", "://user:") survives.
            start, end = match.span("secret")
            whole_start = match.start()
            return match.group(0)[: start - whole_start] + marker + match.group(0)[end - whole_start :]

        return _sub

    out = text
    for label, pattern in SECRET_PATTERNS:
        out = pattern.sub(make_sub(label), out)

    return RedactionReport(out, hits)


def scrub(text: str) -> str:
    """Redact and return just the text. Convenience for call sites that do not
    need to know whether anything was hit."""
    return redact(text).text
