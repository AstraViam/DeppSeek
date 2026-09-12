"""Exception hierarchy.

Every failure mode the agent can hit gets a distinct type so the loop can decide
whether to retry, surface to the model as a tool result, or abort the run.
"""

from __future__ import annotations


class DeppSeekError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(DeppSeekError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ProviderError(DeppSeekError):
    """The model provider returned an error or unusable response."""


class RetryableProviderError(ProviderError):
    """A provider error that is worth retrying (rate limit, 5xx, transport)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceeded(DeppSeekError):
    """The run hit its configured cost or token ceiling."""


class PermissionDenied(DeppSeekError):
    """A permission rule or the user refused an action.

    This is recoverable: it is reported back to the model as a tool result so it
    can choose a different approach, rather than aborting the run.
    """


class PathEscapesWorkspace(PermissionDenied):
    """A path resolved outside the workspace root."""


class ToolError(DeppSeekError):
    """A tool failed in a way the model should see and reason about."""


class ToolNotFound(ToolError):
    """The model called a tool that is not registered."""
