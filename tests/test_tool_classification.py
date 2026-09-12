"""Every registered tool must be classified by the permission presets.

An unclassified tool falls through to the ASK fallback. Under the autonomous
tier that turns a silent background operation into a prompt the user did not
expect, and under a non-interactive run it becomes an outright denial. This test
fails whenever a tool is added without deciding what it is.
"""

import pytest

from deppseek.permissions.rules import (
    DESTRUCTIVE_TOOLS,
    EXEC_TOOLS,
    GIT_REMOTE_TOOLS,
    GIT_WRITE_TOOLS,
    NETWORK_READ_TOOLS,
    NETWORK_UPLOAD_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    Decision,
    PermissionEngine,
    Request,
)
from deppseek.tools import registry

ALL_CLASSIFIED = set(
    READ_TOOLS
    + WRITE_TOOLS
    + DESTRUCTIVE_TOOLS
    + EXEC_TOOLS
    + NETWORK_READ_TOOLS
    + NETWORK_UPLOAD_TOOLS
    + GIT_WRITE_TOOLS
    + GIT_REMOTE_TOOLS
)


@pytest.mark.parametrize("tool_name", sorted(registry()))
def test_every_tool_is_classified(tool_name):
    assert tool_name in ALL_CLASSIFIED, (
        f"{tool_name} is registered but not in any permission group, so it falls "
        f"through to the ASK fallback. Add it to the right group in "
        f"deppseek/permissions/rules.py."
    )


@pytest.mark.parametrize("tool_name", sorted(registry()))
def test_no_tool_hits_the_unclassified_fallback(tool_name):
    engine = PermissionEngine(autonomy="autonomous")
    verdict = engine.evaluate(Request(tool_name, path="scratch/example.txt"))
    assert verdict.reason != "unclassified tool", f"{tool_name} is unclassified"


def test_mutating_tools_are_all_in_a_write_or_destructive_group():
    """A mutating tool outside those groups would not be checkpointed correctly
    by the presets that treat writes and deletes differently."""
    mutating = {name for name, spec in registry().items() if spec.mutates}
    classified = set(WRITE_TOOLS + DESTRUCTIVE_TOOLS)
    assert mutating <= classified, f"unclassified mutating tools: {mutating - classified}"


def test_read_tools_never_mutate():
    """Anything in READ_TOOLS runs unattended at every tier above readonly, so a
    mutating tool listed there would bypass checkpointing entirely."""
    specs = registry()
    for name in READ_TOOLS:
        spec = specs.get(name)
        if spec is not None:
            assert not spec.mutates, f"{name} is in READ_TOOLS but declares mutates=True"
