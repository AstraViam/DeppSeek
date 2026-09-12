"""MCP client, exercised against a real subprocess server."""

import sys
from pathlib import Path

import pytest

from deppseek.config import McpServerConfig
from deppseek.mcp_client import McpError, McpManager, McpServer

FAKE_SERVER = str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")


@pytest.fixture
def server():
    srv = McpServer("fake", sys.executable, (FAKE_SERVER,))
    srv.start()
    yield srv
    srv.stop()


def test_handshake_and_tool_discovery(server):
    assert server.server_info.get("name") == "fake"
    assert {t.name for t in server.tools} == {"echo", "boom"}
    assert server.tools[0].qualified_name == "mcp__fake__echo"


def test_tool_call_round_trip(server):
    assert server.call("echo", {"text": "hello"}) == "echo: hello"


def test_server_reported_error_becomes_a_tool_error(server):
    from deppseek.errors import ToolError

    with pytest.raises(ToolError, match="it broke"):
        server.call("boom", {})


def test_unknown_tool_surfaces_the_jsonrpc_error(server):
    with pytest.raises(McpError, match="no such tool"):
        server.call("nope", {})


def test_missing_command_is_reported_not_raised_opaquely():
    srv = McpServer("ghost", "definitely-not-a-real-command-xyz")
    with pytest.raises(McpError, match="not found"):
        srv.start()


def test_manager_reports_failures_without_aborting_startup():
    manager = McpManager()
    messages = manager.start_all(
        [
            McpServerConfig(name="fake", command=sys.executable, args=(FAKE_SERVER,)),
            McpServerConfig(name="ghost", command="definitely-not-a-real-command-xyz"),
        ]
    )
    try:
        assert any("fake" in m and "2 tool(s)" in m for m in messages)
        assert "ghost" in manager.failures
        # The working server is still usable despite the other one failing.
        assert len(manager.all_tools()) == 2
    finally:
        manager.stop_all()


def test_disabled_servers_are_not_started():
    manager = McpManager()
    manager.start_all([McpServerConfig(name="off", command=sys.executable, enabled=False)])
    assert manager.servers == {}


def test_mcp_tools_are_gated_by_the_permission_engine():
    from deppseek.permissions import Decision, PermissionEngine, Request

    engine = PermissionEngine(autonomy="autonomous")
    verdict = engine.evaluate(Request("mcp__fake__echo"))
    assert verdict.decision is Decision.ASK
    assert "third-party MCP server" in verdict.reason


def test_a_project_config_can_trust_a_specific_server():
    from deppseek.permissions import Decision, PermissionEngine, Request, Rule

    engine = PermissionEngine(autonomy="autonomous")
    engine.user_rules = (
        Rule(Decision.ALLOW, tool="mcp__trusted__*", reason="vetted internally"),
    )
    assert engine.evaluate(Request("mcp__trusted__query")).decision is Decision.ALLOW
    assert engine.evaluate(Request("mcp__other__query")).decision is Decision.ASK
