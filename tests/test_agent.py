"""Agent loop behaviour, driven by a scripted provider so no network is needed."""

import pytest

from deppseek.agent import AgentLoop
from deppseek.checkpoint import CheckpointStore
from deppseek.config import BudgetConfig, Config
from deppseek.errors import ProviderError
from deppseek.permissions import PermissionEngine
from deppseek.permissions.prompt import Approval, Approver
from deppseek.providers.base import Completion, ToolCall
from deppseek.providers.pricing import Usage
from deppseek.session.context import ConversationBuffer
from deppseek.tools import Toolbox, ToolContext


class ScriptedProvider:
    """Returns a fixed list of completions, one per call."""

    def __init__(self, completions, model="deepseek-flash"):
        self.completions = list(completions)
        self.model = model
        self.usage = Usage()
        self.thinking = True
        self.calls = []

    def complete(self, messages, tools=None, *, sink=None, stream=True):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.completions:
            return Completion(content="done")
        result = self.completions.pop(0)
        if isinstance(result, Exception):
            raise result
        self.usage.estimated_cost_usd += 0.01
        return result


@pytest.fixture
def harness(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config(workspace=workspace, budget=BudgetConfig(max_steps=10, max_cost_usd=None))
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    approver = Approver(engine, ask_fn=lambda r, v: Approval(approved=True))
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=approver,
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "test"),
    )
    toolbox = Toolbox.build(ctx)
    buffer = ConversationBuffer("system", soft_limit=10**9, hard_limit=10**9)
    return toolbox, buffer, config, workspace


def build_loop(harness, completions):
    toolbox, buffer, config, workspace = harness
    provider = ScriptedProvider(completions)
    loop = AgentLoop(provider=provider, toolbox=toolbox, buffer=buffer, config=config)
    return loop, provider, workspace


def test_single_turn_answer(harness):
    loop, _, _ = build_loop(harness, [Completion(content="The CFL number is 0.4.")])[:3]
    result = loop.run("What is the CFL number?")
    assert result.answer == "The CFL number is 0.4."
    assert result.stopped_reason == "completed"


def test_tool_call_then_answer(harness):
    _, _, _, workspace = harness
    (workspace / "solver.py").write_text("dt = 0.01\n")
    completions = [
        Completion(
            tool_calls=[ToolCall(id="c1", name="read_file", arguments='{"path": "solver.py"}')]
        ),
        Completion(content="dt is 0.01 s."),
    ]
    loop, provider, _ = build_loop(harness, completions)
    result = loop.run("Read the solver")

    assert result.answer == "dt is 0.01 s."
    assert result.tool_call_count == 1
    # The tool result must have been fed back before the second request.
    second_request = provider.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_request)


def test_tool_pairing_invariant_holds(harness):
    _, buffer, _, workspace = harness
    (workspace / "a.py").write_text("x\n")
    completions = [
        Completion(
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments='{"path": "a.py"}'),
                ToolCall(id="c2", name="git_status", arguments="{}"),
            ]
        ),
        Completion(content="done"),
    ]
    loop, _, _ = build_loop(harness, completions)
    loop.run("look around")
    assert buffer.validate() == []


def test_independent_reads_run_in_parallel(harness):
    _, _, _, workspace = harness
    for name in ("a.py", "b.py", "c.py"):
        (workspace / name).write_text("x\n")

    events = []
    completions = [
        Completion(
            tool_calls=[
                ToolCall(id=f"c{i}", name="read_file", arguments=f'{{"path": "{name}"}}')
                for i, name in enumerate(("a.py", "b.py", "c.py"))
            ]
        ),
        Completion(content="read all three"),
    ]
    loop, _, _ = build_loop(harness, completions)
    loop.on_event = lambda name, payload: events.append((name, payload))
    loop.run("read them")

    assert any(name == "parallel_tools" for name, _ in events)


def test_mutating_tools_are_serialised(harness):
    """Two concurrent edits to one file is a race with no upside."""
    _, _, _, _workspace = harness
    events = []
    completions = [
        Completion(
            tool_calls=[
                ToolCall(id="c1", name="write_file", arguments='{"path": "x.py", "content": "1"}'),
                ToolCall(id="c2", name="write_file", arguments='{"path": "y.py", "content": "2"}'),
            ]
        ),
        Completion(content="wrote"),
    ]
    loop, _, _ = build_loop(harness, completions)
    loop.on_event = lambda name, payload: events.append((name, payload))
    loop.run("write two files")
    assert not any(name == "parallel_tools" for name, _ in events)


def test_step_limit_stops_with_an_explanation(harness):
    completions = [
        Completion(tool_calls=[ToolCall(id=f"c{i}", name="git_status", arguments="{}")])
        for i in range(20)
    ]
    toolbox, buffer, config, _workspace = harness
    provider = ScriptedProvider(completions)
    loop = AgentLoop(provider=provider, toolbox=toolbox, buffer=buffer, config=config)
    result = loop.run("loop forever", max_steps=3)
    assert "step limit" in result.stopped_reason.lower() or "3 steps" in result.stopped_reason


def test_cost_ceiling_stops_the_run(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config(
        workspace=workspace, budget=BudgetConfig(max_steps=50, max_cost_usd=0.025)
    )
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=Approver(engine, ask_fn=lambda r, v: Approval(approved=True)),
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "t"),
    )
    completions = [
        Completion(tool_calls=[ToolCall(id=f"c{i}", name="git_status", arguments="{}")])
        for i in range(20)
    ]
    provider = ScriptedProvider(completions)
    loop = AgentLoop(
        provider=provider,
        toolbox=Toolbox.build(ctx),
        buffer=ConversationBuffer("s", soft_limit=10**9, hard_limit=10**9),
        config=config,
    )
    result = loop.run("spend money")
    assert "cost ceiling" in result.stopped_reason


def test_provider_error_stops_cleanly(harness):
    loop, _, _ = build_loop(harness, [ProviderError("HTTP 401: bad key")])
    result = loop.run("hello")
    assert "401" in result.stopped_reason
    assert not result.interrupted


def test_failed_tool_is_reported_to_the_model_not_fatal(harness):
    completions = [
        Completion(
            tool_calls=[ToolCall(id="c1", name="read_file", arguments='{"path": "nope.py"}')]
        ),
        Completion(content="that file does not exist"),
    ]
    loop, provider, _ = build_loop(harness, completions)
    result = loop.run("read a missing file")

    assert result.answer == "that file does not exist"
    tool_messages = [
        m for m in provider.calls[1]["messages"] if m.get("role") == "tool"
    ]
    assert "does not exist" in tool_messages[0]["content"]


def test_reasoning_content_is_carried_back(harness):
    """DeepSeek thinking mode wants reasoning_content preserved on assistant
    messages across tool turns; dropping it degrades tool-calling quality."""
    completions = [
        Completion(
            reasoning="I should check git first.",
            tool_calls=[ToolCall(id="c1", name="git_status", arguments="{}")],
        ),
        Completion(content="clean"),
    ]
    loop, provider, _ = build_loop(harness, completions)
    loop.run("status?")

    assistant = [m for m in provider.calls[1]["messages"] if m.get("role") == "assistant"]
    assert assistant[0].get("reasoning_content") == "I should check git first."


def test_compaction_triggers_and_preserves_pairing(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config(workspace=workspace, budget=BudgetConfig(max_steps=30, max_cost_usd=None))
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=Approver(engine, ask_fn=lambda r, v: Approval(approved=True)),
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "t"),
    )
    buffer = ConversationBuffer("s", soft_limit=400, hard_limit=10**9, keep_recent_turns=2)
    completions = [
        Completion(
            content="x" * 600,
            tool_calls=[ToolCall(id=f"c{i}", name="git_status", arguments="{}")],
        )
        for i in range(8)
    ] + [Completion(content="finished")]

    provider = ScriptedProvider(completions)
    events = []
    loop = AgentLoop(
        provider=provider,
        toolbox=Toolbox.build(ctx),
        buffer=buffer,
        config=config,
        on_event=lambda name, payload: events.append(name),
    )
    loop.run("long task")

    assert "compacted" in events
    assert buffer.validate() == []
