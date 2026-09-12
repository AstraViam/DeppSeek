"""End-to-end streaming over real HTTP, against a fake DeepSeek endpoint.

This is the only test that exercises the server-sent-events path: fragmented
tool-call deltas, the usage-only final chunk, and the sink callbacks. A test
that stubs the provider cannot see any of it.
"""

import pytest

from deppseek.providers import DeepSeekProvider
from deppseek.providers.pricing import Usage
from fixtures.fake_deepseek_server import Handler, chunk, start


@pytest.fixture
def server(monkeypatch):
    # localhost must not be routed through an outbound proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv, url = start()
    Handler.script = []
    Handler.requests = []
    yield url
    srv.shutdown()


class RecordingSink:
    def __init__(self):
        self.reasoning, self.content, self.tools, self.done = [], [], [], 0

    def on_reasoning(self, delta):
        self.reasoning.append(delta)

    def on_content(self, delta):
        self.content.append(delta)

    def on_tool_call_start(self, name):
        self.tools.append(name)

    def on_done(self):
        self.done += 1


def make_provider(url, **kwargs):
    return DeepSeekProvider(
        api_key="sk-fake", model="deepseek-flash", base_url=url, usage=Usage(), **kwargs
    )


def test_prose_streams_through_the_sink(server):
    Handler.script = [[
        chunk({"role": "assistant"}),
        chunk({"content": "The CFL "}),
        chunk({"content": "number is "}),
        chunk({"content": "0.4."}),
        chunk({}, "stop"),
    ]]
    sink = RecordingSink()
    result = make_provider(server).complete([{"role": "user", "content": "?"}], sink=sink)

    assert result.content == "The CFL number is 0.4."
    assert "".join(sink.content) == "The CFL number is 0.4."
    assert sink.done == 1


def test_reasoning_streams_separately_from_content(server):
    Handler.script = [[
        chunk({"reasoning_content": "Check "}),
        chunk({"reasoning_content": "the mesh."}),
        chunk({"content": "Mesh is fine."}),
        chunk({}, "stop"),
    ]]
    sink = RecordingSink()
    result = make_provider(server).complete([{"role": "user", "content": "?"}], sink=sink)

    assert result.reasoning == "Check the mesh."
    assert result.content == "Mesh is fine."
    assert "".join(sink.reasoning) == "Check the mesh."


def test_fragmented_tool_call_arguments_are_reassembled(server):
    """Arguments arrive a few characters at a time and must be concatenated in
    index order, not treated as separate calls."""
    Handler.script = [[
        chunk({"tool_calls": [{"index": 0, "id": "call_1",
                               "function": {"name": "read_file", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'th": "sol'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ver.py"}'}}]}),
        chunk({}, "tool_calls"),
    ]]
    sink = RecordingSink()
    result = make_provider(server).complete([{"role": "user", "content": "?"}], sink=sink)

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "read_file"
    assert call.arguments == '{"path": "solver.py"}'
    # The UI is told the tool's name as soon as it is known, not at the end.
    assert sink.tools == ["read_file"]


def test_several_interleaved_tool_calls_stay_separate(server):
    Handler.script = [[
        chunk({"tool_calls": [{"index": 0, "id": "a",
                               "function": {"name": "read_file", "arguments": '{"path":'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "b",
                               "function": {"name": "git_status", "arguments": "{"}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "a.py"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]}),
        chunk({}, "tool_calls"),
    ]]
    result = make_provider(server).complete([{"role": "user", "content": "?"}])

    assert [c.name for c in result.tool_calls] == ["read_file", "git_status"]
    assert result.tool_calls[0].arguments == '{"path": "a.py"}'
    assert result.tool_calls[1].arguments == "{}"


def test_usage_from_the_final_chunk_is_recorded(server):
    """Streamed responses carry usage only in a trailing chunk with no choices.
    Missing it is how a streaming client silently reports zero cost."""
    usage_frame = {
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
        "model": "deepseek-flash", "choices": [],
        "usage": {
            "prompt_tokens": 10_000,
            "prompt_cache_hit_tokens": 9_000,
            "prompt_cache_miss_tokens": 1_000,
            "completion_tokens": 500,
            "completion_tokens_details": {"reasoning_tokens": 300},
        },
    }
    Handler.script = [[chunk({"content": "hi"}), chunk({}, "stop"), usage_frame]]

    provider = make_provider(server)
    provider.complete([{"role": "user", "content": "?"}])

    assert provider.usage.cache_hit_input_tokens == 9_000
    assert provider.usage.cache_miss_input_tokens == 1_000
    assert provider.usage.output_tokens == 500
    assert provider.usage.reasoning_tokens == 300
    assert provider.usage.estimated_cost_usd > 0
    # Cache-aware pricing must be far below the flat-rate figure v1 would report.
    assert provider.usage.estimated_cost_usd < 10_000 / 1e6 * 0.30 + 500 / 1e6 * 1.20


def test_request_carries_thinking_and_usage_options(server):
    Handler.script = [[chunk({"content": "ok"}, "stop")]]
    make_provider(server, reasoning_effort="max").complete([{"role": "user", "content": "?"}])

    sent = Handler.requests[-1]
    assert sent["stream"] is True
    assert sent["stream_options"]["include_usage"] is True
    assert sent["reasoning_effort"] == "max"
    assert sent["thinking"] == {"type": "enabled"}


def test_thinking_can_be_disabled(server):
    Handler.script = [[chunk({"content": "ok"}, "stop")]]
    make_provider(server, thinking=False).complete([{"role": "user", "content": "?"}])

    sent = Handler.requests[-1]
    assert sent["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in sent


def test_full_stack_read_edit_and_answer(server, tmp_path):
    """The whole thing, over real HTTP: the model streams a tool call, the tool
    runs under the permission engine with a checkpoint taken, the result is fed
    back, and the final answer streams in. This is the composition that unit
    tests of each layer cannot prove."""
    from deppseek.agent import AgentLoop
    from deppseek.checkpoint import CheckpointStore
    from deppseek.config import BudgetConfig, Config
    from deppseek.permissions import PermissionEngine
    from deppseek.permissions.prompt import Approval, Approver
    from deppseek.session.context import ConversationBuffer
    from deppseek.tools import Toolbox, ToolContext

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "solver.py").write_text("dt = 0.01\nnu = 1e-6\n")

    Handler.script = [
        # Step 1: read the file.
        [
            chunk({"reasoning_content": "Read it first."}),
            chunk({"tool_calls": [{"index": 0, "id": "c1",
                                   "function": {"name": "read_file", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "sol'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ver.py"}'}}]}),
            chunk({}, "tool_calls"),
        ],
        # Step 2: edit it.
        [
            chunk({"tool_calls": [{"index": 0, "id": "c2", "function": {
                "name": "edit_file",
                "arguments": '{"path": "solver.py", "old_text": "dt = 0.01", "new_text": "dt = 0.001"}',
            }}]}),
            chunk({}, "tool_calls"),
        ],
        # Step 3: answer.
        [chunk({"content": "Reduced the timestep to 0.001 s for CFL stability."}), chunk({}, "stop")],
    ]

    config = Config(
        workspace=workspace,
        base_url=server,
        budget=BudgetConfig(max_steps=10, max_cost_usd=None),
    )
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=Approver(engine, ask_fn=lambda r, v: Approval(approved=True)),
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "e2e"),
    )
    provider = make_provider(server)
    ctx.state["provider"] = provider

    buffer = ConversationBuffer("system", soft_limit=10**9, hard_limit=10**9)
    sink = RecordingSink()
    loop = AgentLoop(
        provider=provider,
        toolbox=Toolbox.build(ctx),
        buffer=buffer,
        config=config,
        sink=sink,
    )
    result = loop.run("lower the timestep")

    assert result.answer == "Reduced the timestep to 0.001 s for CFL stability."
    assert (workspace / "solver.py").read_text() == "dt = 0.001\nnu = 1e-6\n"
    assert result.tool_call_count == 2
    assert buffer.validate() == []
    assert sink.tools == ["read_file", "edit_file"]

    # The edit is undoable, which is what makes unattended writes acceptable.
    ctx.checkpoints.undo()
    assert (workspace / "solver.py").read_text() == "dt = 0.01\nnu = 1e-6\n"
