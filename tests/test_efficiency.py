"""Performance and cost properties.

These are asserted rather than measured where possible: a wall-clock threshold
makes a flaky test on shared CI, but complexity, allocation behaviour, and
byte-level prefix stability can be checked exactly.
"""

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from deppseek.session.context import ConversationBuffer
from deppseek.session.tokens import TokenEstimator
from deppseek.tools.fs import looks_binary, read_line_range, read_text

# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def build_conversation(steps: int) -> ConversationBuffer:
    buffer = ConversationBuffer("system " * 200)
    for index in range(steps):
        buffer.append_user(f"step {index}")
        buffer.append({
            "role": "assistant", "content": None,
            "reasoning_content": "thinking " * 200,
            "tool_calls": [{"id": f"c{index}", "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"}}],
        })
        buffer.append_tool_result(f"c{index}", "x = 1\n" * 500)
    return buffer


def test_repeated_estimation_does_not_rescan_the_whole_history():
    """Regression: estimating the conversation once per step re-scanned every
    earlier message, which is quadratic in session length. Measured at 52 ms per
    call after 160 steps, called twice per step, for 16.6 s of pure accounting.

    Asserted by counting the work done rather than the time taken, so the test
    does not depend on how loaded the machine is."""
    buffer = build_conversation(40)
    estimator = buffer.estimator

    calls = {"n": 0}
    original = estimator._estimate_message_uncached

    def counting(message):
        calls["n"] += 1
        return original(message)

    estimator._estimate_message_uncached = counting

    buffer.estimate()
    first_pass = calls["n"]
    assert first_pass == len(buffer.messages) + 1  # +1 for the system message

    calls["n"] = 0
    for _ in range(10):
        buffer.estimate()
    assert calls["n"] == 0, "repeat estimation re-scanned messages it had already seen"


def test_only_the_new_message_is_scanned_as_the_conversation_grows():
    buffer = build_conversation(10)
    estimator = buffer.estimator
    buffer.estimate()

    calls = {"n": 0}
    original = estimator._estimate_message_uncached

    def counting(message):
        calls["n"] += 1
        return original(message)

    estimator._estimate_message_uncached = counting
    buffer.append_user("one more")
    buffer.estimate()
    assert calls["n"] == 1


def test_memo_is_invalidated_when_a_message_is_mutated():
    estimator = TokenEstimator()
    message = {"role": "user", "content": "short"}
    small = estimator.estimate_message(message)

    message["content"] = "much longer content " * 100
    large = estimator.estimate_message(message)
    assert large > small * 5


def test_memo_is_invalidated_by_calibration():
    estimator = TokenEstimator()
    message = {"role": "user", "content": "some content here " * 50}
    before = estimator.estimate_message(message)

    estimator.calibrate(estimated=1000, actual=4000)
    after = estimator.estimate_message(message)
    assert after != before, "calibration did not change a previously cached estimate"


def test_memo_is_bounded():
    estimator = TokenEstimator()
    for index in range(6000):
        estimator.estimate_message({"role": "user", "content": f"message {index}"})
    assert len(estimator._memo) <= 4096


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def test_binary_sniff_closes_its_file_handle():
    """Regression: the sniff opened a handle and left it to the garbage
    collector. It runs once per candidate in the Python search fallback, so a
    search over a large tree opened thousands; on Windows an open handle also
    blocks renaming or deleting the file, which this tool does routinely."""
    path = Path(tempfile.mkdtemp()) / "x.txt"
    path.write_text("hello\n" * 100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        looks_binary(path)
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


@pytest.mark.parametrize(
    "content",
    [
        "one\ntwo\nthree\n",
        "one\ntwo\nthree",       # no trailing newline
        "crlf\r\nlines\r\n",     # Windows line endings
        "\n\n\n",                # blank lines only
        "",                      # empty file
        "single line no newline",
    ],
)
@pytest.mark.parametrize("start,count", [(1, 1), (1, 10), (2, 2), (3, 1), (5, 5)])
def test_windowed_read_matches_the_straightforward_implementation(content, start, count, tmp_path):
    path = tmp_path / "f.txt"
    path.write_text(content, newline="")

    lines = read_text(path).splitlines()
    expected = (lines[start - 1 : start - 1 + count], len(lines))
    assert read_line_range(path, start, count) == expected


def test_windowed_read_does_not_materialise_the_whole_file(tmp_path):
    """A 400-line read of a large result file should not allocate one string per
    line for the entire file."""
    import tracemalloc

    path = tmp_path / "results.csv"
    path.write_text("".join(f"{i},0.01,turbulent\n" for i in range(200_000)))
    size = path.stat().st_size

    tracemalloc.start()
    selected, total = read_line_range(path, 1000, 400)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(selected) == 400
    assert total == 200_000
    # The file itself has to be read; what must not happen is allocating the
    # decoded text plus a string object per line on top of it.
    assert peak < size * 2.5, f"peak {peak:,} bytes for a {size:,} byte file"


# ---------------------------------------------------------------------------
# Prompt-cache prefix stability
# ---------------------------------------------------------------------------

def test_tool_schemas_are_byte_stable_across_rebuilds(tmp_path):
    from deppseek.checkpoint import CheckpointStore
    from deppseek.config import Config
    from deppseek.permissions import PermissionEngine
    from deppseek.permissions.prompt import Approval, Approver
    from deppseek.tools import Toolbox, ToolContext

    config = Config(workspace=tmp_path)
    ctx = ToolContext(
        workspace=tmp_path, config=config,
        approver=Approver(PermissionEngine(), ask_fn=lambda r, v: Approval(True)),
        checkpoints=CheckpointStore(tmp_path / ".deppseek", tmp_path, "t"),
    )
    box = Toolbox.build(ctx)
    first = json.dumps(box.schemas())
    rebuilt = json.dumps(Toolbox.build(ctx).schemas())
    assert first == rebuilt


def test_system_prompt_contains_nothing_that_varies_between_steps():
    """The system prompt is the first thing in every request. Anything in it
    that changes per step, a timestamp or a step counter, moves the cache
    boundary to position zero and turns every cached token into a miss."""
    from deppseek.prompts import build_system_prompt

    first = build_system_prompt(Path("/proj"), autonomy="autonomous")
    second = build_system_prompt(Path("/proj"), autonomy="autonomous")
    assert first == second

    import re

    assert not re.search(r"\d{4}-\d{2}-\d{2}", first), "prompt contains a date"
    assert not re.search(r"\d{2}:\d{2}:\d{2}", first), "prompt contains a time"
