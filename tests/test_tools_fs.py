"""Filesystem tool behaviour, especially the guards that matter when writes are
unattended."""

import pytest

from deppseek.checkpoint import CheckpointStore
from deppseek.config import Config
from deppseek.permissions import PermissionEngine
from deppseek.permissions.prompt import Approval, Approver
from deppseek.tools import Toolbox, ToolContext
from deppseek.tools.fs import detect_newline, is_ignored


@pytest.fixture
def box(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config(workspace=workspace)
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    approver = Approver(engine, ask_fn=lambda r, v: Approval(approved=True))
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=approver,
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "test"),
    )
    return Toolbox.build(ctx), workspace, ctx


def test_ignored_dirs_are_matched_relatively(tmp_path):
    """Regression: v1 tested every component of the *absolute* path, so a
    workspace under any directory named env/ or node_modules/ saw its whole tree
    as ignored."""
    assert is_ignored("node_modules/pkg/index.js")
    assert is_ignored("src/__pycache__/x.pyc")
    assert not is_ignored("src/solver.py")
    assert not is_ignored("environments/case1/run.py")


def test_read_then_edit_round_trip(box):
    toolbox, workspace, _ = box
    (workspace / "solver.py").write_text("dt = 0.01\nnu = 1e-6\n")

    read = toolbox.execute("read_file", {"path": "solver.py"})
    assert "1| dt = 0.01" in read.content

    edited = toolbox.execute(
        "edit_file", {"path": "solver.py", "old_text": "dt = 0.01", "new_text": "dt = 0.001"}
    )
    assert not edited.is_error
    assert (workspace / "solver.py").read_text().startswith("dt = 0.001")


def test_edit_refuses_ambiguous_match_without_writing(box):
    toolbox, workspace, _ = box
    (workspace / "a.py").write_text("x = 1\ny = 2\nx = 1\n")
    original = (workspace / "a.py").read_text()

    result = toolbox.execute("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 9"})
    assert result.is_error
    assert "appears 2 time(s)" in result.content
    assert (workspace / "a.py").read_text() == original


def test_edit_can_replace_all_when_asked(box):
    toolbox, workspace, _ = box
    (workspace / "a.py").write_text("x = 1\ny = 2\nx = 1\n")
    result = toolbox.execute(
        "edit_file",
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 9", "expected_matches": 2},
    )
    assert not result.is_error
    assert (workspace / "a.py").read_text().count("x = 9") == 2


def test_edit_miss_explains_whitespace_cause(box):
    """A failed exact match is almost always indentation, so say so rather than
    leaving the model to guess and retry blindly."""
    toolbox, workspace, _ = box
    # The file uses aligned assignment; the model reproduces it with single
    # spaces. Not a substring, but identical once whitespace is normalised.
    (workspace / "a.py").write_text("dt   =  0.01\nnu   =  1e-6\n")

    result = toolbox.execute(
        "edit_file", {"path": "a.py", "old_text": "dt = 0.01", "new_text": "dt = 0.001"}
    )
    assert result.is_error
    assert "whitespace" in result.content.lower()
    assert (workspace / "a.py").read_text() == "dt   =  0.01\nnu   =  1e-6\n"


def test_edit_matching_exact_indentation_succeeds(box):
    toolbox, workspace, _ = box
    (workspace / "a.py").write_text("def f():\n        deep = True\n")
    result = toolbox.execute(
        "edit_file",
        {"path": "a.py", "old_text": "        deep = True", "new_text": "        deep = False"},
    )
    assert not result.is_error
    assert "deep = False" in (workspace / "a.py").read_text()


def test_overwriting_an_unread_file_is_refused(box):
    """The most common way an autonomous agent destroys work."""
    toolbox, workspace, _ = box
    (workspace / "important.py").write_text("# 500 lines of solver\n")

    result = toolbox.execute("write_file", {"path": "important.py", "content": "pass\n"})
    assert result.is_error
    assert "has not been read" in result.content
    assert (workspace / "important.py").read_text() == "# 500 lines of solver\n"

    toolbox.execute("read_file", {"path": "important.py"})
    result = toolbox.execute("write_file", {"path": "important.py", "content": "pass\n"})
    assert not result.is_error


def test_creating_a_new_file_needs_no_prior_read(box):
    toolbox, workspace, _ = box
    result = toolbox.execute("write_file", {"path": "new/deep/file.py", "content": "x = 1\n"})
    assert not result.is_error
    assert (workspace / "new" / "deep" / "file.py").read_text() == "x = 1\n"


def test_multi_edit_is_atomic(box):
    toolbox, workspace, _ = box
    (workspace / "a.py").write_text("alpha\nbeta\ngamma\n")
    original = (workspace / "a.py").read_text()

    result = toolbox.execute(
        "multi_edit",
        {
            "path": "a.py",
            "edits": [
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "does-not-exist", "new_text": "x"},
            ],
        },
    )
    assert result.is_error
    assert (workspace / "a.py").read_text() == original


def test_crlf_line_endings_are_preserved(box):
    toolbox, workspace, _ = box
    (workspace / "win.m").write_bytes(b"a = 1;\r\nb = 2;\r\n")
    toolbox.execute("read_file", {"path": "win.m"})
    toolbox.execute("edit_file", {"path": "win.m", "old_text": "a = 1;", "new_text": "a = 3;"})
    assert b"\r\n" in (workspace / "win.m").read_bytes()
    assert detect_newline("a\r\nb") == "\r\n"


def test_binary_files_are_refused_with_a_reason(box):
    toolbox, workspace, _ = box
    (workspace / "data.mat").write_bytes(bytes(range(256)) * 40)
    result = toolbox.execute("read_file", {"path": "data.mat"})
    assert result.is_error and "binary" in result.content.lower()


def test_path_escape_is_denied(box):
    toolbox, _, _ = box
    result = toolbox.execute("read_file", {"path": "../../../etc/passwd"})
    assert result.is_error
    assert "outside the workspace" in result.content


def test_credential_read_is_denied_through_the_toolbox(box):
    toolbox, workspace, _ = box
    # Assembled from fragments: a credential-shaped literal in a source file
    # trips secret scanners on push, even when it is entirely synthetic.
    fake_key = "sk-" + "a" * 32
    (workspace / ".env").write_text(f"DEEPSEEK_API_KEY={fake_key}\n")
    result = toolbox.execute("read_file", {"path": ".env"})
    assert result.is_error
    assert "sk-" not in result.content


def test_secrets_in_a_legitimate_file_are_redacted(box):
    """The path was allowed, so redaction is the thing standing between a
    hardcoded key and the API."""
    toolbox, workspace, _ = box
    fake_key = "sk-" + "1234567890abcdef" * 2
    (workspace / "config.py").write_text(f'KEY = "{fake_key}"\n')
    result = toolbox.execute("read_file", {"path": "config.py"})
    assert fake_key not in result.content
    assert "REDACTED" in result.content


def test_unknown_tool_lists_what_is_available(box):
    toolbox, _, _ = box
    result = toolbox.execute("frobnicate", {})
    assert result.is_error and "read_file" in result.content


def test_unknown_argument_is_reported_actionably(box):
    toolbox, _, _ = box
    result = toolbox.execute("read_file", {"path": "a.py", "lines": 5})
    assert result.is_error and "unknown argument" in result.content


def test_malformed_json_arguments_are_repaired_or_explained(box):
    toolbox, workspace, _ = box
    (workspace / "a.py").write_text("x\n")
    assert not toolbox.execute("read_file", '{"path": "a.py",}').is_error
    assert toolbox.execute("read_file", "not json at all").is_error


def test_oversized_result_is_spilled_not_dumped_into_context(box):
    toolbox, workspace, ctx = box
    ctx.config = Config(workspace=workspace)
    big = "\n".join(f"line {i} " + "x" * 80 for i in range(2000))
    (workspace / "big.log").write_text(big)

    result = toolbox.execute("read_file", {"path": "big.log", "line_count": 2000})
    assert len(result.content) < ctx.config.context.max_tool_result_chars * 1.2
    assert result.spilled_to is not None and result.spilled_to.exists()
    assert "characters omitted" in result.content


def test_mutations_are_checkpointed_and_undoable(box):
    toolbox, workspace, ctx = box
    (workspace / "a.py").write_text("v1\n")
    toolbox.execute("read_file", {"path": "a.py"})
    toolbox.execute("write_file", {"path": "a.py", "content": "v2\n"})

    assert (workspace / "a.py").read_text() == "v2\n"
    ctx.checkpoints.undo()
    assert (workspace / "a.py").read_text() == "v1\n"


def test_modified_paths_are_tracked_for_commit_staging(box):
    toolbox, workspace, ctx = box
    toolbox.execute("write_file", {"path": "new.py", "content": "x\n"})
    assert "new.py" in ctx.state.get("modified_paths", set())


def test_tool_schemas_are_stable_across_calls(box):
    """Byte-identical schemas between steps are what lets the provider's prefix
    cache hit; an unstable ordering would multiply input cost."""
    import json

    toolbox, _, _ = box
    first = json.dumps(toolbox.schemas(), sort_keys=False)
    second = json.dumps(toolbox.schemas(), sort_keys=False)
    assert first == second
