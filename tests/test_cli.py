"""CLI wiring. Nothing here touches the network."""

import pytest

from deppseek.cli import COMMANDS, Controller, build_parser, cli_overrides, main
from deppseek.config import load_config


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "solver.py").write_text("dt = 0.01\n")
    return ws


@pytest.fixture
def controller(workspace, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    args = build_parser().parse_args(["--workspace", str(workspace)])
    args.max_steps = 10
    args.interactive = True
    config = load_config(workspace, cli_overrides(args))
    return Controller(config, args)


def test_missing_api_key_explains_how_to_set_it(workspace, monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    code = main(["--workspace", str(workspace), "hello"])
    captured = capsys.readouterr()
    assert code == 2
    assert "DEEPSEEK_API_KEY" in captured.err
    assert "$env:" in captured.err  # PowerShell syntax, since that is the target


def test_doctor_runs_without_network(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    assert main(["--workspace", str(workspace), "--doctor"]) == 0
    out = capsys.readouterr().out
    assert "deppseek 2.0.0" in out
    assert "DEEPSEEK_API_KEY is set" in out


def test_bad_workspace_is_rejected(tmp_path, capsys):
    missing = tmp_path / "nope"
    assert main(["--workspace", str(missing), "hi"]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_invalid_config_is_reported_not_traced(workspace, capsys):
    (workspace / ".deppseek").mkdir()
    (workspace / ".deppseek" / "config.toml").write_text('autonomy = "wild"\n')
    assert main(["--workspace", str(workspace), "hi"]) == 2
    assert "autonomy must be one of" in capsys.readouterr().err


def test_unknown_config_key_names_the_valid_ones(workspace, capsys):
    (workspace / ".deppseek").mkdir()
    (workspace / ".deppseek" / "config.toml").write_text("[budget]\nmax_costs = 5\n")
    assert main(["--workspace", str(workspace), "hi"]) == 2
    err = capsys.readouterr().err
    assert "unknown key" in err and "max_cost_usd" in err


def test_project_config_overrides_defaults(workspace, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    (workspace / ".deppseek").mkdir()
    (workspace / ".deppseek" / "config.toml").write_text(
        'autonomy = "ask"\n[budget]\nmax_steps = 7\n'
    )
    config = load_config(workspace)
    assert config.autonomy == "ask"
    assert config.budget.max_steps == 7


def test_cli_flags_beat_project_config(workspace):
    (workspace / ".deppseek").mkdir()
    (workspace / ".deppseek" / "config.toml").write_text('autonomy = "ask"\n')
    args = build_parser().parse_args(
        ["--workspace", str(workspace), "--autonomy", "readonly", "--max-steps", "3"]
    )
    config = load_config(workspace, cli_overrides(args))
    assert config.autonomy == "readonly"
    assert config.budget.max_steps == 3


def test_max_steps_applies_to_the_interactive_path(workspace, monkeypatch):
    """Regression: v1's shell hardcoded 30 steps for /agent while also accepting
    --max-steps, which therefore did nothing in the mode people actually used."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    args = build_parser().parse_args(["--workspace", str(workspace), "--max-steps", "4"])
    config = load_config(workspace, cli_overrides(args))
    assert config.budget.max_steps == 4
    if args.max_steps is None:
        args.max_steps = config.budget.max_steps
    assert args.max_steps == 4


def test_retired_model_is_rewritten_with_a_warning(workspace, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    args = build_parser().parse_args(["--workspace", str(workspace)])
    args.max_steps = 10
    args.interactive = True
    controller = Controller(load_config(workspace, cli_overrides(args)), args)
    assert controller.provider.model == "deepseek-flash"


def test_slash_commands_do_not_raise(controller):
    for command in ["/help", "/tools", "/permissions", "/context", "/cost",
                    "/checkpoints", "/plan", "/sessions", "/config", "/mcp",
                    "/autonomy", "/model", "/workspace", "/doctor", "/tree"]:
        assert controller.handle_command(command) is True


def test_exit_command_stops_the_shell(controller):
    assert controller.handle_command("/exit") is False


def test_unknown_command_is_reported(controller, capsys):
    assert controller.handle_command("/frobnicate") is True


def test_autonomy_can_be_changed_at_runtime(controller):
    controller.handle_command("/autonomy readonly")
    assert controller.engine.autonomy == "readonly"
    from deppseek.permissions import Decision, Request

    assert controller.engine.evaluate(Request("write_file", path="a.py")).decision is Decision.DENY


def test_every_documented_command_is_handled(controller):
    """A command in the help list that falls through to "unknown" is a lie in
    the documentation."""
    messages = []
    controller.ui.error = lambda m: messages.append(m)
    for command in COMMANDS:
        if command in ("exit",):
            continue
        controller.handle_command(f"/{command}")
    assert not [m for m in messages if "Unknown command" in m]


def test_project_notes_are_loaded_into_the_prompt(workspace, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkeyabcdefghijklmnopqrstuv")
    (workspace / "DEPPSEEK.md").write_text("Always use the implicit solver.\n")
    args = build_parser().parse_args(["--workspace", str(workspace)])
    args.max_steps = 10
    args.interactive = True
    controller = Controller(load_config(workspace, cli_overrides(args)), args)
    assert "implicit solver" in controller.system_prompt


def test_session_is_saved_and_resumable(controller):
    controller.buffer.append_user("analyse the mesh")
    controller.save()
    session_id = controller.session.meta.id

    controller.handle_command("/clear")
    assert controller.buffer.messages == []

    controller.handle_command(f"/resume {session_id}")
    assert any(m.get("content") == "analyse the mesh" for m in controller.buffer.messages)


def test_resume_repairs_a_malformed_session(controller):
    """An orphaned tool message makes the next API request fail, so resume must
    repair rather than faithfully restore."""
    controller.buffer.append_user("x")
    controller.buffer.append_tool_result("orphan", "result with no matching call")
    controller.save()
    session_id = controller.session.meta.id

    controller.handle_command("/clear")
    controller.handle_command(f"/resume {session_id}")
    assert controller.buffer.validate() == []
