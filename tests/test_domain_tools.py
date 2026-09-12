"""MATLAB, notebook, and figure tooling."""

import json

import pytest

from deppseek.checkpoint import CheckpointStore
from deppseek.config import Config, VisionConfig
from deppseek.permissions import PermissionEngine
from deppseek.permissions.prompt import Approval, Approver
from deppseek.tools import Toolbox, ToolContext
from deppseek.tools import matlab as matlab_module


@pytest.fixture
def box(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config(workspace=workspace)
    engine = PermissionEngine(autonomy="autonomous", secret_paths=config.secret_paths)
    ctx = ToolContext(
        workspace=workspace,
        config=config,
        approver=Approver(engine, ask_fn=lambda r, v: Approval(approved=True)),
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "t"),
    )
    return Toolbox.build(ctx), workspace, ctx


NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Heat exchanger\n"]},
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["U = 450 W/m2K\n"]},
                {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo="}},
            ],
            "source": ["U = 450  # W/m2K\n", "print(f'U = {U} W/m2K')\n"],
        },
    ],
    "metadata": {"kernelspec": {"language": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def test_notebook_reads_as_cells_not_json(box):
    toolbox, workspace, _ = box
    (workspace / "hx.ipynb").write_text(json.dumps(NOTEBOOK))

    result = toolbox.execute("read_notebook", {"path": "hx.ipynb"})
    assert "cell 0 [markdown]" in result.content
    assert "U = 450" in result.content
    # Base64 image data must be summarised, not inlined into the context.
    assert "iVBORw0KGgo" not in result.content
    assert "image/png output, not shown" in result.content


def test_editing_a_cell_clears_its_stale_outputs(box):
    """Leaving outputs attached to changed source lets a later read report
    results that the code now in the cell never produced."""
    toolbox, workspace, _ = box
    (workspace / "hx.ipynb").write_text(json.dumps(NOTEBOOK))
    toolbox.execute("read_notebook", {"path": "hx.ipynb"})

    result = toolbox.execute(
        "edit_notebook",
        {"path": "hx.ipynb", "cell_index": 1, "new_source": "U = 900\n"},
    )
    assert not result.is_error

    data = json.loads((workspace / "hx.ipynb").read_text())
    assert data["cells"][1]["outputs"] == []
    assert data["cells"][1]["execution_count"] is None


def test_editing_without_reading_first_is_refused(box):
    toolbox, workspace, _ = box
    (workspace / "hx.ipynb").write_text(json.dumps(NOTEBOOK))
    result = toolbox.execute(
        "edit_notebook", {"path": "hx.ipynb", "cell_index": 0, "new_source": "x"}
    )
    assert result.is_error and "read_notebook" in result.content


def test_out_of_range_cell_index_reports_the_range(box):
    toolbox, workspace, _ = box
    (workspace / "hx.ipynb").write_text(json.dumps(NOTEBOOK))
    toolbox.execute("read_notebook", {"path": "hx.ipynb"})
    result = toolbox.execute(
        "edit_notebook", {"path": "hx.ipynb", "cell_index": 9, "new_source": "x"}
    )
    assert result.is_error and "0-1" in result.content


def test_matlab_probe_explains_an_unsupported_interpreter():
    """On Python 3.14 the MATLAB engine cannot be installed at all, so the
    message must say that rather than suggesting a pip command that will fail."""
    probe = matlab_module.probe_matlab(version_info=(3, 14))
    described = probe.describe()
    assert not probe.engine_supported_here
    assert "3.13 or earlier" in described
    assert "cannot be installed on this interpreter" in described


def test_matlab_probe_suggests_install_on_a_supported_interpreter():
    probe = matlab_module.probe_matlab(version_info=(3, 12))
    assert probe.engine_supported_here
    assert "pip install matlabengine" in probe.describe()


@pytest.mark.parametrize(
    "version,supported",
    [((3, 8), False), ((3, 9), True), ((3, 13), True), ((3, 14), False), ((3, 15), False)],
)
def test_engine_support_window_matches_mathworks_declaration(version, supported):
    """matlabengine 26.1.12 (R2026a) declares python_requires ">=3.9, <3.14"."""
    assert matlab_module.probe_matlab(version_info=version).engine_supported_here is supported


def test_matlab_missing_executable_names_the_config_key(box):
    toolbox, _, ctx = box
    ctx.state["matlab_engine"] = False
    import deppseek.tools.matlab as module
    from deppseek.tools.matlab import MatlabProbe

    original = module.probe_matlab
    module.probe_matlab = lambda exe="matlab": MatlabProbe(
        False, "not installed", None, "3.14", False
    )
    try:
        result = toolbox.execute("run_matlab", {"command": "disp(1)"})
    finally:
        module.probe_matlab = original

    assert result.is_error
    assert "MATLAB_EXE" in result.content


def test_figure_inspection_without_a_vision_model_refuses_rather_than_inventing(box):
    toolbox, workspace, _ctx = box
    (workspace / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    result = toolbox.execute("inspect_figure", {"path": "plot.png"})
    assert result.is_error
    assert "not multimodal" in result.content
    assert "did not look at the plot" in result.content


def test_list_figures_says_when_it_cannot_look(box):
    toolbox, workspace, _ = box
    (workspace / "u.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    result = toolbox.execute("list_figures", {})
    assert "u.png" in result.content
    assert "No vision model is configured" in result.content


def test_vision_config_enablement_requires_all_three_parts(monkeypatch):
    monkeypatch.delenv("DEPPSEEK_VISION_API_KEY", raising=False)
    assert not VisionConfig(base_url="https://x", model="m").enabled
    monkeypatch.setenv("DEPPSEEK_VISION_API_KEY", "k")
    assert VisionConfig(base_url="https://x", model="m").enabled
    assert not VisionConfig(model="m").enabled


def test_unit_mismatch_is_reported_as_an_error(box):
    pytest.importorskip("pint")
    toolbox, _, _ = box
    # Hydrostatic pressure expressed correctly.
    ok = toolbox.execute(
        "check_units",
        {"expression": "1000 * kg/m**3 * 9.81 * m/s**2 * 10 * m", "expected": "Pa"},
    )
    assert not ok.is_error
    # Same expression claimed to be a temperature.
    bad = toolbox.execute(
        "check_units",
        {"expression": "1000 * kg/m**3 * 9.81 * m/s**2 * 10 * m", "expected": "K"},
    )
    assert bad.is_error and "MISMATCH" in bad.content


def test_unit_tools_explain_a_missing_dependency(box, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pint":
            raise ImportError("No module named 'pint'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    toolbox, _, _ = box
    result = toolbox.execute("check_units", {"expression": "1 * m"})
    assert result.is_error and "pip install pint" in result.content


def test_external_content_is_labelled_untrusted():
    from deppseek.tools.research import UNTRUSTED_BANNER

    assert "verified" in UNTRUSTED_BANNER
    assert "not as instructions" in UNTRUSTED_BANNER


def test_web_search_without_a_key_says_what_to_configure(box):
    toolbox, _, _ = box
    result = toolbox.execute("web_search", {"query": "nusselt correlation"})
    assert result.is_error
    assert "search.provider" in result.content
    assert "fetch_paper works without any key" in result.content
