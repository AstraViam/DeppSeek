"""Jupyter notebook tools.

A notebook is JSON, so `read_file` technically works on it and produces an
unreadable wall of escaped source, base64 image outputs, and execution metadata.
That wastes context and makes editing nearly impossible. These tools present a
notebook as numbered cells and edit it cell by cell, preserving everything the
agent did not touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool

MAX_OUTPUT_CHARS = 2000


def load_notebook(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"{path.name} is not valid notebook JSON: {exc}") from exc
    if "cells" not in data:
        raise ToolError(f"{path.name} has no 'cells' key; it may not be a notebook.")
    return data


def save_notebook(path: Path, data: dict[str, Any]) -> None:
    # nbformat writes one-space indentation and a trailing newline. Matching that
    # keeps the diff to the cells actually changed rather than the whole file.
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def render_outputs(cell: dict[str, Any]) -> str:
    """Render a cell's outputs as text, summarising images rather than inlining them."""
    rendered: list[str] = []
    for output in cell.get("outputs", []) or []:
        kind = output.get("output_type")
        if kind == "stream":
            text = "".join(output.get("text", []))
            rendered.append(text)
        elif kind in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "text/plain" in data:
                rendered.append("".join(data["text/plain"]))
            for mime in data:
                if mime.startswith("image/"):
                    rendered.append(f"[{mime} output, not shown]")
        elif kind == "error":
            rendered.append(
                f"{output.get('ename')}: {output.get('evalue')}\n"
                + "\n".join(output.get("traceback", [])[:12])
            )
    text = "\n".join(rendered).strip()
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} more characters]"
    return text


@tool(path_arg="path")
def read_notebook(ctx: ToolContext, path: str, include_outputs: bool = True) -> ToolResult:
    """Read a Jupyter notebook as numbered cells.

    Use this instead of read_file for .ipynb: raw notebook JSON is mostly escape
    sequences and base64 images.

    Args:
        path: Workspace-relative path to a .ipynb file.
        include_outputs: Include cell outputs, with images summarised not inlined.
    """
    target, relative, _ = ctx.resolve(path)
    if not target.is_file():
        raise ToolError(f"Notebook not found: {path}")

    data = load_notebook(target)
    cells = data.get("cells", [])
    language = (
        data.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    )

    blocks = [f"{relative}  ({len(cells)} cells, kernel language: {language})"]
    for index, cell in enumerate(cells):
        kind = cell.get("cell_type", "code")
        source = cell_source(cell)
        header = f"--- cell {index} [{kind}]"
        execution = cell.get("execution_count")
        if execution is not None:
            header += f" (ran as In[{execution}])"
        blocks.append(header + " ---")
        blocks.append(source.rstrip() or "(empty)")
        if include_outputs and kind == "code":
            outputs = render_outputs(cell)
            if outputs:
                blocks.append(f"--- cell {index} output ---")
                blocks.append(outputs)

    ctx.read_files.add(relative)
    return ToolResult(
        content="\n".join(blocks),
        display=f"{relative}: {len(cells)} cells",
        touched=(target,),
    )


@tool(path_arg="path", mutates=True)
def edit_notebook(
    ctx: ToolContext,
    path: str,
    cell_index: int,
    new_source: str,
    mode: str = "replace",
) -> ToolResult:
    """Edit one cell of a Jupyter notebook.

    Args:
        path: Workspace-relative path to a .ipynb file.
        cell_index: Zero-based cell index, as shown by read_notebook.
        new_source: New cell source. Ignored when mode is "delete".
        mode: One of "replace", "insert_after", "insert_before", "delete".
    """
    target, relative, _ = ctx.resolve(path)
    if not target.is_file():
        raise ToolError(f"Notebook not found: {path}")
    if relative not in ctx.read_files:
        raise ToolError(
            f"Read {relative} with read_notebook before editing it, so the cell "
            f"indices you are using are current."
        )

    data = load_notebook(target)
    cells = data.get("cells", [])
    if mode not in {"replace", "insert_after", "insert_before", "delete"}:
        raise ToolError(f"Unknown mode {mode!r}. Use replace, insert_after, insert_before, or delete.")
    if not cells and mode != "insert_after":
        raise ToolError("The notebook has no cells to edit.")
    if mode != "insert_after" and not 0 <= cell_index < len(cells):
        raise ToolError(
            f"cell_index {cell_index} is out of range; the notebook has {len(cells)} cells (0-{len(cells) - 1})."
        )

    def make_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    if mode == "replace":
        cell = cells[cell_index]
        cell["source"] = new_source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            # The stored outputs no longer correspond to the source, and leaving
            # them would let a later read report results that were never produced
            # by the code now in the cell.
            cell["outputs"] = []
            cell["execution_count"] = None
        detail = f"Replaced cell {cell_index} and cleared its stale outputs"
    elif mode == "delete":
        cells.pop(cell_index)
        detail = f"Deleted cell {cell_index}"
    elif mode == "insert_before":
        cells.insert(cell_index, make_cell(new_source))
        detail = f"Inserted a new cell before index {cell_index}"
    else:
        position = min(cell_index + 1, len(cells))
        cells.insert(position, make_cell(new_source))
        detail = f"Inserted a new cell at index {position}"

    data["cells"] = cells
    save_notebook(target, data)
    return ToolResult(
        content=f"{detail} in {relative}. The notebook now has {len(cells)} cells.",
        display=detail,
        touched=(target,),
    )
