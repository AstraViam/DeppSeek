"""Plan and progress tracking.

An agent working a multi-step task without an explicit plan drifts: it fixes the
first thing it finds, forgets the second, and reports success. A written todo
list is a cheap correction, because it is re-sent with every request and so acts
as a standing reminder of what is not done.

The list lives in the session file, so it survives resume.
"""

from __future__ import annotations

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool

VALID_STATUS = ("pending", "in_progress", "done", "blocked")
STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}


def _todos(ctx: ToolContext) -> list[dict]:
    return ctx.state.setdefault("todos", [])


def render_todos(todos: list[dict]) -> str:
    if not todos:
        return "No plan recorded."
    lines: list[str] = []
    for index, item in enumerate(todos, start=1):
        mark = STATUS_MARK.get(item.get("status", "pending"), "[ ]")
        note = item.get("note", "")
        lines.append(f"{index:>2}. {mark} {item.get('task', '')}" + (f"  -- {note}" if note else ""))
    done = sum(1 for t in todos if t.get("status") == "done")
    lines.append(f"\n{done}/{len(todos)} complete")
    return "\n".join(lines)


@tool()
def todo_write(ctx: ToolContext, tasks: list[dict]) -> ToolResult:
    """Record or update the plan for the current task.

    Write the plan before starting anything that needs more than about three tool
    calls, then update it as you go. Replaces the whole list, so include every
    item each time, with its current status.

    Args:
        tasks: List of objects with "task" (what to do), "status" (one of
            pending, in_progress, done, blocked), and optional "note".
    """
    if not isinstance(tasks, list):
        raise ToolError("tasks must be a list of objects.")

    cleaned: list[dict] = []
    for index, item in enumerate(tasks, start=1):
        if not isinstance(item, dict):
            raise ToolError(f"tasks[{index}] must be an object with a 'task' key.")
        text = str(item.get("task", "")).strip()
        if not text:
            raise ToolError(f"tasks[{index}] has an empty 'task'.")
        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUS:
            raise ToolError(
                f"tasks[{index}] has status {status!r}; use one of {', '.join(VALID_STATUS)}."
            )
        cleaned.append({"task": text, "status": status, "note": str(item.get("note", "")).strip()})

    in_progress = [t for t in cleaned if t["status"] == "in_progress"]
    warning = ""
    if len(in_progress) > 1:
        warning = (
            f"\n\nNote: {len(in_progress)} items are marked in_progress. Work one at "
            f"a time so the plan reflects what is actually happening."
        )

    ctx.state["todos"] = cleaned
    return ToolResult(
        content=render_todos(cleaned) + warning,
        display=f"plan: {sum(1 for t in cleaned if t['status'] == 'done')}/{len(cleaned)} done",
    )


@tool()
def todo_read(ctx: ToolContext) -> ToolResult:
    """Show the current plan and what remains."""
    return ToolResult(content=render_todos(_todos(ctx)), display="plan")
