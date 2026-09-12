"""Delegation to a sub-agent.

The reason to delegate is context economy, not parallelism. Searching a large
codebase for where a boundary condition is applied may take fifteen tool calls
and produce fifty thousand tokens of file excerpts, of which only the answer
matters. Running that in a separate conversation and returning only the
conclusion keeps the main thread's context, and therefore its cost, bounded.

The sub-agent inherits the same permission engine and checkpoint store, so
delegation cannot be used to escape the safety model. It gets a read-biased
toolset by default, because a summarising task has no business writing files.
"""

from __future__ import annotations

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool

# Tools a read-only investigator needs and nothing more.
INVESTIGATE_TOOLS = (
    "list_workspace", "tree", "read_file", "glob_files", "search_workspace",
    "git_status", "git_diff", "git_log", "read_notebook", "list_figures",
    "check_units", "convert_units",
)

MAX_DELEGATE_STEPS = 25


@tool(slow=True)
def delegate(
    ctx: ToolContext,
    task: str,
    mode: str = "investigate",
    max_steps: int = 12,
) -> ToolResult:
    """Hand a self-contained investigation to a sub-agent and get back its findings.

    Use this when answering a question would take many file reads whose contents
    you do not need to keep, for example "find every place the inlet boundary
    condition is set and summarise how they differ". The sub-agent runs its own
    conversation and returns only its conclusion.

    Do not use it for work that changes files: make those edits yourself, where
    you can see them.

    Args:
        task: A complete, self-contained instruction. The sub-agent sees none of
            this conversation, so include every detail it needs.
        mode: "investigate" for read-only exploration, "full" to also allow
            running code. Writing is never permitted.
        max_steps: Step ceiling for the sub-agent, 1 to 25.
    """
    from ..agent import AgentLoop
    from ..prompts import build_system_prompt
    from ..session.context import ConversationBuffer
    from .registry import Toolbox

    provider = ctx.state.get("provider")
    if provider is None:
        raise ToolError(
            "Delegation is unavailable: no provider is attached to this session."
        )

    depth = ctx.state.get("delegate_depth", 0)
    if depth >= 1:
        raise ToolError(
            "A sub-agent cannot delegate further. Do this investigation directly."
        )

    if mode not in ("investigate", "full"):
        raise ToolError(f"mode must be 'investigate' or 'full', got {mode!r}")
    max_steps = max(1, min(max_steps, MAX_DELEGATE_STEPS))

    all_names = set(Toolbox.build(ctx).all_specs())
    if mode == "investigate":
        allowed = set(INVESTIGATE_TOOLS)
    else:
        allowed = set(INVESTIGATE_TOOLS) | {
            "run_python", "run_python_snippet", "run_tests", "run_matlab",
            "matlab_status", "matlab_workspace", "fetch_paper",
        }
    excluded = tuple(sorted(all_names - allowed))

    # The sub-agent shares the permission engine and checkpoint store, so it is
    # bound by exactly the same rules as the parent.
    sub_ctx = ToolContext(
        workspace=ctx.workspace,
        config=ctx.config,
        approver=ctx.approver,
        checkpoints=ctx.checkpoints,
        ui=None,
    )
    sub_ctx.state["delegate_depth"] = depth + 1
    sub_ctx.state["provider"] = provider

    sub_toolbox = Toolbox.build(sub_ctx, exclude=excluded)

    system_prompt = build_system_prompt(
        ctx.workspace,
        autonomy=ctx.config.autonomy,
        extra=(
            "You are a sub-agent handling one self-contained investigation. You "
            "cannot ask the user anything and you cannot modify files. Investigate "
            "thoroughly, then answer with your findings: what you found, in which "
            "files and at which lines, and what you could not determine. Quote the "
            "specific lines that matter. Do not summarise vaguely; the agent "
            "reading your answer cannot see anything you saw."
        ),
    )
    buffer = ConversationBuffer(
        system_prompt,
        soft_limit=ctx.config.context.soft_limit,
        hard_limit=ctx.config.context.hard_limit,
        keep_recent_turns=ctx.config.context.keep_recent_turns,
    )

    loop = AgentLoop(
        provider=provider,
        toolbox=sub_toolbox,
        buffer=buffer,
        config=ctx.config,
        sink=None,
        on_event=lambda name, payload: None,
    )
    result = loop.run(task, max_steps=max_steps)

    header = (
        f"Sub-agent ({mode}) finished after {len(result.steps)} step(s) and "
        f"{result.tool_call_count} tool call(s)."
    )
    if result.stopped_reason != "completed":
        header += f"\nStopped because: {result.stopped_reason}"

    return ToolResult(
        content=f"{header}\n\n--- sub-agent findings ---\n{result.answer}",
        display=f"delegate: {len(result.steps)} step(s)",
    )
