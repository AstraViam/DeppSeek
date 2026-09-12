"""Figure handling and visual inspection.

For CFD and battery work the figure *is* the result: a velocity field, a
polarisation curve, a residual history. An agent that can generate plots but
never look at them is working blind, and will report that a simulation "ran
successfully" while the plot shows an obviously diverged solution.

DeepSeek's text models are not multimodal, so looking at an image requires a
separate endpoint. That is configured, not assumed:

    [vision]
    base_url = "https://api.openai.com/v1"
    model = "gpt-4o-mini"
    api_key_env = "DEPPSEEK_VISION_API_KEY"

With nothing configured, `list_figures` still works and `inspect_figure` says
plainly that no vision model is available rather than inventing a description.
Uploading a figure sends workspace content to a third party, so it always
requires approval regardless of autonomy tier.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def find_figures(ctx: ToolContext, limit: int = 50) -> list[tuple[float, Path]]:
    """Find image files in the workspace, newest first."""
    found: list[tuple[float, Path]] = []
    for path in ctx.workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(ctx.workspace).as_posix()
        except ValueError:
            continue
        if "/node_modules/" in relative or relative.startswith("node_modules/"):
            continue
        try:
            found.append((path.stat().st_mtime, path))
        except OSError:
            continue
    found.sort(reverse=True)
    return found[:limit]


@tool()
def list_figures(ctx: ToolContext, limit: int = 25) -> ToolResult:
    """List image files in the workspace, newest first.

    Use this after running code that produces plots, to confirm what was actually
    written and when.

    Args:
        limit: Maximum number of figures to list.
    """
    import time

    figures = find_figures(ctx, max(1, min(limit, 200)))
    if not figures:
        return ToolResult(
            content="No image files found in the workspace.", display="figures: none"
        )

    lines = [f"{len(figures)} figure(s), newest first:"]
    for mtime, path in figures:
        relative = path.relative_to(ctx.workspace).as_posix()
        age = time.time() - mtime
        when = f"{int(age)}s ago" if age < 300 else time.strftime("%H:%M", time.localtime(mtime))
        lines.append(f"  {relative:<60} {path.stat().st_size:>9,} bytes  {when}")

    if not ctx.config.vision.enabled:
        lines.append(
            "\nNo vision model is configured, so these cannot be looked at. "
            "Judge the results from the underlying numbers instead, or configure "
            "[vision] in .deppseek/config.toml."
        )
    return ToolResult(content="\n".join(lines), display=f"figures: {len(figures)}")


@tool(path_arg="path", slow=True)
def inspect_figure(ctx: ToolContext, path: str, question: str = "") -> ToolResult:
    """Look at a figure and describe what it shows.

    Requires a configured vision endpoint. Sending the image transmits workspace
    content to a third-party service, so this always asks for approval.

    Args:
        path: Workspace-relative path to an image file.
        question: What to look for, e.g. "has the residual converged?" or
            "is there an unphysical discontinuity at the inlet?".
    """
    vision = ctx.config.vision
    if not vision.enabled:
        missing = []
        if not vision.base_url:
            missing.append("vision.base_url")
        if not vision.model:
            missing.append("vision.model")
        import os

        if not os.getenv(vision.api_key_env):
            missing.append(f"${vision.api_key_env}")
        raise ToolError(
            "No vision model is configured, so this image cannot be examined. "
            f"Missing: {', '.join(missing)}. DeepSeek's text models are not "
            "multimodal, so figure inspection needs a separate OpenAI-compatible "
            "vision endpoint. Assess the result from the numbers instead, and say "
            "that you did not look at the plot."
        )

    target, relative, _ = ctx.resolve(path)
    if not target.is_file():
        raise ToolError(f"Image not found: {path}")
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ToolError(f"{path} is not a recognised image format.")

    size = target.stat().st_size
    if size > vision.max_image_bytes:
        raise ToolError(
            f"{relative} is {size:,} bytes, above the {vision.max_image_bytes:,} "
            f"byte limit. Re-export it at a lower resolution."
        )

    mime = mimetypes.guess_type(target.name)[0] or "image/png"
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")

    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv(vision.api_key_env), base_url=vision.base_url)
    prompt = question.strip() or (
        "Describe this engineering figure. State what is plotted on each axis "
        "with units if labelled, the qualitative shape of the data, and anything "
        "that looks wrong: divergence, discontinuities, clipping, missing data, "
        "or values that are physically implausible."
    )

    try:
        response = client.chat.completions.create(
            model=vision.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure is reportable
        raise ToolError(f"Vision request failed: {exc}") from exc

    description = (response.choices[0].message.content or "").strip()
    return ToolResult(
        content=(
            f"Visual inspection of {relative} using {vision.model}:\n\n{description}\n\n"
            f"(This is a vision model's reading of the image, not a measurement. "
            f"Verify anything load-bearing against the underlying data.)"
        ),
        display=f"inspected {relative}",
    )
