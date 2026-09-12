#!/usr/bin/env python3
"""Measure the hot paths, so a regression shows up as a number.

Run it before and after a change that touches context accounting, file reading,
or startup:

    python scripts/benchmark.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def timed(fn, repeats: int = 5) -> float:
    """Median wall-clock milliseconds over `repeats` runs."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return sorted(samples)[len(samples) // 2]


def bench_startup() -> None:
    print("Startup (cold process, median of 5)")
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}

    def run():
        subprocess.run(
            [sys.executable, "-m", "deppseek.cli", "--version"],
            env=env, capture_output=True, check=False,
        )

    print(f"  deppseek --version            {timed(run):8.0f} ms")


def bench_context() -> None:
    from deppseek.session.context import ConversationBuffer

    print("\nContext accounting (estimate() twice per step, as the loop does)")
    for steps in (20, 40, 80, 160, 320):
        buffer = ConversationBuffer("system " * 200)

        start = time.perf_counter()
        for index in range(steps):
            buffer.append_user(f"step {index}")
            buffer.append({
                "role": "assistant", "content": None,
                "reasoning_content": "thinking " * 300,
                "tool_calls": [{"id": f"c{index}", "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"}}],
            })
            buffer.append_tool_result(f"c{index}", "x = 1\n" * 1500)
            buffer.estimate()
            buffer.estimate()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {steps:>3} steps, {len(buffer.messages):>4} messages  {elapsed:8.0f} ms")


def bench_file_read() -> None:
    from deppseek.tools.fs import read_line_range

    print("\nReading a 400-line window from a large result file")
    workspace = Path(tempfile.mkdtemp())
    path = workspace / "results.csv"
    path.write_text("".join(f"{i},0.01,1.2e-3,turbulent\n" for i in range(200_000)))
    size = path.stat().st_size

    tracemalloc.start()
    elapsed = timed(lambda: read_line_range(path, 1000, 400))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {size / 1e6:.1f} MB, 200k lines        {elapsed:8.1f} ms   peak {peak / 1e6:.1f} MB")


def bench_tools() -> None:
    from deppseek.checkpoint import CheckpointStore
    from deppseek.config import Config
    from deppseek.permissions import PermissionEngine
    from deppseek.permissions.prompt import Approval, Approver
    from deppseek.tools import Toolbox, ToolContext

    workspace = Path(tempfile.mkdtemp())
    config = Config(workspace=workspace)
    ctx = ToolContext(
        workspace=workspace, config=config,
        approver=Approver(PermissionEngine(), ask_fn=lambda r, v: Approval(True)),
        checkpoints=CheckpointStore(workspace / ".deppseek", workspace, "bench"),
    )
    box = Toolbox.build(ctx)

    print("\nPer-step tool overhead")
    print(f"  schemas() x100                {timed(lambda: [box.schemas() for _ in range(100)]):8.2f} ms")
    print(f"  all_specs() x1000             {timed(lambda: [box.all_specs() for _ in range(1000)]):8.2f} ms")


def bench_checkpoint() -> None:
    from deppseek.checkpoint import CheckpointStore

    print("\nCheckpointing a large file")
    workspace = Path(tempfile.mkdtemp())
    workspace.mkdir(exist_ok=True)
    mesh = workspace / "mesh.dat"
    mesh.write_bytes(b"0123456789abcdef" * 1_000_000)  # 16 MB
    store = CheckpointStore(workspace / ".deppseek", workspace, "bench")

    tracemalloc.start()
    start = time.perf_counter()
    checkpoint = store.snapshot([mesh], tool="write_file", description="bench")
    store.commit(checkpoint)
    elapsed = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  16 MB snapshot + commit       {elapsed:8.0f} ms   peak {peak / 1e6:.1f} MB")


def main() -> int:
    bench_startup()
    bench_context()
    bench_file_read()
    bench_tools()
    bench_checkpoint()
    print("\nAll figures are for this machine; compare runs, not absolutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
