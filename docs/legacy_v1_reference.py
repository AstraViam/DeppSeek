import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ============================================================
# DeepSeek Engineering Agent
# ============================================================
# A local, approval-gated coding/research agent for Windows.
#
# Features:
#   - DeepSeek OpenAI-compatible API
#   - Thinking mode + reasoning effort
#   - Interactive chat
#   - Agent mode with function/tool calling
#   - Workspace-scoped file operations
#   - Recursive code/text search
#   - Python execution
#   - PowerShell execution (approval required)
#   - MATLAB execution (approval required)
#   - Git status/diff/commit (commit requires approval)
#   - Conversation persistence
#   - Basic usage/cost logging from API usage metadata
#   - Explicit approval before destructive or executable actions
#
# SECURITY MODEL:
#   The agent is allowed to READ only inside the workspace.
#   WRITE and EXECUTION actions require user approval unless
#   --auto-approve is explicitly supplied.
#
# NOTE:
#   Never put your API key in this file.
#   Use DEEPSEEK_API_KEY as an environment variable.
# ============================================================

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEFAULT_REASONING = os.getenv("DEEPSEEK_REASONING", "high")

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}

TEXT_EXTENSIONS = {
    ".py", ".pyw", ".m", ".mlx", ".mat", ".mli",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".java", ".js", ".ts", ".tsx", ".jsx",
    ".cs", ".rs", ".go", ".sql", ".sh", ".bat", ".ps1",
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".xml", ".html", ".css", ".tex", ".bib",
}

MAX_READ_CHARS = 300_000
MAX_SEARCH_RESULTS = 200
MAX_FILE_SIZE = 2_000_000

# These are intentionally approximate reference rates. The API response's
# usage metadata is still the authoritative source for token counts.
# Update them when you want a billing estimate matching your account pricing.
PRICE_USD_PER_1M = {
    "deepseek-v4-flash": {
        "input": 0.22,
        "output": 0.66,
    },
    "deepseek-v4-pro": {
        "input": 0.66,
        "output": 1.98,
    },
    "deepseek-flash": {
        "input": 0.22,
        "output": 0.66,
    },
}

SYSTEM_PROMPT = r"""
You are an expert engineering and scientific-computing agent operating inside a
local project workspace.

The user is a chemical engineering student working on demanding technical
projects including:
- CFD and fluid mechanics
- thermodynamics and transport phenomena
- process safety
- batteries and energy systems
- MATLAB / Simulink / Simscape
- Python and scientific computing
- optimization and AI/ML
- research papers and experimental analysis

Your job is to help inspect, reason about, modify, and test the user's project.

CORE RULES
1. Work from evidence in the workspace. Do not invent file contents.
2. Before changing code, inspect the relevant files and understand the existing
   architecture.
3. For scientific/engineering work, state assumptions and units, identify
   governing equations, and flag approximations.
4. Prefer the smallest correct change over large rewrites.
5. After making a change, test it when practical.
6. When a tool fails, diagnose the error from the actual output before retrying.
7. Never claim a simulation, test, or command succeeded unless its output shows
   that it succeeded.
8. Treat numerical results as engineering evidence, not as proof of validity.
9. When modifying research code, preserve reproducibility and explain what was
   changed.
10. Ask for approval before any tool that can change files, execute programs,
    commit changes, or otherwise modify external state. The local harness
    enforces this even if you request the tool.

ENGINEERING STYLE
- First principles first.
- Keep notation explicit.
- Use SI units unless the project clearly uses another system.
- Distinguish model assumptions from measured data.
- Call out numerical stability, convergence, mesh dependence, boundary
  conditions, and conservation issues in CFD.
- For batteries, check sign conventions, SOC/SOH definitions, temperature
  dependence, current limits, heat generation, and energy/power consistency.
- For MATLAB, preserve vectorization and toolbox compatibility where practical.

TOOLS
You have tools to inspect files, search the workspace, edit files, run Python,
run MATLAB, execute PowerShell, and inspect Git state. Use them deliberately.
Do not call tools unnecessarily.
""".strip()


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def print_banner(model: str, workspace: Path, thinking: bool, reasoning: str) -> None:
    print("\n" + "=" * 72)
    print("DEEPSEEK ENGINEERING AGENT")
    print("=" * 72)
    print(f"Model:      {model}")
    print(f"Workspace:  {workspace}")
    print(f"Thinking:   {'ON' if thinking else 'OFF'}")
    print(f"Reasoning:  {reasoning if thinking else 'disabled'}")
    print("=" * 72)
    print("/help /exit /clear /model /workspace /tree /pwd /tools /usage")
    print("/agent <task>       Run autonomous tool-assisted task")
    print("/chat <prompt>      Plain DeepSeek response")
    print("/approve on|off     Toggle approval requirement")
    print("=" * 72 + "\n")


def safe_json_loads(value: str) -> Dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tool JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return data


class Agent:
    def __init__(
        self,
        workspace: Path,
        model: str,
        thinking: bool,
        reasoning_effort: str,
        auto_approve: bool = False,
        history_file: Optional[Path] = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.auto_approve = auto_approve
        self.history_file = history_file or (self.workspace / ".deepseek_agent_history.json")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.estimated_cost_usd = 0.0

        self.client = self._make_client()
        self.messages: List[Dict[str, Any]] = []
        self._load_history()

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------
    def _make_client(self) -> OpenAI:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. "
                'Set it with: $env:DEEPSEEK_API_KEY = "YOUR_NEW_KEY"'
            )
        return OpenAI(api_key=api_key, base_url=BASE_URL)

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    def _load_history(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not self.history_file.exists():
            return
        try:
            saved = json.loads(self.history_file.read_text(encoding="utf-8"))
            if isinstance(saved, list):
                self.messages.extend(saved)
        except Exception as exc:
            print(f"[warning] Could not load history: {exc}")

    def _save_history(self) -> None:
        try:
            # Keep the system prompt out of the saved file.
            self.history_file.write_text(
                json.dumps(self.messages[1:], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[warning] Could not save history: {exc}")

    def clear_history(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        try:
            if self.history_file.exists():
                self.history_file.unlink()
        except Exception:
            pass

    # --------------------------------------------------------
    # Paths / security
    # --------------------------------------------------------
    def resolve_workspace_path(self, raw_path: str) -> Path:
        raw = Path(raw_path).expanduser()
        candidate = raw if raw.is_absolute() else self.workspace / raw
        candidate = candidate.resolve()

        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(
                f"Path escapes workspace: {candidate}\n"
                f"Workspace: {self.workspace}"
            ) from exc

        return candidate

    def _is_ignored(self, path: Path) -> bool:
        return any(part in IGNORE_DIRS for part in path.parts)

    # --------------------------------------------------------
    # Approval
    # --------------------------------------------------------
    def approve(self, action: str, details: str) -> bool:
        if self.auto_approve:
            return True

        print("\n" + "!" * 72)
        print("APPROVAL REQUIRED")
        print(f"Action: {action}")
        print(details)
        print("!" * 72)
        answer = input("Approve? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------
    def tool_list_workspace(self, path: str = ".") -> str:
        root = self.resolve_workspace_path(path)
        if not root.exists():
            return f"Path does not exist: {root}"
        if not root.is_dir():
            return f"Not a directory: {root}"

        items = []
        for p in sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if self._is_ignored(p):
                continue
            marker = "[DIR]" if p.is_dir() else "[FILE]"
            items.append(f"{marker} {p.relative_to(self.workspace)}")

        return "\n".join(items[:500]) or "(empty)"

    def tool_tree(self, max_depth: int = 3) -> str:
        max_depth = max(1, min(max_depth, 8))
        lines: List[str] = [self.workspace.name + "/"]

        def walk(root: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                children = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except OSError as exc:
                lines.append(prefix + f"[ERROR] {exc}")
                return
            children = [p for p in children if not self._is_ignored(p)]
            for i, p in enumerate(children):
                last = i == len(children) - 1
                branch = "└── " if last else "├── "
                if p.is_dir():
                    lines.append(prefix + branch + p.name + "/")
                    walk(p, prefix + ("    " if last else "│   "), depth + 1)
                else:
                    lines.append(prefix + branch + p.name)

        walk(self.workspace, "", 1)
        return "\n".join(lines)

    def tool_read_file(self, path: str, start_line: int = 1, end_line: int = 500) -> str:
        target = self.resolve_workspace_path(path)
        if not target.exists():
            return f"File does not exist: {path}"
        if not target.is_file():
            return f"Not a file: {path}"
        if target.stat().st_size > MAX_FILE_SIZE:
            return f"File is too large ({target.stat().st_size} bytes). Use search_file or read smaller files."

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Could not read file: {exc}"

        lines = text.splitlines()
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        selected = lines[start - 1:end]

        if sum(len(x) for x in selected) > MAX_READ_CHARS:
            selected = selected[: max(1, end - start)]

        output = []
        for idx, line in enumerate(selected, start=start):
            output.append(f"{idx:6}: {line}")
        return "\n".join(output)

    def tool_search(self, pattern: str, glob: str = "*", max_results: int = 100) -> str:
        import re

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
        results = []

        for path in self.workspace.rglob(glob):
            if len(results) >= max_results:
                break
            if not path.is_file() or self._is_ignored(path):
                continue
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
                if path.suffix.lower() not in TEXT_EXTENSIONS and path.name.lower() not in {"readme", "license"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        results.append(f"{path.relative_to(self.workspace)}:{lineno}: {line[:500]}")
                        if len(results) >= max_results:
                            break
            except (OSError, UnicodeError):
                continue

        return "\n".join(results) if results else "No matches found."

    def tool_write_file(self, path: str, content: str, create_dirs: bool = True) -> str:
        target = self.resolve_workspace_path(path)

        if len(content) > MAX_READ_CHARS * 4:
            return "Refused: content is too large for a single write operation."

        existed = target.exists()
        detail = f"Path: {target}\nMode: {'OVERWRITE' if existed else 'CREATE'}\nCharacters: {len(content)}"
        if not self.approve("write_file", detail):
            return "User denied write operation."

        try:
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} characters to {target.relative_to(self.workspace)}"
        except Exception as exc:
            return f"Write failed: {exc}"

    def tool_make_directory(self, path: str) -> str:
        target = self.resolve_workspace_path(path)
        if not self.approve("make_directory", f"Create directory: {target}"):
            return "User denied directory creation."
        try:
            target.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {target.relative_to(self.workspace)}"
        except Exception as exc:
            return f"Directory creation failed: {exc}"

    def tool_run_python(self, script: str, args: Optional[List[str]] = None, timeout: int = 120) -> str:
        target = self.resolve_workspace_path(script)
        if not target.exists() or not target.is_file():
            return f"Python script not found: {script}"

        args = args or []
        command = [sys.executable, str(target), *map(str, args)]
        detail = f"CWD: {self.workspace}\nCommand: {subprocess.list2cmdline(command)}\nTimeout: {timeout}s"
        if not self.approve("run_python", detail):
            return "User denied Python execution."

        return self._run_process(command, timeout)

    def tool_run_powershell(self, command: str, timeout: int = 120) -> str:
        detail = f"CWD: {self.workspace}\nPowerShell command:\n{command}\nTimeout: {timeout}s"
        if not self.approve("run_powershell", detail):
            return "User denied PowerShell execution."

        return self._run_process(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            timeout,
        )

    def tool_run_matlab(self, command: str, timeout: int = 300) -> str:
        matlab_exe = os.getenv("MATLAB_EXE", "matlab")
        detail = f"CWD: {self.workspace}\nMATLAB executable: {matlab_exe}\nMATLAB command:\n{command}\nTimeout: {timeout}s"
        if not self.approve("run_matlab", detail):
            return "User denied MATLAB execution."

        return self._run_process(
            [matlab_exe, "-batch", command],
            timeout,
        )

    def _run_process(self, command: List[str], timeout: int) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=max(1, min(timeout, 1800)),
                errors="replace",
            )
        except FileNotFoundError as exc:
            return f"Executable not found: {exc}"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return f"PROCESS TIMED OUT after {timeout}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        except Exception as exc:
            return f"Process execution failed: {exc}"

        return (
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout[-20_000:]}\n"
            f"STDERR:\n{completed.stderr[-20_000:]}"
        )

    def tool_git_status(self) -> str:
        return self._run_process(
            ["git", "status", "--short", "--branch"],
            timeout=30,
        )

    def tool_git_diff(self) -> str:
        return self._run_process(
            ["git", "diff", "--no-ext-diff", "--", "."],
            timeout=30,
        )

    def tool_git_commit(self, message: str) -> str:
        status = self.tool_git_status()
        detail = f"Commit message: {message}\n\nCurrent Git status:\n{status}"
        if not self.approve("git_commit", detail):
            return "User denied Git commit."

        add_result = self._run_process(["git", "add", "--", "."], timeout=30)
        if "Exit code: 0" not in add_result:
            return f"git add failed:\n{add_result}"

        return self._run_process(["git", "commit", "-m", message], timeout=60)

    # --------------------------------------------------------
    # Tool definitions for DeepSeek function calling
    # --------------------------------------------------------
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_workspace",
                    "description": "List files and directories inside the project workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "default": "."}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tree",
                    "description": "Show a compact recursive directory tree of the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"max_depth": {"type": "integer", "minimum": 1, "maximum": 8}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a text file with line numbers. Use this before editing code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_workspace",
                    "description": "Search project text/code files using a regular expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "glob": {"type": "string", "default": "*"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite a text file inside the workspace. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "create_dirs": {"type": "boolean", "default": True},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "make_directory",
                    "description": "Create a directory inside the workspace. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Run a Python script from the workspace. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}},
                            "timeout": {"type": "integer", "minimum": 1, "maximum": 1800},
                        },
                        "required": ["script"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_powershell",
                    "description": "Execute a PowerShell command in the workspace. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer", "minimum": 1, "maximum": 1800},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_matlab",
                    "description": "Run a MATLAB -batch command. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer", "minimum": 1, "maximum": 1800},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show Git branch and working tree status.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show the current unstaged Git diff for the workspace.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Stage all changes and create a Git commit. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            },
        ]

    # --------------------------------------------------------
    # Tool dispatch
    # --------------------------------------------------------
    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "list_workspace":
                return self.tool_list_workspace(**args)
            if name == "tree":
                return self.tool_tree(**args)
            if name == "read_file":
                return self.tool_read_file(**args)
            if name == "search_workspace":
                return self.tool_search(**args)
            if name == "write_file":
                return self.tool_write_file(**args)
            if name == "make_directory":
                return self.tool_make_directory(**args)
            if name == "run_python":
                return self.tool_run_python(**args)
            if name == "run_powershell":
                return self.tool_run_powershell(**args)
            if name == "run_matlab":
                return self.tool_run_matlab(**args)
            if name == "git_status":
                return self.tool_git_status()
            if name == "git_diff":
                return self.tool_git_diff()
            if name == "git_commit":
                return self.tool_git_commit(**args)
            return f"Unknown tool: {name}"
        except TypeError as exc:
            return f"Tool argument error for {name}: {exc}"
        except Exception as exc:
            return f"Tool execution error for {name}: {exc}"

    # --------------------------------------------------------
    # Usage logging
    # --------------------------------------------------------
    def record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        price = PRICE_USD_PER_1M.get(self.model) or PRICE_USD_PER_1M["deepseek-v4-flash"]
        self.estimated_cost_usd += (
            prompt_tokens / 1_000_000 * price["input"]
            + completion_tokens / 1_000_000 * price["output"]
        )

    # --------------------------------------------------------
    # Plain chat
    # --------------------------------------------------------
    def chat(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
        }
        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request["extra_body"] = {"thinking": {"type": "disabled"}}

        try:
            stream = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.messages.pop()
            return f"API ERROR: {exc}"

        parts: List[str] = []
        print("\nDeepSeek:\n")
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                print(content, end="", flush=True)
                parts.append(content)
        print("\n")

        answer = "".join(parts)
        self.messages.append({"role": "assistant", "content": answer})
        self._save_history()
        return answer

    # --------------------------------------------------------
    # Agent mode
    # --------------------------------------------------------
    def run_agent(self, task: str, max_steps: int = 30) -> str:
        self.messages.append({"role": "user", "content": task})

        tools = self.tool_definitions()
        final_answer = ""

        for step in range(1, max_steps + 1):
            print(f"\n[agent step {step}/{max_steps}]")

            request: Dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            }

            if self.thinking:
                request["reasoning_effort"] = self.reasoning_effort
                request["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                request["extra_body"] = {"thinking": {"type": "disabled"}}

            try:
                response = self.client.chat.completions.create(**request)
                self.record_usage(response)
            except Exception as exc:
                print(f"API ERROR: {exc}")
                return str(exc)

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            # Critical for DeepSeek thinking-mode tool use: preserve the
            # reasoning_content returned on the assistant message.
            assistant_message = {
                "role": "assistant",
                "content": getattr(message, "content", None),
            }

            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content

            if tool_calls:
                assistant_message["tool_calls"] = []
                for tc in tool_calls:
                    assistant_message["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

                self.messages.append(assistant_message)

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = safe_json_loads(tc.function.arguments)
                    except ValueError as exc:
                        result = str(exc)
                    else:
                        print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)})")
                        result = self.execute_tool(name, args)

                    print(textwrap.indent(result[-4_000:], "    "))
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                self._save_history()
                continue

            final_answer = getattr(message, "content", None) or ""
            print("\n" + final_answer + "\n")
            self.messages.append({"role": "assistant", "content": final_answer})
            self._save_history()
            return final_answer

        final_answer = (
            f"Agent stopped after reaching the maximum of {max_steps} steps. "
            "Review the latest tool output and continue with another /agent task."
        )
        print("\n" + final_answer + "\n")
        self.messages.append({"role": "assistant", "content": final_answer})
        self._save_history()
        return final_answer

    # --------------------------------------------------------
    # Cost/usage
    # --------------------------------------------------------
    def usage_summary(self) -> str:
        return (
            f"Prompt tokens:     {self.total_prompt_tokens:,}\n"
            f"Completion tokens: {self.total_completion_tokens:,}\n"
            f"Estimated API cost: ${self.estimated_cost_usd:.6f}\n"
            "Note: cost is an estimate from the configured reference rates; "
            "cached/peak/off-peak billing may make the actual charge differ."
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek Engineering Agent"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt/task to send",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Project root. Default: current directory",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DeepSeek model ID",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--reasoning",
        choices=["low", "high", "max"],
        default=DEFAULT_REASONING,
        help="Reasoning effort",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run the supplied prompt in tool-using agent mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum tool/agent steps",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve write/execute/commit tools",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive shell",
    )
    return parser.parse_args()


def interactive_shell(agent: Agent) -> None:
    print_banner(
        agent.model,
        agent.workspace,
        agent.thinking,
        agent.reasoning_effort,
    )

    while True:
        try:
            raw = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw == "/exit":
            break
        if raw == "/clear":
            agent.clear_history()
            print("Conversation cleared.")
            continue
        if raw == "/model":
            print(agent.model)
            continue
        if raw == "/workspace":
            print(agent.workspace)
            continue
        if raw == "/pwd":
            print(Path.cwd())
            continue
        if raw == "/tree":
            print(agent.tool_tree())
            continue
        if raw == "/tools":
            for item in agent.tool_definitions():
                print("- " + item["function"]["name"])
            continue
        if raw == "/usage":
            print(agent.usage_summary())
            continue
        if raw.startswith("/thinking"):
            pieces = raw.split()
            if len(pieces) == 1:
                agent.thinking = not agent.thinking
            else:
                agent.thinking = pieces[1].lower() in {"on", "true", "1"}
            print(f"Thinking: {'ON' if agent.thinking else 'OFF'}")
            continue
        if raw.startswith("/approve"):
            pieces = raw.split()
            if len(pieces) < 2:
                print(f"Auto-approve: {'ON' if agent.auto_approve else 'OFF'}")
            else:
                agent.auto_approve = pieces[1].lower() in {"on", "true", "1", "yes"}
                print(f"Auto-approve: {'ON' if agent.auto_approve else 'OFF'}")
            continue
        if raw == "/help":
            print(
                "\n"
                "/agent <task>  Tool-assisted task\n"
                "/chat <prompt> Plain model response\n"
                "/thinking on|off\n"
                "/approve on|off\n"
                "/tree /pwd /workspace /tools /usage\n"
                "/clear /model /exit\n"
            )
            continue

        if raw.startswith("/agent "):
            agent.run_agent(raw[len("/agent "):], max_steps=30)
        elif raw.startswith("/chat "):
            agent.chat(raw[len("/chat "):])
        else:
            # Default interactive behavior = normal chat.
            agent.chat(raw)


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    if not workspace.is_dir():
        print(f"Workspace is not a directory: {workspace}")
        return 1

    try:
        agent = Agent(
            workspace=workspace,
            model=args.model,
            thinking=not args.no_thinking,
            reasoning_effort=args.reasoning,
            auto_approve=args.auto_approve,
        )
    except Exception as exc:
        print(f"Startup error: {exc}")
        return 1

    prompt = " ".join(args.prompt).strip()

    if args.interactive or not prompt:
        interactive_shell(agent)
        return 0

    if args.agent:
        agent.run_agent(prompt, max_steps=max(1, min(args.max_steps, 100)))
    else:
        agent.chat(prompt)

    print(agent.usage_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
