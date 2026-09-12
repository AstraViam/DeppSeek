# DeppSeek

An approval-gated DeepSeek engineering agent for Windows and PowerShell, built
for scientific computing: CFD, thermodynamics, batteries, process safety, and
MATLAB work alongside ordinary Python.

```powershell
.\setup.ps1
$env:DEEPSEEK_API_KEY = "sk-your-key"
.\deppseek.cmd --doctor
.\deppseek.cmd "why does case3 diverge after 200 steps?"
```

## What it does

Reads and searches your project, edits code surgically, runs Python, PowerShell,
pytest and MATLAB, inspects notebooks and figures, checks dimensions, looks up
papers, and commits. Every action passes a permission check first, and every file
change is checkpointed so it can be taken back.

Run `--doctor` before anything else. It reports what is missing and what each
gap costs you.

## Install

`setup.ps1` picks an interpreter, builds a virtual environment, installs the
package, and finds MATLAB.

```powershell
.\setup.ps1              # core install
.\setup.ps1 -Full        # plus dashboard, unit checking, notebooks, MCP
.\setup.ps1 -Python "C:\Path\To\python.exe"
```

It prefers a Python the MATLAB Engine API can bind to, for the reason in
[MATLAB](#matlab) below. To use the command from anywhere, add the project
directory to your PATH; the last lines of the setup output give the command.

### Standalone single file

For a machine where you will not run an install:

```powershell
python scripts\bundle.py --check
python dist\deppseek.py --doctor
```

The bundle embeds every module and serves them through a custom importer, so
relative imports, package structure, and traceback line numbers behave exactly
as they do in the installed package. Only `openai` is required at runtime; the
rest degrade gracefully.

## Autonomy

Four tiers, set with `--autonomy` or in config. The default is `autonomous`.

| Tier | Reads | Writes | Execution | Deletes, pushes, uploads |
|---|---|---|---|---|
| `readonly` | run | denied | denied | denied |
| `ask` | run | confirm | confirm | confirm |
| `standard` | run | run | confirm | confirm |
| `autonomous` | run | run | run | confirm |

Under `autonomous` the recovery path is checkpoints and git rather than prompts,
which is what makes a forty-step refactor practical. What still stops: deleting
or moving files, pushing, uploading a figure to a vision endpoint, and fourteen
shell patterns that are wide-reaching but legitimate, such as `Remove-Item
-Recurse`, `git reset --hard`, package removal, and registry writes.

Some things never run, at any tier, and no config file can enable them: reading
credential files (`.env`, `.ssh/`, `*.pem`, AWS and Docker credentials), and
catastrophic commands (formatting a disk, `Set-ExecutionPolicy Bypass`, piping a
download into an interpreter, deleting shadow copies, disabling the firewall).
The denylist sits above project config deliberately, because a project config is
repository content and must not be able to grant itself credential access.

Approval prompts show the actual diff or command, and offer "always this
session" so a long task does not become forty questions.

### Undo

```
/undo              revert the last change the agent made
/checkpoints       list recent checkpoints
/undo cp0007-...   revert a specific one
```

Checkpoints are content-addressed and independent of git, so they work in a
workspace that is not a repository or that has unrelated uncommitted work.

## MATLAB

Two execution paths, chosen automatically. `/matlab` says which is active.

**Warm engine.** When `matlab.engine` imports, one MATLAB session starts and is
kept. Startup is paid once and variables persist across tool calls, so you can
load a mesh, inspect it, and operate on it across separate steps.

**Batch.** Otherwise each call runs `matlab -batch`, but the base workspace is
saved to a `.mat` afterwards and reloaded before the next call. Variables still
persist; you lose startup amortisation and figure-handle continuity.

The engine's availability is an interpreter-version question, not a preference.
MathWorks pins each engine release to one MATLAB release *and* caps the Python
version: for MATLAB R2026a the engine is `matlabengine` 26.1.x, which declares
`python_requires ">=3.9, <3.14"`. On Python 3.14 it cannot be installed at all.

```powershell
python --version
python -c "import matlab.engine; print('warm engine OK')"
```

If your Python is 3.14 or newer and you want the warm workspace, install Python
3.13 and re-run `setup.ps1`; it will pick the compatible interpreter for the
virtual environment and leave your system Python alone.

Open figures are exported to PNG under `.deppseek\figures\` and closed after
every MATLAB call, and every command is journalled to
`.deppseek\matlab\journal.m` as a reproducibility record.

## Cost

DeepSeek bills prompt tokens that hit its context cache at roughly 2% of the
cache-miss rate, and rates differ by 2x between peak and off-peak hours. An
agent loop resends a nearly identical prefix every step and therefore hits cache
constantly, so ignoring the cache overstates a step's cost several-fold.

`/cost` breaks out cache-hit, cache-miss, output and reasoning tokens, and shows
the cache hit rate. A low hit rate usually means something is changing the
request prefix between steps.

Runs stop at a cost ceiling, `$5` by default. Set it with `--max-cost` or in
config.

## Context

The conversation is token-budgeted. When it approaches the soft limit, the
oldest turns are summarised and dropped. `/context` shows usage and the token
estimator's calibration; `/compact` compacts on demand.

Tool results larger than the configured cap are written to `.deppseek\output\`
and replaced in context with a pointer plus head and tail excerpts, so one large
file read cannot dominate the rest of the session.

## Commands

Type `/help` in the shell. The ones worth knowing:

```
/doctor        check the environment
/cost          token usage and estimated spend
/context       context usage and estimator state
/undo          revert the last change
/permissions   show the active rules
/autonomy T    change tier for this session
/matlab        which MATLAB path is active, and why
/plan          the agent's current plan
/sessions      list saved sessions
/resume ID     continue one
/branch        fork the current session
```

Type `@` to complete a workspace file path. Alt+Enter inserts a newline.

## Interfaces

Inline by default: a scrolling transcript that keeps native scrollback, copy and
paste, and output redirection. `--tui` opens a full-screen dashboard showing the
plan, token and cost meters, tool log, and transcript at once, which is worth
the trade on a long unattended run. It needs `textual`, from `setup.ps1 -Full`.

## Project configuration

Two files, both optional:

- `.deppseek\config.toml` — model, autonomy, budgets, MCP servers, permission
  rules. Copy `.deppseek\config.toml.example` to start. See that file for every
  option.
- `DEPPSEEK.md` — free-text project notes loaded into the system prompt. Use it
  for conventions you would otherwise repeat every session: which solver to use,
  how to run the suite, which directories are generated.

Resolution order, lowest first: defaults, user config at
`%USERPROFILE%\.deppseek\config.toml`, project config, environment variables,
command-line flags.

## Extending

**MCP servers** are configured under `[mcp.servers]` and appear as
`mcp__<name>__<tool>`. They ask for confirmation once per session, because an
MCP server is third-party code that can do anything. A `[[permissions.rules]]`
entry can trust one server outright.

**Delegation** (`delegate`) runs a self-contained investigation in a separate
conversation and returns only its findings. It exists for context economy: a
search that takes fifteen tool calls and fifty thousand tokens of file excerpts
should not leave all of that in the main thread. The sub-agent shares the
parent's permission engine and cannot write files.

**Figure inspection** needs an OpenAI-compatible vision endpoint, since
DeepSeek's text models are not multimodal. Without one configured, the agent is
told to say it did not look at the plot rather than inventing a description.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy
python scripts\bundle.py --check
```

Tests run without network access or an API key; the provider is scripted and the
MCP client is exercised against a real subprocess server.

## Upgrading from the single-file version

The old `deepseek_engineering_agent.py` still works but is superseded. Two
things need attention:

- Its default model, `deepseek-v4-flash`, was retired on 10 September 2026.
  DeppSeek rewrites retired model IDs to `deepseek-flash` and says so.
- Its history file, `.deepseek_agent_history.json`, is not read. Sessions now
  live in `.deppseek\sessions\`, outside the tree the agent searches, so it can
  no longer read its own history as project content.

`--autonomy ask` reproduces the old confirm-everything behaviour.
