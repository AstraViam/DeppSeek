"""Permission rule engine.

v1 asked `y/N` for every write and execution, with one global `--auto-approve`
escape hatch. That is the wrong shape for a 30-step refactor: the user either
answers forty prompts or turns the gate off entirely and loses it everywhere.

The model here is a short ordered rule list producing one of three decisions:

  ALLOW  run it, log it
  ASK    prompt the user, and optionally remember the answer for this session
  DENY   refuse, and report the refusal to the model as a tool result so it can
         try another approach instead of the run dying

Evaluation order, first match wins:

  1. HARD_DENY   -- not overridable by any config file. Credential reads and
                    catastrophic shell commands live here.
  2. user rules  -- from [[permissions.rules]] in config.toml.
  3. session grants -- answers the user chose to remember.
  4. autonomy preset -- the tier selected by --autonomy.
  5. fallback    -- ASK, because an unrecognised tool should never be silent.

Putting HARD_DENY above config is deliberate. A project-level config file is
repository content; a malicious or careless repo must not be able to grant itself
permission to read `~/.ssh/id_rsa` or run `Remove-Item -Recurse C:\\`.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from ..errors import ConfigError


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Tool name groups, used to keep the presets readable.
READ_TOOLS = (
    "list_workspace", "tree", "read_file", "search_workspace", "glob_files",
    "git_status", "git_diff", "git_log", "read_notebook", "todo_write",
    "check_units", "list_figures", "session_info",
)
WRITE_TOOLS = ("write_file", "edit_file", "multi_edit", "make_directory", "edit_notebook")
DESTRUCTIVE_TOOLS = ("delete_path", "move_path")
EXEC_TOOLS = (
    "run_powershell", "run_python", "run_python_snippet", "run_matlab",
    "run_tests", "matlab_workspace",
)
# Tools whose only outbound payload is a query string or a public identifier.
NETWORK_READ_TOOLS = ("web_search", "fetch_paper", "web_fetch")
# Tools that transmit workspace *content* to a third party. Distinct from the
# above because the risk is exfiltration, not egress.
NETWORK_UPLOAD_TOOLS = ("inspect_figure",)
GIT_WRITE_TOOLS = ("git_commit", "git_stage")
GIT_REMOTE_TOOLS = ("git_push",)


@dataclass(frozen=True)
class Rule:
    """One permission rule.

    All supplied predicates must match for the rule to fire. An omitted
    predicate matches anything.

    tool:    glob on the tool name, e.g. "run_*" or "write_file".
    path:    glob on the workspace-relative POSIX path the tool targets.
    command: regex searched against the command text, for shell-like tools.
    """

    decision: Decision
    tool: str = "*"
    path: str | None = None
    command: str | None = None
    reason: str = ""
    # Whether the user may be offered "remember for this session" on an ASK.
    remberable: bool = True

    def __post_init__(self) -> None:
        if self.command is not None:
            try:
                re.compile(self.command)
            except re.error as exc:
                raise ConfigError(f"Invalid command regex in rule {self!r}: {exc}") from exc

    def matches(self, request: Request) -> bool:
        if not fnmatch.fnmatchcase(request.tool, self.tool):
            return False
        if self.path is not None:
            if request.path is None:
                return False
            if not _path_matches(self.path, request.path):
                return False
        if self.command is not None:
            if not request.command:
                return False
            if not re.search(self.command, request.command, re.IGNORECASE):
                return False
        return True


@dataclass
class Request:
    """A pending tool invocation being evaluated."""

    tool: str
    # Workspace-relative POSIX path, or None when the tool targets no path.
    path: str | None = None
    # Command text for shell-like tools, used by `command` predicates.
    command: str | None = None
    # Set when the resolved path lies outside the workspace root.
    outside_workspace: bool = False
    # Human-readable summary shown at an approval prompt.
    summary: str = ""
    details: str = ""
    # Populated by the tool so the prompt can show a unified diff.
    diff: str | None = None

    def key(self) -> str:
        """Identity used for remembering a session-scoped grant.

        Keyed on the tool plus the *directory* of the target rather than the exact
        file, so approving one edit in `src/` does not re-prompt for the next file
        in the same directory, while still not silently extending to `~/.ssh`.
        """
        if self.path is None:
            return self.tool
        parent = str(PurePosixPath(self.path).parent)
        return f"{self.tool}:{parent}"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    rule: Rule | None
    source: str

    @property
    def reason(self) -> str:
        if self.rule and self.rule.reason:
            return self.rule.reason
        return f"{self.decision.value} by {self.source}"


# ---------------------------------------------------------------------------
# Hard denylist -- not overridable by config
# ---------------------------------------------------------------------------

# Credential-bearing paths. Reading these is never legitimate for a coding agent,
# and a single read would ship them to a third-party API.
HARD_DENY_PATHS = (
    # Dotenv files are where credentials live by convention. Example and
    # template variants are exempted below, since those are the files a user
    # actually wants the agent to read when wiring up configuration.
    "**/.env", "**/.env.*", "**/*.env",
    "**/.ssh/**", "**/id_rsa*", "**/id_ed25519*", "**/id_ecdsa*", "**/id_dsa*",
    "**/.aws/credentials", "**/.aws/config",
    "**/.netrc", "**/_netrc",
    "**/.gnupg/**", "**/*.pem", "**/*.pfx", "**/*.p12", "**/*.jks", "**/*.keystore",
    "**/.docker/config.json", "**/.kube/config",
    "**/.git-credentials", "**/.npmrc", "**/.pypirc",
    # The agent's own config can hold API keys for the vision/search providers.
    "**/.deppseek/credentials*",
)

# Shell commands that are unrecoverable or that disable the safety model. Phrased
# as regexes against the command text. These are denied, not asked: there is no
# engineering task for which the agent needs to repartition a disk.
HARD_DENY_COMMANDS = (
    # Recursive deletion of a drive root or the user profile.
    (r"remove-item.{0,40}-recurse.{0,40}(?:[a-z]:\\?\s*$|[a-z]:\\\*|\$env:userprofile\s*$|\$home\s*$)",
     "recursive delete of a drive root or user profile"),
    (r"\brd\b.{0,20}/s.{0,20}[a-z]:\\?\s*$", "recursive delete of a drive root"),
    (r"\bdel\b.{0,20}/[sq].{0,20}[a-z]:\\\*", "mass delete of a drive root"),
    (r"rm\s+-[a-z]*r[a-z]*f?\s+/\s*$", "recursive delete of filesystem root"),
    (r"rm\s+-[a-z]*r[a-z]*f?\s+(?:~|\$HOME)\s*$", "recursive delete of home directory"),
    # Disk and partition operations.
    (r"\b(?:format|diskpart|mkfs(?:\.\w+)?)\b", "disk formatting"),
    (r"\bclear-disk\b|\bremove-partition\b|\binitialize-disk\b", "disk partitioning"),
    (r"\bcipher\b.{0,20}/w", "free-space wiping"),
    # Defeating execution policy or fetching and running remote code.
    (r"set-executionpolicy\s+(?:unrestricted|bypass)", "disabling PowerShell execution policy"),
    (r"(?:iwr|invoke-webrequest|curl|wget).{0,120}\|\s*(?:iex|invoke-expression|powershell|bash|sh)\b",
     "piping downloaded content straight into an interpreter"),
    (r"\b(?:iex|invoke-expression)\b.{0,60}(?:downloadstring|invoke-restmethod|invoke-webrequest)",
     "executing a downloaded string"),
    # Credential harvesting.
    (r"\bmimikatz\b|\bsekurlsa\b", "credential dumping"),
    (r"reg\s+save\s+.{0,20}\bhk(?:lm|cu)\\sam\b", "dumping the SAM database"),
    (r"get-content.{0,60}\.ssh\\id_", "reading a private SSH key"),
    # Tampering with system protection.
    (r"\bvssadmin\b.{0,30}delete\s+shadows", "deleting volume shadow copies"),
    (r"\bbcdedit\b.{0,40}(?:recoveryenabled\s+no|safeboot)", "altering boot configuration"),
    (r"set-mppreference.{0,40}disablerealtimemonitoring\s*\$?true", "disabling antivirus"),
    (r"\bnetsh\b.{0,30}firewall.{0,30}(?:off|disable)", "disabling the firewall"),
    # Fork bomb / resource exhaustion.
    (r"while\s*\(\s*\$?true\s*\)\s*\{\s*start-process", "process fork bomb"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", "shell fork bomb"),
)

# Filenames that match a denied pattern but exist precisely to be read: they
# document which variables are needed without carrying their values.
HARD_DENY_EXEMPTIONS = (
    "*.example", "*.example.*", "*.sample", "*.sample.*",
    "*.template", "*.template.*", "*.dist", "*.defaults",
    "*.md", "*.rst", "*.txt",
)


def is_exempt(path: str | None) -> bool:
    """True when a path matches the denylist only incidentally."""
    if not path:
        return False
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in HARD_DENY_EXEMPTIONS)


HARD_DENY_RULES: tuple[Rule, ...] = tuple(
    Rule(
        decision=Decision.DENY,
        tool="*",
        path=pattern,
        reason=f"credential-bearing path ({pattern}) is never readable by the agent",
        remberable=False,
    )
    for pattern in HARD_DENY_PATHS
) + tuple(
    Rule(
        decision=Decision.DENY,
        tool="run_powershell",
        command=pattern,
        reason=f"blocked: {why}",
        remberable=False,
    )
    for pattern, why in HARD_DENY_COMMANDS
)


# ---------------------------------------------------------------------------
# Autonomy presets
# ---------------------------------------------------------------------------

def _group(tools: tuple[str, ...], decision: Decision, reason: str) -> list[Rule]:
    return [Rule(decision=decision, tool=t, reason=reason) for t in tools]


def preset_rules(autonomy: str) -> tuple[Rule, ...]:
    """Build the rule list for an autonomy tier."""
    rules: list[Rule] = []

    # Anything resolving outside the workspace is refused at every tier. The
    # workspace jail is the one invariant that does not scale with autonomy.
    rules.append(
        Rule(
            decision=Decision.DENY,
            tool="*",
            path="!outside",
            reason="path resolves outside the workspace root",
            remberable=False,
        )
    )

    if autonomy == "readonly":
        rules += _group(READ_TOOLS, Decision.ALLOW, "read-only tool")
        rules += _group(NETWORK_READ_TOOLS, Decision.ALLOW, "read-only lookup")
        rules.append(Rule(Decision.DENY, "*", reason="readonly autonomy: no side effects"))
        return tuple(rules)

    if autonomy == "ask":
        rules += _group(READ_TOOLS, Decision.ALLOW, "read-only tool")
        rules.append(Rule(Decision.ASK, "*", reason="ask autonomy: confirm every side effect"))
        return tuple(rules)

    if autonomy == "standard":
        rules += _group(READ_TOOLS, Decision.ALLOW, "read-only tool")
        rules += _group(NETWORK_READ_TOOLS, Decision.ALLOW, "read-only lookup")
        rules += _group(WRITE_TOOLS, Decision.ALLOW, "in-workspace write, checkpointed first")
        rules += _group(DESTRUCTIVE_TOOLS, Decision.ASK, "deletes and moves are confirmed")
        rules += _group(EXEC_TOOLS, Decision.ASK, "execution is confirmed")
        rules += _group(GIT_WRITE_TOOLS, Decision.ASK, "git history changes are confirmed")
        rules += _group(GIT_REMOTE_TOOLS, Decision.ASK, "pushing is always confirmed")
        rules += _group(NETWORK_UPLOAD_TOOLS, Decision.ASK, "uploads workspace content off-machine")
        rules.append(Rule(Decision.ASK, "*", reason="unclassified tool"))
        return tuple(rules)

    if autonomy == "autonomous":
        # "Mostly autonomous": reads, writes, and execution run unattended,
        # because the recovery path is checkpoints plus git rather than prompts.
        # What still stops is anything that is hard to undo or that leaves the
        # machine carrying workspace content.
        rules += _group(READ_TOOLS, Decision.ALLOW, "read-only tool")
        rules += _group(NETWORK_READ_TOOLS, Decision.ALLOW, "sends only a query or public id")
        rules += _group(WRITE_TOOLS, Decision.ALLOW, "in-workspace write, checkpointed first")
        rules += _group(EXEC_TOOLS, Decision.ALLOW, "execution inside the workspace")
        rules += _group(GIT_WRITE_TOOLS, Decision.ALLOW, "local git history, recoverable via reflog")

        # Destructive and outbound actions remain gated even here.
        rules += _group(DESTRUCTIVE_TOOLS, Decision.ASK, "deletes and moves are confirmed")
        rules += _group(GIT_REMOTE_TOOLS, Decision.ASK, "pushing publishes work; always confirmed")
        rules += _group(
            NETWORK_UPLOAD_TOOLS,
            Decision.ASK,
            "transmits workspace content to a third-party endpoint",
        )
        # Shell commands that are recoverable but wide-reaching still ask, even
        # though run_powershell is otherwise allowed above. These rules are
        # ordered before the blanket EXEC allow by being inserted at the front.
        gated_shell = [
            (r"\bremove-item\b.{0,60}-recurse", "recursive delete"),
            (r"\brd\b\s+/s|\brmdir\b\s+/s", "recursive directory delete"),
            (r"\bgit\b.{0,40}\bpush\b", "git push via shell"),
            (r"\bgit\b.{0,40}reset\s+--hard", "discards uncommitted work"),
            (r"\bgit\b.{0,40}\bclean\b.{0,20}-[a-z]*[fd]", "deletes untracked files"),
            (r"\bpip\b.{0,40}\buninstall\b|\bconda\b.{0,20}\bremove\b", "removes packages"),
            (r"\b(?:shutdown|restart-computer|stop-computer)\b", "shuts down the machine"),
            (r"\bstop-process\b.{0,40}-force", "force-kills processes"),
            (r"\bnew-item\b.{0,60}-itemtype\s+symboliclink", "creates a symlink that can escape the jail"),
            (r"\[environment\]::setenvironmentvariable", "changes persistent environment"),
            (r"\bset-itemproperty\b.{0,30}hk(?:lm|cu):", "writes to the registry"),
            (r"\bschtasks\b|\bregister-scheduledtask\b", "installs a scheduled task"),
            (r"\binvoke-webrequest\b|\biwr\b|\bcurl\b|\bwget\b", "outbound network request"),
        ]
        shell_rules = [
            Rule(Decision.ASK, "run_powershell", command=pattern, reason=f"confirm: {why}")
            for pattern, why in gated_shell
        ]
        # Insert shell gates ahead of the blanket allows: first match wins.
        rules = rules[:1] + shell_rules + rules[1:]

        rules.append(Rule(Decision.ASK, "*", reason="unclassified tool"))
        return tuple(rules)

    raise ConfigError(f"Unknown autonomy tier: {autonomy!r}")


def rule_from_dict(raw: dict[str, Any]) -> Rule:
    """Build a Rule from a config table, with a useful error on typos."""
    known = {"decision", "tool", "path", "command", "reason", "remberable", "rememberable"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Permission rule has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: decision, tool, path, command, reason, rememberable"
        )
    decision_raw = str(raw.get("decision", "")).lower()
    try:
        decision = Decision(decision_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Permission rule decision must be allow/ask/deny, got {decision_raw!r}"
        ) from exc
    return Rule(
        decision=decision,
        tool=str(raw.get("tool", "*")),
        path=(str(raw["path"]) if raw.get("path") is not None else None),
        command=(str(raw["command"]) if raw.get("command") is not None else None),
        reason=str(raw.get("reason", "from project config")),
        remberable=bool(raw.get("rememberable", raw.get("remberable", True))),
    )


@dataclass
class PermissionEngine:
    autonomy: str = "autonomous"
    user_rules: tuple[Rule, ...] = ()
    # Extra credential path globs from Config.secret_paths. Treated as hard
    # denies rather than config rules: the point of the list is to be a floor,
    # so a later permissive rule must not be able to lift it.
    secret_paths: tuple[str, ...] = ()
    # Session-scoped grants, keyed by Request.key().
    session_grants: dict[str, Decision] = field(default_factory=dict)
    _preset: tuple[Rule, ...] = field(init=False, repr=False)
    _secret_rules: tuple[Rule, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._preset = preset_rules(self.autonomy)
        self._secret_rules = tuple(
            Rule(
                decision=Decision.DENY,
                tool="*",
                path=pattern if pattern.startswith("**/") else f"**/{pattern}",
                reason=f"matches configured secret path pattern {pattern!r}",
                remberable=False,
            )
            for pattern in self.secret_paths
        )

    def set_autonomy(self, autonomy: str) -> None:
        self.autonomy = autonomy
        self._preset = preset_rules(autonomy)

    def grant_for_session(self, request: Request, decision: Decision) -> None:
        self.session_grants[request.key()] = decision

    def evaluate(self, request: Request) -> Verdict:
        # 1. Hard denies, including the outside-workspace invariant. Paths that
        #    match only incidentally (.env.example and friends) are exempted.
        exempt = is_exempt(request.path)
        for rule in (*HARD_DENY_RULES, *self._secret_rules):
            if rule.path is not None and exempt:
                continue
            if rule.matches(request):
                return Verdict(Decision.DENY, rule, "hard denylist")
        if request.outside_workspace:
            return Verdict(
                Decision.DENY,
                Rule(Decision.DENY, reason="path resolves outside the workspace root"),
                "workspace jail",
            )

        # 2. Project/user config rules.
        for rule in self.user_rules:
            if rule.matches(request):
                return Verdict(rule.decision, rule, "config rule")

        # 3. Grants the user chose to remember this session.
        remembered = self.session_grants.get(request.key()) or self.session_grants.get(request.tool)
        if remembered is not None:
            return Verdict(
                remembered,
                Rule(remembered, tool=request.tool, reason="remembered for this session"),
                "session grant",
            )

        # 4. Autonomy preset.
        for rule in self._preset:
            if rule.path == "!outside":
                continue  # handled above; the sentinel is not a real glob
            if rule.matches(request):
                return Verdict(rule.decision, rule, f"{self.autonomy} preset")

        # 5. Nothing matched: be conservative.
        return Verdict(
            Decision.ASK,
            Rule(Decision.ASK, reason="no rule matched"),
            "fallback",
        )

    def describe(self) -> str:
        lines = [f"Autonomy tier: {self.autonomy}", ""]
        if self.user_rules:
            lines.append("Project/user rules (highest precedence after hard denies):")
            for rule in self.user_rules:
                lines.append(f"  {_fmt_rule(rule)}")
            lines.append("")
        if self.session_grants:
            lines.append("Remembered this session:")
            for key, decision in sorted(self.session_grants.items()):
                lines.append(f"  {decision.value:5}  {key}")
            lines.append("")
        lines.append(f"Preset rules ({len(self._preset)}):")
        for rule in self._preset:
            lines.append(f"  {_fmt_rule(rule)}")
        lines.append("")
        lines.append(
            f"Hard denylist: {len(HARD_DENY_PATHS)} built-in credential path patterns, "
            f"{len(self._secret_rules)} from secret_paths config, "
            f"{len(HARD_DENY_COMMANDS)} catastrophic command patterns. Not overridable."
        )
        lines.append(
            "Exempted filenames (read normally): "
            + ", ".join(HARD_DENY_EXEMPTIONS)
        )
        return "\n".join(lines)


def _fmt_rule(rule: Rule) -> str:
    bits = [f"{rule.decision.value:5}", f"tool={rule.tool}"]
    if rule.path:
        bits.append(f"path={rule.path}")
    if rule.command:
        bits.append(f"cmd=/{rule.command}/")
    if rule.reason:
        bits.append(f"-- {rule.reason}")
    return "  ".join(bits)


def _path_matches(pattern: str, path: str) -> bool:
    """Glob a workspace-relative POSIX path.

    `PurePosixPath.full_match` (3.13+) is not available on 3.11, and fnmatch does
    not treat `**` as crossing separators, so `**/` prefixes are handled by also
    testing the pattern with that prefix stripped.
    """
    path = path.replace("\\", "/")
    # A literal "./" prefix only. str.lstrip takes a character *set*, so
    # lstrip("./") would turn ".env" into "env" and silently un-deny every
    # dotfile on the denylist.
    while path.startswith("./"):
        path = path[2:]
    path = path.removeprefix("/")
    candidates = [pattern]
    if pattern.startswith("**/"):
        candidates.append(pattern[3:])
    for cand in candidates:
        if fnmatch.fnmatch(path, cand):
            return True
        # Match a directory pattern against anything beneath it.
        if cand.endswith("/**") and fnmatch.fnmatch(path, cand[:-3]):
            return True
    # "**/x/**" should match "a/b/x/c/d": test every suffix of the path.
    if "**/" in pattern:
        tail = pattern.rsplit("**/", 1)[-1]
        parts = path.split("/")
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            if fnmatch.fnmatch(suffix, tail):
                return True
            if tail.endswith("/**") and fnmatch.fnmatch(suffix, tail[:-3]):
                return True
    return False


def workspace_relative(path: Path, workspace: Path) -> tuple[str, bool]:
    """Return a POSIX-style workspace-relative path and whether it escaped.

    Both paths are fully resolved first so that symlinks, `..` segments, and
    Windows 8.3 short names cannot be used to step outside the root.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    root = workspace.resolve()
    try:
        return resolved.relative_to(root).as_posix(), False
    except ValueError:
        return resolved.as_posix(), True
