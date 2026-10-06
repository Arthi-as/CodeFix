"""Validation for suggested corrections.

A correction is only reported as successful when it passes every gate:

1. it parses (``ast.parse``);
2. it compiles to a code object (``compile``), which also catches things the
   parser accepts but the compiler rejects, such as ``return`` outside a
   function;
3. the original finding is gone when the analyzer runs over it again;
4. it is not a no-op.

If a gate fails the caller must present the result as a suggestion, never as a
fix. Nothing in this module executes user code.

Sandboxing note
---------------
Executing a snippet to prove a fix works needs more than an AST check, and it is
deliberately not implemented here. When execution is added it must run in an
isolated sandbox/container with CPU, memory, wall-clock, filesystem and network
limits (``--network=none``, read-only root, dropped capabilities, seccomp, and
either a container runtime or a microVM). Running ``exec`` in a worker thread
inside the API process is not a sandbox: ``os``, ``sys`` and ``ctypes`` remain
reachable, and a fork bomb or a hung loop takes the API down with it.
``safe_execute`` documents the required contract and refuses to run by default.
"""

from __future__ import annotations

import ast
import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_LINE_LENGTH = 500
MAX_CODE_CHARS = 50_000
MAX_DEPTH = 60

# Modules that reach outside the process. Reported, never executed.
UNSAFE_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "urllib",
    "http", "requests", "ftplib", "telnetlib", "smtplib", "pickle", "marshal",
    "shelve", "ctypes", "multiprocessing", "threading", "importlib", "runpy",
    "pty", "signal", "resource", "webbrowser",
}

# Calls that matter only if code is ever executed inside a sandbox.
UNSAFE_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "exit", "quit"}

# Interactive builtins are expected in a teaching tool and are not unsafe.
INTERACTIVE_CALLS = {"input", "open"}


@dataclass
class ValidationReport:
    """Outcome of validating one candidate correction."""

    validated: bool = False
    syntax_valid: bool = False
    compiles: bool = False
    changed: bool = False
    issue_resolved: Optional[bool] = None
    local: Optional[bool] = None
    edit_lines: Optional[int] = None
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validated": self.validated,
            "syntax_valid": self.syntax_valid,
            "compiles": self.compiles,
            "changed": self.changed,
            "issue_resolved": self.issue_resolved,
            "local": self.local,
            "edit_lines": self.edit_lines,
            "reason": self.reason,
            "warnings": self.warnings,
        }



def check_syntax(code: str) -> tuple[bool, str]:
    """``ast.parse`` with the interpreter's own message."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"line {exc.lineno}: {exc.msg}"
    except (ValueError, MemoryError, RecursionError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_compiles(code: str) -> tuple[bool, str]:
    """Compile without executing, to catch parser-accepted but invalid code."""
    try:
        compile(code, "<candidate>", "exec", dont_inherit=True)
        return True, ""
    except SyntaxError as exc:
        return False, f"line {exc.lineno}: {exc.msg}"
    except (ValueError, MemoryError, RecursionError, OverflowError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_structure(code: str) -> tuple[bool, str]:
    """Cheap sanity limits before anything else looks at the candidate."""
    if not code.strip():
        return False, "The corrected code is empty."
    if len(code) > MAX_CODE_CHARS:
        return False, f"The corrected code is larger than {MAX_CODE_CHARS} characters."
    for index, line in enumerate(code.splitlines(), start=1):
        if len(line) > MAX_LINE_LENGTH:
            return False, f"Line {index} is longer than {MAX_LINE_LENGTH} characters."
    return True, ""


def collect_warnings(code: str) -> List[str]:
    """Non-blocking notes about the candidate: risky imports and calls."""
    warnings: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return warnings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in UNSAFE_MODULES:
                    warnings.append(
                        f"Line {node.lineno} imports `{alias.name}`, which can touch the "
                        "filesystem, network or process."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in UNSAFE_MODULES:
                warnings.append(
                    f"Line {node.lineno} imports from `{node.module}`, which can touch the "
                    "filesystem, network or process."
                )
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in UNSAFE_CALLS:
                warnings.append(
                    f"Line {node.lineno} calls `{name}()`, which executes code at runtime."
                )
            elif name == "open":
                warnings.append(
                    f"Line {node.lineno} calls `open()`, which reads or writes files."
                )
    return warnings


def _finding_key(error_type: str, message: str) -> tuple:
    """Identity of a finding that survives a rewrite of the source.

    The line number cannot be part of it: deleting an unused import shifts every
    later line up by one, so the *next* unused import then sits on the line the
    first one used to occupy and the old check reported "still contains the
    original problem" for a fix that was correct.
    """
    return (error_type, " ".join((message or "").split()))


def original_issue_gone(code: str, analysis: Dict[str, Any], *, trusted: bool = True) -> Optional[bool]:
    """Re-run static analysis and report whether the finding was resolved.

    ``None`` means the check could not run (for example the finding has no
    signature to compare), in which case the caller must not claim the fix
    resolved the bug.

    ``trusted=False`` is for candidates written by the language model. It adds
    the requirement that the rewrite introduces no finding that was not there
    before, because model output routinely silences the message it was shown
    while leaving the same class of bug behind: ``print(age)`` rewritten as
    ``defprint(name)`` still has an undefined name, and passes a "did the
    original message disappear" test while being nonsense.
    """
    error_type = analysis.get("error_type")
    line = analysis.get("line")
    if not error_type or line is None:
        return None

    from analyzer import find_issues  # local import: avoids an import cycle

    try:
        remaining = find_issues(code)
    except Exception:
        return None

    before = Counter(
        _finding_key(issue.get("error_type", ""), issue.get("message", ""))
        for issue in (analysis.get("all_issues") or [])
    )
    after = Counter(_finding_key(issue.error_type, issue.message) for issue in remaining)
    key = _finding_key(error_type, analysis.get("message") or "")

    # A finding of the same type still reported on the same line is the original
    # problem, unless it is a *different* finding that the edit moved onto this
    # line - which is what happens to the next unused import once the first one
    # is deleted.
    on_line = [
        issue for issue in remaining if issue.error_type == error_type and issue.line == line
    ]
    if on_line:
        still_reported = any(_finding_key(i.error_type, i.message) == key for i in on_line)
        shifted_onto_this_line = after[key] < before.get(key, 0)
        if still_reported or not shifted_onto_this_line:
            return False

    if not trusted:
        for entry in after:
            if entry not in before:
                return False

    return True


def changed_line_numbers(original: str, corrected: str) -> List[int]:
    """1-based line numbers of the original that the candidate changed."""
    before = original.splitlines()
    after = corrected.splitlines()
    changed: List[int] = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        span = list(range(i1 + 1, max(i2, i1 + 1) + 1))
        if not span:
            # A pure insertion or truncation changes no original line, but it
            # still changes the program; charge the line it happened next to.
            span = [max(1, min(i1 + 1, len(before)))]
        changed.extend(span)
    return sorted(set(changed))


# A rewrite that touches three lines next to the error is a fix. One that
# rewrites the file is a different program, whatever the gates say about it.
MAX_UNTRUSTED_CHANGED_LINES = 3
UNTRUSTED_LINE_RADIUS = 2


def rewrite_is_local(
    original: str, corrected: str, line: Optional[int]
) -> Optional[str]:
    """``None`` when the candidate stays near the reported line, else a reason.

    This is the gate that matters most for model output. Every mechanical check
    can be passed by a rewrite that is simply not the same program: asked to fix
    ``print(age)`` the previous checkpoint returned ``print(sys.argv[1])``, which
    parses, compiles, has no findings and changes one line. Only the distance
    from the reported line gives that away.
    """
    before = original.splitlines()
    after = corrected.splitlines()
    if len(after) != len(before):
        # Editing a line is a fix. Adding or removing one is a different program,
        # and the commonest form of it is the model inventing the value it says
        # is missing (`print(age)` -> `age = 10`).
        return (
            f"The corrected code has {len(after)} lines instead of {len(before)}, "
            "so it adds or removes code rather than repairing the reported line."
        )

    changed = changed_line_numbers(original, corrected)
    if not changed:
        return "The corrected code is identical to the original."

    if len(changed) > MAX_UNTRUSTED_CHANGED_LINES:
        return (
            f"The corrected code rewrites {len(changed)} lines (line"
            f"{'' if len(changed) == 1 else 's'} {', '.join(str(n) for n in changed[:6])}), "
            "which is a different program rather than a fix."
        )
    if line is None:
        return None
    far = [n for n in changed if abs(n - line) > UNTRUSTED_LINE_RADIUS]
    if far:
        return (
            f"The corrected code changes line {', '.join(str(n) for n in far)}, "
            f"which is nowhere near the reported line {line}."
        )
    return None


def validate_correction(
    original: str,
    corrected: str,
    analysis: Dict[str, Any],
    *,
    trusted: bool = True,
) -> ValidationReport:
    """Run every gate and decide whether ``corrected`` may be called a fix.

    ``trusted=False`` marks a candidate produced by the language model rather
    than by :mod:`fixes`, which additionally has to survive
    :func:`original_issue_gone`'s "introduces nothing new" gate.
    """
    report = ValidationReport()

    if not corrected or corrected.strip() == original.strip():
        report.reason = "No correction was produced, so there is nothing to validate."
        return report
    report.changed = True
    # How big the edit is, measured rather than claimed: a repair that touches
    # one line is what "minimal modification" means, and this number is what the
    # panel shows for it.
    report.edit_lines = len(changed_line_numbers(original, corrected))

    ok, detail = check_structure(corrected)
    if not ok:
        report.reason = detail
        return report

    ok, detail = check_syntax(corrected)
    report.syntax_valid = ok
    if not ok:
        report.reason = f"The corrected code does not parse: {detail}."
        return report

    ok, detail = check_compiles(corrected)
    report.compiles = ok
    if not ok:
        report.reason = f"The corrected code does not compile: {detail}."
        return report

    report.warnings = collect_warnings(corrected)

    if not trusted:
        problem = rewrite_is_local(original, corrected, analysis.get("line"))
        report.local = problem is None
        if problem:
            report.reason = problem
            return report

    resolved = original_issue_gone(corrected, analysis, trusted=trusted)
    report.issue_resolved = resolved
    if resolved is False:
        report.reason = (
            "The corrected code still contains the original problem."
            if trusted
            else "The corrected code silences the finding but leaves the same kind of problem, "
            "or introduces a new one."
        )
        return report

    report.validated = True
    report.reason = (
        "The corrected code parses, compiles, and the original finding no longer reports."
        if resolved
        else "The corrected code parses and compiles."
    )
    return report


def validate_code(code: str) -> Dict[str, Any]:
    """Standalone syntax/safety check, used when there is no finding to resolve."""
    report = ValidationReport()
    report.changed = True

    ok, detail = check_structure(code)
    if not ok:
        report.reason = detail
        return report.to_dict()

    ok, detail = check_syntax(code)
    report.syntax_valid = ok
    if not ok:
        report.reason = f"The code does not parse: {detail}."
        return report.to_dict()

    ok, detail = check_compiles(code)
    report.compiles = ok
    if not ok:
        report.reason = f"The code does not compile: {detail}."
        return report.to_dict()

    report.warnings = collect_warnings(code)
    report.validated = True
    # There is no finding to clear here, so treat it as resolved by definition.
    report.issue_resolved = True
    report.reason = "The code parses and compiles."
    return report.to_dict()


def safe_execute(
    code: str,
    *,
    enabled: bool = False,
    timeout: float = 3.0,
    memory_mb: int = 256,
) -> Dict[str, Any]:
    """Placeholder contract for sandboxed execution. Never runs by default.

    Returns ``{"executed": False, ...}`` unless a real sandbox is configured.
    See the module docstring for the isolation this would require.
    """
    if not enabled:
        return {
            "executed": False,
            "ok": False,
            "reason": (
                "Execution is disabled. Static validation only - see the sandboxing note "
                "in validator.py for what enabling this safely requires."
            ),
            "timeout": timeout,
            "memory_mb": memory_mb,
        }
    return {
        "executed": False,
        "ok": False,
        "reason": (
            "No sandbox backend is configured. Wire this to a container or microVM runner "
            "with CPU, memory, wall-clock, filesystem and network limits before setting "
            "enabled=True."
        ),
        "timeout": timeout,
        "memory_mb": memory_mb,
    }