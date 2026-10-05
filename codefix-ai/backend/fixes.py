"""Deterministic, AST-driven fix generation.

Every fixer rewrites source using AST source positions, so the output is a
predictable transformation of the input rather than a guess. Nothing here
executes user code.

A fixer returns ``None`` when it cannot handle a finding. That is a normal
outcome: the pipeline then reports the suggestion without a corrected snippet
and the UI shows "Suggested fix - validation unsuccessful" instead of claiming
a fix was applied.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import keyword
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

Fixer = Callable[[str, Optional[ast.AST], Dict[str, Any], "FixOutcome"], Optional[str]]


@dataclass
class FixOutcome:
    """The rewritten source plus a short description of what changed."""

    code: str
    applied: str = ""
    summary: str = ""


# --------------------------------------------------------------------------
# source splicing
# --------------------------------------------------------------------------


def _splice(code: str, start: Tuple[int, int], end: Tuple[int, int], replacement: str) -> str:
    """Replace the 1-based ``(line, column)`` span ``start..end`` with text.

    Everything inside the span is replaced, so lines strictly between the first
    and last line of the span disappear. ``ast`` reports ``col_offset`` as a
    UTF-8 *byte* offset, so each line is encoded before slicing.
    """
    lines = code.splitlines(keepends=True)
    start_line, start_col = start
    end_line, end_col = end
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return code

    first = lines[start_line - 1].encode("utf-8")
    last = lines[end_line - 1].encode("utf-8")

    prefix = first[: max(0, start_col)].decode("utf-8")
    suffix = last[max(0, end_col) :].decode("utf-8")

    return (
        "".join(lines[: start_line - 1])
        + prefix
        + replacement
        + suffix
        + "".join(lines[end_line:])
    )


def _span(node: ast.AST) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    end_line = getattr(node, "end_lineno", None) or node.lineno
    return (node.lineno, node.col_offset), (
        end_line,
        getattr(node, "end_col_offset", None) or node.col_offset,
    )


def _parse_or_none(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _is_input_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input"


def _is_already_converted(source: str) -> bool:
    return bool(re.search(r"\b(?:int|float)\s*\(\s*$", source.strip())) or bool(
        re.match(r"^\s*(?:int|float)\s*\(", source)
    )


# --------------------------------------------------------------------------
# semantic fixers
# --------------------------------------------------------------------------


def _is_singleton_literal(node: ast.AST) -> bool:
    """True only for ``None``, ``True`` and ``False``.

    Not ``node.value in (None, True, False)``: ``0 == False`` in Python, so
    that test reports ``count == 0`` as a singleton comparison and rewrites it
    to ``count is 0``, which changes what the program does.
    """
    return isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, bool)
    )


def _fix_input_type_mismatch(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``x = input(...)`` used against a number -> wrap the call in int()/float()."""
    if tree is None:
        return None

    convert_to = meta.get("convert_to") or "int"
    if convert_to not in ("int", "float"):
        convert_to = "int"
    var = meta.get("var_name")
    reported_line = meta.get("line")
    reported_end = meta.get("end_line") or reported_line

    # Bind input() calls to the assignment that feeds them, so the fix touches
    # the binding of `var` rather than the first input() in the file.
    targets: List[ast.Call] = []
    if var is None:
        # The reported line holds `input(...)` inline, so only a call inside the
        # reported expression is part of this finding. Converting every input()
        # in the file would change code the user never asked about. The range is
        # a line span because the comparison may wrap across lines.
        targets = [
            node
            for node in ast.walk(tree)
            if _is_input_call(node)
            and reported_line is not None
            and reported_line <= node.lineno <= (reported_end or reported_line)
        ]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_input_call(node.value):
                if any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
                    targets.append(node.value)

    if not targets:
        return None

    rewritten = code
    changed = False
    # Bottom-up so earlier offsets remain valid while splicing.
    for call in sorted(targets, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        source = ast.get_source_segment(code, call)
        if not source or _is_already_converted(source):
            continue
        candidate = _splice(rewritten, _span(call)[0], _span(call)[1], f"{convert_to}({source})")
        if candidate == rewritten:
            continue
        rewritten = candidate
        changed = True

    if not changed:
        return None

    outcome.applied = "input-type-mismatch"
    outcome.summary = (
        f"Wrapped the input in `{convert_to}()` so the comparison with a number is valid."
    )
    return rewritten


def _fix_singleton_comparison(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``x == None`` -> ``x is None``, ``None == x`` -> ``None is x``, and the
    True/False equivalents.

    Splicing works on the gap between two comparison operands rather than on the
    operator node itself: on Python 3.10 ``ast.Eq`` and friends carry no
    ``col_offset``, so their position cannot be used. The gap is the same
    whichever operand holds the singleton.
    """
    if tree is None:
        return None

    target_line = meta.get("line")
    edits: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not node.ops:
            continue
        if target_line is not None and node.lineno != target_line:
            continue

        operands = [node.left, *node.comparators]
        for index, op in enumerate(node.ops):
            left_operand = operands[index]
            right_operand = operands[index + 1]
            if not (
                _is_singleton_literal(right_operand) or _is_singleton_literal(left_operand)
            ):
                continue

            replacement = " is " if isinstance(op, ast.Eq) else " is not "
            start = (
                getattr(left_operand, "end_lineno", None) or left_operand.lineno,
                getattr(left_operand, "end_col_offset", None) or left_operand.col_offset,
            )
            end = (right_operand.lineno, right_operand.col_offset)
            edits.append((start, end, replacement))

    if not edits:
        return None

    rewritten = code
    for start, end, text in sorted(edits, key=lambda item: item[0], reverse=True):
        candidate = _splice(rewritten, start, end, text)
        if candidate != rewritten:
            rewritten = candidate

    if rewritten == code:
        return None
    outcome.applied = "singleton-comparison"
    outcome.summary = "Used `is` / `is not` when comparing against None, True or False."
    return rewritten


def _fix_unused_import(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """Delete an import statement Pylint reported as unused."""
    if meta.get("symbol") != "unused-import" or tree is None:
        return None
    line = meta.get("line")

    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if line is not None and not (node.lineno <= line <= (node.end_lineno or node.lineno)):
            continue
        names = ", ".join(alias.name for alias in node.names)
        outcome.applied = "unused-import"
        outcome.summary = f"Removed the unused import `{names}`."
        return "".join(lines[: node.lineno - 1] + lines[node.end_lineno :])
    return None


def _fix_unused_variable(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """Delete an unused assignment whose right-hand side cannot have side effects."""
    if meta.get("symbol") != "unused-variable" or tree is None:
        return None
    line = meta.get("line")

    lines = code.splitlines(keepends=True)
    pure = (ast.Constant, ast.Name, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.UnaryOp)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or node.lineno != line:
            continue
        if not all(isinstance(target, ast.Name) for target in node.targets):
            continue
        if not isinstance(node.value, pure):
            continue
        if node.lineno != node.end_lineno:
            continue

        index = node.lineno - 1
        end = index + 1
        if end < len(lines) and not lines[end].strip():
            end += 1
        outcome.applied = "unused-variable"
        outcome.summary = f"Removed the unused variable `{node.targets[0].id}`."
        return "".join(lines[:index] + lines[end:])
    return None


TYPO_CUTOFF = 0.75
TYPO_MARGIN = 0.05

# Builtins are a pool of several hundred short, unrelated names, so a merely
# similar one is far more likely to be a coincidence than a typo: `age` scores
# 0.75 against `range`, which would rewrite `print(age)` into `print(range)`.
BUILTIN_TYPO_CUTOFF = 0.8


def _fix_undefined_variable(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """Repair a likely typo, but only when one close name stands out.

    Three things have to hold, otherwise the "fix" is a guess that changes what
    the program does: the name must be genuinely undefined, the best match must
    be unambiguous, and the replacement must not already be the call that
    encloses the name (renaming an argument to its own callee nests the call).
    """
    if meta.get("symbol") != "undefined-variable" or tree is None:
        return None

    defined_here: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_here.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_here.append(node.name)
        elif isinstance(node, ast.arg):
            defined_here.append(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined_here.append(alias.asname or alias.name.split(".")[0])

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])

    # Name of the innermost call each node sits inside, so a candidate that is
    # already the enclosing callee can be rejected. BFS order means the
    # outermost call is recorded first, which is the one that matters.
    enclosing_call: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            owner = ast.unparse(node.func)
            for child in ast.walk(node):
                if child is not node.func:
                    enclosing_call.setdefault(id(child), owner)

    undefined: List[Tuple[str, int, List[ast.Name]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        name = node.id
        if name in defined_here or name in imported or keyword.iskeyword(name):
            continue
        if name.startswith("__") or hasattr(builtins, name):
            continue
        undefined.append((name, node.lineno, [node]))

    if not undefined:
        return None

    # Pylint reports the line of the use, so repair that name. Anything else in
    # the file is a different finding with its own turn through this pipeline.
    reported_line = meta.get("line")
    on_reported_line = [item for item in undefined if item[1] == reported_line]
    name = min(on_reported_line or undefined, key=lambda item: item[1])[0]

    # Names bound in this file are the trustworthy targets. Builtins are only a
    # fallback, and only for a near-anagram.
    in_file = [
        candidate
        for candidate in dict.fromkeys(defined_here)
        if not keyword.iskeyword(candidate) and candidate != name
    ]
    builtins_pool = [
        candidate
        for candidate in dir(builtins)
        if not candidate.startswith("_")
        and hasattr(builtins, candidate)
        and candidate != name
    ]

    occurrences = [node for item in undefined for node in item[2] if item[0] == name]

    for pool, cutoff in ((in_file, TYPO_CUTOFF), (builtins_pool, BUILTIN_TYPO_CUTOFF)):
        matches = difflib.get_close_matches(name, pool, n=2, cutoff=cutoff)
        if not matches:
            continue
        best = difflib.SequenceMatcher(None, name, matches[0]).ratio()
        runner_up = (
            difflib.SequenceMatcher(None, name, matches[1]).ratio() if len(matches) > 1 else 0.0
        )
        if best - runner_up < TYPO_MARGIN:
            continue
        if abs(len(name) - len(matches[0])) > 2:
            continue
        replacement = matches[0]
        # `print(prnt('hi'))` must not become `print(print('hi'))`.
        if any(enclosing_call.get(id(node)) == replacement for node in occurrences):
            continue

        pattern = re.compile(rf"(?<![\w.]){re.escape(name)}\b")
        rewritten, count = pattern.subn(replacement, code)
        if not count or rewritten == code:
            continue
        outcome.applied = "undefined-variable"
        outcome.summary = f"Replaced `{name}` with `{replacement}` (likely typo)."
        return rewritten

    return None


# --------------------------------------------------------------------------
# syntax fixers (these run when the source does not parse at all)
# --------------------------------------------------------------------------


_HEADER_LINE_IN_MESSAGE = re.compile(r"on line (\d+)")


def _fix_syntax_expected_colon(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``if x > 3`` -> ``if x > 3:`` when the parser asks for a colon."""
    line = meta.get("line")
    msg = (meta.get("syntax_msg") or "").lower()
    if line is None or "expected ':'" not in msg:
        return None

    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None
    body = lines[line - 1].rstrip("\r\n")
    if body.rstrip().endswith(":"):
        return None

    outcome.applied = "expected-colon"
    outcome.summary = f"Added the missing `:` at the end of line {line}."
    return (
        "".join(lines[: line - 1])
        + body
        + ":"
        + lines[line - 1][len(body) :]
        + "".join(lines[line:])
    )


def _fix_syntax_missing_indent(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``def f():`` with no indented body -> insert an indented ``pass``.

    CPython reports the *offending* line (the unindented statement) while the
    block header sits on the line named in the message, so the insertion point
    comes from the message, not from ``line``.
    """
    line = meta.get("line")
    msg = (meta.get("syntax_msg") or "")
    if line is None or "expected an indented block" not in msg.lower():
        return None

    lines = code.splitlines(keepends=True)
    header_line = line
    match = _HEADER_LINE_IN_MESSAGE.search(msg)
    if match:
        header_line = int(match.group(1))
    if not 1 <= header_line <= len(lines):
        return None

    header = lines[header_line - 1]
    body = header.rstrip("\r\n")
    if not body.rstrip().endswith(":"):
        return None
    # Safe only when the next line is blank or unindented. If it is already
    # indented then something else is wrong and inserting `pass` would hide it.
    following = lines[header_line] if header_line < len(lines) else ""
    if following.strip() and re.match(r"[ \t]", following):
        return None

    indent = re.match(r"[ \t]*", body).group(0)
    inserted = f"{indent}    pass\n"
    if header_line == len(lines) and not header.endswith("\n"):
        inserted = "\n" + inserted

    outcome.applied = "missing-indent"
    outcome.summary = (
        f"Line {header_line} opens a block, so an indented `pass` was added to keep the file valid."
    )
    return "".join(lines[:header_line]) + inserted + "".join(lines[header_line:])


_PY2_PRINT = re.compile(r"^(\s*)print\s+(.+?)\s*$")


def _fix_syntax_py2_print(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``print "hi"`` -> ``print("hi")`` (Python 2 syntax in a Python 3 file)."""
    line = meta.get("line")
    msg = (meta.get("syntax_msg") or "").lower()
    if line is None or "missing parentheses in call to 'print'" not in msg:
        return None

    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None

    trailing = lines[line - 1][len(lines[line - 1].rstrip("\r\n")) :]
    match = _PY2_PRINT.match(lines[line - 1].rstrip("\r\n"))
    if not match:
        return None

    indent, argument = match.group(1), match.group(2)
    outcome.applied = "py2-print"
    outcome.summary = f"Converted the Python 2 `print` statement on line {line} to a function call."
    return (
        "".join(lines[: line - 1])
        + f"{indent}print({argument})"
        + trailing
        + "".join(lines[line:])
    )


Predicate = Callable[[Dict[str, Any]], bool]

# Each fixer states which findings it is allowed to touch. Without this gate a
# fixer would happily rewrite code that has no defect at all, because most of
# them key off AST shape rather than the reported error.
SEMANTIC_FIXERS: List[Tuple[Predicate, Fixer]] = [
    (
        lambda context: context.get("error_type") == "TypeError" and "convert_to" in context,
        _fix_input_type_mismatch,
    ),
    (
        lambda context: context.get("error_type") == "SingletonComparison",
        _fix_singleton_comparison,
    ),
    (lambda context: context.get("error_type") == "unused-import", _fix_unused_import),
    (lambda context: context.get("error_type") == "unused-variable", _fix_unused_variable),
    (lambda context: context.get("error_type") == "undefined-variable", _fix_undefined_variable),
]

SYNTAX_FIXERS: List[Tuple[Predicate, Fixer]] = [
    (lambda context: "expected ':'" in (context.get("syntax_msg") or "").lower(), _fix_syntax_expected_colon),
    (
        lambda context: "expected an indented block" in (context.get("syntax_msg") or "").lower(),
        _fix_syntax_missing_indent,
    ),
    (
        lambda context: "missing parentheses in call to 'print'" in (context.get("syntax_msg") or "").lower(),
        _fix_syntax_py2_print,
    ),
]


def propose_fix(code: str, analysis: Dict[str, Any]) -> Optional[FixOutcome]:
    """Try every applicable fixer and return the first one that changes the source."""
    if not analysis.get("has_error"):
        return None

    # The issue's own metadata (symbol, convert_to, ...) is merged in so a fixer
    # can read it without digging into a nested dict.
    context: Dict[str, Any] = {**analysis, **(analysis.get("meta") or {})}

    if analysis.get("is_syntax_error"):
        candidates = SYNTAX_FIXERS
        tree = None
    else:
        candidates = SEMANTIC_FIXERS
        tree = _parse_or_none(code)
        if tree is None:
            return None

    for applies, fixer in candidates:
        try:
            if not applies(context):
                continue
        except Exception:
            continue
        outcome = FixOutcome(code=code)
        try:
            candidate = fixer(code, tree, context, outcome)
        except Exception:  # a fixer bug must not break analysis
            continue
        if candidate and candidate != code:
            outcome.code = candidate
            return outcome
    return None


def diff_summary(original: str, corrected: str, context: int = 1) -> List[Dict[str, Any]]:
    """Line-level change rows for the UI.

    Emits ``{"kind": "added"|"removed"|"context", "line": int, "text": str}``
    with line numbers taken from the corrected file where it exists, so the
    frontend can render a diff without parsing difflib markers itself.
    """
    kind_map = {"+": "added", "-": "removed", " ": "context"}

    original_lines = original.splitlines()
    corrected_lines = corrected.splitlines()

    rows: List[Dict[str, Any]] = []
    old_no = 0
    new_no = 0

    for row in difflib.ndiff(original_lines, corrected_lines):
        marker, text = row[:1], row[2:]
        kind = kind_map.get(marker, "context")
        if marker == " ":
            old_no += 1
            new_no += 1
            line = new_no
        elif marker == "-":
            old_no += 1
            line = old_no
        else:
            new_no += 1
            line = new_no

        rows.append({"kind": kind, "line": line, "text": text})

        if len(rows) >= 200:
            rows.append({"kind": "context", "line": new_no, "text": "… diff truncated"})
            break

    return rows