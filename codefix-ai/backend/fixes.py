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


def _reproducible_number(text: str, converter: Callable[[str], Any]) -> bool:
    """True when ``str(converter(text)) == text``, e.g. ``'1'`` for ``int``."""
    stripped = text.strip()
    if not stripped:
        return False
    try:
        return str(converter(stripped)) == stripped
    except ValueError:
        return False


def _string_only_use(
    node: ast.AST, name: str, reported_line: Optional[int], converter: Callable[[str], Any]
) -> bool:
    """True when ``name`` is used somewhere that only makes sense as a ``str``.

    Wrapping ``choice = input('1/2/3? ')`` in ``int()`` fixes ``choice > 2`` and
    silently breaks ``choice == 'y'`` three lines above, so the conversion cannot
    be offered while such a use exists. A quoted number that the converter
    reproduces exactly is not such a use: it is rewritten to a bare number, which
    matches every value the quoted form matched.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == name
    if isinstance(node, ast.Compare) and reported_line is not None:
        if node.lineno == reported_line:
            return False  # the comparison this finding is about
        operands = [node.left, *node.comparators]
        for i, operand in enumerate(operands):
            if not (isinstance(operand, ast.Name) and operand.id == name):
                continue
            for other in operands[:i] + operands[i + 1 :]:
                if not (isinstance(other, ast.Constant) and isinstance(other.value, str)):
                    continue
                if not _reproducible_number(other.value, converter):
                    return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if isinstance(node.left, ast.Name) and node.left.id == name:
            return isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
        if isinstance(node.right, ast.Name) and node.right.id == name:
            return isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
    return False


def _numeric_literal_to_rewrite(
    node: ast.AST, name: str, converter: Callable[[str], Any]
) -> Optional[ast.Constant]:
    """A quoted number compared to ``name`` that ``converter`` reproduces exactly."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return None
    if not _reproducible_number(node.value, converter):
        return None
    parent = getattr(node, "parent", None)
    if not isinstance(parent, ast.Compare) or node not in parent.comparators:
        return None
    if not (isinstance(parent.left, ast.Name) and parent.left.id == name):
        return None
    return node


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
    rewrites: List[Tuple[ast.expr, str]] = []
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
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent  # type: ignore[attr-defined]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_input_call(node.value):
                if any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
                    targets.append(node.value)
        # Refuse rather than change what the variable means. The one recoverable
        # case is a quoted number compared to the same variable, which is
        # rewritten alongside the conversion.
        converter = float if convert_to == "float" else int
        for node in ast.walk(tree):
            if _string_only_use(node, var, reported_line, converter):
                return None
            literal = _numeric_literal_to_rewrite(node, var, converter)
            if literal is not None:
                rewrites.append((literal, literal.value.strip()))

    if not targets:
        return None

    edits: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], str]] = []
    for call in targets:
        source = ast.get_source_segment(code, call)
        if not source or _is_already_converted(source):
            continue
        edits.append((_span(call), f"{convert_to}({source})"))
    for literal, number in rewrites:
        edits.append((_span(literal), number))

    if not edits:
        return None

    # Bottom-up so earlier offsets remain valid while splicing.
    rewritten = code
    for (start, end), replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        candidate = _splice(rewritten, start, end, replacement)
        if candidate == rewritten:
            continue
        rewritten = candidate

    if rewritten == code:
        return None

    quoted = sorted({number for _, number in rewrites})
    if not quoted:
        rewrites_note = ""
    elif len(quoted) == 1:
        rewrites_note = f" and compared it with the number {quoted[0]} instead of a quoted string"
    else:
        rewrites_note = f" and compared it with the numbers {', '.join(quoted)} instead of quoted strings"

    outcome.applied = "input-type-mismatch"
    outcome.summary = (
        f"Wrapped the input in `{convert_to}()` so the comparison with a number is valid"
        f"{rewrites_note}."
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


_CLOSERS = {"(": ")", "[": "]", "{": "}"}
_UNCLOSED_BRACKET = re.compile(r"'([(\[{])' was never closed")


def _fix_syntax_unclosed_bracket(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``print(1 + 2`` -> ``print(1 + 2)`` when a bracket was never closed.

    The closing bracket cannot already have been written anywhere, so putting
    the one the message names at the end of the file is a repair rather than a
    guess about intent. Everything else has to be right too: if the candidate
    does not parse - two open brackets, a closer landing inside a comment, a
    bracket inside an unterminated string - nothing is offered.
    """
    line = meta.get("line")
    msg = meta.get("syntax_msg") or ""
    if line is None:
        return None
    match = _UNCLOSED_BRACKET.search(msg)
    if not match:
        return None
    opener = match.group(1)
    closer = _CLOSERS[opener]

    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines) or opener not in lines[line - 1]:
        return None

    last = len(lines) - 1
    while last > 0 and not lines[last].strip():
        last -= 1
    ending = lines[last][len(lines[last].rstrip("\r\n")) :]
    candidate = (
        "".join(lines[:last])
        + lines[last].rstrip("\r\n")
        + closer
        + ending
        + "".join(lines[last + 1 :])
    )
    if candidate == code:
        return None
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None

    outcome.applied = "unclosed-bracket"
    outcome.summary = f"Closed the `{opener}` opened on line {line} with `{closer}` at the end of the file."
    return candidate


def _numeric_converter(node: ast.AST) -> Optional[str]:
    """``int`` or ``float`` when ``node`` is a literal of that kind."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, float):
            return "float"
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return _numeric_converter(node.operand)
    return None


def _used_as_text(tree: ast.AST, name: str) -> bool:
    """True when ``name`` is used in a way that only works for a string.

    Wrapping ``age = input()`` in ``int()`` fixes ``age + 10`` and turns
    ``age.strip()`` into an ``AttributeError`` three lines down, so the
    conversion is not offered while such a use exists.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == name:
                return True
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == name:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len":
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == name:
                        return True
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(isinstance(op, ast.Name) and op.id == name for op in operands):
                return any(
                    isinstance(op, ast.Constant) and isinstance(op.value, str)
                    for op in operands
                )
    return False


def _fix_input_string_arithmetic(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``age = input()`` + ``print(age + 10)`` -> ``age = int(input())``.

    ``input()`` returns ``str``, so adding a number to it is the string/number
    boundary the analyzer reports. Reading a number and adding ten to it is the
    only reading of that line that keeps ``10`` a number: turning the ``+`` into
    text concatenation would make the result ``"3610"``, which is not what the
    programmer wrote either. So the input is converted - and only when the other
    operand is a numeric literal, because only then is the intended type known,
    and only when nothing else in the file needs the value to stay a string.
    """
    if tree is None or meta.get("binop") != "str-plus-number":
        return None
    if meta.get("template"):
        return None  # a string literal is involved: the f-string fixer owns this
    line = meta.get("line")
    end_line = meta.get("end_line") or line
    if line is None:
        return None

    targets: List[Tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if not line <= getattr(node, "lineno", 0) <= end_line:
                continue
            left_converter = _numeric_converter(node.left)
            right_converter = _numeric_converter(node.right)
            for other, converter in (
                (node.right, left_converter),
                (node.left, right_converter),
            ):
                if converter is None:
                    continue
                if not isinstance(other, ast.Name):
                    continue
                if _used_as_text(tree, other.id):
                    return None
                targets.append((other.id, converter))

    if len(targets) != 1 or len({t[0] for t in targets}) != 1:
        return None
    name, converter = targets[0]

    # Bind the conversion to the assignment that feeds this name, and only when
    # there is exactly one such assignment to change.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]

    calls = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and _is_input_call(node.value)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    if getattr(call, "parent", None) is None or not isinstance(call.parent, ast.Assign):
        return None
    assignment_line = call.parent.lineno

    rewritten = _splice(code, _span(call)[0], _span(call)[1], f"{converter}({ast.unparse(call)})")
    if rewritten == code:
        return None
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return None

    outcome.applied = "input-string-arithmetic"
    outcome.summary = (
        f"Converted the value read on line {assignment_line} with `input()` to "
        f"`{converter}()` so it can be added to a number."
    )
    return rewritten


def _fix_str_number_concat(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``'Age: ' + 18`` -> an f-string, which is the only reading of that code.

    ``+`` cannot cross the str/number boundary, so the programmer meant to embed
    the value in the text. ``f'text: {x}'`` and ``'text: ' + str(x)`` produce the
    same string, so this rewrites rather than guesses.
    """
    if meta.get("binop") != "str-plus-number":
        return None
    str_side = meta.get("str_side")
    line = meta.get("line")
    end_line = meta.get("end_line") or line
    if str_side not in ("left", "right") or line is None or tree is None:
        return None

    template = meta.get("template") or ""
    expression = meta.get("expression") or ""
    if not template or not expression:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        if not line <= getattr(node, "lineno", 0) <= end_line:
            continue

        placeholder = "{" + expression + "}"
        inner = template + placeholder if meta.get("str_side") == "left" else placeholder + template
        # An f-string expression may not contain a backslash before Python 3.12,
        # and the quote that delimits it may not appear in the expression.
        replacement = None
        if "\\" not in expression:
            for quote in ("'", '"'):
                if quote not in inner:
                    replacement = "f" + quote + inner + quote
                    break
        if replacement is None:
            for quote in ("'", '"'):
                if quote in template:
                    continue
                left = f"{quote}{template}{quote}" if meta.get("str_side") == "left" else f"str({expression})"
                right = f"str({expression})" if meta.get("str_side") == "left" else f"{quote}{template}{quote}"
                replacement = f"{left} + {right}"
                break
        if replacement is None:
            return None

        rewritten = _splice(code, _span(node)[0], _span(node)[1], replacement)
        if rewritten == code:
            return None

        outcome.applied = "str-number-concat"
        outcome.summary = (
            "Turned the `+` into an f-string, which is how a number is put into text "
            "in Python."
        )
        return rewritten
    return None


_COMPOUND_ASSIGN_HEADS = ("!", "<", ">", "+", "-", "*", "/", "%", "&", "|", "^", "@", ":", "=")
_CONDITION_KEYWORDS = ("if", "elif", "while", "assert")


def _single_assignment_in_condition(body: str) -> Optional[int]:
    """Index of the lone ``=`` in a condition, or ``None`` when it is ambiguous."""
    stripped = body.strip()
    keyword = stripped.split(None, 1)[0] if stripped.split(None, 1) else ""
    if keyword not in _CONDITION_KEYWORDS:
        return None

    candidates = []
    for index, char in enumerate(body):
        if char != "=":
            continue
        if index and body[index - 1] in _COMPOUND_ASSIGN_HEADS:
            continue
        if index + 1 < len(body) and body[index + 1] == "=":
            continue
        candidates.append(index)
    return candidates[0] if len(candidates) == 1 else None


def _fix_syntax_assignment_in_condition(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``if x = 5:`` -> ``if x == 5:``.

    CPython says "Maybe you meant '==' or ':=' instead of '='?" but points at the
    start of the condition rather than at the character, so the line itself is
    searched - and only when the statement is a condition and the line holds
    exactly one candidate. The file does not parse, so the validator is what
    confirms the result.
    """
    msg = (meta.get("syntax_msg") or "").lower()
    line = meta.get("line")
    if "maybe you meant '=='" not in msg or line is None:
        return None

    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None
    body = lines[line - 1].rstrip("\r\n")
    index = _single_assignment_in_condition(body)
    if index is None:
        return None

    rewritten = (
        "".join(lines[: line - 1])
        + body[:index]
        + "=="
        + body[index + 1 :]
        + lines[line - 1][len(body) :]
        + "".join(lines[line:])
    )
    if rewritten == code:
        return None

    outcome.applied = "assignment-in-condition"
    outcome.summary = (
        f"Changed the single `=` on line {line} to `==`, which is what the parser asked for."
    )
    return rewritten


# Lines that must keep the block header's indentation. They are never pulled
# into a block that happens to have been written above them unindented, because
# doing so would silently change which statement belongs to which construct.
_BLOCK_SIBLINGS = re.compile(
    r"^[ \t]*(@|else|elif|except|finally|if|for|while|def|class|with|try|async)\b"
)


def _indent_unit(header_indent: str) -> str:
    """One indentation step, in whatever the header already uses."""
    return "\t" if "\t" in header_indent else "    "


def _indent_run(lines: List[str], start: int, header_indent: str) -> List[int]:
    """Indices of the consecutive unindented lines that belong to the block.

    The run stops at a blank line, at a sibling block header, or at anything
    indented differently - each of those is a place where the programmer's own
    formatting says the block ended, and guessing past it restructures the file.
    """
    run: List[int] = []
    for index in range(start, len(lines)):
        text = lines[index]
        if not text.strip():
            break
        if index > start and _BLOCK_SIBLINGS.match(text):
            break
        if re.match(r"[ \t]*", text).group(0) != header_indent:
            break
        run.append(index)
    return run


def _only_comments_after(lines: List[str], start: int) -> bool:
    """True when nothing but comments follows line ``start`` (0-based)."""
    return all(not text.strip() or text.lstrip().startswith("#") for text in lines[start:])


def _fix_syntax_missing_indent(
    code: str, tree: Optional[ast.AST], meta: Dict[str, Any], outcome: FixOutcome
) -> Optional[str]:
    """``if age > 18:\\nprint("Adult")`` -> indent the statement into the block.

    CPython reports the *offending* line (the unindented statement) while the
    block header sits on the line named in the message, so both are needed.

    Indenting the statement is the repair. This fixer used to insert an
    indented ``pass`` instead, which parses just as happily and leaves
    ``print("Adult")`` running for every input rather than only for adults -
    the code goes green and the bug survives.
    """
    line = meta.get("line")
    msg = meta.get("syntax_msg") or ""
    if line is None or "expected an indented block" not in msg.lower():
        return None

    lines = code.splitlines(keepends=True)
    header_line = line
    match = _HEADER_LINE_IN_MESSAGE.search(msg)
    if match:
        header_line = int(match.group(1))
    if not 1 <= header_line <= len(lines) or not 1 <= line <= len(lines):
        return None

    header = lines[header_line - 1]
    body = header.rstrip("\r\n")
    if not body.rstrip().endswith(":"):
        return None
    header_indent = re.match(r"[ \t]*", body).group(0)

    # Nothing follows the header, or only comments do: there is no statement to
    # move, so an empty body really is the repair.
    if header_line >= len(lines) or _only_comments_after(lines, line - 1):
        inserted = header_indent + _indent_unit(header_indent) + "pass\n"
        if header_line == len(lines) and not header.endswith("\n"):
            inserted = "\n" + inserted
        outcome.applied = "missing-indent"
        outcome.summary = (
            f"Line {header_line} opens a block with no body, so `pass` was added to it."
        )
        return "".join(lines[:header_line]) + inserted + "".join(lines[header_line:])

    offender = lines[line - 1]
    offender_indent = re.match(r"[ \t]*", offender).group(0)
    # Only the plain case where the statement sits at the header's own level. A
    # deeper or shallower indent is a mismatch between levels, and choosing
    # which level the programmer meant restructures the program.
    if offender_indent != header_indent:
        return None

    run = _indent_run(lines, line - 1, header_indent)
    if not run:
        return None

    unit = _indent_unit(header_indent)
    patched = list(lines)
    for index in run:
        patched[index] = header_indent + unit + lines[index][len(header_indent) :]

    candidate = "".join(patched)
    if candidate == code:
        return None
    # The repair has to be a repair: if indenting the run still leaves a
    # syntax error, the block boundary was not where this rule assumed.
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None

    first, last = run[0] + 1, run[-1] + 1
    where = f"line {first}" if first == last else f"lines {first}-{last}"
    outcome.applied = "missing-indent"
    outcome.summary = (
        f"Indented {where} into the block opened on line {header_line}, which is "
        "where the parser expected it."
    )
    return candidate


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
        lambda context: context.get("error_type") == "TypeError"
        and context.get("binop") == "str-plus-number",
        _fix_str_number_concat,
    ),
    (
        lambda context: context.get("error_type") == "TypeError"
        and context.get("binop") == "str-plus-number",
        _fix_input_string_arithmetic,
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
        lambda context: "maybe you meant '=='" in (context.get("syntax_msg") or "").lower(),
        _fix_syntax_assignment_in_condition,
    ),
    (
        lambda context: "missing parentheses in call to 'print'" in (context.get("syntax_msg") or "").lower(),
        _fix_syntax_py2_print,
    ),
    (
        lambda context: "was never closed" in (context.get("syntax_msg") or ""),
        _fix_syntax_unclosed_bracket,
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