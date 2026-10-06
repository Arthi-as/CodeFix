"""Static code analysis for CodeFix AI.

Three layers, in decreasing confidence order:

1. ``syntax``    - CPython's own parser (``compile``/``ast``). Exact line and
                   message come from the interpreter, so these findings are
                   ground truth.
2. ``semantic``  - small type/value inference over the AST that catches the
                   bugs beginners actually hit (str vs number comparison,
                   division by a literal zero, ``x == None``, ...). Every
                   finding is derived from concrete literal types, so it does
                   not fire on genuinely dynamic code.
3. ``pylint``    - Pylint's error and warning messages for everything else
                   (undefined names, unused variables, bad attribute access).

Findings are ranked by (layer, severity, line) rather than by line number.
That ordering matters: Pylint reports ``missing-module-docstring`` on line 1,
and a naive "first message wins" would report the docstring instead of the
real bug on line 3.

Nothing in this module executes user code. Pylint only parses it.
"""

from __future__ import annotations

import ast
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

_LOG = logging.getLogger(__name__)

MAX_CODE_CHARS = 50_000

SEVERITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
LAYER_RANK = {"syntax": 0, "semantic": 1, "pylint": 2}

PYLINT_SEVERITY = {"error": "High", "warning": "Medium"}

# Pylint checkers that produce noise rather than findings for a debugging
# assistant aimed at beginners.
PYLINT_DISABLE = [
    # environment dependent - the interpreter may simply not have the module
    "import-error",
    "no-name-in-module",
    # style / formatting
    "line-too-long",
    "trailing-whitespace",
    "missing-final-newline",
    "trailing-newlines",
    "bad-continuation",
    # CPython accepts consistent tabs; "expected 4 spaces" is a style opinion
    # that reports valid tab-indented files as bugs. Genuinely inconsistent
    # indentation is a SyntaxError and the syntax layer already reports it.
    "bad-indentation",
    "superfluous-parens",
    "consider-using-f-string",
    "consider-using-with",
    "consider-using-enumerate",
    "consider-using-generator",
    "consider-using-dict-items",
    "consider-using-ternary",
    "consider-using-max-builtin",
    "consider-using-set-comprehension",
    "consider-iterating-dictionary",
    "use-list-literal",
    "use-dict-literal",
    "unnecessary-lambda",
    # name style is not a bug
    "invalid-name",
    # configuration noise
    "fixme",
    "useless-object-inheritance",
]

# Friendly category names so the UI can group findings.
PYLINT_CATEGORY = {
    "undefined-variable": "Name Error",
    "used-before-assignment": "Name Error",
    "undefined-loop-variable": "Name Error",
    "unused-variable": "Unused Code",
    "unused-import": "Unused Code",
    "redefined-builtin": "Naming Issue",
    "not-callable": "Type Error",
    "unsubscriptable-object": "Type Error",
    "unsupported-assignment-operation": "Type Error",
    "unsupported-membership-test": "Type Error",
    "no-member": "Attribute Error",
    "no-value-for-parameter": "Type Error",
    "too-many-function-args": "Type Error",
    "unexpected-keyword-arg": "Type Error",
    "assignment-from-no-return": "Type Error",
    "pointless-statement": "Logic Error",
    "expression-not-assigned": "Logic Error",
    "comparison-with-itself": "Logic Error",
    "singleton-comparison": "Logic Error",
    "duplicate-key": "Logic Error",
    "unreachable": "Logic Error",
    "bare-except": "Error Handling",
    "broad-except": "Error Handling",
    "raise-missing-from": "Error Handling",
    "cell-var-from-loop": "Logic Error",
    "duplicate-argument-name": "Syntax Error",
    "keyword-arg-before-vararg": "Type Error",
    "abstract-method": "Type Error",
}

PYLINT_SUGGESTION = {
    "undefined-variable": "The name is not defined anywhere in this file. Check the spelling, or import/assign it before use.",
    "used-before-assignment": "This name is read before it is assigned. Move the assignment above the first use.",
    "unused-variable": "This variable is assigned but never read. Remove it or use it.",
    "unused-import": "This import is never used. Remove it.",
    "redefined-builtin": "This name shadows a Python built-in. Rename it to avoid confusion.",
    "not-callable": "This object is not callable. Check whether you meant to call a function or index a sequence.",
    "no-member": "The object has no such attribute. Check the spelling and confirm the object type.",
    "unsubscriptable-object": "This object cannot be indexed with []. Check the type.",
    "no-value-for-parameter": "A required argument is missing from this call.",
    "too-many-function-args": "Too many positional arguments for this call.",
    "unexpected-keyword-arg": "This function does not accept that keyword argument.",
    "pointless-statement": "This expression is evaluated but its result is discarded.",
    "expression-not-assigned": "This expression has no effect. Assign it or remove it.",
    "bare-except": "A bare `except:` also catches KeyboardInterrupt and SystemExit. Catch a specific exception.",
    "broad-except": "This catches every exception and hides real failures. Catch a specific exception.",
    "duplicate-key": "This dictionary literal defines the same key twice; the last value wins.",
    "unreachable": "This code can never run because the statement above always exits or raises.",
    "cell-var-from-loop": "This closure captures a loop variable by reference, so every callback sees the final value.",
}


@dataclass
class CodeIssue:
    """A single finding produced by static analysis."""

    line: int
    column: int
    error_type: str
    severity: str
    message: str
    category: str
    suggestion: str = ""
    layer: str = "pylint"
    end_line: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> tuple:
        return (
            LAYER_RANK.get(self.layer, 9),
            SEVERITY_RANK.get(self.severity, 9),
            self.line,
            self.column,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line or self.line,
            "error_type": self.error_type,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "suggestion": self.suggestion,
            "detected_by": self.layer,
        }


# --------------------------------------------------------------------------
# type inference
# --------------------------------------------------------------------------

# ``num`` is what :func:`_literal_type` returns for an arithmetic result such as
# ``1 + 2``. It has to be in this set or ``'items: ' + (1 + 2)`` goes unreported.
_NUMERIC = {"int", "float", "num"}
_CONCRETE_NUMERIC = ("int", "float")


def _is_singleton_literal(node: ast.AST) -> bool:
    """True only for ``None``, ``True`` and ``False``.

    ``value in (None, True, False)`` cannot be used for this: ``0 == False`` and
    ``1 == True`` in Python, so every ``count == 0`` looks like a comparison
    against ``False`` and gets rewritten to ``count is 0``.
    """
    return isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, bool)
    )


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _literal_type(node: ast.AST) -> str:
    """Best-effort type of an expression, ``unknown`` when it is dynamic."""
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        return {
            "input": "str",
            "raw_input": "str",
            "int": "int",
            "float": "float",
            "str": "str",
            "bool": "bool",
            "len": "int",
        }.get(name, "unknown")
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, str):
            return "str"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        return "unknown"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_type(node.left)
        right = _literal_type(node.right)
        if "str" in (left, right):
            return "str"
        if left in _NUMERIC and right in _NUMERIC:
            return "num"
        return "unknown"
    if isinstance(node, ast.FormattedValue):
        return "unknown"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, (ast.ListComp,)):
        return "list"
    return "unknown"


class _Binding(NamedTuple):
    """One assignment to a simple name, kept in source order."""

    line: int
    column: int
    name: str
    type: str
    value: Optional[ast.AST]


class _Scope:
    """A module body, a function body or a class body."""

    __slots__ = ("parent", "kind", "def_line", "bindings")

    def __init__(self, parent: Optional["_Scope"], kind: str, def_line: int) -> None:
        self.parent = parent
        self.kind = kind
        self.def_line = def_line
        self.bindings: List[_Binding] = []


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _arguments(args: ast.arguments) -> List[ast.arg]:
    """Every parameter node of a function or lambda."""
    collected = (
        list(getattr(args, "posonlyargs", []))
        + list(args.args)
        + list(args.kwonlyargs)
    )
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            collected.append(extra)
    return collected


class _BindingTable:
    """Every assignment in the file, grouped by the scope it happens in.

    Name resolution follows Python: the innermost scope that binds the name
    *anywhere* wins, then the enclosing scopes, then the module. Resolving by
    line number across the whole file instead - which is what this used to do -
    let a string bound inside one function be reported as the type of the same
    name in an unrelated function, and the fixer then rewrote that other
    function's line to satisfy a finding that was never real.
    """

    def __init__(self, tree: ast.AST) -> None:
        self.module = _Scope(None, "module", 1)
        self._scopes: Dict[int, _Scope] = {id(tree): self.module}
        self._build(tree)

    # -- construction ---------------------------------------------------

    def _build(self, tree: ast.AST) -> None:
        def bind(scope: _Scope, name: str, node: ast.AST, inferred: str, value: Any) -> None:
            scope.bindings.append(
                _Binding(node.lineno, node.col_offset, name, inferred, value)
            )

        def parameters(inner: _Scope, node: ast.AST) -> None:
            for arg in _arguments(node.args):  # type: ignore[attr-defined]
                bind(inner, arg.arg, arg, "unknown", None)

        def walk(node: ast.AST, scope: _Scope) -> None:
            self._scopes[id(node)] = scope

            if isinstance(node, ast.Assign):
                inferred = _literal_type(node.value)
                for target in node.targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            bind(scope, name.id, name, inferred, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                inferred = _literal_type(node.value)
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        bind(scope, name.id, name, inferred, node.value)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                bind(scope, node.target.id, node.target, "unknown", node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        bind(scope, name.id, name, "unknown", node.iter)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, _FUNCTION_NODES):
                    # The name belongs to the enclosing scope, the body does not.
                    bind(scope, child.name, child, "unknown", None)
                    inner = _Scope(scope, "function", child.lineno)
                    parameters(inner, child)
                    walk(child, inner)
                elif isinstance(child, ast.Lambda):
                    inner = _Scope(scope, "function", child.lineno)
                    parameters(inner, child)
                    walk(child, inner)
                elif isinstance(child, ast.ClassDef):
                    # A class body is not a closure: its names are not visible
                    # in the scope that encloses the class.
                    bind(scope, child.name, child, "unknown", None)
                    walk(child, _Scope(self.module, "class", child.lineno))
                else:
                    walk(child, scope)

        walk(tree, self.module)

    # -- lookup ----------------------------------------------------------

    @staticmethod
    def _chain(scope: _Scope) -> List[_Scope]:
        chain: List[_Scope] = []
        current: Optional[_Scope] = scope
        while current is not None:
            chain.append(current)
            current = current.parent
        return chain

    def scope_of(self, node: ast.AST) -> _Scope:
        return self._scopes.get(id(node), self.module)

    def visible(self, name: str, scope: _Scope) -> List[_Binding]:
        """Every binding of ``name`` reachable from ``scope``."""
        found: List[_Binding] = []
        for candidate in self._chain(scope):
            found.extend(binding for binding in candidate.bindings if binding.name == name)
        return found

    def binding_at(self, name: str, line: int, scope: _Scope) -> Optional[_Binding]:
        """The binding of ``name`` in effect at ``line``, or ``None``.

        The walk stops at the first scope that binds the name *at all*, even
        when every such binding is later than ``line``: Python fixes the owning
        scope at compile time, so a parameter shadows a module-level variable
        of the same name for the whole function body.
        """
        for candidate in self._chain(scope):
            matching = [binding for binding in candidate.bindings if binding.name == name]
            if not matching:
                continue
            for binding in reversed(matching):
                if binding.line <= line:
                    return binding
            return None
        return None

    def type_at(self, name: str, line: int, scope: _Scope) -> str:
        """Type of ``name`` at ``line``, honouring rebinding order and scope."""
        binding = self.binding_at(name, line, scope)
        return binding.type if binding else "unknown"


def _expr_type(node: ast.AST, bindings: _BindingTable, scope: _Scope, line: int) -> str:
    if isinstance(node, ast.Name):
        return bindings.type_at(node.id, getattr(node, "lineno", line), scope)
    return _literal_type(node)


# --------------------------------------------------------------------------
# analyzer
# --------------------------------------------------------------------------


class PythonAnalyzer:
    """Runs every static analysis layer over a snippet of Python code."""

    def analyze_syntax(self, code: str) -> List[CodeIssue]:
        """Ground-truth syntax errors straight from the CPython parser."""
        try:
            ast.parse(code)
        except SyntaxError as exc:
            line = exc.lineno or 1
            return [
                CodeIssue(
                    line=line,
                    column=exc.offset or 0,
                    end_line=getattr(exc, "end_lineno", None) or line,
                    error_type="SyntaxError",
                    severity="High",
                    category="Syntax Error",
                    message=exc.msg or str(exc),
                    suggestion=self._syntax_suggestion(exc, code),
                    layer="syntax",
                    meta={"syntax_msg": exc.msg or "", "offset": exc.offset},
                )
            ]
        except (ValueError, MemoryError, RecursionError) as exc:
            # Null bytes, excessively deep nesting, etc.
            return [
                CodeIssue(
                    line=1,
                    column=0,
                    error_type=type(exc).__name__,
                    severity="High",
                    category="Parse Error",
                    message=f"The code could not be parsed: {exc}",
                    suggestion="Check the file for invalid characters or excessive nesting.",
                    layer="syntax",
                )
            ]
        return []

    @staticmethod
    def _syntax_suggestion(exc: SyntaxError, code: str) -> str:
        """Actionable advice for the most common beginner syntax errors."""
        msg = (exc.msg or "").lower()
        lines = code.splitlines()
        line_text = lines[exc.lineno - 1] if exc.lineno and exc.lineno <= len(lines) else ""

        if "expected an indented block" in msg:
            return (
                f"Line {exc.lineno} opens a block but the next line is not indented. "
                "Add 4 spaces of indentation (or any consistent amount)."
            )
        if "expected ':'" in msg or "perhaps you forgot a colon" in msg:
            return f"Add a `:` at the end of line {exc.lineno} to open the block."
        if "unterminated string literal" in msg or "eol while scanning string" in msg:
            return f"Line {exc.lineno} has a string literal that is never closed. Add the matching quote."
        if "'(' was never closed" in msg or "invalid syntax" in msg and "(" in line_text:
            return f"Line {exc.lineno} has unbalanced parentheses or brackets."
        if "invalid indentation" in msg or "unexpected indent" in msg:
            return (
                f"Line {exc.lineno} is indented unexpectedly. Indentation must be "
                "consistent - do not mix tabs and spaces."
            )
        if "was never closed" in msg:
            return f"A bracket or quote opened on an earlier line is never closed."
        if "assignment to literal" in msg or "cannot assign to" in msg:
            return f"You cannot assign to this expression. Assign to a plain name."
        if "keyword argument repeated" in msg:
            return "The same keyword argument appears more than once. Keep only one."
        if "positional argument follows keyword" in msg:
            return "Move positional arguments before keyword arguments."
        if "unexpected indent" in msg:
            return "Remove the extra leading whitespace on this line."
        return "Review the syntax around this line."

    # -- semantic layer ----------------------------------------------------

    def analyze_semantics(self, code: str) -> List[CodeIssue]:
        """Literal-typed checks for the bugs beginners hit most often."""
        issues: List[CodeIssue] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        bindings = _BindingTable(tree)

        for node in ast.walk(tree):
            scope = bindings.scope_of(node)
            if isinstance(node, ast.Compare):
                issues.extend(self._check_compare(node, bindings, scope))
            elif isinstance(node, ast.BinOp):
                issues.extend(self._check_binop(node, bindings, scope))
            elif isinstance(node, ast.Subscript):
                issues.extend(self._check_subscript(node, bindings, tree))
        return issues

    def _check_compare(
        self, node: ast.Compare, bindings: _BindingTable, scope: _Scope
    ) -> List[CodeIssue]:
        issues: List[CodeIssue] = []
        operands = [node.left, *node.comparators]
        types = [_expr_type(operand, bindings, scope, node.lineno) for operand in operands]

        # string vs number comparison
        if "str" in types and any(kind in _NUMERIC for kind in types):
            index = next(i for i, kind in enumerate(types) if kind == "str")
            issues.append(
                self._string_number_comparison(node, operands[index], types, bindings, scope)
            )
            return issues

        # `x == None` / `None == x` / `== True` should use `is`. Already-correct
        # `is None` comparisons are the operators we are steering towards, so
        # skip those. Both operand orders are checked; pylint does not report
        # this at all, because `singleton-comparison` is a convention message.
        for index, op in enumerate(node.ops):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for operand_index, on_left in ((index + 1, False), (index, True)):
                if _is_singleton_literal(operands[operand_index]):
                    issues.append(
                        self._singleton_comparison(
                            node, operands[operand_index], op, on_left=on_left
                        )
                    )
                    return issues
        return issues

    def _string_number_comparison(
        self,
        node: ast.Compare,
        operand: ast.AST,
        types: List[str],
        bindings: _BindingTable,
        scope: _Scope,
    ) -> CodeIssue:
        numbers = [kind for kind in types if kind in _NUMERIC]
        number_word = "integer" if numbers == ["int"] else "number"
        convert_to = next((kind for kind in numbers if kind in _CONCRETE_NUMERIC), "int")

        if isinstance(operand, ast.Name):
            var = operand.id
            binding = bindings.binding_at(var, node.lineno, scope)
            subject = f"`{var}`"
            if binding and _is_input_call(binding.value):
                reason = f"it is assigned from input() on line {binding.line}"
                hint = f"Wrap the input in `{convert_to}()` so the comparison works."
            else:
                origin = f" on line {binding.line}" if binding else ""
                reason = f"it holds a string value{origin}"
                hint = (
                    f"Convert it to {_article(number_word)} {number_word} where the value is "
                    f"created or used, for example `{convert_to}({var})`."
                )
            meta: Dict[str, Any] = {"var_name": var, "convert_to": convert_to}
        else:
            var = None
            subject = "The `input()` result"
            reason = "input() always returns a string"
            hint = f"Wrap the call in `{convert_to}()`: `{convert_to}(input(...))`."
            meta = {"var_name": None, "convert_to": convert_to}

        return CodeIssue(
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, "end_lineno", None),
            error_type="TypeError",
            severity="Medium",
            category="Type Error",
            message=(
                f"{subject} is a string because {reason}, but line {node.lineno} compares it with "
                f"{_article(number_word)} {number_word}. Comparing a str with an int or float "
                f"raises TypeError, for example \"'>' not supported between instances of "
                f"'str' and 'int'\"."
            ),
            suggestion=hint,
            layer="semantic",
            meta=meta,
        )

    @staticmethod
    def _singleton_comparison(
        node: ast.Compare, operand: ast.Constant, op: ast.AST, on_left: bool = False
    ) -> CodeIssue:
        shown = "None" if operand.value is None else ("True" if operand.value else "False")
        operator = "==" if isinstance(op, ast.Eq) else "!="
        replacement = "is" if operator == "==" else "is not"
        found = f"`{shown} {operator}`" if on_left else f"`{operator} {shown}`"
        wanted = f"`{shown} {replacement}`" if on_left else f"`{replacement} {shown}`"
        return CodeIssue(
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, "end_lineno", None),
            error_type="SingletonComparison",
            severity="Medium",
            category="Logic Error",
            message=(
                f"Line {node.lineno} writes {found} on `{ast.unparse(node)}`. `{shown}` is one "
                f"single shared object in Python, so compare identity with {wanted} instead "
                f"of value."
            ),
            suggestion=f"Change {found} to {wanted}.",
            layer="semantic",
            meta={"replacement": f"{replacement} {shown}"},
        )

    def _check_binop(
        self, node: ast.BinOp, bindings: _BindingTable, scope: _Scope
    ) -> List[CodeIssue]:
        issues: List[CodeIssue] = []
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                if right.value == 0 and not isinstance(right.value, bool):
                    issues.append(
                        CodeIssue(
                            line=node.lineno,
                            column=node.col_offset,
                            end_line=getattr(node, "end_lineno", None),
                            error_type="ZeroDivisionError",
                            severity="High",
                            category="Runtime Error",
                            message=(
                                f"Line {node.lineno} divides by the literal 0, which always raises "
                                "`ZeroDivisionError: division by zero`."
                            ),
                            suggestion="Check that the divisor is non-zero before dividing.",
                            layer="semantic",
                        )
                    )

        if isinstance(node.op, ast.Add):
            left_kind = _expr_type(node.left, bindings, scope, node.lineno)
            right_kind = _expr_type(node.right, bindings, scope, node.lineno)
            if "str" in (left_kind, right_kind) and any(
                kind in _NUMERIC for kind in (left_kind, right_kind)
            ):
                # Suggest a concrete f-string only when one side is a plain literal.
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    template = node.left.value.replace("{", "{{").replace("}", "}}")
                    hint = f'Use an f-string: f"{template}{{{ast.unparse(node.right)}}}".'
                elif isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                    template = node.right.value.replace("{", "{{").replace("}", "}}")
                    hint = f'Use an f-string: f"{{{ast.unparse(node.left)}}}{template}".'
                else:
                    hint = "Use an f-string to embed the number in the text, e.g. f\"value: {x}\"."
                # A concrete rewrite is only possible when one side is a plain
                # string literal; otherwise the f-string is only a hint.
                meta: Dict[str, Any] = {"binop": "str-plus-number"}
                left_literal = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                right_literal = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                if left_literal or right_literal:
                    literal = node.left if left_literal else node.right
                    other = node.right if left_literal else node.left
                    meta["str_side"] = "left" if left_literal else "right"
                    meta["template"] = literal.value.replace("{", "{{").replace("}", "}}")
                    meta["expression"] = ast.unparse(other)
                issues.append(
                    CodeIssue(
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=getattr(node, "end_lineno", None),
                        error_type="TypeError",
                        severity="Medium",
                        category="Type Error",
                        message=(
                            f"Line {node.lineno} adds a string to a number with `+`. Python's `+` "
                            "concatenates two strings or adds two numbers; it does not convert "
                            "between them."
                        ),
                        suggestion=hint,
                        layer="semantic",
                        meta=meta,
                    )
                )
        return issues

    def _check_subscript(
        self, node: ast.Subscript, bindings: _BindingTable, tree: ast.AST
    ) -> List[CodeIssue]:
        """Flag ``items[i]`` only when the sequence length is statically known."""
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return []
        if isinstance(node.slice.value, bool):
            return []
        if not isinstance(node.value, ast.Name):
            return []
        name = node.value.id
        index = node.slice.value

        # Only a single visible binding proves the length: a second assignment
        # anywhere in scope could have replaced the sequence.
        visible = bindings.visible(name, bindings.scope_of(node))
        if len(visible) != 1:
            return []
        if _membership_is_mutated(tree, name):
            return []

        elements = _literal_elements(visible[0].value)
        if elements is None:
            return []
        length = len(elements)
        if length and index >= length:
            return [
                CodeIssue(
                    line=node.lineno,
                    column=node.col_offset,
                    end_line=getattr(node, "end_lineno", None),
                    error_type="IndexError",
                    severity="High",
                    category="Runtime Error",
                    message=(
                        f"Line {node.lineno} accesses `{name}[{index}]` but `{name}` has only "
                        f"{length} element(s) (indices 0 to {length - 1}). This raises "
                        "`IndexError: list index out of range`."
                    ),
                    suggestion=f"Use an index between 0 and {length - 1}, or guard with `if {index} < len({name})`.",
                    layer="semantic",
                )
            ]
        return []

    # -- pylint layer ------------------------------------------------------

    def analyze_pylint(self, code: str) -> List[CodeIssue]:
        """Pylint errors and warnings, run in-process (no PATH/subprocess)."""
        collector = _MessageCollector()
        disabled = _supported_disable_symbols()
        handle = None
        path = ""
        try:
            handle, path = tempfile.mkstemp(suffix=".py", prefix="codefix_")
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(code)
                handle = None

            # Pylint runs in this process, so its console script does not have to
            # be on PATH (it is not, when uvicorn is started by absolute path).
            from pylint.lint import Run  # noqa: PLC0415

            Run(
                [path, "--score=n", "--reports=n", "--persistent=n"]
                + [f"--disable={name}" for name in disabled],
                reporter=collector,
                exit=False,
            )
        except Exception as exc:  # pragma: no cover - pylint unavailable/broken
            _LOG.warning("pylint run failed, continuing without it: %s", exc)
            return []
        finally:
            if handle is not None:
                try:
                    os.close(handle)
                except OSError:
                    pass
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        issues: List[CodeIssue] = []
        for message in collector.messages:
            symbol = getattr(message, "symbol", "") or ""
            kind = getattr(message, "category", "") or ""
            if kind not in ("error", "warning"):
                continue
            if symbol in disabled:
                continue
            line = getattr(message, "line", None) or 1
            try:
                line = int(line)
            except (TypeError, ValueError):
                line = 1
            if line < 1:
                line = 1

            issues.append(
                CodeIssue(
                    line=line,
                    column=int(getattr(message, "column", 0) or 0),
                    error_type=symbol,
                    severity=PYLINT_SEVERITY.get(kind, "Medium"),
                    category=PYLINT_CATEGORY.get(symbol, "Static Analysis"),
                    message=(getattr(message, "msg", "") or "").strip(),
                    suggestion=PYLINT_SUGGESTION.get(
                        symbol,
                        f"Review Pylint's `{symbol}` finding on line {line}.",
                    ),
                    layer="pylint",
                    meta={"symbol": symbol},
                )
            )
        return issues

    # -- orchestration -----------------------------------------------------

    def analyze(self, code: str) -> List[CodeIssue]:
        """All findings, most important first."""
        if not code.strip():
            return []

        syntax_issues = self.analyze_syntax(code)
        if syntax_issues:
            # Nothing else can be analysed reliably, and Pylint would just
            # repeat the parser error.
            return syntax_issues

        issues: List[CodeIssue] = []
        issues.extend(self.analyze_semantics(code))
        issues.extend(self.analyze_pylint(code))

        deduped: List[CodeIssue] = []
        seen = set()
        for issue in sorted(issues, key=lambda item: item.rank):
            key = (issue.layer, issue.error_type, issue.line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped


# --------------------------------------------------------------------------
# pylint plumbing
# --------------------------------------------------------------------------


try:  # Pylint is a hard requirement, but keep the analyzer usable without it.
    from pylint.reporters import BaseReporter as _BaseReporter

    _PYLINT_AVAILABLE = True
except Exception:  # pragma: no cover - only when pylint is not installed
    _BaseReporter = object  # type: ignore[assignment,misc]
    _PYLINT_AVAILABLE = False


def _supported_disable_symbols() -> List[str]:
    """Intersect PYLINT_DISABLE with what the installed Pylint accepts.

    Pylint's message set moves between releases (``missing-self-use`` became an
    optional extension in 3.x), and an unknown name makes it emit
    ``unknown-option-value`` instead of honouring the rest of the list.
    """
    try:
        from pylint.lint import PyLinter  # noqa: PLC0415

        known = {name.lower() for name in PyLinter().msgs_store.messages}
    except Exception:  # pragma: no cover - pylint missing or changed API
        return list(PYLINT_DISABLE)
    return [name for name in PYLINT_DISABLE if name.lower() in known]


class _MessageCollector(_BaseReporter):  # type: ignore[misc,valid-type]
    """A Pylint reporter that records messages instead of printing them."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[Any] = []

    def _display(self, layout: Any) -> None:
        return None

    def handle_message(self, message: Any) -> None:
        self.messages.append(message)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _is_input_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    )


def _literal_elements(node: Optional[ast.AST]) -> Optional[List[Any]]:
    """The elements a literal assignment binds, or ``None`` when not literal.

    A string literal counts as a sequence of its characters, which is how
    ``name = 'bob'; name[10]`` becomes an ``IndexError``.
    """
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [None] * len(node.value)
    return None


_MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "update", "add", "discard", "setdefault", "popitem",
}


def _membership_is_mutated(tree: ast.AST, name: str) -> bool:
    """True when the container may change size, invalidating a length guess."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _MUTATING_METHODS:
            value = node.value
            if isinstance(value, ast.Name) and value.id == name:
                return True
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            if node.iter.id == name:
                return True
        if isinstance(node, ast.comprehension) and isinstance(node.iter, ast.Name):
            if node.iter.id == name:
                return True
    return False


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def find_issues(code: str) -> List[CodeIssue]:
    """Every finding for ``code``, highest priority first."""
    if len(code) > MAX_CODE_CHARS:
        return [
            CodeIssue(
                line=1,
                column=0,
                error_type="CodeTooLarge",
                severity="Low",
                category="Input",
                message=(
                    f"The snippet is {len(code)} characters; the limit is {MAX_CODE_CHARS}. "
                    "Analyze a smaller file."
                ),
                suggestion="Trim the snippet to the smallest part that reproduces the problem.",
                layer="syntax",
            )
        ]
    return PythonAnalyzer().analyze(code)


def analyze_code(code: str) -> Dict[str, Any]:
    """Analyse ``code`` and return the primary finding plus the full list."""
    issues = find_issues(code)

    if not issues:
        return {
            "has_error": False,
            "line": None,
            "column": None,
            "end_line": None,
            "error_type": None,
            "severity": None,
            "category": None,
            "detected_by": None,
            "message": "No issues found. The code parses cleanly and Pylint reported no errors or warnings.",
            "suggestion": None,
            "corrected_code": code,
            "validated": True,
            "is_syntax_error": False,
            "all_issues": [],
        }

    primary = issues[0]
    return {
        "has_error": True,
        "line": primary.line,
        "column": primary.column,
        "end_line": primary.end_line or primary.line,
        "error_type": primary.error_type,
        "severity": primary.severity,
        "category": primary.category,
        "detected_by": primary.layer,
        "message": primary.message,
        "suggestion": primary.suggestion,
        "corrected_code": code,
        "validated": False,
        "is_syntax_error": primary.layer == "syntax",
        "meta": dict(primary.meta),
        "all_issues": [issue.to_dict() for issue in issues[:25]],
    }