"""Deterministic explanations.

These are the fallback wording used when CodeT5 is unavailable or its output
fails the quality gates in :mod:`codet5`. They are derived entirely from the
static finding, so they are always specific to the reported line and never
invent a cause.
"""

from __future__ import annotations

from typing import Dict, List

# What the value at the fault line actually evaluates to, and what breaks.
EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "SyntaxError": {
        "title": "The code does not follow Python's grammar rules",
        "body": (
            "Python could not turn this file into a program, so nothing in it runs - not even "
            "the lines above the mistake. The reported line is where the parser gave up, which "
            "is often one line after the real problem: an unclosed bracket or a missing colon on "
            "the previous line shows up here."
        ),
    },
    "TypeError": {
        "title": "Two incompatible types are being used together",
        "body": (
            "Python is dynamically typed but strict about types once values exist. Here a value "
            "that holds a string is combined or compared with a number, and there is no operation "
            "that makes sense for both, so Python raises TypeError. Converting the value at the "
            "point it is created fixes it for every later use as well."
        ),
    },
    "ZeroDivisionError": {
        "title": "Division by zero always fails in Python",
        "body": (
            "Unlike most languages, Python refuses to divide by zero rather than returning "
            "infinity or NaN. Because the divisor is a literal 0 here, this branch is "
            "unconditional, so the line fails every single time it runs."
        ),
    },
    "IndexError": {
        "title": "The index is outside the sequence",
        "body": (
            "Sequences are numbered from 0, so a sequence of length n has valid indices 0 to "
            "n - 1. Negative indices are allowed but count backwards from the end. Here the "
            "index is a fixed number that is past the end of a sequence whose length is known."
        ),
    },
    "SingletonComparison": {
        "title": "Comparing values with == is not the same as identity",
        "body": (
            "None, True and False are single shared objects in Python. Writing `== None` asks "
            "whether two objects have the same value, which a custom __eq__ can override to "
            "return True for unrelated objects. `is` asks whether they are the same object, "
            "which is both faster and correct for these three values."
        ),
    },
    "undefined-variable": {
        "title": "The name does not exist when this line runs",
        "body": (
            "Pylint could not find any assignment, import, function or parameter that defines "
            "this name, so Python raises NameError the moment the line executes. Common causes "
            "are a typo, an import that is missing, or code at module level that runs before "
            "the assignment further down."
        ),
    },
    "unused-variable": {
        "title": "This assignment has no effect",
        "body": (
            "The value is stored and then never read, so the assignment does not change what "
            "the program does. It often signals that a variable was renamed in one place but "
            "not another, leaving the original behind as dead code."
        ),
    },
    "unused-import": {
        "title": "This import is never used",
        "body": (
            "Nothing in the file references anything from this module. Removing it keeps the "
            "file honest about its dependencies and makes startup marginally faster, since the "
            "module does not need to be imported at all."
        ),
    },
    "redefined-builtin": {
        "title": "This name hides a Python built-in",
        "body": (
            "The variable has the same name as a built-in such as list, id or sum. Your code "
            "still works, but any built-in used in the same scope now resolves to your variable, "
            "and readers cannot tell the two apart."
        ),
    },
    "not-callable": {
        "title": "This value cannot be called",
        "body": (
            "Something other than a function is being invoked with parentheses. Python raises "
            "TypeError at that point. Common causes are a missing pair of parentheses when a "
            "function was passed around, or a variable shadowing a function name."
        ),
    },
    "no-member": {
        "title": "This attribute does not exist on the object",
        "body": (
            "The object on the left of the dot has no such attribute, so Python raises "
            "AttributeError. Either the name is misspelled, or the object is not the type the "
            "code assumes - for example a str where a list was expected."
        ),
    },
    "bare-except": {
        "title": "A bare except swallows every error",
        "body": (
            "`except:` also catches KeyboardInterrupt and SystemExit, so Ctrl+C and shutdown "
            "requests are treated as ordinary failures and quietly ignored. Catch the specific "
            "exception you expect instead."
        ),
    },
    "broad-except": {
        "title": "Catching Exception hides real failures",
        "body": (
            "This catches everything, including bugs such as a typo'd attribute name, and "
            "discards the detail that would explain what went wrong. Narrow it to the exception "
            "type you expect, and keep the message."
        ),
    },
}

DEFAULT_TITLE = "Static analysis found a problem on this line"
DEFAULT_BODY = (
    "This was reported by Pylint. The finding is deterministic: the same code always produces "
    "it, so fixing the cause removes it. Read the suggestion below and apply the change that "
    "matches your intent."
)

CATEGORY_NOTES: Dict[str, str] = {
    "Syntax Error": "Fixed by correcting the grammar so the file parses.",
    "Type Error": "Fixed by converting the value to the type the operation needs.",
    "Runtime Error": "The code parses, but this line raises an exception when it runs.",
    "Logic Error": "The code runs without error, but it does not do what it looks like it does.",
    "Name Error": "Python cannot resolve this name at runtime.",
    "Unused Code": "Nothing reads this value, so it can be removed.",
    "Naming Issue": "Valid Python, but the name is likely to confuse a reader.",
    "Attribute Error": "The object does not provide the attribute being read.",
    "Error Handling": "The code compiles, but the error handling hides useful information.",
    "Static Analysis": "Reported by Pylint's error and warning checks.",
}


def build_explanation(analysis: Dict[str, Any]) -> str:
    """Structured fallback explanation for a finding.

    Returns the combined title, cause and advice as one string so the UI has a
    single field to render.
    """
    error_type = analysis.get("error_type") or ""
    category = analysis.get("category") or ""
    entry = EXPLANATIONS.get(error_type)

    title = entry["title"] if entry else DEFAULT_TITLE
    body = entry["body"] if entry else DEFAULT_BODY

    parts: List[str] = [f"{title}.", body]

    note = CATEGORY_NOTES.get(category)
    if note:
        parts.append(note)

    if analysis.get("suggestion"):
        parts.append(f"What to do: {analysis['suggestion']}")

    return " ".join(parts)


def split_explanation(analysis: Dict[str, Any]) -> Dict[str, str]:
    """The same content as separate fields, for richer UI layouts."""
    error_type = analysis.get("error_type") or ""
    entry = EXPLANATIONS.get(error_type)
    return {
        "title": entry["title"] if entry else DEFAULT_TITLE,
        "body": entry["body"] if entry else DEFAULT_BODY,
        "category_note": CATEGORY_NOTES.get(analysis.get("category") or "", ""),
    }