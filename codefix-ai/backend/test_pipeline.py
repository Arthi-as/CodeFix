"""Pipeline tests: which finding is reported and what the user is offered.

Runs with the model disabled, so these cover ``main.run_pipeline``'s own
decisions rather than the model. Run with: python test_pipeline.py
"""
import os

os.environ["CODEFIX_ENABLE_MODEL"] = "0"

from main import run_pipeline

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if condition:
        print(f"ok    {label}")
    else:
        failures += 1
        print(f"FAIL  {label}{(': ' + detail) if detail else ''}")


# A finding nobody can fix is a worse headline than one we can correct, but the
# fix must never make the code look clean when it is not.
result = run_pipeline("import os\nimport sys\nprint(toatl)\n")
check(
    "a fixable finding is promoted over one with no safe fix",
    result["validated"] and result["error_type"] == "unused-import" and result["line"] == 1,
    f"got line={result['line']} type={result['error_type']} validated={result['validated']}",
)
check(
    "the promoted fix is applied",
    result["corrected_code"] == "import sys\nprint(toatl)\n",
    repr(result["corrected_code"]),
)
check(
    "the fix summary says what was left open",
    "still open" in (result["fix_summary"] or "") and "line 3" in (result["fix_summary"] or ""),
    repr(result["fix_summary"]),
)
check(
    "the unfixed finding is still listed",
    any(i["line"] == 3 and i["error_type"] == "undefined-variable" for i in result["all_issues"]),
)

# Two comparisons of one input() are two findings. Fixing the first and calling
# it validated is how "it worked on the demo" stops being true.
two = run_pipeline(
    "a = input('a: ')\nif a > 1:\n    print('a')\nb = input('b: ')\nif b > 2:\n    print('b')\n"
)
check(
    "every occurrence of the same bug is fixed in one response",
    two["corrected_code"] == (
        "a = int(input('a: '))\nif a > 1:\n    print('a')\nb = int(input('b: '))\nif b > 2:\n    print('b')\n"
    ),
    repr(two["corrected_code"]),
)
check(
    "both edits are described",
    (two["fix_summary"] or "").count("Wrapped the input") == 2,
    repr(two["fix_summary"]),
)

# int() on a variable that is also compared to a word would silently break the
# program, so nothing is offered.
for label, code in [
    (
        "variable compared to a word is not converted",
        "c = input('y/n? ')\nif c == 'y':\n    print('yes')\nelif c > 2:\n    print('many')\n",
    ),
    (
        "variable used as a string is not converted",
        "s = input('x: ')\nif s > 2:\n    print(s.strip())\n",
    ),
    (
        "quoted number that does not round-trip is not converted",
        "c = input('0-9? ')\nif c == '01':\n    print('one')\nelif c > 2:\n    print('many')\n",
    ),
]:
    outcome = run_pipeline(code)
    check(
        label,
        not outcome["validated"] and outcome["corrected_code"] == code,
        f"validated={outcome['validated']} code={outcome['corrected_code']!r}",
    )
    check(
        f"{label}: the reason is honest",
        "No automatic correction is safe to offer" in (outcome["validation_details"] or {}).get(
            "reason", ""
        ),
    )

# A quoted number that the converter reproduces exactly is rewritten with it, so
# the menu keeps working.
menu = run_pipeline(
    "choice = input('1/2/3? ')\n"
    "if choice == '1':\n"
    "    print('one')\n"
    "elif choice > 2:\n"
    "    print('many')\n"
)
check(
    "quoted number is rewritten with the conversion",
    menu["corrected_code"] == (
        "choice = int(input('1/2/3? '))\n"
        "if choice == 1:\n"
        "    print('one')\n"
        "elif choice > 2:\n"
        "    print('many')\n"
    ),
    repr(menu["corrected_code"]),
)

# Nothing to promote, nothing to fix: the headline stays the real problem.
alone = run_pipeline("x = 10 / 0\nprint(x)\n")
check(
    "a real error with no fix stays the headline",
    alone["error_type"] == "ZeroDivisionError" and not alone["validated"],
    f"type={alone['error_type']} validated={alone['validated']}",
)

clean = run_pipeline("print('hello')\n")
check("clean code is untouched", clean["validated"] and clean["corrected_code"] == "print('hello')\n")

print()
print("FAILURES:", failures)
raise SystemExit(1 if failures else 0)
