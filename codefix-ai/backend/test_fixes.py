"""Fix-generation + validation tests. Run with: python test_fixes.py"""
import ast

from analyzer import analyze_code
from fixes import propose_fix
from validator import validate_correction

FLAGSHIP = '''age = input("Enter your age: ")

if age > 18:
    print("Adult")
else:
    print("Minor")'''

CASES = [
    (
        "wrap input in int()",
        FLAGSHIP,
        'age = int(input("Enter your age: "))\n\nif age > 18:\n    print("Adult")\nelse:\n    print("Minor")',
    ),
    (
        "wrap input in float() for float comparison",
        "price = input('p: ')\nif price < 10.5:\n    print('cheap')",
        "price = float(input('p: '))\nif price < 10.5:\n    print('cheap')",
    ),
    (
        "inline input comparison",
        "if 18 < input('age: '):\n    print('adult')",
        "if 18 < int(input('age: ')):\n    print('adult')",
    ),
    (
        "singleton comparison to is None",
        "def find(x):\n    if x == None:\n        return 0\n    return 1",
        "def find(x):\n    if x is None:\n        return 0\n    return 1",
    ),
    (
        "singleton comparison is not",
        "def f(x):\n    if x != None:\n        return 1",
        "def f(x):\n    if x is not None:\n        return 1",
    ),
    (
        "singleton comparison with the constant on the left",
        "def find(x):\n    if None == x:\n        return 0\n    return 1",
        "def find(x):\n    if None is x:\n        return 0\n    return 1",
    ),
    (
        # Removing line 1 shifts the second unused import onto the reported
        # line, which used to make validation reject the correct fix.
        "remove only the reported unused import",
        "import os\nimport sys\n\nprint('hi')",
        "import sys\n\nprint('hi')",
    ),
    (
        # An inline input() fix must not convert every input() in the file.
        "inline input fix touches only the reported line",
        "a = input('a: ')\nif 18 < input('age: '):\n    print(a)",
        "a = input('a: ')\nif 18 < int(input('age: ')):\n    print(a)",
    ),
    (
        "add missing colon",
        "if x > 3\n    print('a')",
        "if x > 3:\n    print('a')",
    ),
    (
        "indent pass for empty block",
        "def greet():\nprint('hi')",
        "def greet():\n    pass\nprint('hi')",
    ),
    (
        "python 2 print statement",
        'print "hello"',
        'print("hello")',
    ),
    (
        "typo: pritn -> print",
        "def f():\n    pritn('hi')\n",
        "def f():\n    print('hi')\n",
    ),
    (
        "remove unused import",
        "import os\n\nprint('hi')",
        "\nprint('hi')",
    ),
    (
        "multi-line input binding keeps body",
        'name = input(\n    "Name: "\n)\nif name == 5:\n    print(1)',
        'name = int(input(\n    "Name: "\n))\nif name == 5:\n    print(1)',
    ),
    (
        "correct code with input() is never rewritten",
        'name = input("Name: ")\nif name == "bob":\n    print(1)',
        None,
    ),
]

NO_FIX = [
    ("zero division needs a real decision", "x = 10 / 0"),
    ("dynamic comparison is left alone", "def cmp(a, b):\n    return a > b\n"),
    ("unresolvable name is not guessed", "def f():\n    return a + b + c\n"),
    ("already correct code is not touched", "def f():\n    return 1\n"),
    # 0 == False and 1 == True, so `x == 0` used to be rewritten to `x is 0`.
    ("comparison with a number is not a singleton fix", "count = 3\nif count == 0:\n    print('zero')\n"),
    # `age` is 0.75 similar to the builtin `range`, which is a coincidence and
    # not a typo; rewriting `print(age)` to `print(range)` breaks the program.
    ("undefined name is not replaced with an unrelated builtin", "print(age)\n"),
    # The correct repair would be `print('hi')`, which means deleting an
    # argument. Renaming `prnt` to `print` instead nests the call.
    ("callee is not renamed into its own call", "print(pritn('hi'))\n"),
    # Two different problems; fixing the second would leave the first in place.
    ("a second undefined name is left for its own analysis", "print(age)\nprint(country)\n"),
]

failures = 0

for label, code, expected in CASES:
    analysis = analyze_code(code)

    if expected is None:
        outcome = propose_fix(code, analysis)
        if outcome is not None:
            failures += 1
            print(f"FAIL  {label}: rewrote correct code -> {outcome.code!r}")
        else:
            print(f"ok    {label}")
        continue

    outcome = propose_fix(code, analysis)
    if outcome is None:
        print(f"FAIL  {label}: no fix proposed")
        failures += 1
        continue

    corrected = outcome.code
    report = validate_correction(code, corrected, analysis)

    try:
        ast.parse(corrected)
        parses = True
    except SyntaxError:
        parses = False

    problems = []
    if corrected != expected:
        problems.append(f"got {corrected!r}\n        want {expected!r}")
    if not parses:
        problems.append("corrected code does not parse")
    if not report.validated:
        problems.append(f"validation failed: {report.reason}")
    if report.issue_resolved is not True:
        problems.append(f"issue not confirmed resolved: {report.issue_resolved}")

    if problems:
        failures += 1
        print(f"FAIL  {label}: {'; '.join(problems)}")
    else:
        print(f"ok    {label}  [fix={outcome.applied}]")

print()
for label, code in NO_FIX:
    analysis = analyze_code(code)
    outcome = propose_fix(code, analysis)
    if outcome is not None:
        failures += 1
        print(f"FAIL  {label}: proposed a guess -> {outcome.code!r}")
    else:
        print(f"ok    {label}  (correctly no auto-fix)")


# A candidate written by the language model has to clear a stricter gate than a
# deterministic edit: it may not silence the reported message while leaving the
# same kind of bug behind.
UNTRUSTED = [
    (
        "model output that renames the undefined name into another one",
        "print(age)\n",
        "defprint(name)\n",
        False,
    ),
    (
        # Inventing the missing value is not repairing the reported line. The
        # rewrite model is never asked for `undefined-variable` in the first
        # place, but the gate has to hold if that policy ever changes.
        "model output that invents a value for the missing name",
        "print(age)\n",
        "age = 10\nprint(age)\n",
        False,
    ),
]

print()
for label, code, candidate, expected in UNTRUSTED:
    analysis = analyze_code(code)
    report = validate_correction(code, candidate, analysis, trusted=False)
    if report.validated is not expected:
        failures += 1
        print(f"FAIL  {label}: validated={report.validated} (want {expected}): {report.reason}")
    else:
        print(f"ok    {label}  (validated={report.validated})")


# Passing every static gate is still not enough: the candidate has to be the same
# program with the reported line repaired. Asked to fix `print(age)` the previous
# checkpoint answered `print(sys.argv[1])` -- one line changed, parses, compiles,
# and `sys` is not imported, so only the distance from the reported line gives
# the substitution away.
LOCALITY = [
    (
        "rewrites the whole file instead of the reported line",
        "import os\nimport sys\nprint(toatl)\n",
        "import sys\nprint(total)\n",
        False,
    ),
    (
        "edits a line far from the reported error",
        "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\ng = 7\nh = 8\ni = 9\nprint(a + nope)\n",
        "a = 1\nb = 2\nc = 4\nd = 4\ne = 5\nf = 6\ng = 7\nh = 8\ni = 9\nprint(a + nope)\n",
        False,
    ),
    (
        "repairs the reported line and stays local",
        "name = 'Ada'\nage = 36\nprint(name + ' is ' + age)\n",
        "name = 'Ada'\nage = 36\nprint(name + ' is ' + str(age))\n",
        True,
    ),
]

print()
for label, code, candidate, expected in LOCALITY:
    analysis = analyze_code(code)
    report = validate_correction(code, candidate, analysis, trusted=False)
    if report.local is None:
        failures += 1
        print(f"FAIL  {label}: locality was not reported")
    elif report.validated is not expected or report.local is not expected:
        failures += 1
        print(
            f"FAIL  {label}: validated={report.validated} local={report.local} "
            f"(want {expected}): {report.reason}"
        )
    else:
        print(f"ok    {label}  (validated={report.validated}, local={report.local})")

print()
print("FAILURES:", failures)
raise SystemExit(1 if failures else 0)
