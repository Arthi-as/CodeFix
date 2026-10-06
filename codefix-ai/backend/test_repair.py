"""Repair status, model admission gates and truncation handling.

Everything here runs without loading weights: the model's answer is fed to the
parser as text, and the pipeline cases are the ones that must never reach the
model at all. Run with: python test_repair.py
"""
import ast
import json

from analyzer import analyze_code
from fixes import propose_fix
from main import EXPLANATION_ONLY_MESSAGE, run_pipeline
import rewrite_model
from validator import validate_correction

failures = 0


def ok(label, detail=""):
    print(f"ok    {label}{('  ' + detail) if detail else ''}")


def fail(label, detail):
    global failures
    failures += 1
    print(f"FAIL  {label}: {detail}")


# --------------------------------------------------------------------------
# 1. Deterministic repairers the panel must offer before the model is asked.
# --------------------------------------------------------------------------

DETERMINISTIC = [
    (
        "input arithmetic is wrapped at the binding",
        "age = input()\nprint(age + 10)\n",
        "age = int(input())\nprint(age + 10)\n",
    ),
    (
        "input arithmetic uses float for a decimal",
        "age = input()\nprint(age + 10.5)\n",
        "age = float(input())\nprint(age + 10.5)\n",
    ),
    (
        "unindented body is indented into the block",
        "def greet():\nprint('hi')",
        "def greet():\n    print('hi')",
    ),
    (
        "bracket left open at end of file is closed",
        "print(sum(1, 2)",
        "print(sum(1, 2))",
    ),
    (
        "multi-line expression is closed on its last line",
        "total = (1 +\n         2\n",
        "total = (1 +\n         2)\n",
    ),
]

print("--- deterministic repairers ---")
for label, code, expected in DETERMINISTIC:
    analysis = analyze_code(code)
    outcome = propose_fix(code, analysis)
    if outcome is None:
        fail(label, "no fix proposed")
        continue
    report = validate_correction(code, outcome.code, analysis)
    if outcome.code != expected:
        fail(label, f"got {outcome.code!r}, want {expected!r}")
    elif not report.validated:
        fail(label, f"validation failed: {report.reason}")
    elif report.edit_lines is None or report.edit_lines < 1:
        fail(label, f"edit_lines was not measured: {report.edit_lines}")
    else:
        ok(label, f"[edit_lines={report.edit_lines}]")

# The parser has no safe repair for `sum(1, 2` split across a following
# statement: closing the bracket there produces `2 print(x)`, which does not
# parse, so the fixer has to decline rather than offer it.
label = "unclosed bracket across statements is declined"
code = "x = sum(1, 2\nprint(x)\n"
outcome = propose_fix(code, analyze_code(code))
if outcome is None:
    ok(label)
else:
    fail(label, f"proposed {outcome.code!r}")

# --------------------------------------------------------------------------
# 2. The repair object the API returns.
# --------------------------------------------------------------------------

print("\n--- repair status ---")

result = run_pipeline("def greet():\nprint('hi')\n")
repair = result.get("repair")
label = "deterministic fix reports fixed"
if repair is None:
    fail(label, "repair object missing")
elif repair["status"] != "fixed" or repair["source"] != "static-analysis":
    fail(label, f"got status={repair['status']!r} source={repair['source']!r}")
elif repair["line"] != 2 or repair["replacement"] != "    print('hi')":
    fail(label, f"got line={repair['line']} replacement={repair['replacement']!r}")
elif repair["message"] is not None:
    fail(label, f"message should be null for a fix: {repair['message']!r}")
elif not result["validated"]:
    fail(label, "the pipeline did not validate its own fix")
else:
    ok(label, f"[line={repair['line']}, edit_lines={repair['edit_lines']}]")

label = "deterministic fix keeps the original contract keys"
missing = [key for key in ("validated", "validation_details", "fix_summary", "diff", "explanation") if key not in result]
if missing:
    fail(label, f"missing {missing}")
else:
    ok(label)

for label, code, needle in (
    ("undefined name explains why nothing was substituted", "print(username)\n", "cannot be determined"),
    ("zero division explains why the divisor was left alone", "x = 10 / 0\nprint(x)\n", "change what the program calculates"),
):
    result = run_pipeline(code)
    repair = result["repair"]
    if repair["status"] != "explanation_only":
        fail(label, f"got status={repair['status']!r}")
    elif repair["source"] is not None or repair["replacement"] is not None:
        fail(label, f"an unfixable finding offered a replacement: {repair}")
    elif repair["message"] != EXPLANATION_ONLY_MESSAGE:
        fail(label, f"got message={repair['message']!r}")
    elif needle not in repair["reason"]:
        fail(label, f"reason {repair['reason']!r} does not mention {needle!r}")
    elif result["validated"] or result["corrected_code"] != code:
        fail(label, "an explanation-only result changed the code")
    else:
        ok(label, f"[reason={repair['reason'][:60]!r}]")

label = "clean code reports clean, not explanation-only"
result = run_pipeline("print(1)\n")
repair = result["repair"]
if repair["status"] != "clean" or repair["line"] is not None:
    fail(label, f"got {repair}")
else:
    ok(label)

label = "explanation-only line still points at the finding"
result = run_pipeline("print(username)\n")
if result["repair"]["line"] != result["line"]:
    fail(label, f"repair line {result['repair']['line']} != finding line {result['line']}")
else:
    ok(label)

# --------------------------------------------------------------------------
# 3. The model is never asked about a finding that needs a decision.
# --------------------------------------------------------------------------

print("\n--- model admission ---")

engine = rewrite_model.get_engine()
asked = []
real_chat = engine._chat


def spy(*args, **kwargs):
    asked.append(args)
    return ""


engine._chat = spy
try:
    for label, code in (
        ("undefined-variable never reaches the model", "print(username)\n"),
        ("ZeroDivisionError never reaches the model", "x = 10 / 0\nprint(x)\n"),
        ("IndexError never reaches the model", "values = [1, 2]\nprint(values[10])\n"),
        ("clean code never reaches the model", "print(1)\n"),
    ):
        result = run_pipeline(code)
        if asked:
            fail(label, f"the model was called {len(asked)} time(s)")
        elif result["repair"]["status"] not in ("explanation_only", "clean"):
            fail(label, f"got status={result['repair']['status']!r}")
        else:
            ok(label)
finally:
    engine._chat = real_chat

label = "a decision finding is declined with a reason"
repair, reason = rewrite_model.propose_local_repair("print(username)\n", analyze_code("print(username)\n"))
if repair is not None or not reason:
    fail(label, f"got repair={repair} reason={reason!r}")
else:
    ok(label, f"[{reason}]")

# --------------------------------------------------------------------------
# 4. Admission gates on a model answer - pure text, no weights.
# --------------------------------------------------------------------------

print("\n--- answer gates ---")

CODE = "name = 'Ada'\nage = 36\nprint(name + ' is ' + age)\n"
ANALYSIS = {"line": 3, "error_type": "TypeError", "message": "can only concatenate str to str"}
LINE = "print(name + ' is ' + age)"


def answer(**fields):
    base = {
        "line": 3,
        "confidence": 0.9,
        "original": LINE,
        "replacement": 'print(f"{name} is {age}")',
        "reason": "concatenating str and int needs an f-string",
    }
    base.update(fields)
    return json.dumps(base)


GATES = [
    ("well-formed answer is accepted", answer(), True, ""),
    ("prose is not an answer", "you should use an f-string here", False, "JSON"),
    ("a no_fix answer is a decline", json.dumps({"no_fix": True, "reason": "unclear"}), False, "declined"),
    ("line outside the file", answer(line=99), False, "not in the file"),
    ("original does not quote the line", answer(original="print(name)"), False, "not what is there"),
    ("low confidence is a guess", answer(confidence=0.4), False, "below"),
    ("missing confidence", json.dumps({"line": 3, "original": LINE, "replacement": "x"}), False, "confidence"),
    ("replacement is pass", answer(replacement="pass"), False, "pass"),
    ("replacement is a comment", answer(replacement="# fixed"), False, "comment"),
    ("replacement changes nothing", answer(replacement=LINE), False, "already says"),
    ("empty replacement", answer(replacement="  "), False, "empty"),
    ("no line named", json.dumps({"original": LINE, "replacement": "x = 1"}), False, "line"),
    ("not JSON at all", "{line: 3}", False, "JSON"),
    ("empty answer", "", False, "nothing"),
    (
        "replacement longer than any real line",
        answer(replacement="x" * 400),
        False,
        "longer",
    ),
]

for label, text, should_pass, needle in GATES:
    repair, reason = rewrite_model.parse_repair_response(text, CODE, ANALYSIS)
    if should_pass:
        if repair is None:
            fail(label, f"rejected: {reason}")
        else:
            ok(label)
        continue
    if repair is not None:
        fail(label, f"accepted {repair!r}")
    elif needle.lower() not in reason.lower():
        fail(label, f"reason {reason!r} does not mention {needle!r}")
    else:
        ok(label, f"[{reason}]")

label = "line far from the reported one"
long_code = "\n".join(f"value_{n} = {n}" for n in range(1, 9)) + "\n"
far = json.dumps(
    {
        "line": 8,
        "confidence": 0.9,
        "original": "value_8 = 8",
        "replacement": "value_8 = 0",
        "reason": "wrong value here",
    }
)
repair, reason = rewrite_model.parse_repair_response(
    far, long_code, {"line": 4, "error_type": "TypeError", "message": "m"}
)
if repair is not None:
    fail(label, f"accepted a repair four lines from the report: {repair!r}")
else:
    ok(label, f"[{reason}]")

# --------------------------------------------------------------------------
# 5. Answers cut short by the time budget.
# --------------------------------------------------------------------------

print("\n--- truncated answers ---")

label = "cut inside the reason still yields the repair"
full = answer()
cut = full[: full.rindex("concatenating") + 12]
repair, reason = rewrite_model.parse_repair_response(cut, CODE, ANALYSIS)
if repair is None:
    fail(label, f"rejected: {reason}")
elif repair.reason != "":
    fail(label, f"half-written reason was kept: {repair.reason!r}")
elif repair.replacement != 'print(f"{name} is {age}")':
    fail(label, f"got replacement {repair.replacement!r}")
else:
    ok(label)

label = "cut inside the replacement is caught by parsing the result"
cut = answer()[: answer().index("is {age}") + 8]
repair, reason = rewrite_model.parse_repair_response(cut, CODE, ANALYSIS)
if repair is None:
    ok(label, f"[{reason}]")
else:
    try:
        ast.parse(repair.code)
    except SyntaxError as exc:
        ok(label, f"[validator will reject: {exc.msg}]")
    else:
        fail(label, f"a partial replacement parsed and was offered: {repair.replacement!r}")

label = "cut inside the original fails the exact-quote gate"
cut = answer()[: answer().index("is ' + age") + 4]
repair, reason = rewrite_model.parse_repair_response(cut, CODE, ANALYSIS)
if repair is not None:
    fail(label, f"accepted {repair!r}")
else:
    ok(label, f"[{reason}]")

# --------------------------------------------------------------------------
# 6. edit_lines is reported for every changed candidate.
# --------------------------------------------------------------------------

print("\n--- edit measurement ---")

for label, original, corrected, expected in (
    ("one line changed", "a = 1\nb = 2\n", "a = 1\nb = 3\n", 1),
    ("two lines changed", "a = 1\nb = 2\n", "a = 2\nb = 3\n", 2),
    ("identical code is not an edit", "a = 1\n", "a = 1\n", None),
):
    analysis = analyze_code(original)
    report = validate_correction(original, corrected, analysis)
    if report.edit_lines != expected:
        fail(label, f"got edit_lines={report.edit_lines}, want {expected}")
    elif report.to_dict().get("edit_lines") != expected:
        fail(label, "edit_lines missing from to_dict()")
    else:
        ok(label, f"[edit_lines={report.edit_lines}]")

print()
print("FAILURES:", failures)
raise SystemExit(1 if failures else 0)
