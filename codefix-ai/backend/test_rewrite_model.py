"""Model-admission policy tests. Run with: python test_rewrite_model.py

These never load weights. What is checked is the gate: which findings the
rewrite model is allowed to be asked about, and that a finding needing a
decision returns nothing instead of a guess.
"""
import rewrite_model
from analyzer import analyze_code

failures = 0


def check(label, condition, detail=""):
    global failures
    if condition:
        print(f"ok    {label}")
    else:
        failures += 1
        print(f"FAIL  {label}: {detail}")


# The rewrite model is asked only about mechanical repairs. These need a
# decision about intent, and a 0.5B model makes it up: asked to fix an undefined
# name it answered `print(sys.argv[1])`, which parses, compiles and has no
# findings for the analyzer to complain about.
DECISION_TYPES = [
    ("undefined-variable", "print(age)\n"),
    ("no-member", "import random\nrandom.no_such_function()\n"),
    ("IndexError", "name = 'bob'\nprint(name[10])\n"),
    ("ZeroDivisionError", "x = 10 / 0\nprint(x)\n"),
    ("not-callable", "x = 5\nx()\n"),
    ("unsubscriptable-object", "x = 5\nprint(x[0])\n"),
]

for error_type, code in DECISION_TYPES:
    analysis = analyze_code(code)
    if analysis["error_type"] != error_type:
        check(
            f"{error_type} is still detected",
            False,
            f"analyzer reported {analysis['error_type']} for {code!r}",
        )
        continue
    candidate = rewrite_model.propose_model_fix(code, analysis)
    check(
        f"model not asked about {error_type}",
        candidate is None,
        f"returned {candidate!r}",
    )

# The eligible list is policy, so it is pinned here: adding a type to it is a
# decision that has to show up in a diff, not a quiet side effect of a new
# detector.
check(
    "eligible types are exactly the mechanical repairs",
    set(rewrite_model.ELIGIBLE_TYPES)
    == {"TypeError", "SyntaxError", "SingletonComparison", "unused-import", "unused-variable"},
    f"got {sorted(rewrite_model.ELIGIBLE_TYPES)}",
)
check(
    "every eligible type is one the API documents",
    set(rewrite_model.ELIGIBLE_TYPES) <= set(rewrite_model.available_error_types()),
    "documented types and eligible types disagree",
)
check(
    "explanation generation is opt-in",
    rewrite_model.EXPLAIN_ENABLED is False,
    "the model explanation costs ~10 s per sentence on CPU and is on by default",
)

print()
print("FAILURES:", failures)
raise SystemExit(1 if failures else 0)
