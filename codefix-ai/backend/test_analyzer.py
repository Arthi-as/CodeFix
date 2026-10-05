"""Static analysis regression tests. Run with: python test_analyzer.py"""
from analyzer import analyze_code, find_issues

FLAGSHIP = '''age = input("Enter your age: ")

if age > 18:
    print("Adult")
else:
    print("Minor")'''

CASES = [
    (
        "flagship input/int comparison",
        FLAGSHIP,
        {"has_error": True, "line": 3, "error_type": "TypeError", "severity": "Medium",
         "category": "Type Error", "detected_by": "semantic"},
    ),
    (
        "missing indented block",
        "def greet():\nprint('hi')",
        {"has_error": True, "error_type": "SyntaxError", "severity": "High"},
    ),
    (
        "undefined name outranks docstring noise",
        "def f():\n    return totl\n",
        {"has_error": True, "error_type": "undefined-variable"},
    ),
    (
        "literal zero division",
        "x = 10 / 0",
        {"has_error": True, "error_type": "ZeroDivisionError", "line": 1},
    ),
    (
        "string compared to int literal",
        "name = 'bob'\nif name == 5:\n    print('x')",
        {"has_error": True, "error_type": "TypeError"},
    ),
    (
        "singleton comparison",
        "def find(x):\n    if x == None:\n        return 0\n",
        {"has_error": True, "error_type": "SingletonComparison"},
    ),
    (
        "float comparison picks float conversion",
        "price = input('price: ')\nif price < 10.5:\n    print('cheap')",
        {"has_error": True, "error_type": "TypeError"},
    ),
    (
        "clean code reports nothing",
        "def greet(name):\n    return f\"Hello, {name}!\"\n\n\nprint(greet('World'))",
        {"has_error": False},
    ),
    (
        "str + int is a type error",
        "count = 3\nlabel = 'items: ' + count\n",
        {"has_error": True, "error_type": "TypeError"},
    ),
    (
        "out of range literal index",
        "colors = ['red', 'green']\nprint(colors[5])",
        {"has_error": True, "error_type": "IndexError"},
    ),
    (
        "negative index is idiomatic, not a bug",
        "items = [1, 2, 3]\nlast = items[-1]\nprint(last)\n",
        {"has_error": False},
    ),
    (
        "input compared in the other order",
        "if 18 < input('age: '):\n    print('adult')",
        {"has_error": True, "error_type": "TypeError"},
    ),
    (
        # 0 == False in Python, so a value test against None/True/False used to
        # be read as a singleton comparison and rewritten to `count is 0`.
        "comparison with 0 is not a singleton comparison",
        "count = 3\nif count == 0:\n    print('zero')",
        {"has_error": False},
    ),
    (
        "comparison with 1 is not a singleton comparison",
        "count = 3\nif count == 1:\n    print('one')",
        {"has_error": False},
    ),
    (
        "singleton comparison with the constant on the left",
        "def find(x):\n    if None == x:\n        return 0\n    return 1\n",
        {"has_error": True, "line": 2, "error_type": "SingletonComparison",
         "detected_by": "semantic"},
    ),
    (
        # A string bound in one function is not the type of the same name in
        # another; this used to be reported as a TypeError on line 5.
        "a name bound in another function is not a type error",
        "def g():\n    name = input('n: ')\n\ndef f():\n    if name > 5:\n        print(name)",
        {"has_error": True, "line": 5, "error_type": "undefined-variable"},
    ),
    (
        "a parameter is not another function's local variable",
        "def read_limit():\n    limit = input('limit: ')\n    return limit\n\n"
        "def check(limit):\n    if limit > 5:\n        print(limit)\n\ncheck(read_limit())\n",
        {"has_error": False},
    ),
    (
        "arithmetic result added to text",
        "label = 'items: ' + (1 + 2)\nprint(label)",
        {"has_error": True, "line": 1, "error_type": "TypeError", "detected_by": "semantic"},
    ),
    (
        # Pylint's bad-indentation only objects to spaces; the parser already
        # rejects indentation that is genuinely inconsistent.
        "tab-indented code is not an indentation bug",
        "def f():\n\tif len('ab') > 1:\n\t\tprint(1)\n",
        {"has_error": False},
    ),
    (
        "mixed tabs and spaces are still a syntax error",
        "def f():\n\tx = 1\n    y = 2\n",
        {"has_error": True, "error_type": "SyntaxError", "detected_by": "syntax"},
    ),
]

failures = 0
for name, code, expected in CASES:
    result = analyze_code(code)
    problems = [
        f"{key}={result.get(key)!r} (want {want!r})"
        for key, want in expected.items()
        if result.get(key) != want
    ]
    if problems:
        failures += 1
        print(f"FAIL  {name}: {'; '.join(problems)}")
        for issue in find_issues(code)[:5]:
            print(f"        L{issue.line} [{issue.layer}] {issue.error_type}: {issue.message[:90]}")
    else:
        print(f"ok    {name}  -> line={result.get('line')} type={result.get('error_type')} "
              f"sev={result.get('severity')} cat={result.get('category')}")

print()
print("FAILURES:", failures)
raise SystemExit(1 if failures else 0)