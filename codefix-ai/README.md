# CodeFix AI

An AI-assisted Python debugging system. Paste Python code, and it finds the
problematic line, explains what is wrong, and proposes a correction that has
been checked before it is offered.

## Why this works the way it does

The important design decision here is **who is allowed to be confident**.

| Question | Answered by | Why |
|---|---|---|
| Is there an error, and on which line? | `ast` + Pylint | Deterministic and exhaustive over the checks it runs |
| What should the code become? | AST rewriter | A precise edit is provably safer than generated text |
| Is the correction acceptable? | `validator.py` | It must parse, compile, and clear the original finding |
| How do I explain it in words? | CodeT5, with a deterministic fallback | Wording is where a language model genuinely helps |

Static analysis is the **only** source of truth for the error type, line, and
severity. The model is advisory: when a deterministic fix already validates,
the model is not even consulted for code, because a validated exact edit
beats a generated one.

Where the model cannot be trusted, the system says so rather than guessing.
For `x = 10 / 0` or an undefined name, there is no safe automatic correction,
so CodeFix AI reports and explains the problem and offers nothing.

## Honest limitations

- **`Salesforce/codet5-base` is a pretrained checkpoint, not a chat or
  instruction model.** It was trained to fill in code, not to follow
  instructions. Asked to explain a bug it frequently emits unrelated code or
  repetition. Its output is therefore passed through quality gates and
  discarded when it fails, which in practice means the deterministic
  explanation is usually what you read. Point `CODEFIX_MODEL` at an
  instruction-tuned variant to change that.
- **Static analysis cannot see runtime behaviour.** Code that parses and
  passes Pylint can still be logically wrong. "No issues found" means "no
  issues found", not "correct".
- **Code is never executed.** See [Validation](#validation-and-safety).
- **Analysis runs in-process.** Pylint imports the file's module environment,
  so it is not a security boundary against hostile input. This is a local
  developer tool.

## Validation and safety

`validator.py` never runs the submitted code. It checks, in order:

1. structural limits (size, nesting, encoding),
2. `ast.parse` — does it parse,
3. `compile()` — does it compile without executing,
4. re-analysis — does the original finding actually disappear,
5. warnings collected for display.

A correction is reported as `validated: true` only if all of these pass and the
original error is gone. Step 4 matches findings by error type and wording rather
than by line number, because deleting an unused import shifts every later line
up by one; and a candidate written by the language model must additionally
introduce no finding the original code did not have. That last gate matters:
`print(age)` rewritten as `defprint(name)` silences the reported message while
leaving the same bug, so it is rejected rather than shown as code to copy.

`safe_execute()` exists as an explicit, **disabled by default** contract for
sandboxed execution. Turning it on requires container or process isolation
with seccomp/AppArmor, a PID namespace, a read-only filesystem, dropped
capabilities, no network, and enforced CPU, memory and wall-clock limits.

## Project layout

```
backend/
  analyzer.py     AST, compiler and Pylint checks -> ranked findings
  fixes.py        Deterministic AST-positioned correctors + diff rows
  validator.py    Parse/compile/re-analyze gate. No execution.
  codet5.py       Lazy, thread-safe model wrapper with quality gates
  explain.py      Deterministic explanation text
  main.py         FastAPI app: orchestration, caching, routes
frontend/
  src/App.js      Monaco editor, analysis panel, Apply Fix
```

## Running it

### Backend

```powershell
cd backend
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

Interactive API docs at <http://127.0.0.1:8000/docs>.

The model is ~850 MB and is loaded in the background at startup, so the server
accepts requests immediately. The first analysis pays the load cost (~15–20 s
on CPU); later ones take ~0.5–3 s. Set `CODEFIX_ENABLE_CODET5=0` to run
without the model entirely.

### Frontend

```powershell
cd frontend
npm install
npm start
```

Open <http://localhost:3000>. `package.json` proxies `/analyze` to
`http://127.0.0.1:8000`. For a deployed build, set the backend URL:

```powershell
$env:REACT_APP_API_URL = "https://api.example.com"; npm run build
```

The backend must then allow that origin via `CODEFIX_CORS_ORIGINS`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CODEFIX_ENABLE_CODET5` | `1` | `0` disables the model |
| `CODEFIX_MODEL` | `Salesforce/codet5-base` | Model id or local path |
| `CODEFIX_CORS_ORIGINS` | localhost dev ports | Comma-separated origins |
| `CODEFIX_HOST` / `CODEFIX_PORT` | `127.0.0.1` / `8000` | Bind address |
| `REACT_APP_API_URL` | *(empty, uses proxy)* | Frontend backend URL |

## API

### `POST /analyze`

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "age = input(\"Enter your age: \")\n\nif age > 18:\n    print(\"Adult\")"}'
```

```json
{
  "has_error": true,
  "line": 3,
  "column": 5,
  "error_type": "TypeError",
  "severity": "Medium",
  "category": "Type Error",
  "detected_by": "semantic",
  "message": "A value read from input() is a string, but it is compared with a number.",
  "explanation": "...",
  "corrected_code": "age = int(input(\"Enter your age: \"))\n\nif age > 18:\n    print(\"Adult\")",
  "validated": true,
  "fix_summary": "Wrapped the input in `int()` so the comparison with a number is valid.",
  "diff": [{ "kind": "removed", "line": 1, "text": "age = input(...)" }],
  "ai": { "model": "Salesforce/codet5-base", "fix_source": "static-analysis" },
  "all_issues": [],
  "analysis_ms": 21
}
```

`400` empty code, `413` over `MAX_CODE_CHARS`, `500` unexpected failure.

### `GET /health`

Returns API version and model state (`disabled`, `not-loaded`, `loading`, `ready`, `failed`).

## Detected categories

Syntax errors, type errors, name errors, division by zero, index errors,
singleton comparison (`== None`), unused imports and variables, redefined
builtins, non-callable and missing-attribute access, broad exception
handlers — plus everything Pylint's error and warning checks report.

## Corrections offered automatically

`input()` wrapped in `int()`/`float()`; `== None`/`None ==` → `is None`/`is not
None`; missing colon; missing indented block; Python 2 `print`; likely
identifier typos; removal of unused imports and variables.

Each is gated on the finding type. Clean code is never modified, and a fixer
that fails validation is discarded rather than shown as a fix. Type
inference is scoped per function: a name assigned inside one function never
leaks into another, and `count == 0` is a numeric comparison, not a
singleton check.

## Tests

```powershell
cd backend
.\venv\Scripts\python.exe test_analyzer.py    # detection
.\venv\Scripts\python.exe test_fixes.py       # correctors + validator
.\venv\Scripts\python.exe test_api.py         # live server, end to end
.\venv\Scripts\python.exe test_contract.py    # UI/backend field contract
```

`test_api.py` and `test_contract.py` start and stop their own server on a free
port. `test_contract.py` exists because the UI reads nested paths like
`validation_details.reason`; a missing key renders as `undefined` in the
browser instead of raising, so it fails silently. The test asserts every field
the UI consumes is present.