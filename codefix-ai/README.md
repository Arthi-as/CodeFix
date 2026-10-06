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
| Is the correction acceptable? | `validator.py` | It must parse, compile, stay local, and clear the original finding |
| How do I explain it in words? | `explain.py` | The static text is line-specific and free; the model cannot be trusted to beat it |

Static analysis is the **only** source of truth for the error type, line, and
severity. The model is advisory: when a deterministic fix already validates,
the model is not even consulted for code, because a validated exact edit
beats a generated one.

Where the model cannot be trusted, the system says so rather than guessing.
For `x = 10 / 0` or an undefined name, there is no safe automatic correction,
so CodeFix AI reports and explains the problem and offers nothing.

## Honest limitations

- **`Qwen/Qwen2.5-Coder-0.5B-Instruct` is small, and it is asked to do very
  little.** It is an instruction model, so it can be prompted to return a whole
  file, but 0.5B parameters on CPU produce a fix in 10–30 s and only for
  mechanical repairs. It is therefore consulted *after* the deterministic
  correctors, and only for `TypeError`, `SyntaxError`, `SingletonComparison`,
  `unused-import` and `unused-variable`. Findings that need a decision about
  intent — an undefined name, a wrong index, `x / 0`, a missing attribute — are
  never sent to it, because it answers them by inventing something: asked to fix
  an undefined `age` it returned `print(sys.argv[1])`, which parses, compiles and
  leaves the analyzer with nothing to complain about.
- **A model rewrite is accepted only if it is the same program with the reported
  line repaired.** It must keep the line count, touch at most three original
  lines, and touch nothing more than two lines away from the finding. Those
  gates, not the model's own confidence, are what keep a rewrite a fix.
- **The model does not write the explanation.** `explain.py` already says
  something specific to the reported line at no cost, while a 0.5B model needs
  about ten seconds per sentence on this machine and gets cut off mid-word inside
  the request budget. Set `CODEFIX_MODEL_EXPLAIN=1` with a larger
  `CODEFIX_MODEL_EXPLAIN_TIMEOUT` to see it.
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
4. locality (untrusted candidates only) — same line count, at most three
   original lines touched, nothing more than two lines from the finding,
5. re-analysis — does the original finding actually disappear,
6. warnings collected for display.

A correction is reported as `validated: true` only if all of these pass and the
original error is gone. Step 5 matches findings by error type and wording rather
than by line number, because deleting an unused import shifts every later line
up by one; and a candidate written by the language model must additionally
introduce no finding the original code did not have. Step 4 exists because of
what that gate alone missed: asked to fix `print(age)`, the previous checkpoint
answered `print(sys.argv[1])`, which silences the reported message, introduces
no new finding, and is not a fix at all — only its distance from the reported
line gives it away.

`safe_execute()` exists as an explicit, **disabled by default** contract for
sandboxed execution. Turning it on requires container or process isolation
with seccomp/AppArmor, a PID namespace, a read-only filesystem, dropped
capabilities, no network, and enforced CPU, memory and wall-clock limits.

## Project layout

```
backend/
  analyzer.py     AST, compiler and Pylint checks -> ranked findings
  fixes.py        Deterministic AST-positioned correctors + diff rows
  validator.py    Parse/compile/locality/re-analyze gate. No execution.
  rewrite_model.py  Lazy, thread-safe Qwen wrapper: prompts, budgets, admission
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

The model is ~1 GB and is loaded in the background at startup, so the server
accepts requests immediately. Analyses the deterministic correctors cover take
~0.1–3 s. The startup thread also runs one throwaway generation with a
real-length prompt, because the first `generate` call in a process costs ~25 s
here and every later one costs ~10 s — without that warm-up the first real
request is cut off mid-file by its own budget and reported as unfixable. So
startup reaches `model_state: ready` after ~30 s, and an analysis that ends with
a model fix takes ~10–25 s. Set `CODEFIX_ENABLE_MODEL=0` to run without the
model entirely — everything the deterministic correctors cover still works.

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
| `CODEFIX_ENABLE_MODEL` | `1` | `0` disables the model; the old `CODEFIX_ENABLE_CODET5` is still honoured |
| `CODEFIX_MODEL` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Model id or local path |
| `CODEFIX_MODEL_EXPLAIN` | `0` | `1` also asks the model for explanation wording |
| `CODEFIX_MODEL_TIMEOUT` | `25` | Seconds allowed for a rewrite before it is abandoned |
| `CODEFIX_MODEL_THREADS` | all cores | Torch threads used for generation |
| `CODEFIX_MODEL_DTYPE` | `float32` | `bfloat16` is faster but less precise on older CPUs |
| `CODEFIX_MODEL_MAX_CHARS` | `1400` | Longest snippet sent to the model |
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
  "ai": { "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "fix_source": "static-analysis" },
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
None`; missing colon; missing indented block; `=` inside `if`/`while` →
`==`; string + number → f-string or `str()`; Python 2 `print`; likely
identifier typos; removal of unused imports and variables.

Each is gated on the finding type. Clean code is never modified, and a fixer
that fails validation is discarded rather than shown as a fix. Type
inference is scoped per function: a name assigned inside one function never
leaks into another, and `count == 0` is a numeric comparison, not a
singleton check.

Four rules keep "corrected" from meaning "changed":

- **Every occurrence is fixed.** Two comparisons of one `input()` are two
  findings, so the fixer keeps going (up to five rounds) instead of handing back
  a "validated" fix with the same error still on the next line.
- **A fix may not change what a name means.** `int()` is only wrapped around an
  input when nothing else in the file needs that variable to stay a string. A
  quoted number that the converter reproduces exactly (`'1'` -> `1`) is rewritten
  with it; a quoted number that does not (`'01'`) or a comparison with a word
  (`choice == 'y'`) makes the conversion unsafe, so nothing is offered.
- **A fixable finding is reported over one that cannot be fixed.** If the
  highest-priority finding has no safe correction but another one does, that one
  becomes the headline and the summary names the finding left open. Everything
  remains in the issue list either way.
- **A generated rewrite is held to a stricter gate than an edit.** The model is
  only asked after every deterministic fixer has failed, and its answer is
  discarded unless it keeps the line count, touches at most three lines, stays
  within two lines of the finding, and leaves no new finding behind. `fix_source`
  says which of the two produced the code you are shown.

## Tests

```powershell
cd backend
.\venv\Scripts\python.exe test_analyzer.py    # detection
.\venv\Scripts\python.exe test_fixes.py       # correctors + validator
.\venv\Scripts\python.exe test_pipeline.py    # which finding is reported, and what is offered
.\venv\Scripts\python.exe test_rewrite_model.py # which findings the model may be asked about
.\venv\Scripts\python.exe test_api.py         # live server, end to end
.\venv\Scripts\python.exe test_contract.py    # UI/backend field contract
```

`test_api.py` and `test_contract.py` start and stop their own server on a free
port. `test_contract.py` exists because the UI reads nested paths like
`validation_details.reason`; a missing key renders as `undefined` in the
browser instead of raising, so it fails silently. The test asserts every field
the UI consumes is present.