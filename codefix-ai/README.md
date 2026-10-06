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
| One line, when no rule covers it? | `rewrite_model.py` → `validator.py` | The model proposes one line as JSON; the validator, not the model, decides whether it is a fix |
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
  little.** It is asked for exactly one thing: a JSON object naming the single
  line to replace, quoting that line, and giving the replacement. It is
  consulted *after* the deterministic correctors, and only for `TypeError`,
  `SyntaxError`, `SingletonComparison`, `unused-import` and `unused-variable`.
  Findings that need a decision about intent — an undefined name, a wrong
  index, `x / 0`, a missing attribute — are never sent to it, because it
  answers them by inventing something: asked to fix an undefined `age` it
  returned `print(sys.argv[1])`, which parses, compiles and leaves the
  analyzer with nothing to complain about.
- **The budget cuts answers short, and a cut answer is handled, not trusted.**
  Generation stops when `CODEFIX_MODEL_TIMEOUT` is spent, which on a CPU under
  memory pressure lands in the middle of the JSON. An answer whose first
  fields (`line`, `original`, `replacement`, `confidence`) are complete is
  still usable — the JSON is closed and every field is checked as usual, and
  the half-written `reason` is dropped. An answer cut inside the line it is
  replacing fails the exact-quote gate or does not parse, and the panel falls
  back to explaining the finding instead of offering a repair.
- **A model repair is accepted only if it is the same program with the reported
  line repaired.** It must quote the reported line exactly, stay within two
  lines of the finding, keep the line count, touch at most three original
  lines, and leave no new finding behind. Those gates, not the model's own
  confidence number, are what keep a generation a fix.
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
original error is gone. How far the candidate moved from the original is
measured as well: `validation_details.edit_lines` is the number of lines that
actually differ, so "one line changed" is a count rather than a claim. Step 5
matches findings by error type and wording rather
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
  fixes.py        Deterministic correctors: types, names, structure; diff rows
  validator.py    Parse/compile/locality/re-analyze gate. No execution.
  rewrite_model.py  Qwen wrapper: one-line JSON repair, budgets, admission gates
  explain.py      Deterministic explanation text
  main.py         FastAPI app: orchestration, repair status, caching, routes
frontend/
  src/App.js      Monaco editor, analysis panel, status chip, Apply Fix
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
startup reaches `model_state: ready` after ~30–45 s.

An analysis that reaches the model spends `CODEFIX_MODEL_TIMEOUT` (25 s) on it,
and that budget covers the prompt as well as the answer: on this machine the
prefill of ~300 tokens takes ~11 s, leaving ~14 s of decoding, which is enough
for the fields the gates need but usually not for the whole `reason` string.
When the answer arrives intact or is cut only inside `reason`, the result is a
validated one-line change reported as `ai_suggested`; when it is cut inside the
line itself, nothing is offered and the finding is reported with an
explanation instead. Either outcome arrives in ~25–35 s. Set
`CODEFIX_ENABLE_MODEL=0` to run without the model entirely — everything the
deterministic correctors cover still works.

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
| `CODEFIX_MODEL_TIMEOUT` | `25` | Seconds for one repair answer, prompt included; the answer is cut when it runs out |
| `CODEFIX_MODEL_EXPLAIN_TIMEOUT` | `10` | Seconds for one explanation answer |
| `CODEFIX_MODEL_MIN_CONFIDENCE` | `0.7` | Answers scoring lower than this are treated as guesses and discarded |
| `CODEFIX_MODEL_CONTEXT_LINES` | `4` | Lines shown either side of the finding in the model's window |
| `CODEFIX_MODEL_MAX_TOKENS` | `256` | Tokens the repair answer may generate |
| `CODEFIX_MODEL_MAX_CHARS` | `1400` | Longest snippet sent to the model (40 lines maximum) |
| `CODEFIX_MODEL_THREADS` | all cores | Torch threads used for generation |
| `CODEFIX_MODEL_DTYPE` | `float32` | Keep at `float32` on this machine: `bfloat16` halves the resident weights but took ~3 minutes to convert during load, against ~16 s |
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
  "repair": {
    "status": "fixed",
    "line": 1,
    "original": "age = input(\"Enter your age: \")",
    "replacement": "age = int(input(\"Enter your age: \"))",
    "source": "static-analysis",
    "message": null,
    "reason": "Wrapped the input in `int()` so the comparison with a number is valid.",
    "confidence": null,
    "edit_lines": 1
  },
  "ai": { "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "fix_source": "static-analysis" },
  "all_issues": [],
  "analysis_ms": 21
}
```

`repair.status` is what the result panel keys off:

| `status` | Meaning | `source` |
|---|---|---|
| `fixed` | A deterministic corrector produced the code | `static-analysis` |
| `ai_suggested` | The local model proposed the line; `confidence` and `reason` come from it | `local-llm` |
| `explanation_only` | The finding was detected and deliberately left alone; `message` says so and `reason` says why | `null` |
| `clean` | No finding, so nothing to correct | `null` |

`edit_lines` is measured, not claimed: it is the number of lines the correction
actually changes, which for every accepted model repair is one.

`400` empty code, `413` over `MAX_CODE_CHARS`, `500` unexpected failure.

### `GET /health`

Returns API version and model state (`disabled`, `not-loaded`, `loading`, `ready`, `failed`).

## Detected categories

Syntax errors, type errors, name errors, division by zero, index errors,
singleton comparison (`== None`), unused imports and variables, redefined
builtins, non-callable and missing-attribute access, broad exception
handlers — plus everything Pylint's error and warning checks report.

## Corrections offered automatically

`input()` wrapped in `int()`/`float()`; `input()` read into a name and then
used in arithmetic (`age = input()` followed by `age + 10` → `int(input())` at
the binding); `== None`/`None ==` → `is None`/`is not None`; missing colon; a
body left unindented under a header indented into its block, or `pass` when the
block is genuinely empty; a bracket left open at the end of the file closed;
`=` inside `if`/`while` → `==`; string + number → f-string or `str()`; Python 2
`print`; likely identifier typos; removal of unused imports and variables.

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
- **A generated repair is held to a stricter gate than an edit.** The model is
  only asked after every deterministic fixer has failed, and it must name the
  reported line and quote it exactly. Its answer is discarded unless it keeps
  the line count, touches at most three lines, stays within two lines of the
  finding, and leaves no new finding behind. `repair.status` (`fixed` or
  `ai_suggested`) and `fix_source` say which of the two produced the code you
  are shown; `repair.edit_lines` says how much of it changed.

## Tests

```powershell
cd backend
.\venv\Scripts\python.exe test_analyzer.py    # detection
.\venv\Scripts\python.exe test_fixes.py       # correctors + validator
.\venv\Scripts\python.exe test_pipeline.py    # which finding is reported, and what is offered
.\venv\Scripts\python.exe test_rewrite_model.py # which findings the model may be asked about
.\venv\Scripts\python.exe test_repair.py      # repair status, model admission gates, cut answers
.\venv\Scripts\python.exe test_api.py         # live server, end to end
.\venv\Scripts\python.exe test_contract.py    # UI/backend field contract
```

`test_api.py` and `test_contract.py` start and stop their own server on a free
port. `test_contract.py` exists because the UI reads nested paths like
`validation_details.reason`; a missing key renders as `undefined` in the
browser instead of raising, so it fails silently. The test asserts every field
the UI consumes is present. `test_repair.py` never loads weights: model
answers are fed to the parser as text, and the pipeline cases it covers are
the findings that must never reach the model at all.
