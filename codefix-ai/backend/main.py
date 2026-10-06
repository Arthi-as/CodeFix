"""CodeFix AI API.

Pipeline for ``POST /analyze``:

    code -> static analysis -> deterministic correction -> local model (advisory)
          -> validation -> structured result

Static analysis is the only source of truth for *what* is wrong and *where*, and
the deterministic correctors in :mod:`fixes` come first because they are edits
rather than generations. The local model in :mod:`rewrite_model` may improve the
wording or offer an alternative patch, and the validator has the final say on
whether anything may be called a fix.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import rewrite_model
from analyzer import MAX_CODE_CHARS, analyze_code, find_issues
from explain import build_explanation, split_explanation
from fixes import diff_summary, propose_fix
from validator import ValidationReport, validate_correction, validate_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("codefix")

API_VERSION = "1.0.0"
VALIDATION_FAILED_NOTICE = "Suggested fix - validation unsuccessful."
CACHE_SIZE = 128

DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

# Analysis is deterministic, so identical input always yields identical output.
# Re-running the same code while iterating is common, so cache it.
_cache: "OrderedDict[str, AnalyzeResponse]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(code: str) -> Optional[AnalyzeResponse]:
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    with _cache_lock:
        result = _cache.get(key)
        if result is not None:
            _cache.move_to_end(key)
    return result


def _cache_put(code: str, response: AnalyzeResponse) -> None:
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    with _cache_lock:
        _cache[key] = response
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model up before the first request arrives."""
    app.state.model_ready = False

    async def _warm() -> None:
        status = await run_in_threadpool(rewrite_model.warm_up)
        app.state.model_ready = bool(status.get("loaded"))
        logger.info("rewrite model warm-up finished: %s", status)

    # Load in the background so the server accepts traffic immediately.
    task = asyncio.create_task(_warm())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass


app = FastAPI(
    title="CodeFix AI",
    description=(
        "An AI-assisted Python debugging system that identifies problematic code lines, "
        "explains coding errors, and suggests validated corrections using static analysis "
        "and a pretrained code language model."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.environ.get("CODEFIX_CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Python source to analyze.")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 'age = input("Enter your age: ")\n\nif age > 18:\n    print("Adult")'
            }
        }


class Issue(BaseModel):
    line: int
    column: int
    end_line: int
    error_type: str
    severity: str
    category: str
    message: str
    suggestion: Optional[str] = None
    detected_by: str


class DiffRow(BaseModel):
    kind: str = Field(..., description="added, removed or context")
    line: int
    text: str


class AnalyzeResponse(BaseModel):
    has_error: bool
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    error_type: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    detected_by: Optional[str] = Field(
        default=None,
        description="Which analysis layer produced the finding: syntax, semantic or pylint.",
    )
    message: str
    suggestion: Optional[str] = None
    explanation: Optional[str] = None
    explanation_title: Optional[str] = None
    corrected_code: str
    validated: bool
    validation_details: Dict[str, Any] = Field(default_factory=dict)
    fix_summary: Optional[str] = None
    diff: List[DiffRow] = Field(default_factory=list)
    ai: Dict[str, Any] = Field(default_factory=dict)
    all_issues: List[Issue] = Field(default_factory=list)
    analysis_ms: int = 0


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# deterministic fixing
# --------------------------------------------------------------------------

# Enough rounds for "one bug per line" pasted code without looping on a fixer
# that oscillates.
MAX_FIX_ROUNDS = 5

# Findings a deterministic fixer knows how to handle. Used only as a cheap
# pre-filter before building a context for the other findings.
_FIXABLE_TYPES = {
    "TypeError",
    "SingletonComparison",
    "unused-import",
    "unused-variable",
    "undefined-variable",
}


def _promotable_context(code: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The highest ranked *other* finding that has a deterministic fix.

    Reporting "nothing to offer" while a one-line unused import sits in the same
    file is what makes the tool look broken on code it was not written for, so a
    fixable finding is allowed to become the headline. The finding it replaced
    stays in ``all_issues``, which the panel already lists.
    """
    try:
        issues = find_issues(code)
    except Exception:
        return None

    for issue in issues[1:]:
        if issue.error_type not in _FIXABLE_TYPES:
            continue
        context = {**analysis, **issue.to_dict(), "meta": dict(issue.meta)}
        if propose_fix(code, context) is not None:
            return context
    return None


def _best_static_fix(
    code: str, analysis: Dict[str, Any]
) -> Optional[Tuple[Dict[str, Any], str, List[str], ValidationReport]]:
    """Fix ``code`` deterministically, or return ``None`` if nothing is safe.

    Returns the (possibly promoted) analysis, the corrected source, one summary
    per applied edit and the validation report for the final result.
    """
    context = analysis
    outcome = propose_fix(code, context)
    if outcome is None:
        promoted = _promotable_context(code, analysis)
        if promoted is None:
            return None
        context = promoted
        outcome = propose_fix(code, context)
        if outcome is None:
            return None

    corrected = outcome.code
    summaries = [outcome.summary]
    report = validate_correction(code, corrected, context, trusted=True)
    if not report.validated:
        return None

    # Keep going while the same kind of finding is still there. Two comparisons
    # of one input() are two findings; fixing only the first and calling the
    # result validated is how "it worked on the demo" stops being true.
    repeated = sum(
        1
        for issue in (context.get("all_issues") or [])
        if issue.get("error_type") == context.get("error_type")
    )
    while repeated > 1 and len(summaries) < MAX_FIX_ROUNDS:
        round_analysis = analyze_code(corrected)
        if not round_analysis.get("has_error"):
            break
        outcome = propose_fix(corrected, round_analysis)
        if outcome is None or outcome.code == corrected:
            break
        round_report = validate_correction(
            corrected, outcome.code, round_analysis, trusted=True
        )
        if not round_report.validated:
            break
        corrected = outcome.code
        summaries.append(outcome.summary)
        repeated = sum(
            1
            for issue in (round_analysis.get("all_issues") or [])
            if issue.get("error_type") == round_analysis.get("error_type")
        )

    if len(summaries) > 1:
        # Re-check the accumulated result against the originally reported finding.
        report = validate_correction(code, corrected, context, trusted=True)
        if not report.validated:
            return None

    return context, corrected, summaries, report


# --------------------------------------------------------------------------
# repair status
#
# The panel needs to tell three outcomes apart without reading validation
# internals: something was corrected (`fixed`), the model proposed the
# correction (`ai_suggested`), or an error was found and deliberately left
# alone (`explanation_only`).
# --------------------------------------------------------------------------

EXPLANATION_ONLY_MESSAGE = (
    "The error was detected, but CodeFix could not safely determine the intended correction."
)

REASON_NO_CORRECTION = "No automatic correction is safe to offer for this finding."

# What the *programmer* has to decide, for the finding types no automatic rule
# may touch. Saying which decision is missing is the whole point of the
# explanation-only state: "no fix" without this reads as "your code is fine".
NO_FIX_REASONS = {
    "undefined-variable": (
        "The intended variable cannot be determined from the code alone, so no name "
        "was substituted for it."
    ),
    "ZeroDivisionError": (
        "Changing the divisor would change what the program calculates, so no divisor "
        "was substituted for it."
    ),
    "IndexError": "Which index was intended cannot be determined from the code alone.",
    "no-member": "Which attribute was intended cannot be determined from the code alone.",
    "not-callable": "What to call instead cannot be determined from the code alone.",
    "unsubscriptable-object": (
        "What should be indexed instead cannot be determined from the code alone."
    ),
}


def _repair_edit(original: str, corrected: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Line number, original text and replacement text of the first change.

    Most corrections replace one line; when a fix spans several (the rounds in
    :func:`_best_static_fix`, or an import removal with nothing added) the pair
    is ``None`` and the diff carries the detail instead.
    """
    rows = [row for row in diff_summary(original, corrected) if row["kind"] in ("added", "removed")]
    removed = [row for row in rows if row["kind"] == "removed"]
    added = [row for row in rows if row["kind"] == "added"]
    if len(removed) == 1 and len(added) == 1:
        if removed[0]["line"] == added[0]["line"]:
            return removed[0]["line"], removed[0]["text"], added[0]["text"]
        # Different numbering before and after: report the line as it stands in
        # the corrected file, which is what the user is about to look at.
        return added[0]["line"], removed[0]["text"], added[0]["text"]
    if len(removed) == 1 and not added:
        return removed[0]["line"], removed[0]["text"], None
    if len(added) == 1 and not removed:
        return added[0]["line"], None, added[0]["text"]
    return None, None, None


def _build_repair(
    code: str,
    corrected: str,
    analysis: Dict[str, Any],
    chosen: Optional[Dict[str, Any]],
    validation: Dict[str, Any],
    model_repair: Optional[Any] = None,
    decline_reason: str = "",
) -> Dict[str, Any]:
    """The repair block the panel renders. Additive: existing keys are untouched."""
    if chosen is None:
        error_type = analysis.get("error_type") or ""
        reason = validation.get("reason") or ""
        if decline_reason and validation.get("reason") in ("", REASON_NO_CORRECTION):
            reason = decline_reason
        policy = NO_FIX_REASONS.get(error_type)
        return {
            "status": "explanation_only",
            "line": analysis.get("line"),
            "original": None,
            "replacement": None,
            "source": None,
            "message": EXPLANATION_ONLY_MESSAGE,
            "reason": policy or reason or EXPLANATION_ONLY_MESSAGE,
            "confidence": None,
            "edit_lines": validation.get("edit_lines"),
        }

    line, original, replacement = _repair_edit(code, corrected)
    if model_repair is not None:
        line = model_repair.line
        original = model_repair.original
        replacement = model_repair.replacement
        confidence: Optional[float] = round(model_repair.confidence, 2)
        reason = model_repair.reason or chosen["summary"]
        status = "ai_suggested"
    else:
        confidence = None
        reason = chosen["summary"]
        status = "fixed"
    return {
        "status": status,
        "line": line,
        "original": original,
        "replacement": replacement,
        "source": chosen["source"],
        "message": None,
        "reason": reason,
        "confidence": confidence,
        "edit_lines": validation.get("edit_lines"),
    }


def run_pipeline(code: str) -> Dict[str, Any]:
    """Blocking analysis pipeline. Runs in a worker thread."""
    started = time.perf_counter()

    analysis = analyze_code(code)

    if not analysis["has_error"]:
        report = validate_code(code)
        return {
            **analysis,
            "explanation": (
                "The file parses and Pylint reported no errors or warnings. That is not proof "
                "the logic is correct - static analysis cannot see runtime behaviour."
            ),
            "explanation_title": "No issues found",
            "corrected_code": code,
            "validated": bool(report.get("validated")),
            "validation_details": report,
            "fix_summary": None,
            "diff": [],
            "repair": {
                "status": "clean",
                "line": None,
                "original": None,
                "replacement": None,
                "source": None,
                "message": "No error was detected, so there is nothing to correct.",
                "reason": "",
                "confidence": None,
                "edit_lines": None,
            },
            "ai": _ai_metadata(explanation_source="static-analysis", fix_source=None),
            "analysis_ms": int((time.perf_counter() - started) * 1000),
        }

    # 1. Deterministic fix first: it decides *which* finding is reported, because
    #    a finding nobody can fix is a worse headline than one we can correct.
    chosen: Optional[Dict[str, Any]] = None
    rejected: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}

    static = _best_static_fix(code, analysis)
    if static is not None:
        promoted, fixed_code, summaries, report = static
        demoted = analysis if promoted is not analysis else None
        chosen = {
            "code": fixed_code,
            "source": "static-analysis",
            "summary": " ".join(summaries),
        }
        if demoted is not None:
            # Say out loud what was *not* fixed, so a promoted fix never reads as
            # "your code is clean now". The demoted finding also stays at the top
            # of all_issues, which the panel lists.
            chosen["summary"] = (
                f"{chosen['summary']} The finding on line {demoted['line']} "
                f"({demoted['error_type']}) has no safe automatic correction and is still open."
            )
            analysis = promoted
        validation = report.to_dict()

    # 2. Deterministic explanation, optionally enriched by the model.
    parts = split_explanation(analysis)
    explanation = build_explanation(analysis)
    explanation_source = "static-analysis"

    model_explanation = rewrite_model.explain_with_model(code, analysis)
    if model_explanation:
        explanation = f"{explanation} Model adds: {model_explanation}"
        explanation_source = "local-llm"

    # 3. Nothing deterministic worked, so it is worth asking the model for one
    #    line to replace - never for a rewritten program.
    model_repair = None
    decline_reason = ""
    if chosen is None:
        model_repair, decline_reason = rewrite_model.propose_local_repair(code, analysis)
        if model_repair is not None:
            report = validate_correction(code, model_repair.code, analysis, trusted=False)
            if report.validated:
                chosen = {
                    "code": model_repair.code,
                    "source": "local-llm",
                    "summary": (
                        f"Rewritten by the local model on line {model_repair.line}: "
                        f"{model_repair.reason}"
                        if model_repair.reason
                        else f"Rewritten by the local model on line {model_repair.line}."
                    ),
                }
                validation = report.to_dict()
            else:
                rejected.append(
                    {
                        "candidate": {"code": model_repair.code, "source": "local-llm"},
                        "report": report,
                    }
                )
                logger.info("rejected model repair: %s", report.reason)
                decline_reason = report.reason or decline_reason
                model_repair = None

    validated = chosen is not None
    corrected_code = chosen["code"] if chosen else code

    if not validated:
        # Show the best near-miss, clearly marked as not a fix. A deterministic
        # edit that parses but does not clear the finding is still informative;
        # model output is shown only when it both stayed near the reported line
        # and cleared the finding, because a fragment that merely parses
        # (`print(age)` -> `defprint(name)`) is worse than nothing to copy.
        for entry in rejected:
            report = entry["report"]
            if entry["candidate"]["source"] != "static-analysis":
                if not report.local or report.issue_resolved is False:
                    continue
            if report.syntax_valid and report.compiles:
                corrected_code = entry["candidate"]["code"]
                validation = report.to_dict()
                break
        if not validation:
            report = ValidationReport()
            report.reason = REASON_NO_CORRECTION
            report.warnings = analysis.get("warnings", [])
            validation = report.to_dict()

    message = analysis["message"]
    if not validated and corrected_code != code:
        message = f"{message} {VALIDATION_FAILED_NOTICE}"

    return {
        **analysis,
        "explanation": explanation,
        "explanation_title": parts["title"],
        "corrected_code": corrected_code,
        "validated": validated,
        "validation_details": validation,
        "fix_summary": chosen["summary"] if chosen else None,
        "diff": diff_summary(code, corrected_code) if corrected_code != code else [],
        "repair": _build_repair(
            code, corrected_code, analysis, chosen, validation,
            model_repair=model_repair, decline_reason=decline_reason,
        ),
        "ai": _ai_metadata(explanation_source, chosen["source"] if chosen else None),
        "analysis_ms": int((time.perf_counter() - started) * 1000),
    }


def _ai_metadata(explanation_source: str, fix_source: Optional[str]) -> Dict[str, Any]:
    status = rewrite_model.model_status()
    return {
        "model": status["model"],
        "model_loaded": status["loaded"],
        "model_state": status["state"],
        "device": status["device"],
        "explanation_source": explanation_source,
        "fix_source": fix_source,
        "note": (
            "Static analysis determines the error type, line and severity, and the "
            "deterministic correctors are tried first. The local model is advisory: "
            "its output is only used when it passes validation."
        ),
    }


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    return {
        "name": "CodeFix AI",
        "description": (
            "An AI-assisted Python debugging system that identifies problematic code lines, "
            "explains coding errors, and suggests validated corrections using static analysis "
            "and a pretrained code language model."
        ),
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {"analyze": "POST /analyze", "health": "GET /health"},
    }


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": API_VERSION,
        "model": rewrite_model.model_status(),
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze Python source and return a validated diagnosis."""
    code = request.code

    if code is None or not code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")
    if len(code) > MAX_CODE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Code is {len(code)} characters; the limit is {MAX_CODE_CHARS}.",
        )

    logger.info("analyze: %d chars", len(code))

    cached = _cache_get(code)
    if cached is not None:
        logger.info("cache hit")
        return cached

    try:
        result = await run_in_threadpool(run_pipeline, code)
    except Exception as exc:  # pragma: no cover - unexpected pipeline failure
        logger.exception("analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    response = AnalyzeResponse(**result)
    _cache_put(code, response)

    logger.info(
        "analyze done in %dms: has_error=%s line=%s type=%s validated=%s",
        result.get("analysis_ms", 0),
        result.get("has_error"),
        result.get("line"),
        result.get("error_type"),
        result.get("validated"),
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("CODEFIX_HOST", "127.0.0.1"),
        port=int(os.environ.get("CODEFIX_PORT", "8000")),
    )