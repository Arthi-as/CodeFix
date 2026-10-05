"""CodeFix AI API.

Pipeline for ``POST /analyze``:

    code -> static analysis -> candidate correction -> CodeT5 (advisory)
          -> validation -> structured result

Static analysis is the only source of truth for *what* is wrong and *where*.
CodeT5 may improve the wording or offer an alternative patch, and the
validator has the final say on whether anything may be called a fix.
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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import codet5
from analyzer import MAX_CODE_CHARS, analyze_code
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
        status = await run_in_threadpool(codet5.warm_up)
        app.state.model_ready = bool(status.get("loaded"))
        logger.info("CodeT5 warm-up finished: %s", status)

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
            "ai": _ai_metadata(explanation_source="static-analysis", fix_source=None),
            "analysis_ms": int((time.perf_counter() - started) * 1000),
        }

    # 1. Deterministic explanation, optionally enriched by CodeT5.
    parts = split_explanation(analysis)
    explanation = build_explanation(analysis)
    explanation_source = "static-analysis"

    model_explanation = codet5.explain_with_codet5(code, analysis)
    if model_explanation:
        explanation = f"{explanation} CodeT5 adds: {model_explanation}"
        explanation_source = "codet5"

    # 2. Candidate corrections, cheapest and most reliable first.
    # The static fixer is exact, so when it validates there is no reason to
    # spend seconds asking the model for an alternative.
    candidates: List[Dict[str, Any]] = []

    static_fix = propose_fix(code, analysis)
    if static_fix is not None:
        candidates.append(
            {"code": static_fix.code, "source": "static-analysis", "summary": static_fix.summary}
        )

    chosen: Optional[Dict[str, Any]] = None
    rejected: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}

    for candidate in candidates:
        # A rewrite from the language model has to clear a stricter gate than a
        # deterministic edit: it must not introduce findings the original did
        # not have, or silence one message while leaving the same bug.
        report = validate_correction(
            code,
            candidate["code"],
            analysis,
            trusted=candidate["source"] == "static-analysis",
        )
        if report.validated:
            chosen = candidate
            validation = report.to_dict()
            break
        rejected.append({"candidate": candidate, "report": report})
        logger.info("rejected %s candidate: %s", candidate["source"], report.reason)

    if chosen is None:
        # Only now is it worth asking the model.
        model_fix = codet5.propose_codet5_correction(code, analysis)
        if model_fix:
            report = validate_correction(code, model_fix, analysis, trusted=False)
            if report.validated:
                chosen = {"code": model_fix, "source": "codet5", "summary": "Proposed by CodeT5."}
                validation = report.to_dict()
            else:
                rejected.append(
                    {"candidate": {"code": model_fix, "source": "codet5"}, "report": report}
                )
                logger.info("rejected codet5 candidate: %s", report.reason)

    validated = chosen is not None
    corrected_code = chosen["code"] if chosen else code

    if not validated:
        # Show the best near-miss, clearly marked as not a fix. A deterministic
        # edit that parses but does not clear the finding is still informative;
        # model output is shown only when it did clear it, because a fragment
        # that merely parses (`print(age)` -> `defprint(name)`) is worse than
        # nothing to copy.
        for entry in rejected:
            report = entry["report"]
            if entry["candidate"]["source"] == "codet5" and report.issue_resolved is False:
                continue
            if report.syntax_valid and report.compiles:
                corrected_code = entry["candidate"]["code"]
                validation = report.to_dict()
                break
        if not validation:
            report = ValidationReport()
            report.reason = "No automatic correction is safe to offer for this finding."
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
        "ai": _ai_metadata(explanation_source, chosen["source"] if chosen else None),
        "analysis_ms": int((time.perf_counter() - started) * 1000),
    }


def _ai_metadata(explanation_source: str, fix_source: Optional[str]) -> Dict[str, Any]:
    status = codet5.model_status()
    return {
        "model": status["model"],
        "model_loaded": status["loaded"],
        "model_state": status["state"],
        "device": status["device"],
        "explanation_source": explanation_source,
        "fix_source": fix_source,
        "note": (
            "Static analysis determines the error type, line and severity. CodeT5 is advisory: "
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
        "model": codet5.model_status(),
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