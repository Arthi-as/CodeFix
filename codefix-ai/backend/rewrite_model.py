"""Local instruction-tuned repair model.

This module is deliberately *advisory*. It proposes; :mod:`validator` decides.
The previous checkpoint here (``Salesforce/codet5-base``) is a pretrained
code-completion model, not an instruction model: asked to repair a file it
emitted unrelated code, and ``print(age)`` came back as ``defprint(name)`` - text
that parses, compiles and silences the reported message, so every mechanical gate
passed. An instruct-tuned checkpoint fixes most of that, but not all of it, so
three rules keep it in its lane:

* it only sees findings whose correction is mechanical (:data:`ELIGIBLE_TYPES`),
  because a missing name, an out-of-range index or a zero divisor is a decision
  the programmer has to make, and the model answers those by inventing something;
* it is never asked for a whole program. It is shown a window around the
  reported line and asked for one line to replace - the rest of the file is
  spliced back byte-for-byte by :func:`_apply_repair`, so an unrelated rewrite
  is not something the model can express;
* every candidate it produces must still clear the untrusted gates in
  :mod:`validator` - syntax, no new findings, and the locality gate that rejects
  an edit outside the reported region.

``CODEFIX_MODEL`` points at whatever checkpoint suits the machine. On CPU,
``Qwen/Qwen2.5-Coder-0.5B-Instruct`` is roughly 1 GB and answers a one-line
repair in about 10 s; 1.5B or 7B are better if there is RAM or a GPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from validator import UNTRUSTED_LINE_RADIUS


_LOG = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("CODEFIX_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")


def _enabled() -> bool:
    """``CODEFIX_ENABLE_MODEL`` with the old ``CODEFIX_ENABLE_CODET5`` honoured."""
    raw = os.environ.get("CODEFIX_ENABLE_MODEL")
    if raw is None:
        raw = os.environ.get("CODEFIX_ENABLE_CODET5", "1")
    return raw not in ("0", "false", "False")


ENABLED = _enabled()

DTYPE = os.environ.get("CODEFIX_MODEL_DTYPE", "float32")
THREADS = int(os.environ.get("CODEFIX_MODEL_THREADS", "0")) or None

# A rewrite has to arrive inside the request budget. Generation stops when the
# budget is spent; a truncated answer does not parse, so it fails validation and
# nothing is offered. The two calls are budgeted separately because a request
# pays for both, and the explanation is the one users are more likely to be
# waiting on.
FIX_TIME_BUDGET_S = float(os.environ.get("CODEFIX_MODEL_TIMEOUT", "25"))
EXPLAIN_TIME_BUDGET_S = float(os.environ.get("CODEFIX_MODEL_EXPLAIN_TIMEOUT", "10"))

MAX_NEW_TOKENS_FIX = int(os.environ.get("CODEFIX_MODEL_MAX_TOKENS", "256"))
MAX_NEW_TOKENS_EXPLAIN = int(os.environ.get("CODEFIX_MODEL_MAX_TOKENS_EXPLAIN", "140"))

# Off by default. The static explanation in :mod:`explain` is specific to the
# reported line and free, while a 0.5B model on CPU needs ~10 s for one sentence
# and gets cut off mid-word inside the request budget. Enable it with a larger
# ``CODEFIX_MODEL_EXPLAIN_TIMEOUT`` if the machine can afford to wait.
EXPLAIN_ENABLED = os.environ.get("CODEFIX_MODEL_EXPLAIN", "0") not in ("0", "false", "False")

# Beyond this the model cannot finish a full-file rewrite at CPU speed, and a
# partial file is worse than no answer.
MAX_CHARS = int(os.environ.get("CODEFIX_MODEL_MAX_CHARS", "1400"))
MAX_LINES = 40

MIN_EXPLANATION_CHARS = 40
MAX_EXPLANATION_CHARS = 400

# Findings whose fix is mechanical. Everything else is a decision, and a model
# asked for one invents a plausible-looking answer: `print(age)` came back as
# `print(sys.argv[1])`, `name[10]` came back as `name[10:]`.
ELIGIBLE_TYPES = {
    "TypeError",
    "SyntaxError",
    "SingletonComparison",
    "unused-import",
    "unused-variable",
}

# Below this the model is saying it is guessing, and a guess is not a repair.
MIN_CONFIDENCE = float(os.environ.get("CODEFIX_MODEL_MIN_CONFIDENCE", "0.7"))

# How much of the file the model is shown. The reported line sits in the
# middle; everything outside the window is never the model's business. The
# radius is small because the prompt is paid for twice over on CPU: prefill of
# the window is the single biggest cost inside the request budget, and a wide
# window buys context the model has no time to decode an answer for.
WINDOW_RADIUS = int(os.environ.get("CODEFIX_MODEL_CONTEXT_LINES", "4"))
MAX_REPAIR_LENGTH = 160


@dataclass
class LocalRepair:
    """One line the model says should be different, with its justification."""

    line: int
    original: str
    replacement: str
    reason: str = ""
    confidence: float = 0.0
    code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
        }


_SYSTEM_REPAIR = (
    "Fix exactly one line of a Python file. Answer with one JSON object only:\n"
    '{"line": 0, "confidence": 0.9, "original": "", "replacement": "", "reason": ""}\n'
    "line = the line you replace. original = that line copied exactly as written, "
    "not its output. replacement = the corrected line, no newline. confidence = 0.9 "
    "when the fix follows from the error, below 0.7 when you are guessing. reason = "
    "at most 5 words.\n"
    "No other line may change. Never invent variables, imports or values, never "
    "delete or comment out code, never use pass, never change a number or string "
    "just to make the program run.\n"
    'If it cannot be determined, answer {"no_fix": true, "reason": "why"}.'
)


_SYSTEM_EXPLAIN = (
    "You explain Python bugs to a beginner. Answer in two or three plain sentences. "
    "Do not use code blocks, do not repeat the file, and do not add suggestions "
    "beyond the one asked about."
)


class LocalRewriteModel:
    """Lazily loaded causal LM, safe to share across requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._state: str = "not-loaded"
        self._detail: str = ""
        self.device: str = "cpu"

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> bool:
        """Load the model once. Returns whether it is usable."""
        if not ENABLED:
            self._state = "disabled"
            self._detail = "CODEFIX_ENABLE_MODEL is set to 0."
            return False
        if self._model is not None:
            return True
        if self._state in ("failed", "disabled"):
            return False

        with self._lock:
            if self._model is not None:
                return True
            try:
                import torch  # noqa: PLC0415
                from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

                if THREADS:
                    torch.set_num_threads(THREADS)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    # low_cpu_mem_usage streams the weights instead of building a
                    # full copy in RAM first, which matters on a small machine.
                    low_cpu_mem_usage=True,
                    torch_dtype=getattr(torch, DTYPE, torch.float32),
                )
                model.to(self.device)
                model.eval()
                self._model = model
                self._state = "ready"
                self._detail = f"{MODEL_NAME} loaded on {self.device} as {DTYPE}"
                _LOG.info("rewrite model %s", self._detail)
                self._warm_generation(torch)
                return True
            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._state = "failed"
                self._detail = f"{type(exc).__name__}: {exc}"
                _LOG.warning("rewrite model unavailable, static analysis only: %s", self._detail)
                return False

    def _render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # A base checkpoint without a chat template still gets the turn
            # markers most instruct models were trained with.
            return (
                "<|im_start|>system\n" + system + "<|im_end|>\n"
                "<|im_start|>user\n" + user + "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    def _warm_generation(self, torch) -> None:
        """Run one throwaway generation so the first real request is not the one
        that pays for it.

        Measured here: the first ``generate`` call built from a full-length chat
        prompt costs ~25 s and every later one costs ~10 s, whatever
        ``max_new_tokens`` is. It is the prompt, not the answer — a two-token
        warm-up on a five-token prompt takes 2.7 s and leaves the penalty in
        place, while a prompt of the same length as a real request pays it once
        at load time. The generation is cut off after a single token because the
        cost being paid here is the prefill and the cache allocation, not the
        decoding.
        """
        if self._model is None or self._tokenizer is None:
            return
        from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: PLC0415

        class _OneToken(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return input_ids.shape[1] >= 2

        filler = "\n".join(f"value_{index} = {index}" for index in range(20))
        started = time.perf_counter()
        try:
            prompt = self._render(
                _SYSTEM_REPAIR,
                f"Line 20 has an error: TypeError\n\n{filler}\n\nReturn the whole file.\n",
            )
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                self._model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS_FIX,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    stopping_criteria=StoppingCriteriaList([_OneToken()]),
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            _LOG.info(
                "generation path warm after %.1f s (%d prompt tokens)",
                time.perf_counter() - started,
                encoded["input_ids"].shape[1],
            )
        except Exception as exc:
            # A failed warm-up only costs latency, so it must not disable the model.
            _LOG.warning("generation warm-up skipped: %s", exc)

    def is_available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self) -> Dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "state": self._state,
            "loaded": self.is_available(),
            "enabled": ENABLED,
            "device": self.device,
            "detail": self._detail,
        }

    # -- generation --------------------------------------------------------

    def _chat(self, system: str, user: str, max_new_tokens: int, budget_s: float) -> str:
        if not self.is_available() and not self.load():
            return ""
        try:
            import torch  # noqa: PLC0415
            from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: PLC0415
        except Exception as exc:
            _LOG.warning("transformers unavailable: %s", exc)
            return ""

        deadline = time.perf_counter() + budget_s

        class _OutOfTime(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return time.perf_counter() > deadline

        prompt = self._render(system, user)

        try:
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    # The checkpoint ships sampling defaults; with greedy decoding
                    # they only produce a warning.
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=1.05,
                    stopping_criteria=StoppingCriteriaList([_OutOfTime()]),
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        except Exception as exc:
            _LOG.warning("rewrite model generation failed: %s", exc)
            return ""
        return text.strip()

    # -- tasks -------------------------------------------------------------

    def explain(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Model-written description of the finding, or ``None`` to keep ours."""
        if not ENABLED or not self._within_budget(code):
            return None
        user = (
            "Explain this Python problem to a beginner in two or three sentences.\n\n"
            f"Error on line {analysis.get('line')}: {analysis.get('error_type')}\n"
            f"{analysis.get('message') or ''}\n\n"
            f"File:\n```python\n{code}\n```"
        )
        text = self._chat(_SYSTEM_EXPLAIN, user, MAX_NEW_TOKENS_EXPLAIN, EXPLAIN_TIME_BUDGET_S)
        if not _looks_like_prose(text):
            return None
        return text

    def propose_repair(self, code: str, analysis: Dict[str, Any]) -> Tuple[Optional[LocalRepair], str]:
        """Ask for one line to replace. Returns ``(repair, reason)``.

        ``repair`` is ``None`` whenever the answer is not usable, and ``reason``
        says why, so the panel can show *why* nothing was offered instead of an
        empty box. The returned candidate is still *unvalidated*: the caller
        must run :func:`validator.validate_correction` before showing it.
        """
        if not ENABLED:
            return None, "the model is disabled"
        error_type = analysis.get("error_type") or ""
        if error_type not in ELIGIBLE_TYPES:
            _LOG.info("model not asked about %s: it needs a decision, not a rewrite", error_type)
            return None, f"{error_type} needs a decision about intent, not a rewrite"
        line = analysis.get("line")
        if not isinstance(line, int) or not 1 <= line <= len(code.splitlines()):
            return None, "the reported line is not inside the file"
        if not self._within_budget(code):
            return None, "the file is too long to repair at CPU speed"

        # The line to copy is spelled out on its own: asked to quote it from the
        # window alone, this checkpoint quoted the line's *output* instead
        # (`Ada is 36` for `print(name + ' is ' + age)`), which never matches.
        user = (
            f"Error on line {line}: {error_type}\n"
            f"{analysis.get('message') or ''}\n\n"
            f"Line {line} to replace, copied as written: {code.splitlines()[line - 1]}\n\n"
            f"File window:\n{_repair_window(code, line)}"
        )
        answer = self._chat(_SYSTEM_REPAIR, user, MAX_NEW_TOKENS_FIX, FIX_TIME_BUDGET_S)
        return parse_repair_response(answer, code, analysis)

    def propose_fix(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Full-file candidate from :meth:`propose_repair`, *unvalidated*.

        The caller must run :func:`validator.validate_correction` before showing
        it. Kept for callers that only need the text.
        """
        repair, _reason = self.propose_repair(code, analysis)
        return repair.code if repair else None


    def _within_budget(self, code: str) -> bool:
        if len(code) > MAX_CHARS or len(code.splitlines()) > MAX_LINES:
            _LOG.info("model skipped: snippet is too long to rewrite at CPU speed")
            return False
        return True


_ENGINE = LocalRewriteModel()


def get_engine() -> LocalRewriteModel:
    return _ENGINE


# --------------------------------------------------------------------------
# output gates
# --------------------------------------------------------------------------

_CODE_CHARS = re.compile(r"[{}()\[\];]|=>|::")
_REPEATED_CHAR = re.compile(r"(.)\1{6,}")
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*")


def _looks_like_prose(text: str) -> bool:
    """Reject degenerate output before it reaches the panel."""
    if not text:
        return False
    stripped = text.strip()
    if not MIN_EXPLANATION_CHARS <= len(stripped) <= MAX_EXPLANATION_CHARS:
        return False
    if " " not in stripped:
        return False
    if len(re.findall(r"[A-Za-z]{2,}", stripped)) < 8:
        return False
    if _REPEATED_CHAR.search(stripped) or _CODE_CHARS.search(stripped):
        return False
    return True


def _extract_code(text: str) -> str:
    """Pull the file out of the answer, fenced or not."""
    if not text:
        return ""
    blocks = _FENCE.split(text)
    # blocks: [before, "python\n<file>", between, "<file>", after] for a fenced
    # answer; odd indices are the fenced bodies.
    if len(blocks) >= 3 and blocks[1].strip():
        body = blocks[1]
        return body[body.index("\n") + 1 :] if "\n" in body else body
    return text.strip()


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def _close_truncated_json(text: str) -> Optional[str]:
    """Closing characters for an object the answer did not finish.

    The time budget cuts generation mid-object, and the fields written first
    (line, original, replacement, confidence) are the ones already complete.
    Completing the *syntax* invents no content: everything between the last
    quote and the cut stays exactly as the model wrote it, and the gates in
    :func:`parse_repair_response` still have to accept each value.
    """
    in_string = False
    escaped = False
    stack: List[str] = []
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or (char == "}" and stack[-1] != "{") or (char == "]" and stack[-1] != "["):
                return None
            stack.pop()
    if escaped or (not in_string and not stack):
        # A dangling escape would swallow the closing quote, and text with
        # nothing left open is not truncation - it is malformed.
        return None
    out = text + ('"' if in_string else "")
    out += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return out


def _extract_json_object(text: str) -> Tuple[Dict[str, Any], bool]:
    """First JSON object in the answer.

    Returns ``(object, complete)``. ``complete`` is False when the answer was
    cut short and the object had to be closed before it would parse, which is
    the only case where a field may be missing its ending.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        chunk = text[index:]
        try:
            data, _end = decoder.raw_decode(chunk)
        except ValueError:
            closed = _close_truncated_json(chunk)
            if closed is None:
                continue
            try:
                data = json.loads(closed)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data, False
        if isinstance(data, dict):
            return data, True
    return {}, True


def _repair_window(code: str, line: int) -> str:
    """A numbered slice of the file around ``line`` with original line numbers."""
    lines = code.splitlines()
    start = max(1, line - WINDOW_RADIUS)
    end = min(len(lines), line + WINDOW_RADIUS)
    return "\n".join(f"{n:>4} | {lines[n - 1]}" for n in range(start, end + 1))


def _apply_repair(code: str, line: int, replacement: str) -> Optional[str]:
    """Put the model's line back into the file, keeping everything else byte-for-byte.

    The replacement is re-indented to whatever the original line used, because a
    model that drops the indentation produces an ``IndentationError`` for the
    block it was repairing, and indentation is not the user's question.
    """
    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None
    text = lines[line - 1]
    ending = text[len(text.rstrip("\r\n")) :]
    indent = re.match(r"[ \t]*", text).group(0)
    lines[line - 1] = indent + replacement.strip() + ending
    return "".join(lines)


def parse_repair_response(
    text: str, code: str, analysis: Dict[str, Any]
) -> Tuple[Optional[LocalRepair], str]:
    """Turn one model answer into a repair, or explain why it cannot be one.

    Pure function: no weights, no I/O, so every admission rule here can be
    tested without loading the model. The answer must name the line it is
    replacing *and* quote it, which is what stops a model from quietly editing
    something it was never shown.
    """
    if not text or not text.strip():
        return None, "the model returned nothing"

    data, complete = _extract_json_object(text)
    if not data:
        return None, "the model did not return a JSON object"

    if data.get("no_fix"):
        why = str(data.get("reason") or "").strip() if complete else ""
        return None, f"the model declined to guess: {why}" if why else "the model declined to guess"

    try:
        line = int(data["line"])
    except (KeyError, TypeError, ValueError):
        return None, "the answer did not name a line"

    lines = code.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None, f"line {line} is not in the file"

    reported = analysis.get("line") if isinstance(analysis, dict) else None
    if isinstance(reported, int) and abs(line - reported) > UNTRUSTED_LINE_RADIUS:
        # The validator would reject the candidate for this reason later; saying
        # it here gives the panel the model's own wording instead of a gate's.
        return None, (
            f"it named line {line}, which is {abs(line - reported)} lines away from "
            f"the reported line {reported}"
        )

    original = data.get("original")
    replacement = data.get("replacement")
    if not isinstance(original, str) or not isinstance(replacement, str):
        return None, "the answer carried no original and replacement lines"

    actual = lines[line - 1].rstrip("\r\n")
    if original.rstrip() != actual.rstrip():
        return None, (
            f'it quoted line {line} as "{original.strip()}", which is not what is there'
        )

    candidate = replacement.strip()
    if not candidate:
        return None, "the replacement is empty"
    if candidate.startswith("#"):
        return None, "the replacement is a comment, which would delete what the line did"
    if candidate == "pass":
        return None, "the replacement is `pass`, which removes what the line did"
    if len(candidate) > MAX_REPAIR_LENGTH:
        return None, "the replacement is far longer than the line it replaces"
    if candidate == actual.strip():
        return None, "the replacement is what the line already says"

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "the answer carried no confidence score"
    if float(confidence) < MIN_CONFIDENCE:
        return None, (
            f"its confidence {float(confidence):.2f} is below {MIN_CONFIDENCE:.2f}"
        )

    applied = _apply_repair(code, line, candidate)
    if applied is None or applied == code:
        return None, "the replacement could not be placed in the file"

    # The time budget can cut the answer inside the last field. The repair
    # itself is intact, but a half-written reason is shown to the user, so it
    # is dropped rather than quoted.
    reason = str(data.get("reason") or "").strip() if complete else ""
    if len(reason) < 8:
        reason = ""

    return (
        LocalRepair(
            line=line,
            original=actual,
            replacement=candidate,
            reason=reason,
            confidence=float(confidence),
            code=applied,
        ),
        "",
    )


def model_status() -> Dict[str, Any]:
    return _ENGINE.status()


def warm_up() -> Dict[str, Any]:
    """Load the model outside the request path."""
    _ENGINE.load()
    return _ENGINE.status()


def explain_with_model(code: str, analysis: Dict[str, Any]) -> Optional[str]:
    """Extra wording from the model, or ``None`` when it is switched off."""
    if not EXPLAIN_ENABLED:
        return None
    return _ENGINE.explain(code, analysis)


def propose_local_repair(
    code: str, analysis: Dict[str, Any]
) -> Tuple[Optional[LocalRepair], str]:
    """One-line repair candidate plus the reason it is (or is not) usable."""
    return _ENGINE.propose_repair(code, analysis)


def propose_model_fix(code: str, analysis: Dict[str, Any]) -> Optional[str]:
    """Full-file candidate text, *unvalidated*. Validate before presenting it."""
    repair, _reason = _ENGINE.propose_repair(code, analysis)
    return repair.code if repair else None



def available_error_types() -> List[str]:
    """Documented finding types, for the API's self-description."""
    return [
        "SyntaxError",
        "TypeError",
        "ZeroDivisionError",
        "IndexError",
        "SingletonComparison",
        "undefined-variable",
        "unused-variable",
        "unused-import",
        "redefined-builtin",
        "not-callable",
        "no-member",
        "bare-except",
    ]
