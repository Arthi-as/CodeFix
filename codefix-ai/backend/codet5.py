"""CodeT5 integration for CodeFix AI.

What this module actually does
------------------------------
``Salesforce/codet5-base`` is a *pretrained* code encoder-decoder. It was
trained with identifier-aware and bimodal dual denoising objectives, not
instruction tuning, so it has no notion of "explain this error" or "fix this
code". Measured on this checkpoint, instruction-style prompts return fragments
such as ``'age ) ) ;'`` or ``'..'`` rather than usable output.

Rather than pretend otherwise, this module treats the model as an *advisory*
source with hard quality gates:

* :func:`explain` asks the model to summarise the offending code. The result is
  accepted only if it passes :func:`_looks_like_prose`; otherwise the caller
  falls back to the deterministic explanation built from the static finding.
* :func:`propose_correction` lets the model attempt a rewrite. The candidate is
  returned unvalidated and the pipeline only prefers it if
  :mod:`validator` confirms it parses, compiles and resolves the finding.

Static analysis therefore decides *what* is wrong and *where*; CodeT5 may
contribute wording or an alternative patch; the validator decides whether any
of it may be presented as a fix. Every response reports which source produced
the wording and the correction, so the UI never implies the model was
authoritative.

Operational notes
-----------------
* The tokenizer for this checkpoint is ``RobertaTokenizer`` (``AutoTokenizer``),
  **not** ``T5Tokenizer`` - there is no ``spiece.model`` in the repository, and
  ``T5Tokenizer.from_pretrained`` fails with ``TypeError: not a string``.
* Loading happens lazily behind a lock, on a worker thread, so neither the event
  loop nor concurrent requests block.
* If the weights are missing (offline, no cache) every function degrades to the
  static-analysis path and says so.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("CODEFIX_MODEL", "Salesforce/codet5-base")
ENABLED = os.environ.get("CODEFIX_ENABLE_CODET5", "1") not in ("0", "false", "False")

# Generation is deliberately cheap: this runs on CPU inside a request.
MAX_NEW_TOKENS_EXPLAIN = 48
MAX_NEW_TOKENS_FIX = 96
NUM_BEAMS = 1

MIN_EXPLANATION_CHARS = 40
MAX_EXPLANATION_CHARS = 400


class CodeT5Engine:
    """Lazily loaded CodeT5 wrapper, safe to share across requests."""

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
            self._detail = "CODEFIX_ENABLE_CODET5 is set to 0."
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
                from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: PLC0415

                self.device = "cuda" if torch.cuda.is_available() else "cpu"

                # AutoTokenizer, not T5Tokenizer: this checkpoint is RoBERTa-tokenised.
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
                model.to(self.device)
                model.eval()
                self._model = model
                self._state = "ready"
                self._detail = f"{MODEL_NAME} loaded on {self.device}"
                _LOG.info("CodeT5 %s", self._detail)
                return True
            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._state = "failed"
                self._detail = f"{type(exc).__name__}: {exc}"
                _LOG.warning("CodeT5 unavailable, using static analysis only: %s", self._detail)
                return False

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

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        if not self.is_available() and not self.load():
            return ""
        try:
            import torch  # noqa: PLC0415

            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                )
            return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception as exc:
            _LOG.warning("CodeT5 generation failed: %s", exc)
            return ""

    # -- explanation -------------------------------------------------------

    def explain(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Model-written description of the failing snippet, if it is usable.

        Returns ``None`` when the model is unavailable or its output fails the
        prose checks, so the caller can fall back to the static explanation.
        """
        if not ENABLED:
            return None

        snippet = _snippet_around(code, analysis.get("line"))
        prompt = f"summarize: {snippet}"
        text = self._generate(prompt, MAX_NEW_TOKENS_EXPLAIN)
        if not _looks_like_prose(text):
            return None
        return text

    # -- correction --------------------------------------------------------

    def propose_correction(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Model-proposed rewrite, returned unvalidated.

        The caller must run :func:`validator.validate_correction` before
        presenting it. ``None`` means no usable candidate was produced.
        """
        if not ENABLED:
            return None

        error_type = analysis.get("error_type", "error")
        suggestion = analysis.get("suggestion") or ""
        prompt = (
            f"input: fix the {error_type}. {suggestion}\n"
            f"output: {code}"
        )
        text = _strip_code_fence(self._generate(prompt, MAX_NEW_TOKENS_FIX))
        if not text.strip():
            return None
        return text


_ENGINE = CodeT5Engine()


def get_engine() -> CodeT5Engine:
    return _ENGINE


# --------------------------------------------------------------------------
# output quality gates
# --------------------------------------------------------------------------

_CODE_CHARS = re.compile(r"[{}()\[\];]|=>|::")
_REPEATED_CHAR = re.compile(r"(.)\1{6,}")


def _looks_like_prose(text: str) -> bool:
    """Reject the degenerate output this checkpoint usually produces.

    The pretrained model tends to emit code fragments, single punctuation
    marks, or long runs of one character. Showing any of those as an
    explanation would be worse than showing nothing.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_EXPLANATION_CHARS or len(stripped) > MAX_EXPLANATION_CHARS:
        return False
    # Must contain real words and spaces, not just punctuation or code.
    if " " not in stripped:
        return False
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    if len(words) < 5:
        return False
    if _REPEATED_CHAR.search(stripped):
        return False
    if _CODE_CHARS.search(stripped):
        return False
    return True


_FENCE = re.compile(r"^```(?:python|py)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def _snippet_around(code: str, line: Optional[int], radius: int = 3) -> str:
    """The reported line plus a little context, for a summarisation prompt."""
    lines = code.splitlines()
    if not lines:
        return ""
    if not line or line < 1:
        return "\n".join(lines[: radius * 2 + 1])
    start = max(0, line - 1 - radius)
    end = min(len(lines), line + radius)
    return "\n".join(lines[start:end])


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def model_status() -> Dict[str, Any]:
    return _ENGINE.status()


def warm_up() -> Dict[str, Any]:
    """Load the model outside the request path."""
    _ENGINE.load()
    return _ENGINE.status()


def explain_with_codet5(code: str, analysis: Dict[str, Any]) -> Optional[str]:
    """Best-effort model explanation, or ``None`` to use the static one."""
    return _ENGINE.explain(code, analysis)


def propose_codet5_correction(code: str, analysis: Dict[str, Any]) -> Optional[str]:
    """Best-effort model rewrite. Unvalidated - the caller must check it."""
    return _ENGINE.propose_correction(code, analysis)


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