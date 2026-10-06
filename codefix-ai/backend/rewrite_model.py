"""Local instruction-tuned rewrite model.

This module is deliberately *advisory*. It proposes; :mod:`validator` decides.
The previous checkpoint here (``Salesforce/codet5-base``) is a pretrained
code-completion model, not an instruction model: asked to repair a file it
emitted unrelated code, and ``print(age)`` came back as ``defprint(name)`` - text
that parses, compiles and silences the reported message, so every mechanical gate
passed. An instruct-tuned checkpoint fixes most of that, but not all of it, so two
rules keep it in its lane:

* it only sees findings whose correction is mechanical (:data:`ELIGIBLE_TYPES`),
  because a missing name, an out-of-range index or a zero divisor is a decision
  the programmer has to make, and the model answers those by inventing something;
* every candidate it produces must clear the untrusted gates in
  :mod:`validator` - including the locality gate, which rejects a rewrite that
  touches lines far from the reported one.

``CODEFIX_MODEL`` points at whatever checkpoint suits the machine. On CPU,
``Qwen/Qwen2.5-Coder-0.5B-Instruct`` is roughly 2 GB and answers in 6-15 s;
1.5B or 7B are better if there is RAM or a GPU.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

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

_SYSTEM_REPAIR = (
    "You repair one Python file. Reply with the complete corrected file inside a single "
    "```python code block. Make the smallest change that fixes the reported error and keep "
    "every other line exactly as it is. Do not add features. No prose."
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

    def propose_fix(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Model rewrite of the whole file, returned *unvalidated*.

        The caller must run :func:`validator.validate_correction` before showing
        it. ``None`` means no candidate was produced, or the finding is one the
        model is not allowed to touch.
        """
        if not ENABLED:
            return None
        error_type = analysis.get("error_type") or ""
        if error_type not in ELIGIBLE_TYPES:
            _LOG.info("model not asked about %s: it needs a decision, not a rewrite", error_type)
            return None
        if not self._within_budget(code):
            return None

        user = (
            f"Fix this error.\n\nError on line {analysis.get('line')}: {error_type}\n"
            f"{analysis.get('message') or ''}\n\n"
            f"File:\n```python\n{code}\n```\n\n"
            "Return the whole file with only that error fixed."
        )
        candidate = _extract_code(
            self._chat(_SYSTEM_REPAIR, user, MAX_NEW_TOKENS_FIX, FIX_TIME_BUDGET_S)
        )
        if not candidate.strip() or candidate.strip() == code.strip():
            return None
        return candidate

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


def propose_model_fix(code: str, analysis: Dict[str, Any]) -> Optional[str]:
    """Unvalidated candidate. The caller must validate before presenting it."""
    return _ENGINE.propose_fix(code, analysis)


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
