"""Contract test: every field the React UI reads must exist in a real response.

App.js reads nested paths like `analysis.validation_details.reason` and
`analysis.diff[0].kind`. A missing key renders as `undefined` in the browser
rather than raising, so it fails silently. This asserts the keys are present.
"""

from __future__ import annotations

import os
import sys

import requests

from test_api import free_port, start_server, wait_for

REQUIRED_TOP_LEVEL = [
    "has_error",
    "line",
    "column",
    "end_line",
    "error_type",
    "severity",
    "category",
    "detected_by",
    "message",
    "suggestion",
    "explanation",
    "explanation_title",
    "corrected_code",
    "validated",
    "validation_details",
    "fix_summary",
    "diff",
    "ai",
    "all_issues",
    "analysis_ms",
]

REQUIRED_ISSUE = ["line", "column", "error_type", "severity", "category", "message", "detected_by"]
REQUIRED_VALIDATION = ["validated", "syntax_valid", "compiles", "changed", "issue_resolved", "reason", "warnings"]
REQUIRED_AI = ["model", "model_loaded", "model_state", "explanation_source", "fix_source", "note"]
REQUIRED_DIFF_ROW = ["kind", "line", "text"]

ERROR_CASES = [
    ('age = input("Enter your age: ")\n\nif age > 18:\n    print("Adult")', "fixable"),
    ("x = 10 / 0", "unfixable"),
    ("if x > 3\n    print('a')", "fixable"),
]
CLEAN_CASE = "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n"


def check(payload: dict, where: str, failures: list) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            failures.append(f"{where}: missing top-level key {key!r}")

    for key in REQUIRED_AI:
        if key not in (payload.get("ai") or {}):
            failures.append(f"{where}: missing ai.{key}")

    for key in REQUIRED_VALIDATION:
        if key not in (payload.get("validation_details") or {}):
            failures.append(f"{where}: missing validation_details.{key}")

    for index, issue in enumerate(payload.get("all_issues") or []):
        for key in REQUIRED_ISSUE:
            if key not in issue:
                failures.append(f"{where}: missing all_issues[{index}].{key}")

    for index, row in enumerate(payload.get("diff") or []):
        for key in REQUIRED_DIFF_ROW:
            if key not in row:
                failures.append(f"{where}: missing diff[{index}].{key}")
        if row.get("kind") not in ("added", "removed", "context"):
            failures.append(f"{where}: diff[{index}].kind is {row.get('kind')!r}, expected added/removed/context")


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server, log = start_server(port)
    failures: list = []

    try:
        wait_for(base)
        for code, label in ERROR_CASES:
            response = requests.post(f"{base}/analyze", json={"code": code}, timeout=300)
            if response.status_code != 200:
                failures.append(f"{label}: HTTP {response.status_code}")
                continue
            payload = response.json()
            check(payload, label, failures)

            if payload.get("validated"):
                if not payload.get("fix_summary"):
                    failures.append(f"{label}: validated but no fix_summary")
                if not payload.get("diff"):
                    failures.append(f"{label}: validated but no diff")
                if payload.get("ai", {}).get("fix_source") is None:
                    failures.append(f"{label}: validated but fix_source is null")
                print(f"ok    {label}: validated, fix_source={payload['ai']['fix_source']}, "
                      f"diff={len(payload['diff'])} rows")
            else:
                if payload.get("corrected_code") != code:
                    failures.append(f"{label}: unvalidated but corrected_code was changed")
                print(f"ok    {label}: reported without a fix (honest)")

        response = requests.post(f"{base}/analyze", json={"code": CLEAN_CASE}, timeout=300)
        payload = response.json()
        check(payload, "clean", failures)
        if payload.get("has_error"):
            failures.append("clean: has_error should be False")
        else:
            print("ok    clean: has_error=False")

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()
        log.close()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_test.log")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            for line in text.splitlines():
                if "ERROR" in line or "Traceback" in line or "Exception" in line:
                    failures.append(f"server log: {line.strip()}")

    print()
    if failures:
        print(f"CONTRACT FAILURES ({len(failures)}):")
        for item in failures:
            print("  -", item)
        return 1
    print("CONTRACT OK: all UI-consumed fields present")
    return 0


if __name__ == "__main__":
    sys.exit(main())