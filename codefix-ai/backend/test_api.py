"""End-to-end API tests against a real uvicorn server.

Run with: python test_api.py
Starts the server on a free port, exercises POST /analyze over HTTP, then
shuts it down. Self-contained so it needs no separately running service.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(HERE, "venv", "Scripts", "python.exe")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int) -> tuple[subprocess.Popen, object]:
    """Start uvicorn with its output going to a file.

    A PIPE is not usable here: nobody drains it, so once the OS buffer fills,
    the server blocks on write and requests hang.
    """
    env = dict(os.environ)
    env["CODEFIX_ENABLE_MODEL"] = "1"
    log = open(os.path.join(HERE, "server_test.log"), "w", encoding="utf-8")
    process = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=HERE,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def wait_for(base: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as exc:
            last = exc
        time.sleep(1.0)
    raise RuntimeError(f"server did not become ready: {last}")


FLAGSHIP = '''age = input("Enter your age: ")

if age > 18:
    print("Adult")
else:
    print("Minor")'''

EXPECTATIONS = [
    (
        "flagship input/int comparison",
        FLAGSHIP,
        {
            "has_error": True,
            "line": 3,
            "error_type": "TypeError",
            "severity": "Medium",
            "category": "Type Error",
            "validated": True,
        },
    ),
    (
        "missing colon",
        "if x > 3\n    print('a')",
        {"has_error": True, "error_type": "SyntaxError", "severity": "High", "validated": True},
    ),
    (
        # No safe automatic fix exists: the correct value of `tax` is unknown,
        # and inventing one would be a guess. This asserts the honest path -
        # the error is reported and explained, but nothing is presented as a fix.
        "undefined variable is reported, not guessed",
        "def total(prices):\n    return sum(prices) + tax\n",
        {"has_error": True, "line": 2, "error_type": "undefined-variable", "validated": False},
    ),
    (
        # Likewise `10 / 0`: the intended divisor is unknowable.
        "division by zero is reported, not guessed",
        "x = 10 / 0",
        {"has_error": True, "error_type": "ZeroDivisionError", "severity": "High", "validated": False},
    ),
    (
        "clean code",
        "def greet(name):\n    return f\"Hello, {name}!\"\n\n\nprint(greet('World'))",
        {"has_error": False, "validated": True},
    ),
]


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server, server_log = start_server(port)

    failures = 0
    try:
        health = wait_for(base)
        print(f"health: status={health['status']} model={health['model']['state']}")
        print(f"model detail: {health['model']['detail'][:110]}")
        print()

        for label, code, expected in EXPECTATIONS:
            started = time.perf_counter()
            response = requests.post(f"{base}/analyze", json={"code": code}, timeout=300)
            elapsed = int((time.perf_counter() - started) * 1000)

            if response.status_code != 200:
                failures += 1
                print(f"FAIL  {label}: HTTP {response.status_code} {response.text[:200]}")
                continue

            body = response.json()
            problems = [
                f"{key}={body.get(key)!r} want {want!r}"
                for key, want in expected.items()
                if body.get(key) != want
            ]

            if problems:
                failures += 1
                print(f"FAIL  {label}: {'; '.join(problems)}")
                print(f"      message: {body.get('message')}")
            else:
                print(f"ok    {label}  ({elapsed}ms)")
                print(f"      line={body.get('line')} type={body.get('error_type')} "
                      f"sev={body.get('severity')} cat={body.get('category')} "
                      f"validated={body.get('validated')}")
                print(f"      fix_source={body['ai']['fix_source']} "
                      f"explanation_source={body['ai']['explanation_source']}")
                if body.get("fix_summary"):
                    print(f"      fix: {body['fix_summary']}")

            for required in ("message", "explanation", "corrected_code", "validation_details", "ai"):
                if required not in body:
                    failures += 1
                    print(f"FAIL  {label}: missing field {required}")

        # Empty input must be rejected, not crash.
        response = requests.post(f"{base}/analyze", json={"code": "   "}, timeout=30)
        if response.status_code != 400:
            failures += 1
            print(f"FAIL  empty code should give 400, got {response.status_code}")
        else:
            print("ok    empty code rejected with 400")

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()

        log_path = os.path.join(HERE, "server_test.log")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                noise = handle.read().strip()
            if noise:
                print("\n--- server log ---")
                print(noise[-3000:])

    print()
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())