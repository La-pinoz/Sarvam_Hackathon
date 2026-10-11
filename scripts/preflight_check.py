"""
Secret preflight check for FasalVachak.

Run:
    python scripts/preflight_check.py

Exit 0 = OK to start server.py. Exit 1 = fix something first.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Mirrors the exact placeholder string the Sarvam clients check for.
SARVAM_PLACEHOLDER = "your_sarvam_api_key_here"

REQUIRED_VARS = {
    "GROQ_API_KEY": None,
    "SARVAM_API_KEY": lambda v: v != SARVAM_PLACEHOLDER,
    "LIVEKIT_URL": lambda v: v.startswith(("wss://", "ws://")),
    "LIVEKIT_API_KEY": None,
    "LIVEKIT_API_SECRET": None,
}

GENERIC_PLACEHOLDER_MARKERS = ("your_", "_here")


def mask(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def check_var(name: str, validator) -> tuple[bool, str]:
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return False, f"{name}: MISSING"

    if any(marker in value for marker in GENERIC_PLACEHOLDER_MARKERS):
        return False, f"{name}: unfilled placeholder ({mask(value)})"

    if validator is not None and not validator(value):
        return False, f"{name}: set but fails format check ({mask(value)})"

    return True, f"{name}: OK ({mask(value)})"


def main() -> int:
    results = [check_var(name, validator) for name, validator in REQUIRED_VARS.items()]

    ok_count = sum(1 for ok, _ in results if ok)
    print(f"Preflight check: {ok_count}/{len(results)} variables OK\n")

    failed = False
    for ok, message in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {message}")
        failed = failed or not ok

    print()
    print("Preflight FAILED — fix above before running server.py" if failed
          else "Preflight PASSED — safe to run server.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())