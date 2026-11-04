"""Fail if pyright reports more errors than the count in .pyright-baseline.

Usage: uv run pyright --outputjson | python scripts/check_pyright_count.py
"""

import json
import sys
from pathlib import Path

BASELINE_FILE = Path(__file__).resolve().parent.parent / ".pyright-baseline"


def main() -> int:
    baseline = int(BASELINE_FILE.read_text().strip())
    report = json.load(sys.stdin)
    errors = report["summary"]["errorCount"]
    print(f"pyright errors: {errors} (baseline {baseline})")
    if errors > baseline:
        print(
            f"FAIL: {errors - baseline} new pyright error(s). Fix them; "
            "do not raise the baseline."
        )
        return 1
    if errors < baseline:
        print(f"Count dropped: lower {BASELINE_FILE.name} to {errors}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
