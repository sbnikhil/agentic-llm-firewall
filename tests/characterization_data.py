"""Read-only loaders for the test data files (golden and characterization)."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data(relative: str) -> Any:
    with open(DATA_DIR / relative, encoding="utf-8") as f:
        return json.load(f)


def ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["id"] for item in items]
