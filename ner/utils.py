from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path, n: int | None = None) -> list[dict[str, Any]]:
    """Load up to ``n`` non-empty JSONL records into memory."""
    if n is not None and n <= 0:
        return []

    records: list[dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(json.loads(line))

            if n is not None and len(records) >= n:
                break

    return records


def save_responses_to_jsonl(
    responses: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save response dictionaries as one JSON object per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for response in responses:
            file.write(json.dumps(response, ensure_ascii=False) + "\n")
