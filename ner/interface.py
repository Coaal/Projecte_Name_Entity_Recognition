"""GLiNER model loading and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gliner import GLiNER

DEFAULT_MODEL = "gliner-community/gliner_small-v2.5"


def load_gliner_model(model_name: str = DEFAULT_MODEL, device: str | None = None) -> GLiNER:
    """Load a GLiNER model from Hugging Face (downloaded on first use).

    If `device` is None, uses CUDA when available and CPU otherwise.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return GLiNER.from_pretrained(model_name, map_location=device)


def _chunk_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into (offset, chunk) pieces of at most `max_chars`,
    cutting at whitespace when possible."""
    chunks: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append((start, text[start:end]))
        start = end
    return chunks


def predict_entities(
    model: GLiNER,
    text: str,
    labels: list[str],
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
    max_chars: int = 1000,
) -> list[dict[str, Any]]:
    """Predict entities in `text`.

    Returns a list of dicts with keys: start, end, text, label, score.
    `start`/`end` are character offsets in the ORIGINAL text, even if the
    text had to be split into chunks.
    """
    entities: list[dict[str, Any]] = []
    for offset, chunk in _chunk_text(text, max_chars):
        if not chunk.strip():
            continue
        preds = model.predict_entities(
            chunk,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        for p in preds:
            p = dict(p)
            p["start"] += offset
            p["end"] += offset
            entities.append(p)
    return entities


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file (creates parent folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def predict_dataset(
    model: GLiNER,
    records: list[dict[str, Any]] | str | Path,
    labels: list[str],
    text_key: str = "text",
    threshold: float = 0.5,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run predict_entities over a dataset.

    `records` can be a list of dicts or a path to a JSONL file. Each record
    is returned with an added "predictions" key. If `output_path` is given,
    the result is also saved as JSONL.
    """
    if isinstance(records, (str, Path)):
        records = load_jsonl(records)

    results = []
    for record in records:
        preds = predict_entities(
            model, record[text_key], labels, threshold=threshold, **kwargs
        )
        results.append({**record, "predictions": preds})

    if output_path is not None:
        save_jsonl(results, output_path)
    return results