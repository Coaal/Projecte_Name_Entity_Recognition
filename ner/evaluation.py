from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import re

from ner.utils import load_jsonl

ENTITY_LABELS = (
    "PERSON_NAME",
    "ID_CARD_NUMBER",
    "ADDRESS",
    "REGISTRATION_NUMBER",
    "DATE",
    "AMOUNT",
    "ORGANIZATION",
)


def calculate_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Calculate precision, recall and F1 from TP, FP and FN counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _empty_counts() -> dict[str, int]:
    """Create an empty TP/FP/FN counter."""
    return {"tp": 0, "fp": 0, "fn": 0}


def _increment(
    counts_by_label: dict[str, dict[str, int]],
    label: str,
    key: str,
    amount: int = 1,
) -> None:
    """Increment one label-level counter, including unexpected labels."""
    if label not in counts_by_label:
        counts_by_label[label] = _empty_counts()

    counts_by_label[label][key] += amount


def _build_metric_block(
    counts_by_label: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build micro metrics and per-entity-type metrics from raw counts."""
    labels = list(ENTITY_LABELS)
    labels.extend(sorted(label for label in counts_by_label if label not in ENTITY_LABELS))

    per_entity_type = {
        label: calculate_metrics(**counts_by_label.get(label, _empty_counts()))
        for label in labels
    }

    micro = calculate_metrics(
        tp=sum(counts["tp"] for counts in counts_by_label.values()),
        fp=sum(counts["fp"] for counts in counts_by_label.values()),
        fn=sum(counts["fn"] for counts in counts_by_label.values()),
    )

    return {
        "micro": micro,
        "per_entity_type": per_entity_type,
    }


def _load_aligned_samples(
    gold_path: str | Path,
    predictions_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load predictions and the matching prefix of the gold dataset."""
    predicted_samples = load_jsonl(predictions_path)
    gold_samples = load_jsonl(gold_path, n=len(predicted_samples))

    if len(gold_samples) < len(predicted_samples):
        raise ValueError(
            "The gold dataset contains fewer documents than the predictions file."
        )

    for index, (gold, predicted) in enumerate(zip(gold_samples, predicted_samples)):
        gold_id = gold.get("id")
        predicted_id = predicted.get("id")

        if gold_id is not None and predicted_id is not None and gold_id != predicted_id:
            raise ValueError(
                f"Document id mismatch at position {index}: "
                f"gold={gold_id!r}, prediction={predicted_id!r}."
            )

    return gold_samples, predicted_samples


def _entity_text(entity: dict[str, Any], document_text: str) -> str:
    """Read entity text directly or recover it from character offsets."""
    if entity.get("text") is not None:
        return str(entity["text"])

    return document_text[int(entity["start"]):int(entity["end"])]


def _normalize_relaxed_text(text: str) -> str:
    """Collapse whitespace so PDF line wrapping does not change entity identity."""
    return re.sub(r"\s+", " ", text).strip()


def _print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    """Print micro and per-entity metrics with counts in a compact table."""
    micro = metrics["micro"]

    print(f"\n{title}")
    print("=" * 92)
    print(
        "Micro -> "
        f"P: {micro['precision']:.2f} | "
        f"R: {micro['recall']:.2f} | "
        f"F1: {micro['f1']:.2f} | "
        f"TP: {micro['true_positives']} | "
        f"FP: {micro['false_positives']} | "
        f"FN: {micro['false_negatives']}"
    )

    print()
    print(
        f"{'ENTITY TYPE':<25} {'P':>6} {'R':>6} {'F1':>6} "
        f"{'TP':>7} {'FP':>7} {'FN':>7}"
    )
    print("-" * 92)

    for label, values in metrics["per_entity_type"].items():
        print(
            f"{label:<25} "
            f"{values['precision']:>6.2f} "
            f"{values['recall']:>6.2f} "
            f"{values['f1']:>6.2f} "
            f"{values['true_positives']:>7} "
            f"{values['false_positives']:>7} "
            f"{values['false_negatives']:>7}"
        )


def evaluate_ner_strict(
    gold_path: str | Path,
    predictions_path: str | Path,
    print_results: bool = True,
) -> dict[str, Any]:
    """Evaluate exact matches of entity start, end and label. """
    gold_samples, predicted_samples = _load_aligned_samples(
        gold_path,
        predictions_path,
    )
    counts_by_label = {label: _empty_counts() for label in ENTITY_LABELS}

    for gold, predicted in zip(gold_samples, predicted_samples):
        gold_entities = {
            (int(entity["start"]), int(entity["end"]), str(entity["label"]))
            for entity in gold.get("entities", [])
        }
        predicted_entities = {
            (int(entity["start"]), int(entity["end"]), str(entity["label"]))
            for entity in predicted.get("entities", [])
        }

        for _, _, label in gold_entities & predicted_entities:
            _increment(counts_by_label, label, "tp")

        for _, _, label in predicted_entities - gold_entities:
            _increment(counts_by_label, label, "fp")

        for _, _, label in gold_entities - predicted_entities:
            _increment(counts_by_label, label, "fn")

    results = {
        "compared_samples": len(predicted_samples),
        **_build_metric_block(counts_by_label),
    }

    if print_results:
        _print_metric_block("STRICT EVALUATION", results)

    return results


def _evaluate_relaxed_mode(
    gold_samples: list[dict[str, Any]],
    predicted_samples: list[dict[str, Any]],
    unique: bool,
) -> dict[str, Any]:
    """Evaluate label and whitespace-normalized text while ignoring character spans."""
    counts_by_label = {label: _empty_counts() for label in ENTITY_LABELS}

    for gold, predicted in zip(gold_samples, predicted_samples):
        gold_pairs = [
            (str(entity["label"]), _normalize_relaxed_text(_entity_text(entity, str(gold.get("text", "")))))
            for entity in gold.get("entities", [])
        ]
        predicted_pairs = [
            (
                str(entity["label"]),
                _normalize_relaxed_text(_entity_text(entity, str(predicted.get("text", "")))),
            )
            for entity in predicted.get("entities", [])
        ]

        if unique:
            gold_counter = Counter({pair: 1 for pair in set(gold_pairs)})
            predicted_counter = Counter({pair: 1 for pair in set(predicted_pairs)})
        else:
            gold_counter = Counter(gold_pairs)
            predicted_counter = Counter(predicted_pairs)

        for label, text in set(gold_counter) | set(predicted_counter):
            gold_count = gold_counter[(label, text)]
            predicted_count = predicted_counter[(label, text)]

            _increment(counts_by_label, label, "tp", min(gold_count, predicted_count))
            _increment(
                counts_by_label,
                label,
                "fp",
                max(0, predicted_count - gold_count),
            )
            _increment(
                counts_by_label,
                label,
                "fn",
                max(0, gold_count - predicted_count),
            )

    return _build_metric_block(counts_by_label)


def evaluate_ner_relaxed(
    gold_path: str | Path,
    predictions_path: str | Path,
    print_results: bool = True,
) -> dict[str, Any]:
    """Evaluate label and whitespace-normalized text without spans in two views."""
    gold_samples, predicted_samples = _load_aligned_samples(
        gold_path,
        predictions_path,
    )

    occurrences = _evaluate_relaxed_mode(
        gold_samples,
        predicted_samples,
        unique=False,
    )
    unique = _evaluate_relaxed_mode(
        gold_samples,
        predicted_samples,
        unique=True,
    )

    results = {
        "compared_samples": len(predicted_samples),
        "occurrences": occurrences,
        "unique": unique,
    }

    if print_results:
        _print_metric_block("RELAXED EVALUATION - OCCURRENCES", occurrences)
        _print_metric_block("RELAXED EVALUATION - UNIQUE ENTITIES", unique)

    return results
