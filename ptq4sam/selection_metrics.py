from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _to_float_list(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("expected at least one candidate value")
    return result


def _argmax_first(values: Sequence[float]) -> int:
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        current = float(value)
        if current > best_value:
            best_index = index
            best_value = current
    return best_index


def summarize_selection_sample(
    candidate_ious: Sequence[float],
    candidate_scores: Sequence[float],
) -> dict[str, float | int]:
    """Summarize model-selected vs oracle-selected mask quality for one sample."""

    ious = _to_float_list(candidate_ious)
    scores = _to_float_list(candidate_scores)
    if len(ious) != len(scores):
        raise ValueError(
            "candidate_ious and candidate_scores must have the same number of elements"
        )

    selected_index = _argmax_first(scores)
    oracle_index = _argmax_first(ious)
    selected_miou = float(ious[selected_index])
    oracle_miou = float(ious[oracle_index])
    oracle_gap = oracle_miou - selected_miou

    return {
        "num_candidates": len(ious),
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "selected_miou": selected_miou,
        "oracle_miou": oracle_miou,
        "oracle_gap": oracle_gap,
        "rank_hit_at_1": int(selected_index == oracle_index),
    }


def aggregate_selection_metrics(
    samples: Iterable[Mapping[str, float | int]],
) -> dict[str, float | int]:
    total = 0
    selected_sum = 0.0
    oracle_sum = 0.0
    gap_sum = 0.0
    hit_sum = 0

    for sample in samples:
        total += 1
        selected_sum += float(sample["selected_miou"])
        oracle_sum += float(sample["oracle_miou"])
        gap_sum += float(sample["oracle_gap"])
        hit_sum += int(sample["rank_hit_at_1"])

    if total == 0:
        raise ValueError("expected at least one sample summary to aggregate")

    return {
        "samples_evaluated": total,
        "selected_miou": selected_sum / total,
        "oracle_miou": oracle_sum / total,
        "oracle_gap": gap_sum / total,
        "rank_hit_at_1": hit_sum / total,
    }


def summarize_selection_dataset(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    per_sample = []
    for record in records:
        sample_summary = summarize_selection_sample(
            candidate_ious=record["candidate_ious"],
            candidate_scores=record["candidate_scores"],
        )
        merged = dict(record)
        merged.update(sample_summary)
        per_sample.append(merged)
    return {
        "summary": aggregate_selection_metrics(per_sample),
        "per_sample": per_sample,
    }


def summarize_selection_json(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, object]:
    records = json.loads(Path(input_path).read_text())
    if not isinstance(records, list):
        raise ValueError("expected a JSON list of per-sample records")
    result = summarize_selection_dataset(records)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
