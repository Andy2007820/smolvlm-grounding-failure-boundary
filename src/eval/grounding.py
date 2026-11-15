"""Deterministic box parsing, IoU and aggregate grounding metrics.

Metric contract (frozen before B0):
- Coordinates are continuous ``[x1, y1, x2, y2]``; pixel-corner +1 is not used.
- A prediction is valid only when x2 > x1, y2 > y1 and all coordinates are in
  [0, 1]. Predictions are never clipped or reordered to make them valid.
- Integer coordinates in [0, 1000] are divided by 1000 when the values are not
  already a normalized box. Mixed/fractional quantized coordinates are invalid.
- Invalid model outputs contribute IoU=0 and remain in n for every aggregate.
- Invalid ground truth raises ValueError: it is a data error, not a model error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

ACC_THRESHOLD = 0.5
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9])"
)
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


@dataclass(frozen=True)
class Score:
    """One model output scored against one ground-truth box."""

    predicted_box: tuple[float, float, float, float] | None
    iou: float
    correct_at_0_5: bool
    parse_error: str | None


def validate_box(box: Any, *, normalized: bool = True) -> tuple[float, float, float, float]:
    """Return a finite ordered xyxy tuple or raise ValueError.

    ``normalized=False`` permits arbitrary finite pixel/world coordinates but
    still requires positive width and height.
    """

    if isinstance(box, (str, bytes)):
        raise ValueError("box must be a four-number sequence, not text")
    try:
        values = tuple(box)
    except TypeError as exc:
        raise ValueError("box must be a four-number sequence") from exc
    if len(values) != 4:
        raise ValueError(f"box must have four coordinates, got {len(values)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("box coordinates must be real numbers, not bool/non-numeric values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("box coordinates must be finite")
    if normalized and not all(0.0 <= v <= 1.0 for v in result):
        raise ValueError(f"normalized box is outside [0, 1]: {result}")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"box must have x2>x1 and y2>y1: {result}")
    return result  # type: ignore[return-value]


def _numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def parse_box_with_error(text: Any) -> tuple[list[float] | None, str | None]:
    """Parse one unambiguous normalized or 1000-bin box.

    Bracketed output takes precedence over surrounding prose. Exactly one
    bracketed candidate containing four numbers is accepted. Without such a
    candidate, bracket-free text must contain exactly four numbers in total.
    This prevents unrelated counts in prose from silently becoming coordinates.
    """

    if not isinstance(text, str):
        return None, "output_not_string"
    bracket_groups = BRACKET_RE.findall(text)
    candidates = [values for group in bracket_groups if len(values := _numbers(group)) == 4]
    if len(candidates) > 1:
        return None, "ambiguous_multiple_boxes"
    if candidates:
        tokens = candidates[0]
    elif bracket_groups:
        return None, "no_bracket_with_exactly_four_coordinates"
    else:
        tokens = _numbers(text)
        if len(tokens) != 4:
            return None, "expected_exactly_four_coordinates"
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None, "coordinate_conversion_failed"
    if not all(math.isfinite(v) for v in values):
        return None, "non_finite_coordinate"
    if any(v < 0.0 or v > 1000.0 for v in values):
        return None, "coordinate_outside_0_1_or_0_1000"
    if all(v <= 1.0 for v in values):
        normalized = values
    else:
        if any(not v.is_integer() for v in values):
            return None, "quantized_coordinates_must_all_be_integers"
        normalized = [v / 1000.0 for v in values]
    try:
        box = validate_box(normalized, normalized=True)
    except ValueError:
        return None, "degenerate_or_reversed_box"
    return list(box), None


def parse_box(text: Any) -> list[float] | None:
    """Return a valid normalized xyxy box, otherwise None."""

    box, _ = parse_box_with_error(text)
    return box


def iou(a: Any, b: Any) -> float:
    """Compute continuous xyxy intersection-over-union.

    Both inputs are validated. The function also accepts pixel boxes; the
    caller is responsible for ensuring both use the same coordinate space.
    """

    box_a = validate_box(a, normalized=False)
    box_b = validate_box(b, normalized=False)
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:  # Defensive: ordered finite boxes should never reach this.
        raise ValueError("IoU union must be positive")
    return intersection / union


def score_prediction(output_text: Any, ground_truth: Any) -> Score:
    """Score one output; an invalid prediction has IoU 0 and is not dropped."""

    gt = validate_box(ground_truth, normalized=True)
    prediction, error = parse_box_with_error(output_text)
    if prediction is None:
        return Score(None, 0.0, False, error or "invalid_prediction")
    value = iou(prediction, gt)
    return Score(tuple(prediction), value, value >= ACC_THRESHOLD, None)


def metrics_from_scores(scores: Iterable[Score]) -> dict[str, float | int]:
    """Aggregate over all samples, explicitly including invalid predictions."""

    materialized = list(scores)
    n = len(materialized)
    if n == 0:
        raise ValueError("metrics are undefined for an empty evaluation")
    invalid = sum(score.predicted_box is None for score in materialized)
    correct = sum(score.correct_at_0_5 for score in materialized)
    return {
        "acc@0.5": correct / n,
        "mIoU": math.fsum(score.iou for score in materialized) / n,
        "parse_rate": (n - invalid) / n,
        "invalid": invalid,
    }


def evaluate_outputs(
    records: Sequence[Mapping[str, Any]], output_texts: Sequence[Any]
) -> tuple[dict[str, float | int], list[Score]]:
    """Score aligned dataset records and decoded model outputs."""

    if len(records) != len(output_texts):
        raise ValueError(f"record/output length mismatch: {len(records)} != {len(output_texts)}")
    scores = [score_prediction(text, record["answer"]) for record, text in zip(records, output_texts)]
    return metrics_from_scores(scores), scores
