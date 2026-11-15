"""Independent 0-100 integer grid conversion layer for the T7 ``p1_grid100`` variant.

Training writes grid answers as ``[gx1, gy1, gx2, gy2]`` with each integer in
0-100 (``round(normalized * 100)``).  The frozen evaluator parser must not score
these directly: values > 1 are treated as 1000-bin and divided by 1000, which
would shrink the box by 10x.  This module therefore parses the grid box itself,
validates range/ordering, divides by 100, and only then hands the normalized box
to the frozen ``iou`` / ``Score`` / ``ACC_THRESHOLD`` code.  The frozen metrics
module is never modified.

Grammar mirrors ``src/eval/pixel_convert.py`` and
``src/train/coordinate_variants.py`` so training-time and evaluation-time
interpretation are identical:

1. exactly one bracketed candidate with exactly four numbers wins over prose;
2. multiple four-number bracket candidates are ambiguous and invalid;
3. with brackets present but no four-number candidate the output is invalid;
4. without brackets the whole text must contain exactly four numbers.

Grid values must be finite and integer-valued (``float`` values equal to their
integer are accepted), satisfy ``0 <= gx1 < gx2 <= 100`` and
``0 <= gy1 < gy2 <= 100``.  Invalid outputs become ``Score(None, 0.0, False,
error)`` and therefore remain in ``n`` for parse_rate/Acc@0.5/mIoU exactly like
other invalid predictions.  Degenerate/reversed boxes are never repaired.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .grounding import ACC_THRESHOLD, Score, iou, validate_box
from .pixel_convert import bracketed_numbers


def parse_grid100(output_text: Any) -> tuple[list[int] | None, str | None, dict[str, Any]]:
    """Parse one 0-100 integer grid box; returns ``(grid, error, audit)``."""

    audit: dict[str, Any] = {"conversion_applied": True, "representation": "grid100_integer"}
    values, error = bracketed_numbers(output_text)
    if values is None:
        return None, error or "grid_parse_failed", audit
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_coordinate", audit
    if any(value != int(value) for value in values):
        return None, "grid_coordinates_must_be_integers", audit
    grid = [int(value) for value in values]
    audit["grid_box"] = grid
    if any(value < 0 or value > 100 for value in grid):
        return None, "grid_box_outside_0_100", audit
    if not (grid[2] > grid[0] and grid[3] > grid[1]):
        return None, "grid_box_reversed", audit
    return grid, None, audit


def grid_to_normalized(grid: list[int]) -> tuple[float, float, float, float]:
    """Validate and normalize an integer grid box; raises ValueError on invalid input."""

    if len(grid) != 4:
        raise ValueError(f"grid box must have four coordinates, got {len(grid)}")
    if any(value < 0 or value > 100 for value in grid):
        raise ValueError(f"grid box outside 0-100: {grid}")
    normalized = [grid[0] / 100.0, grid[1] / 100.0, grid[2] / 100.0, grid[3] / 100.0]
    return validate_box(normalized, normalized=True)


def score_prediction_grid(output_text: Any, record: Mapping[str, Any]) -> tuple[Score, dict[str, Any]]:
    """Score one ``p1_grid100`` output with frozen IoU/Acc rules and no repair."""

    grid, error, audit = parse_grid100(output_text)
    ground_truth = validate_box(record["answer"], normalized=True)
    if grid is None:
        return Score(None, 0.0, False, error or "invalid_prediction"), audit
    normalized = grid_to_normalized(grid)
    value = iou(normalized, ground_truth)
    audit["normalized_box"] = list(normalized)
    return Score(tuple(normalized), value, value >= ACC_THRESHOLD, None), audit
