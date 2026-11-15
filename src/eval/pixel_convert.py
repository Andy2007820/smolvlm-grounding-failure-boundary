"""Independent pixel-xyxy conversion layer for the T7 ``p1_pixel`` variant.

The frozen evaluator parser (``src.eval.grounding.parse_box``) must not consume
pixel coordinates: it would treat values > 1 as 1000-bin and divide by 1000.
This module therefore parses a pixel box itself, validates it against the real
image width/height, normalizes it, and only then hands the normalized box to the
frozen ``iou`` / ``Score`` / ``ACC_THRESHOLD`` code. The frozen metrics module is
never modified.

Grammar mirrors ``src/train/coordinate_variants.py`` so training-time and
evaluation-time interpretation are identical:

1. exactly one bracketed candidate with exactly four numbers wins over prose;
2. multiple four-number bracket candidates are ambiguous and invalid;
3. with brackets present but no four-number candidate the output is invalid;
4. without brackets the whole text must contain exactly four numbers.

Pixel values must be finite integers (``float`` values equal to their integer
are accepted), satisfy ``0 <= x1 < x2 <= width`` and ``0 <= y1 < y2 <= height``.
Invalid outputs become ``Score(None, 0.0, False, error)`` and therefore remain
in ``n`` for parse_rate/Acc@0.5/mIoU exactly like other invalid predictions.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .grounding import ACC_THRESHOLD, BRACKET_RE, NUMBER_RE, Score, iou, validate_box


def bracketed_numbers(text: Any) -> tuple[list[float] | None, str | None]:
    """Mirror the training-side bracket grammar without using ``parse_box``."""

    if not isinstance(text, str):
        return None, "output_not_string"
    groups = BRACKET_RE.findall(text)
    candidates = [values for group in groups if len(values := NUMBER_RE.findall(group)) == 4]
    if len(candidates) > 1:
        return None, "ambiguous_multiple_boxes"
    if candidates:
        tokens = candidates[0]
    elif groups:
        return None, "no_bracket_with_exactly_four_coordinates"
    else:
        tokens = NUMBER_RE.findall(text)
        if len(tokens) != 4:
            return None, "expected_exactly_four_coordinates"
    try:
        return [float(token) for token in tokens], None
    except ValueError:
        return None, "coordinate_conversion_failed"


def parse_pixel_box(
    output_text: Any, *, width: int, height: int
) -> tuple[list[int] | None, str | None, dict[str, Any]]:
    """Parse one pixel xyxy box and validate it against the image dimensions."""

    audit: dict[str, Any] = {
        "conversion_applied": True,
        "image_wh": [int(width), int(height)],
    }
    values, error = bracketed_numbers(output_text)
    if values is None:
        return None, error or "pixel_parse_failed", audit
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_coordinate", audit
    if any(value != int(value) for value in values):
        return None, "pixel_coordinates_must_be_integers", audit
    pixels = [int(value) for value in values]
    audit["pixel_box"] = pixels
    if not (
        0 <= pixels[0] < pixels[2] <= int(width)
        and 0 <= pixels[1] < pixels[3] <= int(height)
    ):
        return None, "pixel_box_outside_or_reversed", audit
    return pixels, None, audit


def pixel_box_to_normalized(pixels: list[int], *, width: int, height: int) -> tuple[float, float, float, float]:
    """Validate and normalize a pixel box; raises ValueError on invalid input."""

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    normalized = [
        pixels[0] / width,
        pixels[1] / height,
        pixels[2] / width,
        pixels[3] / height,
    ]
    return validate_box(normalized, normalized=True)


def score_prediction_pixel(output_text: Any, record: Mapping[str, Any]) -> tuple[Score, dict[str, Any]]:
    """Score one ``p1_pixel`` output with frozen IoU/Acc rules and no repair."""

    width = int(record["width"])
    height = int(record["height"])
    pixels, error, audit = parse_pixel_box(output_text, width=width, height=height)
    ground_truth = validate_box(record["answer"], normalized=True)
    if pixels is None:
        return Score(None, 0.0, False, error or "invalid_prediction"), audit
    normalized = pixel_box_to_normalized(pixels, width=width, height=height)
    value = iou(normalized, ground_truth)
    audit["normalized_box"] = list(normalized)
    return Score(tuple(normalized), value, value >= ACC_THRESHOLD, None), audit
