"""Coordinate-representation variants for the T7 ablation (CPU-side spec + helpers).

Three representations share the same data, optimizer budget, LoRA settings and
seed; only the answer encoding and the matching prompt instruction change:

A. ``normalized_decimal`` — frozen p1 prompt + ``[x1, y1, x2, y2]`` normalized
   decimals in [0, 1].  The frozen evaluator parser consumes this directly.
B. ``bin1000_integer`` — prompt asks for integers in the 0–1000 range; answers
   are ``round(normalized * 1000)``.  The frozen evaluator already divides
   integers in [0, 1000] by 1000 when they are not all <= 1, so B needs no
   evaluator code change (only the prompt variant).
C. ``pixel_xyxy`` — prompt asks for pixel coordinates using the image width and
   height; answers are the dataset's ``raw_bbox`` pixel integers.  The frozen
   parser must NOT be used directly for C: it would misinterpret pixel values
   (e.g. divide by 1000).  C therefore needs an independent conversion layer
   that parses the pixel box, validates it against the actual image dimensions,
   normalizes it, and then feeds the frozen metric functions.  This module
   implements that conversion; the official evaluator remains owned by
   eval-keeper.
D. ``grid100_integer`` — prompt asks for integer grid coordinates in the 0–100
   range; answers are ``round(normalized * 100)``.  Like B the frozen parser can
   read the numbers, but it would divide them by 1000, so D also needs an
   independent conversion layer (``src/eval/grid_convert.py``) that validates
   the integer grid box, divides by 100, and then reuses the frozen metrics.
   Degenerate (collapsed) grid boxes are an error, never repaired.

This module is read-only with respect to ``src/eval`` (it imports the frozen
parser/score helpers but never modifies them).  The training pipeline uses
``encode_answer`` and ``build_messages``; the diagnostic probe uses
``score_prediction_variant``.  The official C evaluation must be run by the
independent evaluator using the same conversion contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from src.eval.grounding import (
    ACC_THRESHOLD,
    BRACKET_RE,
    NUMBER_RE,
    Score,
    iou,
    parse_box,
    validate_box,
)
from src.eval.prompts import PROMPT_ID as P1_PROMPT_ID
from src.eval.prompts import PROMPT_TEMPLATE as P1_PROMPT_TEMPLATE

REPRESENTATIONS = ("normalized_decimal", "bin1000_integer", "pixel_xyxy", "grid100_integer")

PROMPT_IDS = {
    "normalized_decimal": P1_PROMPT_ID,
    "bin1000_integer": "p1_bin1000",
    "pixel_xyxy": "p1_pixel",
    "grid100_integer": "p1_grid100",
}

PROMPT_TEMPLATES = {
    "normalized_decimal": P1_PROMPT_TEMPLATE,
    "bin1000_integer": (
        "Locate the object described below in the image.\n"
        "Description: {question}\n"
        "Answer with the bounding box as integer coordinates "
        "[x1, y1, x2, y2] in the 0-1000 range, where 1000 corresponds to "
        "the image width or height."
    ),
    "pixel_xyxy": (
        "Locate the object described below in the image.\n"
        "Description: {question}\n"
        "Answer with the bounding box as pixel coordinates "
        "[x1, y1, x2, y2] using the image width and height."
    ),
    "grid100_integer": (
        "Locate the object described below in the image.\n"
        "Description: {question}\n"
        "Answer with the bounding box as integer grid coordinates "
        "[gx1, gy1, gx2, gy2] in the 0-100 range, where 100 corresponds to "
        "the image width or height."
    ),
}

PROMPT_SHA256 = {
    name: hashlib.sha256(template.encode("utf-8")).hexdigest()
    for name, template in PROMPT_TEMPLATES.items()
}

DEFAULT_REPRESENTATION = "normalized_decimal"


def validate_representation(representation: str) -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS!r}, got {representation!r}")
    return representation


def prompt_template(representation: str) -> str:
    return PROMPT_TEMPLATES[validate_representation(representation)]


def prompt_id(representation: str) -> str:
    return PROMPT_IDS[validate_representation(representation)]


def render_prompt(question: str, representation: str = DEFAULT_REPRESENTATION) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return prompt_template(representation).format(question=question)


def build_messages(question: str, representation: str = DEFAULT_REPRESENTATION) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": render_prompt(question, representation)},
            ],
        }
    ]


def encode_answer(record: Mapping[str, Any], representation: str = DEFAULT_REPRESENTATION) -> tuple[str, dict[str, Any]]:
    """Return the supervised answer text for one representation plus audit metadata."""

    representation = validate_representation(representation)
    normalized = [float(value) for value in record["answer"]]
    width = int(record["width"])
    height = int(record["height"])
    if representation == "normalized_decimal":
        # Reuse the T5/T6 implementation for byte-compatible A behaviour.
        from .sft_data import format_answer

        text, meta = format_answer(normalized, precision=4)
        return text, {**meta, "representation": representation, "prompt_id": PROMPT_IDS[representation]}
    if representation == "bin1000_integer":
        bins = [max(0, min(1000, int(round(value * 1000.0)))) for value in normalized]
        if not (bins[2] > bins[0] and bins[3] > bins[1]):
            raise ValueError(f"bin1000 answer degenerated for {record['id']}: {bins}")
        text = f"[{bins[0]}, {bins[1]}, {bins[2]}, {bins[3]}]"
        parsed = parse_box(text)
        if parsed is None:
            raise ValueError(f"bin1000 answer is not evaluator-parseable: {text}")
        max_error = max(abs(a - b) for a, b in zip(parsed, normalized))
        if max_error > 0.0005 + 1e-12:
            raise ValueError(f"bin1000 roundtrip error too high for {record['id']}: {max_error}")
        return text, {
            "answer_text": text,
            "precision_used": 0,
            "max_abs_rounding_error": max_error,
            "evaluator_parse_ok": True,
            "representation": representation,
            "prompt_id": PROMPT_IDS[representation],
            "encoded_values": bins,
        }
    if representation == "grid100_integer":
        grid = [int(round(value * 100.0)) for value in normalized]
        if any(value < 0 or value > 100 for value in grid):
            raise ValueError(f"grid100 answer outside [0, 100] for {record['id']}: {grid}")
        if not (grid[2] > grid[0] and grid[3] > grid[1]):
            raise ValueError(f"grid100 answer degenerated for {record['id']}: {grid}")
        text = f"[{grid[0]}, {grid[1]}, {grid[2]}, {grid[3]}]"
        max_error = max(abs(value / 100.0 - original) for value, original in zip(grid, normalized))
        if max_error > 0.005 + 1e-12:
            raise ValueError(f"grid100 roundtrip error too high for {record['id']}: {max_error}")
        return text, {
            "answer_text": text,
            "precision_used": 0,
            "max_abs_rounding_error": max_error,
            "evaluator_parse_ok": False,
            "representation": representation,
            "prompt_id": PROMPT_IDS[representation],
            "encoded_values": grid,
            "grid_conversion_required": True,
            "grid_rounding_tolerance": 0.005 + 1e-12,
        }
    # pixel_xyxy: use the dataset's pixel bbox directly (already integer pixels)
    raw = [int(round(float(value))) for value in record["raw_bbox"]]
    if not (0 <= raw[0] < raw[2] <= width and 0 <= raw[1] < raw[3] <= height):
        raise ValueError(f"raw_bbox outside image for {record['id']}: {raw} image={width}x{height}")
    text = f"[{raw[0]}, {raw[1]}, {raw[2]}, {raw[3]}]"
    normalized_from_raw = [raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height]
    max_error = max(abs(a - b) for a, b in zip(normalized_from_raw, normalized))
    # RefCOCO pixels are integer-rounded relative to the stored normalized answer;
    # allow the exact pixel-rounding bound (<=1px on each axis) instead of 1e-8.
    pixel_rounding_tolerance = (1.0 / width) + (1.0 / height) + 1e-8
    if max_error > pixel_rounding_tolerance:
        raise ValueError(
            f"raw_bbox/answer inconsistency for {record['id']}: {max_error} > {pixel_rounding_tolerance}"
        )
    return text, {
        "answer_text": text,
        "precision_used": 0,
        "max_abs_rounding_error": max_error,
        "evaluator_parse_ok": False,
        "representation": representation,
        "prompt_id": PROMPT_IDS[representation],
        "encoded_values": raw,
        "pixel_conversion_required": True,
        "image_wh": [width, height],
        "pixel_rounding_tolerance": pixel_rounding_tolerance,
        "max_abs_pixel_answer_gap": max_error,
    }


def _bracketed_numbers(text: str) -> tuple[list[float] | None, str | None]:
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


def parse_metric_prediction(
    output_text: str, representation: str, *, width: int, height: int
) -> tuple[list[float] | None, str | None, dict[str, Any]]:
    """Convert one generated answer to a normalized xyxy box for frozen metric code.

    A and B use the frozen evaluator parser unchanged.  C parses pixel xyxy,
    validates against ``width``/``height``, and normalizes; D parses the 0–100
    integer grid, validates ordering/range, and divides by 100.  The frozen
    parser is intentionally not used for C or D.  Returns
    ``(normalized_box, error, audit)``.
    """

    representation = validate_representation(representation)
    audit: dict[str, Any] = {"representation": representation, "conversion_applied": False}
    if representation in ("normalized_decimal", "bin1000_integer"):
        normalized = parse_box(output_text)
        if normalized is None:
            return None, "frozen_parser_rejected_output", audit
        audit["parser"] = "src.eval.grounding.parse_box (frozen)"
        return list(normalized), None, audit
    if representation == "grid100_integer":
        values, error = _bracketed_numbers(output_text if isinstance(output_text, str) else "")
        if values is None:
            return None, error or "grid_parse_failed", audit
        if not all(math.isfinite(value) for value in values):
            return None, "non_finite_coordinate", audit
        if any(value != int(value) for value in values):
            return None, "grid_coordinates_must_be_integers", audit
        grid = [int(value) for value in values]
        if any(value < 0 or value > 100 for value in grid):
            return None, "grid_box_outside_0_100", audit
        if not (grid[2] > grid[0] and grid[3] > grid[1]):
            return None, "grid_box_reversed", audit
        normalized = [grid[0] / 100.0, grid[1] / 100.0, grid[2] / 100.0, grid[3] / 100.0]
        try:
            normalized = list(validate_box(normalized, normalized=True))
        except ValueError as exc:
            return None, f"normalized_box_invalid: {exc}", audit
        audit.update({"conversion_applied": True, "grid_box": grid})
        return normalized, None, audit
    values, error = _bracketed_numbers(output_text if isinstance(output_text, str) else "")
    if values is None:
        return None, error or "pixel_parse_failed", audit
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_coordinate", audit
    if any(value != int(value) for value in values):
        return None, "pixel_coordinates_must_be_integers", audit
    pixels = [int(value) for value in values]
    if not (0 <= pixels[0] < pixels[2] <= int(width) and 0 <= pixels[1] < pixels[3] <= int(height)):
        return None, "pixel_box_outside_or_reversed", audit
    normalized = [pixels[0] / width, pixels[1] / height, pixels[2] / width, pixels[3] / height]
    try:
        normalized = list(validate_box(normalized, normalized=True))
    except ValueError as exc:
        return None, f"normalized_box_invalid: {exc}", audit
    audit.update({"conversion_applied": True, "pixel_box": pixels, "image_wh": [int(width), int(height)]})
    return normalized, None, audit


def score_prediction_variant(output_text: str, record: Mapping[str, Any], representation: str) -> tuple[Score, dict[str, Any]]:
    """Diagnostic-only scoring that mirrors the frozen metric contract for all reps."""

    representation = validate_representation(representation)
    normalized, error, audit = parse_metric_prediction(
        output_text, representation, width=int(record["width"]), height=int(record["height"])
    )
    gt = validate_box(record["answer"], normalized=True)
    if normalized is None:
        return Score(None, 0.0, False, error or "invalid_prediction"), audit
    value = iou(normalized, gt)
    return Score(tuple(normalized), value, value >= ACC_THRESHOLD, None), audit
