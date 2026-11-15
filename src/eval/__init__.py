"""Independent grounding evaluation; core metrics have no torch dependency."""

from .grounding import (
    Score,
    evaluate_outputs,
    iou,
    metrics_from_scores,
    parse_box,
    parse_box_with_error,
    score_prediction,
    validate_box,
)

__all__ = [
    "Score",
    "evaluate_outputs",
    "iou",
    "metrics_from_scores",
    "parse_box",
    "parse_box_with_error",
    "score_prediction",
    "validate_box",
]
