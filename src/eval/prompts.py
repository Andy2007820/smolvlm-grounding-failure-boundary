"""Shared prompts for training and independent grounding evaluation.

``p1`` is the frozen normalized-decimal prompt used by B0/T6. The T7 coordinate
ablation adds representation-specific prompts (``p1_bin1000``, ``p1_pixel``,
``p1_grid100``) whose text must match ``src/train/coordinate_variants.py``
byte-for-byte (enforced by a CPU test and by the validator recording the prompt
SHA256).
"""

from __future__ import annotations

import hashlib
from typing import Any

PROMPT_ID = "p1"
PROMPT_TEMPLATE = (
    "Locate the object described below in the image.\n"
    "Description: {question}\n"
    "Answer with the bounding box as normalized coordinates "
    "[x1, y1, x2, y2], each between 0 and 1."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

PROMPT_ID_BIN1000 = "p1_bin1000"
PROMPT_TEMPLATE_BIN1000 = (
    "Locate the object described below in the image.\n"
    "Description: {question}\n"
    "Answer with the bounding box as integer coordinates "
    "[x1, y1, x2, y2] in the 0-1000 range, where 1000 corresponds to "
    "the image width or height."
)
PROMPT_SHA256_BIN1000 = hashlib.sha256(PROMPT_TEMPLATE_BIN1000.encode("utf-8")).hexdigest()

PROMPT_ID_PIXEL = "p1_pixel"
PROMPT_TEMPLATE_PIXEL = (
    "Locate the object described below in the image.\n"
    "Description: {question}\n"
    "Answer with the bounding box as pixel coordinates "
    "[x1, y1, x2, y2] using the image width and height."
)
PROMPT_SHA256_PIXEL = hashlib.sha256(PROMPT_TEMPLATE_PIXEL.encode("utf-8")).hexdigest()

PROMPT_ID_GRID100 = "p1_grid100"
PROMPT_TEMPLATE_GRID100 = (
    "Locate the object described below in the image.\n"
    "Description: {question}\n"
    "Answer with the bounding box as integer grid coordinates "
    "[gx1, gy1, gx2, gy2] in the 0-100 range, where 100 corresponds to "
    "the image width or height."
)
PROMPT_SHA256_GRID100 = hashlib.sha256(PROMPT_TEMPLATE_GRID100.encode("utf-8")).hexdigest()

PROMPT_VARIANTS: dict[str, dict[str, Any]] = {
    PROMPT_ID: {
        "template": PROMPT_TEMPLATE,
        "sha256": PROMPT_SHA256,
        "representation": "normalized_decimal",
        "conversion_required": False,
    },
    PROMPT_ID_BIN1000: {
        "template": PROMPT_TEMPLATE_BIN1000,
        "sha256": PROMPT_SHA256_BIN1000,
        "representation": "bin1000_integer",
        "conversion_required": False,
    },
    PROMPT_ID_PIXEL: {
        "template": PROMPT_TEMPLATE_PIXEL,
        "sha256": PROMPT_SHA256_PIXEL,
        "representation": "pixel_xyxy",
        "conversion_required": True,
    },
    PROMPT_ID_GRID100: {
        "template": PROMPT_TEMPLATE_GRID100,
        "sha256": PROMPT_SHA256_GRID100,
        "representation": "grid100_integer",
        "conversion_required": True,
    },
}

DEFAULT_PROMPT_ID = PROMPT_ID


def validate_prompt_id(prompt_id: str) -> str:
    if prompt_id not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt_id {prompt_id!r}; expected one of {tuple(PROMPT_VARIANTS)}")
    return prompt_id


def prompt_template_for(prompt_id: str) -> str:
    return PROMPT_VARIANTS[validate_prompt_id(prompt_id)]["template"]


def prompt_sha256_for(prompt_id: str) -> str:
    return PROMPT_VARIANTS[validate_prompt_id(prompt_id)]["sha256"]


def representation_for(prompt_id: str) -> str:
    return PROMPT_VARIANTS[validate_prompt_id(prompt_id)]["representation"]


def render_prompt_for(prompt_id: str, question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return prompt_template_for(prompt_id).format(question=question)


def build_messages_for(prompt_id: str, question: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": render_prompt_for(prompt_id, question)},
            ],
        }
    ]


def render_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    # The image content item inserts the sole <image> token through the chat
    # template; do not duplicate it in this text.
    return PROMPT_TEMPLATE.format(question=question)


def build_messages(question: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": render_prompt(question)},
            ],
        }
    ]
