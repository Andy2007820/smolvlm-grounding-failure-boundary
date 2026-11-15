"""Shared visual-token contract for training and independent evaluation.

The zero-shot run recorded a CUDA device-side assert because the processor's
default ``image_seq_len`` (64, valid for 512 px tiles) did not match the 384 px
no-split sequence.  The evaluator fixed this by deriving the placeholder count
from the configured image size, patch size and scale factor:

    image_seq_len = (image_size / (patch_size * scale_factor)) ** 2

Training must use the *same* computation and must never silently fall back to
the processor's 64-token default.  This module re-uses the evaluator helper
(``src.eval.run_zeroshot.compute_image_seq_len``) and adds assertions against the
actual ``pixel_values`` tile geometry and the ``<image>`` placeholder count.

The 64 default is only changed on the in-memory processor instance for the
duration of one call.  ``processor_config.json`` on disk is read but never
modified; every call returns an audit block proving the shipped default stayed
at 64.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.eval.run_zeroshot import compute_image_seq_len as evaluator_compute_image_seq_len

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSOR_CONFIG = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct" / "processor_config.json"

#: Value shipped in models/SmolVLM-256M-Instruct/processor_config.json.
PROCESSOR_DEFAULT_IMAGE_SEQ_LEN = 64


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_image_seq_len(image_size: int, patch_size: int, scale_factor: int) -> int:
    """Delegate to the evaluator helper so train/eval cannot drift apart."""

    return int(
        evaluator_compute_image_seq_len(
            image_size=image_size, patch_size=patch_size, scale_factor=scale_factor
        )
    )


def compute_tokens_per_view(height: int, width: int, patch_size: int, scale_factor: int) -> int:
    """Tokens for one actual (padded, no-split) image tile.

    Idefics3 pixel-shuffles ``patch_size x patch_size`` patches with
    ``scale_factor`` so one output token covers ``patch_size * scale_factor``
    pixels on each axis.  The actual ``pixel_values`` tile is padded to a square
    by the image processor; the tile geometry is therefore authoritative.
    """

    for name, value in (("height", height), ("width", width), ("patch_size", patch_size), ("scale_factor", scale_factor)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    reduction = patch_size * scale_factor
    if height % reduction or width % reduction:
        raise ValueError(
            f"tile {height}x{width} is not an exact multiple of patch*scale={reduction}"
        )
    return (height // reduction) * (width // reduction)


@dataclass(frozen=True)
class VisualTokenContract:
    """Auditable record of one processed image and its placeholder tokens."""

    batch_size: int
    views_per_sample: int
    total_views: int
    channels: int
    view_height: int
    view_width: int
    patch_size: int
    scale_factor: int
    tokens_per_view: int
    derived_visual_tokens: int
    placeholder_count: int
    expected_image_seq_len: int
    image_token_id: int
    pixel_values_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pixel_values_shape"] = list(self.pixel_values_shape)
        return payload


def inspect_visual_token_contract(
    inputs: Mapping[str, Any],
    *,
    patch_size: int,
    scale_factor: int,
    image_size: int,
    image_token_id: int,
) -> VisualTokenContract:
    """Assert views * tokens-per-view == number of ``<image>`` placeholders.

    ``pixel_values`` is expected in the Idefics3 batch layout
    ``[batch, views, channels, height, width]``.  For the project's no-split
    single-view path this is ``[1, 1, 3, 384, 384]``.
    """

    if "pixel_values" not in inputs:
        raise KeyError("processed inputs lack pixel_values")
    pixel_values = inputs["pixel_values"]
    shape = tuple(int(value) for value in pixel_values.shape)
    if len(shape) != 5:
        raise ValueError(f"expected pixel_values [batch, views, 3, H, W], got {shape}")
    batch_size, views_per_sample, channels, view_height, view_width = shape
    if channels != 3:
        raise ValueError(f"expected 3 channels, got {channels}")
    if batch_size != 1:
        raise ValueError(
            "visual-token preflight is defined for one image view set at a time, "
            f"got batch_size={batch_size}"
        )

    total_views = batch_size * views_per_sample
    tokens_per_view = compute_tokens_per_view(view_height, view_width, patch_size, scale_factor)
    derived_visual_tokens = total_views * tokens_per_view
    expected_image_seq_len = compute_image_seq_len(image_size, patch_size, scale_factor)

    if "input_ids" not in inputs:
        raise KeyError("processed inputs lack input_ids")
    placeholder_count = int((inputs["input_ids"] == image_token_id).sum())
    if placeholder_count != derived_visual_tokens:
        raise AssertionError(
            "visual-token contract mismatch: "
            f"views({total_views}) * tokens_per_view({tokens_per_view}) = "
            f"{derived_visual_tokens} != placeholders({placeholder_count})"
        )
    if expected_image_seq_len != derived_visual_tokens:
        raise AssertionError(
            "configured image_seq_len disagrees with the actual tile geometry: "
            f"image_size={image_size} -> {expected_image_seq_len} != {derived_visual_tokens}"
        )

    return VisualTokenContract(
        batch_size=batch_size,
        views_per_sample=views_per_sample,
        total_views=total_views,
        channels=channels,
        view_height=view_height,
        view_width=view_width,
        patch_size=patch_size,
        scale_factor=scale_factor,
        tokens_per_view=tokens_per_view,
        derived_visual_tokens=derived_visual_tokens,
        placeholder_count=placeholder_count,
        expected_image_seq_len=expected_image_seq_len,
        image_token_id=image_token_id,
        pixel_values_shape=shape,
    )


def set_runtime_image_seq_len(
    processor: Any,
    image_seq_len: int,
    *,
    processor_config_path: Path = DEFAULT_PROCESSOR_CONFIG,
    expected_default: int = PROCESSOR_DEFAULT_IMAGE_SEQ_LEN,
) -> dict[str, Any]:
    """Set ``processor.image_seq_len`` for the current call and prove the default is untouched.

    ``processor_config.json`` must stay at its shipped value (64).  The helper
    raises if the on-disk value changed or if writing the runtime attribute
    mutates the file.  Callers invoke it on *every* image-processing call so the
    global default is never permanently overridden.
    """

    if isinstance(image_seq_len, bool) or not isinstance(image_seq_len, int) or image_seq_len <= 0:
        raise ValueError(f"image_seq_len must be a positive integer, got {image_seq_len!r}")
    config_path = Path(processor_config_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    before_bytes = config_path.read_bytes()
    before_payload = json.loads(before_bytes.decode("utf-8"))
    on_disk_default = int(before_payload["image_seq_len"])
    if on_disk_default != expected_default:
        raise RuntimeError(
            f"processor_config.json changed: image_seq_len={on_disk_default}, expected {expected_default}; "
            "the shipped default must not be modified"
        )
    previous = getattr(processor, "image_seq_len", None)
    processor.image_seq_len = image_seq_len
    after_bytes = config_path.read_bytes()
    if sha256_bytes(before_bytes) != sha256_bytes(after_bytes):
        raise RuntimeError("processor_config.json was modified by setting the runtime image_seq_len")
    return {
        "processor_config_path": config_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "processor_config_default_image_seq_len": on_disk_default,
        "processor_config_sha256": sha256_bytes(before_bytes),
        "processor_image_seq_len_before_call": previous,
        "runtime_image_seq_len": image_seq_len,
        "processor_image_seq_len_after_call": int(processor.image_seq_len),
        "global_default_unchanged": True,
    }
