"""Read-only preflight for the project's eight-field grounding JSONL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .grounding import validate_box

REQUIRED_KEYS = ("id", "image", "question", "answer", "raw_bbox", "width", "height", "source")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def validate_record(
    record: dict[str, Any], *, project_root: Path, expected_source: str | None, check_image: bool
) -> None:
    record_id = record.get("id", "<missing id>")
    if set(record) != set(REQUIRED_KEYS):
        raise ValueError(f"{record_id}: expected exactly eight fields {REQUIRED_KEYS}, got {tuple(record)}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("id must be a non-empty string")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError(f"{record_id}: question must be non-empty text")
    if expected_source is not None and record["source"] != expected_source:
        raise ValueError(f"{record_id}: source {record['source']!r} != {expected_source!r}")
    width, height = record["width"], record["height"]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{record_id}: invalid image dimensions {width!r}x{height!r}")
    answer = validate_box(record["answer"], normalized=True)
    raw = validate_box(record["raw_bbox"], normalized=False)
    if not (0.0 <= raw[0] < raw[2] <= width and 0.0 <= raw[1] < raw[3] <= height):
        raise ValueError(f"{record_id}: raw_bbox is outside image bounds: {raw}")
    expected = [raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height]
    if max(abs(a - b) for a, b in zip(answer, expected)) > 1.1e-8:
        raise ValueError(f"{record_id}: answer is inconsistent with raw_bbox/width/height")
    image_value = record["image"]
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{record_id}: image must be a non-empty relative path")
    relative = Path(image_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{record_id}: unsafe image path {image_value!r}")
    image_path = project_root / relative
    if not image_path.is_file():
        raise FileNotFoundError(f"{record_id}: missing image {image_path}")
    if check_image:
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(f"{record_id}: actual image size {image.size} != {(width, height)}")
            image.verify()


def load_dataset(
    path: Path,
    *,
    project_root: Path,
    expected_source: str,
    split: str,
    expected_n: int | None = None,
    limit: int | None = None,
    check_images: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate without modifying any data or manifests."""

    rows = read_jsonl(path)
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} rows, found {len(rows)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    seen: set[str] = set()
    for record in rows:
        record_id = record.get("id", "<missing id>")
        try:
            validate_record(
                record,
                project_root=project_root,
                expected_source=expected_source,
                check_image=check_images,
            )
            if record["id"] in seen:
                raise ValueError("duplicate selected id")
        except Exception as exc:
            raise ValueError(
                f"data/GT validation aborted: dataset={expected_source} split={split} "
                f"id={record_id} reason={type(exc).__name__}: {exc}"
            ) from exc
        seen.add(record["id"])
    return rows
