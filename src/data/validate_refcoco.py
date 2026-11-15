#!/usr/bin/env python3
"""Independently validate final RefCOCO JSONL, manifests, images and QA artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
REQUIRED_KEYS = ["id", "image", "question", "answer", "raw_bbox", "width", "height", "source"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: not an object")
            rows.append(value)
    return rows


def validate_record(record: dict[str, Any], split: str) -> list[str]:
    errors = []
    if list(record) != REQUIRED_KEYS:
        errors.append(f"{record.get('id')}: keys/order {list(record)} != {REQUIRED_KEYS}")
    record_id = record.get("id", "")
    if not isinstance(record_id, str) or not record_id.startswith(f"refcoco_{split}_"):
        errors.append(f"invalid id for {split}: {record_id}")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        errors.append(f"{record_id}: empty question")
    if record.get("source") != "refcoco":
        errors.append(f"{record_id}: wrong source")
    answer = record.get("answer")
    if (
        not isinstance(answer, list)
        or len(answer) != 4
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in answer)
        or not all(0 <= v <= 1 for v in answer)
        or answer[2] <= answer[0]
        or answer[3] <= answer[1]
    ):
        errors.append(f"{record_id}: invalid normalized xyxy answer {answer}")
    raw_bbox = record.get("raw_bbox")
    if (
        not isinstance(raw_bbox, list)
        or len(raw_bbox) != 4
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw_bbox)
        or raw_bbox[2] <= raw_bbox[0]
        or raw_bbox[3] <= raw_bbox[1]
    ):
        errors.append(f"{record_id}: invalid raw pixel xyxy {raw_bbox}")
    width, height = record.get("width"), record.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        errors.append(f"{record_id}: invalid dimensions {width}x{height}")
        return errors
    image_value = record.get("image")
    if not isinstance(image_value, str):
        errors.append(f"{record_id}: image is not a string")
        return errors
    relative = Path(image_value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{record_id}: unsafe image path {image_value}")
        return errors
    image_path = PROJECT_ROOT / relative
    if not image_path.exists():
        errors.append(f"{record_id}: missing image {image_value}")
        return errors
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (width, height):
                errors.append(f"{record_id}: actual size {image.size} != {(width, height)}")
        # Recompute normalized xyxy from raw_bbox with boundary clipping.
        clipped = [
            max(0.0, min(float(width), raw_bbox[0])),
            max(0.0, min(float(height), raw_bbox[1])),
            max(0.0, min(float(width), raw_bbox[2])),
            max(0.0, min(float(height), raw_bbox[3])),
        ]
        expected = [clipped[0] / width, clipped[1] / height, clipped[2] / width, clipped[3] / height]
        if max(abs(a - b) for a, b in zip(answer, expected)) > 1.1e-8:
            errors.append(f"{record_id}: answer does not match raw_bbox/width/height: {answer} vs {expected}")
    except Exception as exc:
        errors.append(f"{record_id}: image/read/bbox validation error {type(exc).__name__}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "started",
        "expected_counts": {"train": args.train_size, "val": args.val_size},
    }
    errors: list[str] = []
    try:
        final = {
            split: read_jsonl(DATA_ROOT / f"refcoco_{split}.jsonl")
            for split in ("train", "val")
        }
        manifests = {
            split: read_jsonl(DATA_ROOT / "manifests" / f"refcoco_{split}.jsonl")
            for split in ("train", "val")
        }
        for split, expected_count in (("train", args.train_size), ("val", args.val_size)):
            rows = final[split]
            manifest = manifests[split]
            if len(rows) != expected_count:
                errors.append(f"{split}: final count {len(rows)} != {expected_count}")
            if len(manifest) != expected_count:
                errors.append(f"{split}: manifest count {len(manifest)} != {expected_count}")
            if len({r["id"] for r in rows}) != len(rows):
                errors.append(f"{split}: duplicate final ids")
            source_keys = [(m["ref_id"], m["sent_id"], m["ann_id"]) for m in manifest]
            if len(set(source_keys)) != len(source_keys):
                errors.append(f"{split}: duplicate manifest source keys")
            manifest_by_id = {m["id"]: m for m in manifest}
            for record in rows:
                errors.extend(validate_record(record, split))
                source = manifest_by_id.get(record["id"])
                if source is None:
                    errors.append(f"{record['id']}: absent from manifest")
                    continue
                for key in REQUIRED_KEYS:
                    if record[key] != source[key]:
                        errors.append(f"{record['id']}: final/manifest mismatch for {key}")
                raw_xywh = source["raw_xywh"]
                expected_raw = [
                    raw_xywh[0], raw_xywh[1], raw_xywh[0] + raw_xywh[2], raw_xywh[1] + raw_xywh[3]
                ]
                if max(abs(a - b) for a, b in zip(record["raw_bbox"], expected_raw)) > 5.1e-6:
                    errors.append(f"{record['id']}: raw_bbox does not match source raw_xywh")
            report[f"{split}_actual_records"] = len(rows)
            report[f"{split}_unique_images"] = len({r["image"] for r in rows})

        for field in ("id", "image"):
            overlap = {r[field] for r in final["train"]} & {r[field] for r in final["val"]}
            report[f"train_val_{field}_intersection_count"] = len(overlap)
            if overlap:
                errors.append(f"train/val {field} overlap: {sorted(overlap)[:10]}")
        for field in ("ref_id", "sent_id", "ann_id", "image_id"):
            overlap = {m[field] for m in manifests["train"]} & {m[field] for m in manifests["val"]}
            report[f"train_val_{field}_intersection_count"] = len(overlap)
            if overlap:
                errors.append(f"train/val source {field} overlap: {sorted(overlap)[:10]}")

        stats_path = DATA_ROOT / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for split in ("train", "val"):
            if stats["splits"][split]["records"] != len(final[split]):
                errors.append(f"stats {split} record count mismatch")
            if stats["splits"][split]["unique_images"] != len({r["image"] for r in final[split]}):
                errors.append(f"stats {split} unique image count mismatch")
            output_name = f"refcoco_{split}.jsonl"
            actual_hash = sha256_file(DATA_ROOT / output_name)
            if stats["outputs"][output_name]["sha256"] != actual_hash:
                errors.append(f"stats hash mismatch for {output_name}")
            report[f"{output_name}_sha256"] = actual_hash

        qa_path = DATA_ROOT / "qa" / "refcoco" / "sample_manifest.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        samples = qa["samples"]
        if len(samples) != 20:
            errors.append(f"QA sample count {len(samples)} != 20")
        for split in ("train", "val"):
            if sum(s["split"] == split for s in samples) != 10:
                errors.append(f"QA {split} sample count is not 10")
        final_by_id = {r["id"]: r for rows in final.values() for r in rows}
        for sample in samples:
            if not (PROJECT_ROOT / sample["qa_image"]).exists():
                errors.append(f"missing QA overlay {sample['qa_image']}")
            source = final_by_id.get(sample["id"])
            if source is None:
                errors.append(f"QA id {sample['id']} not in final JSONL")
            elif sample["answer"] != source["answer"] or sample["image"] != source["image"]:
                errors.append(f"QA sample mismatch for {sample['id']}")
        report["qa_sample_count"] = len(samples)
        report["errors"] = errors
        report["status"] = "passed" if not errors else "failed"
    except Exception as exc:
        errors.append(f"validator exception: {type(exc).__name__}: {exc}")
        report["errors"] = errors
        report["status"] = "failed"
    output = DATA_ROOT / "qa" / "refcoco" / "independent_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
