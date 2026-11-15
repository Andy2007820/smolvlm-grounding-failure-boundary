#!/usr/bin/env python3
"""Independently re-read and validate final synthetic JSONL/images/manifests/QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

import make_synthetic as ms


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def validate_question(manifest: dict[str, Any]) -> None:
    metadata = manifest["question_metadata"]
    target = manifest["objects"][manifest["target_index"]]
    tx, ty = ms.center(target)
    template = metadata["template"]
    if template == "quadrant":
        assert metadata["vertical"] == ("top" if ty < manifest["height"] / 2 else "bottom")
        assert metadata["horizontal"] == ("left" if tx < manifest["width"] / 2 else "right")
    elif template in ("relation", "occlusion"):
        reference = manifest["objects"][metadata["reference_index"]]
        rx, ry = ms.center(reference)
        relation = metadata["relation"]
        if relation == "to the left of":
            assert tx < rx
        elif relation == "to the right of":
            assert tx > rx
        elif relation == "above":
            assert ty < ry
        elif relation == "below":
            assert ty > ry
        elif relation == "behind":
            assert manifest["target_index"] < metadata["reference_index"]
        else:
            raise AssertionError(f"unknown relation {relation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=1000)
    args = parser.parse_args()
    errors: list[str] = []
    counters: Counter = Counter()
    records: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    output_hashes: dict[str, str] = {}

    for split, expected in (("train", args.train_size), ("eval", args.eval_size)):
        jsonl_path = ms.DATA_ROOT / f"synthetic_{split}.jsonl"
        manifest_path = ms.DATA_ROOT / "manifests" / f"synthetic_{split}.jsonl"
        try:
            records[split] = read_jsonl(jsonl_path)
            manifests[split] = read_jsonl(manifest_path)
            output_hashes[jsonl_path.name] = ms.sha256_file(jsonl_path)
        except Exception as exc:
            errors.append(f"{split}: JSONL read failed: {exc}")
            records[split], manifests[split] = [], []
            continue
        if len(records[split]) != expected:
            errors.append(f"{split}: expected {expected} records, got {len(records[split])}")
        if len(manifests[split]) != len(records[split]):
            errors.append(f"{split}: manifest/record count mismatch")
        counters[f"{split}_actual_records"] = len(records[split])
        counters[f"{split}_unique_images"] = len({r.get("image") for r in records[split]})
        for index, (record, manifest) in enumerate(zip(records[split], manifests[split])):
            label = f"{split}[{index}]"
            try:
                assert list(record) == list(ms.FINAL_KEYS), "final JSONL keys/order mismatch"
                assert record["source"] == "synthetic"
                assert record["id"] == f"synthetic_{split}_{index + 1:06d}"
                assert record["image"] == f"data/images/synthetic/{split}/{record['id']}.png"
                for key in ms.FINAL_KEYS:
                    assert record[key] == manifest[key], f"manifest field mismatch: {key}"
                assert manifest["generator_version"] == ms.GENERATOR_VERSION
                assert manifest["split_seed"] == ms.split_seed(42, split)
                width, height = record["width"], record["height"]
                assert width > 0 and height > 0
                raw = record["raw_bbox"]
                assert len(raw) == 4 and all(isinstance(v, int) and not isinstance(v, bool) for v in raw)
                assert 0 <= raw[0] < raw[2] <= width
                assert 0 <= raw[1] < raw[3] <= height
                expected_answer = [round(raw[0] / width, 8), round(raw[1] / height, 8), round(raw[2] / width, 8), round(raw[3] / height, 8)]
                assert record["answer"] == expected_answer, "answer normalization mismatch"
                image_path = ms.PROJECT_ROOT / record["image"]
                with Image.open(image_path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
                    assert image.size == (width, height), "image dimensions mismatch"
                assert ms.sha256_file(image_path) == manifest["image_sha256"], "image hash mismatch"
                assert image_path.stat().st_size == manifest["image_bytes"], "image byte count mismatch"
                rendered = ms.render_scene(manifest)
                if rendered.tobytes() != image.tobytes():
                    counters[f"{split}_render_mismatch_count"] += 1
                objects = manifest["objects"]
                combos = [(obj["color"], obj["shape"]) for obj in objects]
                assert len(combos) == len(set(combos)), "duplicate color/shape combination"
                target = objects[manifest["target_index"]]
                assert target["box"] == raw, "target/raw bbox mismatch"
                assert target["color"] in record["question"] and target["shape"] in record["question"]
                fractions = ms.visibility(manifest)
                assert len(fractions) == len(manifest["visible_fractions"])
                if max(abs(a - b) for a, b in zip(fractions, manifest["visible_fractions"])) > 1e-12:
                    counters[f"{split}_visibility_mismatch_count"] += 1
                assert manifest["target_visible_fraction"] == fractions[manifest["target_index"]]
                assert fractions[manifest["target_index"]] >= 0.55
                if manifest["requested_target_occlusion"]:
                    assert fractions[manifest["target_index"]] <= 0.90
                validate_question(manifest)
                assert manifest["scene_signature"] == ms.scene_signature(manifest), "scene signature mismatch"
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if all(records.values()) and all(manifests.values()):
        checks = ms.cross_split_checks(manifests)
        counters.update(checks)
        for name, value in checks.items():
            if value:
                errors.append(f"{name}={value}")
        for split in ("train", "eval"):
            for field in ("id", "image", "scene_signature", "image_sha256"):
                values = [m[field] for m in manifests[split]]
                duplicates = len(values) - len(set(values))
                counters[f"{split}_duplicate_{field}_count"] = duplicates
                if duplicates:
                    errors.append(f"{split}: duplicate {field}: {duplicates}")

    stats_path = ms.DATA_ROOT / "synthetic_stats.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert stats["config"]["seed"] == 42
        assert stats["config"]["train_split_seed"] == 42
        assert stats["config"]["eval_split_seed"] == 43
        for split, expected in (("train", args.train_size), ("eval", args.eval_size)):
            assert stats["splits"][split]["records"] == expected
            assert len(stats["splits"][split]["target_shape_counts"]) == 5
            assert len(stats["splits"][split]["target_color_counts"]) == 6
            assert stats["splits"][split]["target_at_least_10pct_occluded_count"] > 0
            assert max(int(k) for k in stats["splits"][split]["object_count_distribution"]) == 6
            assert min(int(k) for k in stats["splits"][split]["object_count_distribution"]) >= 1
        for name, digest in output_hashes.items():
            assert stats["outputs"][name]["sha256"] == digest
        shared = json.loads((ms.DATA_ROOT / "stats.json").read_text(encoding="utf-8"))
        assert shared["synthetic"]["outputs"] == stats["outputs"]
    except Exception as exc:
        errors.append(f"stats validation: {type(exc).__name__}: {exc}")

    qa_sample_count = 0
    try:
        qa = json.loads((ms.DATA_ROOT / "qa" / "synthetic" / "sample_manifest.json").read_text(encoding="utf-8"))
        qa_sample_count = len(qa["samples"])
        assert qa_sample_count == 20
        assert Counter(sample["split"] for sample in qa["samples"]) == {"train": 10, "eval": 10}
        assert qa.get("visual_review") == "passed", "visual review is not marked passed"
        for sample in qa["samples"]:
            assert (ms.PROJECT_ROOT / sample["qa_image"]).is_file()
            split = sample["split"]
            match = next(r for r in records[split] if r["id"] == sample["id"])
            for key in ("image", "question", "answer", "raw_bbox"):
                assert sample[key] == match[key]
        for split in ("train", "eval"):
            assert (ms.DATA_ROOT / "qa" / "synthetic" / f"{split}_contact_sheet.png").is_file()
        unit = json.loads((ms.DATA_ROOT / "qa" / "synthetic" / "unit_tests.json").read_text(encoding="utf-8"))
        assert unit["status"] == "passed" and unit["failures"] == 0 and unit["errors"] == 0
    except Exception as exc:
        errors.append(f"QA validation: {type(exc).__name__}: {exc}")

    report = {
        "validated_at": ms.utc_now(),
        "status": "passed" if not errors else "failed",
        "expected_counts": {"train": args.train_size, "eval": args.eval_size},
        **dict(sorted(counters.items())),
        "output_sha256": output_hashes,
        "qa_sample_count": qa_sample_count,
        "errors": errors,
    }
    report_path = ms.DATA_ROOT / "qa" / "synthetic" / "independent_validation.json"
    ms.write_json(report_path, report)
    console_report = {**report, "error_count": len(errors), "errors": errors[:50]}
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
