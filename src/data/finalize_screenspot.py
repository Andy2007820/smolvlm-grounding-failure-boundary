#!/usr/bin/env python3
"""Record vision-assisted ScreenSpot QA without changing JSONL or source images."""

from __future__ import annotations

import json

import prepare_screenspot as ps


OBSERVATIONS = {
    "screenspot_test_000713": "box encloses the selected Rule-based mode control",
    "screenspot_test_001011": "box encloses the Most commented sorting option",
    "screenspot_test_000840": "box encloses the GitLab add-new plus/dropdown control",
    "screenspot_test_000653": "box encloses the Wallet action",
    "screenspot_test_000233": "box encloses the Allow Notifications toggle",
    "screenspot_test_000957": "box encloses the Carnation Breakfast Essentials delete/trash icon",
    "screenspot_test_001143": "box encloses the Wikipedia footer link",
    "screenspot_test_000036": "box encloses Account protection in Windows Security navigation",
    "screenspot_test_000211": "box encloses the macOS Bluetooth back arrow",
    "screenspot_test_001124": "box encloses the OpenStreetMap Sign Up button",
}


def main() -> None:
    qa_path = ps.DATA_ROOT / "qa" / "screenspot" / "sample_manifest.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    sample_ids = [sample["id"] for sample in qa["samples"]]
    if len(sample_ids) != 10 or set(sample_ids) != set(OBSERVATIONS):
        raise RuntimeError(f"QA sample IDs differ from reviewed IDs: {sample_ids}")
    output_hash = ps.sha256_file(ps.DATA_ROOT / "screenspot.jsonl")
    stats_path = ps.DATA_ROOT / "screenspot_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if output_hash != stats["outputs"]["screenspot.jsonl"]["sha256"]:
        raise RuntimeError("screenspot.jsonl changed since generation")
    review = {
        "status": "passed",
        "reviewer": "deepseek-flash via image_understand fallback; coordinates assessed by data-engineer",
        "reviewed_at": ps.utc_now(),
        "full_resolution_samples_reviewed": len(sample_ids),
        "observations": OBSERVATIONS,
        "human_relabeling": False,
    }
    qa["visual_review"] = review
    ps.write_json(qa_path, qa)
    conflict_path = ps.DATA_ROOT / "qa" / "screenspot" / "conflict" / "pair.json"
    conflict = json.loads(conflict_path.read_text(encoding="utf-8"))
    conflict["visual_review"] = {
        "status": "passed_with_source_annotation_variation",
        "observation": "both boxes surround the same Done button on an identical payment-success screenshot; box extents differ",
        "reviewer": review["reviewer"],
        "disposition": "retain both original source annotations; not exact duplicates; no relabeling",
    }
    ps.write_json(conflict_path, conflict)
    finalization = {
        "status": "passed",
        "review": review,
        "conflict_review": conflict["visual_review"] | {"ids": conflict["ids"], "bbox_iou": conflict["bbox_iou"]},
        "screenspot_jsonl_sha256": output_hash,
        "data_files_changed": False,
        "source_images_changed": False,
    }
    ps.write_json(ps.DATA_ROOT / "qa" / "screenspot" / "finalization.json", finalization)
    stats["qa"]["visual_review"] = "passed"
    stats["qa"]["visual_review_details"] = "data/qa/screenspot/finalization.json"
    stats["quality_checks"]["source_annotation_variation_group_count"] = stats["processing_audit"]["conflicting_same_image_instruction_count"]
    ps.write_json(stats_path, stats)
    shared_path = ps.DATA_ROOT / "stats.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    shared["screenspot"] = stats
    ps.write_json(shared_path, shared)
    print(json.dumps({"status": "passed", "reviewed_samples": len(sample_ids), "jsonl_sha256": output_hash,
                      "conflict_ids": conflict["ids"], "conflict_iou": conflict["bbox_iou"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
