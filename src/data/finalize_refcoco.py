#!/usr/bin/env python3
"""Record RefCOCO visual review and derive final download/schema audit fields."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" / "refcoco"
QA_ROOT = DATA_ROOT / "qa" / "refcoco"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-status", choices=["passed", "issues"], required=True)
    parser.add_argument("--review-notes", required=True)
    parser.add_argument(
        "--reviewer",
        default="data-engineer; vision-assisted inspection through the configured fallback vision model (not a human review)",
    )
    args = parser.parse_args()

    stats_path = DATA_ROOT / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    qa_path = QA_ROOT / "sample_manifest.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    review = {
        "status": args.visual_status,
        "reviewed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reviewer": args.reviewer,
        "notes": args.review_notes,
        "sample_count": len(qa["samples"]),
        "contact_sheets": [
            "data/qa/refcoco/train_contact_sheet.png",
            "data/qa/refcoco/val_contact_sheet.png",
        ],
        "review_type": "vision-assisted overlay review; contact sheets remain available for independent human inspection",
    }
    qa["visual_review"] = review
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["qa"]["visual_review"] = review

    download_path = RAW_ROOT / "download_log.jsonl"
    results = read_jsonl(download_path)
    status_counts = Counter(r["status"] for r in results)
    final_rows = {
        split: read_jsonl(DATA_ROOT / f"refcoco_{split}.jsonl")
        for split in ("train", "val")
    }
    final_images = sorted({r["image"] for rows in final_rows.values() for r in rows})
    missing = [image for image in final_images if not (PROJECT_ROOT / image).exists()]
    download_audit = {
        **stats.get("downloads", {}),
        "result_status_counts": dict(sorted(status_counts.items())),
        "http_attempt_count_excluding_cache_hits": sum(r.get("attempt_count", 0) for r in results),
        "retry_attempt_count_after_first_attempt": sum(
            max(0, r.get("attempt_count", 0) - 1) for r in results if r["status"] != "cached"
        ),
        "result_with_retry_count": sum(r.get("attempt_count", 0) > 1 for r in results),
        "successful_download_with_retry_count": sum(
            r["status"] == "downloaded" and r.get("attempt_count", 0) > 1 for r in results
        ),
        "failed_after_retries_result_count": sum(r["status"] == "failed" for r in results),
        "final_selected_unique_image_count": len(final_images),
        "final_selected_missing_image_count": len(missing),
        "final_selected_missing_images": missing,
        "failure_semantics": "failed results are historical attempts; deterministic replacement rounds exclude those images. Final missing count must be zero.",
    }
    if missing:
        raise RuntimeError(f"Final selected images are missing: {missing[:10]}")
    stats["downloads"] = download_audit

    for split, file_info in stats["source"]["files"].items():
        parquet_path = PROJECT_ROOT / file_info["path"]
        schema = pq.ParquetFile(parquet_path).schema_arrow
        file_info["schema_fields"] = list(schema.names)
        file_info["arrow_schema"] = str(schema)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "finalized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "visual_review_status": args.visual_status,
        "train_records": len(final_rows["train"]),
        "val_records": len(final_rows["val"]),
        "train_unique_images": len({r["image"] for r in final_rows["train"]}),
        "val_unique_images": len({r["image"] for r in final_rows["val"]}),
        "download_audit": download_audit,
        "review": review,
    }
    (QA_ROOT / "finalization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.visual_status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
