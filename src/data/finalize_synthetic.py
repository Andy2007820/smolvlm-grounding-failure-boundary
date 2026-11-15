#!/usr/bin/env python3
"""Record truthful visual review results without changing data or image hashes."""

from __future__ import annotations

import argparse
import json

import make_synthetic as ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-status", required=True, choices=("passed", "issues"))
    parser.add_argument("--review-notes", required=True)
    args = parser.parse_args()
    stats_path = ms.DATA_ROOT / "synthetic_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    qa_path = ms.DATA_ROOT / "qa" / "synthetic" / "sample_manifest.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    review = {
        "status": args.visual_status,
        "reviewed_at": ms.utc_now(),
        "reviewer": "data-engineer; vision-assisted inspection through the configured fallback vision model (not a human review)",
        "notes": args.review_notes,
        "sample_count": len(qa["samples"]),
        "contact_sheets": [
            "data/qa/synthetic/train_contact_sheet.png",
            "data/qa/synthetic/eval_contact_sheet.png",
        ],
        "review_type": "vision-assisted overlay review; contact sheets remain available for independent human inspection",
    }
    stats["qa"]["visual_review"] = args.visual_status
    stats["qa"]["review"] = review
    qa["visual_review"] = args.visual_status
    qa["review"] = review
    ms.write_json(stats_path, stats)
    ms.write_json(qa_path, qa)
    shared_path = ms.DATA_ROOT / "stats.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    shared["synthetic"] = stats
    ms.write_json(shared_path, shared)
    finalization = {
        "finalized_at": ms.utc_now(),
        "visual_review_status": args.visual_status,
        "train_records": stats["splits"]["train"]["records"],
        "eval_records": stats["splits"]["eval"]["records"],
        "outputs": stats["outputs"],
        "quality_checks": stats["quality_checks"],
        "review": review,
    }
    ms.write_json(ms.DATA_ROOT / "qa" / "synthetic" / "finalization.json", finalization)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
