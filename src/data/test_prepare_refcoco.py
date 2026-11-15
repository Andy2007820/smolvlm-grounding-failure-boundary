#!/usr/bin/env python3
"""Small CPU-only unit tests for the RefCOCO preparation invariants."""

from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_refcoco import build_candidates, coco_url, final_row, parse_candidate


def synthetic_row(ref_id: int = 7, sent_id: int = 99) -> dict:
    return {
        "raw_anns": json.dumps({
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "iscrowd": 0,
        }),
        "raw_image_info": json.dumps({
            "file_name": "COCO_train2014_000000000001.jpg",
            "id": 1,
            "width": 100,
            "height": 200,
        }),
        "sentences": [
            {"raw": "  A test object  ", "sent": "a test object", "sent_id": sent_id},
            {"raw": "another description", "sent": "another description", "sent_id": sent_id + 1},
        ],
        "bbox": [10.0, 20.0, 40.0, 60.0],
        "ref_id": ref_id,
        "ann_id": 1000 + ref_id,
        "image_id": 1,
        "split": "train",
    }


class RefCOCOPreparerTests(unittest.TestCase):
    def test_coco_url_uses_real_file_name(self) -> None:
        self.assertEqual(
            coco_url("COCO_train2014_000000000001.jpg"),
            "http://images.cocodataset.org/train2014/COCO_train2014_000000000001.jpg",
        )
        self.assertEqual(
            coco_url("COCO_val2014_000000000002.jpg"),
            "http://images.cocodataset.org/val2014/COCO_val2014_000000000002.jpg",
        )
        with self.assertRaises(ValueError):
            coco_url("../COCO_train2014_000000000001.jpg")

    def test_xywh_to_normalized_xyxy_and_raw_question(self) -> None:
        candidate = parse_candidate(synthetic_row(), "train", 42, Counter())
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["raw_bbox"], [10.0, 20.0, 40.0, 60.0])
        self.assertIn(candidate["question"], {"A test object", "another description"})
        self.assertEqual(candidate["width"], 100)
        self.assertEqual(candidate["height"], 200)
        if candidate["question"] == "A test object":
            self.assertEqual(candidate["answer"], [0.1, 0.1, 0.4, 0.3])
        self.assertEqual(
            list(final_row(candidate)),
            ["id", "image", "question", "answer", "raw_bbox", "width", "height", "source"],
        )

    def test_clipping_is_recorded_and_answer_stays_in_unit_range(self) -> None:
        row = synthetic_row()
        row["raw_anns"] = json.dumps({"bbox": [-5.0, -10.0, 120.0, 230.0], "iscrowd": 0})
        row["bbox"] = [0.0, 0.0, 100.0, 200.0]
        candidate = parse_candidate(row, "train", 42, Counter())
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["raw_bbox"], [-5.0, -10.0, 115.0, 220.0])
        self.assertEqual(candidate["answer"], [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(candidate["_meta"]["bbox_was_clipped"])

    def test_selection_is_deterministic_under_row_shuffle(self) -> None:
        rows = []
        for ref_id in range(1, 21):
            row = synthetic_row(ref_id=ref_id, sent_id=ref_id * 10)
            row["raw_image_info"] = json.dumps({
                "file_name": f"COCO_train2014_{ref_id:012d}.jpg",
                "id": ref_id,
                "width": 100,
                "height": 200,
            })
            row["image_id"] = ref_id
            rows.append(row)
        first, invalid_first, _ = build_candidates(rows, "train", 42)
        random.Random(123).shuffle(rows)
        second, invalid_second, _ = build_candidates(rows, "train", 42)
        self.assertEqual(invalid_first, invalid_second)
        self.assertEqual([c["id"] for c in first], [c["id"] for c in second])
        self.assertEqual(
            [c["_meta"]["selection_key"] for c in first],
            sorted(c["_meta"]["selection_key"] for c in first),
        )

    def test_empty_sentences_are_rejected(self) -> None:
        row = synthetic_row()
        row["sentences"] = []
        invalid = Counter()
        candidate = parse_candidate(row, "train", 42, invalid)
        self.assertIsNone(candidate)
        self.assertEqual(invalid["empty_sentences"], 1)


if __name__ == "__main__":
    unittest.main()
