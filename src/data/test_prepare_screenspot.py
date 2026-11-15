#!/usr/bin/env python3
"""CPU-only unit tests for pinned ScreenSpot conversion helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_screenspot as ps


class ScreenSpotPreparationTests(unittest.TestCase):
    def test_normalized_xyxy_is_not_double_normalized(self) -> None:
        answer, raw, clipped = ps.convert_bbox([0.9479166666666666, 0.14444444444444443, 0.99375, 0.2074074074074074], 960, 540)
        self.assertAlmostEqual(answer[0], 0.94791667)
        self.assertAlmostEqual(answer[3], 0.20740741)
        self.assertAlmostEqual(raw[0], 910.0)
        self.assertAlmostEqual(raw[1], 78.0)
        self.assertAlmostEqual(raw[2], 954.0)
        self.assertAlmostEqual(raw[3], 112.0)
        self.assertFalse(clipped)

    def test_clipping_and_unclipped_raw_bbox(self) -> None:
        answer, raw, clipped = ps.convert_bbox([-0.1, 0.2, 1.1, 0.8], 200, 100)
        self.assertEqual(answer, [0.0, 0.2, 1.0, 0.8])
        self.assertEqual(raw, [-20.0, 20.0, 220.0, 80.0])
        self.assertTrue(clipped)

    def test_empty_or_invalid_box_is_rejected(self) -> None:
        for bbox in ([0.2, 0.2, 0.2, 0.8], [0.8, 0.2, 0.2, 0.8], [0, 0, 1], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                ps.convert_bbox(bbox, 100, 100)

    def test_safe_file_name(self) -> None:
        self.assertEqual(ps.safe_file_name("pc_ede36f9b.png"), "pc_ede36f9b.png")
        self.assertEqual(ps.safe_file_name("folder/web shot.png"), "web_shot.png")
        with self.assertRaises(ValueError):
            ps.safe_file_name("..")

    def test_exact_content_key_detects_duplicates_and_bbox_conflicts(self) -> None:
        first = ps.exact_content_key("hash", "close", [0.1, 0.2, 0.3, 0.4], "icon", "windows")
        same = ps.exact_content_key("hash", "close", [0.1, 0.2, 0.3, 0.4], "icon", "windows")
        different_box = ps.exact_content_key("hash", "close", [0.1, 0.2, 0.3, 0.5], "icon", "windows")
        different_image = ps.exact_content_key("other", "close", [0.1, 0.2, 0.3, 0.4], "icon", "windows")
        self.assertEqual(first, same)
        self.assertNotEqual(first, different_box)
        self.assertNotEqual(first, different_image)

    def test_qa_sampling_is_stratified_and_deterministic(self) -> None:
        records = [{"id": f"screenspot_test_{i:06d}"} for i in range(30)]
        sources = ["android", "ios", "macos", "web", "windows"]
        manifests = [{"data_source": sources[i % len(sources)]} for i in range(30)]
        selected = ps.select_qa_samples(records, manifests, 42, 10)
        again = ps.select_qa_samples(records, manifests, 42, 10)
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({row["id"] for row in selected}), 10)
        selected_sources = {manifests[int(row["id"].rsplit("_", 1)[1])]["data_source"] for row in selected}
        self.assertEqual(selected_sources, set(sources))

    def test_probe_confirms_source_bbox_and_image_schema(self) -> None:
        probe = json.loads((ps.RAW_ROOT / "probe_result.json").read_text(encoding="utf-8"))
        self.assertEqual(probe["revision"], ps.REVISION)
        self.assertEqual(probe["sha256"], ps.EXPECTED_FILES[probe["file"]][1])
        sample = probe["samples"][0]
        self.assertEqual(sample["image_size"], [960, 540])
        self.assertTrue(all(0 <= value <= 1 for value in sample["bbox"]))
        self.assertLess(sample["bbox"][0], sample["bbox"][2])
        self.assertLess(sample["bbox"][1], sample["bbox"][3])

    def test_expected_source_is_single_pinned_test_set(self) -> None:
        self.assertEqual(ps.REPO_ID, "bevaya/ScreenSpot")
        self.assertEqual(len(ps.REVISION), 40)
        self.assertEqual(len(ps.EXPECTED_FILES), 3)
        self.assertTrue(all(name.startswith("data/test-") for name in ps.EXPECTED_FILES))
        self.assertEqual(sum(size for size, _ in ps.EXPECTED_FILES.values()), 602316816)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScreenSpotPreparationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    ps.write_json(ps.DATA_ROOT / "qa" / "screenspot" / "unit_tests.json", {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "status": "passed" if result.wasSuccessful() else "failed",
        "test_file": "src/data/test_prepare_screenspot.py",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
