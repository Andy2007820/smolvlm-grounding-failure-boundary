#!/usr/bin/env python3
"""CPU-only tests for the deterministic synthetic grounding generator."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_synthetic as ms


class SyntheticGeneratorTests(unittest.TestCase):
    def candidate(self, split: str, index: int) -> dict:
        for attempt in range(1, 201):
            try:
                return ms.generate_candidate(42, split, index, attempt)
            except ValueError:
                continue
        self.fail(f"no valid candidate for {split}/{index}")

    def test_candidate_is_deterministic_and_split_seeded(self) -> None:
        first = self.candidate("train", 7)
        again = self.candidate("train", 7)
        self.assertEqual(first, again)
        other = self.candidate("eval", 7)
        self.assertNotEqual(ms.scene_signature(first), ms.scene_signature(other))
        self.assertEqual(ms.split_seed(42, "train"), 42)
        self.assertEqual(ms.split_seed(42, "eval"), 43)

    def test_visibility_accounts_for_z_order(self) -> None:
        scene = {
            "width": 64, "height": 64, "background": (255, 255, 255),
            "objects": [
                {"color": "red", "shape": "square", "box": [10, 10, 30, 30]},
                {"color": "blue", "shape": "square", "box": [20, 20, 40, 40]},
            ],
        }
        fractions = ms.visibility(scene)
        self.assertAlmostEqual(fractions[0], 0.75)
        self.assertEqual(fractions[1], 1.0)

    def test_all_shapes_stay_inside_their_boxes(self) -> None:
        for shape in ms.SHAPES:
            obj = {"color": "red", "shape": shape, "box": [17, 19, 83, 89]}
            mask = ms.object_mask(obj, 100, 110)
            ys, xs = mask.nonzero()
            self.assertGreater(len(xs), 0)
            self.assertGreaterEqual(xs.min(), 17)
            self.assertGreaterEqual(ys.min(), 19)
            self.assertLess(xs.max(), 83)
            self.assertLess(ys.max(), 89)

    def test_requested_occlusion_and_unique_combinations(self) -> None:
        for index in (0, 3, 6, 9):
            scene = self.candidate("train", index)
            target = scene["target_index"]
            self.assertTrue(scene["requested_target_occlusion"])
            self.assertGreaterEqual(scene["visible_fractions"][target], 0.55)
            self.assertLessEqual(scene["visible_fractions"][target], 0.90)
            combos = [(obj["color"], obj["shape"]) for obj in scene["objects"]]
            self.assertEqual(len(combos), len(set(combos)))

    def test_question_metadata_matches_geometry(self) -> None:
        seen = set()
        for index in range(30):
            scene = self.candidate("eval", index)
            metadata = scene["question_metadata"]
            seen.add(metadata["template"])
            target = scene["objects"][scene["target_index"]]
            tx, ty = ms.center(target)
            if metadata["template"] == "quadrant":
                self.assertEqual(metadata["vertical"], "top" if ty < scene["height"] / 2 else "bottom")
                self.assertEqual(metadata["horizontal"], "left" if tx < scene["width"] / 2 else "right")
            elif metadata["template"] in ("relation", "occlusion"):
                reference = scene["objects"][metadata["reference_index"]]
                rx, ry = ms.center(reference)
                relation = metadata["relation"]
                if relation == "to the left of":
                    self.assertLess(tx, rx)
                elif relation == "to the right of":
                    self.assertGreater(tx, rx)
                elif relation == "above":
                    self.assertLess(ty, ry)
                elif relation == "below":
                    self.assertGreater(ty, ry)
                elif relation == "behind":
                    self.assertLess(scene["target_index"], metadata["reference_index"])
                else:
                    self.fail(f"unknown relation {relation}")
        self.assertGreaterEqual(len(seen), 3)

    def test_render_is_deterministic(self) -> None:
        scene = self.candidate("train", 11)
        self.assertEqual(ms.render_scene(scene).tobytes(), ms.render_scene(scene).tobytes())

    def test_unified_record_schema_and_normalization(self) -> None:
        root = ms.DATA_ROOT / "raw" / "synthetic_unit_output"
        root.mkdir(parents=True, exist_ok=True)
        record, manifest, counters = ms.generate_record(42, "train", 0, root, set())
        self.assertEqual(tuple(record), ms.FINAL_KEYS)
        self.assertEqual(counters["generated"], 1)
        self.assertEqual(record["source"], "synthetic")
        self.assertEqual(record["raw_bbox"], manifest["objects"][manifest["target_index"]]["box"])
        expected = [
            round(record["raw_bbox"][0] / record["width"], 8),
            round(record["raw_bbox"][1] / record["height"], 8),
            round(record["raw_bbox"][2] / record["width"], 8),
            round(record["raw_bbox"][3] / record["height"], 8),
        ]
        self.assertEqual(record["answer"], expected)
        self.assertTrue(0 <= expected[0] < expected[2] <= 1)
        self.assertTrue(0 <= expected[1] < expected[3] <= 1)
        ms.validate_scene_metadata(manifest)

    def test_cross_split_checks_detect_collision(self) -> None:
        base = {"id": "a", "image": "a.png", "scene_signature": "s", "image_sha256": "h"}
        other = {"id": "b", "image": "b.png", "scene_signature": "s2", "image_sha256": "h2"}
        clean = ms.cross_split_checks({"train": [base], "eval": [other]})
        self.assertTrue(all(value == 0 for value in clean.values()))
        collision = ms.cross_split_checks({"train": [base], "eval": [base]})
        self.assertGreater(collision["train_eval_id_intersection_count"], 0)
        self.assertGreater(collision["train_eval_scene_signature_intersection_count"], 0)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticGeneratorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    destination = ms.DATA_ROOT / "qa" / "synthetic" / "unit_tests.json"
    ms.write_json(destination, {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": "passed" if result.wasSuccessful() else "failed",
        "test_file": "src/data/test_make_synthetic.py",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
