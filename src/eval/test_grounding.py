"""CPU-only regression tests for the independent grounding evaluator."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

from src.eval.dataset import load_dataset, read_jsonl
from src.eval.grounding import (
    evaluate_outputs,
    iou,
    metrics_from_scores,
    parse_box,
    parse_box_with_error,
    score_prediction,
    validate_box,
)
from src.eval.prompts import PROMPT_ID, PROMPT_TEMPLATE, build_messages, render_prompt
from src.eval.run_zeroshot import compute_image_seq_len


class ParseBoxTests(unittest.TestCase):
    def assertBoxAlmostEqual(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), 4)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_normalized_bracket_and_prose(self):
        self.assertBoxAlmostEqual(parse_box("The box is [0.31, 0.42, 0.55, 0.68]."), [0.31, 0.42, 0.55, 0.68])
        self.assertBoxAlmostEqual(parse_box("1 object: [0.1, 0.2, 0.3, 0.4]"), [0.1, 0.2, 0.3, 0.4])

    def test_plain_and_labelled_coordinates(self):
        self.assertBoxAlmostEqual(parse_box("0.31, 0.42, 0.55, 0.68"), [0.31, 0.42, 0.55, 0.68])
        self.assertBoxAlmostEqual(parse_box("x1=0.1 y1=0.2 x2=0.3 y2=0.4"), [0.1, 0.2, 0.3, 0.4])

    def test_quantized_1000_and_loc_style(self):
        self.assertBoxAlmostEqual(parse_box("[0, 100, 500, 1000]"), [0.0, 0.1, 0.5, 1.0])
        self.assertBoxAlmostEqual(parse_box("<loc_0><loc_100><loc_500><loc_1000>"), [0.0, 0.1, 0.5, 1.0])

    def test_scientific_notation(self):
        self.assertBoxAlmostEqual(parse_box("[1e-1, 2e-1, 5e-1, 1e0]"), [0.1, 0.2, 0.5, 1.0])

    def test_invalid_outputs_are_not_repaired(self):
        invalid = [
            "",
            None,
            "[0.1, 0.2, 0.3]",
            "[0.1, 0.2, 0.3, 0.4, 0.5]",
            "[-0.1, 0.2, 0.3, 0.4]",
            "[0.1, 0.2, 0.3, 1001]",
            "[0.1, 0.2, 500, 600]",
            "[0.1, 0.2, 0.3, 1.5]",
            "[0.5, 0.2, 0.4, 0.8]",
            "[0.1, 0.8, 0.4, 0.2]",
            "[0.1, 0.2, 0.1, 0.8]",
            "[0.1, 0.2, 0.4, 0.2]",
            "[NaN, 0.2, 0.4, 0.8]",
            "[inf, 0.2, 0.4, 0.8]",
            "[0.1,0.2,0.3,0.4] [0.2,0.3,0.4,0.5]",
            "there are 2 objects at 0.1, 0.2, 0.3, 0.4",
        ]
        for text in invalid:
            with self.subTest(text=text):
                box, error = parse_box_with_error(text)
                self.assertIsNone(box)
                self.assertIsInstance(error, str)


class IoUTests(unittest.TestCase):
    def test_identical_disjoint_contained_and_touching(self):
        a = [0.1, 0.1, 0.4, 0.4]
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, [0.5, 0.5, 0.8, 0.8]), 0.0)
        self.assertEqual(iou(a, [0.4, 0.1, 0.7, 0.4]), 0.0)
        self.assertAlmostEqual(iou([0, 0, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]), 0.04)

    def test_partial_overlap_and_pixel_coordinates(self):
        self.assertAlmostEqual(iou([0, 0, 0.6, 0.6], [0.4, 0, 1.0, 0.6]), 0.2)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1.0 / 3.0)

    def test_invalid_boxes_raise(self):
        for bad in ([0, 0, 0, 1], [0, 0, 1], [1, 0, 0, 1], [0, 0, math.inf, 1]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    iou(bad, [0, 0, 1, 1])


class MetricTests(unittest.TestCase):
    def test_invalid_stays_in_every_denominator(self):
        records = [
            {"answer": [0.1, 0.1, 0.4, 0.4]},
            {"answer": [0.1, 0.1, 0.4, 0.4]},
            {"answer": [0.1, 0.1, 0.4, 0.4]},
            {"answer": [0.0, 0.0, 0.75, 1.0]},
        ]
        outputs = [
            "[0.1,0.1,0.4,0.4]",
            "[0.5,0.5,0.8,0.8]",
            "not a box",
            "[0.25,0.0,1.0,1.0]",  # Binary-exact IoU 0.5: threshold is inclusive.
        ]
        metrics, scores = evaluate_outputs(records, outputs)
        self.assertEqual(metrics["invalid"], 1)
        self.assertAlmostEqual(metrics["parse_rate"], 0.75)
        self.assertAlmostEqual(metrics["acc@0.5"], 0.5)
        self.assertAlmostEqual(metrics["mIoU"], 0.375)
        self.assertEqual(scores[2].iou, 0.0)
        self.assertTrue(scores[3].correct_at_0_5)

    def test_all_invalid_and_empty(self):
        metrics, _ = evaluate_outputs([{"answer": [0, 0, 1, 1]}], ["none"])
        self.assertEqual(metrics, {"acc@0.5": 0.0, "mIoU": 0.0, "parse_rate": 0.0, "invalid": 1})
        with self.assertRaises(ValueError):
            metrics_from_scores([])
        with self.assertRaises(ValueError):
            evaluate_outputs([{"answer": [0, 0, 1, 1]}], [])

    def test_invalid_ground_truth_is_data_error(self):
        with self.assertRaises(ValueError):
            score_prediction("[0,0,1,1]", [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            validate_box([0, 0, 2, 1], normalized=True)

    def test_one_shot_score_iterator(self):
        scores = evaluate_outputs(
            [{"answer": [0, 0, 1, 1]}, {"answer": [0, 0, 1, 1]}],
            ["invalid", "[0,0,1,1]"],
        )[1]
        metrics = metrics_from_scores(iter(scores))
        self.assertEqual(metrics["invalid"], 1)
        self.assertAlmostEqual(metrics["parse_rate"], 0.5)
        self.assertAlmostEqual(metrics["acc@0.5"], 0.5)
        self.assertAlmostEqual(metrics["mIoU"], 0.5)


class DatasetTests(unittest.TestCase):
    def test_gt_abort_has_dataset_split_id_and_reason(self):
        project_root = Path(__file__).resolve().parents[2]
        data_path = project_root / "data" / "synthetic_eval.jsonl"
        record = read_jsonl(data_path)[0]
        record["answer"] = [0.0, 0.0, 0.0, 1.0]
        with mock.patch("src.eval.dataset.read_jsonl", return_value=[record]):
            with self.assertRaises(ValueError) as caught:
                load_dataset(
                    data_path,
                    project_root=project_root,
                    expected_source="synthetic",
                    split="eval",
                    check_images=False,
                )
        message = str(caught.exception)
        self.assertIn("dataset=synthetic", message)
        self.assertIn("split=eval", message)
        self.assertIn(record["id"], message)
        self.assertIn("reason=", message)


class ProcessorConfigTests(unittest.TestCase):
    def test_image_sequence_length_matches_square_pixels(self):
        self.assertEqual(compute_image_seq_len(384, 16, 4), 36)
        self.assertEqual(compute_image_seq_len(512, 16, 4), 64)
        with self.assertRaises(ValueError):
            compute_image_seq_len(385, 16, 4)


class PromptTests(unittest.TestCase):
    def test_p1_template_and_single_image_item(self):
        self.assertEqual(PROMPT_ID, "p1")
        self.assertIn("Description: the red cup", render_prompt("the red cup"))
        messages = build_messages("the red cup")
        content = messages[0]["content"]
        self.assertEqual(sum(item["type"] == "image" for item in content), 1)
        self.assertNotIn("<image>", PROMPT_TEMPLATE)
        with self.assertRaises(ValueError):
            render_prompt(" ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        report = {
            "run_id": "eval_unit_tests_grounding_v1",
            "timestamp": dt.datetime.now().astimezone().isoformat(),
            "status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "failure_details": [text for _, text in result.failures],
            "error_details": [text for _, text in result.errors],
            "python": sys.version.split()[0],
            "gpu_used": False,
            "source_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in {
                    "grounding.py": Path(__file__).with_name("grounding.py"),
                    "dataset.py": Path(__file__).with_name("dataset.py"),
                    "prompts.py": Path(__file__).with_name("prompts.py"),
                    "run_zeroshot.py": Path(__file__).with_name("run_zeroshot.py"),
                    "test_grounding.py": Path(__file__),
                }.items()
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
