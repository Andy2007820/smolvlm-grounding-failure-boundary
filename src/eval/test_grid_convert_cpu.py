"""CPU-only tests for the T7-D ``p1_grid100`` text-grid conversion layer.

Covers:
- strict parse (exactly four 0-100 integers, ordered, no repair) and the
  invalid/ambiguous/non-integer/out-of-range/reversed cases;
- grid -> normalized -> frozen IoU/Acc@0.5 path, including invalid predictions
  remaining in ``n``;
- training encoding -> evaluation parser roundtrip on all four fixed datasets
  (synthetic/refcoco train+eval) within the half-grid bound ``0.005``;
- agreement between ``src/train/coordinate_variants.py`` and
  ``src/eval/grid_convert.py`` so training-time and evaluation-time
  interpretation cannot drift;
- the degeneracy probe from the handoff brief (zero collapsed boxes, min
  normalized side >= 4.9 grid cells) as a regression test.

No CUDA model is loaded; the GPU remains untouched.  The test writes a
machine-readable report under ``results/`` (default
``results/eval_t7_grid100_cpu_tests.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.eval.dataset import read_jsonl  # noqa: E402
from src.eval.grid_convert import (  # noqa: E402
    grid_to_normalized,
    parse_grid100,
    score_prediction_grid,
)
from src.eval.grounding import (  # noqa: E402
    metrics_from_scores,
    parse_box,
    score_prediction,
)
from src.train.coordinate_variants import (  # noqa: E402
    encode_answer,
    parse_metric_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "synthetic_train": PROJECT_ROOT / "data" / "synthetic_train.jsonl",
    "refcoco_train": PROJECT_ROOT / "data" / "refcoco_train.jsonl",
    "synthetic_eval": PROJECT_ROOT / "data" / "synthetic_eval.jsonl",
    "refcoco_val": PROJECT_ROOT / "data" / "refcoco_val.jsonl",
}
HALF_GRID = 0.005 + 1e-12
OBSERVATIONS: dict[str, Any] = {}


def sample_records(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    return read_jsonl(path)[:limit]


class GridParseTests(unittest.TestCase):
    def test_valid_grid_equals_normalized_frozen_score(self) -> None:
        record = {"answer": [0.2, 0.3, 0.5, 0.6], "width": 320, "height": 200}
        grid_score, audit = score_prediction_grid("[20, 30, 50, 60]", record)
        frozen_score = score_prediction("[0.2, 0.3, 0.5, 0.6]", record["answer"])
        self.assertEqual(grid_score.predicted_box, frozen_score.predicted_box)
        self.assertAlmostEqual(grid_score.iou, 1.0, places=15)
        self.assertTrue(grid_score.correct_at_0_5)
        self.assertIsNone(grid_score.parse_error)
        self.assertTrue(audit["conversion_applied"])
        self.assertEqual(audit["grid_box"], [20, 30, 50, 60])
        self.assertEqual(audit["normalized_box"], [0.2, 0.3, 0.5, 0.6])

    def test_boundary_boxes_are_valid(self) -> None:
        record = {"answer": [0.0, 0.0, 1.0, 1.0], "width": 100, "height": 50}
        score, audit = score_prediction_grid("[0, 0, 100, 100]", record)
        self.assertIsNotNone(score.predicted_box)
        self.assertAlmostEqual(score.iou, 1.0, places=15)
        self.assertEqual(audit["grid_box"], [0, 0, 100, 100])

    def test_frozen_parser_would_misread_grid(self) -> None:
        # Frozen parser treats >1 as 1000-bin and divides by 1000: direct use is wrong.
        self.assertEqual(parse_box("[20, 30, 50, 60]"), [0.02, 0.03, 0.05, 0.06])

    def test_invalid_boxes_are_rejected_without_repair(self) -> None:
        cases = {
            "[0, 0, 101, 100]": "grid_box_outside_0_100",
            "[0, 0, 100, 101]": "grid_box_outside_0_100",
            "[-1, 0, 50, 50]": "grid_box_outside_0_100",
            "[50, 10, 40, 90]": "grid_box_reversed",
            "[10, 50, 90, 40]": "grid_box_reversed",
            "[10, 10, 10, 90]": "grid_box_reversed",
            "[10, 10, 90, 10]": "grid_box_reversed",
            "[1.5, 10, 20, 30]": "grid_coordinates_must_be_integers",
            "[10, 10, 20]": "no_bracket_with_exactly_four_coordinates",
            "[]": "no_bracket_with_exactly_four_coordinates",
        }
        for text, expected_error in cases.items():
            grid, error, audit = parse_grid100(text)
            self.assertIsNone(grid, text)
            self.assertEqual(error, expected_error, text)
            self.assertTrue(audit["conversion_applied"], text)

    def test_ambiguous_boxes_are_invalid(self) -> None:
        self.assertEqual(
            parse_grid100("[10, 10, 20, 20] and [30, 30, 40, 40]")[:2],
            (None, "ambiguous_multiple_boxes"),
        )
        self.assertEqual(
            parse_grid100("box [10, 10, 20]")[:2],
            (None, "no_bracket_with_exactly_four_coordinates"),
        )
        self.assertEqual(
            parse_grid100("10 10 20 20")[:2],
            ([10, 10, 20, 20], None),
        )

    def test_invalid_grid_predictions_stay_in_denominator(self) -> None:
        record = {"answer": [0.0, 0.0, 0.5, 0.5], "width": 100, "height": 100}
        valid, _ = score_prediction_grid("[0, 0, 50, 50]", record)
        invalid, _ = score_prediction_grid("[0, 0, 500, 500]", record)
        metrics = metrics_from_scores([valid, invalid])
        self.assertEqual(metrics["invalid"], 1)
        self.assertEqual(metrics["parse_rate"], 0.5)
        self.assertEqual(metrics["acc@0.5"], 0.5)
        self.assertAlmostEqual(metrics["mIoU"], 0.5)

    def test_grid_to_normalized_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            grid_to_normalized([0, 0, 0, 10])
        with self.assertRaises(ValueError):
            grid_to_normalized([0, 0, 101, 10])
        with self.assertRaises(ValueError):
            grid_to_normalized([0, 0, 10])


class GridDatasetRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stats: dict[str, dict[str, Any]] = {}
        cls.sample: dict[str, list[dict[str, Any]]] = {}
        for name, path in DATA_FILES.items():
            rows = read_jsonl(path)
            cls.sample[name] = rows[:10]
            degenerate = 0
            max_err = 0.0
            min_side = math.inf
            for row in rows:
                x1, y1, x2, y2 = (float(value) for value in row["answer"])
                grid = [int(round(value * 100.0)) for value in (x1, y1, x2, y2)]
                if not (grid[2] > grid[0] and grid[3] > grid[1]):
                    degenerate += 1
                max_err = max(
                    max_err,
                    max(abs(value / 100.0 - original) for value, original in zip(grid, (x1, y1, x2, y2))),
                )
                min_side = min(min_side, x2 - x1, y2 - y1)
            cls.stats[name] = {
                "n": len(rows),
                "degenerate": degenerate,
                "max_abs_rounding_error": max_err,
                "min_normalized_side": min_side,
            }
        OBSERVATIONS["dataset_probe"] = cls.stats

    def test_probe_matches_handoff_brief(self) -> None:
        for name, stats in self.stats.items():
            self.assertGreater(stats["n"], 0, name)
            self.assertEqual(stats["degenerate"], 0, name)
            self.assertLessEqual(stats["max_abs_rounding_error"], HALF_GRID, name)
            self.assertGreaterEqual(stats["min_normalized_side"], 0.049 - 1e-12, name)

    def test_training_encoding_roundtrips_through_eval_parser(self) -> None:
        for name, rows in self.sample.items():
            for record in rows:
                text, audit = encode_answer(record, "grid100_integer")
                self.assertFalse(audit["evaluator_parse_ok"], name)
                grid, error, parse_audit = parse_grid100(text)
                self.assertIsNone(error, f"{name} {record['id']} {text}")
                self.assertTrue(parse_audit["conversion_applied"])
                expected = [float(value) for value in record["answer"]]
                self.assertLessEqual(
                    max(abs(g / 100.0 - original) for g, original in zip(grid, expected)),
                    HALF_GRID,
                )
                score, score_audit = score_prediction_grid(
                    text, {"answer": expected, "width": record["width"], "height": record["height"]}
                )
                self.assertIsNotNone(score.predicted_box, f"{name} {record['id']} {text}")
                self.assertEqual(score_audit["grid_box"], grid)

    def test_training_and_evaluation_parsers_agree(self) -> None:
        for name, rows in self.sample.items():
            for record in rows:
                text, _ = encode_answer(record, "grid100_integer")
                train_box, train_error, _ = parse_metric_prediction(
                    text, "grid100_integer", width=record["width"], height=record["height"]
                )
                eval_box, eval_error, _ = parse_grid100(text)
                self.assertIsNone(train_error, name)
                self.assertIsNone(eval_error, name)
                self.assertEqual(
                    [round(value, 10) for value in train_box],
                    [round(value / 100.0, 10) for value in eval_box],
                )

    def test_sample_record_scores_high_iou(self) -> None:
        # The normalized prediction is the GT rounded to a 1% grid, so the
        # representation itself caps IoU below 1 (typically 0.91-0.97 on these
        # datasets; the 1000-bin variant is 100x finer).  This is a property of
        # the representation, not a scoring bug.
        record = self.sample["synthetic_eval"][0]
        text, _ = encode_answer(record, "grid100_integer")
        score, _ = score_prediction_grid(
            text, {"answer": record["answer"], "width": record["width"], "height": record["height"]}
        )
        self.assertGreater(score.iou, 0.85)
        self.assertLessEqual(score.iou, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval_t7_grid100_cpu_tests.json",
    )
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    import torch

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "run_id": "eval_t7_grid100_cpu_tests_v1",
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "failure_details": [text for _, text in result.failures],
        "error_details": [text for _, text in result.errors],
        "gpu_used": False,
        "device": "cpu",
        "cuda_initialized": torch.cuda.is_initialized(),
        "observations": OBSERVATIONS,
        "contracts": {
            "grid100_integer_only": True,
            "ordered_and_range_validated_no_repair": True,
            "frozen_parse_box_not_used_for_grid": True,
            "invalid_predictions_remain_in_n": True,
            "train_eval_parser_agreement": True,
        },
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "grounding.py": Path(__file__).with_name("grounding.py"),
                "pixel_convert.py": Path(__file__).with_name("pixel_convert.py"),
                "grid_convert.py": Path(__file__).with_name("grid_convert.py"),
                "prompts.py": Path(__file__).with_name("prompts.py"),
                "dataset.py": Path(__file__).with_name("dataset.py"),
                "train/coordinate_variants.py": PROJECT_ROOT / "src" / "train" / "coordinate_variants.py",
                "test_grid_convert_cpu.py": Path(__file__),
            }.items()
        },
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "gpu_used", "cuda_initialized")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
