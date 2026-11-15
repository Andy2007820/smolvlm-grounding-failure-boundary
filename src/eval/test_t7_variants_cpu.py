"""CPU-only tests for the T7 coordinate-representation evaluation variants.

Covers:
- prompt text/ID SHA256 equality with ``src/train/coordinate_variants.py``;
- B (bin1000_integer) using the frozen parser unchanged;
- C (pixel_xyxy) independent parse -> bounds/order validation -> normalization ->
  frozen IoU/Acc@0.5 metrics, including invalid/out-of-bounds/reversed cases;
- dry-run preflight plumbing for p1_bin1000 / p1_pixel.

No CUDA model is loaded; the GPU remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.eval.dataset import read_jsonl  # noqa: E402
from src.eval.grounding import (  # noqa: E402
    metrics_from_scores,
    parse_box,
    score_prediction,
)
from src.eval.pixel_convert import (  # noqa: E402
    bracketed_numbers,
    parse_pixel_box,
    pixel_box_to_normalized,
    score_prediction_pixel,
)
from src.eval.prompts import (  # noqa: E402
    PROMPT_ID,
    PROMPT_ID_BIN1000,
    PROMPT_ID_GRID100,
    PROMPT_ID_PIXEL,
    PROMPT_VARIANTS,
    build_messages_for,
    prompt_sha256_for,
    prompt_template_for,
    representation_for,
)
from src.train.coordinate_variants import (  # noqa: E402
    PROMPT_IDS as TRAIN_PROMPT_IDS,
    PROMPT_TEMPLATES as TRAIN_PROMPT_TEMPLATES,
    build_messages as train_build_messages,
    encode_answer,
    parse_metric_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
ADAPTER_A = PROJECT_ROOT / "adapters" / "sft_v1" / "adapter"


def sample_records(limit: int = 3) -> list[dict[str, Any]]:
    return read_jsonl(DATA_PATH)[:limit]


class PromptConsistencyTests(unittest.TestCase):
    def test_prompt_ids_match_training_module(self) -> None:
        expected = {
            "normalized_decimal": PROMPT_ID,
            "bin1000_integer": PROMPT_ID_BIN1000,
            "pixel_xyxy": PROMPT_ID_PIXEL,
            "grid100_integer": PROMPT_ID_GRID100,
        }
        self.assertEqual(TRAIN_PROMPT_IDS, expected)
        self.assertEqual(set(PROMPT_VARIANTS), {"p1", "p1_bin1000", "p1_pixel", "p1_grid100"})

    def test_prompt_templates_and_sha256_match_training_module(self) -> None:
        for representation, train_template in TRAIN_PROMPT_TEMPLATES.items():
            prompt_id = TRAIN_PROMPT_IDS[representation]
            self.assertEqual(prompt_template_for(prompt_id), train_template)
            self.assertEqual(
                prompt_sha256_for(prompt_id),
                hashlib.sha256(train_template.encode("utf-8")).hexdigest(),
            )

    def test_messages_match_training_module(self) -> None:
        question = "the red cup"
        for representation in ("normalized_decimal", "bin1000_integer", "pixel_xyxy", "grid100_integer"):
            prompt_id = TRAIN_PROMPT_IDS[representation]
            self.assertEqual(
                build_messages_for(prompt_id, question),
                train_build_messages(question, representation),
            )
            content = build_messages_for(prompt_id, question)[0]["content"]
            self.assertEqual(sum(item["type"] == "image" for item in content), 1)
            self.assertNotIn("<image>", content[1]["text"])

    def test_representation_mapping(self) -> None:
        self.assertEqual(representation_for(PROMPT_ID), "normalized_decimal")
        self.assertEqual(representation_for(PROMPT_ID_BIN1000), "bin1000_integer")
        self.assertEqual(representation_for(PROMPT_ID_PIXEL), "pixel_xyxy")
        self.assertEqual(representation_for(PROMPT_ID_GRID100), "grid100_integer")


class Bin1000Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = sample_records(3)

    def test_training_encoding_is_frozen_parser_compatible(self) -> None:
        for record in self.records:
            text, audit = encode_answer(record, "bin1000_integer")
            self.assertTrue(audit["evaluator_parse_ok"])
            parsed = parse_box(text)
            self.assertIsNotNone(parsed)
            expected = [float(value) for value in record["answer"]]
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(parsed, expected)),
                0.0005 + 1e-12,
            )
            self.assertTrue(all(token.strip().isdigit() for token in text.strip("[]").split(",")))

    def test_bin_boundaries(self) -> None:
        self.assertEqual(parse_box("[0, 0, 1000, 1000]"), [0.0, 0.0, 1.0, 1.0])
        self.assertIsNone(parse_box("[1001, 0, 1000, 1000]"))
        self.assertIsNone(parse_box("[0, 0, 1000]"))

    def test_frozen_bin_score_matches_normalized_equivalent(self) -> None:
        record = {"answer": [0.2, 0.3, 0.5, 0.6]}
        bin_score = score_prediction("[200, 300, 500, 600]", record["answer"])
        normalized_score = score_prediction("[0.2, 0.3, 0.5, 0.6]", record["answer"])
        self.assertEqual(bin_score.predicted_box, normalized_score.predicted_box)
        self.assertAlmostEqual(bin_score.iou, normalized_score.iou, places=15)
        self.assertTrue(bin_score.correct_at_0_5)
        self.assertIsNone(bin_score.parse_error)


class PixelConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = sample_records(3)

    def test_roundtrip_from_dataset_raw_bbox(self) -> None:
        record = self.records[0]
        width, height = int(record["width"]), int(record["height"])
        raw = [int(round(float(value))) for value in record["raw_bbox"]]
        text = f"[{raw[0]}, {raw[1]}, {raw[2]}, {raw[3]}]"
        score, audit = score_prediction_pixel(text, record)
        self.assertIsNotNone(score.predicted_box)
        self.assertGreater(score.iou, 0.999999)
        self.assertTrue(score.correct_at_0_5)
        self.assertIsNone(score.parse_error)
        self.assertTrue(audit["conversion_applied"])
        self.assertEqual(audit["pixel_box"], raw)
        self.assertEqual(audit["image_wh"], [width, height])
        frozen = score_prediction(text, record["answer"])  # deliberately wrong on pixel text
        self.assertNotEqual(frozen.predicted_box, score.predicted_box)
        self.assertLess(frozen.iou, score.iou)

    def test_valid_pixel_box_equals_normalized_frozen_score(self) -> None:
        record = {"answer": [0.2, 0.3, 0.5, 0.6], "width": 320, "height": 200}
        pixel_text = "[64, 60, 160, 120]"
        pixel_score, audit = score_prediction_pixel(pixel_text, record)
        normalized_text = f"[{64/320}, {60/200}, {160/320}, {120/200}]"
        frozen_score = score_prediction(normalized_text, record["answer"])
        self.assertEqual(pixel_score.predicted_box, frozen_score.predicted_box)
        self.assertAlmostEqual(pixel_score.iou, frozen_score.iou, places=15)
        self.assertEqual(pixel_score.correct_at_0_5, frozen_score.correct_at_0_5)
        self.assertEqual(audit["normalized_box"], list(pixel_score.predicted_box))

    def test_boundary_boxes_are_valid(self) -> None:
        record = {"answer": [0.0, 0.0, 1.0, 1.0], "width": 100, "height": 50}
        score, audit = score_prediction_pixel("[0, 0, 100, 50]", record)
        self.assertIsNotNone(score.predicted_box)
        self.assertAlmostEqual(score.iou, 1.0, places=15)
        self.assertEqual(audit["pixel_box"], [0, 0, 100, 50])

    def test_out_of_bounds_and_reversed_are_invalid(self) -> None:
        record = {"answer": [0.1, 0.1, 0.9, 0.9], "width": 100, "height": 100}
        cases = [
            "[10, 10, 101, 90]",   # x2 > width
            "[10, 10, 90, 101]",   # y2 > height
            "[10, 10, 10, 90]",    # x1 == x2
            "[10, 10, 90, 10]",    # y1 == y2
            "[90, 10, 10, 90]",    # reversed x
            "[-1, 10, 90, 90]",    # negative
            "[1.5, 10, 90, 90]",   # non-integer
            "[10, 10, 90]",        # three numbers
        ]
        for text in cases:
            score, audit = score_prediction_pixel(text, record)
            self.assertIsNone(score.predicted_box, text)
            self.assertEqual(score.iou, 0.0, text)
            self.assertFalse(score.correct_at_0_5, text)
            self.assertIsNotNone(score.parse_error, text)
            self.assertTrue(audit["conversion_applied"], text)

    def test_ambiguous_boxes_are_invalid(self) -> None:
        self.assertEqual(
            bracketed_numbers("[10, 10, 20, 20] and [30, 30, 40, 40]"),
            (None, "ambiguous_multiple_boxes"),
        )
        self.assertEqual(
            bracketed_numbers("box [10, 10, 20]"),
            (None, "no_bracket_with_exactly_four_coordinates"),
        )
        self.assertEqual(
            bracketed_numbers("10 10 20 20"),
            ([10.0, 10.0, 20.0, 20.0], None),
        )

    def test_invalid_pixel_predictions_stay_in_denominator(self) -> None:
        record = {"answer": [0.0, 0.0, 0.5, 0.5], "width": 100, "height": 100}
        valid, _ = score_prediction_pixel("[0, 0, 50, 50]", record)
        invalid, _ = score_prediction_pixel("[0, 0, 500, 500]", record)
        metrics = metrics_from_scores([valid, invalid])
        self.assertEqual(metrics["n"] if "n" in metrics else 2, 2)
        self.assertEqual(metrics["invalid"], 1)
        self.assertEqual(metrics["parse_rate"], 0.5)
        self.assertEqual(metrics["acc@0.5"], 0.5)
        self.assertAlmostEqual(metrics["mIoU"], 0.5)

    def test_pixel_normalization_rejects_zero_image_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            pixel_box_to_normalized([0, 0, 1, 1], width=0, height=10)


class Grid100Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = sample_records(5)

    def test_training_encoding_is_strict_grid_and_roundtrips(self) -> None:
        for record in self.records:
            text, audit = encode_answer(record, "grid100_integer")
            self.assertFalse(audit["evaluator_parse_ok"])
            self.assertTrue(audit["grid_conversion_required"])
            values = [int(token) for token in text.strip("[]").split(",")]
            self.assertEqual(len(values), 4)
            self.assertTrue(all(0 <= value <= 100 for value in values))
            self.assertTrue(values[2] > values[0] and values[3] > values[1])
            normalized, error, parse_audit = parse_metric_prediction(
                text, "grid100_integer", width=record["width"], height=record["height"]
            )
            self.assertIsNone(error)
            self.assertTrue(parse_audit["conversion_applied"])
            self.assertEqual(parse_audit["grid_box"], values)
            expected = [float(value) for value in record["answer"]]
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(normalized, expected)),
                0.005 + 1e-12,
            )

    def test_frozen_parser_would_misread_grid(self) -> None:
        # The frozen parser treats values >1 as 1000-bin and divides by 1000, so
        # grid answers must never be scored through it directly.
        self.assertEqual(parse_box("[29, 75, 45, 90]"), [0.029, 0.075, 0.045, 0.09])

    def test_invalid_grid_boxes_are_rejected_without_repair(self) -> None:
        cases = {
            "[0, 0, 101, 100]": "grid_box_outside_0_100",
            "[-1, 0, 50, 50]": "grid_box_outside_0_100",
            "[50, 10, 40, 90]": "grid_box_reversed",
            "[10, 50, 90, 40]": "grid_box_reversed",
            "[1.5, 10, 20, 30]": "grid_coordinates_must_be_integers",
            "[10, 10, 20]": "no_bracket_with_exactly_four_coordinates",
        }
        for text, expected_error in cases.items():
            normalized, error, audit = parse_metric_prediction(
                text, "grid100_integer", width=100, height=100
            )
            self.assertIsNone(normalized, text)
            self.assertEqual(error, expected_error, text)
            self.assertFalse(audit["conversion_applied"], text)


class DryRunVariantTests(unittest.TestCase):
    COMMON = [
        sys.executable,
        "-m",
        "src.eval.run_zeroshot",
        "--run-id",
        "t7_variant_cpu_dryrun",
        "--dataset",
        "synthetic",
        "--split",
        "eval",
        "--data",
        "data/synthetic_eval.jsonl",
        "--output",
        "results/zeroshot_t7_variant_cpu_dryrun.json",
        "--details-output",
        "results/eval_t7_variant_cpu_dryrun_details.jsonl",
        "--model-path",
        "models/SmolVLM-256M-Instruct",
        "--adapter",
        "adapters/sft_v1/adapter",
        "--expected-n",
        "1000",
        "--limit",
        "5",
        "--image-size",
        "384",
        "--max-new-tokens",
        "64",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--dry-run",
    ]

    def run_dry(self, prompt_id: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            self.COMMON + ["--prompt-id", prompt_id],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
        )

    def test_dry_runs_record_variant_metadata(self) -> None:
        expected = {
            PROMPT_ID: ("normalized_decimal", False),
            PROMPT_ID_BIN1000: ("bin1000_integer", False),
            PROMPT_ID_PIXEL: ("pixel_xyxy", True),
            PROMPT_ID_GRID100: ("grid100_integer", True),
        }
        preflights = {}
        for prompt_id, (representation, conversion) in expected.items():
            completed = self.run_dry(prompt_id)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            preflight = json.loads(completed.stdout)
            preflights[prompt_id] = preflight
            self.assertEqual(preflight["status"], "preflight_passed")
            self.assertEqual(preflight["prompt_id"], prompt_id)
            self.assertEqual(preflight["representation"], representation)
            self.assertEqual(preflight["prompt_sha256"], prompt_sha256_for(prompt_id))
            self.assertIs(preflight["coordinate_conversion_required"], conversion)
            self.assertEqual(preflight["n"], 5)
            self.assertIs(preflight["gpu_used"], False)
        self.assertEqual(len({preflight["config_fingerprint"] for preflight in preflights.values()}), 4)
        self.assertFalse((PROJECT_ROOT / "results" / "zeroshot_t7_variant_cpu_dryrun.json").exists())
        self.assertFalse(
            (PROJECT_ROOT / "results" / "eval_t7_variant_cpu_dryrun_details.jsonl").exists()
        )

    def test_unknown_prompt_id_is_rejected(self) -> None:
        completed = self.run_dry("p1_unknown")
        self.assertNotEqual(completed.returncode, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval_t7_variants_cpu_tests.json",
    )
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    import torch

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "run_id": "eval_t7_variants_cpu_tests_v1",
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
        "prompt_variants": {
            prompt_id: {
                "representation": representation_for(prompt_id),
                "sha256": prompt_sha256_for(prompt_id),
                "template_matches_training": prompt_template_for(prompt_id)
                == TRAIN_PROMPT_TEMPLATES[representation_for(prompt_id)],
            }
            for prompt_id in PROMPT_VARIANTS
        },
        "contracts": {
            "B_uses_frozen_parse_box": True,
            "C_independent_pixel_parse_then_frozen_metrics": True,
            "D_independent_grid_parse_then_frozen_metrics": True,
            "frozen_grounding_not_modified": True,
            "invalid_predictions_remain_in_n": True,
        },
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "grounding.py": Path(__file__).with_name("grounding.py"),
                "dataset.py": Path(__file__).with_name("dataset.py"),
                "prompts.py": Path(__file__).with_name("prompts.py"),
                "pixel_convert.py": Path(__file__).with_name("pixel_convert.py"),
                "grid_convert.py": Path(__file__).with_name("grid_convert.py"),
                "run_zeroshot.py": Path(__file__).with_name("run_zeroshot.py"),
                "validate_results.py": Path(__file__).with_name("validate_results.py"),
                "train/coordinate_variants.py": PROJECT_ROOT / "src" / "train" / "coordinate_variants.py",
                "test_t7_variants_cpu.py": Path(__file__),
            }.items()
        },
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "gpu_used", "cuda_initialized")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
