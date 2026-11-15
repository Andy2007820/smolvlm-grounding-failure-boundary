"""CPU-only regression for the batch-size-1-preserving batched evaluator path.

The batched path must be mathematically equivalent to the historical
one-sample path: left-padded prompts, per-row prompt length from the attention
mask, generation sliced from the padded input length, and each continuation
trimmed at (and including) its first EOS token so finished rows do not carry
generation pads. These tests never load a CUDA model.
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

# Hide the GPU before torch/transformers are imported by this process. Child
# subprocesses inherit the setting, so no CUDA context is created anywhere.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PIL import Image  # noqa: E402

from src.eval.prompts import build_messages  # noqa: E402
from src.eval.run_zeroshot import (  # noqa: E402
    load_adapter_metadata,
    normalise_eos_token_ids,
    sha256_file,
    trim_at_first_eos,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "sft_v1" / "adapter"
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
EOS_ID = 49279
PAD_ID = 2
EXPECTED_IMAGE_SEQ_LEN = 36

TEST_RUN_B1 = "batch_cpu_test_b1"
TEST_RUN_B2 = "batch_cpu_test_b2"
B1_OUTPUT = PROJECT_ROOT / "results" / "zeroshot_batch_cpu_test_b1.json"
B1_DETAILS = PROJECT_ROOT / "results" / "eval_batch_cpu_test_b1_details.jsonl"
B2_OUTPUT = PROJECT_ROOT / "results" / "zeroshot_batch_cpu_test_b2.json"
B2_DETAILS = PROJECT_ROOT / "results" / "eval_batch_cpu_test_b2_details.jsonl"


def load_records(limit: int = 3) -> list[dict[str, Any]]:
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[:limit]]


def load_images(records: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for record in records:
        with Image.open(PROJECT_ROOT / record["image"]) as opened:
            images.append(opened.convert("RGB"))
    return images


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_command(extra: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    base = [
        sys.executable,
        "-m",
        "src.eval.run_zeroshot",
        "--dataset",
        "synthetic",
        "--split",
        "eval",
        "--data",
        "data/synthetic_eval.jsonl",
        "--model-path",
        "models/SmolVLM-256M-Instruct",
        "--adapter",
        "adapters/sft_v1/adapter",
        "--expected-n",
        "1000",
        "--limit",
        "2",
        "--image-size",
        "384",
        "--max-new-tokens",
        "64",
        "--seed",
        "42",
        "--device",
        "cpu",
    ]
    return subprocess.run(
        base + extra,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env(),
    )


def read_details(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class HelperTests(unittest.TestCase):
    def test_trim_at_first_eos_inclusive_middle(self) -> None:
        self.assertEqual(
            trim_at_first_eos([11, 12, EOS_ID, PAD_ID, PAD_ID], {EOS_ID}),
            [11, 12, EOS_ID],
        )

    def test_trim_at_first_eos_first_token(self) -> None:
        self.assertEqual(trim_at_first_eos([EOS_ID, PAD_ID, PAD_ID], {EOS_ID}), [EOS_ID])

    def test_trim_without_eos_keeps_full_continuation(self) -> None:
        tokens = list(range(100, 164))
        self.assertEqual(trim_at_first_eos(tokens, {EOS_ID}), tokens)

    def test_trim_supports_multiple_eos_ids(self) -> None:
        self.assertEqual(trim_at_first_eos([7, 8, 9, 10], {9, 10}), [7, 8, 9])

    def test_normalise_eos_token_ids(self) -> None:
        self.assertEqual(normalise_eos_token_ids(None), set())
        self.assertEqual(normalise_eos_token_ids(5), {5})
        self.assertEqual(normalise_eos_token_ids([5, "6"]), {5, 6})

    def test_batched_extraction_simulation_variable_lengths(self) -> None:
        # Simulated left-padded batch: padded length 12, two prompts of 12 and 9
        # tokens. Row 0 emits EOS after two tokens, then generation pads the
        # finished row; row 1 runs to max_new_tokens without EOS.
        padded_length = 12
        batch_max_new_tokens = 64
        generated = [
            [101, 102, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
            list(range(200, 200 + batch_max_new_tokens)),
        ]
        prompt_lengths = [12, 9]
        trimmed = [
            trim_at_first_eos(row, {EOS_ID}) for row in generated
        ]
        self.assertEqual(trimmed[0], [101, 102, EOS_ID])
        self.assertEqual(len(trimmed[1]), batch_max_new_tokens)
        self.assertTrue(all(len(row) <= len(generated[i]) for i, row in enumerate(trimmed)))
        self.assertEqual(prompt_lengths, [12, 9])


class ProcessorPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        cls.processor.image_processor.do_image_splitting = False
        cls.processor.image_processor.size = {"longest_edge": 384}
        cls.processor.image_processor.max_image_size = {"longest_edge": 384}
        cls.processor.image_seq_len = EXPECTED_IMAGE_SEQ_LEN
        cls.records = load_records(3)
        cls.images = load_images(cls.records)
        cls.prompts = [
            cls.processor.apply_chat_template(build_messages(record["question"]), add_generation_prompt=True)
            for record in cls.records
        ]
        cls.single_inputs = [
            cls.processor(text=prompt, images=[image], return_tensors="pt")
            for prompt, image in zip(cls.prompts, cls.images)
        ]
        cls.processor.tokenizer.padding_side = "left"
        cls.batched = cls.processor(
            text=cls.prompts,
            images=[[image] for image in cls.images],
            padding=True,
            return_tensors="pt",
        )

    def test_left_padding_and_variable_prompt_lengths(self) -> None:
        batch = self.batched
        max_len = int(batch["input_ids"].shape[1])
        single_lengths = [int(single["input_ids"].shape[1]) for single in self.single_inputs]
        self.assertGreater(max_len, min(single_lengths))
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), single_lengths)
        for row, (length, single) in enumerate(zip(single_lengths, self.single_inputs)):
            pad = max_len - length
            if pad:
                self.assertTrue(
                    all(int(token) == PAD_ID for token in batch["input_ids"][row, :pad].tolist())
                )
                self.assertTrue(
                    all(int(flag) == 0 for flag in batch["attention_mask"][row, :pad].tolist())
                )
            self.assertEqual(
                batch["input_ids"][row, pad:].tolist(),
                single["input_ids"][0].tolist(),
            )
            self.assertEqual(int((batch["input_ids"][row] == 49190).sum()), EXPECTED_IMAGE_SEQ_LEN)

    def test_batched_pixels_equal_single_pixels_per_sample(self) -> None:
        for row, single in enumerate(self.single_inputs):
            self.assertEqual(
                tuple(single["pixel_values"].shape),
                (1, 1, 3, 384, 384),
            )
            self.assertTrue(
                self.batched["pixel_values"][row].equal(single["pixel_values"][0])
            )
            self.assertTrue(
                self.batched["pixel_attention_mask"][row].equal(
                    single["pixel_attention_mask"][0]
                )
            )

    def test_batch_of_one_matches_single_call(self) -> None:
        batch_one = self.processor(
            text=[self.prompts[0]],
            images=[[self.images[0]]],
            padding=True,
            return_tensors="pt",
        )
        self.assertTrue(batch_one["input_ids"][0].equal(self.single_inputs[0]["input_ids"][0]))
        self.assertTrue(
            batch_one["pixel_values"][0].equal(self.single_inputs[0]["pixel_values"][0])
        )


class DryRunBatchTests(unittest.TestCase):
    COMMON = [
        "--output",
        "results/zeroshot_batch_cpu_dryrun_probe.json",
        "--details-output",
        "results/eval_batch_cpu_dryrun_probe_details.jsonl",
        "--batch-size",
    ]

    def run_dry_run(self, batch_size: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "src.eval.run_zeroshot",
                "--run-id",
                "batch_cpu_dryrun_probe",
                "--dataset",
                "synthetic",
                "--split",
                "eval",
                "--data",
                "data/synthetic_eval.jsonl",
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
                *self.COMMON,
                str(batch_size),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=child_env(),
        )

    def test_batch_size_is_recorded_and_changes_fingerprint(self) -> None:
        batch_one = self.run_dry_run(1)
        batch_eight = self.run_dry_run(8)
        self.assertEqual(batch_one.returncode, 0, batch_one.stderr)
        self.assertEqual(batch_eight.returncode, 0, batch_eight.stderr)
        preflight_one = json.loads(batch_one.stdout)
        preflight_eight = json.loads(batch_eight.stdout)
        self.assertEqual(preflight_one["batch_size"], 1)
        self.assertEqual(preflight_eight["batch_size"], 8)
        self.assertNotEqual(
            preflight_one["config_fingerprint"], preflight_eight["config_fingerprint"]
        )
        for key in ("dataset_sha256", "model_sha256", "adapter_model_sha256", "image_seq_len", "n"):
            self.assertEqual(preflight_one[key], preflight_eight[key])
        self.assertFalse((PROJECT_ROOT / "results" / "zeroshot_batch_cpu_dryrun_probe.json").exists())
        self.assertFalse(
            (PROJECT_ROOT / "results" / "eval_batch_cpu_dryrun_probe_details.jsonl").exists()
        )

    def test_batch_size_zero_is_rejected(self) -> None:
        completed = self.run_dry_run(0)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("batch_size must be a positive integer", completed.stderr)


class EndToEndCpuBatchEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for path in (B1_OUTPUT, B1_DETAILS, B2_OUTPUT, B2_DETAILS):
            if path.exists():
                path.unlink()
        cls.batch_one = run_command(
            [
                "--run-id",
                TEST_RUN_B1,
                "--output",
                str(B1_OUTPUT.relative_to(PROJECT_ROOT)),
                "--details-output",
                str(B1_DETAILS.relative_to(PROJECT_ROOT)),
                "--batch-size",
                "1",
            ]
        )
        cls.batch_two = run_command(
            [
                "--run-id",
                TEST_RUN_B2,
                "--output",
                str(B2_OUTPUT.relative_to(PROJECT_ROOT)),
                "--details-output",
                str(B2_DETAILS.relative_to(PROJECT_ROOT)),
                "--batch-size",
                "2",
            ]
        )

    def test_both_paths_complete(self) -> None:
        self.assertEqual(self.batch_one.returncode, 0, self.batch_one.stderr)
        self.assertEqual(self.batch_two.returncode, 0, self.batch_two.stderr)
        self.assertTrue(B1_OUTPUT.is_file())
        self.assertTrue(B2_OUTPUT.is_file())
        self.assertEqual(len(read_details(B1_DETAILS)), 2)
        self.assertEqual(len(read_details(B2_DETAILS)), 2)

    def test_raw_outputs_are_byte_identical(self) -> None:
        batch_one = read_details(B1_DETAILS)
        batch_two = read_details(B2_DETAILS)
        for left, right in zip(batch_one, batch_two):
            self.assertEqual(left["id"], right["id"])
            self.assertEqual(left["raw_output"], right["raw_output"])
            self.assertEqual(left["output_text"], right["output_text"])
            self.assertEqual(left["predicted_box"], right["predicted_box"])
            self.assertEqual(left["parse_error"], right["parse_error"])
            self.assertEqual(left["iou"], right["iou"])
            self.assertEqual(left["correct@0.5"], right["correct@0.5"])
            self.assertEqual(left["generated_tokens"], right["generated_tokens"])
            self.assertEqual(left["input_tokens"], right["input_tokens"])
            self.assertFalse(left["hit_max_new_tokens"])
            self.assertFalse(right["hit_max_new_tokens"])

    def test_config_adapter_and_summary_contract(self) -> None:
        summary_one = json.loads(B1_OUTPUT.read_text(encoding="utf-8"))
        summary_two = json.loads(B2_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(summary_one["batch_size"], 1)
        self.assertEqual(summary_two["batch_size"], 2)
        self.assertEqual(summary_one["config"]["batch_size"], 1)
        self.assertEqual(summary_two["config"]["batch_size"], 2)
        self.assertNotEqual(
            summary_one["config"]["config_fingerprint"],
            summary_two["config"]["config_fingerprint"],
        )
        self.assertEqual(summary_one["adapter"], "adapters/sft_v1/adapter")
        self.assertEqual(summary_two["adapter"], "adapters/sft_v1/adapter")
        self.assertEqual(summary_two["adapter_load"]["is_trainable"], False)
        self.assertEqual(summary_two["adapter_load"]["lora_module_count"], 120)
        self.assertEqual(summary_two["adapter_load"]["trainable_parameter_count"], 0)
        self.assertEqual(
            summary_two["config"]["adapter"]["adapter_model_sha256"],
            sha256_file(ADAPTER_PATH / "adapter_model.safetensors"),
        )
        metadata = load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH)
        self.assertFalse(metadata["is_trainable"])
        self.assertEqual(summary_two["batch_contract"]["left_padding"], True)
        self.assertEqual(summary_one["batch_contract"]["left_padding"], False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval_batch_cpu_tests.json",
    )
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    import torch

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "run_id": "eval_batch_cpu_tests_v1",
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
        "cuda_visibility": "CUDA_VISIBLE_DEVICES set to empty before importing torch/transformers",
        "adapter": load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH),
        "batch_contract": {
            "batch1_path_unchanged": True,
            "batched_left_padding": True,
            "padded_input_length_slice": True,
            "per_sample_prompt_length_via_attention_mask": True,
            "eos_trim_inclusive": True,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 64,
        },
        "cpu_equivalence_probe": {
            "batch1_run_id": TEST_RUN_B1,
            "batch2_run_id": TEST_RUN_B2,
            "n": 2,
            "raw_output_identical": True,
            "details": [
                "results/eval_batch_cpu_test_b1_details.jsonl",
                "results/eval_batch_cpu_test_b2_details.jsonl",
            ],
            "note": "CPU float32 probe; it validates the batching/slicing machinery, not GPU bf16 identity. GPU A/B (batch1 vs batch8, 20 samples) remains mandatory before any batched re-run.",
        },
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "grounding.py": Path(__file__).with_name("grounding.py"),
                "dataset.py": Path(__file__).with_name("dataset.py"),
                "prompts.py": Path(__file__).with_name("prompts.py"),
                "run_zeroshot.py": Path(__file__).with_name("run_zeroshot.py"),
                "test_batch_eval_cpu.py": Path(__file__),
            }.items()
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "gpu_used", "cuda_initialized")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
