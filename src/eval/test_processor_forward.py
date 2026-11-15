"""CPU-only regression for the 384px Idefics3 visual-token contract."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.eval.dataset import read_jsonl
from src.eval.prompts import build_messages
from src.eval.run_zeroshot import compute_image_seq_len, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
OBSERVATIONS: list[dict] = []


class ProcessorForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Heavy imports are deliberately isolated from dependency-free metric tests.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cls.torch = torch
        cls.records = {row["id"]: row for row in read_jsonl(DATA_PATH)}
        cls.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        cls.processor.image_processor.do_image_splitting = False
        cls.processor.image_processor.size = {"longest_edge": 384}
        cls.processor.image_processor.max_image_size = {"longest_edge": 384}
        cls.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            dtype=torch.float32,
            local_files_only=True,
        ).eval()
        cls.patch_size = int(cls.model.config.vision_config.patch_size)
        cls.scale_factor = int(cls.model.config.scale_factor)
        cls.image_seq_len = compute_image_seq_len(384, cls.patch_size, cls.scale_factor)
        cls.processor.image_seq_len = cls.image_seq_len

    def _check_sample(self, sample_id: str, expected_shape: str) -> None:
        torch = self.torch
        row = self.records[sample_id]
        image_path = (PROJECT_ROOT / row["image"]).resolve()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            original_size = list(image.size)
        if expected_shape == "square":
            self.assertEqual(original_size[0], original_size[1])
        else:
            self.assertNotEqual(original_size[0], original_size[1])
        prompt = self.processor.apply_chat_template(
            build_messages(row["question"]), add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        placeholders = int((inputs["input_ids"] == self.model.config.image_token_id).sum())
        self.assertEqual(placeholders, 36)
        self.assertEqual(self.image_seq_len, 36)
        self.assertEqual(tuple(inputs["pixel_values"].shape), (1, 1, 3, 384, 384))
        self.assertTrue(all(value.device.type == "cpu" for value in inputs.values()))
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
            )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        self.assertEqual(tuple(generated.shape), (1, prompt_tokens + 1))
        token_id = int(generated[0, prompt_tokens])
        OBSERVATIONS.append(
            {
                "id": sample_id,
                "shape": expected_shape,
                "original_size_wh": original_size,
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "input_tokens": prompt_tokens,
                "image_placeholders": placeholders,
                "expected_visual_tokens": self.image_seq_len,
                "forward_generate_passed": True,
                "generated_token_id": token_id,
                "device": "cpu",
            }
        )

    def test_square_000001(self) -> None:
        self._check_sample("synthetic_eval_000001", "square")

    def test_rectangular_000005(self) -> None:
        self._check_sample("synthetic_eval_000005", "rectangular")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results" / "eval_cpu_forward_tests.json")
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessorForwardTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    import torch
    import transformers

    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "gpu_used": False,
        "device": "cpu",
        "dtype": "float32",
        "image_size": 384,
        "do_image_splitting": False,
        "image_seq_len": 36,
        "generation": {"max_new_tokens": 1, "do_sample": False, "num_beams": 1},
        "observations": OBSERVATIONS,
        "failure_details": [str(item) for item in result.failures + result.errors],
        "elapsed_seconds": elapsed,
        "model_sha256": sha256_file(MODEL_PATH / "model.safetensors"),
        "dataset_sha256": sha256_file(DATA_PATH),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("run_zeroshot.py", "prompts.py", "dataset.py", "test_processor_forward.py")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "gpu_used", "elapsed_seconds")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
