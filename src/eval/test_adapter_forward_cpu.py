"""CPU-only real forward regression for read-only PEFT adapter evaluation.

Loads the base SmolVLM-256M-Instruct on CPU, wraps it with
``PeftModel.from_pretrained(..., is_trainable=False)`` and runs one greedy
one-token generation. No CUDA is initialised and no weights are written.
"""

from __future__ import annotations

import argparse
import json
import importlib.metadata
import os
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CPU-only regression: hide the GPU before torch/peft are imported so that no
# CUDA context or GPU memory is created. This test must never touch the device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PIL import Image

from src.eval.dataset import read_jsonl
from src.eval.prompts import build_messages
from src.eval.run_zeroshot import compute_image_seq_len, load_adapter_metadata, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "sft_v1" / "adapter"
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
OBSERVATIONS: list[dict[str, Any]] = []


class AdapterForwardCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cls.torch = torch
        cls.metadata = load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH)
        cls.records = {row["id"]: row for row in read_jsonl(DATA_PATH)}
        cls.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        cls.processor.image_processor.do_image_splitting = False
        cls.processor.image_processor.size = {"longest_edge": 384}
        cls.processor.image_processor.max_image_size = {"longest_edge": 384}
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            dtype=torch.float32,
            local_files_only=True,
        ).eval()
        cls.base_model_class = type(base_model).__name__
        base_config = base_model.config
        cls.patch_size = int(base_config.vision_config.patch_size)
        cls.scale_factor = int(base_config.scale_factor)
        cls.image_token_id = int(base_config.image_token_id)
        cls.image_seq_len = compute_image_seq_len(384, cls.patch_size, cls.scale_factor)
        cls.processor.image_seq_len = cls.image_seq_len
        cls.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            is_trainable=False,
            torch_device="cpu",
        )
        cls.model.eval()
        cls.cuda_initialized = bool(torch.cuda.is_initialized())
        cls.cuda_memory_allocated_bytes = int(torch.cuda.memory_allocated()) if cls.cuda_initialized else 0
        cls.model_devices = sorted({str(parameter.device) for parameter in cls.model.parameters()})
        cls.lora_module_count = sum(
            1 for _, module in cls.model.named_modules() if hasattr(module, "lora_A")
        )
        cls.trainable_parameters = [
            name for name, parameter in cls.model.named_parameters() if parameter.requires_grad
        ]

    def test_peft_adapter_is_read_only_and_frozen(self) -> None:
        from peft import PeftModel

        self.assertIsInstance(self.model, PeftModel)
        self.assertFalse(self.model.training)
        self.assertGreater(self.lora_module_count, 0)
        self.assertEqual(self.trainable_parameters, [])
        metadata = load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH)
        self.assertEqual(metadata["adapter_config_sha256"], self.metadata["adapter_config_sha256"])
        self.assertEqual(metadata["adapter_model_sha256"], self.metadata["adapter_model_sha256"])
        self.assertEqual(metadata["is_trainable"], False)
        self.assertFalse(self.cuda_initialized)
        self.assertEqual(self.cuda_memory_allocated_bytes, 0)
        self.assertEqual(self.model_devices, ["cpu"])

    def test_one_token_generation_uses_adapter(self) -> None:
        torch = self.torch
        row = self.records["synthetic_eval_000001"]
        image_path = (PROJECT_ROOT / row["image"]).resolve()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            original_size = list(image.size)
        prompt = self.processor.apply_chat_template(
            build_messages(row["question"]), add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        placeholders = int((inputs["input_ids"] == self.image_token_id).sum())
        self.assertEqual(self.image_seq_len, 36)
        self.assertEqual(placeholders, 36)
        self.assertEqual(tuple(inputs["pixel_values"].shape), (1, 1, 3, 384, 384))
        self.assertTrue(all(value.device.type == "cpu" for value in inputs.values()))
        self.assertFalse(self.cuda_initialized)
        self.assertEqual(self.model_devices, ["cpu"])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
            )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        self.assertEqual(tuple(generated.shape), (1, prompt_tokens + 1))
        OBSERVATIONS.append(
            {
                "id": "synthetic_eval_000001",
                "original_size_wh": original_size,
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "input_tokens": prompt_tokens,
                "image_placeholders": placeholders,
                "expected_visual_tokens": self.image_seq_len,
                "generate_passed": True,
                "generated_token_id": int(generated[0, prompt_tokens]),
                "device": "cpu",
                "lora_module_count": self.lora_module_count,
                "trainable_parameter_count": len(self.trainable_parameters),
                "cuda_initialized": self.cuda_initialized,
                "cuda_memory_allocated_bytes": self.cuda_memory_allocated_bytes,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval_adapter_cpu_forward.json",
    )
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdapterForwardCpuTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - started

    import torch

    report = {
        "run_id": "eval_adapter_cpu_forward_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "failure_details": [str(item) for item in result.failures + result.errors],
        "gpu_used": False,
        "device": "cpu",
        "dtype": "float32",
        "lora_module_count": AdapterForwardCpuTests.lora_module_count,
        "trainable_parameter_count": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_initialized() else 0,
        "model_devices": sorted(
            {str(parameter.device) for parameter in AdapterForwardCpuTests.model.parameters()}
        ),
        "cuda_visibility": "CUDA_VISIBLE_DEVICES set to empty before importing torch/peft",
        "adapter": load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH),
        "observations": OBSERVATIONS,
        "elapsed_seconds": elapsed,
        "model_sha256": sha256_file(MODEL_PATH / "model.safetensors"),
        "dataset_sha256": sha256_file(DATA_PATH),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "run_zeroshot.py",
                "grounding.py",
                "dataset.py",
                "prompts.py",
                "test_adapter_forward_cpu.py",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "tests_run", "failures", "errors", "gpu_used", "elapsed_seconds")
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
