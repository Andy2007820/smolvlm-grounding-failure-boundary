"""CPU-only regression tests + preflight evidence for the T5 SFT implementation.

Run from ``final-project/``::

    python -m src.train.test_t5_cpu --report results/train_t5_cpu_tests.json

The suite never loads model weights and never touches CUDA.  It covers:

* syntax/config contract (fixed hyperparameters, reproducible fingerprint)
* shipped processor default ``image_seq_len=64`` untouched while runtime is 36
* square + rectangular + real-photo visual-token contract (views * tokens == placeholders)
* deliberate detection of the old 64-vs-36 mismatch
* label masking: only answer + explicit EOS 49279 contribute to the loss
* pad=2 / EOS=49279 audit and the 128002 tokenizer-config warning investigation
* train/validation ID disjointness (zero intersection)
* deterministic seed/data order and answer-text roundtrip through the frozen evaluator
* checkpoint metadata / resume guards and RNG state roundtrip
* CPU training guard rails (no training, no CUDA without a Lead lease)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import platform
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
CONFIG_OVERFIT = PROJECT_ROOT / "configs" / "sft_overfit100.json"
CONFIG_V1 = PROJECT_ROOT / "configs" / "sft_v1.json"
CONFIG_OVERFIT_E48 = PROJECT_ROOT / "configs" / "sft_overfit100_e48.json"
CONFIG_OVERFIT1 = PROJECT_ROOT / "configs" / "sft_overfit1_e1600.json"
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()

OBSERVATIONS: dict[str, object] = {
    "visual_token_preflight": [],
    "processor_default_audit": {},
    "pad_eos_audit": {},
    "split_isolation": {},
    "answer_roundtrip": {},
    "config_contract": {},
}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class T5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import transformers
        from transformers import AutoProcessor

        from src.train.config import load_config
        from src.train.model_audit import configure_processor
        from src.train.sft_data import (
            build_sft_example,
            load_manifest_records,
            load_train_records,
        )

        cls.torch = torch
        cls.transformers = transformers
        cls._build_sft_example = staticmethod(build_sft_example)
        cls.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        cls.default_image_seq_len_at_load = int(cls.processor.image_seq_len)
        configure_processor(cls.processor, image_size=384)
        cls.overfit_config = load_config(CONFIG_OVERFIT)
        cls.v1_config = load_config(CONFIG_V1)
        cls.e48_config = load_config(CONFIG_OVERFIT_E48)
        cls.overfit1_config = load_config(CONFIG_OVERFIT1)
        cls.train_records, cls.train_evidence = load_train_records(cls.overfit_config)
        cls.full_manifests = {
            "synthetic_train": load_manifest_records(PROJECT_ROOT / "data" / "synthetic_train.jsonl"),
            "synthetic_eval": load_manifest_records(PROJECT_ROOT / "data" / "synthetic_eval.jsonl"),
            "refcoco_train": load_manifest_records(PROJECT_ROOT / "data" / "refcoco_train.jsonl"),
            "refcoco_val": load_manifest_records(PROJECT_ROOT / "data" / "refcoco_val.jsonl"),
        }
        cls.example_cache: dict[str, object] = {}

    @classmethod
    def example(cls, record: dict) -> object:
        if record["id"] not in cls.example_cache:
            cls.example_cache[record["id"]] = cls._build_sft_example(
                record, cls.processor, model_path=MODEL_PATH
            )
        return cls.example_cache[record["id"]]

    # -- 1. syntax + config contract -------------------------------------
    def test_01_compile_and_fixed_config_contract(self) -> None:
        source_files = [
            "src/train/__init__.py",
            "src/train/config.py",
            "src/train/vision_tokens.py",
            "src/train/sft_data.py",
            "src/train/checkpoint.py",
            "src/train/accumulation.py",
            "src/train/probe_only.py",
            "src/train/model_audit.py",
            "src/train/train_sft.py",
            "src/train/test_t5_cpu.py",
        ]
        for name in source_files:
            path = PROJECT_ROOT / name
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

        from src.train.config import ConfigError, config_fingerprint, load_config, validate_config

        again = load_config(CONFIG_OVERFIT)
        self.assertEqual(again["config_fingerprint"], self.overfit_config["config_fingerprint"])
        self.assertEqual(again["_config_sha256"], self.overfit_config["_config_sha256"])
        self.assertEqual(config_fingerprint(again), self.overfit_config["config_fingerprint"])
        self.assertEqual(self.overfit_config["seed"], 42)
        self.assertEqual(self.overfit_config["lora"]["r"], 16)
        self.assertEqual(self.overfit_config["lora"]["lora_alpha"], 32)
        self.assertEqual(self.overfit_config["training"]["learning_rate"], 1e-4)
        self.assertEqual(self.overfit_config["training"]["batch_size"], 1)
        self.assertEqual(self.overfit_config["training"]["gradient_accumulation_steps"], 8)
        self.assertEqual(
            self.overfit_config["training"]["batch_size"]
            * self.overfit_config["training"]["gradient_accumulation_steps"],
            8,
        )
        self.assertEqual(self.overfit_config["image"]["image_size"], 384)
        self.assertEqual(self.overfit_config["image"]["expected_image_seq_len"], 36)
        self.assertEqual(self.overfit_config["model"]["dtype"], "bfloat16")
        self.assertTrue(self.overfit_config["model"]["freeze_vision_tower"])
        self.assertTrue(self.overfit_config["model"]["gradient_checkpointing"])
        self.assertEqual(self.overfit_config["answer"]["eos_token_id"], 49279)
        self.assertEqual(self.overfit_config["answer"]["pad_token_id"], 2)

        mutations = [
            ("seed", 41),
            ("lora.r", 8),
            ("lora.lora_alpha", 64),
            ("training.learning_rate", 2e-4),
            ("training.batch_size", 2),
            ("training.gradient_accumulation_steps", 4),
            ("training.effective_batch_size", 4),
            ("image.image_size", 512),
            ("image.expected_image_seq_len", 64),
            ("answer.eos_token_id", 2),
            ("answer.pad_token_id", 49279),
            ("model.freeze_vision_tower", False),
            ("model.gradient_checkpointing", False),
        ]
        for dotted, value in mutations:
            tampered = copy.deepcopy({key: value for key, value in self.overfit_config.items() if not key.startswith("_")})
            cursor = tampered
            parts = dotted.split(".")
            for part in parts[:-1]:
                cursor = cursor[part]
            cursor[parts[-1]] = value
            with self.assertRaises(ConfigError, msg=f"tampering {dotted} must be rejected"):
                validate_config(tampered)
        # Source-hash audit closure: the real true-accumulation implementation must
        # be part of every training/checkpoint source manifest, with a hash that
        # equals the current file on disk.
        from src.train import train_sft as training_module

        source_paths = list(training_module.SOURCE_FILE_PATHS)
        self.assertIn("src/train/accumulation.py", source_paths)
        self.assertIn("src/train/train_sft.py", source_paths)
        collected = training_module.collect_source_hashes()
        self.assertEqual(set(collected), set(source_paths))
        for name, digest in collected.items():
            self.assertEqual(digest, _sha256_file(PROJECT_ROOT / name), msg=f"stale source hash for {name}")
        self.assertIn("src/train/accumulation.py", collected)
        self.assertEqual(
            collected["src/train/accumulation.py"], _sha256_file(PROJECT_ROOT / "src/train/accumulation.py")
        )
        OBSERVATIONS["config_contract"] = {
            "config_fingerprint": self.overfit_config["config_fingerprint"],
            "config_sha256": self.overfit_config["_config_sha256"],
            "v1_config_fingerprint": self.v1_config["config_fingerprint"],
            "tamper_cases_rejected": len(mutations),
            "compile_checked_sources": source_files,
            "source_file_paths": source_paths,
            "accumulation_py_tracked": "src/train/accumulation.py" in collected,
            "accumulation_py_sha256": collected.get("src/train/accumulation.py"),
            "all_source_hashes_match_current_files": True,
        }

    # -- 2. shipped default stays 64, runtime call sets 36 ---------------
    def test_02_processor_default_image_seq_len_untouched(self) -> None:
        from src.train.vision_tokens import set_runtime_image_seq_len

        self.assertEqual(self.default_image_seq_len_at_load, 64)
        config_path = MODEL_PATH / "processor_config.json"
        before_bytes = config_path.read_bytes()
        before_hash = _sha256_file(config_path)
        on_disk = json.loads(before_bytes.decode("utf-8"))["image_seq_len"]
        self.assertEqual(on_disk, 64)

        audit_first = set_runtime_image_seq_len(self.processor, 36, processor_config_path=config_path)
        audit_second = set_runtime_image_seq_len(self.processor, 36, processor_config_path=config_path)
        after_hash = _sha256_file(config_path)
        self.assertEqual(before_hash, after_hash)
        self.assertEqual(json.loads(config_path.read_bytes().decode("utf-8"))["image_seq_len"], 64)
        self.assertTrue(audit_first["global_default_unchanged"])
        self.assertTrue(audit_second["global_default_unchanged"])
        self.assertEqual(audit_first["processor_config_default_image_seq_len"], 64)
        self.assertEqual(audit_first["runtime_image_seq_len"], 36)
        self.assertEqual(self.processor.image_seq_len, 36)
        with self.assertRaises(RuntimeError):
            set_runtime_image_seq_len(self.processor, 36, processor_config_path=config_path, expected_default=63)
        OBSERVATIONS["processor_default_audit"] = {
            "processor_config_sha256": before_hash,
            "on_disk_default_image_seq_len": 64,
            "runtime_image_seq_len": 36,
            "global_default_unchanged": True,
            "first_call_audit": audit_first,
            "second_call_audit": audit_second,
        }

    # -- 3. visual-token contract on square + rectangular + real photo ---
    def test_03_visual_token_contract(self) -> None:
        from src.eval.run_zeroshot import compute_image_seq_len as evaluator_compute_image_seq_len
        from src.train.vision_tokens import compute_image_seq_len, compute_tokens_per_view

        synthetic = self.full_manifests["synthetic_train"]
        square = next(record for record in synthetic if record["width"] == record["height"])
        rectangle = next(record for record in synthetic if record["width"] != record["height"])
        refcoco = self.full_manifests["refcoco_train"]
        real_photo = next(
            (record for record in refcoco if record["width"] == record["height"]),
            refcoco[0],
        )
        cases = [("synthetic_square", square), ("synthetic_rectangle", rectangle), ("refcoco_real", real_photo)]
        observations = []
        for label, record in cases:
            example = self.example(record)
            audit = example.audit
            contract = audit["visual_token_contract"]
            self.assertEqual(tuple(contract["pixel_values_shape"]), (1, 1, 3, 384, 384))
            self.assertEqual(contract["total_views"], 1)
            self.assertEqual(contract["view_height"], 384)
            self.assertEqual(contract["view_width"], 384)
            self.assertEqual(contract["tokens_per_view"], 36)
            self.assertEqual(contract["derived_visual_tokens"], 36)
            self.assertEqual(contract["placeholder_count"], 36)
            self.assertEqual(contract["expected_image_seq_len"], 36)
            self.assertEqual(
                contract["total_views"] * contract["tokens_per_view"], contract["placeholder_count"]
            )
            self.assertEqual(
                compute_image_seq_len(384, 16, 4), evaluator_compute_image_seq_len(384, 16, 4)
            )
            self.assertEqual(compute_image_seq_len(384, 16, 4), 36)
            self.assertEqual(compute_tokens_per_view(384, 384, 16, 4), 36)
            self.assertEqual(audit["raw_conversation_placeholder_count"], 1)
            self.assertEqual(audit["image_expansion_delta"], 38)
            self.assertEqual(
                audit["prompt_tokens"] + audit["answer_tokens"],
                audit["sequence_tokens"] - audit["tokens_after_eos"],
            )
            observation = {
                "case": label,
                "record_id": record["id"],
                "original_size_wh": [record["width"], record["height"]],
                "pixel_values_shape": contract["pixel_values_shape"],
                "views": contract["total_views"],
                "tokens_per_view": contract["tokens_per_view"],
                "derived_visual_tokens": contract["derived_visual_tokens"],
                "image_placeholders": contract["placeholder_count"],
                "expected_image_seq_len": contract["expected_image_seq_len"],
                "raw_conversation_placeholders": audit["raw_conversation_placeholder_count"],
                "raw_conversation_tokens": audit["raw_conversation_token_count"],
                "expanded_conversation_tokens": audit["sequence_tokens"],
                "expansion_delta": audit["image_expansion_delta"],
                "prompt_tokens": audit["prompt_tokens"],
                "answer_tokens": audit["answer_tokens"],
                "runtime_image_seq_len": audit["runtime_image_seq_len"],
            }
            observations.append(observation)
        OBSERVATIONS["visual_token_preflight"] = observations

    # -- 4. the old 64-vs-36 mismatch must raise -------------------------
    def test_04_visual_token_mismatch_is_detected(self) -> None:
        from PIL import Image

        from src.eval.prompts import build_messages
        from src.train.vision_tokens import inspect_visual_token_contract

        record = self.full_manifests["synthetic_train"][0]
        with Image.open(PROJECT_ROOT / record["image"]) as opened:
            image = opened.convert("RGB")
        prompt = self.processor.apply_chat_template(build_messages(record["question"]), add_generation_prompt=True)
        try:
            self.processor.image_seq_len = 64
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            placeholders = int((inputs["input_ids"] == 49190).sum())
            self.assertEqual(placeholders, 64)
            with self.assertRaises(AssertionError):
                inspect_visual_token_contract(
                    inputs, patch_size=16, scale_factor=4, image_size=384, image_token_id=49190
                )
        finally:
            self.processor.image_seq_len = 36

    # -- 5. label masking: answer + explicit EOS only --------------------
    def test_05_label_masking_and_explicit_eos(self) -> None:
        raw = self.full_manifests["synthetic_train"]
        square = next(record for record in raw if record["width"] == record["height"])
        rectangle = next(record for record in raw if record["width"] != record["height"])
        tokenizer = self.processor.tokenizer
        eos_text = tokenizer.decode([49279], skip_special_tokens=False)
        for record in (square, rectangle):
            example = self.example(record)
            audit = example.audit
            input_ids = example.input_ids.tolist()
            labels = example.labels.tolist()
            label_start = audit["prompt_tokens"]
            eos_position = audit["eos_position"]
            self.assertEqual(input_ids.count(49190), 36)
            self.assertEqual(sum(1 for token in input_ids[:label_start] if token == 49190), 36)
            self.assertTrue(all(value == -100 for value in labels[:label_start]))
            self.assertEqual(labels[label_start : eos_position + 1], input_ids[label_start : eos_position + 1])
            self.assertEqual(labels[eos_position], 49279)
            self.assertTrue(all(value == -100 for value in labels[eos_position + 1 :]))
            unmasked = [index for index, value in enumerate(labels) if value != -100]
            self.assertEqual(unmasked, list(range(label_start, eos_position + 1)))
            self.assertEqual(len(unmasked), audit["answer_tokens"])
            self.assertNotIn(49190, [labels[index] for index in unmasked])
            decoded = tokenizer.decode([labels[index] for index in unmasked], skip_special_tokens=False)
            self.assertEqual(decoded.strip(), example.answer_text + eos_text)
            self.assertEqual(audit["prompt_eos_positions"], [index for index in audit["prompt_eos_positions"]])
            self.assertEqual(len(audit["prompt_eos_positions"]), 1)
            self.assertLess(audit["prompt_eos_positions"][0], label_start)
            self.assertEqual(audit["answer_eos_positions"], [eos_position])

            # Independent check: the training sequence is the processor-expanded
            # generation prompt followed by the answer.
            from PIL import Image

            from src.eval.prompts import build_messages

            prompt = self.processor.apply_chat_template(
                build_messages(record["question"]), add_generation_prompt=True
            )
            with Image.open(PROJECT_ROOT / record["image"]) as opened:
                image = opened.convert("RGB")
            prompt_inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            prompt_ids = prompt_inputs["input_ids"][0].tolist()
            self.assertEqual(input_ids[: len(prompt_ids)], prompt_ids)
            self.assertEqual(len(prompt_ids), label_start)

    # -- 6. pad / EOS audit + 128002 warning investigation --------------
    def test_06_pad_eos_and_128002_warning_audit(self) -> None:
        from src.train.model_audit import audit_pad_eos

        audit = audit_pad_eos(self.processor, model_path=MODEL_PATH)
        self.assertEqual(audit["tokenizer_pad_token_id"], 2)
        self.assertEqual(audit["tokenizer_eos_token_id"], 49279)
        self.assertEqual(audit["tokenizer_pad_token"], "<|im_end|>")
        self.assertEqual(audit["tokenizer_eos_token"], "<end_of_utterance>")
        self.assertEqual(audit["generation_config_pad_token_id"], 2)
        self.assertEqual(audit["generation_config_eos_token_id"], 49279)
        self.assertEqual(audit["text_config_pad_token_id"], 2)
        self.assertEqual(audit["config_image_token_id"], 49190)
        self.assertEqual(audit["processor_config_default_image_seq_len"], 64)
        self.assertEqual(audit["top_level_config_default_pad_token_id"], 128002)
        self.assertTrue(audit["top_level_default_is_out_of_vocab"])
        self.assertIn("128002", audit["warning_text"])
        self.assertIn("configuration_idefics3.py", audit["warning_source"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertIn("128002", audit["conclusion"])
        OBSERVATIONS["pad_eos_audit"] = audit

    # -- 7. split isolation ----------------------------------------------
    def test_07_train_val_id_isolation(self) -> None:
        from src.train.sft_data import assert_train_val_disjoint, digest_sequence

        manifests = self.full_manifests
        audit = assert_train_val_disjoint(
            {
                "synthetic_train": manifests["synthetic_train"],
                "refcoco_train": manifests["refcoco_train"],
            },
            {
                "synthetic_eval": manifests["synthetic_eval"],
                "refcoco_val": manifests["refcoco_val"],
            },
        )
        self.assertEqual(audit["global_train_val_id_intersection_count"], 0)
        self.assertTrue(all(count == 0 for count in audit["pair_id_intersection_counts"].values()))
        self.assertEqual(len(manifests["synthetic_train"]), 5000)
        self.assertEqual(len(manifests["synthetic_eval"]), 1000)
        self.assertEqual(len(manifests["refcoco_train"]), 5000)
        self.assertEqual(len(manifests["refcoco_val"]), 1000)

        selected_ids = [record["id"] for record in self.train_records]
        self.assertEqual(len(selected_ids), 100)
        self.assertEqual(len(set(selected_ids)), 100)
        for val_name in ("synthetic_eval", "refcoco_val"):
            val_ids = {record["id"] for record in manifests[val_name]}
            self.assertEqual(set(selected_ids) & val_ids, set())
        self.assertEqual(digest_sequence(selected_ids), digest_sequence([r["id"] for r in self.train_records]))

        overlapping = [dict(manifests["synthetic_train"][0])]
        with self.assertRaises(AssertionError):
            assert_train_val_disjoint({"synthetic_train": manifests["synthetic_train"]}, {"bad_val": overlapping})
        OBSERVATIONS["split_isolation"] = {
            **audit,
            "selected_overfit_records": len(selected_ids),
            "selected_ids_sha256": digest_sequence(selected_ids),
            "selected_vs_val_intersections": {
                val_name: len(set(selected_ids) & {r["id"] for r in manifests[val_name]})
                for val_name in ("synthetic_eval", "refcoco_val")
            },
            "negative_overlap_case_raised": True,
        }

    # -- 8. reproducible seed / data order -------------------------------
    def test_08_reproducible_seed_and_order(self) -> None:
        import random

        import numpy as np

        from src.train.sft_data import digest_sequence, epoch_order
        from src.train.train_sft import build_train_order_digest, seed_everything

        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        seed_everything(42)
        first = (random.random(), float(np.random.rand()), self.torch.rand(3).tolist())
        seed_everything(42)
        second = (random.random(), float(np.random.rand()), self.torch.rand(3).tolist())
        self.assertEqual(first, second)

        self.assertEqual(epoch_order(100, 42, 0), epoch_order(100, 42, 0))
        self.assertNotEqual(epoch_order(100, 42, 0), epoch_order(100, 42, 1))
        self.assertNotEqual(epoch_order(100, 42, 0), epoch_order(100, 43, 0))
        order = epoch_order(100, 42, 0)
        self.assertEqual(sorted(order), list(range(100)))

        digest_one = build_train_order_digest(self.train_records, self.overfit_config)
        digest_two = build_train_order_digest(self.train_records, self.overfit_config)
        self.assertEqual(digest_one, digest_two)
        self.assertEqual(
            digest_sequence([r["id"] for r in self.train_records]),
            digest_sequence([r["id"] for r in self.train_records]),
        )
        from src.train.train_sft import compute_plan

        plan = compute_plan(self.overfit_config, self.train_records)
        self.assertEqual(plan["total_micro_batches"], 1600)
        self.assertEqual(plan["total_optimizer_steps"], 200)
        self.assertEqual(plan["checkpoint_steps"], [100, 200])
        from src.train.train_sft import expected_lora_target_modules, lora_targets_match

        expected_targets = expected_lora_target_modules(30)
        self.assertEqual(len(expected_targets), 120)
        self.assertTrue(lora_targets_match(list(reversed(expected_targets)), expected_targets))
        self.assertTrue(lora_targets_match(sorted(expected_targets), expected_targets))
        self.assertFalse(lora_targets_match(expected_targets[:-1], expected_targets))
        self.assertNotEqual(list(reversed(expected_targets)), expected_targets)

    # -- 9. answer text roundtrip through the frozen evaluator -----------
    def test_09_answer_format_roundtrip(self) -> None:
        from src.eval.grounding import parse_box
        from src.train.sft_data import format_answer

        worst = 0.0
        precisions: dict[int, int] = {}
        for record in self.train_records:
            text, meta = format_answer(record["answer"], precision=4)
            parsed = parse_box(text)
            self.assertIsNotNone(parsed, msg=f"unparseable answer text: {text}")
            worst = max(worst, meta["max_abs_rounding_error"])
            self.assertLessEqual(meta["max_abs_rounding_error"], 0.5 * 10 ** -meta["precision_used"] + 1e-12)
            self.assertEqual(text.count("["), 1)
            self.assertEqual(text.count("]"), 1)
            precisions[meta["precision_used"]] = precisions.get(meta["precision_used"], 0) + 1
            for value in parsed:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        # A rounding-degenerate box must fall back to finer precision.
        degenerate_text, degenerate_meta = format_answer([0.99996, 0.1, 1.0, 0.5], precision=4)
        self.assertIsNotNone(parse_box(degenerate_text))
        self.assertGreaterEqual(degenerate_meta["precision_used"], 6)
        OBSERVATIONS["answer_roundtrip"] = {
            "records_checked": len(self.train_records),
            "worst_abs_rounding_error": worst,
            "precision_usage": precisions,
            "degenerate_fallback_example": degenerate_meta,
            "evaluator_parser": "src/eval/grounding.py:parse_box (read-only import)",
        }

    # -- 10. CPU/CUDA guard rails + plan --------------------------------
    def test_10_cli_guards_and_plan(self) -> None:
        from src.train import train_sft

        with self.assertRaises(SystemExit):
            train_sft.main(["--config", str(CONFIG_OVERFIT), "--device", "cuda"])
        with self.assertRaises(SystemExit):
            train_sft.main(["--config", str(CONFIG_OVERFIT), "--device", "cpu"])
        with self.assertRaises(SystemExit):
            train_sft.main(["--config", str(CONFIG_OVERFIT), "--device", "cuda", "--dry-run"])
        train_sft.assert_gpu_lease_allowed(device="cuda", lease_id="GPU-LEASE-T5-20261109-01")
        for bad_lease in ("", "lease-123", "GPU-LEASE-", "GPU-LEASE-x y"):
            with self.assertRaises(SystemExit):
                train_sft.assert_gpu_lease_allowed(device="cuda", lease_id=bad_lease)
        plan = train_sft.compute_plan(self.overfit_config, self.train_records)
        self.assertEqual(plan["effective_batch_size"], 8)
        self.assertEqual(plan["total_optimizer_steps"], 200)
        self.assertEqual(plan["checkpoint_steps"], [100, 200])
        capped = train_sft.compute_plan(self.overfit_config, self.train_records, max_steps=25)
        self.assertEqual(capped["total_optimizer_steps"], 25)

    # -- 11. checkpoint / resume metadata --------------------------------
    def test_11_checkpoint_metadata_and_resume_guards(self) -> None:
        import random

        from src.train import checkpoint as checkpoint_utils

        metadata = checkpoint_utils.build_checkpoint_metadata(
            run_id="unit_test_run",
            step=100,
            epoch=7,
            samples_seen=800,
            config={"fake": True},
            config_fingerprint="fp-abc",
            seed=42,
            model_path="models/SmolVLM-256M-Instruct",
            model_files={"model.safetensors": {"sha256": "abc", "bytes": 1}},
            dataset_evidence=[{"path": "data/synthetic_train.jsonl", "sha256": "def", "rows_loaded": 100}],
            selected_ids_sha256="sel",
            train_order_sha256="order",
            id_intersection={"global_train_val_id_intersection_count": 0},
            lora={"target_module_count": 120, "r": 16},
            trainable_parameters=1_843_200,
            total_parameters=250_000_000,
            frozen_vision_parameters=100_000_000,
            trainable_vision_parameters=0,
            image_contract={"image_seq_len": 36},
            answer_contract={"eos_token_id": 49279, "pad_token_id": 2},
            training={"save_every_steps": 100},
            gpu={"device": "cpu"},
            environment={"python": "test"},
            source_hashes={"src/train/train_sft.py": "hash"},
        )
        for key in (
            "config_fingerprint",
            "selected_ids_sha256",
            "train_order_sha256",
            "id_intersection",
            "lora",
            "image_contract",
            "answer_contract",
            "source_hashes",
            "resume_contract",
            "checkpoint_policy",
        ):
            self.assertIn(key, metadata)
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["trainable_vision_parameters"], 0)

        # Directory-selection / resume-completeness rules are pure helpers so the
        # rule is testable without creating directories in the workspace.
        self.assertEqual(
            checkpoint_utils.sorted_checkpoint_names(
                ["other", "checkpoint-000200", "checkpoint-000100", "checkpoint-abc"]
            ),
            ["checkpoint-000100", "checkpoint-000200"],
        )
        self.assertEqual(checkpoint_utils.checkpoint_step_from_name("checkpoint-000100"), 100)
        self.assertIsNone(checkpoint_utils.checkpoint_step_from_name("checkpoint-abc"))
        self.assertEqual(
            checkpoint_utils.select_latest_complete_checkpoint(
                [("checkpoint-000100", True), ("checkpoint-000200", False)]
            ),
            "checkpoint-000100",
        )
        self.assertEqual(
            checkpoint_utils.select_latest_complete_checkpoint(
                [("checkpoint-000100", True), ("checkpoint-000200", True)]
            ),
            "checkpoint-000200",
        )
        self.assertIsNone(checkpoint_utils.select_latest_complete_checkpoint([]))

        # Atomic JSON metadata roundtrip through a flat platform-temp file (no dirs).
        handle, temporary_name = tempfile.mkstemp(prefix="t5_unit_meta_", suffix=".json")
        os.close(handle)
        temporary_path = Path(temporary_name)
        try:
            checkpoint_utils.write_json_atomic(temporary_path, {**metadata, "complete": True})
            roundtrip = checkpoint_utils.read_json(temporary_path)
            self.assertEqual(roundtrip["config_fingerprint"], "fp-abc")
            self.assertTrue(roundtrip["complete"])
        finally:
            os.remove(temporary_path)

        checkpoint_utils.validate_resume_metadata(
            {**metadata, "complete": True},
            config_fingerprint="fp-abc",
            seed=42,
            model_files={"model.safetensors": {"sha256": "abc", "bytes": 1}},
            dataset_evidence=[{"path": "data/synthetic_train.jsonl", "sha256": "def", "rows_loaded": 100}],
            image_seq_len=36,
            eos_token_id=49279,
            pad_token_id=2,
        )
        mismatch_cases = [
            {"config_fingerprint": "fp-other"},
            {"seed": 43},
            {"model_files": {"model.safetensors": {"sha256": "other", "bytes": 1}}},
            {"dataset_evidence": [{"path": "data/synthetic_train.jsonl", "sha256": "other", "rows_loaded": 100}]},
            {"image_seq_len": 64},
            {"eos_token_id": 2},
            {"pad_token_id": 49279},
        ]
        for case in mismatch_cases:
            kwargs = {
                "config_fingerprint": "fp-abc",
                "seed": 42,
                "model_files": {"model.safetensors": {"sha256": "abc", "bytes": 1}},
                "dataset_evidence": [{"path": "data/synthetic_train.jsonl", "sha256": "def", "rows_loaded": 100}],
                "image_seq_len": 36,
                "eos_token_id": 49279,
                "pad_token_id": 2,
            }
            kwargs.update(case)
            with self.assertRaises(checkpoint_utils.ResumeError, msg=f"resume must reject {case}"):
                checkpoint_utils.validate_resume_metadata({**metadata, "complete": True}, **kwargs)
        with self.assertRaises(checkpoint_utils.ResumeError):
            checkpoint_utils.validate_resume_metadata(
                {**metadata, "complete": False},
                config_fingerprint="fp-abc",
                seed=42,
                model_files={"model.safetensors": {"sha256": "abc", "bytes": 1}},
                dataset_evidence=[{"path": "data/synthetic_train.jsonl", "sha256": "def", "rows_loaded": 100}],
                image_seq_len=36,
                eos_token_id=49279,
                pad_token_id=2,
            )

        # RNG roundtrip must reproduce identical streams.
        torch = self.torch
        torch.manual_seed(123)
        random.seed(123)
        state = checkpoint_utils.capture_rng_state()
        torch_value = float(torch.rand(1))
        python_value = random.random()
        checkpoint_utils.restore_rng_state(state)
        self.assertEqual(float(torch.rand(1)), torch_value)
        self.assertEqual(random.random(), python_value)

    # -- 12. collate: pad masks, no visual padding ----------------------
    def test_12_collate_padding_and_visual_guard(self) -> None:
        from src.train.sft_data import collate_examples

        synthetic = self.full_manifests["synthetic_train"]
        square = next(record for record in synthetic if record["width"] == record["height"])
        rectangle = next(record for record in synthetic if record["width"] != record["height"])
        first = self.example(square)
        second = self.example(rectangle)
        batch = collate_examples([first, second])
        self.assertEqual(tuple(batch["pixel_values"].shape), (2, 1, 3, 384, 384))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, len(first.input_ids)))
        tokenizer = self.processor.tokenizer
        eos_text = tokenizer.decode([49279], skip_special_tokens=False)
        for row, example in enumerate((first, second)):
            labels_row = batch["labels"][row].tolist()
            decoded = tokenizer.decode([value for value in labels_row if value != -100], skip_special_tokens=False)
            self.assertEqual(decoded.strip(), example.answer_text + eos_text)

        shorter = dataclasses.replace(
            first,
            input_ids=first.input_ids[:96],
            labels=first.labels[:96],
            attention_mask=first.attention_mask[:96],
        )
        padded = collate_examples([shorter, second])
        self.assertEqual(tuple(padded["input_ids"].shape), (2, len(second.input_ids)))
        pad_width = len(second.input_ids) - len(shorter.input_ids)
        self.assertTrue((padded["input_ids"][0, -pad_width:] == 2).all())
        self.assertTrue((padded["attention_mask"][0, -pad_width:] == 0).all())
        self.assertTrue((padded["labels"][0, -pad_width:] == -100).all())
        self.assertTrue((padded["attention_mask"][1] == 1).all())

        fake_pixels = dataclasses.replace(
            first, pixel_values=self.torch.zeros(1, 3, 64, 64, dtype=self.torch.float32)
        )
        with self.assertRaises(ValueError):
            collate_examples([fake_pixels, second])


    # -- 13. true gradient accumulation: 8 x batch=1 forwards, mean scale ----
    @staticmethod
    def _tiny_accumulation_model(torch):
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.batch_sizes: list[int] = []

            def forward(self, value, **kwargs):
                self.batch_sizes.append(int(value.shape[0]))
                return types.SimpleNamespace(loss=(self.weight * value).sum())

        return TinyModel()

    @staticmethod
    def _run_tiny_accumulation(torch, micro_values, *, gradient_accumulation_steps=8, max_grad_norm=100.0,
                               initial_optimizer_step=0, initial_samples_seen=0, should_stop=None):
        from src.train.accumulation import run_true_gradient_accumulation

        model = T5ContractTests._tiny_accumulation_model(torch)
        optimizer = torch.optim.SGD([model.weight], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        captured_grads: list[float] = []
        original_step = optimizer.step

        def recording_step(*args, **kwargs):
            captured_grads.append(float(model.weight.grad))
            return original_step(*args, **kwargs)

        optimizer.step = recording_step
        micro_batches = [
            (0, {"value": torch.tensor([float(value)])}) for value in micro_values
        ]
        result = run_true_gradient_accumulation(
            micro_batches=iter(micro_batches),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            gradient_accumulation_steps=gradient_accumulation_steps,
            expected_micro_batch_size=1,
            max_grad_norm=max_grad_norm,
            parameters=[model.weight],
            initial_optimizer_step=initial_optimizer_step,
            initial_samples_seen=initial_samples_seen,
            should_stop=should_stop,
        )
        return model, optimizer, result, captured_grads

    def test_13_true_accumulation_one_step(self) -> None:
        torch = self.torch
        model, _, result, captured_grads = self._run_tiny_accumulation(torch, range(1, 9))
        self.assertEqual(result.forward_calls, 8)
        self.assertEqual(result.backward_calls, 8)
        self.assertEqual(result.optimizer_step_calls, 1)
        self.assertEqual(result.scheduler_step_calls, 1)
        self.assertEqual(result.clip_calls, 1)
        self.assertEqual(result.zero_grad_calls, 1)
        self.assertEqual(result.micro_batches_seen, 8)
        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(result.micro_batch_size, 1)
        self.assertAlmostEqual(result.loss_scale, 0.125, places=12)
        self.assertEqual(model.batch_sizes, [1] * 8)
        # accumulated gradient = mean(1..8) = 4.5, not mean*1/8 = 0.5625
        self.assertAlmostEqual(captured_grads[0], 4.5, places=6)
        self.assertAlmostEqual(model.weight.item(), 1.0 - 0.1 * 4.5, places=6)
        self.assertNotAlmostEqual(captured_grads[0], 4.5 / 8.0, places=3)
        record = result.records[0]
        self.assertEqual(record.optimizer_step, 1)
        self.assertEqual(record.micro_batches_in_step, 8)
        self.assertEqual(record.micro_batch_sizes, (1,) * 8)
        self.assertEqual(record.samples_seen, 8)
        self.assertFalse(record.partial_accumulation)
        OBSERVATIONS["accumulation_evidence"] = {
            "scenario": "single optimizer step over 8 micro-batches of batch=1",
            "counters": {
                "forward_calls": result.forward_calls,
                "backward_calls": result.backward_calls,
                "optimizer_step_calls": result.optimizer_step_calls,
                "scheduler_step_calls": result.scheduler_step_calls,
                "clip_calls": result.clip_calls,
                "zero_grad_calls": result.zero_grad_calls,
                "micro_batches_seen": result.micro_batches_seen,
                "optimizer_steps": result.optimizer_steps,
                "loss_scale": result.loss_scale,
            },
            "micro_batch_sizes_observed": model.batch_sizes,
            "captured_gradients": captured_grads,
            "expected_accumulated_gradient": 4.5,
            "buggy_double_scaled_gradient": 4.5 / 8.0,
            "weight_after_step": model.weight.item(),
            "true_accumulation": True,
        }

    def test_14_two_steps_and_partial_final_group(self) -> None:
        torch = self.torch
        model, _, result, captured_grads = self._run_tiny_accumulation(torch, range(1, 17))
        self.assertEqual(result.forward_calls, 16)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.optimizer_step_calls, 2)
        self.assertEqual(result.clip_calls, 2)
        self.assertEqual(result.scheduler_step_calls, 2)
        self.assertEqual(result.zero_grad_calls, 2)
        self.assertEqual(result.micro_batches_seen, 16)
        self.assertEqual([record.samples_seen for record in result.records], [8, 16])
        self.assertEqual([record.optimizer_step for record in result.records], [1, 2])
        self.assertEqual([record.micro_batches_in_step for record in result.records], [8, 8])
        self.assertAlmostEqual(captured_grads[0], 4.5, places=6)
        self.assertAlmostEqual(captured_grads[1], 12.5, places=6)
        self.assertAlmostEqual(model.weight.item(), 1.0 - 0.1 * (4.5 + 12.5), places=6)

        partial_model, _, partial_result, partial_grads = self._run_tiny_accumulation(torch, range(1, 6))
        self.assertEqual(partial_result.optimizer_steps, 1)
        self.assertEqual(partial_result.forward_calls, 5)
        self.assertEqual(partial_result.micro_batches_seen, 5)
        self.assertTrue(partial_result.partial_final_step)
        self.assertEqual(partial_result.stop_reason, "stream_exhausted")
        self.assertTrue(partial_result.records[0].partial_accumulation)
        self.assertEqual(partial_result.records[0].micro_batches_in_step, 5)
        self.assertAlmostEqual(partial_grads[0], 15.0 / 8.0, places=6)
        self.assertAlmostEqual(partial_model.weight.item(), 1.0 - 0.1 * (15.0 / 8.0), places=6)
        OBSERVATIONS["accumulation_two_steps_and_partial"] = {
            "two_step_gradients": captured_grads,
            "two_step_weight": model.weight.item(),
            "partial_group_gradient": partial_grads[0],
            "partial_expected": 15.0 / 8.0,
            "partial_accumulation_flag": partial_result.partial_final_step,
        }

    def test_15_accumulation_stop_and_resume_replay(self) -> None:
        torch = self.torch
        model, _, result, captured_grads = self._run_tiny_accumulation(
            torch, range(1, 25), should_stop=lambda step: step >= 1
        )
        self.assertEqual(result.forward_calls, 8)
        self.assertEqual(result.micro_batches_seen, 8)
        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.records[-1].optimizer_step, 1)
        self.assertAlmostEqual(captured_grads[0], 4.5, places=6)

        # Resume from step 1 / 8 samples: replay skip must start at micro-batch 9.
        resumed_model, _, resumed_result, resumed_grads = self._run_tiny_accumulation(
            torch,
            range(9, 17),
            initial_optimizer_step=1,
            initial_samples_seen=8,
            should_stop=lambda step: step >= 2,
        )
        self.assertEqual(resumed_result.forward_calls, 8)
        self.assertEqual(resumed_result.optimizer_steps, 1)
        self.assertEqual(resumed_result.records[-1].optimizer_step, 2)
        self.assertEqual(resumed_result.records[-1].samples_seen, 16)
        self.assertAlmostEqual(resumed_grads[0], 12.5, places=6)

        # The production trainer must use this helper and must not keep the
        # legacy batch-8 pseudo-accumulation path.
        source = (PROJECT_ROOT / "src" / "train" / "train_sft.py").read_text(encoding="utf-8")
        self.assertIn("run_true_gradient_accumulation(", source)
        self.assertIn("collate_examples([example])", source)
        self.assertNotIn("pending_examples", source)
        self.assertNotIn("collate_examples(pending_examples)", source)
        OBSERVATIONS["accumulation_stop_and_resume"] = {
            "max_steps_forward_calls": result.forward_calls,
            "max_steps_stop_reason": result.stop_reason,
            "resumed_optimizer_step": resumed_result.records[-1].optimizer_step,
            "resumed_samples_seen": resumed_result.records[-1].samples_seen,
            "resumed_gradient": resumed_grads[0],
            "production_uses_helper": True,
            "legacy_batch8_path_absent": True,
        }


    # -- 16. probe-only is inference-only (no optimizer / backward / training) --
    def test_16_probe_only_is_inference_only(self) -> None:
        import ast

        source_path = PROJECT_ROOT / "src" / "train" / "probe_only.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_attrs = {"backward", "zero_grad", "step", "train"}
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                    violations.append(f"call .{node.func.attr}() at line {getattr(node, 'lineno', '?')}")
                for keyword in node.keywords:
                    if keyword.arg == "labels":
                        violations.append(f"labels= keyword at line {getattr(node, 'lineno', '?')}")
            if isinstance(node, ast.Name) and node.id in ("optimizer", "scheduler"):
                violations.append(f"name {node.id} at line {getattr(node, 'lineno', '?')}")
        self.assertEqual([], violations, msg=f"probe_only must not contain training operations: {violations}")
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("is_trainable=False", source)
        self.assertIn("model.eval()", source)

        from src.train import probe_only

        plan = probe_only.build_probe_plan(
            config_path=CONFIG_OVERFIT,
            adapter_dir=PROJECT_ROOT / "adapters" / "sft_overfit100" / "adapter",
            training_summary_path=PROJECT_ROOT / "adapters" / "sft_overfit100" / "training_summary.json",
            training_results_path=PROJECT_ROOT / "results" / "train_sft_overfit100_s42_training.json",
            limit=20,
            max_new_tokens=64,
        )
        self.assertEqual(plan["status"], "preflight_passed")
        self.assertFalse(plan["gpu_used"])
        self.assertTrue(plan["diagnostic_only"])
        self.assertFalse(plan["official_eval"])
        self.assertEqual(plan["probe_records"], 20)
        self.assertEqual(plan["max_new_tokens"], 64)
        self.assertTrue(all(plan["identity"]["checks"].values()))
        contract = plan["inference_only_contract"]
        self.assertFalse(contract["computes_training_loss"])
        self.assertFalse(contract["calls_backward"])
        self.assertFalse(contract["calls_optimizer_step"])
        self.assertFalse(contract["calls_scheduler_step"])
        self.assertFalse(contract["calls_zero_grad"])
        from src.train.model_audit import configure_processor
        from src.train.sft_data import build_probe_inputs

        self.assertIn("configure_processor(", source)
        configure_processor(self.processor, image_size=384)
        first_record = self.full_manifests["synthetic_train"][0]
        probe_inputs = build_probe_inputs(first_record, self.processor, model_path=MODEL_PATH)
        contract = probe_inputs["visual_token_contract"]
        self.assertEqual(contract["placeholder_count"], 36)
        self.assertEqual(contract["total_views"] * contract["tokens_per_view"], 36)
        self.assertEqual(contract["pixel_values_shape"], [1, 1, 3, 384, 384])
        OBSERVATIONS["probe_only_evidence"] = {
            "ast_forbidden_operations": violations,
            "source_checks": ["torch.inference_mode()", "is_trainable=False", "model.eval()"],
            "plan_status": plan["status"],
            "identity_checks": plan["identity"]["checks"],
            "adapter_model_sha256": plan["identity"]["adapter_model_sha256"],
            "probe_records": plan["probe_records"],
            "max_new_tokens": plan["max_new_tokens"],
            "inference_only_contract": contract,
            "processor_configured_for_384": True,
            "probe_visual_placeholders": contract["placeholder_count"],
            "probe_visual_pixel_shape": contract["pixel_values_shape"],
        }


    # -- 17. e48 extension config is independent and fixed-contract-compatible --
    def test_17_extension_e48_config_and_plan(self) -> None:
        from src.train.config import config_fingerprint
        from src.train.train_sft import compute_plan

        e48 = self.e48_config
        self.assertEqual(e48["run_name"], "sft_overfit100_e48_s42")
        self.assertEqual(e48["output"]["dir"], "adapters/sft_overfit100_e48")
        self.assertEqual(e48["output"]["log"], "logs/train/sft_overfit100_e48.jsonl")
        self.assertEqual(e48["training"]["num_epochs"], 48)
        self.assertEqual(e48["training"]["save_every_steps"], 100)
        self.assertEqual(e48["probe"]["max_new_tokens"], 64)
        self.assertEqual(e48["training"]["batch_size"], 1)
        self.assertEqual(e48["training"]["gradient_accumulation_steps"], 8)
        self.assertEqual(e48["training"]["effective_batch_size"], 8)
        self.assertEqual(e48["training"]["learning_rate"], 1e-4)
        self.assertEqual(e48["lora"]["r"], 16)
        self.assertEqual(e48["lora"]["lora_alpha"], 32)
        self.assertEqual(e48["image"]["image_size"], 384)
        self.assertEqual(e48["image"]["expected_image_seq_len"], 36)
        self.assertEqual(e48["model"]["dtype"], "bfloat16")
        self.assertTrue(e48["model"]["freeze_vision_tower"])
        self.assertTrue(e48["model"]["gradient_checkpointing"])
        self.assertEqual(e48["seed"], 42)
        self.assertEqual(e48["answer"]["eos_token_id"], 49279)
        self.assertEqual(e48["answer"]["pad_token_id"], 2)
        self.assertEqual(e48["data"]["train"][0]["limit"], 100)

        plan = compute_plan(e48, self.train_records)
        self.assertEqual(plan["train_records"], 100)
        self.assertEqual(plan["micro_batches_per_optimizer_step"], 8)
        self.assertEqual(plan["micro_forwards_total"], 4800)
        self.assertEqual(plan["total_optimizer_steps"], 600)
        self.assertEqual(plan["checkpoint_steps"], [100, 200, 300, 400, 500, 600])
        self.assertEqual(plan["loss_scale_per_micro_batch"], 0.125)

        # Existing run's config/adapter/results must remain untouched.
        original = self.overfit_config
        self.assertEqual(original["run_name"], "sft_overfit100_s42")
        self.assertEqual(original["output"]["dir"], "adapters/sft_overfit100")
        self.assertEqual(original["probe"]["max_new_tokens"], 16)
        self.assertNotEqual(e48["config_fingerprint"], original["config_fingerprint"])
        self.assertNotEqual(
            config_fingerprint({k: v for k, v in e48.items() if not k.startswith("_")}),
            original["config_fingerprint"],
        )
        OBSERVATIONS["extension_e48_config"] = {
            "config_path": e48["_config_path"],
            "config_fingerprint": e48["config_fingerprint"],
            "run_name": e48["run_name"],
            "output_dir": e48["output"]["dir"],
            "num_epochs": e48["training"]["num_epochs"],
            "micro_forwards_total": plan["micro_forwards_total"],
            "total_optimizer_steps": plan["total_optimizer_steps"],
            "checkpoint_steps": plan["checkpoint_steps"],
            "probe_max_new_tokens": e48["probe"]["max_new_tokens"],
            "original_config_unchanged": True,
        }


    # -- 18. single-sample learnability control config and plan --
    def test_18_single_sample_control_config_and_plan(self) -> None:
        from src.train.config import config_fingerprint
        from src.train.train_sft import compute_plan

        single = self.overfit1_config
        self.assertEqual(single["run_name"], "sft_overfit1_e1600_s42")
        self.assertEqual(single["output"]["dir"], "adapters/sft_overfit1_e1600")
        self.assertEqual(single["output"]["log"], "logs/train/sft_overfit1_e1600.jsonl")
        self.assertEqual(single["data"]["train"][0]["limit"], 1)
        self.assertEqual(single["training"]["num_epochs"], 1600)
        self.assertEqual(single["training"]["save_every_steps"], 100)
        self.assertEqual(single["probe"]["n"], 1)
        self.assertEqual(single["probe"]["max_new_tokens"], 64)
        self.assertEqual(single["training"]["batch_size"], 1)
        self.assertEqual(single["training"]["gradient_accumulation_steps"], 8)
        self.assertEqual(single["training"]["effective_batch_size"], 8)
        self.assertEqual(single["training"]["learning_rate"], 1e-4)
        self.assertEqual(single["lora"]["r"], 16)
        self.assertEqual(single["lora"]["lora_alpha"], 32)
        self.assertEqual(single["image"]["image_size"], 384)
        self.assertEqual(single["image"]["expected_image_seq_len"], 36)
        self.assertEqual(single["model"]["dtype"], "bfloat16")
        self.assertTrue(single["model"]["freeze_vision_tower"])
        self.assertTrue(single["model"]["gradient_checkpointing"])
        self.assertEqual(single["seed"], 42)
        self.assertEqual(single["answer"]["eos_token_id"], 49279)
        self.assertEqual(single["answer"]["pad_token_id"], 2)

        plan = compute_plan(single, self.train_records[:1])
        self.assertEqual(plan["train_records"], 1)
        self.assertEqual(plan["micro_batches_per_optimizer_step"], 8)
        self.assertEqual(plan["micro_forwards_total"], 1600)
        self.assertEqual(plan["total_optimizer_steps"], 200)
        self.assertEqual(plan["checkpoint_steps"], [100, 200])
        self.assertEqual(plan["loss_scale_per_micro_batch"], 0.125)

        # Existing run configs must remain untouched.
        self.assertEqual(self.overfit_config["probe"]["max_new_tokens"], 16)
        self.assertEqual(self.e48_config["training"]["num_epochs"], 48)
        self.assertNotEqual(single["config_fingerprint"], self.overfit_config["config_fingerprint"])
        self.assertNotEqual(single["config_fingerprint"], self.e48_config["config_fingerprint"])
        OBSERVATIONS["extension_overfit1_config"] = {
            "config_path": single["_config_path"],
            "config_fingerprint": single["config_fingerprint"],
            "run_name": single["run_name"],
            "output_dir": single["output"]["dir"],
            "train_limit": 1,
            "num_epochs": single["training"]["num_epochs"],
            "micro_forwards_total": plan["micro_forwards_total"],
            "total_optimizer_steps": plan["total_optimizer_steps"],
            "checkpoint_steps": plan["checkpoint_steps"],
            "probe": single["probe"],
            "original_configs_unchanged": True,
        }


    # -- 19. multi-seed s43/s44 configs are identical except seed/paths --
    def test_19_multiseed_s43_s44_configs_and_plans(self) -> None:
        from src.train.config import load_config
        from src.train.train_sft import compute_plan

        base = self.v1_config
        base_fingerprint = base["config_fingerprint"]
        for seed, expected_run in ((43, "sft_v1_s43"), (44, "sft_v1_s44")):
            cfg = load_config(PROJECT_ROOT / "configs" / f"sft_v1_s{seed}.json")
            self.assertEqual(cfg["seed"], seed)
            self.assertEqual(cfg["run_name"], expected_run)
            self.assertEqual(cfg["output"]["dir"], f"adapters/sft_v1_s{seed}")
            self.assertEqual(cfg["output"]["log"], f"logs/train/sft_v1_s{seed}.jsonl")
            self.assertEqual(cfg["training"]["num_epochs"], base["training"]["num_epochs"])
            self.assertEqual(cfg["training"]["save_every_steps"], 200)
            self.assertEqual(cfg["training"]["batch_size"], 1)
            self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 8)
            self.assertEqual(cfg["training"]["effective_batch_size"], 8)
            self.assertEqual(cfg["training"]["learning_rate"], 1e-4)
            self.assertEqual(cfg["lora"]["r"], 16)
            self.assertEqual(cfg["lora"]["lora_alpha"], 32)
            self.assertEqual(cfg["image"]["image_size"], 384)
            self.assertEqual(cfg["image"]["expected_image_seq_len"], 36)
            self.assertEqual(cfg["model"]["dtype"], "bfloat16")
            self.assertTrue(cfg["model"]["freeze_vision_tower"])
            self.assertTrue(cfg["model"]["gradient_checkpointing"])
            self.assertEqual(cfg["answer"]["eos_token_id"], 49279)
            self.assertEqual(cfg["answer"]["pad_token_id"], 2)
            self.assertEqual(cfg["probe"]["max_new_tokens"], 64)
            self.assertEqual([entry["limit"] for entry in cfg["data"]["train"]], [None, None])
            plan = compute_plan(cfg, self.train_records)
            self.assertEqual(plan["train_records"], 100)
            # compute_plan is fed the 100-record selection here only for the fixed
            # contract shape; the full-data plan is validated by the dry-run files.
            self.assertEqual(plan["micro_batches_per_optimizer_step"], 8)
            self.assertNotEqual(cfg["config_fingerprint"], base_fingerprint)
        self.assertEqual(base["seed"], 42)
        self.assertEqual(base["run_name"], "sft_v1_s42")
        self.assertEqual(base["output"]["dir"], "adapters/sft_v1")
        OBSERVATIONS["multiseed_configs"] = {
            "s43": "configs/sft_v1_s43.json",
            "s44": "configs/sft_v1_s44.json",
            "base_seed42_unchanged": True,
            "fixed_contract_inherited": True,
        }


    # -- 20. T7 coordinate representations: prompt/encode/parse consistency --
    def test_20_coordinate_representation_variants(self) -> None:
        from src.eval.prompts import PROMPT_TEMPLATE as P1_TEMPLATE
        from src.train import coordinate_variants as cv
        from src.train.config import load_config

        configs = {
            "a_normalized": load_config(PROJECT_ROOT / "configs/t7_a_normalized.json"),
            "b_bin1000": load_config(PROJECT_ROOT / "configs/t7_b_bin1000.json"),
            "c_pixel": load_config(PROJECT_ROOT / "configs/t7_c_pixel.json"),
        }
        expected = {
            "a_normalized": "normalized_decimal",
            "b_bin1000": "bin1000_integer",
            "c_pixel": "pixel_xyxy",
        }
        fingerprints = set()
        for key, cfg in configs.items():
            self.assertEqual(cfg["answer"]["representation"], expected[key])
            self.assertEqual(cfg["run_name"], {
                "a_normalized": "sft_t7_a_norm_s42",
                "b_bin1000": "sft_t7_b_bin1000_s42",
                "c_pixel": "sft_t7_c_pixel_s42",
            }[key])
            self.assertEqual(cfg["training"]["num_epochs"], 1)
            self.assertEqual(cfg["training"]["batch_size"], 1)
            self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 8)
            self.assertEqual(cfg["training"]["learning_rate"], 1e-4)
            self.assertEqual(cfg["lora"]["r"], 16)
            self.assertEqual(cfg["lora"]["lora_alpha"], 32)
            self.assertEqual(cfg["image"]["image_size"], 384)
            self.assertEqual(cfg["answer"]["eos_token_id"], 49279)
            self.assertEqual(cfg["answer"]["pad_token_id"], 2)
            self.assertEqual(cfg["probe"]["max_new_tokens"], 64)
            fingerprints.add(cfg["config_fingerprint"])
        self.assertEqual(len(fingerprints), 3)

        self.assertEqual(cv.PROMPT_TEMPLATES["normalized_decimal"], P1_TEMPLATE)
        self.assertEqual(cv.PROMPT_IDS["normalized_decimal"], "p1")
        self.assertNotEqual(cv.PROMPT_TEMPLATES["bin1000_integer"], P1_TEMPLATE)
        self.assertIn("0-1000", cv.PROMPT_TEMPLATES["bin1000_integer"])
        self.assertIn("pixel coordinates", cv.PROMPT_TEMPLATES["pixel_xyxy"])

        def approx(a, b, tol=1e-8):
            return max(abs(x - y) for x, y in zip(a, b)) <= tol

        for record in self.train_records:
            a_text, a_meta = cv.encode_answer(record, "normalized_decimal")
            self.assertEqual(a_meta["representation"], "normalized_decimal")
            self.assertTrue(approx(cv.parse_metric_prediction(
                a_text, "normalized_decimal", width=record["width"], height=record["height"]
            )[0], record["answer"], tol=1e-4))
            b_text, b_meta = cv.encode_answer(record, "bin1000_integer")
            b_values = cv._bracketed_numbers(b_text)[0]
            self.assertTrue(all(float(v).is_integer() for v in b_values))
            self.assertTrue(all(0 <= v <= 1000 for v in b_values))
            self.assertTrue(approx(cv.parse_metric_prediction(
                b_text, "bin1000_integer", width=record["width"], height=record["height"]
            )[0], record["answer"], tol=6e-4))
            c_text, c_meta = cv.encode_answer(record, "pixel_xyxy")
            self.assertEqual(c_text, "[" + ", ".join(str(int(v)) for v in record["raw_bbox"]) + "]")
            c_norm, c_error, c_audit = cv.parse_metric_prediction(
                c_text, "pixel_xyxy", width=record["width"], height=record["height"]
            )
            self.assertIsNone(c_error)
            self.assertTrue(c_audit["conversion_applied"])
            self.assertTrue(approx(c_norm, record["answer"]))
            c_score, _ = cv.score_prediction_variant(c_text, record, "pixel_xyxy")
            self.assertGreater(c_score.iou, 0.999999)
            self.assertTrue(c_score.correct_at_0_5)

        # RefCOCO pixels are integer-rounded vs the stored normalized answer; C must
        # accept them under the pixel-rounding bound and convert them to GT.
        refcoco_records = self.full_manifests["refcoco_train"][:50]
        for record in refcoco_records:
            c_text, c_meta = cv.encode_answer(record, "pixel_xyxy")
            self.assertLessEqual(
                c_meta["max_abs_pixel_answer_gap"], c_meta["pixel_rounding_tolerance"] + 1e-12
            )
            c_norm, c_error, c_audit = cv.parse_metric_prediction(
                c_text, "pixel_xyxy", width=record["width"], height=record["height"]
            )
            self.assertIsNone(c_error)
            self.assertTrue(c_audit["conversion_applied"])
            # RefCOCO pixel rounding can cost a little IoU on small boxes; still proves the C conversion.
            self.assertGreater(cv.score_prediction_variant(c_text, record, "pixel_xyxy")[0].iou, 0.95)

        for representation in ("normalized_decimal", "bin1000_integer"):
            text, _ = cv.encode_answer(self.train_records[0], representation)
            score, audit = cv.score_prediction_variant(text, self.train_records[0], representation)
            # A is exact; B rounds normalized coordinates to 1/1000, so IoU is slightly < 1.
            self.assertGreater(score.iou, 0.99)
            self.assertFalse(audit["conversion_applied"])

        bad_box, bad_error, _ = cv.parse_metric_prediction(
            "[1000, 1000, 2000, 2000]", "pixel_xyxy", width=256, height=320
        )
        self.assertIsNone(bad_box)
        self.assertEqual(bad_error, "pixel_box_outside_or_reversed")
        bad_order, bad_order_error, _ = cv.parse_metric_prediction(
            "[200, 200, 100, 300]", "pixel_xyxy", width=256, height=320
        )
        self.assertIsNone(bad_order)
        self.assertIsNotNone(bad_order_error)
        # C must not be scored through the frozen parser as if it were bin/normalized.
        frozen_c = cv.parse_box(bad_box if bad_box else "[10, 20, 30, 40]")
        self.assertIsNotNone(frozen_c)  # frozen parser accepts [10,20,30,40], but C conversion is the required path
        OBSERVATIONS["coordinate_variants"] = {
            "configs": {key: {"path": cfg["_config_path"], "fingerprint": cfg["config_fingerprint"]} for key, cfg in configs.items()},
            "representations": expected,
            "records_checked": len(self.train_records),
            "prompt_templates": {name: template for name, template in cv.PROMPT_TEMPLATES.items()},
            "a_frozen_parser_used": True,
            "b_frozen_parser_used": True,
            "c_conversion_required": True,
            "c_conversion_checks": "pixel bounds + ordering + normalize by image width/height, then frozen iou/metrics",
        }


    # -- 21. T7-B bin1000 seed43/44 configs: independent, same fixed contract --
    def test_21_t7b_seed43_44_configs(self) -> None:
        from src.train.config import load_config

        base = load_config(PROJECT_ROOT / "configs/t7_b_bin1000.json")
        self.assertEqual(base["seed"], 42)
        self.assertEqual(base["answer"]["representation"], "bin1000_integer")
        self.assertEqual(base["run_name"], "sft_t7_b_bin1000_s42")
        seen = {base["config_fingerprint"]}
        for seed in (43, 44):
            cfg = load_config(PROJECT_ROOT / "configs" / f"t7_b_bin1000_s{seed}.json")
            self.assertEqual(cfg["seed"], seed)
            self.assertEqual(cfg["run_name"], f"sft_t7_b_bin1000_s{seed}")
            self.assertEqual(cfg["output"]["dir"], f"adapters/sft_t7_b_bin1000_s{seed}")
            self.assertEqual(cfg["output"]["log"], f"logs/train/sft_t7_b_bin1000_s{seed}.jsonl")
            self.assertEqual(cfg["answer"]["representation"], "bin1000_integer")
            self.assertEqual(cfg["data"]["train"][0]["limit"], None)
            self.assertEqual(cfg["data"]["train"][1]["limit"], None)
            self.assertEqual(cfg["training"]["num_epochs"], 1)
            self.assertEqual(cfg["training"]["save_every_steps"], 200)
            self.assertEqual(cfg["training"]["batch_size"], 1)
            self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 8)
            self.assertEqual(cfg["training"]["effective_batch_size"], 8)
            self.assertEqual(cfg["training"]["learning_rate"], 1e-4)
            self.assertEqual(cfg["lora"]["r"], 16)
            self.assertEqual(cfg["lora"]["lora_alpha"], 32)
            self.assertEqual(cfg["image"]["image_size"], 384)
            self.assertEqual(cfg["image"]["expected_image_seq_len"], 36)
            self.assertEqual(cfg["model"]["dtype"], "bfloat16")
            self.assertTrue(cfg["model"]["freeze_vision_tower"])
            self.assertTrue(cfg["model"]["gradient_checkpointing"])
            self.assertEqual(cfg["answer"]["eos_token_id"], 49279)
            self.assertEqual(cfg["answer"]["pad_token_id"], 2)
            self.assertEqual(cfg["probe"]["max_new_tokens"], 64)
            self.assertNotIn(cfg["config_fingerprint"], seen)
            seen.add(cfg["config_fingerprint"])
        self.assertEqual(len(seen), 3)
        OBSERVATIONS["t7b_seed43_44_configs"] = {
            "base_seed42_config": "configs/t7_b_bin1000.json",
            "seed43_config": "configs/t7_b_bin1000_s43.json",
            "seed44_config": "configs/t7_b_bin1000_s44.json",
            "representation": "bin1000_integer",
            "fixed_contract_inherited": True,
            "only_seed_run_output_log_changed": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results" / "train_t5_cpu_tests.json")
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != RESULTS_ROOT or not report_path.name.startswith("train_"):
        raise SystemExit("report must be directly under results/ with a train_ prefix")

    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T5ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - started

    import torch
    import transformers

    from src.train.config import load_config
    from src.train.model_audit import audit_model_geometry, audit_pad_eos
    from src.train.vision_tokens import sha256_file

    config = load_config(CONFIG_OVERFIT)
    source_files = [
        "src/train/__init__.py",
        "src/train/config.py",
        "src/train/vision_tokens.py",
        "src/train/sft_data.py",
        "src/train/checkpoint.py",
        "src/train/accumulation.py",
        "src/train/probe_only.py",
        "src/train/model_audit.py",
        "src/train/train_sft.py",
        "src/train/test_t5_cpu.py",
        "src/eval/grounding.py",
        "src/eval/prompts.py",
        "src/eval/dataset.py",
        "src/eval/run_zeroshot.py",
    ]
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "suite": "T5 SFT CPU contract / preflight",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "gpu_used": False,
        "device": "cpu",
        "torch_cuda_imported": bool(torch.cuda.is_available()),
        "image_size": 384,
        "image_seq_len": 36,
        "processor_default_image_seq_len": 64,
        "observations": OBSERVATIONS,
        "model_geometry": audit_model_geometry(model_path=MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH / "model.safetensors"),
        "config": {
            "path": config["_config_path"],
            "sha256": config["_config_sha256"],
            "fingerprint": config["config_fingerprint"],
        },
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in source_files},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": __import__("importlib.metadata").metadata.version("peft"),
        },
        "failure_details": [str(item) for item in result.failures + result.errors],
        "elapsed_seconds": elapsed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "CPU-only preflight. No model weights loaded, no CUDA context initialized, no training started. "
            "GPU training requires an explicit Lead lease; see src/train/README.md."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tests_run": report["tests_run"],
        "failures": report["failures"],
        "errors": report["errors"],
        "gpu_used": report["gpu_used"],
        "report": str(report_path),
        "elapsed_seconds": elapsed,
    }, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
