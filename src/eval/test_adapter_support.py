"""CPU-only regression for the read-only PEFT adapter path of run_zeroshot.

These tests never load CUDA or model weights. The separate
test_adapter_forward_cpu.py performs the real CPU base+adapter forward.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any

from src.eval.run_zeroshot import ADAPTER_FILES, load_adapter_metadata, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "sft_v1" / "adapter"
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
TEMP_ROOT = PROJECT_ROOT / "results" / "eval_adapter_test_tmp"

DRY_RUN_BASE = [
    sys.executable,
    "-m",
    "src.eval.run_zeroshot",
    "--run-id",
    "adapter_cpu_dryrun_probe",
    "--dataset",
    "synthetic",
    "--split",
    "eval",
    "--data",
    "data/synthetic_eval.jsonl",
    "--output",
    "results/zeroshot_adapter_cpu_dryrun_probe.json",
    "--details-output",
    "results/eval_adapter_cpu_dryrun_probe_details.jsonl",
    "--model-path",
    "models/SmolVLM-256M-Instruct",
    "--expected-n",
    "1000",
    "--limit",
    "25",
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


def hash_tree(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def run_dry_run(extra: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        DRY_RUN_BASE + extra,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def make_temp_adapter(
    *,
    base_model: str = str(MODEL_PATH),
    peft_type: str = "LORA",
    task_type: str = "CAUSAL_LM",
    with_weights: bool = True,
) -> Path:
    # tempfile.mkdtemp creates directories that the file sandbox treats as
    # unwritable, so use a deterministic, explicitly created subdirectory.
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    directory = TEMP_ROOT / f"case_{uuid.uuid4().hex}"
    directory.mkdir()
    config = json.loads((ADAPTER_PATH / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model
    config["peft_type"] = peft_type
    config["task_type"] = task_type
    (directory / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if with_weights:
        (directory / "adapter_model.safetensors").write_bytes(b"deterministic-test-bytes\n")
    return directory


class AdapterMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_real_sft_v1_adapter_metadata(self) -> None:
        metadata = load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH)
        self.assertEqual(metadata["path"], "adapters/sft_v1/adapter")
        self.assertEqual(metadata["peft_type"], "LORA")
        self.assertEqual(metadata["task_type"], "CAUSAL_LM")
        self.assertEqual(metadata["lora_r"], 16)
        self.assertEqual(metadata["lora_alpha"], 32)
        self.assertIs(metadata["is_trainable"], False)
        self.assertEqual(
            metadata["adapter_config_sha256"],
            sha256_file(ADAPTER_PATH / "adapter_config.json"),
        )
        self.assertEqual(
            metadata["adapter_model_sha256"],
            sha256_file(ADAPTER_PATH / "adapter_model.safetensors"),
        )
        self.assertEqual(
            metadata["adapter_model_bytes"],
            (ADAPTER_PATH / "adapter_model.safetensors").stat().st_size,
        )
        self.assertEqual(metadata["peft_version"], importlib.metadata.version("peft"))
        self.assertEqual(metadata["base_model_name_or_path"], str(MODEL_PATH))

    def test_loading_metadata_is_read_only(self) -> None:
        before = hash_tree(ADAPTER_PATH)
        names_before = sorted(path.name for path in ADAPTER_PATH.iterdir())
        load_adapter_metadata(ADAPTER_PATH, model_path=MODEL_PATH)
        self.assertEqual(hash_tree(ADAPTER_PATH), before)
        self.assertEqual(sorted(path.name for path in ADAPTER_PATH.iterdir()), names_before)

    def test_missing_adapter_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_adapter_metadata(PROJECT_ROOT / "adapters" / "does_not_exist", model_path=MODEL_PATH)

    def test_missing_weight_file_raises(self) -> None:
        directory = make_temp_adapter(with_weights=False)
        try:
            with self.assertRaises(FileNotFoundError):
                load_adapter_metadata(directory, model_path=MODEL_PATH)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_wrong_base_model_raises(self) -> None:
        directory = make_temp_adapter(base_model=str(PROJECT_ROOT / "models" / "wrong-model"))
        try:
            with self.assertRaises(ValueError):
                load_adapter_metadata(directory, model_path=MODEL_PATH)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_non_lora_adapter_raises(self) -> None:
        directory = make_temp_adapter(peft_type="IA3")
        try:
            with self.assertRaises(ValueError):
                load_adapter_metadata(directory, model_path=MODEL_PATH)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_non_causal_lm_task_raises(self) -> None:
        directory = make_temp_adapter(task_type="SEQ_CLS")
        try:
            with self.assertRaises(ValueError):
                load_adapter_metadata(directory, model_path=MODEL_PATH)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class AdapterDryRunTests(unittest.TestCase):
    def test_dry_run_with_adapter_preflight(self) -> None:
        completed = run_dry_run(["--adapter", "adapters/sft_v1/adapter"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        preflight = json.loads(completed.stdout)
        self.assertEqual(preflight["status"], "preflight_passed")
        self.assertEqual(preflight["adapter"], "adapters/sft_v1/adapter")
        self.assertEqual(
            preflight["adapter_model_sha256"],
            sha256_file(ADAPTER_PATH / "adapter_model.safetensors"),
        )
        self.assertIs(preflight["adapter_is_trainable"], False)
        self.assertEqual(preflight["n"], 25)
        self.assertEqual(preflight["image_seq_len"], 36)
        self.assertIs(preflight["gpu_used"], False)
        self.assertFalse((PROJECT_ROOT / "results" / "zeroshot_adapter_cpu_dryrun_probe.json").exists())
        self.assertFalse(
            (PROJECT_ROOT / "results" / "eval_adapter_cpu_dryrun_probe_details.jsonl").exists()
        )

    def test_dry_run_without_adapter_has_null_adapter(self) -> None:
        completed = run_dry_run([])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        preflight = json.loads(completed.stdout)
        self.assertIsNone(preflight["adapter"])
        self.assertIsNone(preflight["adapter_model_sha256"])
        self.assertIsNone(preflight["adapter_is_trainable"])
        self.assertIs(preflight["gpu_used"], False)

    def test_adapter_changes_config_fingerprint(self) -> None:
        base = run_dry_run([])
        adapted = run_dry_run(["--adapter", "adapters/sft_v1/adapter"])
        self.assertEqual(base.returncode, 0, base.stderr)
        self.assertEqual(adapted.returncode, 0, adapted.stderr)
        base_preflight = json.loads(base.stdout)
        adapted_preflight = json.loads(adapted.stdout)
        self.assertNotEqual(
            base_preflight["config_fingerprint"], adapted_preflight["config_fingerprint"]
        )
        self.assertEqual(base_preflight["dataset_sha256"], adapted_preflight["dataset_sha256"])
        self.assertEqual(base_preflight["model_sha256"], adapted_preflight["model_sha256"])

    def test_dry_run_rejects_directory_without_adapter_files(self) -> None:
        completed = run_dry_run(["--adapter", "models/SmolVLM-256M-Instruct"])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("adapter directory is missing", completed.stderr)

    def test_dry_run_rejects_missing_adapter_directory(self) -> None:
        completed = run_dry_run(["--adapter", "adapters/does_not_exist"])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("adapter directory not found", completed.stderr)

    def test_dry_run_rejects_adapter_wrong_base_model(self) -> None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        directory = make_temp_adapter(base_model=str(PROJECT_ROOT / "models" / "wrong-model"))
        try:
            relative = directory.relative_to(PROJECT_ROOT).as_posix()
            completed = run_dry_run(["--adapter", relative])
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("does not match --model-path", completed.stderr)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval_adapter_tests.json",
    )
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    if report_path.parent.resolve() != (PROJECT_ROOT / "results").resolve() or not report_path.name.startswith("eval_"):
        raise SystemExit("report must be directly under results/ with an eval_ prefix")

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    report: dict[str, Any] = {
        "run_id": "eval_adapter_support_tests_v1",
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "failure_details": [text for _, text in result.failures],
        "error_details": [text for _, text in result.errors],
        "gpu_used": False,
        "device": "cpu",
        "adapter": "adapters/sft_v1/adapter",
        "adapter_config_sha256": sha256_file(ADAPTER_PATH / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(ADAPTER_PATH / "adapter_model.safetensors"),
        "adapter_files": list(ADAPTER_FILES),
        "peft_version": importlib.metadata.version("peft"),
        "python": sys.version.split()[0],
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "grounding.py": Path(__file__).with_name("grounding.py"),
                "dataset.py": Path(__file__).with_name("dataset.py"),
                "prompts.py": Path(__file__).with_name("prompts.py"),
                "run_zeroshot.py": Path(__file__).with_name("run_zeroshot.py"),
                "validate_results.py": Path(__file__).with_name("validate_results.py"),
                "test_adapter_support.py": Path(__file__),
            }.items()
        },
        "dataset_sha256": sha256_file(DATA_PATH),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "gpu_used")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
