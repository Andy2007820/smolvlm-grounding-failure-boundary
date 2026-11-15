"""Merge the T6 SFT LoRA adapter into a standalone CPU checkpoint (T14 step 1).

Modes (all CPU-only):

``--dry-run``
    Validate paths, adapter identity and disk budget, print the plan.  No torch
    import, no write.
``--verify-only --artifact <dir>``
    Re-check an existing merged checkpoint from its files alone (no weights load).
``--verify-equivalence --artifact <dir>``
    Load base+adapter and the merged checkpoint, generate on 5 fixed samples and
    require byte-identical decoded outputs.  This is the scientific guard that
    the merge did not silently change the model.
default
    Merge on CPU in the requested dtype, save the full checkpoint + processor
    files, hash everything and write ``results/deploy_merge_report.json``.

The merge candidate is **provisional**: T12/KD has not run, so the deployed
adapter must be labelled ``provisional sft_v1 seed42`` until the Lead freezes the
final model.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    COPY_FROM_BASE_FILES,
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    IMAGE_SIZE,
    MERGED_REQUIRED_FILES,
    MODEL_PATH,
    PROJECT_ROOT,
    checked_results_path,
    cpu_count,
    expected_image_seq_len,
    file_facts,
    now_iso,
    project_relative,
    read_json,
    resolve_project_path,
    write_json_atomic,
)
from .env import adapter_facts, model_facts
from .infer import classify_artifact, peak_process_memory_mib

MERGE_DTYPES = ("float32", "bfloat16")
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="base SmolVLM-256M-Instruct directory")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="PEFT LoRA adapter directory")
    parser.add_argument("--output", default=str(DEFAULT_MERGED), help="merged checkpoint output directory")
    parser.add_argument("--dtype", choices=MERGE_DTYPES, default="float32", help="merge/save dtype (CPU)")
    parser.add_argument("--report", default="results/deploy_merge_report.json", help="JSON report under results/deploy_*")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan; no torch import, no write")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing merged checkpoint; no weights load")
    parser.add_argument("--verify-equivalence", action="store_true", help="compare base+adapter vs merged on 5 fixed samples")
    parser.add_argument("--artifact", default=None, help="existing merged checkpoint for --verify-only/--verify-equivalence")
    parser.add_argument("--samples", type=int, default=5, help="equivalence sample count (default 5)")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory / report")
    return parser


# --------------------------------------------------------------------- checks
def validate_merge_inputs(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Read-only validation of a merge request; raises ``ValueError``/``FileNotFoundError``."""

    if dtype not in MERGE_DTYPES:
        raise ValueError(f"dtype must be one of {MERGE_DTYPES}")
    model = model_facts(model_path, with_hash=False)
    if not model["all_required_files_present"]:
        raise FileNotFoundError(f"base model is incomplete: missing {model['missing_required_files']}")
    adapter = adapter_facts(adapter_path, with_hash=True)
    if not adapter["all_required_files_present"]:
        raise FileNotFoundError(f"adapter is incomplete: missing {adapter['missing_required_files']}")
    if not adapter["base_model_matches_project_model"]:
        raise ValueError(
            "adapter base_model_name_or_path does not match --model-path: "
            f"{adapter['base_model_name_or_path']!r} vs {str(model_path)!r}"
        )
    if adapter.get("peft_type") != "LORA":
        raise ValueError(f"adapter peft_type must be LORA, got {adapter.get('peft_type')!r}")
    if adapter.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"adapter task_type must be CAUSAL_LM, got {adapter.get('task_type')!r}")
    if output_path.exists() and not output_path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_path}")

    params = int(model.get("approx_parameters_millions") or 0) * 1e6
    if params <= 0:
        raise ValueError("cannot estimate parameter count from model.safetensors")
    expected_bytes = int(params * DTYPE_BYTES[dtype])
    disk = __import__("shutil").disk_usage(PROJECT_ROOT)
    required_free = int(expected_bytes * 1.5)
    if disk.free < required_free:
        raise OSError(
            f"not enough disk for the merged checkpoint: free={disk.free} required~{required_free} bytes"
        )
    return {
        "model_path": project_relative(model_path),
        "adapter_path": project_relative(adapter_path),
        "output_path": project_relative(output_path),
        "dtype": dtype,
        "estimated_parameters_millions": model.get("approx_parameters_millions"),
        "expected_output_bytes": expected_bytes,
        "disk_free_bytes": disk.free,
        "disk_required_bytes": required_free,
        "adapter": adapter,
        "model": model,
        "provisional": True,
        "provisional_reason": "T12/KD not run; final adapter not frozen by the Lead",
        "image_size": IMAGE_SIZE,
        "expected_image_seq_len": expected_image_seq_len(IMAGE_SIZE),
    }


def verify_merged_directory(artifact: Path, *, dtype: str | None = None, with_hash: bool = True) -> dict[str, Any]:
    """File-level verification of a merged checkpoint (never loads the weights)."""

    resolved = Path(artifact)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    record("directory_exists", resolved.is_dir(), str(resolved))
    if not resolved.is_dir():
        return {"artifact": project_relative(resolved), "ok": False, "checks": checks}

    missing = [name for name in MERGED_REQUIRED_FILES if not (resolved / name).is_file()]
    record("required_files_present", not missing, {"missing": missing})

    config_path = resolved / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = read_json(config_path)
        architecture = (config.get("architectures") or [None])[0]
        record("architecture_is_idefics3", architecture == "Idefics3ForConditionalGeneration", architecture)
        record("config_has_no_peft_block", "peft_config" not in config, sorted(config.keys())[:8])
        record("model_type_preserved", config.get("model_type") == "idefics3", config.get("model_type"))
    else:
        record("config_json_parse", False, "config.json missing")

    weights = resolved / "model.safetensors"
    weight_facts = file_facts(weights, with_hash=with_hash)
    record("weights_present_and_nonempty", bool(weight_facts.get("exists") and weight_facts.get("bytes")), weight_facts)

    processor_config = resolved / "processor_config.json"
    if processor_config.is_file():
        processor = read_json(processor_config)
        record(
            "processor_default_image_seq_len_untouched",
            processor.get("image_seq_len") == 64,
            {"on_disk": processor.get("image_seq_len"), "runtime_expected": expected_image_seq_len(IMAGE_SIZE)},
        )
    else:
        record("processor_config_present", False, "processor_config.json missing")

    kinds = classify_artifact(resolved, model_path=MODEL_PATH)
    record("classified_as_merged", kinds["kind"] == "merged", kinds["kind"])

    ok = all(check["passed"] for check in checks)
    return {
        "artifact": project_relative(resolved),
        "dtype": dtype,
        "ok": ok,
        "checks": checks,
        "files": {
            name: file_facts(resolved / name, with_hash=with_hash)
            for name in MERGED_REQUIRED_FILES
            if (resolved / name).is_file()
        },
    }


# ---------------------------------------------------------------------- merge
def merge_and_save(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    dtype: str = "float32",
    force: bool = False,
) -> dict[str, Any]:
    """Execute the CPU merge.  Importing torch happens here, never at module import."""

    if output_path.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_path} (pass --force)")
        import shutil

        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    import torch
    from transformers import AutoModelForImageTextToText

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    started = time.perf_counter()
    torch.manual_seed(42)
    threads = int(torch.get_num_threads())

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype_map[dtype], local_files_only=True
    )
    base_class = type(base_model).__name__
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    lora_modules_before = [name for name, module in peft_model.named_modules() if hasattr(module, "lora_A")]
    if not lora_modules_before:
        raise RuntimeError("adapter attached no LoRA modules; nothing to merge")

    merged_model = peft_model.merge_and_unload()
    lora_modules_after = [name for name, module in merged_model.named_modules() if hasattr(module, "lora_A")]
    if lora_modules_after:
        raise RuntimeError(f"merge left {len(lora_modules_after)} LoRA modules; refusing to save")
    if hasattr(merged_model, "peft_config") and getattr(merged_model, "peft_config"):
        raise RuntimeError("merge left a peft_config on the model; refusing to save")
    merged_class = type(merged_model).__name__
    if "Peft" in merged_class:
        raise RuntimeError(f"merged model class still looks wrapped: {merged_class}")

    merged_model.save_pretrained(output_path, safe_serialization=True)

    # The processor/tokenizer files are copied verbatim from the base model so
    # the deployed checkpoint keeps the shipped contract on disk:
    #   processor_config.json image_seq_len = 64, do_image_splitting = true,
    #   size longest_edge = 2048.
    # Deployment code sets 384/no-split/36 at runtime (same as the evaluator);
    # the on-disk default must never be silently rewritten by a merge.
    import shutil

    copied: list[str] = []
    for name in COPY_FROM_BASE_FILES:
        source = model_path / name
        if source.is_file():
            shutil.copyfile(source, output_path / name)
            copied.append(name)
    missing = [name for name in MERGED_REQUIRED_FILES if not (output_path / name).is_file()]
    if missing:
        raise RuntimeError(f"merged checkpoint is missing files after processor copy: {missing}")

    elapsed = time.perf_counter() - started
    report = {
        "status": "completed",
        "mode": "merge",
        "started_at": now_iso(),
        "artifact": project_relative(output_path),
        "dtype": dtype,
        "base_model": project_relative(model_path),
        "adapter": project_relative(adapter_path),
        "base_model_class": base_class,
        "merged_model_class": merged_class,
        "lora_modules_before_merge": len(lora_modules_before),
        "lora_modules_after_merge": len(lora_modules_after),
        "peft_config_removed": True,
        "torch_threads": threads,
        "cpu_count": cpu_count(),
        "peak_process_memory_mib": peak_process_memory_mib(),
        "elapsed_seconds": elapsed,
        "copied_from_base": copied,
        "provisional": True,
        "provisional_reason": "T12/KD not run; the Lead has not frozen the final adapter",
        "environment": {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
        },
    }
    merged_model = None
    peft_model = None
    base_model = None
    del merged_model, peft_model, base_model
    return report


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


# -------------------------------------------------------- equivalence check
def verify_merge_equivalence(
    *,
    artifact: Path,
    adapter: Path,
    model_path: Path = MODEL_PATH,
    samples: int = 5,
) -> dict[str, Any]:
    """Base+adapter vs merged must decode to identical text on fixed samples."""

    from .config import read_jsonl

    dataset = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"
    records = read_jsonl(dataset, limit=samples)
    if len(records) < samples:
        raise RuntimeError(f"dataset {dataset} has fewer than {samples} rows")

    from PIL import Image

    from .infer import HFPredictor

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with HFPredictor(model_path=model_path, adapter=adapter, device="cpu", dtype="float32") as reference:
        with HFPredictor(model_path=model_path, merged=artifact, device="cpu", dtype="float32") as deployed:
            for record in records:
                image_path = PROJECT_ROOT / record["image"]
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    reference_output = reference.predict(image, record["question"])
                    merged_output = deployed.predict(image, record["question"])
                results.append(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "reference_output_text": reference_output["output_text"],
                        "merged_output_text": merged_output["output_text"],
                        "reference_box": reference_output["predicted_box"],
                        "merged_box": merged_output["predicted_box"],
                        "text_identical": reference_output["output_text"] == merged_output["output_text"],
                        "box_identical": reference_output["predicted_box"] == merged_output["predicted_box"],
                    }
                )
    mismatches = [row for row in results if not row["text_identical"]]
    return {
        "status": "completed" if not mismatches else "merge_failed",
        "passed": not mismatches,
        "artifact": project_relative(artifact),
        "adapter": project_relative(adapter),
        "dataset": project_relative(dataset),
        "samples": len(results),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "rows": results,
        "elapsed_seconds": time.perf_counter() - started,
        "contract": "base+adapter and merged checkpoint must produce byte-identical output_text for the same image/question",
        "caveat": "5 fixed diagnostic samples, not an accuracy evaluation",
    }


# ------------------------------------------------------------------------ CLI
def run_verify_only(args: argparse.Namespace, report_path: Path, artifact: Path) -> dict[str, Any]:
    verification = verify_merged_directory(artifact, dtype=args.dtype)
    payload = {
        "status": "verified" if verification["ok"] else "verification_failed",
        "mode": "verify_only",
        "started_at": now_iso(),
        "verification": verification,
        "provisional": True,
    }
    write_json_atomic(report_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = resolve_project_path(args.model_path)
    adapter_path = resolve_project_path(args.adapter)
    output_path = resolve_project_path(args.output)
    report_path = checked_results_path(args.report, "deploy_")

    artifact = resolve_project_path(args.artifact) if args.artifact else output_path

    if args.verify_equivalence:
        payload = verify_merge_equivalence(
            artifact=artifact, adapter=adapter_path, model_path=model_path, samples=args.samples
        )
        payload["started_at"] = now_iso()
        write_json_atomic(report_path, payload)
        print(json.dumps({"status": payload["status"], "passed": payload["passed"], "report": project_relative(report_path)}, ensure_ascii=False, indent=2))
        return 0 if payload["passed"] else 2

    if args.verify_only:
        payload = run_verify_only(args, report_path, artifact)
        print(json.dumps({"status": payload["status"], "report": project_relative(report_path)}, ensure_ascii=False, indent=2))
        return 0 if payload["verification"]["ok"] else 2

    plan = validate_merge_inputs(model_path, adapter_path, output_path, dtype=args.dtype)
    plan.update(
        {
            "mode": "merge",
            "dry_run": bool(args.dry_run),
            "required_files_after_merge": list(MERGED_REQUIRED_FILES),
            "follow_up": [
                "python -m src.deploy.merge_lora --verify-only --artifact <merged> --report results/deploy_merge_verify.json",
                "python -m src.deploy.merge_lora --verify-equivalence --artifact <merged> --report results/deploy_merge_equivalence.json",
            ],
        }
    )
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", **plan}, ensure_ascii=False, indent=2, default=str))
        return 0

    merge_report = merge_and_save(model_path, adapter_path, output_path, dtype=args.dtype, force=args.force)
    merge_report.update({"started_at": plan.get("started_at") or now_iso(), "plan": plan})
    verification = verify_merged_directory(output_path, dtype=args.dtype)
    merge_report["verification"] = verification
    merge_report["status"] = "completed" if verification["ok"] else "verification_failed"
    write_json_atomic(report_path, merge_report)
    print(
        json.dumps(
            {
                "status": merge_report["status"],
                "artifact": merge_report["artifact"],
                "elapsed_seconds": merge_report["elapsed_seconds"],
                "verification_ok": verification["ok"],
                "report": project_relative(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if merge_report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
