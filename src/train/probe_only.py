"""Inference-only diagnostic probe for an already-trained SFT adapter.

Purpose: re-measure the T5 100-record overfit diagnostic with a sufficient
``max_new_tokens`` (64) after the first probe was accidentally configured with
16 tokens and truncated every answer mid-box.  This script is strictly
inference-only:

* loads ``adapters/<run>/adapter`` with ``PeftModel.from_pretrained(..., is_trainable=False)``
* calls only ``model.eval()`` + ``torch.inference_mode()`` + ``model.generate(...)``
* never computes a training loss, never calls ``backward``, ``optimizer.step``,
  ``scheduler.step`` or ``optimizer.zero_grad``, never calls ``model.train()``
* never writes to the adapter, data, ground truth or evaluator

The output is explicitly labelled ``diagnostic_only=true`` and
``official_eval=false``.  It is not the project's official G3 metric; that is
eval-keeper's independent job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    PROJECT_ROOT,
    config_fingerprint,
    load_config,
    project_relative,
    resolve_project_path,
    sha256_file,
)
from .model_audit import configure_processor
from .sft_data import (
    EOS_TOKEN_ID,
    build_probe_inputs,
    load_manifest_records,
    load_split_records,
    load_train_records,
)
from .vision_tokens import compute_image_seq_len

RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
LOG_ROOT = (PROJECT_ROOT / "logs" / "train").resolve()
LEASE_RE = __import__("re").compile(r"^GPU-LEASE-[A-Za-z0-9-]+$")


def checked_results_path(value: str | Path) -> Path:
    path = resolve_project_path(value)
    if path.parent != RESULTS_ROOT or not path.name.startswith("train_"):
        raise ValueError(f"output must be directly under results/ and start with 'train_': {path}")
    return path


def write_result_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def nvidia_smi_memory_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one GPU memory row, got {completed.stdout!r}")
    return int(lines[0])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_identity(
    *,
    config_path: Path,
    adapter_dir: Path,
    training_summary_path: Path,
    training_results_path: Path,
) -> dict[str, Any]:
    """Verify the adapter belongs to the recorded training run before inference."""

    config = load_config(config_path)
    current_fingerprint = config["config_fingerprint"]
    summary = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    training_results = json.loads(Path(training_results_path).read_text(encoding="utf-8"))
    adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    if not (Path(adapter_dir) / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(Path(adapter_dir) / "adapter_model.safetensors")

    checks = {
        "summary_config_fingerprint_matches_config": summary.get("config_fingerprint") == current_fingerprint,
        "training_results_config_fingerprint_matches_config": (
            training_results.get("config_fingerprint") == current_fingerprint
        ),
        "summary_and_results_fingerprints_equal": (
            summary.get("config_fingerprint") == training_results.get("config_fingerprint")
        ),
        "seed_is_42": int(config["seed"]) == 42 and int(summary.get("seed", -1)) == 42,
        "lora_r_is_16": int(adapter_config.get("r")) == 16,
        "lora_alpha_is_32": int(adapter_config.get("lora_alpha")) == 32,
        "task_type_is_causal_lm": adapter_config.get("task_type") == "CAUSAL_LM",
        "adapter_is_for_causal_lm": adapter_config.get("peft_type") == "LORA",
        "model_path_matches": summary.get("model", {}).get("path") == config["model"]["path"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"adapter identity mismatch: {failed}")
    if int(adapter_config.get("r")) != int(config["lora"]["r"]):
        raise RuntimeError("adapter LoRA rank differs from the config")
    return {
        "checks": checks,
        "config_fingerprint": current_fingerprint,
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "training_summary_config_fingerprint": summary.get("config_fingerprint"),
        "training_results_config_fingerprint": training_results.get("config_fingerprint"),
        "adapter_config_sha256": sha256_file(Path(adapter_dir) / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(Path(adapter_dir) / "adapter_model.safetensors"),
        "adapter_saved_target_modules_count": len(adapter_config.get("target_modules", [])),
        "adapter_config_note": (
            "PEFT 0.21 persisted a truncated-looking target_modules list in adapter_config.json (91 entries), "
            "but PeftModel.from_pretrained resolves all 120 text decoder q/k/v/o LoRA modules from the 240 "
            "adapter weight tensors; this is recorded as a display/serialisation anomaly, not a failed load."
        ),
    }


def build_probe_plan(
    *,
    config_path: Path,
    adapter_dir: Path,
    training_summary_path: Path,
    training_results_path: Path,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """CPU-only plan/identity validation; no model weights are loaded."""

    config = load_config(config_path)
    identity = validate_identity(
        config_path=config_path,
        adapter_dir=adapter_dir,
        training_summary_path=training_summary_path,
        training_results_path=training_results_path,
    )
    records, train_evidence = load_train_records(config)
    if limit <= 0 or limit > len(records):
        raise ValueError(f"limit must be in [1, {len(records)}], got {limit}")
    sample = records[:limit]
    return {
        "status": "preflight_passed",
        "gpu_used": False,
        "device": "cpu",
        "diagnostic_only": True,
        "official_eval": False,
        "run_id": config["run_name"],
        "adapter_dir": project_relative(Path(adapter_dir)),
        "training_summary": project_relative(Path(training_summary_path)),
        "training_results": project_relative(Path(training_results_path)),
        "identity": identity,
        "records_available": len(records),
        "probe_records": len(sample),
        "probe_record_ids": [record["id"] for record in sample],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "image_seq_len": compute_image_seq_len(
            int(config["image"]["image_size"]),
            int(config["image"]["patch_size"]),
            int(config["image"]["scale_factor"]),
        ),
        "train_dataset_evidence": train_evidence,
        "inference_only_contract": {
            "loads_adapter_with_is_trainable_false": True,
            "calls_model_eval": True,
            "calls_torch_inference_mode": True,
            "computes_training_loss": False,
            "calls_backward": False,
            "calls_optimizer_step": False,
            "calls_scheduler_step": False,
            "calls_zero_grad": False,
            "writes_adapter": False,
            "modifies_data": False,
            "modifies_evaluator": False,
        },
    }


def run_probe(
    *,
    config_path: Path,
    adapter_dir: Path,
    training_summary_path: Path,
    training_results_path: Path,
    output_path: Path,
    gpu_log_path: Path,
    limit: int,
    max_new_tokens: int,
    device: str,
    gpu_lease_id: str | None,
) -> dict[str, Any]:
    if device not in ("cpu", "cuda"):
        raise SystemExit(f"unsupported device {device!r}")
    if device == "cuda":
        if not gpu_lease_id or not LEASE_RE.fullmatch(gpu_lease_id):
            raise SystemExit("CUDA inference requires --gpu-lease-id GPU-LEASE-... issued by the Lead")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.eval.grounding import metrics_from_scores, score_prediction

    plan = build_probe_plan(
        config_path=config_path,
        adapter_dir=adapter_dir,
        training_summary_path=training_summary_path,
        training_results_path=training_results_path,
        limit=limit,
        max_new_tokens=max_new_tokens,
    )
    config = load_config(config_path)
    model_path = resolve_project_path(config["model"]["path"])
    records, _ = load_train_records(config)
    sample = records[:limit]
    model = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        memory_before = nvidia_smi_memory_mib() if device == "cuda" else None
        if memory_before is not None and memory_before > 7500:
            raise RuntimeError(f"GPU memory before probe is {memory_before} MiB > 7500 MiB; aborting")
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # The 384 px no-split contract must be configured before any image is
        # processed; otherwise the processor default (splitting + 512 tiles) is
        # used and the 36-placeholder visual-token assertion rightly fails.
        processor_geometry = configure_processor(processor, image_size=int(config["image"]["image_size"]))
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to(device)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model.eval()
        if model.training:
            raise RuntimeError("probe-only model is unexpectedly in training mode")
        lora_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
        if not lora_modules:
            raise RuntimeError("adapter did not attach any LoRA modules")
        if device == "cuda":
            torch.cuda.synchronize()
            memory_after_load = nvidia_smi_memory_mib()
            if memory_after_load > 7500:
                raise RuntimeError(f"GPU memory after adapter load is {memory_after_load} MiB > 7500 MiB; aborting")
        else:
            memory_after_load = None

        log_path = Path(gpu_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        details: list[dict[str, Any]] = []
        scores = []
        for record in sample:
            inputs = build_probe_inputs(
                record,
                processor,
                model_path=model_path,
                image_size=int(config["image"]["image_size"]),
                patch_size=int(config["image"]["patch_size"]),
                scale_factor=int(config["image"]["scale_factor"]),
            )
            prompt_tokens = int(inputs["prompt_tokens"])
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    pixel_values=inputs["pixel_values"].to(device),
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                )
            generated_ids = generated[0, prompt_tokens:]
            generated_tokens = int(generated_ids.shape[0])
            eos_reached = int(EOS_TOKEN_ID) in generated_ids.tolist()
            hit_max = generated_tokens >= int(max_new_tokens)
            raw_output = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=False)[0]
            output_text = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
            score = score_prediction(output_text, record["answer"])
            scores.append(score)
            details.append(
                {
                    "id": record["id"],
                    "image": record["image"],
                    "question": record["question"],
                    "ground_truth": record["answer"],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "max_new_tokens": int(max_new_tokens),
                    "hit_max_new_tokens": hit_max,
                    "eos_reached": eos_reached,
                    "truncated": hit_max and not eos_reached,
                    "raw_output": raw_output,
                    "output_text": output_text,
                    "predicted_box": list(score.predicted_box) if score.predicted_box is not None else None,
                    "parse_error": score.parse_error,
                    "iou": score.iou,
                    "correct@0.5": score.correct_at_0_5,
                }
            )
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "probe_sample", **details[-1]}, ensure_ascii=False) + "\n")
        metrics = metrics_from_scores(scores)
        elapsed = time.perf_counter() - started
        if device == "cuda":
            torch.cuda.synchronize()
            memory_after_generation = nvidia_smi_memory_mib()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        else:
            memory_after_generation = None
            peak_allocated = None
            peak_reserved = None
        payload = {
            **plan,
            "status": "completed",
            "adapter_dir": project_relative(Path(adapter_dir)),
            "n": len(sample),
            "metrics": metrics,
            "details": details,
            "truncation_audit": {
                "max_new_tokens": int(max_new_tokens),
                "samples_hit_max_new_tokens": sum(1 for item in details if item["hit_max_new_tokens"]),
                "samples_eos_reached": sum(1 for item in details if item["eos_reached"]),
                "samples_truncated": sum(1 for item in details if item["truncated"]),
            },
            "lora_module_count_loaded": len(lora_modules),
            "processor_geometry": processor_geometry,
            "runtime": {
                "device": device,
                "dtype": "bfloat16" if device == "cuda" else "float32",
                "gpu_lease_id": gpu_lease_id,
                "nvidia_smi_memory_before_mib": memory_before,
                "nvidia_smi_memory_after_adapter_load_mib": memory_after_load,
                "nvidia_smi_memory_after_generation_mib": memory_after_generation,
                "torch_peak_memory_allocated_mib": peak_allocated,
                "torch_peak_memory_reserved_mib": peak_reserved,
                "elapsed_seconds": elapsed,
            },
            "source_sha256": {
                "src/train/probe_only.py": sha256_file(Path(__file__)),
                "src/train/sft_data.py": sha256_file(PROJECT_ROOT / "src/train/sft_data.py"),
                "src/train/config.py": sha256_file(PROJECT_ROOT / "src/train/config.py"),
                "src/eval/grounding.py": sha256_file(PROJECT_ROOT / "src/eval/grounding.py"),
            },
            "note": (
                "Diagnostic-only re-measurement of the T5 100-record overfit probe with a sufficient "
                "max_new_tokens. Not the official evaluation and not a G3 decision; eval-keeper produces "
                "official metrics."
            ),
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        }
        write_result_new(output_path, payload)
        return payload
    finally:
        if model is not None:
            del model
        if device == "cuda":
            gc = __import__("gc")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default="adapters/sft_overfit100/adapter")
    parser.add_argument("--training-summary", default="adapters/sft_overfit100/training_summary.json")
    parser.add_argument("--training-results", default="results/train_sft_overfit100_s42_training.json")
    parser.add_argument("--output", default="results/train_sft_overfit100_s42_probe_maxtok64.json")
    parser.add_argument("--gpu-log", default="logs/train/sft_overfit100_probe_gpu.log")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-lease-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--limit and --max-new-tokens must be positive")
    config_path = resolve_project_path(args.config)
    adapter_dir = resolve_project_path(args.adapter_dir)
    training_summary_path = resolve_project_path(args.training_summary)
    training_results_path = resolve_project_path(args.training_results)
    if args.dry_run:
        if args.device == "cuda":
            raise SystemExit("--dry-run is CPU-only")
        plan = build_probe_plan(
            config_path=config_path,
            adapter_dir=adapter_dir,
            training_summary_path=training_summary_path,
            training_results_path=training_results_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    output_path = checked_results_path(args.output)
    gpu_log_path = resolve_project_path(args.gpu_log)
    payload = run_probe(
        config_path=config_path,
        adapter_dir=adapter_dir,
        training_summary_path=training_summary_path,
        training_results_path=training_results_path,
        output_path=output_path,
        gpu_log_path=gpu_log_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        gpu_lease_id=args.gpu_lease_id,
    )
    print(json.dumps({
        "status": payload["status"],
        "output": project_relative(output_path),
        "n": payload["n"],
        "metrics": payload["metrics"],
        "truncation_audit": payload["truncation_audit"],
        "runtime": payload["runtime"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
