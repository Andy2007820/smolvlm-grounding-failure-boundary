"""Reproducible LoRA SFT for SmolVLM-256M-Instruct grounding.

Contract (enforced by ``configs/*.json`` and ``src/train/config.py``):
LoRA r=16 / alpha=32, lr=1e-4, batch 1 x gradient accumulation 8 (effective 8),
384 px no-split single view, bf16, frozen vision tower, seed 42, gradient
checkpointing, explicit answer EOS=49279, pad=2, and a training-time assertion
that the train/validation ID intersection is exactly zero.

Safety:
* ``--dry-run`` performs a CPU-only preflight (processor, data, visual-token
  contract, label masking, plan) and never loads model weights or CUDA.
* Real training requires ``--device cuda`` plus ``--gpu-lease-id`` issued by the
  Team Lead; a plain CPU training start is refused during the T5 phase.
* nvidia-smi is recorded before the process, after model load, after the first
  ``--profile-steps`` optimizer steps, and after cleanup; >7500 MiB aborts.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import checkpoint as checkpoint_utils
from .accumulation import run_true_gradient_accumulation
from .coordinate_variants import score_prediction_variant
from .config import (
    PROJECT_ROOT,
    FIXED_EFFECTIVE_BATCH,
    load_config,
    project_relative,
    resolve_project_path,
    sha256_file,
)
from .model_audit import (
    DEFAULT_MODEL_PATH,
    audit_model_geometry,
    audit_pad_eos,
    configure_processor,
)
from .sft_data import (
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_MAX_LENGTH,
    EOS_TOKEN_ID,
    IMAGE_TOKEN_ID,
    PAD_TOKEN_ID,
    SftDataset,
    assert_train_val_disjoint,
    build_probe_inputs,
    build_sft_example,
    digest_sequence,
    epoch_order,
    format_answer,
    load_manifest_records,
    load_split_records,
    load_train_records,
    project_relative as data_project_relative,
)
from .vision_tokens import compute_image_seq_len

RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
LEASE_RE = re.compile(r"^GPU-LEASE-[A-Za-z0-9-]+$")
LORA_TARGET_RE = re.compile(r"^model\.text_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")

MODEL_FILE_NAMES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
)
SOURCE_FILE_PATHS = (
    "src/train/__init__.py",
    "src/train/config.py",
    "src/train/vision_tokens.py",
    "src/train/sft_data.py",
    "src/train/checkpoint.py",
    "src/train/accumulation.py",
    "src/train/coordinate_variants.py",
    "src/train/model_audit.py",
    "src/train/train_sft.py",
    "src/eval/grounding.py",
    "src/eval/prompts.py",
    "src/eval/dataset.py",
    "src/eval/run_zeroshot.py",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return checkpoint_utils.now_iso()


def checked_results_path(value: str | Path) -> Path:
    path = resolve_project_path(value)
    if path.parent != RESULTS_ROOT:
        raise ValueError(f"output must be directly under results/: {path}")
    if not path.name.startswith("train_"):
        raise ValueError(f"result file must start with 'train_': {path}")
    return path


def write_result_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    checkpoint_utils.write_json_atomic(path, payload)


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


def torch_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    for package in ("transformers", "peft", "torch", "accelerate"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    environment["gpu_name"] = torch_gpu_name()
    return environment


def collect_model_files(model_path: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in MODEL_FILE_NAMES:
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"missing model file: {path}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def collect_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILE_PATHS:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"missing source file: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def collect_val_sets(config: dict[str, Any], *, with_images: bool) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    sets: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for entry in config["data"]["val"]:
        if with_images:
            rows, audit = load_split_records(entry)
        else:
            path = resolve_project_path(entry["path"])
            rows = load_manifest_records(path)
            audit = {
                "path": data_project_relative(path),
                "source": entry["source"],
                "split": entry["split"],
                "limit": entry.get("limit"),
                "rows_loaded": len(rows),
                "sha256": sha256_file(path),
            }
        sets[audit["path"]] = rows
        evidence.append(audit)
    return sets, evidence


def collect_full_train_sets(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    sets: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for entry in config["data"]["train"]:
        path = resolve_project_path(entry["path"])
        rows = load_manifest_records(path)
        sets[data_project_relative(path)] = rows
        evidence.append(
            {
                "path": data_project_relative(path),
                "source": entry["source"],
                "split": entry["split"],
                "limit": entry.get("limit"),
                "rows_total": len(rows),
                "sha256": sha256_file(path),
            }
        )
    return sets, evidence


def expected_lora_target_modules(
    num_layers: int = 30, leaves: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")
) -> list[str]:
    return [
        f"model.text_model.layers.{layer}.self_attn.{leaf}"
        for layer in range(num_layers)
        for leaf in leaves
    ]


def lora_targets_match(discovered: Sequence[str], expected: Sequence[str]) -> bool:
    """Compare LoRA target module names as sets; discovery order is not semantic."""

    return sorted(discovered) == sorted(expected)


def discover_lora_target_modules(model: Any) -> list[str]:
    """Find q/k/v/o projections under the text decoder only (never the vision tower)."""

    import torch.nn as nn

    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and LORA_TARGET_RE.match(name):
            names.append(name)
    names.sort(key=lambda name: (int(LORA_TARGET_RE.match(name).group(1)), LORA_TARGET_RE.match(name).group(2)))
    if not names:
        raise RuntimeError("no text-model q/k/v/o projection modules found for LoRA")
    if any("vision_model" in name for name in names):
        raise RuntimeError("refusing to attach LoRA to the frozen vision tower")
    return names


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def assert_gpu_lease_allowed(*, device: str, lease_id: str | None) -> None:
    """Refuse CUDA work unless the Lead issued an explicit lease id."""

    if device not in ("cpu", "cuda"):
        raise SystemExit(f"unsupported device: {device!r}")
    if device == "cuda":
        if not lease_id:
            raise SystemExit(
                "CUDA execution is blocked: pass --gpu-lease-id GPU-LEASE-... issued by the Lead. "
                "No GPU load or training may start without an explicit lease."
            )
        if not LEASE_RE.fullmatch(str(lease_id)):
            raise SystemExit(f"invalid GPU lease id {lease_id!r}; expected GPU-LEASE-<...>")


def assert_execution_mode(*, device: str, lease_id: str | None, dry_run: bool) -> None:
    if dry_run:
        if device == "cuda":
            raise SystemExit("--dry-run is CPU-only; do not touch the GPU during the T5 CPU phase")
        return
    if device == "cpu":
        raise SystemExit(
            "refusing to start a CPU training run in the T5 phase; use --dry-run for the CPU "
            "preflight, or --device cuda with a Lead GPU lease for real training"
        )
    assert_gpu_lease_allowed(device=device, lease_id=lease_id)


# ---------------------------------------------------------------------------
# Planning / preflight
# ---------------------------------------------------------------------------


def compute_plan(config: dict[str, Any], records: Sequence[dict[str, Any]], *, max_steps: int | None = None) -> dict[str, Any]:
    training = config["training"]
    effective = int(training["effective_batch_size"])
    epochs = int(training["num_epochs"])
    total_micro_batches = len(records) * epochs
    total_updates = total_micro_batches // effective + (1 if total_micro_batches % effective else 0)
    if max_steps is not None:
        total_updates = min(total_updates, int(max_steps))
        total_micro_batches = min(total_micro_batches, total_updates * effective)
    save_every = int(training["save_every_steps"])
    checkpoint_steps = list(range(save_every, total_updates + 1, save_every))
    if total_updates and (not checkpoint_steps or checkpoint_steps[-1] != total_updates):
        checkpoint_steps.append(total_updates)
    return {
        "train_records": len(records),
        "num_epochs": epochs,
        "batch_size": int(training["batch_size"]),
        "micro_batch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "micro_batches_per_optimizer_step": int(training["gradient_accumulation_steps"]),
        "true_accumulation": True,
        "micro_forwards_total": total_micro_batches,
        "loss_scale_per_micro_batch": 1.0 / float(training["gradient_accumulation_steps"]),
        "effective_batch_size": effective,
        "total_micro_batches": total_micro_batches,
        "total_optimizer_steps": total_updates,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_every_steps": save_every,
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "max_steps_override": max_steps,
    }


def build_train_order_digest(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> str:
    indices: list[int] = []
    for epoch in range(int(config["training"]["num_epochs"])):
        indices.extend(epoch_order(len(records), int(config["seed"]), epoch))
    ids = [records[index]["id"] for index in indices]
    return digest_sequence(ids)


def dry_run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only preflight: no model weights, no CUDA, no training."""

    model_path = resolve_project_path(config["model"]["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    geometry = audit_model_geometry(model_path=model_path)
    image_cfg = config["image"]
    if int(geometry["vision_patch_size"]) != int(image_cfg["patch_size"]):
        raise SystemExit("model patch_size disagrees with config")
    if int(geometry["scale_factor"]) != int(image_cfg["scale_factor"]):
        raise SystemExit("model scale_factor disagrees with config")
    if int(geometry["image_token_id"]) != IMAGE_TOKEN_ID:
        raise SystemExit("model image_token_id disagrees with the hard-coded contract")
    expected_image_seq_len = compute_image_seq_len(
        int(image_cfg["image_size"]), int(image_cfg["patch_size"]), int(image_cfg["scale_factor"])
    )
    if expected_image_seq_len != int(image_cfg["expected_image_seq_len"]):
        raise SystemExit("config expected_image_seq_len disagrees with the derived value")

    train_records, train_evidence = load_train_records(config)
    full_train_sets, full_train_evidence = collect_full_train_sets(config)
    val_sets, val_evidence = collect_val_sets(config, with_images=False)
    id_audit = assert_train_val_disjoint(full_train_sets, val_sets)
    plan = compute_plan(config, train_records, max_steps=args.max_steps)
    selected_ids_sha256 = digest_sequence([record["id"] for record in train_records])
    train_order_sha256 = build_train_order_digest(train_records, config)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor_geometry = configure_processor(processor, image_size=int(image_cfg["image_size"]))
    pad_eos_audit = audit_pad_eos(processor, model_path=model_path)
    if int(pad_eos_audit["processor_config_default_image_seq_len"]) != 64:
        raise SystemExit("processor default image_seq_len is not the shipped 64")

    # A single-record learnability control cannot itself contain both a square
    # and a rectangular image; in that case run the visual-token preflight on the
    # full source manifest (read-only) so both geometries are still covered.
    preflight_records = train_records
    preflight_source = "selected_train_records"
    has_square = any(record["width"] == record["height"] for record in train_records)
    has_rect = any(record["width"] != record["height"] for record in train_records)
    if not has_square or not has_rect:
        first_train_path = config["data"]["train"][0]["path"]
        preflight_records = load_manifest_records(first_train_path)
        preflight_source = "full_train_manifest_fallback_for_single_sample_preflight"
    dataset = SftDataset(
        preflight_records,
        processor,
        model_path=model_path,
        image_size=int(image_cfg["image_size"]),
        patch_size=int(image_cfg["patch_size"]),
        scale_factor=int(image_cfg["scale_factor"]),
        answer_precision=int(config["answer"]["precision"]),
        answer_format=str(config["answer"]["format"]),
        representation=str(config["answer"].get("representation", "normalized_decimal")),
    )
    observations: list[dict[str, Any]] = []
    square_index = next((i for i, r in enumerate(preflight_records) if r["width"] == r["height"]), None)
    rect_index = next((i for i, r in enumerate(preflight_records) if r["width"] != r["height"]), None)
    if square_index is None or rect_index is None:
        raise SystemExit("preflight source must include both square and rectangular images")
    for index in (square_index, rect_index):
        example = dataset.build_example(index)
        observations.append(example.audit)
        if example.audit["visual_token_contract"]["total_views"] * example.audit["visual_token_contract"]["tokens_per_view"] != example.audit["visual_token_contract"]["placeholder_count"]:
            raise SystemExit("visual-token assertion failed during preflight")

    source_hashes = collect_source_hashes()
    plan_payload = {
        "status": "preflight_passed",
        "gpu_used": False,
        "device": "cpu",
        "dry_run": True,
        "run_name": config["run_name"],
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "config_fingerprint": config["config_fingerprint"],
        "seed": config["seed"],
        "model": {"path": project_relative(model_path), "geometry": geometry, "files": collect_model_files(model_path)},
        "processor_geometry": processor_geometry,
        "pad_eos_audit": pad_eos_audit,
        "plan": plan,
        "training_semantics": {
            "true_accumulation": True,
            "micro_batch_size": 1,
            "micro_batches_per_optimizer_step": int(config["training"]["gradient_accumulation_steps"]),
            "loss_scaling": "loss / gradient_accumulation_steps before backward",
            "optimizer_step_order": "clip_grad_norm -> optimizer.step -> scheduler.step -> optimizer.zero_grad",
            "micro_forwards_total": plan["micro_forwards_total"],
            "optimizer_steps_total": plan["total_optimizer_steps"],
            "checkpoint_every_optimizer_steps": plan["checkpoint_every_steps"],
        },
        "selected_ids_sha256": selected_ids_sha256,
        "train_order_sha256": train_order_sha256,
        "train_dataset_evidence": train_evidence,
        "full_train_manifest_evidence": full_train_evidence,
        "val_manifest_evidence": val_evidence,
        "id_intersection": id_audit,
        "preflight_observations": observations,
        "preflight_source": preflight_source,
        "preflight_records_used": len(preflight_records),
        "answer_contract": {
            "format": config["answer"]["format"],
            "precision": config["answer"]["precision"],
            "eos_token_id": config["answer"]["eos_token_id"],
            "pad_token_id": config["answer"]["pad_token_id"],
        },
        "environment": collect_environment(),
        "source_hashes": source_hashes,
        "execution_requirements": {
            "gpu_lease_required": True,
            "exact_gpu_command": (
                f"{sys.executable} -m src.train.train_sft --config {config['_config_path']} "
                "--device cuda --gpu-lease-id <GPU-LEASE-ID>"
            ),
            "nvidia_smi_logging": (
                "shell wrapper records `nvidia-smi --query-gpu=memory.used --format=csv` before process "
                "start and after process exit to logs/train/<run>_gpu.log; the script records before-load, "
                "post-profile, post-training and post-cleanup values in its JSON outputs"
            ),
            "abort_conditions": [
                "nvidia-smi memory.used > 7500 MiB",
                "any competing game/training process",
                "CUDA assert or non-finite loss",
            ],
        },
        "generated_at": now_iso(),
    }
    if args.plan_out:
        output_path = checked_results_path(args.plan_out)
        write_result_new(output_path, plan_payload)
        plan_payload["written_to"] = project_relative(output_path)
    return plan_payload


# ---------------------------------------------------------------------------
# Real training
# ---------------------------------------------------------------------------


def build_lora_model(model: Any, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Freeze the vision tower, attach text-only LoRA, and audit parameters."""

    import torch
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["lora"]
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    vision_names = [name for name, _ in model.named_parameters() if "vision_model" in name]
    for name, parameter in model.named_parameters():
        if "vision_model" in name:
            parameter.requires_grad = False
    targets = discover_lora_target_modules(model)
    expected_targets = expected_lora_target_modules(
        num_layers=int(model.config.text_config.num_hidden_layers), leaves=tuple(lora_cfg["target_modules"])
    )
    if not lora_targets_match(targets, expected_targets):
        missing = sorted(set(expected_targets) - set(targets))
        raise RuntimeError(f"LoRA target discovery mismatch; missing={missing[:5]} extra={sorted(set(targets) - set(expected_targets))[:5]}")
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=targets,
        bias=str(lora_cfg["bias"]),
        task_type=str(lora_cfg["task_type"]),
    )
    peft_model = get_peft_model(model, peft_config)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()

    total_parameters = sum(parameter.numel() for parameter in peft_model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    frozen_vision_parameters = sum(
        parameter.numel()
        for name, parameter in peft_model.named_parameters()
        if "vision_model" in name and not parameter.requires_grad
    )
    trainable_vision_parameters = sum(
        parameter.numel()
        for name, parameter in peft_model.named_parameters()
        if "vision_model" in name and parameter.requires_grad
    )
    if trainable_vision_parameters != 0:
        raise RuntimeError("vision tower is not fully frozen")
    if frozen_vision_parameters == 0:
        raise RuntimeError("vision tower parameters were not found")
    if trainable_parameters / total_parameters >= 0.01:
        raise RuntimeError(
            f"trainable parameter ratio {trainable_parameters / total_parameters:.4f} is not LoRA-like (<1%)"
        )
    audit = {
        "lora_r": int(lora_cfg["r"]),
        "lora_alpha": int(lora_cfg["lora_alpha"]),
        "lora_dropout": float(lora_cfg["lora_dropout"]),
        "target_module_count": len(targets),
        "target_modules_sha256": checkpoint_utils.sha256_json(targets),
        "target_modules_prefix": "model.text_model.layers.{i}.self_attn.{q,k,v,o}_proj",
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_ratio": trainable_parameters / total_parameters,
        "frozen_vision_parameters": frozen_vision_parameters,
        "trainable_vision_parameters": trainable_vision_parameters,
        "vision_frozen": trainable_vision_parameters == 0,
        "gradient_checkpointing": True,
        "use_cache": False,
    }
    return peft_model, audit


def build_optimizer_scheduler(model: Any, config: dict[str, Any], total_steps: int) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    training = config["training"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = max(1, int(total_steps * float(training["warmup_ratio"])))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, {
        "optimizer": "AdamW",
        "scheduler": "cosine_with_linear_warmup",
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "betas": list(optimizer.defaults.get("betas", ())),
        "eps": optimizer.defaults.get("eps"),
    }


def load_model_for_training(config: dict[str, Any], *, device: str) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForImageTextToText

    model_path = resolve_project_path(config["model"]["path"])
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["model"]["dtype"]]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    )
    geometry = audit_model_geometry(model_path=model_path)
    if int(model.config.vision_config.patch_size) != int(geometry["vision_patch_size"]):
        raise RuntimeError("loaded model patch size differs from config.json")
    if int(model.config.scale_factor) != int(geometry["scale_factor"]):
        raise RuntimeError("loaded model scale factor differs from config.json")
    model.to(device)
    return model


def resolve_resume_checkpoint(output_dir: Path, resume_argument: str | None) -> Path | None:
    if resume_argument is None:
        return None
    if resume_argument == "latest":
        candidate = checkpoint_utils.find_latest_checkpoint(output_dir)
        if candidate is None:
            raise FileNotFoundError(f"no complete checkpoint under {output_dir}")
        return candidate
    candidate = resolve_project_path(resume_argument)
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    metadata_path = candidate / "checkpoint_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return candidate


def make_micro_stream(records: Sequence[dict[str, Any]], config: dict[str, Any]):
    for epoch in range(int(config["training"]["num_epochs"])):
        for index in epoch_order(len(records), int(config["seed"]), epoch):
            yield epoch, index


def run_training(config: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel

    model_path = resolve_project_path(config["model"]["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    image_cfg = config["image"]
    training_cfg = config["training"]
    model_files = collect_model_files(model_path)
    source_hashes = collect_source_hashes()
    environment = collect_environment()
    output_dir = resolve_project_path(config["output"]["dir"])
    log_path = resolve_project_path(config["output"]["log"])
    results_summary_path = checked_results_path(f"results/train_{config['run_name']}_training.json")
    probe_results_path = checked_results_path(f"results/train_{config['run_name']}_probe.json")

    train_records, train_evidence = load_train_records(config)
    full_train_sets, _ = collect_full_train_sets(config)
    val_sets, val_evidence = collect_val_sets(config, with_images=False)
    id_audit = assert_train_val_disjoint(full_train_sets, val_sets)
    plan = compute_plan(config, train_records, max_steps=args.max_steps)
    selected_ids_sha256 = digest_sequence([record["id"] for record in train_records])
    train_order_sha256 = build_train_order_digest(train_records, config)
    expected_image_seq_len = compute_image_seq_len(
        int(image_cfg["image_size"]), int(image_cfg["patch_size"]), int(image_cfg["scale_factor"])
    )

    seed_everything(int(config["seed"]))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor_geometry = configure_processor(processor, image_size=int(image_cfg["image_size"]))
    pad_eos_audit = audit_pad_eos(processor, model_path=model_path)
    if int(pad_eos_audit["processor_config_default_image_seq_len"]) != 64:
        raise RuntimeError("processor default image_seq_len is not the shipped 64")

    device = args.device
    nvidia_before = nvidia_smi_memory_mib() if device == "cuda" else None
    if nvidia_before is not None and nvidia_before > 7500:
        raise RuntimeError(f"GPU memory before training is {nvidia_before} MiB > 7500 MiB; aborting")
    started = time.perf_counter()
    started_at = now_iso()

    model = load_model_for_training(config, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
        nvidia_after_load = nvidia_smi_memory_mib()
        if nvidia_after_load > 7500:
            raise RuntimeError(f"GPU memory after model load is {nvidia_after_load} MiB > 7500 MiB; aborting")
        torch.cuda.reset_peak_memory_stats()
    else:  # pragma: no cover - real training is CUDA-only
        nvidia_after_load = None

    peft_model, parameter_audit = build_lora_model(model, config)
    optimizer, scheduler, optimizer_audit = build_optimizer_scheduler(peft_model, config, plan["total_optimizer_steps"])
    accumulation_contract = {
        "true_accumulation": True,
        "micro_batch_size": int(training_cfg["batch_size"]),
        "micro_batches_per_optimizer_step": int(training_cfg["gradient_accumulation_steps"]),
        "micro_forwards_are_batch_1": True,
        "loss_scaling": "loss / gradient_accumulation_steps before backward",
        "optimizer_step_order": "clip_grad_norm -> optimizer.step -> scheduler.step -> optimizer.zero_grad",
        "steps_per_optimizer_step": {
            "forward": int(training_cfg["gradient_accumulation_steps"]),
            "backward": int(training_cfg["gradient_accumulation_steps"]),
            "clip_grad_norm": 1,
            "optimizer_step": 1,
            "scheduler_step": 1,
            "zero_grad": 1,
        },
        "samples_seen_increments_per": "micro-batch (batch=1)",
        "global_step_increments_per": "optimizer step",
    }
    schedule_plan = {
        "total_steps": plan["total_optimizer_steps"],
        **optimizer_audit,
        "gradient_accumulation": accumulation_contract,
    }

    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume)
    start_step = 0
    start_samples = 0
    resume_metadata: dict[str, Any] | None = None
    resume_adapter_audit: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_metadata = checkpoint_utils.read_json(resume_checkpoint / "checkpoint_meta.json")
        checkpoint_utils.validate_resume_metadata(
            resume_metadata,
            config_fingerprint=config["config_fingerprint"],
            seed=int(config["seed"]),
            model_files=model_files,
            dataset_evidence=train_evidence,
            image_seq_len=expected_image_seq_len,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )
        # A resume must restore the LoRA weights too.  The previous implementation
        # only restored optimizer/scheduler/RNG state and silently continued from
        # a freshly initialised adapter (found during T7-D seed42 recovery).
        adapter_state_path = resume_checkpoint / "adapter" / "adapter_model.safetensors"
        if not adapter_state_path.is_file():
            raise FileNotFoundError(f"resume checkpoint lacks adapter weights: {adapter_state_path}")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_state = load_file(str(adapter_state_path))
        if not adapter_state:
            raise RuntimeError(f"resume adapter state is empty: {adapter_state_path}")
        load_result = set_peft_model_state_dict(peft_model, adapter_state)
        unexpected = list(load_result.get("unexpected_keys") or []) if isinstance(load_result, dict) else []
        missing = list(load_result.get("missing_keys") or []) if isinstance(load_result, dict) else []
        if unexpected or missing:
            raise RuntimeError(
                "resume adapter load incomplete: "
                f"unexpected={unexpected[:5]} missing={missing[:5]} (total unexpected={len(unexpected)})"
            )
        resume_adapter_audit = {
            "path": project_relative(adapter_state_path),
            "sha256": checkpoint_utils.sha256_file(adapter_state_path),
            "tensors": len(adapter_state),
            "unexpected_keys": unexpected,
            "missing_keys": missing,
        }
        trainer_state = checkpoint_utils.load_trainer_state(resume_checkpoint)
        optimizer.load_state_dict(trainer_state["optimizer"])
        if trainer_state.get("scheduler") is not None:
            scheduler.load_state_dict(trainer_state["scheduler"])
        checkpoint_utils.restore_rng_state(trainer_state["rng"])
        start_step = int(resume_metadata["step"])
        start_samples = int(resume_metadata["samples_seen"])
        if start_step > plan["total_optimizer_steps"]:
            raise RuntimeError(
                f"checkpoint step {start_step} exceeds planned total steps {plan['total_optimizer_steps']}"
            )

    dataset = SftDataset(
        train_records,
        processor,
        model_path=model_path,
        image_size=int(image_cfg["image_size"]),
        patch_size=int(image_cfg["patch_size"]),
        scale_factor=int(image_cfg["scale_factor"]),
        answer_precision=int(config["answer"]["precision"]),
        answer_format=str(config["answer"]["format"]),
        representation=str(config["answer"].get("representation", "normalized_decimal")),
        cache_max=int(config["data"]["example_cache_max"]),
    )
    if len(train_records) <= int(config["data"]["example_cache_max"]):
        dataset.precompute()

    from .sft_data import collate_examples

    access = "a" if resume_checkpoint is not None else "w"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_history: list[float] = []
    step_records: list[dict[str, Any]] = []
    profile_steps = max(1, int(args.profile_steps))
    telemetry: dict[str, Any] = {
        "nvidia_smi_memory_before_mib": nvidia_before,
        "nvidia_smi_memory_after_model_load_mib": nvidia_after_load,
        "profile_step": profile_steps,
        "profile_recorded": False,
    }
    global_step = start_step
    samples_seen = start_samples
    stop_reason = "completed"
    model.train()

    def build_metadata(step: int, epoch: int, samples: int) -> dict[str, Any]:
        gpu_payload: dict[str, Any] = {
            "device": device,
            "gpu_name": environment.get("gpu_name"),
            "lease_id": args.gpu_lease_id,
            "nvidia_smi_memory_before_mib": nvidia_before,
            "nvidia_smi_memory_at_checkpoint_mib": nvidia_smi_memory_mib() if device == "cuda" else None,
            "torch_peak_memory_allocated_mib": (
                torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else None
            ),
            "torch_peak_memory_reserved_mib": (
                torch.cuda.max_memory_reserved() / (1024 * 1024) if device == "cuda" else None
            ),
        }
        return checkpoint_utils.build_checkpoint_metadata(
            run_id=config["run_name"],
            step=step,
            epoch=epoch,
            samples_seen=samples,
            config=config,
            config_fingerprint=config["config_fingerprint"],
            seed=int(config["seed"]),
            model_path=project_relative(model_path),
            model_files=model_files,
            dataset_evidence=train_evidence,
            selected_ids_sha256=selected_ids_sha256,
            train_order_sha256=train_order_sha256,
            id_intersection=id_audit,
            lora=parameter_audit,
            trainable_parameters=parameter_audit["trainable_parameters"],
            total_parameters=parameter_audit["total_parameters"],
            frozen_vision_parameters=parameter_audit["frozen_vision_parameters"],
            trainable_vision_parameters=parameter_audit["trainable_vision_parameters"],
            image_contract={
                "image_size": int(image_cfg["image_size"]),
                "image_seq_len": expected_image_seq_len,
                "do_image_splitting": False,
                "patch_size": int(image_cfg["patch_size"]),
                "scale_factor": int(image_cfg["scale_factor"]),
                "processor_config_default_image_seq_len": 64,
                "processor_geometry": processor_geometry,
            },
            answer_contract={
                "format": config["answer"]["format"],
                "precision": int(config["answer"]["precision"]),
                "eos_token_id": EOS_TOKEN_ID,
                "pad_token_id": PAD_TOKEN_ID,
                "eos_explicit": True,
                "evidence": pad_eos_audit,
            },
            training=schedule_plan,
            gpu=gpu_payload,
            environment=environment,
            source_hashes=source_hashes,
            resumed_from=(project_relative(resume_checkpoint) if resume_checkpoint is not None else None),
            resume_count=(int(resume_metadata.get("resume_count", 0)) + 1 if resume_metadata else 0),
        )

    with log_path.open(access, encoding="utf-8", newline="\n") as log_handle:
        def log_event(event: dict[str, Any]) -> None:
            log_handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            log_handle.flush()

        log_event(
            {
                "event": "start",
                "run_id": config["run_name"],
                "config_fingerprint": config["config_fingerprint"],
                "resumed_from": project_relative(resume_checkpoint) if resume_checkpoint is not None else None,
                "start_step": start_step,
                "plan": plan,
                "id_intersection": id_audit,
                "pad_eos_audit": pad_eos_audit,
                "parameter_audit": parameter_audit,
                "started_at": started_at,
            }
        )

        if resume_adapter_audit is not None:
            log_event({"event": "resume_adapter_loaded", **resume_adapter_audit})

        model.train()

        def micro_batch_iterator():
            consumed = 0
            for epoch, index in make_micro_stream(train_records, config):
                if consumed < start_samples:
                    # Replay of already-consumed micro-batches after --resume.
                    consumed += 1
                    continue
                if consumed >= plan["total_micro_batches"]:
                    return
                example = dataset.build_example(index)
                batch = collate_examples([example])
                if int(batch["input_ids"].shape[0]) != int(training_cfg["batch_size"]):
                    raise RuntimeError(
                        "true accumulation requires one sample per micro-batch; "
                        f"got collated batch={int(batch['input_ids'].shape[0])}"
                    )
                consumed += 1
                yield epoch, batch

        def on_optimizer_step(record) -> None:
            step = record.optimizer_step
            samples = record.samples_seen
            loss_history.append(record.loss_mean_of_micro_losses)
            step_record = {
                "step": step,
                "epoch": record.epoch,
                "samples_seen": samples,
                "micro_batches_in_step": record.micro_batches_in_step,
                "loss": record.loss_mean_of_micro_losses,
                "learning_rate": record.learning_rate,
                "grad_norm": record.grad_norm,
                "partial_accumulation": record.partial_accumulation,
                "elapsed_seconds": time.perf_counter() - started,
            }
            step_records.append(step_record)
            log_event({"event": "step", **step_record})
            if not telemetry["profile_recorded"] and step >= profile_steps:
                if device == "cuda":
                    torch.cuda.synchronize()
                telemetry["profile_recorded"] = True
                telemetry.update(
                    {
                        "profile_step": step,
                        "profile_elapsed_seconds": time.perf_counter() - started,
                        "seconds_per_optimizer_step": (time.perf_counter() - started)
                        / max(1, step - start_step),
                        "nvidia_smi_memory_at_profile_mib": (
                            nvidia_smi_memory_mib() if device == "cuda" else None
                        ),
                        "torch_peak_allocated_mib": (
                            torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else None
                        ),
                        "torch_peak_reserved_mib": (
                            torch.cuda.max_memory_reserved() / (1024 * 1024) if device == "cuda" else None
                        ),
                    }
                )
                telemetry["projected_total_seconds"] = (
                    telemetry["seconds_per_optimizer_step"] * plan["total_optimizer_steps"]
                )
                telemetry["projected_remaining_seconds"] = telemetry["projected_total_seconds"] - (
                    time.perf_counter() - started
                )
                log_event({"event": "profile", **telemetry})
            if step % int(training_cfg["save_every_steps"]) == 0:
                checkpoint_path = checkpoint_utils.save_checkpoint(
                    output_dir,
                    step=step,
                    model=peft_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=build_metadata(step, record.epoch, samples),
                )
                log_event({"event": "checkpoint", "step": step, "path": project_relative(checkpoint_path)})

        accumulation_result = run_true_gradient_accumulation(
            micro_batches=micro_batch_iterator(),
            model=peft_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            expected_micro_batch_size=int(training_cfg["batch_size"]),
            max_grad_norm=float(training_cfg["max_grad_norm"]),
            parameters=[parameter for parameter in peft_model.parameters() if parameter.requires_grad],
            initial_optimizer_step=start_step,
            initial_samples_seen=start_samples,
            on_optimizer_step=on_optimizer_step,
            should_stop=lambda step: step >= plan["total_optimizer_steps"],
        )
        global_step = start_step + accumulation_result.optimizer_steps
        samples_seen = start_samples + accumulation_result.micro_batches_seen
        stop_reason = accumulation_result.stop_reason
        log_event({"event": "accumulation_complete", **accumulation_result.as_dict()})

        if device == "cuda":
            torch.cuda.synchronize()
            nvidia_after_training = nvidia_smi_memory_mib()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        else:  # pragma: no cover
            nvidia_after_training = None
            peak_allocated = None
            peak_reserved = None

        final_epoch = (
            accumulation_result.records[-1].epoch
            if accumulation_result.records
            else max(0, int(config["training"]["num_epochs"]) - 1)
        )
        final_metadata = build_metadata(global_step, final_epoch, samples_seen)
        final_metadata["stop_reason"] = stop_reason
        final_metadata["loss_history"] = loss_history
        final_metadata["loss_history_sha256"] = checkpoint_utils.sha256_json(loss_history)
        final_metadata["training_seconds"] = time.perf_counter() - started
        final_metadata["nvidia_smi_memory_after_training_mib"] = nvidia_after_training
        final_metadata["torch_peak_memory_allocated_mib"] = peak_allocated
        final_metadata["torch_peak_memory_reserved_mib"] = peak_reserved
        final_metadata["telemetry"] = telemetry
        final_metadata["log_path"] = project_relative(log_path)
        final_metadata["plan"] = plan
        final_metadata["train_dataset_evidence"] = train_evidence
        final_metadata["val_manifest_evidence"] = val_evidence
        final_metadata["pad_eos_audit"] = pad_eos_audit
        final_metadata["processor_geometry"] = processor_geometry
        final_metadata["parameter_audit"] = parameter_audit
        final_metadata["optimizer_audit"] = optimizer_audit
        final_metadata["accumulation_contract"] = accumulation_contract
        final_metadata["accumulation_stats"] = accumulation_result.as_dict()
        final_metadata["micro_batches_seen"] = accumulation_result.micro_batches_seen
        final_metadata["forward_calls"] = accumulation_result.forward_calls
        final_metadata["backward_calls"] = accumulation_result.backward_calls
        final_metadata["partial_final_step"] = accumulation_result.partial_final_step
        final_metadata["started_at"] = started_at
        final_metadata["finished_at"] = now_iso()
        final_metadata["step_records"] = step_records
        final_metadata["stop_reason"] = stop_reason
        if resume_adapter_audit is not None:
            final_metadata["resume_adapter_load"] = resume_adapter_audit

        summary_path = checkpoint_utils.save_final_adapter(output_dir, peft_model, final_metadata)
        log_event({"event": "final_adapter", "path": project_relative(output_dir), "summary": project_relative(output_dir / "training_summary.json")})

        probe_payload: dict[str, Any] | None = None
        if not args.no_probe and int(config["probe"]["n"]) > 0:
            probe_payload = run_probe(
                peft_model,
                processor,
                config,
                device=device,
                model_path=model_path,
                log_event=log_event,
            )
            write_result_new(probe_results_path, probe_payload)
            log_event({"event": "probe", "path": project_relative(probe_results_path), "metrics": probe_payload["metrics"]})

        if device == "cuda":
            del peft_model
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            nvidia_after_cleanup = nvidia_smi_memory_mib()
        else:  # pragma: no cover
            nvidia_after_cleanup = None

        training_summary = dict(final_metadata)
        training_summary.update(
            {
                "nvidia_smi_memory_after_cleanup_mib": nvidia_after_cleanup,
                "probe_results_path": (project_relative(probe_results_path) if probe_payload else None),
                "probe_metrics_diagnostic_only": (probe_payload["metrics"] if probe_payload else None),
                "final_adapter_summary": summary_path,
                "official_eval_note": (
                    "probe metrics are training-time diagnostics computed with the frozen evaluator parser; "
                    "official metrics must be produced independently by eval-keeper"
                ),
            }
        )
        write_result_new(results_summary_path, training_summary)
        log_event({"event": "completed", "summary": project_relative(results_summary_path), **{
            key: training_summary[key] for key in (
                "nvidia_smi_memory_after_cleanup_mib", "probe_results_path", "stop_reason"
            )
        }})

    print(json.dumps({
        "status": "completed",
        "run_id": config["run_name"],
        "steps": global_step,
        "samples_seen": samples_seen,
        "output_dir": project_relative(output_dir),
        "training_summary": project_relative(results_summary_path),
        "probe_results": project_relative(probe_results_path) if probe_payload else None,
        "telemetry": telemetry,
        "stop_reason": stop_reason,
        "accumulation": accumulation_result.as_dict(),
    }, ensure_ascii=False, indent=2))


def run_probe(model: Any, processor: Any, config: dict[str, Any], *, device: str, model_path: Path, log_event) -> dict[str, Any]:
    """Training-time diagnostic generation on a few train/val records (not official eval)."""

    import torch

    from src.eval.grounding import metrics_from_scores, score_prediction

    probe_cfg = config["probe"]
    if probe_cfg["records"] == "train":
        records, _ = load_train_records(config)
    else:
        records = []
        for entry in config["data"]["val"]:
            rows, _ = load_split_records(entry)
            records.extend(rows)
    sample = records[: int(probe_cfg["n"])]
    details = []
    scores = []
    model.eval()
    started = time.perf_counter()
    for record in sample:
        representation = str(config["answer"].get("representation", "normalized_decimal"))
        inputs = build_probe_inputs(
            record,
            processor,
            model_path=model_path,
            image_size=int(config["image"]["image_size"]),
            patch_size=int(config["image"]["patch_size"]),
            scale_factor=int(config["image"]["scale_factor"]),
            representation=representation,
        )
        prompt_tokens = int(inputs["prompt_tokens"])
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                max_new_tokens=int(probe_cfg["max_new_tokens"]),
                do_sample=False,
                num_beams=1,
            )
        generated_ids = generated[0, prompt_tokens:]
        raw_output = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=False)[0]
        output_text = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
        score, conversion_audit = score_prediction_variant(output_text, record, representation)
        scores.append(score)
        details.append(
            {
                "id": record["id"],
                "image": record["image"],
                "question": record["question"],
                "ground_truth": record["answer"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated_ids.shape[0]),
                "raw_output": raw_output,
                "output_text": output_text,
                "predicted_box": list(score.predicted_box) if score.predicted_box is not None else None,
                "parse_error": score.parse_error,
                "representation": representation,
                "coordinate_conversion": conversion_audit,
                "iou": score.iou,
                "correct@0.5": score.correct_at_0_5,
                "visual_token_contract": inputs["visual_token_contract"],
            }
        )
    metrics = metrics_from_scores(scores)
    payload = {
        "run_id": config["run_name"],
        "config_fingerprint": config["config_fingerprint"],
        "probe_kind": "training_diagnostic",
        "official_eval": False,
        "note": (
            "Diagnostic overfitting check only: metrics are computed with the frozen evaluator parser but "
            "are not the project's official evaluation. eval-keeper produces the official numbers."
        ),
        "records": probe_cfg["records"],
        "n": len(sample),
        "metrics": metrics,
        "details": details,
        "elapsed_seconds": time.perf_counter() - started,
        "generated_at": now_iso(),
    }
    log_event({"event": "probe_detail", "n": len(sample), "metrics": metrics})
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to configs/*.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-lease-id", default=None, help="Lead-issued lease id; mandatory for CUDA")
    parser.add_argument("--dry-run", action="store_true", help="CPU-only preflight; never loads model weights")
    parser.add_argument("--plan-out", default=None, help="optional results/train_*.json destination for the dry-run plan")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="resume from a checkpoint directory (or the latest complete checkpoint)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="optional optimizer-step cap for lease profiling")
    parser.add_argument("--profile-steps", type=int, default=25, help="record throughput/memory after this many steps")
    parser.add_argument("--no-probe", action="store_true", help="skip the training-time diagnostic generation probe")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.profile_steps <= 0:
        raise SystemExit("--profile-steps must be positive")
    config = load_config(args.config)
    if not RUN_ID_RE.fullmatch(config["run_name"]):
        raise SystemExit(f"invalid run_name: {config['run_name']!r}")
    assert_execution_mode(device=args.device, lease_id=args.gpu_lease_id, dry_run=args.dry_run)
    if args.dry_run:
        plan = dry_run(config, args)
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    run_training(config, args)


if __name__ == "__main__":
    main()
