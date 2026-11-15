"""Run deterministic SmolVLM zero-shot grounding with resumable trace files.

No CUDA model is loaded in --dry-run mode. Actual CUDA execution requires a
Lead-issued GPU lease; the caller must also record nvidia-smi before process
start and after process exit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any

from .dataset import load_dataset, read_jsonl, sha256_file
from .grid_convert import score_prediction_grid
from .grounding import metrics_from_scores, score_prediction
from .pixel_convert import score_prediction_pixel
from .prompts import (
    DEFAULT_PROMPT_ID,
    PROMPT_VARIANTS,
    build_messages_for,
    prompt_sha256_for,
    prompt_template_for,
    representation_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
)
SOURCE_FILES = (
    "grounding.py",
    "dataset.py",
    "prompts.py",
    "pixel_convert.py",
    "grid_convert.py",
    "run_zeroshot.py",
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def compute_image_seq_len(image_size: int, patch_size: int, scale_factor: int) -> int:
    """Match Idefics3 square no-split pixels to its placeholder token count."""

    reduction = patch_size * scale_factor
    if image_size <= 0 or patch_size <= 0 or scale_factor <= 0 or image_size % reduction:
        raise ValueError(f"image_size={image_size} must be a positive multiple of {reduction}")
    return (image_size // reduction) ** 2


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def checked_result_path(value: str | Path, prefix: str) -> Path:
    path = resolve_path(value)
    if path.parent != RESULTS_ROOT or not path.name.startswith(prefix):
        raise ValueError(f"output must be directly under results/ and start with {prefix!r}: {path}")
    return path


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a final JSON, refusing to overwrite an earlier run."""

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary.replace(path)


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


def git_commit() -> tuple[str | None, str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"unavailable: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, "not_a_git_repository"
    return completed.stdout.strip(), "available"


def selected_ids_digest(records: list[dict[str, Any]]) -> str:
    serialized = json.dumps([record["id"] for record in records], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def fingerprint_config(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_adapter_metadata(adapter_path: Path, *, model_path: Path) -> dict[str, Any]:
    """Read-only PEFT adapter preflight: validate identity and hash every weight file.

    The evaluator is inference-only. ``is_trainable=False`` is recorded here and
    enforced again after ``PeftModel.from_pretrained`` loads the adapter.
    """

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
    missing = [name for name in ADAPTER_FILES if not (adapter_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"adapter directory is missing {missing}: {adapter_path}")
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(f"adapter peft_type must be LORA, got {adapter_config.get('peft_type')!r}")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"adapter task_type must be CAUSAL_LM, got {adapter_config.get('task_type')!r}")
    base_name = adapter_config.get("base_model_name_or_path")
    if not base_name:
        raise ValueError("adapter_config.json lacks base_model_name_or_path")
    recorded_base = Path(str(base_name))
    if not recorded_base.is_absolute():
        recorded_base = adapter_path / recorded_base
    if recorded_base.resolve() != model_path.resolve():
        raise ValueError(
            f"adapter base model {str(recorded_base)!r} does not match --model-path {str(model_path)!r}"
        )
    return {
        "path": project_relative(adapter_path),
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "adapter_model_bytes": (adapter_path / "adapter_model.safetensors").stat().st_size,
        "peft_type": adapter_config.get("peft_type"),
        "task_type": adapter_config.get("task_type"),
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_dropout": adapter_config.get("lora_dropout"),
        "base_model_name_or_path": str(base_name),
        "peft_version": importlib.metadata.version("peft"),
        "is_trainable": False,
    }


def normalise_eos_token_ids(value: Any) -> set[int]:
    """Return the generation stop-token ids as a plain int set."""

    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def trim_at_first_eos(token_ids: list[int], eos_ids: set[int]) -> list[int]:
    """Cut a padded batched continuation at (and including) its first EOS token.

    Batched greedy generation keeps a finished row alive by filling its slot
    with pad tokens until every row in the batch stops. Batch-size 1 stops at
    the first EOS, so the equivalent batched continuation is the prefix up to
    and including that EOS. Rows that never emit EOS are returned unchanged,
    which preserves the no-truncation contract at ``max_new_tokens``.
    """

    for position, token in enumerate(token_ids):
        if token in eos_ids:
            return token_ids[: position + 1]
    return token_ids


def score_output_for_prompt(
    prompt_id: str, output_text: Any, record: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Score one output with the representation-appropriate frozen metric path."""

    if prompt_id == "p1_pixel":
        return score_prediction_pixel(output_text, record)
    if prompt_id == "p1_grid100":
        return score_prediction_grid(output_text, record)
    return score_prediction(output_text, record["answer"]), {
        "representation": representation_for(prompt_id),
        "conversion_applied": False,
    }


def load_resume_rows(
    path: Path,
    *,
    run_id: str,
    config_fingerprint: str,
    dataset_name: str,
    split: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    if len(rows) > len(records):
        raise ValueError(f"resume has {len(rows)} rows but selected dataset has only {len(records)}")
    for index, (detail, record) in enumerate(zip(rows, records)):
        expected = {
            "run_id": run_id,
            "config_fingerprint": config_fingerprint,
            "dataset": dataset_name,
            "split": split,
            "index": index,
            "id": record["id"],
            "image": record["image"],
            "question": record["question"],
            "ground_truth": record["answer"],
        }
        for key, value in expected.items():
            if detail.get(key) != value:
                raise ValueError(
                    f"resume mismatch at row {index}, field {key}: {detail.get(key)!r} != {value!r}"
                )
        if not isinstance(detail.get("raw_output"), str) or not isinstance(detail.get("output_text"), str):
            raise ValueError(f"resume row {index} lacks raw/decoded output text")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    # ScreenSpot is enabled for a later, separately leased transfer run. Its
    # unified raw_bbox is already pixel xyxy derived once from normalized source.
    parser.add_argument("--dataset", required=True, choices=("synthetic", "refcoco", "screenspot"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--model-path", default="models/SmolVLM-256M-Instruct")
    parser.add_argument("--model-name", default="SmolVLM-256M-Instruct")
    parser.add_argument(
        "--adapter",
        default=None,
        help="optional read-only PEFT LoRA adapter directory; loaded with is_trainable=False",
    )
    parser.add_argument(
        "--prompt-id",
        default=DEFAULT_PROMPT_ID,
        choices=tuple(PROMPT_VARIANTS),
        help="prompt variant; p1=normalized decimal (default), p1_bin1000, p1_pixel, p1_grid100",
    )
    parser.add_argument("--expected-n", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "samples per greedy generation call; 1 keeps the historical "
            "one-sample path unchanged, >1 uses left-padded equivalent batching"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="optional torch CPU intra-op thread count (deployment diagnostics only)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="CPU-only data/model preflight; do not load torch")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("run_id may contain only letters, digits, underscore, dash and dot")
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer when provided")

    output_path = checked_result_path(args.output, "zeroshot_")
    details_path = checked_result_path(args.details_output, "eval_")
    data_path = resolve_path(args.data)
    model_path = resolve_path(args.model_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    adapter_path = resolve_path(args.adapter) if args.adapter else None
    adapter_metadata = (
        load_adapter_metadata(adapter_path, model_path=model_path)
        if adapter_path is not None
        else None
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    if details_path.exists() and not args.resume:
        raise FileExistsError(f"details exist; use --resume or choose a new run_id: {details_path}")

    records = load_dataset(
        data_path,
        project_root=PROJECT_ROOT,
        expected_source=args.dataset,
        split=args.split,
        expected_n=args.expected_n,
        limit=args.limit,
        check_images=True,
    )
    model_hashes = {}
    for name in MODEL_FILES:
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"missing model file: {path}")
        model_hashes[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    source_hashes = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in SOURCE_FILES
    }
    model_definition = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    vision_patch_size = int(model_definition["vision_config"]["patch_size"])
    vision_scale_factor = int(model_definition["scale_factor"])
    image_seq_len = compute_image_seq_len(args.image_size, vision_patch_size, vision_scale_factor)
    prompt_id = args.prompt_id
    prompt_template = prompt_template_for(prompt_id)
    prompt_sha256 = prompt_sha256_for(prompt_id)
    representation = representation_for(prompt_id)
    conversion_required = bool(PROMPT_VARIANTS[prompt_id]["conversion_required"])
    config: dict[str, Any] = {
        "image_size": args.image_size,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha256,
        "prompt_template": prompt_template,
        "representation": representation,
        "coordinate_conversion": {
            "applied": conversion_required,
            "rule": (
                "pixel xyxy integers validated against width/height, then divided by width/height; "
                "frozen iou/Acc@0.5/mIoU are unchanged"
                if prompt_id == "p1_pixel"
                else (
                    "grid100 integers in 0-100 validated (ordered, in range), then divided by 100; "
                    "frozen iou/Acc@0.5/mIoU are unchanged"
                    if prompt_id == "p1_grid100"
                    else "frozen parse_box path unchanged"
                )
            ),
        },
        "seed": args.seed,
        "do_sample": False,
        "num_beams": 1,
        "decoding": "deterministic_greedy",
        "device": args.device,
        "dtype": "bfloat16" if args.device == "cuda" else "float32",
        "do_image_splitting": False,
        "vision_patch_size": vision_patch_size,
        "vision_scale_factor": vision_scale_factor,
        "image_seq_len": image_seq_len,
        "image_processor_size": {"longest_edge": args.image_size},
        "image_processor_max_image_size": {"longest_edge": args.image_size},
        "model_path": project_relative(model_path),
        "model_files": model_hashes,
        "adapter": adapter_metadata,
        "dataset_path": project_relative(data_path),
        "dataset_sha256": sha256_file(data_path),
        "selected_ids_sha256": selected_ids_digest(records),
        "source_sha256": source_hashes,
        "local_files_only": True,
    }
    config_fingerprint = fingerprint_config(config)
    config["config_fingerprint"] = config_fingerprint
    resumed = load_resume_rows(
        details_path,
        run_id=args.run_id,
        config_fingerprint=config_fingerprint,
        dataset_name=args.dataset,
        split=args.split,
        records=records,
    )
    preflight = {
        "status": "preflight_passed",
        "dry_run": bool(args.dry_run),
        "run_id": args.run_id,
        "dataset": args.dataset,
        "split": args.split,
        "n": len(records),
        "resume_rows": len(resumed),
        "dataset_sha256": config["dataset_sha256"],
        "model_sha256": model_hashes["model.safetensors"]["sha256"],
        "image_seq_len": image_seq_len,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "prompt_id": prompt_id,
        "representation": representation,
        "prompt_sha256": prompt_sha256,
        "coordinate_conversion_required": conversion_required,
        "config_fingerprint": config_fingerprint,
        "adapter": adapter_metadata["path"] if adapter_metadata else None,
        "adapter_model_sha256": adapter_metadata["adapter_model_sha256"] if adapter_metadata else None,
        "adapter_is_trainable": False if adapter_metadata else None,
        "gpu_used": False,
    }
    if args.dry_run:
        if resumed:
            raise ValueError("--dry-run does not accept an existing resume trace")
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if len(resumed) == len(records):
        raise ValueError("trace already contains all selected rows; run validate_results instead of reloading the model")

    # Offline flags are set before importing transformers. No network fallback
    # can silently change the model or processor revision.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    memory_before = nvidia_smi_memory_mib() if args.device == "cuda" else None
    if memory_before is not None and memory_before > 7500:
        raise RuntimeError(f"GPU memory before evaluation is {memory_before} MiB > 7500 MiB; aborting")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    started = time.perf_counter()
    started_at = now_iso()
    model = None
    scores = [
        score_output_for_prompt(
            prompt_id,
            row["output_text"],
            {"answer": row["ground_truth"], "width": row["width"], "height": row["height"]},
        )[0]
        for row in resumed
    ]
    memory_after_load = None
    memory_after_generation = None
    memory_after_cleanup = None
    peak_allocated = None
    peak_reserved = None
    model_class = None
    adapter_load: dict[str, Any] | None = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": args.image_size}
        processor.image_processor.max_image_size = {"longest_edge": args.image_size}
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        ).to(args.device)
        base_model_class = type(model).__name__
        base_config = model.config
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                is_trainable=False,
                torch_device=args.device,
            )
            lora_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
            if not lora_modules:
                raise RuntimeError("adapter attached no LoRA modules")
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            if trainable:
                raise RuntimeError(
                    f"adapter model has {len(trainable)} trainable parameters; "
                    "is_trainable=False must freeze every parameter"
                )
            adapter_load = {
                "loaded": True,
                "is_trainable": False,
                "torch_device": args.device,
                "lora_module_count": len(lora_modules),
                "trainable_parameter_count": 0,
                "base_model_class": base_model_class,
                "wrapped_model_class": type(model).__name__,
                "adapter_metadata": adapter_metadata,
            }
            if model.training:
                raise RuntimeError("adapter model is unexpectedly in training mode before eval()")
        model.eval()
        model_class = type(model).__name__
        if (
            int(base_config.vision_config.patch_size) != vision_patch_size
            or int(base_config.scale_factor) != vision_scale_factor
        ):
            raise RuntimeError("loaded model vision patch/scale differs from preflight config")
        # Idefics3's no-split path emits square max_image_size tiles. At 384,
        # 24x24 patches followed by 4x4 pixel shuffle yield exactly 36 tokens;
        # the default 64 is only correct for 512 tiles and otherwise crashes.
        processor.image_seq_len = image_seq_len
        if args.device == "cuda":
            torch.cuda.synchronize()
            memory_after_load = nvidia_smi_memory_mib()
            if memory_after_load > 7500:
                raise RuntimeError(
                    f"GPU memory after model/adapter load is {memory_after_load} MiB > 7500 MiB; aborting"
                )
            torch.cuda.reset_peak_memory_stats()

        mode = "a" if args.resume else "x"
        generation_config = getattr(model, "generation_config", None)
        eos_token_ids = normalise_eos_token_ids(getattr(generation_config, "eos_token_id", None))
        if not eos_token_ids:
            eos_token_ids = normalise_eos_token_ids(getattr(base_config, "eos_token_id", None))
        if args.batch_size > 1 and not eos_token_ids:
            raise RuntimeError("cannot determine the EOS token id required for batched slicing")
        with details_path.open(mode, encoding="utf-8", newline="\n") as detail_handle:

            def emit_sample(
                *,
                index: int,
                record: dict[str, Any],
                input_tokens: int,
                generated_ids: Any,
                elapsed_seconds: float,
            ) -> None:
                """Write one sample exactly as the batch-size-1 path always has."""

                raw_output = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=False)[0]
                output_text = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
                score, conversion_audit = score_output_for_prompt(prompt_id, output_text, record)
                scores.append(score)
                generated_tokens = int(generated_ids.shape[0])
                detail = {
                    "run_id": args.run_id,
                    "config_fingerprint": config_fingerprint,
                    "dataset": args.dataset,
                    "split": args.split,
                    "index": index,
                    "id": record["id"],
                    "image": record["image"],
                    "question": record["question"],
                    "ground_truth": record["answer"],
                    "raw_bbox": record["raw_bbox"],
                    "width": record["width"],
                    "height": record["height"],
                    "prompt_id": prompt_id,
                    "representation": representation,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "hit_max_new_tokens": generated_tokens == args.max_new_tokens,
                    "raw_output": raw_output,
                    "output_text": output_text,
                    "predicted_box": list(score.predicted_box) if score.predicted_box is not None else None,
                    "parse_error": score.parse_error,
                    "iou": score.iou,
                    "correct@0.5": score.correct_at_0_5,
                    "pixel_conversion": conversion_audit if prompt_id == "p1_pixel" else None,
                    "pixel_box": conversion_audit.get("pixel_box") if prompt_id == "p1_pixel" else None,
                    "grid_conversion": conversion_audit if prompt_id == "p1_grid100" else None,
                    "grid_box": conversion_audit.get("grid_box") if prompt_id == "p1_grid100" else None,
                    "elapsed_seconds_diagnostic": elapsed_seconds,
                    "batch_size": args.batch_size,
                }
                detail_handle.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
                detail_handle.flush()

            def report_progress(completed: int) -> None:
                if (
                    completed == 1
                    or completed % 25 == 0
                    or completed == len(records)
                    or completed - args.batch_size < 25 <= completed
                ):
                    running = metrics_from_scores(scores)
                    print(json.dumps({
                        "run_id": args.run_id,
                        "completed": completed,
                        "n": len(records),
                        "elapsed_seconds": time.perf_counter() - started,
                        "running_metrics": running,
                    }, ensure_ascii=False), flush=True)

            index = len(resumed)
            while index < len(records):
                if args.batch_size == 1:
                    record = records[index]
                    sample_started = time.perf_counter()
                    with Image.open(PROJECT_ROOT / record["image"]) as opened:
                        image = opened.convert("RGB")
                    prompt = processor.apply_chat_template(
                        build_messages_for(prompt_id, record["question"]), add_generation_prompt=True
                    )
                    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(args.device)
                    input_tokens = int(inputs["input_ids"].shape[1])
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                        )
                    generated_ids = generated[0, input_tokens:]
                    emit_sample(
                        index=index,
                        record=record,
                        input_tokens=input_tokens,
                        generated_ids=generated_ids,
                        elapsed_seconds=time.perf_counter() - sample_started,
                    )
                    index += 1
                    report_progress(index)
                    del inputs, generated, generated_ids, image
                else:
                    batch_records = records[index:index + args.batch_size]
                    sample_started = time.perf_counter()
                    images = []
                    for record in batch_records:
                        with Image.open(PROJECT_ROOT / record["image"]) as opened:
                            images.append(opened.convert("RGB"))
                    prompts = [
                        processor.apply_chat_template(
                            build_messages_for(prompt_id, record["question"]), add_generation_prompt=True
                        )
                        for record in batch_records
                    ]
                    # Equivalence contract: left padding keeps every real prompt
                    # at the right edge, so generated continuations always start
                    # at the padded length and each row's own prompt length is
                    # recovered from its attention mask.
                    processor.tokenizer.padding_side = "left"
                    inputs = processor(
                        text=prompts,
                        images=[[image] for image in images],
                        padding=True,
                        return_tensors="pt",
                    ).to(args.device)
                    prompt_lengths = [
                        int(length) for length in inputs["attention_mask"].sum(dim=1).tolist()
                    ]
                    padded_length = int(inputs["input_ids"].shape[1])
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                        )
                    batch_elapsed = time.perf_counter() - sample_started
                    for offset, record in enumerate(batch_records):
                        row_generated = generated[offset, padded_length:]
                        trimmed = trim_at_first_eos(
                            [int(token) for token in row_generated.tolist()],
                            eos_token_ids,
                        )
                        generated_ids = torch.tensor(
                            trimmed,
                            dtype=row_generated.dtype,
                            device=row_generated.device,
                        )
                        emit_sample(
                            index=index + offset,
                            record=record,
                            input_tokens=prompt_lengths[offset],
                            generated_ids=generated_ids,
                            elapsed_seconds=batch_elapsed / len(batch_records),
                        )
                        report_progress(index + offset + 1)
                    index += len(batch_records)
                    del inputs, generated, images
        if args.device == "cuda":
            torch.cuda.synchronize()
            memory_after_generation = nvidia_smi_memory_mib()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    finally:
        if model is not None:
            del model
        gc.collect()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_after_cleanup = nvidia_smi_memory_mib()

    metrics = metrics_from_scores(scores)
    commit, commit_status = git_commit()
    runtime = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds_including_load": time.perf_counter() - started,
        "nvidia_smi_memory_before_mib": memory_before,
        "nvidia_smi_memory_after_model_load_mib": memory_after_load,
        "nvidia_smi_memory_after_generation_mib": memory_after_generation,
        "torch_peak_memory_allocated_mib": peak_allocated,
        "torch_peak_memory_reserved_mib": peak_reserved,
        "nvidia_smi_memory_after_cleanup_mib": memory_after_cleanup,
        "cpu_threads_requested": args.cpu_threads,
        "torch_num_threads_effective": torch.get_num_threads(),
        "post_process_exit_check": "recorded by the calling shell, not claimed by this in-process value",
        "timing_caveat": "Diagnostic single-run wall time, not the Table 3 warmed latency benchmark.",
    }
    summary = {
        "run_id": args.run_id,
        "model": args.model_name,
        "adapter": adapter_metadata["path"] if adapter_metadata else None,
        "dataset": args.dataset,
        "split": args.split,
        "n": len(records),
        "batch_size": args.batch_size,
        "prompt_id": prompt_id,
        "representation": representation,
        "metrics": metrics,
        "config": config,
        "git_commit": commit,
        "git_commit_status": commit_status,
        "timestamp": now_iso(),
        "status": "completed",
        "details_path": project_relative(details_path),
        "details_sha256": sha256_file(details_path),
        "details_rows": len(scores),
        "model_class": model_class,
        "adapter_load": adapter_load,
        "batch_contract": {
            "batch_size": args.batch_size,
            "left_padding": args.batch_size > 1,
            "generated_start_index": "padded input length",
            "per_sample_prompt_length": "attention mask sum",
            "eos_trim_inclusive": args.batch_size > 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "coordinate_contract": {
            "prompt_id": prompt_id,
            "representation": representation,
            "prompt_sha256": prompt_sha256,
            "conversion_required": conversion_required,
            "conversion_rule": config["coordinate_conversion"]["rule"],
            "frozen_metric_code_unchanged": True,
            "invalid_predictions_remain_in_n": True,
        },
        "aggregate_contract": "invalid predictions remain in n; Acc@0.5 uses IoU>=0.5; invalid IoU=0; mIoU and parse_rate use n",
        "ground_truth_contract": "data/GT errors abort with dataset/split/id; synthetic occlusion boxes are pre-occlusion full-object GT",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_threads_requested": args.cpu_threads,
        },
        "runtime": runtime,
    }
    write_json_new(output_path, summary)
    print(json.dumps({
        "status": "completed",
        "run_id": args.run_id,
        "output": project_relative(output_path),
        "details": project_relative(details_path),
        "n": len(records),
        "batch_size": args.batch_size,
        "prompt_id": prompt_id,
        "representation": representation,
        "adapter": summary["adapter"],
        "adapter_load": adapter_load,
        "metrics": metrics,
        "runtime": runtime,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
