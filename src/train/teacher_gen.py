#!/usr/bin/env python3
"""T11 (P3): 3B teacher generation for response distillation (KD).

For every selected prompt the teacher (Qwen2.5-VL-3B-Instruct) is asked with the
**frozen student prompt** (``src/eval/prompts.PROMPT_TEMPLATE``, prompt id ``p1``)
and answers a normalized ``[x1, y1, x2, y2]`` box.  The answer is parsed and
scored with the frozen evaluator (:mod:`src.eval.grounding`) against the GT box;
rows with ``IoU >= tau`` are the KD training pool.

This is *response* distillation, not logit KD: teacher and student have different
vocabularies (paper §6.5), so only the teacher's text output can be transferred.

Outputs (never overwrite existing files):
  * ``data/kd/kd_train.jsonl``            KD pool, 8-field pool schema, source="kd"
  * ``data/kd/kd_teacher_audit.jsonl``    one row per generation (IoU / raw text)
  * ``data/kd/kd_teacher_progress.jsonl`` resume bookkeeping
  * ``results/train_kd_teacher_v1.json``  summary: teacher, params, memory, stats

Usage::

    python -m src.train.teacher_gen --dry-run          # CPU preflight, no weights
    python -m src.train.teacher_gen --device cuda --load-mode 4bit \\
        --gpu-lease-id GPU-LEASE-KD-TEACHER-... --per-dataset 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config, project_relative, resolve_project_path
from .probe_only import nvidia_smi_memory_mib, write_result_new
from .rest_sample import dataset_of, select_records
from .sft_data import load_train_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAU = 0.5
TEACHER_FORMAT_SUFFIX = (
    "\nOutput ONLY the box as [x1, y1, x2, y2] with four numbers between 0 and 1, "
    "no JSON, no labels, no extra words."
)

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval import grounding  # noqa: E402
from eval.prompts import PROMPT_ID, PROMPT_SHA256, PROMPT_TEMPLATE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T11 3B teacher generation (response distillation)")
    parser.add_argument("--config", default="configs/rest_r1.json", help="prompt 来源配置（复用 r1/r2 同一批 prompt）")
    parser.add_argument("--teacher-path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--per-dataset", type=int, default=500)
    parser.add_argument("--datasets", default="synthetic,refcoco")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--load-mode", choices=("bf16", "4bit", "offload"), default="4bit")
    parser.add_argument("--gpu-lease-id", default=None)
    parser.add_argument("--output-data", default="data/kd/kd_train.jsonl")
    parser.add_argument("--audit-output", default="data/kd/kd_teacher_audit.jsonl")
    parser.add_argument("--progress-output", default="data/kd/kd_teacher_progress.jsonl")
    parser.add_argument("--output-summary", default="results/train_kd_teacher_v1.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def checked_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_done_ids(progress_path: Path) -> set[str]:
    done: set[str] = set()
    if not progress_path.is_file():
        return done
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = payload.get("id")
            if isinstance(record_id, str):
                done.add(record_id)
    return done


def build_pool_row(record: dict[str, Any], box: list[float]) -> dict[str, Any]:
    width, height = int(record["width"]), int(record["height"])
    return {
        "id": f"{record['id']}#teacher",
        "image": record["image"],
        "question": record["question"],
        "answer": [round(float(value), 4) for value in box],
        "raw_bbox": [float(value) * width for value in box[:2]] + [float(value) * height for value in box[2:]],
        "width": width,
        "height": height,
        "source": "kd",
    }


_NUMBER_TOKEN = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_BRACKET_GROUP = r"\[([^\[\]]*)\]"


def extract_raw_box(text: Any) -> list[float] | None:
    """Pull exactly one four-number bracketed group out of the teacher's answer."""
    if not isinstance(text, str):
        return None
    for group in re.findall(_BRACKET_GROUP, text):
        values = [float(token) for token in re.findall(_NUMBER_TOKEN, group)]
        if len(values) == 4:
            return values
    values = [float(token) for token in re.findall(_NUMBER_TOKEN, text)]
    if len(values) == 4:
        return values
    return None


def normalize_teacher_box(raw: list[float], width: int, height: int) -> tuple[list[float] | None, str]:
    """Convert the teacher's raw box to normalized xyxy.

    Qwen2.5-VL answers in **absolute pixels of the original image** (verified on
    the T11 smoke: raw [423,265,597,476] on a 640x488 image matches its GT box).
    A deterministic rule resolves the ambiguity with the 0-1000 convention: if the
    largest coordinate exceeds the image extent the answer must be on the 0-1000
    scale, otherwise it is absolute pixels.  The chosen convention is recorded in
    the audit for every row.
    """

    if not raw or any(value != value for value in raw):
        return None, "unparseable"
    if max(raw) <= 1.0:
        box = [min(1.0, max(0.0, value)) for value in raw]
        convention = "already_normalized_0_1"
    elif max(raw) > max(width, height):
        box = [min(1.0, max(0.0, value / 1000.0)) for value in raw]
        convention = "relative_0_1000"
    else:
        box = [
            min(1.0, max(0.0, raw[0] / width)),
            min(1.0, max(0.0, raw[1] / height)),
            min(1.0, max(0.0, raw[2] / width)),
            min(1.0, max(0.0, raw[3] / height)),
        ]
        convention = "absolute_pixels_original_image"
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None, f"{convention}_degenerate"
    return box, convention


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    records, evidence = load_train_records(config)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    sample = select_records(records, limit=None, per_dataset=args.per_dataset, datasets=datasets)
    if not sample:
        raise SystemExit("no prompts selected")
    teacher_path = resolve_project_path(args.teacher_path)
    if not (teacher_path / "config.json").is_file():
        raise SystemExit(f"teacher model not found: {teacher_path}")
    if args.device == "cuda" and not args.gpu_lease_id:
        raise SystemExit("CUDA teacher generation requires --gpu-lease-id issued by the Lead")
    return {
        "status": "preflight_passed",
        "gpu_used": False,
        "mode": "dry_run" if args.dry_run else "planned",
        "teacher_path": project_relative(teacher_path),
        "load_mode": args.load_mode,
        "prompts_selected": len(sample),
        "datasets": datasets,
        "prompt_id": PROMPT_ID,
        "prompt_sha256": PROMPT_SHA256,
        "tau": args.tau,
        "max_new_tokens": args.max_new_tokens,
        "train_evidence": [
            {"path": item.get("path"), "rows_loaded": item.get("rows_loaded"), "sha256": (item.get("sha256") or "")[:16]}
            for item in evidence
        ],
        "outputs": {
            "data": args.output_data,
            "audit": args.audit_output,
            "progress": args.progress_output,
            "summary": args.output_summary,
        },
    }


def load_teacher(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load the teacher without AutoProcessor.

    ``AutoProcessor`` for Qwen2.5-VL composes ``Qwen2VLVideoProcessor``, which
    hard-requires ``torchvision`` (not installed in this environment).  We only
    need image inputs, so we pair the tokenizer with ``Qwen2VLImageProcessor``
    directly — the chat template still emits the ``<|vision_start|>...<|image_pad|>
    ...<|vision_end|>`` placeholders, and the image processor yields
    ``pixel_values`` + ``image_grid_thw`` exactly as the composed processor would.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen2VLImageProcessor  # noqa: PLC0415

    teacher_path = str(resolve_project_path(args.teacher_path))
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(teacher_path, local_files_only=True)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if args.load_mode == "4bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif args.load_mode == "offload":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
        kwargs["offload_folder"] = str(PROJECT_ROOT / "artifacts" / "tmp" / "teacher_offload")
    else:
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(teacher_path, **kwargs)
    model.eval()
    merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
    image_token_id = int(model.config.image_token_id)
    return model, (tokenizer, image_processor, image_token_id, merge_size)


def generate_one(model: Any, processor: Any, image: Any, question: str, max_new_tokens: int) -> str:
    import torch  # noqa: PLC0415

    tokenizer, image_processor, image_token_id, merge_size = processor
    # The frozen student prompt (prompts.PROMPT_TEMPLATE) is kept verbatim; the
    # teacher additionally gets a format-only suffix because Qwen2.5-VL otherwise
    # answers with its native JSON grounding form ({"bbox_2d": [...]}), which the
    # student's answer contract cannot absorb.  The student is still trained on the
    # frozen prompt -> teacher-box pairs, so evaluation semantics are untouched.
    prompt_text = PROMPT_TEMPLATE.format(question=question) + TEACHER_FORMAT_SUFFIX
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image_inputs = image_processor(images=[image], return_tensors="pt")
    grid = image_inputs["image_grid_thw"]
    # Qwen2.5-VL merges merge_size x merge_size patches into one visual token, so the
    # single <|image_pad|> placeholder in the chat text must be expanded to
    # grid_t * grid_h * grid_w / merge_size**2 copies (the composed Qwen processor
    # normally does this; we bypass it because its video processor needs torchvision).
    visual_tokens = int(grid.prod().item()) // (merge_size * merge_size)
    ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    expanded: list[int] = []
    for token in ids:
        if token == image_token_id:
            expanded.extend([image_token_id] * visual_tokens)
        else:
            expanded.append(token)
    device = model.device
    inputs = {
        "input_ids": torch.tensor([expanded], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(expanded)), dtype=torch.long, device=device),
        "pixel_values": image_inputs["pixel_values"].to(device),
        "image_grid_thw": grid.to(device),
    }
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image  # noqa: PLC0415
    import torch  # noqa: PLC0415

    config = load_config(args.config)
    records, _ = load_train_records(config)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    sample = select_records(records, limit=None, per_dataset=args.per_dataset, datasets=datasets)

    data_path = checked_dir(resolve_project_path(args.output_data))
    audit_path = checked_dir(resolve_project_path(args.audit_output))
    progress_path = checked_dir(resolve_project_path(args.progress_output))
    summary_path = resolve_project_path(args.output_summary)
    for path in (data_path, audit_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {project_relative(path)}")

    done = load_done_ids(progress_path)
    memory_before = nvidia_smi_memory_mib()
    teacher, processor = load_teacher(args)
    memory_after_load = nvidia_smi_memory_mib()

    accepted_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    iou_buckets = {"lt_0.25": 0, "0.25_to_0.5": 0, "0.5_to_0.75": 0, "ge_0.75": 0}
    parse_failures = 0
    audit_count = 0
    accepted_by_dataset: dict[str, int] = {}
    rescale_counts: dict[str, int] = {}
    started = time.time()

    with data_path.open("w", encoding="utf-8") as data_handle, audit_path.open("w", encoding="utf-8") as audit_handle:
        for index, record in enumerate(sample, 1):
            record_id = record["id"]
            if record_id in done:
                continue
            image_path = resolve_project_path(record["image"])
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            output_text = generate_one(teacher, processor, image, record["question"], args.max_new_tokens)
            raw_box = extract_raw_box(output_text)
            convention = None
            parsed = None
            if raw_box is not None:
                parsed, convention = normalize_teacher_box(raw_box, int(record["width"]), int(record["height"]))
            if parsed is not None:
                iou_value = float(grounding.iou(parsed, list(grounding.validate_box(record["answer"], normalized=True))))
            else:
                iou_value = 0.0
            if parsed is None:
                parse_failures += 1
            bucket = (
                "ge_0.75"
                if iou_value >= 0.75
                else "0.5_to_0.75"
                if iou_value >= 0.5
                else "0.25_to_0.5"
                if iou_value >= 0.25
                else "lt_0.25"
            )
            iou_buckets[bucket] += 1
            accept = parsed is not None and iou_value >= args.tau
            rescale_counts[convention or "unparsed"] = rescale_counts.get(convention or "unparsed", 0) + 1
            audit = {
                "id": record_id,
                "source": dataset_of(record),
                "iou": iou_value,
                "parsed": parsed is not None,
                "raw_box": raw_box,
                "coordinate_convention": convention,
                "accepted": accept,
                "output_text": output_text,
            }
            audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            audit_handle.flush()
            audit_count += 1
            if accept:
                row = build_pool_row(record, parsed)
                accepted_rows.append(row)
                accepted_by_dataset[dataset_of(record)] = accepted_by_dataset.get(dataset_of(record), 0) + 1
                data_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                data_handle.flush()
            progress_handle = progress_path.open("a", encoding="utf-8")
            try:
                progress_handle.write(json.dumps({"id": record_id, "accepted": accept}, ensure_ascii=False) + "\n")
            finally:
                progress_handle.close()
            if index % 25 == 0 or index == len(sample):
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "done": index,
                            "of": len(sample),
                            "accepted": len(accepted_rows),
                            "survival_rate": round(len(accepted_rows) / max(index - len(done), 1), 4),
                            "elapsed_s": round(elapsed, 1),
                            "eta_s": round(elapsed / max(index, 1) * (len(sample) - index), 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    peak_allocated = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    memory_after = nvidia_smi_memory_mib()
    summary = {
        "status": "completed",
        "tag": "kd_teacher_v1",
        "teacher": {
            "path": project_relative(resolve_project_path(args.teacher_path)),
            "load_mode": args.load_mode,
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt_id": PROMPT_ID,
        "prompt_sha256": PROMPT_SHA256,
        "tau": args.tau,
        "prompts_total": len(sample),
        "skipped_resumed": len(done),
        "generations": len(sample) - len(done),
        "accepted": len(accepted_rows),
        "survival_rate": round(len(accepted_rows) / max(len(sample) - len(done), 1), 6),
        "accepted_by_dataset": {dataset: accepted_by_dataset.get(dataset, 0) for dataset in datasets},
        "parse_failures": parse_failures,
        "rescale_applied_counts": rescale_counts,
        "teacher_format_suffix": TEACHER_FORMAT_SUFFIX,
        "iou_buckets": iou_buckets,
        "memory": {
            "nvidia_smi_before_mib": memory_before,
            "nvidia_smi_after_load_mib": memory_after_load,
            "nvidia_smi_after_generation_mib": memory_after,
            "torch_peak_allocated_mib": peak_allocated,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "data_manifest": {
            "path": project_relative(data_path),
            "records": len(accepted_rows),
            "bytes": data_path.stat().st_size,
            "sha256": __import__("hashlib").sha256(data_path.read_bytes()).hexdigest(),
        },
        "audit": {"path": project_relative(audit_path), "rows": audit_count},
        "design_note": (
            "response distillation (teacher text answer -> student SFT); NOT logit KD, because the teacher "
            "(Qwen2.5-VL-3B) and the student (SmolVLM-256M) have different vocabularies (paper §6.5)"
        ),
        "gate_note": "P3 gate: kd_train.jsonl = accepted rows; teacher params + memory peaks recorded here",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_result_new(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = preflight(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
