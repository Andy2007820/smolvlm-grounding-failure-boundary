"""T8/T9 ReST：采样 → 几何 IoU 过滤 → 训练数据池（`not_run` 状态的第一个缺口）。

设计约束（照计划 `handoff/PLAN-2026-11-10-rest-kd-tiered.md` §2 P0）：
* **单 seed pilot**（默认 seed 42），产出的池**只能**标 single-seed pilot，不是主表。
* K 条候选**一次** ``generate(num_return_sequences=K)`` 出，不做 K 次前向（省 2–3 倍 GPU 时间）。
* 过滤用冻结的几何 IoU 验证器（``src.eval.grounding.score_prediction``，与评测同一条代码路径），
  τ 默认 0.5，即与 Acc@0.5 同一判据，不做第二种口径。
* 训练目标是 4 个归一化坐标（``sft_data.format_answer``），因此池里 ``answer`` 存
  **模型自己预测框的坐标**；模型原始文本另存 ``*_pool_audit.jsonl``，便于复核。
* 支持中断续跑：每完成一条 prompt 追加一行进度，重跑时跳过已完成，避免 7.5h 采样白费。
* 绝不覆盖已有结果：summary 用 ``write_result_new``，池用独立文件名。

本模块**只产数据池**，不训练、不评测；G5 判定由后续 train/eval 阶段完成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    PROJECT_ROOT,
    load_config,
    project_relative,
    resolve_project_path,
    sha256_file,
)
from .model_audit import configure_processor
from .probe_only import nvidia_smi_memory_mib, validate_identity, write_result_new
from .sft_data import EOS_TOKEN_ID, build_probe_inputs, load_train_records
from .vision_tokens import compute_image_seq_len

DATA_ROOT = (PROJECT_ROOT / "data").resolve()
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
LOG_ROOT = (PROJECT_ROOT / "logs" / "train").resolve()
LEASE_RE = __import__("re").compile(r"^GPU-LEASE-[A-Za-z0-9-]+$")

POOL_SCHEMA = {
    "id": "str, 池内唯一（原文 id + '#k{i}'）",
    "image": "str, 与训练数据同一相对路径",
    "question": "str, 原 prompt 文本",
    "answer": "list[float], 模型自身预测框的 4 个归一化坐标（训练目标）",
    "raw_bbox": "list[float], 由 answer 推导的像素坐标（= answer × [w,h]，浮点不取整）",
    "width": "int, 原图像宽（train_sft.py 视觉 token 预检必需）",
    "height": "int, 原图像高（同上）",
    "source": "str, 数据来源 synthetic/refcoco（train_sft 的 split 校验用）",
}
TRAINER_REQUIRED_FIELDS = ("image", "question", "width", "height", "source")
# src/eval/dataset.py:44 要求**恰好八字段**，且 answer 必须与 raw_bbox/width,height
# 在 1.1e-8 内一致；raw_bbox 只能由预测框推导，绝不能抄 GT（否则等于把 GT 泄漏进池子）。
POOL_FIELDS = ("id", "image", "question", "answer", "raw_bbox", "width", "height", "source")


def now_iso() -> str:
    import datetime

    return datetime.datetime.now().astimezone().isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checked_data_path(value: str | Path, *, tag: str) -> Path:
    """池文件必须落在 data/rest/ 下并带 tag，避免写到别处或覆盖历史数据。"""

    path = resolve_project_path(value)
    root = (DATA_ROOT / "rest").resolve()
    if path.parent != root:
        raise ValueError(f"pool must be directly under data/rest/: {path}")
    if tag not in path.name:
        raise ValueError(f"pool filename must contain tag {tag!r}: {path}")
    return path


def checked_summary_path(value: str | Path) -> Path:
    path = resolve_project_path(value)
    if path.parent != RESULTS_ROOT or not path.name.startswith("train_"):
        raise ValueError(f"summary must be directly under results/ and start with 'train_': {path}")
    return path


def iou_bucket(iou: float | None) -> str:
    if iou is None:
        return "parse_fail"
    if iou >= 0.75:
        return "ge_0.75"
    if iou >= 0.5:
        return "0.5_to_0.75"
    if iou >= 0.25:
        return "0.25_to_0.5"
    return "lt_0.25"


def dataset_of(record: dict[str, Any]) -> str:
    for key in ("dataset", "source_dataset", "source", "split"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    image = str(record.get("image", ""))
    if "synthetic" in image:
        return "synthetic"
    if "refcoco" in image or "refcoco" in str(record.get("id", "")).lower():
        return "refcoco"
    return "unknown"


def build_pool_record(
    record: dict[str, Any],
    *,
    k_index: int,
    predicted_box: list[float],
    iou: float,
) -> dict[str, Any]:
    """训练用池记录：沿用原记录的 schema（width/height/source 是 trainer 必需的），
    只把 ``id`` 改成池内唯一、把 ``answer`` 换成模型自己的预测框。

    注意：``raw_bbox`` 是 GT 派生字段，只服务 pixel 变体编码，本配置（normalized_decimal）
    不读它，因此池里不携带，避免出现「训练目标到底是预测还是 GT」的歧义。
    """

    missing = [name for name in TRAINER_REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"source record {record.get('id')!r} lacks trainer-required fields: {missing}")
    box = [float(value) for value in predicted_box]
    width = int(record["width"])
    height = int(record["height"])
    return {
        "id": f"{record['id']}#k{k_index}",
        "image": record["image"],
        "question": record["question"],
        "answer": box,
        "raw_bbox": [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
        "width": width,
        "height": height,
        "source": record["source"],
    }


def build_audit_record(
    record: dict[str, Any],
    *,
    k_index: int,
    accepted: bool,
    output_text: str,
    raw_output: str,
    score: Any,
    tau: float,
) -> dict[str, Any]:
    return {
        "id": f"{record['id']}#k{k_index}",
        "source_id": record["id"],
        "dataset": dataset_of(record),
        "k_index": int(k_index),
        "accepted": bool(accepted),
        "tau": float(tau),
        "iou": score.iou,
        "iou_bucket": iou_bucket(score.iou),
        "predicted_box": list(score.predicted_box) if score.predicted_box is not None else None,
        "parse_error": score.parse_error,
        "output_text": output_text,
        "raw_output": raw_output,
        "ground_truth": record["answer"],
    }


def accepted(score: Any, *, tau: float) -> bool:
    """唯一判据：可解析出框且几何 IoU ≥ τ（与 Acc@0.5 同一条口径，不引入第二种）。"""

    if score.predicted_box is None or score.parse_error is not None:
        return False
    box = [float(value) for value in score.predicted_box]
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        return False  # 退化框会让 dataset.validate_record 的 raw_bbox 检查失败
    return score.iou is not None and float(score.iou) >= float(tau)


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
            source_id = payload.get("source_id")
            if isinstance(source_id, str) and payload.get("status") == "done":
                done.add(source_id)
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--training-results", required=True)
    parser.add_argument("--tag", required=True, help="r1 / r2 / probe，用于文件名与结果命名")
    parser.add_argument("--output-pool", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--audit-output", default=None, help="默认由 pool 推出 *_pool_audit.jsonl")
    parser.add_argument("--progress-output", default=None, help="默认由 pool 推出 *_progress.jsonl")
    parser.add_argument("--log-output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条 prompt（探针用 200）")
    parser.add_argument("--per-dataset", type=int, default=None, help="按数据集各取 N 条并交错（混合池用）")
    parser.add_argument("--datasets", default=None, help="逗号分隔，如 synthetic,refcoco；默认全部")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-lease-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only 预检，不加载 torch/权重")
    return parser


def resolve_aux_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    pool = checked_data_path(args.output_pool, tag=args.tag)
    audit = (
        resolve_project_path(args.audit_output)
        if args.audit_output
        else pool.with_name(pool.stem + "_audit.jsonl")
    )
    progress = (
        resolve_project_path(args.progress_output)
        if args.progress_output
        else pool.with_name(pool.stem + "_progress.jsonl")
    )
    log = (
        resolve_project_path(args.log_output)
        if args.log_output
        else LOG_ROOT / f"rest_sample_{args.tag}.log"
    )
    return audit, progress, log


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    per_dataset: int | None,
    datasets: list[str] | None,
) -> list[dict[str, Any]]:
    """选取 prompt：默认取前 N 条；``--per-dataset`` 时按数据集各取 N 条并交错。

    交错是必需的：训练数据是 synthetic(5000) 后接 refcoco(5000)，直接 ``[:N]`` 只会拿到
    synthetic，混合池会系统性偏向合成数据，存活率也无法分集比较。
    """

    pool = records if not datasets else [record for record in records if dataset_of(record) in set(datasets)]
    if not pool:
        raise SystemExit(f"no records match datasets={datasets}")
    if per_dataset is not None:
        if per_dataset <= 0:
            raise SystemExit("--per-dataset must be >= 1")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in pool:
            grouped.setdefault(dataset_of(record), []).append(record)
        selected: list[dict[str, Any]] = []
        index = 0
        while True:
            added = False
            for name in sorted(grouped):
                if index < len(grouped[name]) and index < per_dataset:
                    selected.append(grouped[name][index])
                    added = True
            if not added:
                break
            index += 1
        return selected
    if limit is None:
        return pool
    if limit <= 0:
        raise SystemExit("--limit must be >= 1")
    return pool[: int(limit)]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only 计划，不加载任何权重；dry-run 与真跑都先走这里。"""

    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if not (0.0 < args.tau <= 1.0):
        raise SystemExit("--tau must be in (0, 1]")
    if args.device == "cuda":
        if not args.gpu_lease_id or not LEASE_RE.fullmatch(args.gpu_lease_id):
            raise SystemExit("CUDA sampling requires --gpu-lease-id GPU-LEASE-... issued by the Lead")
    config = load_config(args.config)
    adapter_dir = resolve_project_path(args.adapter_dir)
    identity = validate_identity(
        config_path=resolve_project_path(args.config),
        adapter_dir=adapter_dir,
        training_summary_path=resolve_project_path(args.training_summary),
        training_results_path=resolve_project_path(args.training_results),
    )
    records, train_evidence = load_train_records(config)
    dataset_filter = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
    sample = select_records(
        records,
        limit=args.limit,
        per_dataset=args.per_dataset,
        datasets=dataset_filter,
    )
    total = len(sample)
    if total <= 0:
        raise SystemExit("no records selected")
    return {
        "status": "preflight_passed",
        "gpu_used": False,
        "mode": "dry_run" if args.dry_run else "planned",
        "tag": args.tag,
        "pilot_scope": "single-seed pilot (seed 42); NOT a 3-seed main table and NOT a G5 verdict",
        "adapter_dir": project_relative(adapter_dir),
        "identity": identity,
        "records_available": len(records),
        "prompts_selected": total,
        "prompt_ids_head": [record["id"] for record in sample[:5]],
        "k": args.k,
        "tau": args.tau,
        "expected_generations": total * args.k,
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_return_sequences": args.k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "num_beams": 1,
        },
        "dataset_mix": {
            name: sum(1 for record in sample if dataset_of(record) == name)
            for name in sorted({dataset_of(record) for record in sample})
        },
        "train_dataset_evidence": train_evidence,
        "image_seq_len": compute_image_seq_len(
            int(config["image"]["image_size"]),
            int(config["image"]["patch_size"]),
            int(config["image"]["scale_factor"]),
        ),
        "generated_at": now_iso(),
    }


def run_sampling(args: argparse.Namespace) -> dict[str, Any]:
    """真正的采样；调用者已确认 GPU 租约。"""

    audit_path, progress_path, log_path = resolve_aux_paths(args)
    pool_path = checked_data_path(args.output_pool, tag=args.tag)
    summary_path = checked_summary_path(args.output_summary)
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {summary_path}")
    if pool_path.exists():
        raise FileExistsError(f"refusing to overwrite existing pool: {pool_path} (resume uses the same file)")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    plan = preflight(args)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.eval.grounding import score_prediction

    config = load_config(args.config)
    model_path = resolve_project_path(config["model"]["path"])
    records, _ = load_train_records(config)
    dataset_filter = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
    sample = select_records(
        records,
        limit=args.limit,
        per_dataset=args.per_dataset,
        datasets=dataset_filter,
    )
    done_ids = load_done_ids(progress_path)
    todo = [record for record in sample if record["id"] not in done_ids]

    model = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        memory_before = nvidia_smi_memory_mib() if args.device == "cuda" else None
        if memory_before is not None and memory_before > 7500:
            raise RuntimeError(f"GPU memory before sampling is {memory_before} MiB > 7500 MiB; aborting")
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        geometry = configure_processor(processor, image_size=int(config["image"]["image_size"]))
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to(args.device)
        model = PeftModel.from_pretrained(model, resolve_project_path(args.adapter_dir), is_trainable=False)
        model.eval()
        if not [name for name, module in model.named_modules() if hasattr(module, "lora_A")]:
            raise RuntimeError("adapter did not attach any LoRA modules")
        if args.device == "cuda":
            torch.cuda.synchronize()
            memory_after_load = nvidia_smi_memory_mib()
            if memory_after_load > 7500:
                raise RuntimeError(f"GPU memory after adapter load is {memory_after_load} MiB > 7500 MiB; aborting")
        else:
            memory_after_load = None

        # SmolVLM/Idefics3 的 generate() 不接受 ``generator`` 关键字（transformers 会以
        # "model_kwargs are not used by the model: ['generator']" 直接报错），因此采样随机性
        # 由 torch 全局 RNG 播种控制；跨机器/跨版本不保证逐 token 位级可复现，summary 里如实记录。
        torch.manual_seed(args.seed)

        started = time.perf_counter()
        stats: dict[str, Any] = {
            "prompts_done_this_run": 0,
            "prompts_skipped_resume": len(sample) - len(todo),
            "generations": 0,
            "accepted": 0,
            "prompts_with_at_least_one_accept": 0,
            "accepted_by_dataset": {},
            "generated_by_dataset": {},
            "iou_buckets": {},
            "parse_failures": 0,
            "hit_max_new_tokens": 0,
        }

        pool_handle = pool_path.open("a", encoding="utf-8", newline="\n")
        audit_handle = audit_path.open("a", encoding="utf-8", newline="\n")
        try:
            for record in todo:
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
                        input_ids=inputs["input_ids"].to(args.device),
                        attention_mask=inputs["attention_mask"].to(args.device),
                        pixel_values=inputs["pixel_values"].to(args.device),
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=True,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        num_return_sequences=int(args.k),
                        num_beams=1,
                    )
                dataset = dataset_of(record)
                stats["generated_by_dataset"][dataset] = stats["generated_by_dataset"].get(dataset, 0) + args.k
                prompt_accepted = 0
                for k_index in range(generated.shape[0]):
                    continuation = generated[k_index, prompt_tokens:]
                    if int(continuation.shape[0]) >= int(args.max_new_tokens):
                        stats["hit_max_new_tokens"] += 1
                    raw_output = processor.batch_decode(continuation.unsqueeze(0), skip_special_tokens=False)[0]
                    output_text = processor.batch_decode(continuation.unsqueeze(0), skip_special_tokens=True)[0]
                    score = score_prediction(output_text, record["answer"])
                    ok = accepted(score, tau=args.tau)
                    stats["generations"] += 1
                    if score.predicted_box is None or score.parse_error is not None:
                        stats["parse_failures"] += 1
                    bucket = iou_bucket(score.iou)
                    stats["iou_buckets"][bucket] = stats["iou_buckets"].get(bucket, 0) + 1
                    if ok:
                        line = build_pool_record(
                            record,
                            k_index=k_index,
                            predicted_box=score.predicted_box,
                            iou=float(score.iou),
                        )
                        pool_handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                        stats["accepted"] += 1
                        stats["accepted_by_dataset"][dataset] = stats["accepted_by_dataset"].get(dataset, 0) + 1
                        prompt_accepted += 1
                    audit_handle.write(
                        json.dumps(
                            build_audit_record(
                                record,
                                k_index=k_index,
                                accepted=ok,
                                output_text=output_text,
                                raw_output=raw_output,
                                score=score,
                                tau=args.tau,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if prompt_accepted:
                    stats["prompts_with_at_least_one_accept"] += 1
                pool_handle.flush()
                audit_handle.flush()
                with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(
                        json.dumps(
                            {"source_id": record["id"], "status": "done", "accepted": prompt_accepted},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stats["prompts_done_this_run"] += 1
                if stats["prompts_done_this_run"] % 50 == 0:
                    elapsed = time.perf_counter() - started
                    done = stats["prompts_done_this_run"]
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = (len(todo) - done) / rate if rate > 0 else float("nan")
                    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "done": done,
                                    "of": len(todo),
                                    "accepted": stats["accepted"],
                                    "survival_rate": stats["accepted"] / max(1, stats["generations"]),
                                    "elapsed_s": round(elapsed, 1),
                                    "eta_s": round(remaining, 1),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            pool_handle.close()
            audit_handle.close()

        elapsed = time.perf_counter() - started
        if args.device == "cuda":
            torch.cuda.synchronize()
            memory_after = nvidia_smi_memory_mib()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        else:
            memory_after = None
            peak_allocated = None
            peak_reserved = None

        generations = max(1, stats["generations"])
        payload = {
            **plan,
            "status": "completed",
            "mode": "sampled",
            "pool": {
                "path": project_relative(pool_path),
                "records": stats["accepted"],
                "sha256": sha256_file(pool_path),
                "schema": POOL_SCHEMA,
                "resume_progress": project_relative(progress_path),
                "audit": project_relative(audit_path),
            },
            "survival": {
                "generations": stats["generations"],
                "accepted": stats["accepted"],
                "survival_rate": stats["accepted"] / generations,
                "prompts_total": len(sample),
                "prompts_with_at_least_one_accept": stats["prompts_with_at_least_one_accept"],
                "accepted_by_dataset": stats["accepted_by_dataset"],
                "generated_by_dataset": stats["generated_by_dataset"],
                "iou_buckets": stats["iou_buckets"],
                "parse_failures": stats["parse_failures"],
                "hit_max_new_tokens": stats["hit_max_new_tokens"],
            },
            "resume": {
                "skipped_prompts": stats["prompts_skipped_resume"],
                "done_prompts_this_run": stats["prompts_done_this_run"],
            },
            "runtime": {
                "device": args.device,
                "dtype": "bfloat16" if args.device == "cuda" else "float32",
                "gpu_lease_id": args.gpu_lease_id,
                "nvidia_smi_memory_before_mib": memory_before,
                "nvidia_smi_memory_after_adapter_load_mib": memory_after_load,
                "nvidia_smi_memory_after_sampling_mib": memory_after,
                "torch_peak_memory_allocated_mib": peak_allocated,
                "torch_peak_memory_reserved_mib": peak_reserved,
                "elapsed_seconds": elapsed,
                "seconds_per_generation": elapsed / generations,
            },
            "processor_geometry": geometry,
            "source_sha256": {
                "src/train/rest_sample.py": sha256_file(Path(__file__)),
                "src/train/sft_data.py": sha256_file(PROJECT_ROOT / "src/train/sft_data.py"),
                "src/train/probe_only.py": sha256_file(PROJECT_ROOT / "src/train/probe_only.py"),
                "src/eval/grounding.py": sha256_file(PROJECT_ROOT / "src/eval/grounding.py"),
            },
            "design_deviation": None,
            "note": (
                "ReST sampling pool only. Diagnostic pilot data; not an evaluation, not a G5 verdict. "
                "Accepted samples are the model's own predictions whose geometric IoU >= tau against "
                "ground truth, stored as normalised box coordinates (the trainer's supervised target)."
            ),
            "generated_at": now_iso(),
        }
        write_result_new(summary_path, payload)
        return payload
    finally:
        if model is not None:
            del model
        if args.device == "cuda":
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        plan = preflight(args)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(
            f"\n[dry-run] {plan['prompts_selected']} prompts x K={plan['k']} = "
            f"{plan['expected_generations']} generations; tau={plan['tau']}; "
            f"estimated GPU hours at 1.35 s/gen = {plan['expected_generations'] * 1.35 / 3600:.2f} h",
            file=sys.stderr,
        )
        return
    payload = run_sampling(args)
    survival = payload["survival"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pool": payload["pool"]["path"],
                "pool_records": payload["pool"]["records"],
                "survival_rate": survival["survival_rate"],
                "accepted_by_dataset": survival["accepted_by_dataset"],
                "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
