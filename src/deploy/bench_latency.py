"""Warmed deployment latency benchmark (T14 Table 3 source).

Contract (must match what the paper reports):

* deterministic 50-image subset selected by seed 42, id digest recorded so the
  CPU / quantized / GPU columns measure the *same* images;
* 5 untimed warmup calls, then one timed greedy generation per image;
* mean / median / p95 / p99 / min / max, peak process RSS, model-load time;
* accuracy is recorded as a **diagnostic** only — official metrics stay with
  the independent evaluator.

``--plan-only`` validates everything and prints the plan without loading a
model (the prep phase runs only this mode).  Real CPU benchmarking is allowed
only after the Lead announces the evaluation GPU is released; ``--engine
hf-cuda`` additionally requires an explicit ``--gpu-lease-id``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    BENCH_SAMPLES,
    BENCH_SEED,
    BENCH_WARMUP,
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    EXPECTED_IMAGE_SEQ_LEN,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PROJECT_ROOT,
    benchmark_time_estimate,
    checked_results_path,
    latency_stats,
    make_scratch_dir,
    now_iso,
    project_relative,
    read_jsonl,
    remove_scratch_dir,
    resolve_project_path,
    select_benchmark_records,
    selected_ids_digest,
    write_json_atomic,
    write_jsonl,
)
from .env import detect_llama_cpp, nvidia_smi_snapshot
from .infer import HFPredictor, classify_artifact, peak_process_memory_mib

PYTHON = sys.executable or "python"
ENGINES = ("hf-cpu", "hf-cpu-adapter", "hf-cuda", "torch-int8", "gguf-q4")
DEFAULT_DATASET = "data/synthetic_eval.jsonl"
GPU_MEMORY_ABORT_MIB = 7500
INT8_SCOPE = "model.model.text_model Linear -> qint8 dynamic (runtime-only)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--artifact", default=None, help="merged dir (hf-*) or GGUF dir/file (gguf-q4)")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="adapter for --engine hf-cpu-adapter")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="base model for hf-cpu-adapter")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n", type=int, default=BENCH_SAMPLES)
    parser.add_argument("--warmup", type=int, default=BENCH_WARMUP)
    parser.add_argument("--seed", type=int, default=BENCH_SEED)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="defaults from engine")
    parser.add_argument("--gpu-lease-id", default=None, help="required for --engine hf-cuda")
    parser.add_argument("--output", default=None, help="results/deploy_latency_<engine>.json")
    parser.add_argument("--details-output", default=None, help="optional results/deploy_latency_*_details.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU intra-op threads (default: torch default)")
    parser.add_argument("--plan-only", action="store_true", help="validate and print the plan; never load a model")
    parser.add_argument("--force", action="store_true")
    return parser


def default_output(engine: str) -> str:
    return f"results/deploy_latency_{engine.replace('-', '_')}.json"


def _engine_device(engine: str, requested: str | None) -> str:
    if engine == "hf-cuda":
        if requested not in (None, "cuda"):
            raise ValueError("--engine hf-cuda requires --device cuda")
        return "cuda"
    if requested not in (None, "cpu"):
        raise ValueError(f"--engine {engine} is CPU-only; --device cuda is not allowed")
    return "cpu"


def resolve_gguf_artifact(value: str | None) -> dict[str, Any]:
    if value is None:
        return {"kind": "gguf", "usable": False, "reason": "no --artifact given"}
    path = resolve_project_path(value)
    if path.is_file() and path.suffix.lower() == ".gguf":
        candidates = [path]
        root = path.parent
    elif path.is_dir():
        candidates = sorted(path.rglob("*.gguf"))
        root = path
    else:
        return {"kind": "gguf", "usable": False, "reason": f"path not found: {path}"}
    quantized = [item for item in candidates if "q4_k_m" in item.name.lower() or "q4_k" in item.name.lower()]
    mmproj = [item for item in candidates if "mmproj" in item.name.lower()]
    model = (quantized or candidates or [None])[0]
    usable = bool(model and mmproj)
    return {
        "kind": "gguf",
        "usable": usable,
        "path": project_relative(root),
        "model": project_relative(model) if model else None,
        "mmproj": project_relative(mmproj[0]) if mmproj else None,
        "files": [{"path": project_relative(item), "bytes": item.stat().st_size} for item in candidates],
        "reason": None if usable else "need both a quantized .gguf model file and a *mmproj*.gguf file",
    }


def build_benchmark_plan(args: argparse.Namespace) -> dict[str, Any]:
    device = _engine_device(args.engine, args.device)
    dataset_path = resolve_project_path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    records = read_jsonl(dataset_path)
    selected = select_benchmark_records(records, args.n, seed=args.seed)
    digest = selected_ids_digest(selected)
    if args.engine == "gguf-q4":
        artifact = resolve_gguf_artifact(args.artifact)
    elif args.engine == "hf-cpu-adapter":
        artifact = {
            "kind": "adapter",
            "usable": classify_artifact(resolve_project_path(args.adapter), model_path=resolve_project_path(args.model_path))["usable"],
            "adapter": classify_artifact(resolve_project_path(args.adapter), model_path=resolve_project_path(args.model_path)),
            "base_model": project_relative(resolve_project_path(args.model_path)),
        }
    else:
        merged_path = resolve_project_path(args.artifact or DEFAULT_MERGED)
        classified = classify_artifact(merged_path, model_path=resolve_project_path(args.model_path))
        artifact = {"kind": classified["kind"], "usable": classified["usable"], "checkpoint": classified}

    if not artifact["usable"]:
        raise FileNotFoundError(f"benchmark artifact is not usable: {json.dumps(artifact, ensure_ascii=False)[:400]}")
    if device == "cuda" and not args.gpu_lease_id:
        raise RuntimeError("--engine hf-cuda requires --gpu-lease-id from the Lead")

    estimate = benchmark_time_estimate(args.n, args.warmup)
    output = checked_results_path(args.output or default_output(args.engine), "deploy_latency_")
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force or a new --output")
    details = None
    if args.details_output:
        details = checked_results_path(args.details_output, "deploy_latency_")
        if details.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {details}; pass --force")
    return {
        "status": "plan_only",
        "engine": args.engine,
        "device": device,
        "artifact": artifact,
        "dataset": {"path": project_relative(dataset_path), "rows": len(records)},
        "selection": {
            "n": args.n,
            "warmup": args.warmup,
            "seed": args.seed,
            "selected_ids_digest": digest,
            "first_ids": [record["id"] for record in selected[:5]],
        },
        "inference": {
            "image_size": args.image_size,
            "image_seq_len": EXPECTED_IMAGE_SEQ_LEN,
            "max_new_tokens": args.max_new_tokens,
            "prompt_id": "p1",
            "do_sample": False,
            "num_beams": 1,
            "quantization": (
                {"scope": INT8_SCOPE, "artifact_type": "runtime_only", "dtype": "qint8"}
                if args.engine == "torch-int8"
                else None
            ),
            "torch_threads_requested": args.threads,
        },
        "gpu": {
            "required": device == "cuda",
            "gpu_lease_id": args.gpu_lease_id,
            "snapshot_before": nvidia_smi_snapshot() if device == "cuda" else None,
        },
        "estimate": estimate,
        "output": project_relative(output),
        "details_output": project_relative(details) if details else None,
        "plan_note": "official accuracy metrics are eval-closer's; this benchmark records diagnostics only",
    }


# --------------------------------------------------------------- gguf engine
class GgufPredictor:
    """Minimal llama.cpp mtmd CLI driver; used only when the GGUF route exists."""

    def __init__(self, artifact: dict[str, Any], *, max_new_tokens: int, llama: dict[str, Any] | None = None) -> None:
        self.artifact = artifact
        self.max_new_tokens = max_new_tokens
        self.llama = llama or detect_llama_cpp()
        self.binary = self.llama.get("binaries", {}).get("llama-mtmd-cli", {}).get("path")
        if not self.binary:
            raise FileNotFoundError("llama-mtmd-cli is not installed; GGUF latency is not available")

    def load(self) -> dict[str, Any]:
        return {
            "status": "loaded",
            "engine": "gguf-q4",
            "backend": "llama-mtmd-cli",
            "binary": self.binary,
            "model": self.artifact["model"],
            "mmproj": self.artifact["mmproj"],
        }

    def predict_path(self, image_path: Path, question: str) -> dict[str, Any]:
        from src.eval.grounding import parse_box_with_error
        from src.eval.prompts import render_prompt

        prompt = render_prompt(question)
        argv = [
            str(self.binary),
            "-m",
            str(resolve_project_path(self.artifact["model"])),
            "--mmproj",
            str(resolve_project_path(self.artifact["mmproj"])),
            "--image",
            str(image_path),
            "-p",
            prompt,
            "-n",
            str(self.max_new_tokens),
            "--temp",
            "0",
        ]
        started = time.perf_counter()
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=600, errors="replace")
        latency = time.perf_counter() - started
        text = (completed.stdout or "").strip()
        box, error = parse_box_with_error(text)
        return {
            "raw_output": text,
            "output_text": text,
            "predicted_box": box,
            "parse_error": error,
            "latency_seconds": latency,
            "exit_code": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-500:],
        }


# ------------------------------------------------------------------ accuracy
def _score(record: dict[str, Any], output_text: str) -> dict[str, Any]:
    from src.eval.grounding import score_prediction

    score = score_prediction(output_text, record["answer"])
    return {
        "iou": score.iou,
        "correct@0.5": score.correct_at_0_5,
        "parse_error": score.parse_error,
    }


def run_hf_benchmark(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    records = read_jsonl(resolve_project_path(args.dataset))
    selected = select_benchmark_records(records, args.n, seed=args.seed)
    if args.engine == "hf-cpu-adapter":
        predictor = HFPredictor(
            model_path=resolve_project_path(args.model_path),
            adapter=resolve_project_path(args.adapter),
            device=plan["device"],
            dtype="float32" if plan["device"] == "cpu" else "bfloat16",
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        predictor = HFPredictor(
            model_path=resolve_project_path(args.model_path),
            merged=resolve_project_path(args.artifact or DEFAULT_MERGED),
            device=plan["device"],
            dtype="float32" if plan["device"] == "cpu" else "bfloat16",
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
        )
    with predictor:
        load_report = predictor.load_report or {}
        int8_report: dict[str, Any] | None = None
        if args.engine == "torch-int8":
            int8_report = quantize_text_decoder_dynamic(predictor.model)
            load_report = {**load_report, "quantization": int8_report}
        for _ in range(args.warmup):
            with Image.open(PROJECT_ROOT / selected[0]["image"]) as opened:
                predictor.predict(opened.convert("RGB"), selected[0]["question"])
        rows: list[dict[str, Any]] = []
        for record in selected:
            with Image.open(PROJECT_ROOT / record["image"]) as opened:
                image = opened.convert("RGB")
            started = time.perf_counter()
            output = predictor.predict(image, record["question"])
            latency = time.perf_counter() - started
            score = _score(record, output["output_text"])
            rows.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "image": record["image"],
                    "latency_seconds": latency,
                    "inference_seconds": output["inference_seconds"],
                    "generated_tokens": output["generated_tokens"],
                    "hit_max_new_tokens": output["hit_max_new_tokens"],
                    "output_text": output["output_text"],
                    "predicted_box": output["predicted_box"],
                    **score,
                }
            )
    if plan["device"] == "cuda":
        import torch

        plan["gpu"]["torch_peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        plan["gpu"]["torch_peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        plan["gpu"]["snapshot_after"] = nvidia_smi_snapshot()
    return {"rows": rows, "load_report": load_report}


def quantize_text_decoder_dynamic(model: Any) -> dict[str, Any]:
    """INT8 dynamic quantization of the text decoder (runtime-only, CPU).

    This is a real quantization experiment, not a file artifact: the quantized
    modules live in this process only and the command reproduces them exactly.
    """

    import torch
    from torch.ao.quantization import quantize_dynamic

    text_model = model.model.text_model
    linear_before = sum(1 for module in text_model.modules() if isinstance(module, torch.nn.Linear))
    import time as _time

    started = _time.perf_counter()
    quantized = quantize_dynamic(text_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
    model.model.text_model = quantized
    quantized_linear = sum(
        1 for module in quantized.modules() if type(module).__name__.startswith("Linear") and hasattr(module, "_packed_params")
    )
    return {
        "scope": INT8_SCOPE,
        "artifact_type": "runtime_only",
        "dtype": "qint8",
        "linear_modules_before": linear_before,
        "dynamic_quantized_modules": quantized_linear,
        "quantize_seconds": _time.perf_counter() - started,
        "note": "torch.ao.quantization is deprecated in torch 2.11; kept for a comparable INT8 baseline",
    }


def run_gguf_benchmark(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    records = read_jsonl(resolve_project_path(args.dataset))
    selected = select_benchmark_records(records, args.n, seed=args.seed)
    artifact = plan["artifact"]
    predictor = GgufPredictor(artifact, max_new_tokens=args.max_new_tokens)
    load_report = predictor.load()
    rows: list[dict[str, Any]] = []
    scratch = make_scratch_dir("gguf_bench")
    try:
        for index, record in enumerate(selected):
            with Image.open(PROJECT_ROOT / record["image"]) as opened:
                image_path = scratch / f"sample_{index:03d}.png"
                opened.convert("RGB").save(image_path)
            output = predictor.predict_path(image_path, record["question"])
            score = _score(record, output["output_text"])
            rows.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "image": record["image"],
                    "latency_seconds": output["latency_seconds"],
                    "exit_code": output["exit_code"],
                    "output_text": output["output_text"],
                    "predicted_box": output["predicted_box"],
                    **score,
                }
            )
    finally:
        remove_scratch_dir(scratch)
    return {"rows": rows, "load_report": load_report}


def summarise(rows: list[dict[str, Any]], *, plan: dict[str, Any], load_report: dict[str, Any]) -> dict[str, Any]:
    from src.eval.grounding import metrics_from_scores

    latencies = [row["latency_seconds"] for row in rows]
    metrics = metrics_from_scores(
        [
            type(
                "Score",
                (),
                {
                    "predicted_box": row["predicted_box"],
                    "iou": row["iou"],
                    "correct_at_0_5": row["correct@0.5"],
                    "parse_error": row["parse_error"],
                },
            )()
            for row in rows
        ]
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "engine": plan["engine"],
        "device": plan["device"],
        "n": len(rows),
        "warmup": plan["selection"]["warmup"],
        "latency": latency_stats(latencies),
        "throughput_images_per_second": len(rows) / sum(latencies) if sum(latencies) else None,
        "diagnostic_metrics": metrics,
        "diagnostic_metric_note": (
            "computed with the frozen parser/metric code for deployment sanity only; "
            "official Acc@0.5/mIoU remain eval-closer's independent results"
        ),
        "generated_tokens": {
            "mean": sum(row.get("generated_tokens", 0) or 0 for row in rows) / len(rows),
            "hit_max_new_tokens_count": sum(1 for row in rows if row.get("hit_max_new_tokens")),
        },
        "load_report": load_report,
        "gpu": plan["gpu"],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n <= 0 or args.warmup < 0:
        raise ValueError("--n must be positive and --warmup non-negative")
    try:
        plan = build_benchmark_plan(args)
    except (FileNotFoundError, RuntimeError) as exc:
        payload = {
            "status": "blocked",
            "engine": args.engine,
            "plan_only": bool(args.plan_only),
            "reason": str(exc),
            "captured_at": now_iso(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    plan["run_id"] = args.run_id or f"deploy_latency_{args.engine.replace('-', '_')}"

    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    started = time.perf_counter()
    threads_used: int | None = None
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        import torch

        torch.set_num_threads(int(args.threads))
        threads_used = int(torch.get_num_threads())
    if args.engine == "gguf-q4":
        result = run_gguf_benchmark(args, plan)
    else:
        result = run_hf_benchmark(args, plan)
    payload = {
        "run_id": plan["run_id"],
        "started_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - started,
        "plan": plan,
        "details_rows": len(result["rows"]),
        "environment": {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "onnxruntime": _version("onnxruntime"),
        },
        "peak_process_memory_mib": peak_process_memory_mib(),
        "cpu_threads": threads_used,
        **summarise(result["rows"], plan=plan, load_report=result["load_report"]),
    }
    output = resolve_project_path(plan["output"])
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    write_json_atomic(output, payload)
    if plan["details_output"]:
        write_jsonl(resolve_project_path(plan["details_output"]), result["rows"])
    print(
        json.dumps(
            {
                "status": payload["status"],
                "engine": args.engine,
                "output": project_relative(output),
                "n": payload["n"],
                "latency": payload["latency"],
                "diagnostic_metrics": payload["diagnostic_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
