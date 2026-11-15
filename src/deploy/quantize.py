"""T14 quantization: GGUF Q4_K_M (preferred), ONNX dynamic INT8, torch INT8, fp32.

Two things this module refuses to do:

1. It never treats the upstream ONNX files under ``models/.../onnx`` as *our*
   quantized model: they are the base SmolVLM-256M-Instruct export without the
   project LoRA (marked ``carries_project_lora=false`` everywhere).
2. It never invents a quantized artifact.  A route that cannot run is recorded
   as ``blocked``/``not_run`` with the real reason.

``--plan-only`` is CPU-light: it validates paths and prints the exact commands
without importing torch/onnx/gguf.  Real execution requires torch (torch-int8)
or external tools (llama.cpp / optimum / onnx), which only run after the Lead's
lease.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import (
    DEFAULT_MERGED,
    DEFAULT_ONNX_OUT,
    DEFAULT_QUANTIZED_GGUF,
    EXPECTED_IMAGE_SEQ_LEN,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PROJECT_ROOT,
    benchmark_time_estimate,
    checked_results_path,
    now_iso,
    project_relative,
    resolve_project_path,
    sha256_file,
    write_json_atomic,
)
from .env import detect_llama_cpp, merged_facts, packages, quantized_facts
from .infer import classify_artifact, peak_process_memory_mib

PYTHON = sys.executable or "python"
ROUTES = ("gguf", "onnx", "torch-int8", "fp32")
SAMPLE_IMAGE = PROJECT_ROOT / "models" / "g0" / "smoke_image.png"
SAMPLE_PROMPT = (
    "Locate the object described below in the image.\n"
    "Description: the red circle\n"
    "Answer with the bounding box as normalized coordinates "
    "[x1, y1, x2, y2], each between 0 and 1."
)


# --------------------------------------------------------------------- plans
def _route_estimate(route: str, environment: dict[str, Any] | None = None) -> float:
    estimates = {
        "gguf": 35.0,
        "onnx": 25.0,
        "torch-int8": 8.0,
        "fp32": 1.0,
    }
    return estimates[route]


def build_gguf_plan(
    merged_path: Path,
    output_dir: Path,
    *,
    llama: dict[str, Any] | None = None,
    sample_image: Path = SAMPLE_IMAGE,
) -> dict[str, Any]:
    llama = llama or detect_llama_cpp()
    packages_payload = packages()
    missing: list[str] = list(llama.get("missing") or [])
    if not packages_payload["gguf"]["available"]:
        missing.append("python package gguf")
    f16 = output_dir / "smolvlm-256m-f16.gguf"
    q4 = output_dir / "smolvlm-256m-q4_k_m.gguf"
    commands: list[dict[str, Any]] = []
    if llama.get("convert_script"):
        commands.append(
            {
                "step": "convert_f16_mmproj",
                "argv": [
                    PYTHON,
                    str(llama["convert_script"]),
                    str(merged_path),
                    "--outfile",
                    str(f16),
                    "--mmproj",
                ],
                "cwd": llama.get("home") or str(PROJECT_ROOT),
                "expected_outputs": [str(f16)],
                "note": (
                    "convert_hf_to_gguf.py writes the F16 model plus a *mmproj*.gguf; "
                    "the exact mmproj filename/flag changed across llama.cpp releases, "
                    "so the executor detects produced files instead of assuming"
                ),
                "timeout_seconds": 1800,
            }
        )
    if llama.get("binaries", {}).get("llama-quantize", {}).get("path"):
        commands.append(
            {
                "step": "quantize_q4_k_m",
                "argv": [
                    llama["binaries"]["llama-quantize"]["path"],
                    str(f16),
                    str(q4),
                    "Q4_K_M",
                ],
                "cwd": llama.get("home") or str(PROJECT_ROOT),
                "expected_outputs": [str(q4)],
                "timeout_seconds": 900,
            }
        )
    mtmd = llama.get("binaries", {}).get("llama-mtmd-cli", {}).get("path")
    if mtmd and q4:
        commands.append(
            {
                "step": "smoke_mmproj_generation",
                "argv": [
                    mtmd,
                    "-m",
                    str(q4),
                    "--mmproj",
                    "<detected mmproj>.gguf",
                    "--image",
                    str(sample_image),
                    "-p",
                    SAMPLE_PROMPT,
                    "-n",
                    str(MAX_NEW_TOKENS),
                    "--temp",
                    "0",
                ],
                "cwd": llama.get("home") or str(PROJECT_ROOT),
                "expected_outputs": [],
                "note": "flag names differ per release; executor rewrites this argv after `--help` inspection",
                "timeout_seconds": 600,
            }
        )
    status = "ready" if not missing else "blocked_tool_missing"
    return {
        "route": "gguf",
        "title": "llama.cpp GGUF Q4_K_M (+ correct mmproj)",
        "status": status,
        "missing": missing,
        "merged": project_relative(merged_path),
        "output_dir": project_relative(output_dir),
        "artifacts_expected": [str(q4).replace(str(PROJECT_ROOT) + "\\", "")],
        "commands": commands,
        "requires_gpu": False,
        "requires_download": bool(missing),
        "estimated_minutes": _route_estimate("gguf"),
        "acceptance": [
            "F16 GGUF and a *mmproj*.gguf are produced",
            "Q4_K_M GGUF is produced and smaller than the F16 file",
            "llama-mtmd-cli on one fixed image emits four parseable coordinates",
            "sha256 + byte size of every produced file recorded",
        ],
        "notes": [
            "SmolVLM/Idefics3 support and the exact --mmproj flow must be smoke-tested in the first 10 minutes",
            "no GGUF artifact may be quoted as 4-bit until the smoke generation produces a box",
        ],
        "fallback": "onnx -> torch-int8 -> hf-fp32",
    }


def build_onnx_plan(merged_path: Path, output_dir: Path, *, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    packages_payload = packages()
    missing = [name for name in ("optimum", "onnx", "onnxruntime") if not packages_payload[name]["available"]]
    onnx_dir = output_dir / "onnx"
    commands = [
        {
            "step": "export_onnx",
            "argv": [
                PYTHON,
                "-m",
                "optimum.commands.optimum_cli",
                "export",
                "onnx",
                "--model",
                str(merged_path),
                str(onnx_dir),
            ],
            "cwd": str(PROJECT_ROOT),
            "expected_outputs": [str(onnx_dir)],
            "timeout_seconds": 2400,
            "note": (
                "Idefics3 image-text-to-text export support under transformers 5.19/optimum "
                "is unverified; a 10-minute export smoke decides whether this route exists"
            ),
        },
        {
            "step": "quantize_dynamic_int8",
            "argv": [
                PYTHON,
                "-m",
                "src.deploy.quantize",
                "--route",
                "onnx",
                "--quantize-only",
                "--output",
                str(output_dir),
                "--merged",
                str(merged_path),
            ],
            "cwd": str(PROJECT_ROOT),
            "expected_outputs": [],
            "timeout_seconds": 1800,
            "note": "quantize_dynamic on MatMul weights; writes *_int8.onnx next to the exported graph",
        },
    ]
    return {
        "route": "onnx",
        "title": "ONNX export + dynamic INT8 quantization",
        "status": "ready" if not missing else "blocked_tool_missing",
        "missing": missing,
        "merged": project_relative(merged_path),
        "output_dir": project_relative(output_dir),
        "commands": commands,
        "requires_gpu": False,
        "requires_download": bool(missing),
        "estimated_minutes": _route_estimate("onnx"),
        "acceptance": [
            "optimum-cli export completes for the merged checkpoint",
            "quantize_dynamic produces a smaller *_int8.onnx",
            "onnxruntime CPU InferenceSession loads the quantized graph",
            "a runtime smoke produces a parseable box",
        ],
        "notes": [
            "pre-existing models/SmolVLM-256M-Instruct/onnx files are the BASE model without LoRA; never use them as the deployed artifact",
            "if optimum cannot export Idefics3, record blocked_reason and fall through to torch-int8",
        ],
        "fallback": "torch-int8 -> hf-fp32",
    }


def build_torch_int8_plan(merged_path: Path) -> dict[str, Any]:
    return {
        "route": "torch-int8",
        "title": "PyTorch INT8 dynamic quantization of the text decoder (CPU, no export)",
        "status": "ready",
        "missing": [],
        "merged": project_relative(merged_path),
        "output_dir": None,
        "commands": [
            {
                "step": "quantize_and_smoke",
                "argv": [
                    PYTHON,
                    "-m",
                    "src.deploy.quantize",
                    "--route",
                    "torch-int8",
                    "--merged",
                    str(merged_path),
                    "--report",
                    "results/deploy_quantize_torch_int8.json",
                ],
                "cwd": str(PROJECT_ROOT),
                "expected_outputs": ["results/deploy_quantize_torch_int8.json"],
                "timeout_seconds": 1800,
            }
        ],
        "requires_gpu": False,
        "requires_download": False,
        "estimated_minutes": _route_estimate("torch-int8"),
        "acceptance": [
            "quantize_dynamic replaces text-decoder Linear layers with qint8 dynamic modules",
            "generate() still returns four parseable coordinates on a fixed image",
            "artifact_type recorded as runtime_only (no file artifact)",
        ],
        "notes": [
            "this is real INT8 dynamic quantization, but it is not GGUF/Q4_K_M and must not be labelled 4-bit",
            "no file artifact: the quantized modules exist only in the benchmark process; reproducibility comes from the script",
        ],
        "fallback": "hf-fp32",
    }


def build_fp32_plan(merged_path: Path) -> dict[str, Any]:
    return {
        "route": "fp32",
        "title": "CPU fp32 HuggingFace fallback (no quantization)",
        "status": "ready",
        "missing": [],
        "merged": project_relative(merged_path),
        "output_dir": None,
        "commands": [],
        "requires_gpu": False,
        "requires_download": False,
        "estimated_minutes": _route_estimate("fp32"),
        "acceptance": ["merged fp32 checkpoint runs on CPU; latency benchmark produces real numbers"],
        "notes": ["fallback only: '4-bit speed-up' must move to Limitations if this is the deployed artifact"],
        "fallback": None,
    }


def build_route_plan(route: str, merged_path: Path, output_dir: Path, *, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    if route == "gguf":
        return build_gguf_plan(merged_path, output_dir, llama=(environment or {}).get("tools", {}).get("llama_cpp"))
    if route == "onnx":
        return build_onnx_plan(merged_path, output_dir, environment=environment)
    if route == "torch-int8":
        return build_torch_int8_plan(merged_path)
    if route == "fp32":
        return build_fp32_plan(merged_path)
    raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")


# ---------------------------------------------------------------- execution
def _run_command(command: dict[str, Any], *, log_path: Path) -> dict[str, Any]:
    argv = [str(token) for token in command["argv"]]
    cwd = command.get("cwd") or str(PROJECT_ROOT)
    timeout = int(command.get("timeout_seconds") or 1800)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n=== {now_iso()} step={command.get('step')} ===\n")
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.write(f"exception: {type(exc).__name__}: {exc}\n")
            return {
                "step": command.get("step"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
                "log": project_relative(log_path),
            }
        log.write("--- stdout ---\n")
        log.write(completed.stdout or "")
        log.write("--- stderr ---\n")
        log.write(completed.stderr or "")
        log.write(f"\nexit_code={completed.returncode}\n")
    return {
        "step": command.get("step"),
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
        "log": project_relative(log_path),
    }


def _discover_files(directory: Path, patterns: Sequence[str]) -> list[Path]:
    found: set[Path] = set()
    if not directory.is_dir():
        return []
    for pattern in patterns:
        found.update(directory.rglob(pattern))
    return sorted(path for path in found if path.is_file())


def execute_gguf(plan: dict[str, Any], *, log_path: Path) -> dict[str, Any]:
    if plan["status"] != "ready":
        return {"status": "blocked_tool_missing", "missing": plan["missing"], "steps": [], "artifacts": []}
    output_dir = resolve_project_path(plan["output_dir"])
    steps: list[dict[str, Any]] = []
    for command in plan["commands"]:
        if command["step"] == "smoke_mmproj_generation":
            break  # handled after file discovery
        steps.append(_run_command(command, log_path=log_path))
        if steps[-1]["status"] != "completed":
            return {"status": "failed", "steps": steps, "artifacts": []}
    produced = _discover_files(output_dir, ("*.gguf",))
    mmproj = [path for path in produced if "mmproj" in path.name.lower()]
    quantized = [path for path in produced if "q4_k_m" in path.name.lower() or "q4_k" in path.name.lower()]
    artifacts = [
        {
            "path": project_relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in produced
    ]
    return {
        "status": "completed" if quantized else "failed",
        "steps": steps,
        "artifacts": artifacts,
        "mmproj": [project_relative(path) for path in mmproj],
        "quantized": [project_relative(path) for path in quantized],
        "note": "smoke generation is a separate step; a GGUF without a passing smoke is not a usable deployment artifact",
    }


def execute_onnx(plan: dict[str, Any], *, log_path: Path) -> dict[str, Any]:
    if plan["status"] != "ready":
        return {"status": "blocked_tool_missing", "missing": plan["missing"], "steps": [], "artifacts": []}
    steps: list[dict[str, Any]] = []
    for command in plan["commands"]:
        steps.append(_run_command(command, log_path=log_path))
        if steps[-1]["status"] != "completed":
            return {"status": "failed", "steps": steps, "artifacts": []}
    output_dir = resolve_project_path(plan["output_dir"])
    produced = _discover_files(output_dir, ("*.onnx",))
    return {
        "status": "completed" if produced else "failed",
        "steps": steps,
        "artifacts": [
            {"path": project_relative(path), "bytes": path.stat().st_size} for path in produced
        ],
    }


def quantize_exported_onnx(output_dir: Path) -> dict[str, Any]:
    """Quantize every exported ONNX graph with dynamic INT8 MatMul quantization."""

    from onnxruntime.quantization import QuantType, quantize_dynamic

    graphs = _discover_files(output_dir, ("*.onnx",))
    graphs = [path for path in graphs if "_int8" not in path.stem]
    if not graphs:
        raise FileNotFoundError(f"no ONNX graph found under {output_dir} to quantize")
    results: list[dict[str, Any]] = []
    for graph in graphs:
        target = graph.with_name(graph.stem + "_int8.onnx")
        quantize_dynamic(
            model_input=str(graph),
            model_output=str(target),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            per_channel=False,
        )
        results.append(
            {
                "source": project_relative(graph),
                "target": project_relative(target),
                "source_bytes": graph.stat().st_size,
                "target_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return {
        "status": "completed",
        "graphs": results,
        "weight_type": "QInt8",
        "op_types": ["MatMul"],
        "per_channel": False,
        "caveat": "quantizes exported graphs only; a runtime adapter for the split VLM pipeline is still required for end-to-end inference",
    }


def execute_torch_int8(merged_path: Path) -> dict[str, Any]:
    """Real INT8 dynamic quantization of the text decoder, in-process, CPU only."""

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from PIL import Image

    from .infer import HFPredictor
    from .config import EXPECTED_IMAGE_SEQ_LEN, read_jsonl

    started = time.perf_counter()
    records = read_jsonl(PROJECT_ROOT / "data" / "synthetic_eval.jsonl", limit=1)
    record = records[0]
    with HFPredictor(merged=merged_path, device="cpu", dtype="float32") as predictor:
        text_model = predictor.model.model.text_model
        try:
            from torch.ao.quantization import quantize_dynamic

            quantized_text = quantize_dynamic(text_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        except Exception as exc:  # pragma: no cover - execution only
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "route": "torch-int8"}
        predictor.model.model.text_model = quantized_text
        with Image.open(PROJECT_ROOT / record["image"]) as opened:
            image = opened.convert("RGB")
        output = predictor.predict(image, record["question"])
    return {
        "status": "completed",
        "route": "torch-int8",
        "artifact_type": "runtime_only",
        "artifact_path": None,
        "quantized_scope": "model.model.text_model Linear -> qint8 dynamic",
        "dtype": "qint8",
        "image_size": IMAGE_SIZE,
        "image_seq_len": EXPECTED_IMAGE_SEQ_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sample": {"id": record["id"], "question": record["question"], "output_text": output["output_text"], "predicted_box": output["predicted_box"]},
        "parseable": output["predicted_box"] is not None,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_memory_mib": peak_process_memory_mib(),
        "caveat": "runtime-only artifact; re-running this command reproduces the quantized model",
    }


# ------------------------------------------------------------------ handoff
def build_handoff(
    merged_path: Path,
    *,
    quantized_route: str | None = None,
    quantized_artifact: str | None = None,
) -> dict[str, Any]:
    """Machine-readable contract the independent evaluator can follow."""

    return {
        "schema_version": 1,
        "purpose": "independent evaluation of the deployed model by eval-closer",
        "generated_at": now_iso(),
        "deployment_candidate_provisional": True,
        "provisional_reason": "T12/KD not run; final adapter not frozen by the Lead",
        "hf_checkpoint": {
            "path": project_relative(merged_path),
            "exists": merged_path.is_dir(),
            "evaluator_command_examples": [
                (
                    f"{PYTHON} -m src.eval.run_zeroshot --run-id <RUN_ID> --dataset synthetic "
                    f"--split eval --data data/synthetic_eval.jsonl --output results/zeroshot_<RUN_ID>.json "
                    f"--details-output results/eval_<RUN_ID>_details.jsonl --expected-n 1000 "
                    f"--model-path {project_relative(merged_path)} --image-size 384 --max-new-tokens 64 "
                    f"--batch-size 1 --seed 42 --device cpu"
                ),
                (
                    f"{PYTHON} -m src.eval.run_zeroshot --run-id <RUN_ID> --dataset refcoco "
                    f"--split val --data data/refcoco_val.jsonl --output results/zeroshot_<RUN_ID>.json "
                    f"--details-output results/eval_<RUN_ID>_details.jsonl --expected-n 1000 "
                    f"--model-path {project_relative(merged_path)} --image-size 384 --max-new-tokens 64 "
                    f"--batch-size 1 --seed 42 --device cpu"
                ),
            ],
            "note": (
                "src.eval.run_zeroshot loads any full checkpoint through --model-path, so the merged "
                "deployment candidate can be scored with the frozen evaluator unchanged."
            ),
        },
        "quantized_checkpoint": {
            "route": quantized_route,
            "artifact": quantized_artifact,
            "official_metrics_status": "not_run",
            "note": (
                "GGUF/ONNX runtime inference is not an evaluator code path. If a quantized engine works, "
                "its accuracy must be reported as a deployment-harness diagnostic plus an independent "
                "spot-check, or the +量化 accuracy row must be marked provisional/not_run. Never copy "
                "deploy-harness numbers into Table 1 as official metrics."
            ),
        },
        "inference_contract": {
            "prompt_id": "p1",
            "prompt_source": "src.eval.prompts.build_messages",
            "image_size": IMAGE_SIZE,
            "do_image_splitting": False,
            "image_seq_len": EXPECTED_IMAGE_SEQ_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "num_beams": 1,
            "seed": 42,
            "parser": "src.eval.grounding.parse_box_with_error",
        },
        "latency_plan": {
            "warmup": 5,
            "n": 50,
            "selection": "src.deploy.config.select_benchmark_records(seed=42)",
            "output_files": "results/deploy_latency_*.json",
            "estimate": benchmark_time_estimate(),
        },
    }


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=ROUTES, default="gguf")
    parser.add_argument("--merged", default=str(DEFAULT_MERGED))
    parser.add_argument("--output", default=None, help="quantized artifact directory (default per route)")
    parser.add_argument("--report", default=None, help="JSON report under results/deploy_* (default per route)")
    parser.add_argument("--llama-cpp-dir", default=None, help="llama.cpp checkout/bin directory")
    parser.add_argument("--plan-only", action="store_true", help="print the plan; no torch import, no write")
    parser.add_argument("--quantize-only", action="store_true", help="quantize already-exported ONNX graphs")
    parser.add_argument("--emit-handoff", action="store_true", help="write results/deploy_handoff.json")
    parser.add_argument("--quantized-route", default=None, help="handoff: which quantized route actually succeeded")
    parser.add_argument("--quantized-artifact", default=None, help="handoff: path of the quantized artifact")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    return parser


def default_output(route: str) -> Path:
    return {
        "gguf": DEFAULT_QUANTIZED_GGUF,
        "onnx": DEFAULT_ONNX_OUT,
        "torch-int8": PROJECT_ROOT / "adapters" / "quantized" / "sft_v1_torch_int8_runtime",
        "fp32": PROJECT_ROOT / "adapters" / "quantized" / "sft_v1_fp32",
    }[route]


def default_report(route: str) -> str:
    return {
        "gguf": "results/deploy_quantize_gguf.json",
        "onnx": "results/deploy_quantize_onnx.json",
        "torch-int8": "results/deploy_quantize_torch_int8.json",
        "fp32": "results/deploy_quantize_fp32.json",
    }[route]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merged_path = resolve_project_path(args.merged)
    output_dir = resolve_project_path(args.output) if args.output else default_output(args.route)
    if args.llama_cpp_dir:
        os.environ["LLAMA_CPP_DIR"] = str(resolve_project_path(args.llama_cpp_dir))
    report_default = "results/deploy_handoff.json" if args.emit_handoff else default_report(args.route)
    report_path = checked_results_path(args.report or report_default, "deploy_")

    if args.emit_handoff:
        payload = build_handoff(
            merged_path,
            quantized_route=args.quantized_route,
            quantized_artifact=args.quantized_artifact,
        )
        payload["merged_facts"] = merged_facts(merged_path)
        payload["quantized_facts"] = quantized_facts()
        write_json_atomic(report_path, payload)
        print(json.dumps({"status": "handoff_written", "report": project_relative(report_path)}, ensure_ascii=False))
        return 0

    merged = classify_artifact(merged_path, model_path=MODEL_PATH)
    if not merged["usable"]:
        raise FileNotFoundError(f"merged checkpoint is unusable: {merged}")

    plan = build_route_plan(args.route, merged_path, output_dir)
    plan["merged_artifact"] = merged
    plan["report_path"] = project_relative(report_path)
    plan["plan_only"] = bool(args.plan_only)

    if args.plan_only:
        print(json.dumps({"status": "plan_only", **plan}, ensure_ascii=False, indent=2))
        return 0

    if args.quantize_only:
        if args.route != "onnx":
            raise ValueError("--quantize-only is only valid with --route onnx")
        payload = quantize_exported_onnx(output_dir)
        payload.update({"started_at": now_iso(), "plan": plan})
        write_json_atomic(report_path, payload)
        print(json.dumps({"status": payload["status"], "report": project_relative(report_path)}, ensure_ascii=False))
        return 0 if payload["status"] == "completed" else 2

    started = time.perf_counter()
    if args.route == "gguf":
        log_path = PROJECT_ROOT / "results" / "deploy_gguf_build.log"
        result = execute_gguf(plan, log_path=log_path)
    elif args.route == "onnx":
        log_path = PROJECT_ROOT / "results" / "deploy_onnx_build.log"
        result = execute_onnx(plan, log_path=log_path)
    elif args.route == "torch-int8":
        log_path = None
        result = execute_torch_int8(merged_path)
    else:
        log_path = None
        result = {"status": "completed", "route": "fp32", "artifact_type": "checkpoint", "artifact_path": project_relative(merged_path)}
    payload = {
        "started_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - started,
        "plan": plan,
        **result,
        "environment": {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "onnxruntime": _version("onnxruntime"),
            "optimum": _version("optimum"),
        },
        "claims_note": "only routes that actually ran appear under artifacts; everything else is not_run",
    }
    write_json_atomic(report_path, payload)
    print(json.dumps({"status": payload["status"], "route": args.route, "report": project_relative(report_path)}, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "completed" else 2


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
