"""CPU-only T14/T15 prep probe.

Writes ``results/deploy_prep.json``: tool availability, repository assets,
executable step plan, time estimates, risks, fallbacks and explicit ``not_run``
claims.  It never imports torch/transformers, never loads a model, never touches
CUDA and never downloads anything.

Usage::

    python -m src.deploy.check_env
    python -m src.deploy.check_env --output results/deploy_prep.json --print-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    checked_results_path,
    project_relative,
    write_json_atomic,
)
from .env import collect_environment
from .plan import build_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/deploy_prep.json",
        help="target JSON under results/ (must start with deploy_)",
    )
    parser.add_argument("--print-summary", action="store_true", help="print a compact summary after writing")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing prep JSON")
    return parser


def compact_summary(document: dict[str, Any]) -> dict[str, Any]:
    environment = document["environment"]
    packages = environment["packages"]
    return {
        "status": document["status"],
        "output_phase": document["phase"],
        "captured_at": document["generated_at"],
        "deadline_remaining_minutes": document["deadline"]["remaining_minutes_at_prep"],
        "gpu": {
            "state": environment["gpu"].get("state"),
            "memory_used_mib": environment["gpu"].get("memory_used_mib"),
            "memory_total_mib": environment["gpu"].get("memory_total_mib"),
            "utilization_gpu_percent": environment["gpu"].get("utilization_gpu_percent"),
        },
        "tools": {
            "torch": packages["torch"]["version"],
            "transformers": packages["transformers"]["version"],
            "peft": packages["peft"]["version"],
            "onnxruntime": packages["onnxruntime"]["version"],
            "gradio": packages["gradio"]["version"],
            "optimum": packages["optimum"]["version"],
            "gguf": packages["gguf"]["version"],
            "llama_cpp": "available" if environment["tools"]["llama_cpp"]["available"] else "missing",
            "cmake": environment["tools"]["llama_cpp"]["build_tools"]["cmake"],
        },
        "assets": {
            "model_present": document["assets"]["model"]["all_required_files_present"],
            "model_bytes": document["assets"]["model"]["weights"]["bytes"],
            "adapter_present": document["assets"]["adapter"]["all_required_files_present"],
            "adapter_sha256": document["assets"]["adapter"]["adapter_weights"]["sha256"],
            "lora_r": document["assets"]["adapter"]["r"],
            "merged_exists": document["assets"]["merged"]["exists"],
            "quantized_status": document["assets"]["quantized"]["status"],
            "upstream_onnx_files": len(document["assets"]["upstream_onnx"]["files"]),
            "upstream_onnx_carries_lora": document["assets"]["upstream_onnx"]["carries_project_lora"],
        },
        "fallback_chain": document["fallback_chain"],
        "routes": {
            route: {"status": payload["status"], "missing": payload.get("missing", [])}
            for route, payload in document["routes"].items()
        },
        "estimated_total_minutes": document["estimated_total_minutes"],
        "claims": document["claims"],
        "gap_count": len(document["gaps"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if "torch" in sys.modules:
        raise RuntimeError("prep probe must not import torch; aborting to protect the GPU lease")
    output = checked_results_path(args.output, "deploy_")
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    environment = collect_environment()
    document = build_plan(environment)
    document["output_path"] = project_relative(output)
    document["written_at"] = document["generated_at"]
    write_json_atomic(output, document)

    summary = compact_summary(document)
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "prep_complete", "output": project_relative(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
