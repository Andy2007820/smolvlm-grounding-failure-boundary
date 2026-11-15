"""Deterministic T14/T15 execution plan, risks, fallbacks and honest claims.

The plan is built from live, read-only environment detection.  It is the single
source of truth for the hand-off document ``results/deploy_prep.json`` and for
the code skeletons that will be executed after the Lead lifts the CPU/GPU
restriction.  Nothing in this module executes a heavy step.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

from .config import (
    BENCH_SAMPLES,
    BENCH_SEED,
    BENCH_WARMUP,
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    DEFAULT_ONNX_OUT,
    DEFAULT_QUANTIZED_GGUF,
    DEADLINE_LOCAL_ISO,
    EXPECTED_IMAGE_SEQ_LEN,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    PROJECT_ROOT,
    PROVISIONAL_DEPLOYMENT_CANDIDATE,
    QUANTIZED_ROOT,
    benchmark_time_estimate,
    deadline_remaining_minutes,
    project_relative,
)
from .env import (
    adapter_facts,
    collect_environment,
    dataset_facts,
    merged_facts,
    model_facts,
    onnx_asset_inventory,
    prep_time_estimates,
    quantized_facts,
    summarize_missing_tools,
)

PYTHON = sys.executable or "python"

FALLBACK_CHAIN = ("gguf", "onnx", "torch-int8", "hf-fp32")

ROUTE_TITLES = {
    "gguf": "llama.cpp GGUF Q4_K_M (+ mmproj)",
    "onnx": "ONNX export + dynamic quantization",
    "torch-int8": "PyTorch INT8 dynamic quantization (CPU, no export)",
    "hf-fp32": "CPU fp32 HuggingFace fallback (no quantization)",
}


def _command(module: str, args: Sequence[str]) -> dict[str, Any]:
    argv = [PYTHON, "-m", module, *args]
    return {"argv": argv, "display": subprocess_display(argv)}


def subprocess_display(argv: Sequence[str]) -> str:
    """Windows-friendly copy-paste rendering; quoting is display-only."""

    rendered = []
    for token in argv:
        rendered.append(f'"{token}"' if " " in token else str(token))
    return " ".join(rendered)


def route_statuses(environment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages = environment["packages"]
    llama = environment["tools"]["llama_cpp"]

    def available(module: str) -> bool:
        return bool(packages.get(module, {}).get("available"))

    gguf_missing = list(llama["missing"])
    if not available("gguf"):
        gguf_missing.append("python package gguf")
    if not available("sentencepiece"):
        gguf_missing.append("python package sentencepiece (convert script may need it)")
    onnx_missing = [name for name in ("optimum", "onnx") if not available(name)]
    return {
        "gguf": {
            "route": "gguf",
            "title": ROUTE_TITLES["gguf"],
            "status": "ready" if not gguf_missing else "needs_tooling_and_download",
            "missing": gguf_missing,
            "gpu_required": False,
            "download_required": True,
            "download_note": "prebuilt llama.cpp Windows CPU release (approval needed) + pip gguf/sentencepiece",
            "reason": "preferred route per TEAM-BOOTSTRAP; Q4_K_M with a correct mmproj",
        },
        "onnx": {
            "route": "onnx",
            "title": ROUTE_TITLES["onnx"],
            "status": "ready" if not onnx_missing else "conditional",
            "missing": onnx_missing,
            "gpu_required": False,
            "download_required": bool(onnx_missing),
            "download_note": "pip install optimum[onnxruntime] onnx",
            "reason": (
                "official fallback, but Idefics3 ONNX export support under "
                "transformers 5.19/optimum is unverified; a 10-minute export smoke "
                "must pass before promising this route"
            ),
        },
        "torch-int8": {
            "route": "torch-int8",
            "title": ROUTE_TITLES["torch-int8"],
            "status": "ready",
            "missing": [],
            "gpu_required": False,
            "download_required": False,
            "download_note": "none; torch.ao.quantization is bundled",
            "reason": (
                "real INT8 dynamic quantization of the merged model without an export "
                "step; artifact is runtime-only (in-memory), openly labelled as such"
            ),
        },
        "hf-fp32": {
            "route": "hf-fp32",
            "title": ROUTE_TITLES["hf-fp32"],
            "status": "ready",
            "missing": [],
            "gpu_required": False,
            "download_required": False,
            "download_note": "none",
            "reason": "final fallback: runnable but slow; 4-bit speed-up goes to Limitations",
        },
    }


def build_steps(environment: dict[str, Any]) -> list[dict[str, Any]]:
    merged_display = project_relative(DEFAULT_MERGED)
    quantized_display = project_relative(DEFAULT_QUANTIZED_GGUF)
    onnx_display = project_relative(DEFAULT_ONNX_OUT)
    adapter_display = project_relative(DEFAULT_ADAPTER)
    steps: list[dict[str, Any]] = [
        {
            "id": "S1",
            "title": "Merge LoRA adapter into a standalone fp32 CPU checkpoint",
            "owner": "deploy-runner",
            "depends_on": [],
            "requires_gpu": False,
            "requires_model_load": True,
            "requires_download": False,
            "requires_write_scope": ["adapters/merged", "results/deploy_merge_report.json"],
            "estimated_minutes": 6,
            "command": _command(
                "src.deploy.merge_lora",
                [
                    "--adapter",
                    adapter_display,
                    "--output",
                    merged_display,
                    "--dtype",
                    "float32",
                    "--report",
                    "results/deploy_merge_report.json",
                ],
            ),
            "dry_run_command": _command(
                "src.deploy.merge_lora",
                ["--adapter", adapter_display, "--output", merged_display, "--dry-run"],
            ),
            "acceptance": [
                "report status=completed with adapter + base hashes",
                "merged dir has model.safetensors + config + processor/tokenizer files",
                "no remaining lora_* modules in the merged checkpoint config",
                "model.safetensors size is ~2x the bf16 base weight size (fp32 merge)",
            ],
            "artifacts": [merged_display, "results/deploy_merge_report.json"],
            "on_failure": "stop; report blocker; never fabricate a merged artifact",
            "blocking_scope_note": (
                "adapters/merged is outside this prep task's write scope; the Lead must "
                "confirm the T14 write scope before execution"
            ),
        },
        {
            "id": "S2",
            "title": "Merge equivalence: base+adapter vs merged must generate identical boxes",
            "owner": "deploy-runner",
            "depends_on": ["S1"],
            "requires_gpu": False,
            "requires_model_load": True,
            "requires_download": False,
            "estimated_minutes": 8,
            "command": _command(
                "src.deploy.merge_lora",
                [
                    "--verify-equivalence",
                    "--artifact",
                    merged_display,
                    "--report",
                    "results/deploy_merge_equivalence.json",
                ],
            ),
            "acceptance": [
                "5 fixed synthetic_eval samples produce byte-identical output_text",
                "parse_rate 1.0 on those 5 samples",
                "any mismatch is reported as merge_failed, not averaged away",
            ],
            "artifacts": ["results/deploy_merge_equivalence.json"],
            "on_failure": "do not deploy the merged model; investigate dtype/merge target mismatch",
        },
        {
            "id": "S3",
            "title": "GGUF Q4_K_M export smoke (preferred route)",
            "owner": "deploy-runner",
            "depends_on": ["S1", "S2"],
            "requires_gpu": False,
            "requires_model_load": False,
            "requires_download": True,
            "estimated_minutes": 35,
            "command": _command(
                "src.deploy.quantize",
                [
                    "--route",
                    "gguf",
                    "--merged",
                    merged_display,
                    "--output",
                    quantized_display,
                    "--report",
                    "results/deploy_quantize_gguf.json",
                ],
            ),
            "plan_only_command": _command(
                "src.deploy.quantize", ["--route", "gguf", "--plan-only", "--merged", merged_display]
            ),
            "acceptance": [
                "convert_hf_to_gguf.py produced an F16 GGUF and a *mmproj*.gguf",
                "llama-quantize produced a Q4_K_M GGUF",
                "llama-mtmd-cli on one fixed image emits 4 parseable box numbers",
                "file sizes and sha256 recorded in results/deploy_quantize_gguf.json",
            ],
            "artifacts": [quantized_display, "results/deploy_quantize_gguf.json"],
            "on_failure": "capture the exact error, then fall through to S4 (ONNX) or S5 (torch INT8)",
            "setup_note": (
                "first 10 minutes are a tool/protocol smoke (convert --help, mmproj naming, "
                "mtmd-cli flags) because llama.cpp CLI names drift between releases"
            ),
        },
        {
            "id": "S4",
            "title": "ONNX export + dynamic quantization (conditional fallback)",
            "owner": "deploy-runner",
            "depends_on": ["S3"],
            "requires_gpu": False,
            "requires_model_load": False,
            "requires_download": True,
            "estimated_minutes": 25,
            "command": _command(
                "src.deploy.quantize",
                [
                    "--route",
                    "onnx",
                    "--merged",
                    merged_display,
                    "--output",
                    onnx_display,
                    "--report",
                    "results/deploy_quantize_onnx.json",
                ],
            ),
            "acceptance": [
                "optimum-cli export onnx succeeds for Idefics3 (else route recorded blocked)",
                "quantize_dynamic halves or better the decoder artifact size",
                "onnxruntime CPU session loads the quantized graph",
            ],
            "artifacts": [onnx_display, "results/deploy_quantize_onnx.json"],
            "on_failure": "record blocked_tool_missing / unsupported_architecture and fall through to S5",
            "scientific_note": (
                "the ONNX files already present under models/SmolVLM-256M-Instruct/onnx are the "
                "upstream BASE model (no project LoRA); they must not be presented as our quantized model"
            ),
        },
        {
            "id": "S5",
            "title": "PyTorch INT8 dynamic quantization (export-free fallback)",
            "owner": "deploy-runner",
            "depends_on": ["S4"],
            "requires_gpu": False,
            "requires_model_load": True,
            "requires_download": False,
            "estimated_minutes": 8,
            "command": _command(
                "src.deploy.quantize",
                [
                    "--route",
                    "torch-int8",
                    "--merged",
                    merged_display,
                    "--report",
                    "results/deploy_quantize_torch_int8.json",
                ],
            ),
            "acceptance": [
                "quantize_dynamic on the text decoder succeeds and generate() returns a parseable box",
                "runtime label recorded as int8-dynamic-cpu (in-memory artifact, no file)",
            ],
            "artifacts": ["results/deploy_quantize_torch_int8.json"],
            "on_failure": "fall through to S6 CPU fp32; report quantization as not achieved",
        },
        {
            "id": "S6",
            "title": "CPU latency benchmark: merged fp32 (50 images, 5 warmups)",
            "owner": "deploy-runner",
            "depends_on": ["S1", "S2"],
            "requires_gpu": False,
            "requires_model_load": True,
            "requires_download": False,
            "estimated_minutes": 4,
            "command": _command(
                "src.deploy.bench_latency",
                [
                    "--engine",
                    "hf-cpu",
                    "--artifact",
                    merged_display,
                    "--n",
                    str(BENCH_SAMPLES),
                    "--warmup",
                    str(BENCH_WARMUP),
                    "--seed",
                    str(BENCH_SEED),
                    "--output",
                    "results/deploy_latency_hf_cpu_fp32.json",
                    "--details-output",
                    "results/deploy_latency_hf_cpu_fp32_details.jsonl",
                ],
            ),
            "plan_only_command": _command(
                "src.deploy.bench_latency",
                ["--engine", "hf-cpu", "--artifact", merged_display, "--plan-only", "--output", "results/deploy_latency_plan_hf_cpu.json"],
            ),
            "acceptance": [
                "n=50, warmup=5, deterministic id digest",
                "mean/median/p95/p99/min/max + peak RSS recorded",
                "parse_rate and diagnostic accuracy recorded (official metrics remain eval-closer's)",
            ],
            "artifacts": ["results/deploy_latency_hf_cpu_fp32.json"],
            "on_failure": "record not_run with the error; no invented latency numbers",
        },
        {
            "id": "S7",
            "title": "CPU latency benchmark: quantized and/or GGUF engine on the same 50 ids",
            "owner": "deploy-runner",
            "depends_on": ["S3", "S5", "S6"],
            "requires_gpu": False,
            "requires_model_load": False,
            "requires_download": False,
            "estimated_minutes": 6,
            "command": _command(
                "src.deploy.bench_latency",
                [
                    "--engine",
                    "gguf-q4",
                    "--artifact",
                    quantized_display,
                    "--n",
                    str(BENCH_SAMPLES),
                    "--warmup",
                    str(BENCH_WARMUP),
                    "--seed",
                    str(BENCH_SEED),
                    "--output",
                    "results/deploy_latency_gguf_q4.json",
                ],
            ),
            "acceptance": [
                "same selected_ids digest as S6 so the CPU/quantized columns are comparable",
                "per-run backend version and artifact sha256 recorded",
            ],
            "artifacts": ["results/deploy_latency_gguf_q4.json"],
            "on_failure": "benchmark whatever quantized engine actually exists; list the missing route as not_run",
        },
        {
            "id": "S8",
            "title": "GPU bf16 baseline latency (requires Lead GPU lease)",
            "owner": "deploy-runner",
            "depends_on": ["S6"],
            "requires_gpu": True,
            "requires_model_load": True,
            "requires_download": False,
            "estimated_minutes": 10,
            "command": _command(
                "src.deploy.bench_latency",
                [
                    "--engine",
                    "hf-cuda",
                    "--artifact",
                    merged_display,
                    "--device",
                    "cuda",
                    "--gpu-lease-id",
                    "<LEASE-ID>",
                    "--n",
                    str(BENCH_SAMPLES),
                    "--warmup",
                    str(BENCH_WARMUP),
                    "--output",
                    "results/deploy_latency_hf_cuda_bf16.json",
                ],
            ),
            "acceptance": [
                "lease id recorded; nvidia-smi before/after; abort if >7500 MiB or a competing task appears",
                "same 50 ids as S6/S7 for a fair Table 3 row",
            ],
            "artifacts": ["results/deploy_latency_hf_cuda_bf16.json"],
            "on_failure": "skip the GPU column; never report an unmeasured number",
            "gating_note": "blocked until the Lead announces the current evaluation lease is released",
        },
        {
            "id": "S9",
            "title": "Gradio demo: install, launch, real API round-trip, screenshot evidence",
            "owner": "deploy-runner",
            "depends_on": ["S6"],
            "requires_gpu": False,
            "requires_model_load": True,
            "requires_download": True,
            "estimated_minutes": 20,
            "command": _command(
                "demo.app",
                [
                    "--engine",
                    "hf-cpu",
                    "--artifact",
                    merged_display,
                    "--port",
                    "7860",
                    "--smoke",
                ],
            ),
            "acceptance": [
                "gradio installed (needs approval) and server answers HTTP 200",
                "gradio_client round-trip on a real image returns an annotated image and a box string",
                "results/deploy_demo_smoke.json + saved annotated PNG under demo/artifacts/",
                "screenshot/browser confirmation recorded by the Lead (agent cannot claim a human-visible UI without it)",
            ],
            "artifacts": ["demo/app.py", "results/deploy_demo_smoke.json", "demo/artifacts/"],
            "on_failure": "fallback: CLI predictor script + annotated PNG (still honest, but G7 demo part not passed)",
            "gating_note": "requires `python -m pip install gradio` approval (network download)",
        },
        {
            "id": "S10",
            "title": "Hand-off document for independent deployment evaluation",
            "owner": "deploy-runner -> eval-closer",
            "depends_on": ["S3", "S5", "S6", "S7"],
            "requires_gpu": False,
            "requires_model_load": False,
            "requires_download": False,
            "estimated_minutes": 10,
            "command": _command(
                "src.deploy.quantize",
                ["--route", "gguf", "--emit-handoff", "--output", "results/deploy_handoff.json"],
            ),
            "acceptance": [
                "artifact paths + sha256 + exact inference command + prompt id p1 + image_seq_len 36 documented",
                "eval-closer can rerun 1000-image sets without reading deploy code",
                "official +量化 metrics are produced by eval-closer, not by deploy code",
            ],
            "artifacts": ["results/deploy_handoff.json"],
            "on_failure": "block G7; paper must mark deployment row not_run",
        },
    ]
    prior = benchmark_time_estimate()
    for step in steps:
        step["estimated_seconds"] = int(step["estimated_minutes"] * 60)
    steps[5]["estimated_minutes"] = round(prior["total_seconds_upper_bound"] / 60.0, 1)
    steps[5]["estimated_seconds"] = int(prior["total_seconds_upper_bound"])
    return steps


def build_risks(environment: dict[str, Any], routes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "R1",
            "risk": "llama.cpp conversion does not support SmolVLM/Idefics3 in the available release, or mmproj flags differ",
            "probability": "medium",
            "impact": "high (preferred route lost)",
            "mitigation": "10-minute convert smoke first; record exact error; fall back to ONNX then torch-int8 then CPU fp32",
            "owner": "deploy-runner",
        },
        {
            "id": "R2",
            "risk": "no llama.cpp binaries / cmake / MSVC compiler on the machine",
            "probability": "high (confirmed missing)",
            "impact": "high",
            "mitigation": "unpack a prebuilt Windows CPU release (needs approval) or skip GGUF; do not silently claim 4-bit",
            "owner": "Lead (download approval) + deploy-runner",
        },
        {
            "id": "R3",
            "risk": "ONNX export of Idefics3 via optimum fails under transformers 5.19",
            "probability": "medium-high",
            "impact": "medium",
            "mitigation": "record blocked status; use torch INT8 dynamic quantization (real quantization, no export) as the next fallback",
            "owner": "deploy-runner",
        },
        {
            "id": "R4",
            "risk": "the pre-existing ONNX files are the BASE model without our LoRA and could be mistaken for the deployed model",
            "probability": "low (guarded in code)",
            "impact": "high (scientific integrity)",
            "mitigation": "inventory marks carries_project_lora=false; quantize route refuses to use them for Table rows",
            "owner": "deploy-runner",
        },
        {
            "id": "R5",
            "risk": "CPU fp32 is NOT faster than the GPU bf16 baseline (prior CPU ~1.78-1.95 s/img vs measured GPU ~1.33 s/img)",
            "probability": "high",
            "impact": "medium (deployment story becomes size/memory, not speed)",
            "mitigation": "report honestly as a negative result; 4-bit reduces size and enables CPU-only execution; do not fake a speed-up",
            "owner": "deploy-runner + paper-scribe",
        },
        {
            "id": "R6",
            "risk": "gradio is missing and installing it needs a network download/approval",
            "probability": "high (confirmed missing)",
            "impact": "medium (T15 cannot be verified)",
            "mitigation": "request approval; meanwhile deliver CLI predictor + annotated PNG fallback; never present an unlaunched UI as working",
            "owner": "Lead (approval) + deploy-runner",
        },
        {
            "id": "R7",
            "risk": "adapters/merged is outside the current prep write scope",
            "probability": "certain",
            "impact": "low (execution blocker only)",
            "mitigation": "Lead confirms T14 write scope before S1; prep writes nothing outside src/deploy, demo, results/deploy_*, adapters/quantized",
            "owner": "Lead",
        },
        {
            "id": "R8",
            "risk": "deployment candidate is provisional (T12/KD not run; seed44 evaluation in flight)",
            "probability": "certain",
            "impact": "medium",
            "mitigation": "label every artifact 'provisional sft_v1 seed42'; re-merge if the Lead freezes a different adapter",
            "owner": "Lead",
        },
        {
            "id": "R9",
            "risk": "quantized CPU benchmark competes for CPU with an in-flight GPU evaluation driver",
            "probability": "low",
            "impact": "low",
            "mitigation": "wait for the Lead's explicit 'evaluation finished' message before any model load or 50-image benchmark",
            "owner": "Lead + deploy-runner",
        },
        {
            "id": "R10",
            "risk": "RAM/disk pressure: fp32 merge ~1.0 GB + GGUF f16 ~0.5 GB + ONNX/quantized copies",
            "probability": "low (206 GB free measured)",
            "impact": "low",
            "mitigation": "record sizes/hashes; delete only clearly redundant intermediates after the report is written",
            "owner": "deploy-runner",
        },
    ]


def build_plan(environment: dict[str, Any] | None = None) -> dict[str, Any]:
    env = environment or collect_environment()
    routes = route_statuses(env)
    steps = build_steps(env)
    model = model_facts(with_hash=True)
    adapter = adapter_facts(with_hash=True)
    merged = merged_facts()
    quantized = quantized_facts()
    onnx_upstream = onnx_asset_inventory()
    datasets = dataset_facts()
    estimate = benchmark_time_estimate()
    missing_tools = summarize_missing_tools(env)
    deadlines_remaining = round(deadline_remaining_minutes(), 1)

    primary_minutes = sum(
        step["estimated_minutes"] for step in steps if step["id"] in {"S1", "S2", "S3", "S6", "S7", "S9", "S10"}
    )
    fallback_minutes = sum(
        step["estimated_minutes"] for step in steps if step["id"] in {"S1", "S2", "S4", "S5", "S6", "S9", "S10"}
    )
    claims = {
        "merged_model_created": "not_run",
        "quantized_model_created": "not_run",
        "cpu_latency_measured": "not_run",
        "gpu_latency_measured": "not_run",
        "demo_launched": "not_run",
        "demo_browser_verified": "not_run",
        "official_deployment_metrics": "not_run (must be produced by eval-closer)",
        "gpu_used_during_prep": False,
        "model_loaded_during_prep": False,
        "downloads_during_prep": False,
    }
    gaps: list[str] = []
    if not missing_tools["llama_cpp"]["available"]:
        gaps.append("llama.cpp toolchain absent -> GGUF route needs approval to download a prebuilt release or is blocked")
    if not missing_tools["gradio"]["available"]:
        gaps.append("gradio missing -> T15 demo cannot be launched until `pip install gradio` is approved")
    if not missing_tools["optimum"]["available"] or not missing_tools["onnx"]["available"]:
        gaps.append("optimum/onnx missing -> ONNX fallback needs installation before it can be attempted")
    gaps.append("adapters/merged write scope not yet confirmed by the Lead")
    gaps.append("T12/KD not run -> deployment candidate is provisional sft_v1 seed42 adapter")
    gaps.append("seed44 independent evaluation in flight -> final model choice may change")
    gaps.append("no merged/quantized artifact exists yet; every deployment claim is not_run")

    return {
        "schema_version": 1,
        "document": "results/deploy_prep.json",
        "phase": "prep",
        "status": "prep_complete",
        "purpose": (
            "T14/T15 CPU-only preparation: verified environment, executable code skeletons, "
            "planned commands, realistic time estimates, risks and fallbacks. Executed only "
            "after the Lead releases the evaluation GPU and issues the CPU/GPU lease."
        ),
        "generated_at": env["captured_at"],
        "deadline": {
            "local": DEADLINE_LOCAL_ISO,
            "remaining_minutes_at_prep": deadlines_remaining,
            "hard_boundary_note": "no heavy step may run before the Lead announces the evaluation lease ended",
        },
        "execution_gate": {
            "current_gpu_state": env["gpu"],
            "rule": "prep must not load a model, start CUDA work, download large files or run heavy CPU tests",
            "release_condition": "Lead message that the seed44/3-seed evaluation finished and the GPU lease is released",
        },
        "environment": env,
        "tool_availability": {
            "packages": env["packages"],
            "llama_cpp": env["tools"]["llama_cpp"],
            "missing_and_blocked": missing_tools,
            "onnxruntime_providers_note": "onnxruntime 1.31 CPU available; GPU providers not required for T14",
        },
        "assets": {
            "model": model,
            "adapter": adapter,
            "merged": merged,
            "quantized": quantized,
            "upstream_onnx": onnx_upstream,
            "datasets": datasets,
            "write_scopes": {
                "allowed_in_prep": ["src/deploy", "demo", "results/deploy_*", "adapters/quantized"],
                "needed_for_execution": ["adapters/merged", "results/deploy_*", "adapters/quantized", "demo", "src/deploy"],
                "forbidden": ["data", "src/train", "src/eval", "results/eval_*", "results/train_*", "report"],
            },
        },
        "deployment_candidate": {
            "path": PROVISIONAL_DEPLOYMENT_CANDIDATE,
            "status": "provisional",
            "reason": "T12/KD not run; Lead has not frozen the final adapter",
            "inference_contract": {
                "prompt_id": "p1 (src.eval.prompts)",
                "image_size": IMAGE_SIZE,
                "do_image_splitting": False,
                "image_seq_len": EXPECTED_IMAGE_SEQ_LEN,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "num_beams": 1,
                "seed": 42,
                "parser": "src.eval.grounding.parse_box_with_error (frozen)",
                "metric_contract": "src.eval.grounding.metrics_from_scores (invalid in n)",
            },
        },
        "routes": routes,
        "fallback_chain": list(FALLBACK_CHAIN),
        "fallback_chain_note": (
            "try in order; a route that is missing or fails is recorded as not_run/blocked with the "
            "real error. ONNX is marked conditional because Idefics3 export support is unverified."
        ),
        "steps": steps,
        "estimated_total_minutes": {
            "primary_gguf_path": primary_minutes,
            "fallback_onnx_path": fallback_minutes,
            "benchmark_50_images_one_engine": round(estimate["total_seconds_upper_bound"] / 60.0, 1),
            "official_eval_per_1000_image_set": 40,
            "note": "estimates exclude waiting for Lead lease/download approval",
        },
        "time_estimates_detail": prep_time_estimates(env),
        "risks": build_risks(env, routes),
        "claims": claims,
        "gaps": gaps,
        "cpu_test_plan": {
            "runner": "python -m src.deploy.run_cpu_tests",
            "output": "results/deploy_cpu_tests.json",
            "modules": ["src.deploy.test_deploy_cpu", "demo.test_demo_cpu"],
            "constraints": [
                "CUDA_VISIBLE_DEVICES='' so no CUDA context can be created",
                "no model weights loaded, no torch/transformers import",
                "no network, no download, no subprocess that trains",
            ],
            "covered": [
                "plan schema, fallback order, positive estimates",
                "environment detection uses importlib.metadata only (assert torch not imported)",
                "adapter identity/hash checks against the real adapter",
                "merge input validation + merged-directory verification on temp dirs",
                "GGUF/ONNX/torch-int8 plan construction and blocked-route reasons",
                "upstream ONNX inventory cannot carry LoRA",
                "deterministic benchmark subset + latency statistics",
                "box drawing and demo handler with a fake predictor",
                "deploy_prep.json schema and honest not_run claims",
            ],
            "not_covered_here": [
                "actual model load / merge / generate (needs the post-evaluation lease)",
                "gradio UI launch (gradio not installed)",
                "quantization conversion (llama.cpp/optimum missing)",
            ],
        },
        "next_actions_for_lead": [
            "release notification: tell deploy-runner when the in-flight evaluation ends",
            "confirm write scope for adapters/merged (T14 execution)",
            "approve `python -m pip install gradio` (~download) for T15",
            "approve acquisition of a prebuilt llama.cpp Windows CPU release, or accept GGUF as blocked",
            "decide whether the provisional sft_v1 seed42 adapter is the model to deploy or wait for KD/seed freeze",
        ],
    }


def build_prep_document(environment: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_plan(environment)
