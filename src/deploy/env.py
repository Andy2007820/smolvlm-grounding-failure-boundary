"""Read-only environment / tool / asset detection for the deployment prep.

This module deliberately avoids importing ``torch``, ``transformers``,
``onnxruntime`` or ``gradio``.  Package availability is detected with
``importlib.util.find_spec`` and versions with ``importlib.metadata`` only, so
running the prep does **not** create a CUDA context while the GPU is leased to
the evaluator.  It also performs no downloads.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    ADAPTER_REQUIRED_FILES,
    DEADLINE_LOCAL_ISO,
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    DEFAULT_QUANTIZED_GGUF,
    EXPECTED_IMAGE_SEQ_LEN,
    IMAGE_SIZE,
    MERGED_REQUIRED_FILES,
    MODEL_PATH,
    MODEL_REQUIRED_FILES,
    PROJECT_ROOT,
    benchmark_time_estimate,
    deadline_remaining_minutes,
    file_facts,
    now_iso,
    project_relative,
    read_json,
    sha256_file,
)

#: (import name, distribution name, role, expected version in this project)
PACKAGE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("torch", "torch", "model execution / merge / int8 dynamic quantization", "2.11.0+cu128"),
    ("transformers", "transformers", "AutoModelForImageTextToText / AutoProcessor", "5.19.0"),
    ("peft", "peft", "LoRA adapter merge", "0.21.2"),
    ("onnxruntime", "onnxruntime", "ONNX CPU inference + dynamic quantization", "not yet used"),
    ("onnx", "onnx", "graph loading for ONNX quantization", "not yet used"),
    ("optimum", "optimum", "ONNX export fallback (Idefics3 support unverified)", "not yet used"),
    ("gguf", "gguf", "llama.cpp GGUF writer dependency", "not yet used"),
    ("gradio", "gradio", "T15 demo UI", "not yet used"),
    ("bitsandbytes", "bitsandbytes", "optional GPU 4-bit reference only", "not yet used"),
    ("accelerate", "accelerate", "optional device placement helper", "not yet used"),
    ("safetensors", "safetensors", "weight I/O", "0.8.0"),
    ("PIL", "pillow", "image loading / box drawing", "12.3.0"),
    ("sentencepiece", "sentencepiece", "possible llama.cpp tokenizer conversion dep", "not yet used"),
)


def package_status(import_name: str, distribution_name: str, role: str, expected: str) -> dict[str, Any]:
    """Availability + version without importing the package."""

    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError):  # broken/missing parent package
        spec = None
    available = spec is not None
    version: str | None = None
    status = "ok"
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        if available:
            status = "importable_without_distribution_metadata"
        else:
            status = "missing"
    return {
        "module": import_name,
        "distribution": distribution_name,
        "role": role,
        "available": available,
        "version": version,
        "expected_in_project": expected,
        "status": status,
    }


def packages() -> dict[str, dict[str, Any]]:
    return {
        import_name: package_status(import_name, dist, role, expected)
        for import_name, dist, role, expected in PACKAGE_SPECS
    }


def executable_status(name: str, *, version_args: Sequence[str] = ("--version",), timeout: int = 10) -> dict[str, Any]:
    path = shutil.which(name)
    payload: dict[str, Any] = {"name": name, "found": bool(path), "path": path, "version": None, "probe": None}
    if not path:
        return payload
    for args in (tuple(version_args), ("--help",)):
        try:
            completed = subprocess.run(
                [path, *args], capture_output=True, text=True, timeout=timeout, errors="replace"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            payload["probe"] = f"{type(exc).__name__}: {exc}"
            continue
        output = (completed.stdout or "") + (completed.stderr or "")
        payload["probe"] = output.strip().splitlines()[0][:200] if output.strip() else f"exit={completed.returncode}"
        if completed.returncode == 0 and output.strip():
            payload["version"] = payload["probe"]
            break
    return payload


def detect_llama_cpp() -> dict[str, Any]:
    """Locate a llama.cpp checkout/binaries without downloading or building."""

    home_candidates: list[Path] = []
    env_home = os.environ.get("LLAMA_CPP_DIR")
    if env_home:
        home_candidates.append(Path(env_home))
    workspace_root = PROJECT_ROOT.parent
    home_candidates.extend(
        [
            PROJECT_ROOT / "third_party" / "llama.cpp",
            PROJECT_ROOT / "tools" / "llama.cpp",
            workspace_root / "llama.cpp",
            Path("C:/llama.cpp"),
        ]
    )

    binaries: dict[str, Any] = {}
    for name in ("llama-quantize", "llama-mtmd-cli", "llama-cli", "convert_hf_to_gguf.py"):
        binaries[name] = {"path": None, "source": None}

    convert_script: Path | None = None
    home: Path | None = None
    for candidate in home_candidates:
        if not candidate.is_dir():
            continue
        script = candidate / "convert_hf_to_gguf.py"
        local_bin_dir = candidate / "build" / "bin"
        found_any = False
        for name in ("llama-quantize", "llama-mtmd-cli", "llama-cli"):
            for suffix in (".exe", ""):
                for directory in (local_bin_dir, candidate, candidate / "build" / "bin" / "Release"):
                    probe = directory / f"{name}{suffix}"
                    if probe.is_file():
                        binaries[name] = {"path": str(probe), "source": project_relative(probe)}
                        found_any = True
                        break
                if binaries[name]["path"]:
                    break
        if script.is_file():
            convert_script = script
            found_any = True
        if found_any and home is None:
            home = candidate

    # Fall back to PATH-installed binaries (e.g. a prebuilt release unpacked to PATH).
    for name in ("llama-quantize", "llama-mtmd-cli", "llama-cli"):
        if binaries[name]["path"]:
            continue
        path = shutil.which(name)
        if path:
            binaries[name] = {"path": path, "source": "PATH"}

    quantize_ok = binaries["llama-quantize"]["path"] is not None
    mmproj_ok = binaries["llama-mtmd-cli"]["path"] is not None
    convert_ok = convert_script is not None
    missing: list[str] = []
    if not convert_ok:
        missing.append("convert_hf_to_gguf.py")
    if not quantize_ok:
        missing.append("llama-quantize")
    if not mmproj_ok:
        missing.append("llama-mtmd-cli")
    cmake = shutil.which("cmake")
    compilers = {name: shutil.which(name) for name in ("cl", "gcc", "g++", "clang", "clang++")}
    return {
        "available": bool(convert_ok and quantize_ok),
        "home": project_relative(home) if home is not None else None,
        "convert_script": str(convert_script) if convert_script else None,
        "binaries": binaries,
        "missing": missing,
        "build_tools": {
            "cmake": cmake,
            "compilers": compilers,
            "can_build_from_source": bool(cmake and any(compilers.values())),
        },
        "acquisition_note": (
            "No llama.cpp toolchain is installed. GGUF needs either (a) a prebuilt "
            "Windows CPU release unpacked under third_party/llama.cpp or PATH, or "
            "(b) a source checkout plus cmake/MSVC. Both require a download/build "
            "that is out of scope for this CPU-only prep run."
        ),
        "policy_note": "never downloads or builds anything during prep",
    }


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Read-only GPU state snapshot; does not allocate GPU memory."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    created = now_iso()
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "command": " ".join(command), "error": "nvidia-smi not on PATH", "captured_at": created}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "command": " ".join(command), "error": f"{type(exc).__name__}: {exc}", "captured_at": created}
    if completed.returncode != 0:
        return {
            "available": False,
            "command": " ".join(command),
            "error": (completed.stderr or completed.stdout or "nvidia-smi failed").strip()[:500],
            "captured_at": created,
        }
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"available": False, "command": " ".join(command), "error": "empty nvidia-smi output", "captured_at": created}
    fields = [field.strip() for field in lines[0].split(",")]
    payload: dict[str, Any] = {
        "available": True,
        "command": " ".join(command),
        "captured_at": created,
        "raw": lines[0],
        "gpu_compute_started_by_this_module": False,
    }
    if len(fields) >= 4:
        payload.update(
            {
                "name": fields[0],
                "memory_used_mib": _to_number(fields[1]),
                "memory_total_mib": _to_number(fields[2]),
                "utilization_gpu_percent": _to_number(fields[3]),
            }
        )
    busy = payload.get("utilization_gpu_percent")
    used = payload.get("memory_used_mib")
    payload["state"] = (
        "busy"
        if isinstance(busy, (int, float)) and busy > 5
        else "idle_or_small_context"
        if isinstance(used, (int, float)) and used < 2000
        else "unknown"
    )
    payload["interpretation"] = (
        "read-only snapshot captured during prep; a busy state means a leased evaluation "
        "may be running and this prep process must not touch the GPU"
    )
    return payload


def _to_number(value: str) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def disk_facts(path: Path = PROJECT_ROOT) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"path": str(path), "available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "available": True,
        "total_gb": round(usage.total / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
    }


def host_facts() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "project_root": str(PROJECT_ROOT),
        "working_directory": os.getcwd(),
        "captured_at": now_iso(),
    }


def model_facts(model_path: Path = MODEL_PATH, *, with_hash: bool = True) -> dict[str, Any]:
    config_path = model_path / "config.json"
    preprocessor_path = model_path / "preprocessor_config.json"
    processor_path = model_path / "processor_config.json"
    generation_path = model_path / "generation_config.json"
    payload: dict[str, Any] = {
        "path": project_relative(model_path),
        "exists": model_path.is_dir(),
        "required_files": {name: (model_path / name).is_file() for name in MODEL_REQUIRED_FILES},
        "missing_required_files": [name for name in MODEL_REQUIRED_FILES if not (model_path / name).is_file()],
    }
    if config_path.is_file():
        config = read_json(config_path)
        text_config = config.get("text_config") or {}
        vision_config = config.get("vision_config") or {}
        generation = read_json(generation_path) if generation_path.is_file() else {}
        processor = read_json(processor_path) if processor_path.is_file() else {}
        payload.update(
            {
                "architecture": (config.get("architectures") or [None])[0],
                "model_type": config.get("model_type"),
                "image_token_id": config.get("image_token_id"),
                "vocab_size": config.get("vocab_size"),
                "torch_dtype": config.get("torch_dtype"),
                "vision_patch_size": vision_config.get("patch_size"),
                "vision_scale_factor": config.get("scale_factor"),
                "vision_hidden_size": vision_config.get("hidden_size"),
                "vision_layers": vision_config.get("num_hidden_layers"),
                "text_hidden_size": text_config.get("hidden_size"),
                "text_layers": text_config.get("num_hidden_layers"),
                "text_lora_targets_scope": "model.text_model",
                "generation_eos_token_id": generation.get("eos_token_id"),
                "generation_pad_token_id": generation.get("pad_token_id"),
                "processor_default_image_seq_len": processor.get("image_seq_len"),
                "expected_runtime_image_seq_len": EXPECTED_IMAGE_SEQ_LEN,
                "processor_default_matches_384_no_split": processor.get("image_seq_len") == EXPECTED_IMAGE_SEQ_LEN,
            }
        )
    if preprocessor_path.is_file():
        preprocessor = read_json(preprocessor_path)
        payload["preprocessor_do_image_splitting"] = preprocessor.get("do_image_splitting")
        payload["preprocessor_size"] = preprocessor.get("size")
        payload["preprocessor_max_image_size"] = preprocessor.get("max_image_size")
    weights = model_path / "model.safetensors"
    payload["weights"] = file_facts(weights, with_hash=with_hash)
    if payload["weights"]["bytes"]:
        # bf16 safetensors -> approximate parameter count for merge disk estimates.
        payload["approx_parameters_millions"] = round(payload["weights"]["bytes"] / 2 / 1e6, 2)
    payload["all_required_files_present"] = not payload["missing_required_files"]
    return payload


def onnx_asset_inventory(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Inventory the ONNX files that shipped inside the upstream model repo.

    These are the transformers.js export of the **base** SmolVLM-256M-Instruct.
    They do **not** contain the project LoRA adapter, so they may only be used
    for ONNX pipeline bring-up and never for the ``+量化`` Table 1 row.
    """

    onnx_dir = model_path / "onnx"
    payload: dict[str, Any] = {
        "dir": project_relative(onnx_dir),
        "exists": onnx_dir.is_dir(),
        "files": [],
        "total_bytes": 0,
        "carries_project_lora": False,
        "scientific_note": (
            "upstream transformers.js export of the BASE SmolVLM-256M-Instruct; "
            "the project LoRA adapter is not baked in, so these files cannot "
            "produce the +量化 accuracy row. They are at most a runtime smoke asset."
        ),
    }
    if not onnx_dir.is_dir():
        return payload
    files = sorted(onnx_dir.glob("*.onnx"))
    payload["files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "carries_project_lora": False}
        for path in files
    ]
    payload["total_bytes"] = sum(item["bytes"] for item in payload["files"])
    payload["total_gb"] = round(payload["total_bytes"] / (1024**3), 3)
    variants = set()
    for path in files:
        stem = path.stem
        for variant in ("_bnb4", "_fp16", "_int8", "_uint8", "_quantized", "_q4", "_q4f16"):
            if stem.endswith(variant):
                variants.add(variant.lstrip("_"))
                stem = stem[: -len(variant)]
        variants.add("fp32")
    payload["variants_seen"] = sorted(variants)
    payload["has_split_pipeline"] = all(
        (onnx_dir / name).is_file()
        for name in ("vision_encoder.onnx", "embed_tokens.onnx", "decoder_model_merged.onnx")
    )
    return payload


def adapter_facts(adapter_path: Path = DEFAULT_ADAPTER, *, with_hash: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": project_relative(adapter_path),
        "exists": adapter_path.is_dir(),
        "required_files": {name: (adapter_path / name).is_file() for name in ADAPTER_REQUIRED_FILES},
        "missing_required_files": [name for name in ADAPTER_REQUIRED_FILES if not (adapter_path / name).is_file()],
    }
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        payload["all_required_files_present"] = False
        return payload
    config = read_json(config_path)
    base_name = config.get("base_model_name_or_path")
    base_path = Path(str(base_name)) if base_name else None
    target_modules = sorted(config.get("target_modules") or [])
    payload.update(
        {
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "bias": config.get("bias"),
            "use_dora": config.get("use_dora"),
            "base_model_name_or_path": str(base_name) if base_name else None,
            "base_model_matches_project_model": bool(base_path and base_path.resolve() == MODEL_PATH.resolve()),
            "target_module_count": len(target_modules),
            "target_module_sample": target_modules[:6],
            "target_modules_include_q_proj": any(name.endswith("q_proj") for name in target_modules),
            "target_modules_include_o_proj": any(name.endswith("o_proj") for name in target_modules),
            "adapter_config": file_facts(config_path, with_hash=with_hash),
            "adapter_weights": file_facts(adapter_path / "adapter_model.safetensors", with_hash=with_hash),
            "peft_version_recorded": config.get("peft_version"),
        }
    )
    payload["all_required_files_present"] = not payload["missing_required_files"]
    return payload


def merged_facts(merged_path: Path = DEFAULT_MERGED) -> dict[str, Any]:
    return {
        "path": project_relative(merged_path),
        "exists": merged_path.is_dir(),
        "files": {
            name: file_facts(merged_path / name)
            for name in MERGED_REQUIRED_FILES
            if (merged_path / name).is_file()
        },
        "missing_required_files": [
            name for name in MERGED_REQUIRED_FILES if not (merged_path / name).is_file()
        ],
    }


def quantized_facts(quantized_root: Path | None = None) -> dict[str, Any]:
    root = quantized_root or DEFAULT_QUANTIZED_GGUF.parent
    payload: dict[str, Any] = {
        "root": project_relative(root),
        "exists": root.is_dir(),
        "artifacts": [],
        "status": "not_run",
    }
    if not root.is_dir():
        return payload
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.upper() != "README.MD":
            payload["artifacts"].append(
                {"path": project_relative(path), "bytes": path.stat().st_size, "suffix": path.suffix.lower()}
            )
    payload["status"] = "has_artifacts" if payload["artifacts"] else "empty"
    return payload


def dataset_facts() -> dict[str, Any]:
    candidates = {
        "synthetic_eval": PROJECT_ROOT / "data" / "synthetic_eval.jsonl",
        "refcoco_val": PROJECT_ROOT / "data" / "refcoco_val.jsonl",
        "screenspot": PROJECT_ROOT / "data" / "screenspot.jsonl",
    }
    payload: dict[str, Any] = {}
    for name, path in candidates.items():
        entry: dict[str, Any] = {"path": project_relative(path), "exists": path.is_file()}
        if path.is_file():
            entry["bytes"] = path.stat().st_size
            try:
                with path.open("r", encoding="utf-8") as handle:
                    entry["lines"] = sum(1 for line in handle if line.strip())
            except OSError as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        payload[name] = entry
    return payload


def collect_environment() -> dict[str, Any]:
    """Everything the prep run is allowed to observe, with no heavy imports."""

    started = time.perf_counter()
    environment = {
        "captured_at": now_iso(),
        "host": host_facts(),
        "packages": packages(),
        "tools": {
            "llama_cpp": detect_llama_cpp(),
            "git": executable_status("git"),
            "cmake": executable_status("cmake", version_args=("--version",)),
            "nvidia_smi": executable_status("nvidia-smi", version_args=("--help",)),
        },
        "gpu": nvidia_smi_snapshot(),
        "disk": disk_facts(PROJECT_ROOT),
        "deadline": {
            "local": DEADLINE_LOCAL_ISO,
            "remaining_minutes_at_capture": round(deadline_remaining_minutes(), 1),
        },
        "prep_policy": {
            "gpu_compute_used": False,
            "model_loaded": False,
            "downloads_performed": False,
            "torch_imported_by_detection": "torch" in sys.modules,
        },
        "probe_seconds": None,
    }
    environment["probe_seconds"] = round(time.perf_counter() - started, 3)
    return environment


def summarize_missing_tools(environment: dict[str, Any]) -> dict[str, Any]:
    """Which optional deployment toolchains are missing and what that blocks."""

    packages_payload = environment["packages"]

    def availability(module: str) -> bool:
        return bool(packages_payload.get(module, {}).get("available"))

    llama = environment["tools"]["llama_cpp"]
    return {
        "gradio": {
            "available": availability("gradio"),
            "blocks": "T15 demo launch",
            "install_hint": "python -m pip install gradio",
        },
        "optimum": {
            "available": availability("optimum"),
            "blocks": "ONNX export fallback",
            "install_hint": "python -m pip install optimum[onnxruntime] onnx",
        },
        "onnx": {
            "available": availability("onnx"),
            "blocks": "ONNX graph quantization",
            "install_hint": "python -m pip install onnx",
        },
        "gguf": {
            "available": availability("gguf"),
            "blocks": "llama.cpp convert_hf_to_gguf.py import",
            "install_hint": "python -m pip install gguf",
        },
        "llama_cpp": {
            "available": bool(llama["available"]),
            "blocks": "GGUF Q4_K_M export + llama.cpp CPU inference",
            "missing": llama["missing"],
            "install_hint": "unpack a prebuilt Windows llama.cpp release (with llama-quantize/llama-mtmd-cli) or build from source",
        },
        "onnxruntime": {
            "available": availability("onnxruntime"),
            "blocks": "ONNX CPU inference",
            "install_hint": "already installed",
        },
        "torch_int8": {
            "available": availability("torch"),
            "blocks": "n/a (built into torch)",
            "install_hint": "already installed; fallback quantizer needs no export",
        },
    }


def prep_time_estimates(environment: dict[str, Any]) -> dict[str, Any]:
    bench = benchmark_time_estimate()
    return {
        "benchmark_50_images_one_engine": {
            "basis": "prior CPU fp32+adapter diagnostic (upper bound, includes load)",
            "per_sample_seconds_prior": bench["per_sample_seconds_prior"],
            "estimated_minutes": round(bench["total_seconds_upper_bound"] / 60.0, 1),
            "note": "warmup 5 + 50 measured samples, CPU fp32; a quantized/GGUF engine should be faster",
        },
        "merge_cpu_fp32": {
            "estimated_minutes": 6,
            "basis": "load 513 MB bf16 base on CPU, merge 7.4 MB adapter, save ~1.0 GB fp32 safetensors",
        },
        "merge_equivalence_check_5_samples": {"estimated_minutes": 8, "basis": "two CPU model loads + 5 greedy generations each"},
        "gguf_q4_k_m": {
            "estimated_minutes": 35,
            "basis": "tool acquisition 15-20 min (if approved), convert+quantize 5-10 min, smoke 5 min",
            "conditional": "requires llama.cpp binaries/convert script + gguf/sentencepiece packages",
        },
        "onnx_dynamic_quant": {
            "estimated_minutes": 25,
            "basis": "optimum export (may fail for Idefics3) + quantize_dynamic + session smoke",
            "conditional": "requires optimum + onnx; export support unverified",
        },
        "torch_int8_dynamic": {
            "estimated_minutes": 8,
            "basis": "in-process torch.ao.quantization.quantize_dynamic on the text decoder, no export",
        },
        "gradio_demo_install_and_verify": {
            "estimated_minutes": 20,
            "basis": "pip install gradio (needs approval), launch, HTTP/API round-trip with a real image",
            "conditional": "requires gradio install",
        },
        "official_deployment_eval_by_eval_closer": {
            "estimated_minutes_per_1000_image_set": 40,
            "basis": "CPU fp32 upper bound ~1.95 s/image measured on 20 samples; 1000 images ~32.5 min + load",
            "note": "official Table 3/Table 1 +量化 numbers are produced by eval-closer, not by this package",
        },
    }
