"""Shared HuggingFace CPU/GPU predictor used by the benchmark and the demo.

Importing this module is free of heavy dependencies: ``torch``, ``transformers``
and ``PIL`` are imported inside :meth:`HFPredictor.load` / :meth:`HFPredictor.predict`.
The prompt (``p1``) and the box parser are imported from :mod:`src.eval` so the
deployment path cannot invent a second prompt or a second metric contract.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Mapping

from .config import (
    ADAPTER_REQUIRED_FILES,
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    EXPECTED_IMAGE_SEQ_LEN,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    MERGED_REQUIRED_FILES,
    MODEL_PATH,
    file_facts,
    project_relative,
    expected_image_seq_len,
    read_json,
)

DTYPE_NAMES = ("float32", "bfloat16", "float16")


def classify_artifact(path: str | Path, *, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Classify a deployment artifact from its files, without loading any weights."""

    resolved = Path(path)
    is_adapter = (resolved / "adapter_config.json").is_file()
    is_model = (resolved / "config.json").is_file() and (resolved / "model.safetensors").is_file()
    if is_adapter:
        kind = "adapter"
        required = ADAPTER_REQUIRED_FILES
    elif is_model:
        required = MERGED_REQUIRED_FILES
        kind = "base" if resolved.resolve() == Path(model_path).resolve() else "merged"
    else:
        return {
            "path": project_relative(resolved),
            "exists": resolved.is_dir(),
            "kind": "unknown",
            "usable": False,
            "reason": "neither an adapter (adapter_config.json) nor a full model checkpoint (config.json + model.safetensors)",
        }
    missing = [name for name in required if not (resolved / name).is_file()]
    payload: dict[str, Any] = {
        "path": project_relative(resolved),
        "exists": True,
        "kind": kind,
        "usable": not missing,
        "required_files": list(required),
        "missing_required_files": missing,
        "files": {
            name: file_facts(resolved / name)
            for name in ("adapter_config.json", "adapter_model.safetensors", "model.safetensors", "config.json")
            if (resolved / name).is_file()
        },
    }
    if kind == "adapter":
        config = read_json(resolved / "adapter_config.json")
        payload.update(
            {
                "peft_type": config.get("peft_type"),
                "r": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "requires_base_model": True,
            }
        )
    else:
        config = read_json(resolved / "config.json")
        payload.update(
            {
                "architecture": (config.get("architectures") or [None])[0],
                "has_lora_in_config": "peft" in str(config).lower() or "lora" in str(config).lower(),
                "requires_base_model": False,
            }
        )
    return payload


class HFPredictor:
    """Greedy grounding predictor for the merged checkpoint or base+adapter.

    ``merged`` and ``adapter`` are mutually exclusive.  The processor's
    ``image_seq_len`` is derived from ``image_size`` for the current call only
    (the shipped ``processor_config.json`` default of 64 is never modified).
    """

    def __init__(
        self,
        *,
        model_path: str | Path = MODEL_PATH,
        merged: str | Path | None = None,
        adapter: str | Path | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        image_size: int = IMAGE_SIZE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        local_files_only: bool = True,
    ) -> None:
        if merged is not None and adapter is not None:
            raise ValueError("pass either merged or adapter, not both")
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"device must be cpu or cuda, got {device!r}")
        if dtype not in DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {DTYPE_NAMES}, got {dtype!r}")
        if image_size <= 0 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model_path = Path(model_path)
        self.merged = Path(merged) if merged is not None else None
        self.adapter = Path(adapter) if adapter is not None else None
        self.device = device
        self.dtype_name = dtype
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.image_seq_len = expected_image_seq_len(image_size)
        self.model: Any = None
        self.processor: Any = None
        self.load_report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> dict[str, Any]:
        if self.model is not None:
            return self.load_report or {}
        if self.merged is not None:
            artifact = classify_artifact(self.merged, model_path=self.model_path)
            if artifact["kind"] != "merged":
                raise ValueError(f"--merged must point to a merged checkpoint, got kind={artifact['kind']!r}: {self.merged}")
            if not artifact["usable"]:
                raise FileNotFoundError(f"merged checkpoint is incomplete: missing {artifact['missing_required_files']}")
            if artifact.get("has_lora_in_config"):
                raise RuntimeError("merged checkpoint config still mentions peft/lora; refusing to deploy it")
            weights = artifact["files"].get("model.safetensors")
            if not weights or not weights.get("bytes"):
                raise FileNotFoundError("merged checkpoint has no model.safetensors")
        elif self.adapter is not None:
            artifact = classify_artifact(self.adapter, model_path=self.model_path)
            if artifact["kind"] != "adapter":
                raise ValueError(f"--adapter must point to a PEFT adapter dir, got kind={artifact['kind']!r}: {self.adapter}")
            if not artifact["usable"]:
                raise FileNotFoundError(f"adapter is incomplete: missing {artifact['missing_required_files']}")

        started = time.perf_counter()
        import torch  # local import: keep the package import CPU-light
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.manual_seed(42)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype_name]

        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
        # The project's no-split path processes one square 384 tile = 36 tokens.
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": self.image_size}
        processor.image_processor.max_image_size = {"longest_edge": self.image_size}
        processor.image_seq_len = self.image_seq_len

        source = self.merged if self.merged is not None else self.model_path
        model = AutoModelForImageTextToText.from_pretrained(
            source, dtype=dtype, local_files_only=self.local_files_only
        )
        base_class = type(model).__name__
        adapter_load: dict[str, Any] | None = None
        if self.adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, self.adapter, is_trainable=False, torch_device=self.device
            )
            lora_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
            if not lora_modules:
                raise RuntimeError("adapter attached no LoRA modules")
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            if trainable:
                raise RuntimeError(f"adapter load left {len(trainable)} trainable parameters")
            adapter_load = {
                "loaded": True,
                "is_trainable": False,
                "lora_module_count": len(lora_modules),
                "trainable_parameter_count": 0,
                "base_model_class": base_class,
            }
        model.eval()
        model.to(self.device)
        self.processor = processor
        self.model = model
        self.load_report = {
            "status": "loaded",
            "device": self.device,
            "dtype": self.dtype_name,
            "image_size": self.image_size,
            "image_seq_len": self.image_seq_len,
            "max_new_tokens": self.max_new_tokens,
            "source": project_relative(source),
            "merged": project_relative(self.merged) if self.merged is not None else None,
            "adapter": project_relative(self.adapter) if self.adapter is not None else None,
            "adapter_load": adapter_load,
            "model_class": type(model).__name__,
            "base_model_class": base_class,
            "torch_threads": int(torch.get_num_threads()),
            "load_seconds": time.perf_counter() - started,
            "local_files_only": self.local_files_only,
        }
        return self.load_report

    # --------------------------------------------------------------- predict
    def predict(self, image: Any, question: str) -> dict[str, Any]:
        """Run one deterministic greedy generation and parse the box with the frozen parser."""

        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before predict()")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        import time as _time

        import torch

        from src.eval.grounding import parse_box_with_error
        from src.eval.prompts import build_messages

        prompt = self.processor.apply_chat_template(build_messages(question), add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        input_tokens = int(inputs["input_ids"].shape[1])
        started = _time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        inference_seconds = _time.perf_counter() - started
        generated_ids = generated[0, input_tokens:]
        raw_output = self.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=False)[0]
        output_text = self.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
        predicted_box, parse_error = parse_box_with_error(output_text)
        generated_tokens = int(generated_ids.shape[0])
        return {
            "raw_output": raw_output,
            "output_text": output_text,
            "predicted_box": predicted_box,
            "parse_error": parse_error,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "hit_max_new_tokens": generated_tokens == self.max_new_tokens,
            "inference_seconds": inference_seconds,
        }

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self.model is None:
            return
        import torch

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "HFPredictor":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def describe_artifact_spec(
    *,
    merged: str | Path | None = None,
    adapter: str | Path | None = None,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Pure-CPU description used by ``--plan-only`` paths and the prep document."""

    if merged is not None and adapter is not None:
        raise ValueError("pass either merged or adapter, not both")
    if merged is not None:
        return {"role": "merged", **classify_artifact(merged, model_path=model_path)}
    if adapter is not None:
        return {
            "role": "adapter",
            **classify_artifact(adapter, model_path=model_path),
            "base_model": classify_artifact(model_path, model_path=model_path),
        }
    return {"role": "base", **classify_artifact(model_path, model_path=model_path)}


def peak_process_memory_mib() -> float | None:
    """Peak working set of this process (Windows) or max RSS (POSIX), read-only."""

    import sys

    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            process = kernel32.GetCurrentProcess()
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except Exception:  # pragma: no cover - platform defensive
            return None
        return None
    try:  # pragma: no cover - POSIX only
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0
    except Exception:
        return None
