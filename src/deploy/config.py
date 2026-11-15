"""Shared, dependency-free constants and path helpers for the deployment work.

Every helper in this module is pure CPU and imports only the standard library.
Paths are resolved against ``PROJECT_ROOT`` (``final-project/``), never against
the current working directory, so ``python -m src.deploy.<tool>`` behaves the
same from anywhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
SCRATCH_ROOT = PROJECT_ROOT / ".cache" / "deploy_scratch"

MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
DEFAULT_ADAPTER = PROJECT_ROOT / "adapters" / "sft_v1" / "adapter"
MERGED_ROOT = PROJECT_ROOT / "adapters" / "merged"
DEFAULT_MERGED = MERGED_ROOT / "sft_v1_merged"
QUANTIZED_ROOT = PROJECT_ROOT / "adapters" / "quantized"
DEFAULT_QUANTIZED_GGUF = QUANTIZED_ROOT / "sft_v1_q4_k_m"
DEFAULT_ONNX_OUT = QUANTIZED_ROOT / "sft_v1_onnx"

#: Frozen inference geometry shared with training/evaluation (T5/T6 contract).
IMAGE_SIZE = 384
PATCH_SIZE = 16
SCALE_FACTOR = 4
EXPECTED_IMAGE_SEQ_LEN = (IMAGE_SIZE // (PATCH_SIZE * SCALE_FACTOR)) ** 2  # 36
MAX_NEW_TOKENS = 64
BENCH_WARMUP = 5
BENCH_SAMPLES = 50
BENCH_SEED = 42
ACC_THRESHOLD = 0.5

#: ``p1`` is owned by :mod:`src.eval.prompts`; this is only an identity check.
PROMPT_ID = "p1"

DEADLINE_LOCAL_ISO = "2026-11-10T13:00:00+08:00"
DEADLINE_UTC_ISO = "2026-11-10T05:00:00+00:00"

#: The deployment candidate is provisional until the Lead freezes the final
#: adapter (T12/KD is still ``not_run``).  Never quote it as "the final model".
PROVISIONAL_DEPLOYMENT_CANDIDATE = "adapters/sft_v1/adapter"

MODEL_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

MERGED_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "special_tokens_map.json",
)

ADAPTER_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")

#: Files copied from the base model directory into the merged checkpoint when
#: ``processor.save_pretrained`` does not emit them (transformers-version drift).
COPY_FROM_BASE_FILES = (
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)

#: Passed to every ``subprocess`` call that runs a Python entry point.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONIOENCODING": "utf-8",
}


def now_iso() -> str:
    """Timezone-aware local timestamp (UTC+08:00 on this machine)."""

    return dt.datetime.now().astimezone().isoformat()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: str | Path) -> str:
    """POSIX-style path relative to the project root (or absolute if outside)."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_facts(path: str | Path, *, with_hash: bool = False) -> dict[str, Any]:
    resolved = Path(path)
    facts: dict[str, Any] = {
        "path": project_relative(resolved),
        "exists": resolved.is_file(),
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
    }
    if facts["exists"] and with_hash:
        facts["sha256"] = sha256_file(resolved)
    return facts


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON (no BOM) through a temporary file + replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)
    return target


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    return target


def checked_results_path(value: str | Path, prefix: str) -> Path:
    """Deployment results must live directly under ``results/`` with ``prefix``.

    The write scope for the deployment prep task is ``results/deploy_*``; this
    guard makes an accidental write to ``results/eval_*`` or ``results/train_*``
    impossible.
    """

    path = resolve_project_path(value)
    if path.parent != RESULTS_ROOT:
        raise ValueError(f"output must be directly under results/: {path}")
    if not path.name.startswith(prefix):
        raise ValueError(f"output must start with {prefix!r}: {path.name}")
    return path


def expected_image_seq_len(
    image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE, scale_factor: int = SCALE_FACTOR
) -> int:
    """Square no-split visual placeholder count, e.g. ``(384/64)**2 == 36``.

    Mirrors :func:`src.eval.run_zeroshot.compute_image_seq_len`; the deploy code
    must never rely on the processor's shipped default of 64.
    """

    reduction = patch_size * scale_factor
    if image_size <= 0 or reduction <= 0 or image_size % reduction:
        raise ValueError(f"image_size={image_size} must be a positive multiple of {reduction}")
    return (image_size // reduction) ** 2


def select_benchmark_records(
    records: Sequence[Mapping[str, Any]], n: int, seed: int = BENCH_SEED
) -> list[dict[str, Any]]:
    """Deterministic, reproducible ``n``-record subset for the latency table.

    Selection is a seeded ``random.Random(seed).sample`` over record indices,
    then re-sorted by original position, so two runs with the same dataset and
    seed must measure exactly the same images.  The selected id digest is
    recorded in every benchmark JSON.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    materialized = [dict(record) for record in records]
    if n > len(materialized):
        raise ValueError(f"requested {n} benchmark samples but dataset has {len(materialized)}")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(materialized)), n))
    return [materialized[index] for index in chosen]


def selected_ids_digest(records: Sequence[Mapping[str, Any]]) -> str:
    serialized = json.dumps(
        [record.get("id") for record in records], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Descriptive latency statistics with explicit, documented quantiles.

    ``p95``/``p99`` use the nearest-rank definition on the sorted samples:
    ``sorted_values[ceil(q * n) - 1]``.  With 50 samples this is the 48th
    largest sample, which is what a warmed 50-image table can honestly report.
    """

    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarise an empty latency sample")
    ordered = sorted(materialized)
    n = len(ordered)
    mean = math.fsum(ordered) / n
    if n > 1:
        variance = math.fsum((value - mean) ** 2 for value in ordered) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0

    def nearest_rank(q: float) -> float:
        index = max(0, min(n - 1, math.ceil(q * n) - 1))
        return ordered[index]

    return {
        "n": float(n),
        "mean_seconds": mean,
        "median_seconds": (ordered[(n - 1) // 2] + ordered[n // 2]) / 2.0,
        "stdev_seconds": stdev,
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "p95_seconds": nearest_rank(0.95),
        "p99_seconds": nearest_rank(0.99),
    }


def prior_cpu_latency_evidence() -> dict[str, Any]:
    """Existing CPU diagnostics used only to size the planned benchmark.

    ``results/zeroshot_batch_cpu_20_b1.json`` is a 20-sample CPU fp32 +
    LoRA-adapter diagnostic produced earlier by the evaluator.  Its wall time
    includes the one-off model load, so ``elapsed / n`` is a *conservative upper
    bound* for the warmed per-sample latency the deployment benchmark must beat
    or confirm.  This is a planning prior, never a Table 3 number.
    """

    candidates = (
        "results/zeroshot_batch_cpu_20_b1.json",
        "results/zeroshot_batch_cpu_probe_b1.json",
    )
    evidence: list[dict[str, Any]] = []
    for name in candidates:
        path = PROJECT_ROOT / name
        if not path.is_file():
            continue
        payload = read_json(path)
        runtime = payload.get("runtime") or {}
        elapsed = runtime.get("elapsed_seconds_including_load")
        n = payload.get("n")
        if not isinstance(elapsed, (int, float)) or not isinstance(n, int) or n <= 0:
            continue
        evidence.append(
            {
                "source": name,
                "n": n,
                "elapsed_seconds_including_load": float(elapsed),
                "per_sample_upper_bound_seconds": float(elapsed) / n,
                "engine": "hf-cpu-fp32+adapter",
                "caveat": "diagnostic run including load; not a warmed latency benchmark",
            }
        )
    if not evidence:
        return {
            "available": False,
            "estimated_per_sample_seconds": 2.0,
            "source": "no prior CPU evidence; conservative assumption",
        }
    best = min(evidence, key=lambda item: item["per_sample_upper_bound_seconds"])
    return {
        "available": True,
        "evidence": evidence,
        "estimated_per_sample_seconds": best["per_sample_upper_bound_seconds"],
        "source": best["source"],
        "caveat": "upper bound including model load; replace with measured warm latencies",
    }


def estimate_benchmark_seconds(
    n: int = BENCH_SAMPLES, warmup: int = BENCH_WARMUP, *, load_seconds_estimate: float = 10.0
) -> dict[str, float]:
    prior = prior_cpu_latency_evidence()
    per_sample = float(prior["estimated_per_sample_seconds"])
    warm_seconds = (n + warmup) * per_sample
    return {
        "per_sample_seconds_prior": per_sample,
        "warm_seconds_upper_bound": warm_seconds,
        "load_seconds_estimate": load_seconds_estimate,
        "total_seconds_upper_bound": warm_seconds + load_seconds_estimate,
    }


def benchmark_time_estimate(
    n: int = BENCH_SAMPLES, warmup: int = BENCH_WARMUP, *, load_seconds_estimate: float = 10.0
) -> dict[str, Any]:
    """Planning estimate for one warmed benchmark run (never a Table 3 number)."""

    estimate = estimate_benchmark_seconds(n, warmup, load_seconds_estimate=load_seconds_estimate)
    estimate.update(
        {
            "n": n,
            "warmup": warmup,
            "basis": (
                "prior CPU fp32+adapter diagnostic wall time divided by n; conservative "
                "upper bound because it includes the one-off model load"
            ),
            "caveat": "replace with measured warm latencies after the benchmark actually runs",
        }
    )
    return estimate


def deadline_remaining_minutes(now: dt.datetime | None = None) -> float:
    deadline = dt.datetime.fromisoformat(DEADLINE_LOCAL_ISO)
    current = now or dt.datetime.now().astimezone()
    return max(0.0, (deadline - current).total_seconds() / 60.0)


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def make_scratch_dir(prefix: str = "scratch") -> Path:
    """Create a workspace-local scratch directory.

    ``tempfile.mkdtemp`` creates directories with mode ``0o700``; under the DSH
    Windows file sandbox children of such directories are not writable, so all
    deployment scratch files stay inside the workspace and are created with the
    default mode.
    """

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"{prefix}_{uuid.uuid4().hex[:10]}"
    path.mkdir()
    return path


def remove_scratch_dir(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
