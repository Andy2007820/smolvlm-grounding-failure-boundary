"""Auditable checkpoint and resume helpers for SFT.

Every checkpoint directory contains:

* ``adapter/`` - PEFT adapter weights and ``adapter_config.json``
* ``trainer_state.pt`` - optimizer state, scheduler state, CPU/CUDA RNG states
* ``checkpoint_meta.json`` - written *last* and atomically; a directory without a
  complete metadata file is treated as an interrupted write, never resumed.

``checkpoint_meta.json`` records the config fingerprint, data hashes, selected ID
digest, LoRA target modules, trainable/frozen parameter counts, EOS/pad contract,
visual-token contract, environment versions, GPU lease and memory readings, and
the source hashes.  Resume refuses to proceed when any identity field differs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PREFIX = "checkpoint-"


class ResumeError(ValueError):
    """Raised when a checkpoint does not match the current run identity."""


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_directory_files(directory: Path) -> dict[str, str]:
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def capture_rng_state() -> dict[str, Any]:
    """Capture global RNG states; restore with :func:`restore_rng_state`."""

    import torch

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
        "torch_cuda": None,
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # pragma: no cover - numpy is expected to exist for torch
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    if "python_random" in state:
        random.setstate(state["python_random"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(numpy_state)
        except Exception:  # pragma: no cover
            pass
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def checkpoint_step_from_name(name: str) -> int | None:
    """Return the numeric step encoded in ``checkpoint-<digits>`` or ``None``."""

    if not isinstance(name, str) or not name.startswith(CHECKPOINT_PREFIX):
        return None
    suffix = name[len(CHECKPOINT_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def sorted_checkpoint_names(names: Sequence[str]) -> list[str]:
    """Sort valid checkpoint directory names by numeric step (pure helper)."""

    valid = [name for name in names if checkpoint_step_from_name(name) is not None]
    return sorted(valid, key=lambda name: checkpoint_step_from_name(name))


def select_latest_complete_checkpoint(candidates: Sequence[tuple[str, bool]]) -> str | None:
    """Pick the highest-step candidate whose metadata declares ``complete=True``.

    ``candidates`` is ``(directory_name, complete)`` in any order.  Pure helper so
    the resume-selection rule is testable without touching the filesystem.
    """

    best_name: str | None = None
    best_step = -1
    for name, complete in candidates:
        step = checkpoint_step_from_name(name)
        if step is None or not complete:
            continue
        if step > best_step:
            best_step = step
            best_name = name
    return best_name


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    return Path(output_dir) / f"{CHECKPOINT_PREFIX}{step:06d}"


def find_checkpoint_dirs(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    found = []
    for child in output_dir.iterdir():
        if child.is_dir() and checkpoint_step_from_name(child.name) is not None:
            found.append(child)
    return sorted(found, key=lambda path: checkpoint_step_from_name(path.name))


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest checkpoint whose metadata declares it complete."""

    candidates: list[tuple[str, bool]] = []
    for candidate in find_checkpoint_dirs(output_dir):
        metadata_path = candidate / "checkpoint_meta.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = read_json(metadata_path)
        except Exception:
            continue
        candidates.append((candidate.name, metadata.get("complete") is True))
    latest_name = select_latest_complete_checkpoint(candidates)
    return (Path(output_dir) / latest_name) if latest_name is not None else None


def build_checkpoint_metadata(
    *,
    run_id: str,
    step: int,
    epoch: int,
    samples_seen: int,
    config: Mapping[str, Any],
    config_fingerprint: str,
    seed: int,
    model_path: str,
    model_files: Mapping[str, Any],
    dataset_evidence: Sequence[Mapping[str, Any]],
    selected_ids_sha256: str,
    train_order_sha256: str,
    id_intersection: Mapping[str, Any],
    lora: Mapping[str, Any],
    trainable_parameters: int,
    total_parameters: int,
    frozen_vision_parameters: int,
    trainable_vision_parameters: int,
    image_contract: Mapping[str, Any],
    answer_contract: Mapping[str, Any],
    training: Mapping[str, Any],
    gpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    resumed_from: str | None = None,
    resume_count: int = 0,
) -> dict[str, Any]:
    """Assemble the full audit payload stored next to every checkpoint."""

    if not config_fingerprint:
        raise ValueError("config_fingerprint is required")
    trainable_ratio = trainable_parameters / total_parameters if total_parameters else 0.0
    return {
        "format": "t5-sft-checkpoint-v1",
        "complete": False,  # flipped to True only after all files are written
        "run_id": run_id,
        "created_at": now_iso(),
        "step": step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "config": dict(config),
        "config_fingerprint": config_fingerprint,
        "seed": seed,
        "model": {
            "path": model_path,
            "files": dict(model_files),
        },
        "datasets": list(dataset_evidence),
        "selected_ids_sha256": selected_ids_sha256,
        "train_order_sha256": train_order_sha256,
        "id_intersection": dict(id_intersection),
        "lora": dict(lora),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_ratio": trainable_ratio,
        "frozen_vision_parameters": frozen_vision_parameters,
        "trainable_vision_parameters": trainable_vision_parameters,
        "image_contract": dict(image_contract),
        "answer_contract": dict(answer_contract),
        "training": dict(training),
        "gpu": dict(gpu),
        "environment": dict(environment),
        "source_hashes": dict(source_hashes),
        "resumed_from": resumed_from,
        "resume_count": resume_count,
        "checkpoint_policy": (
            f"save adapter + optimizer/scheduler/RNG every training.save_every_steps "
            f"({training.get('save_every_steps')}) optimizer steps; metadata written last and atomically"
        ),
        "resume_contract": (
            "resume is allowed only when config_fingerprint, seed, model file hashes, dataset hashes, "
            "image_seq_len and EOS/pad ids all match"
        ),
    }


def validate_resume_metadata(
    metadata: Mapping[str, Any],
    *,
    config_fingerprint: str,
    seed: int,
    model_files: Mapping[str, Any],
    dataset_evidence: Sequence[Mapping[str, Any]],
    image_seq_len: int,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    """Raise :class:`ResumeError` unless the checkpoint identity matches this run."""

    if metadata.get("format") != "t5-sft-checkpoint-v1":
        raise ResumeError(f"unsupported checkpoint format: {metadata.get('format')!r}")
    if metadata.get("complete") is not True:
        raise ResumeError("checkpoint is incomplete")
    checks = (
        ("config_fingerprint", metadata.get("config_fingerprint"), config_fingerprint),
        ("seed", metadata.get("seed"), seed),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ResumeError(f"resume mismatch for {name}: {actual!r} != {expected!r}")

    stored_model = metadata.get("model", {}).get("files")
    if stored_model != dict(model_files):
        raise ResumeError("resume mismatch: model file hashes differ from the checkpoint")
    stored_datasets = metadata.get("datasets")
    expected_datasets = [dict(entry) for entry in dataset_evidence]
    if stored_datasets != expected_datasets:
        raise ResumeError("resume mismatch: dataset evidence differs from the checkpoint")
    stored_image = metadata.get("image_contract", {})
    if int(stored_image.get("image_seq_len", -1)) != int(image_seq_len):
        raise ResumeError("resume mismatch: image_seq_len differs")
    stored_answer = metadata.get("answer_contract", {})
    if int(stored_answer.get("eos_token_id", -1)) != int(eos_token_id):
        raise ResumeError("resume mismatch: EOS token id differs")
    if int(stored_answer.get("pad_token_id", -1)) != int(pad_token_id):
        raise ResumeError("resume mismatch: pad token id differs")


def save_adapter(model: Any, directory: Path) -> dict[str, Any]:
    """Save a PEFT adapter and return per-file hashes plus byte sizes."""

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite adapter directory: {directory}")
    directory.mkdir(parents=True)
    model.save_pretrained(directory)
    files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    if not files:
        raise RuntimeError("adapter save produced no files")
    return files


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    metadata: Mapping[str, Any],
) -> Path:
    """Write one checkpoint; metadata is the commit marker and is written last."""

    import torch

    directory = checkpoint_dir(output_dir, step)
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    adapter_directory = directory / "adapter"
    adapter_files = save_adapter(model, adapter_directory)

    trainer_state = {
        "step": int(step),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
    }
    state_path = directory / "trainer_state.pt"
    temporary_state = state_path.with_name(state_path.name + ".tmp")
    torch.save(trainer_state, temporary_state)
    os.replace(temporary_state, state_path)

    payload = dict(metadata)
    payload.update(
        {
            "checkpoint_step": int(step),
            "checkpoint_dir": directory.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "adapter_files": adapter_files,
            "trainer_state_file": state_path.name,
            "trainer_state_sha256": sha256_file(state_path),
            "trainer_state_bytes": state_path.stat().st_size,
            "complete": True,
            "completed_at": now_iso(),
        }
    )
    write_json_atomic(directory / "checkpoint_meta.json", payload)
    return directory


def load_trainer_state(checkpoint: Path) -> dict[str, Any]:
    state_path = Path(checkpoint) / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    import torch

    return torch.load(state_path, map_location="cpu", weights_only=False)


def save_final_adapter(output_dir: Path, model: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Save the final adapter at ``output_dir`` with a summary next to it."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_directory = output_dir / "adapter"
    if adapter_directory.exists():
        raise FileExistsError(f"final adapter already exists: {adapter_directory}")
    files = save_adapter(model, adapter_directory)
    payload = dict(metadata)
    payload.update(
        {
            "format": "t5-sft-final-v1",
            "complete": True,
            "adapter_files": files,
            "finished_at": now_iso(),
        }
    )
    write_json_atomic(output_dir / "training_summary.json", payload)
    return payload


def copy_checkpoint(source: Path, destination: Path) -> Path:
    """Copy a checkpoint directory (used only by explicit audit tooling)."""

    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    return destination
