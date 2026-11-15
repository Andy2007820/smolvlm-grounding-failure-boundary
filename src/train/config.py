"""Fixed SFT contract: load, validate and fingerprint a training config.

The T5 task fixes the training recipe.  A config that drifts from these values is
rejected instead of being silently trained, so every reported run is auditable
against the assignment contract:

* LoRA ``r=16``, ``lora_alpha=32``, ``learning_rate=1e-4``
* batch size 1 with 8 gradient-accumulation micro-batches => effective batch 8
* ``image_size=384``, bf16 weights, frozen vision tower, seed 42
* gradient checkpointing enabled
* answer ends with explicit EOS token ``49279``; pad token is ``2``
* no-split single image view; ``image_seq_len=36`` derived from patch/scale

``configs/*.json`` are the single source of truth; the script never accepts CLI
overrides that could weaken the contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .vision_tokens import compute_image_seq_len

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

FIXED_SEED = 42
ALLOWED_SEEDS = (42, 43, 44)
FIXED_LORA_R = 16
FIXED_LORA_ALPHA = 32
FIXED_LEARNING_RATE = 1e-4
FIXED_BATCH_SIZE = 1
FIXED_GRAD_ACCUM = 8
FIXED_EFFECTIVE_BATCH = 8
FIXED_IMAGE_SIZE = 384
FIXED_DTYPE = "bfloat16"
FIXED_PATCH_SIZE = 16
FIXED_SCALE_FACTOR = 4
FIXED_EOS_TOKEN_ID = 49279
FIXED_PAD_TOKEN_ID = 2
ALLOWED_ANSWER_REPRESENTATIONS = ("normalized_decimal", "bin1000_integer", "pixel_xyxy", "grid100_integer")
MIN_CHECKPOINT_INTERVAL = 100
MAX_CHECKPOINT_INTERVAL = 200
_ALLOWED_TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")


class ConfigError(ValueError):
    """Raised when a training config violates the fixed T5 contract."""


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        if ".." in path.parts:
            raise ConfigError(f"path must not contain '..': {value!r}")
        resolved = (PROJECT_ROOT / path).resolve()
    return resolved


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigError(f"{name} must be an object, got {type(payload).__name__}")
    return payload


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    return float(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean, got {value!r}")
    return value


def _check_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ConfigError(f"{name} must be {expected!r}, got {actual!r}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config from ``configs/`` (or a project-relative path)."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = validate_config(payload)
    config["_config_path"] = project_relative(config_path)
    config["_config_sha256"] = sha256_file(config_path)
    config["config_fingerprint"] = config_fingerprint(config)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable sha256 over the semantic config (load-time bookkeeping excluded)."""

    payload = {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key != "config_fingerprint"
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a normalised copy of the config."""

    config = _require_mapping(payload, "config")
    config = json.loads(json.dumps(config))  # deep copy with JSON-normalised types

    _check_equal(_require_int(config.get("config_version"), "config_version"), 1, "config_version")
    run_name = config.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ConfigError("run_name must be a non-empty string")
    if any(character in run_name for character in "/\\ \t"):
        raise ConfigError(f"run_name must be a slug without whitespace or separators: {run_name!r}")
    seed = _require_int(config.get("seed"), "seed")
    if seed not in ALLOWED_SEEDS:
        raise ConfigError(f"seed must be one of {ALLOWED_SEEDS!r} for the fixed T5/T6 contract, got {seed!r}")

    model = _require_mapping(config.get("model"), "model")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ConfigError("model.path must be a non-empty project-relative path")
    resolve_project_path(model_path)
    _check_equal(model.get("dtype"), FIXED_DTYPE, "model.dtype")
    _check_equal(_require_bool(model.get("freeze_vision_tower"), "model.freeze_vision_tower"), True, "model.freeze_vision_tower")
    _check_equal(
        _require_bool(model.get("gradient_checkpointing"), "model.gradient_checkpointing"),
        True,
        "model.gradient_checkpointing",
    )
    _check_equal(
        _require_bool(model.get("local_files_only"), "model.local_files_only"), True, "model.local_files_only"
    )

    lora = _require_mapping(config.get("lora"), "lora")
    _check_equal(_require_int(lora.get("r"), "lora.r"), FIXED_LORA_R, "lora.r")
    _check_equal(_require_int(lora.get("lora_alpha"), "lora.lora_alpha"), FIXED_LORA_ALPHA, "lora.lora_alpha")
    dropout = _require_number(lora.get("lora_dropout", 0.05), "lora.lora_dropout")
    if not 0.0 <= dropout <= 0.1:
        raise ConfigError(f"lora.lora_dropout must be in [0, 0.1], got {dropout}")
    leaves = lora.get("target_modules")
    if not isinstance(leaves, list) or tuple(leaves) != _ALLOWED_TARGET_LEAVES:
        raise ConfigError(
            f"lora.target_modules must be {list(_ALLOWED_TARGET_LEAVES)!r} in that order, got {leaves!r}"
        )
    _check_equal(lora.get("target_scope"), "model.text_model", "lora.target_scope")
    _check_equal(lora.get("task_type"), "CAUSAL_LM", "lora.task_type")
    _check_equal(lora.get("bias"), "none", "lora.bias")

    image = _require_mapping(config.get("image"), "image")
    image_size = _require_int(image.get("image_size"), "image.image_size")
    _check_equal(image_size, FIXED_IMAGE_SIZE, "image.image_size")
    _check_equal(
        _require_bool(image.get("do_image_splitting"), "image.do_image_splitting"),
        False,
        "image.do_image_splitting",
    )
    patch_size = _require_int(image.get("patch_size"), "image.patch_size")
    scale_factor = _require_int(image.get("scale_factor"), "image.scale_factor")
    _check_equal(patch_size, FIXED_PATCH_SIZE, "image.patch_size")
    _check_equal(scale_factor, FIXED_SCALE_FACTOR, "image.scale_factor")
    derived = compute_image_seq_len(image_size, patch_size, scale_factor)
    _check_equal(
        _require_int(image.get("expected_image_seq_len"), "image.expected_image_seq_len"),
        derived,
        "image.expected_image_seq_len",
    )

    answer = _require_mapping(config.get("answer"), "answer")
    _check_equal(_require_int(answer.get("eos_token_id"), "answer.eos_token_id"), FIXED_EOS_TOKEN_ID, "answer.eos_token_id")
    _check_equal(_require_int(answer.get("pad_token_id"), "answer.pad_token_id"), FIXED_PAD_TOKEN_ID, "answer.pad_token_id")
    precision = _require_int(answer.get("precision", 4), "answer.precision")
    if not 1 <= precision <= 8:
        raise ConfigError(f"answer.precision must be in [1, 8], got {precision}")
    answer_format = answer.get("format")
    if not isinstance(answer_format, str) or any(
        token not in answer_format for token in ("{x1", "{y1", "{x2", "{y2")
    ):
        raise ConfigError("answer.format must contain {x1}, {y1}, {x2}, {y2} placeholders")
    representation = answer.get("representation", "normalized_decimal")
    if representation not in ALLOWED_ANSWER_REPRESENTATIONS:
        raise ConfigError(
            f"answer.representation must be one of {ALLOWED_ANSWER_REPRESENTATIONS!r}, got {representation!r}"
        )

    data = _require_mapping(config.get("data"), "data")
    for split in ("train", "val"):
        entries = data.get(split)
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"data.{split} must be a non-empty list")
        for index, entry in enumerate(entries):
            entry = _require_mapping(entry, f"data.{split}[{index}]")
            entry_path = entry.get("path")
            if not isinstance(entry_path, str) or not entry_path.strip():
                raise ConfigError(f"data.{split}[{index}].path must be a non-empty string")
            resolve_project_path(entry_path)
            if not isinstance(entry.get("source"), str) or not entry["source"]:
                raise ConfigError(f"data.{split}[{index}].source must be a non-empty string")
            entry_split = entry.get("split")
            if not isinstance(entry_split, str) or not entry_split:
                raise ConfigError(f"data.{split}[{index}].split must be a non-empty string")
            limit = entry.get("limit")
            if limit is not None:
                _require_int(limit, f"data.{split}[{index}].limit")
                if limit <= 0:
                    raise ConfigError(f"data.{split}[{index}].limit must be positive or null")
            _check_equal(entry.get("select", "first"), "first", f"data.{split}[{index}].select")
    _check_equal(
        _require_bool(data.get("id_intersection_must_be_zero"), "data.id_intersection_must_be_zero"),
        True,
        "data.id_intersection_must_be_zero",
    )
    _check_equal(_require_int(data.get("example_cache_max", 200), "data.example_cache_max"), 200, "data.example_cache_max")

    training = _require_mapping(config.get("training"), "training")
    _check_equal(_require_int(training.get("batch_size"), "training.batch_size"), FIXED_BATCH_SIZE, "training.batch_size")
    _check_equal(
        _require_int(training.get("gradient_accumulation_steps"), "training.gradient_accumulation_steps"),
        FIXED_GRAD_ACCUM,
        "training.gradient_accumulation_steps",
    )
    _check_equal(
        _require_int(training.get("effective_batch_size"), "training.effective_batch_size"),
        FIXED_EFFECTIVE_BATCH,
        "training.effective_batch_size",
    )
    _check_equal(
        _require_number(training.get("learning_rate"), "training.learning_rate"),
        FIXED_LEARNING_RATE,
        "training.learning_rate",
    )
    weight_decay = _require_number(training.get("weight_decay", 0.0), "training.weight_decay")
    if weight_decay < 0:
        raise ConfigError(f"training.weight_decay must be >= 0, got {weight_decay}")
    num_epochs = _require_int(training.get("num_epochs"), "training.num_epochs")
    if num_epochs <= 0:
        raise ConfigError(f"training.num_epochs must be positive, got {num_epochs}")
    save_every = _require_int(training.get("save_every_steps"), "training.save_every_steps")
    if not MIN_CHECKPOINT_INTERVAL <= save_every <= MAX_CHECKPOINT_INTERVAL:
        raise ConfigError(
            f"training.save_every_steps must be within [{MIN_CHECKPOINT_INTERVAL}, {MAX_CHECKPOINT_INTERVAL}], got {save_every}"
        )
    log_every = _require_int(training.get("log_every_steps", 1), "training.log_every_steps")
    if log_every <= 0:
        raise ConfigError("training.log_every_steps must be positive")
    _check_equal(training.get("lr_scheduler"), "cosine", "training.lr_scheduler")
    warmup_ratio = _require_number(training.get("warmup_ratio", 0.03), "training.warmup_ratio")
    if not 0.0 <= warmup_ratio <= 0.5:
        raise ConfigError(f"training.warmup_ratio must be in [0, 0.5], got {warmup_ratio}")
    max_grad_norm = _require_number(training.get("max_grad_norm", 1.0), "training.max_grad_norm")
    if max_grad_norm <= 0:
        raise ConfigError("training.max_grad_norm must be positive")
    max_steps = training.get("max_steps")
    if max_steps is not None:
        _require_int(max_steps, "training.max_steps")
        if max_steps <= 0:
            raise ConfigError("training.max_steps must be positive or null")

    probe = _require_mapping(config.get("probe"), "probe")
    if probe.get("records") not in ("train", "val"):
        raise ConfigError("probe.records must be 'train' or 'val'")
    probe_n = _require_int(probe.get("n"), "probe.n")
    if probe_n < 0:
        raise ConfigError("probe.n must be >= 0")
    if _require_int(probe.get("max_new_tokens"), "probe.max_new_tokens") <= 0:
        raise ConfigError("probe.max_new_tokens must be positive")
    _check_equal(_require_bool(probe.get("do_sample", False), "probe.do_sample"), False, "probe.do_sample")

    output = _require_mapping(config.get("output"), "output")
    output_dir = output.get("dir")
    if not isinstance(output_dir, str) or not output_dir.startswith("adapters/"):
        raise ConfigError("output.dir must be a project-relative path under adapters/")
    resolve_project_path(output_dir)
    log_path = output.get("log")
    if not isinstance(log_path, str) or not log_path.startswith("logs/train/"):
        raise ConfigError("output.log must be a project-relative path under logs/train/")
    resolve_project_path(log_path)

    return config
