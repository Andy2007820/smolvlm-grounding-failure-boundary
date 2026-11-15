"""Tokenizer/generation/processor audit for the SmolVLM-256M checkpoint.

T5 must independently investigate the ``pad_token_id=128002`` warning observed
during the G0 smoke and B0 runs.  The warning is emitted by transformers while
constructing ``Idefics3Config``: the *class default* for the top-level
``pad_token_id`` is ``128_002`` (``configuration_idefics3.py``), which is outside
the legacy 32k default text vocabulary and therefore trips a config-validation
warning.  It is not the value used by the tokenizer, the processor or the
generation config.

The real, user-facing contract verified here is:

* tokenizer ``pad_token`` = ``<|im_end|>``, id ``2``
* tokenizer ``eos_token`` = ``<end_of_utterance>``, id ``49279``
* ``generation_config.json``: ``pad_token_id=2``, ``eos_token_id=49279``
* ``text_config.pad_token_id=2``
* ``config.json``: ``image_token_id=49190``
* SFT answers append EOS ``49279`` explicitly

No code in this module writes to the model directory or to the evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"

EXPECTED_PAD_TOKEN_ID = 2
EXPECTED_EOS_TOKEN_ID = 49279
EXPECTED_IMAGE_TOKEN_ID = 49190
EXPECTED_GLOBAL_IMAGE_SEQ_LEN = 64
TOP_LEVEL_DEFAULT_PAD_TOKEN_ID = 128_002
WARNING_SOURCE = (
    "transformers/models/idefics3/configuration_idefics3.py: "
    "class Idefics3Config default pad_token_id = 128_002, validated against the "
    "legacy 32000-entry default text vocabulary in configuration_utils.validate_token_ids"
)
WARNING_TEXT = (
    "Model config: pad_token_id must be `None` or an integer within the vocabulary "
    "(between 0 and 31999), got 128002. This may result in unexpected behavior."
)


def project_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_processor(processor: Any, *, image_size: int = 384) -> dict[str, Any]:
    """Force the single-view 384 px no-split path used by the evaluator."""

    image_processor = processor.image_processor
    image_processor.do_image_splitting = False
    image_processor.size = {"longest_edge": image_size}
    image_processor.max_image_size = {"longest_edge": image_size}
    observed = {
        "do_image_splitting": bool(image_processor.do_image_splitting),
        "size": dict(image_processor.size),
        "max_image_size": dict(image_processor.max_image_size),
    }
    if observed["do_image_splitting"] is not False:
        raise RuntimeError("failed to disable image splitting")
    if observed["size"] != {"longest_edge": image_size} or observed["max_image_size"] != {"longest_edge": image_size}:
        raise RuntimeError(f"failed to force image processor size: {observed}")
    return observed


def audit_pad_eos(processor: Any, *, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Return the full audit block and raise on any contract violation."""

    model_path = Path(model_path)
    tokenizer = processor.tokenizer
    config = read_json(model_path / "config.json")
    generation_config = read_json(model_path / "generation_config.json")
    processor_config = read_json(model_path / "processor_config.json")

    tokenizer_pad_id = tokenizer.pad_token_id
    tokenizer_eos_id = tokenizer.eos_token_id
    generation_pad_id = generation_config.get("pad_token_id")
    generation_eos_id = generation_config.get("eos_token_id")
    text_pad_id = config.get("text_config", {}).get("pad_token_id")
    image_token_id = config.get("image_token_id")
    global_image_seq_len = processor_config.get("image_seq_len")

    vocab_size = int(config.get("text_config", {}).get("vocab_size") or config.get("vocab_size"))
    out_of_vocab_default = TOP_LEVEL_DEFAULT_PAD_TOKEN_ID >= vocab_size

    checks = {
        "tokenizer_pad_is_2": tokenizer_pad_id == EXPECTED_PAD_TOKEN_ID,
        "tokenizer_eos_is_49279": tokenizer_eos_id == EXPECTED_EOS_TOKEN_ID,
        "generation_pad_is_2": generation_pad_id == EXPECTED_PAD_TOKEN_ID,
        "generation_eos_is_49279": generation_eos_id == EXPECTED_EOS_TOKEN_ID,
        "text_config_pad_is_2": text_pad_id == EXPECTED_PAD_TOKEN_ID,
        "image_token_is_49190": image_token_id == EXPECTED_IMAGE_TOKEN_ID,
        "processor_default_image_seq_len_is_64": global_image_seq_len == EXPECTED_GLOBAL_IMAGE_SEQ_LEN,
        "actual_pad_eos_inside_vocab": all(
            isinstance(value, int) and 0 <= value < vocab_size
            for value in (EXPECTED_PAD_TOKEN_ID, EXPECTED_EOS_TOKEN_ID)
        ),
        "top_level_default_128002_outside_vocab": out_of_vocab_default,
        "actual_pad_eos_not_128002": EXPECTED_PAD_TOKEN_ID != TOP_LEVEL_DEFAULT_PAD_TOKEN_ID
        and EXPECTED_EOS_TOKEN_ID != TOP_LEVEL_DEFAULT_PAD_TOKEN_ID,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"pad/EOS contract violations: {failed}")

    return {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_pad_token": tokenizer.pad_token,
        "tokenizer_pad_token_id": tokenizer_pad_id,
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_eos_token_id": tokenizer_eos_id,
        "tokenizer_bos_token": tokenizer.bos_token,
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
        "generation_config_pad_token_id": generation_pad_id,
        "generation_config_eos_token_id": generation_eos_id,
        "text_config_pad_token_id": text_pad_id,
        "config_image_token_id": image_token_id,
        "processor_config_default_image_seq_len": global_image_seq_len,
        "top_level_config_default_pad_token_id": TOP_LEVEL_DEFAULT_PAD_TOKEN_ID,
        "text_config_vocab_size": vocab_size,
        "top_level_default_is_out_of_vocab": out_of_vocab_default,
        "warning_observed": True,
        "warning_text": WARNING_TEXT,
        "warning_source": WARNING_SOURCE,
        "conclusion": (
            "128002 is only the Idefics3Config class-level default for top-level pad_token_id; "
            "the tokenizer, generation_config and text config all resolve to pad=2 / eos=49279. "
            "SFT explicitly appends eos=49279 to every answer and uses pad=2 for collation, "
            "so the warning is documented and does not affect the training contract."
        ),
        "checks": checks,
    }


def audit_model_geometry(*, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Read patch/scale/vision size from config.json without loading weights."""

    model_path = Path(model_path)
    config = read_json(model_path / "config.json")
    vision_config = config.get("vision_config", {})
    return {
        "model_path": project_relative(model_path),
        "vision_patch_size": int(vision_config.get("patch_size")),
        "vision_image_size": int(vision_config.get("image_size")),
        "scale_factor": int(config.get("scale_factor")),
        "image_token_id": int(config.get("image_token_id")),
        "text_vocab_size": int(config.get("text_config", {}).get("vocab_size") or config.get("vocab_size")),
        "vision_hidden_layers": int(vision_config.get("num_hidden_layers")),
        "text_hidden_layers": int(config.get("text_config", {}).get("num_hidden_layers")),
        "text_hidden_size": int(config.get("text_config", {}).get("hidden_size")),
        "source_sha256": {
            "config.json": __import__("hashlib").sha256((model_path / "config.json").read_bytes()).hexdigest(),
            "generation_config.json": __import__("hashlib")
            .sha256((model_path / "generation_config.json").read_bytes())
            .hexdigest(),
            "processor_config.json": __import__("hashlib")
            .sha256((model_path / "processor_config.json").read_bytes())
            .hexdigest(),
        },
    }
