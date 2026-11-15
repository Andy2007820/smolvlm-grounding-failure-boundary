"""Dataset preparation and label masking for grounding SFT.

Every training sequence is built as:

    <chat template user turn>Assistant: [x1, y1, x2, y2]<end_of_utterance>

The prompt portion (system/user turn plus the ``Assistant:`` prefix and the
``<image>`` placeholder run) receives ``-100`` labels; only the four answer
coordinates and the explicit ``<end_of_utterance>`` (token id 49279) contribute
to the cross-entropy loss.  No prompt token reaches the loss.

The visual-token helper is invoked on *every* image-processing call, so the
processor's shipped ``image_seq_len=64`` default is never permanently changed.
Sequences are never truncated silently and image views are never padded or
cropped independently of the image processor.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from src.eval.grounding import parse_box
from src.eval.prompts import PROMPT_TEMPLATE, build_messages
from .coordinate_variants import (
    PROMPT_SHA256 as VARIANT_PROMPT_SHA256,
    build_messages as build_variant_messages,
    encode_answer as encode_variant_answer,
    prompt_id as variant_prompt_id,
    validate_representation,
)
from src.eval.dataset import load_dataset, read_jsonl, sha256_file

from .vision_tokens import (
    compute_image_seq_len,
    inspect_visual_token_contract,
    set_runtime_image_seq_len,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct"
DEFAULT_ANSWER_FORMAT = "[{x1}, {y1}, {x2}, {y2}]"
DEFAULT_MAX_LENGTH = 1024
PAD_TOKEN_ID = 2
EOS_TOKEN_ID = 49279
IMAGE_TOKEN_ID = 49190


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if ".." in path.parts:
        raise ValueError(f"path must not contain '..': {value!r}")
    return (PROJECT_ROOT / path).resolve()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_sequence(values: Sequence[Any]) -> str:
    serialized = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return sha256_text(serialized)


def format_answer(
    answer: Sequence[float],
    *,
    precision: int = 4,
    template: str = DEFAULT_ANSWER_FORMAT,
    tolerance: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render the supervised answer text and prove it roundtrips through the evaluator parser.

    ``precision`` is the preferred number of decimals.  If rounding at that
    precision produces a box the frozen evaluator parser would reject (for
    example x1==x2 after rounding), a finer precision is used.  The returned
    audit block records the precision actually used and the worst absolute
    coordinate error.
    """

    if len(answer) != 4:
        raise ValueError(f"answer must have four coordinates, got {len(answer)}")
    values = [float(value) for value in answer]
    for candidate in sorted({int(precision), max(int(precision), 6), max(int(precision), 8)}):
        try:
            text = template.format(
                x1=f"{values[0]:.{candidate}f}",
                y1=f"{values[1]:.{candidate}f}",
                x2=f"{values[2]:.{candidate}f}",
                y2=f"{values[3]:.{candidate}f}",
            )
        except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover - config validated
            raise ValueError(f"invalid answer template {template!r}") from exc
        parsed = parse_box(text)
        if parsed is None:
            continue
        max_abs_error = max(abs(a - b) for a, b in zip(parsed, values))
        allowed = tolerance if tolerance is not None else 0.5 * (10.0 ** -candidate) + 1e-12
        if max_abs_error <= allowed:
            return text, {
                "answer_text": text,
                "precision_used": candidate,
                "max_abs_rounding_error": max_abs_error,
                "evaluator_parse_ok": True,
            }
    raise ValueError(f"answer {values!r} cannot be rendered into an evaluator-parseable box")


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows without image checks (fast, for ID intersection evidence)."""

    resolved = resolve_project_path(path)
    return read_jsonl(resolved)


def select_records(records: Sequence[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(records)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer or None, got {limit!r}")
    if limit > len(records):
        raise ValueError(f"limit {limit} exceeds dataset size {len(records)}")
    return list(records[:limit])


def epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    """Deterministic epoch permutation, independent of global RNG state."""

    if size < 0:
        raise ValueError("size must be non-negative")
    order = list(range(size))
    random.Random(seed * 1_000_003 + epoch * 1009 + 17).shuffle(order)
    return order


def assert_train_val_disjoint(
    train_sets: Mapping[str, Sequence[dict[str, Any]]],
    val_sets: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Require zero ID (and, additionally, zero image-path) overlap.

    Returns an audit block; raises ``AssertionError`` if any overlap is found.
    """

    if not train_sets or not val_sets:
        raise ValueError("train_sets and val_sets must be non-empty")
    train_ids: dict[str, set[str]] = {}
    val_ids: dict[str, set[str]] = {}
    for name, records in train_sets.items():
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate ids inside training set {name}")
        train_ids[name] = set(ids)
    for name, records in val_sets.items():
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate ids inside validation set {name}")
        val_ids[name] = set(ids)

    pair_counts: dict[str, int] = {}
    for train_name, train_set in train_ids.items():
        for val_name, val_set in val_ids.items():
            pair_counts[f"{train_name}|{val_name}"] = len(train_set & val_set)
    global_train: set[str] = set()
    for train_set in train_ids.values():
        global_train |= train_set
    global_val: set[str] = set()
    for val_set in val_ids.values():
        global_val |= val_set
    global_train_val_intersection = global_train & global_val

    train_images = {record["image"] for records in train_sets.values() for record in records}
    val_images = {record["image"] for records in val_sets.values() for record in records}
    image_intersection = train_images & val_images

    audit = {
        "train_id_counts": {name: len(values) for name, values in train_ids.items()},
        "val_id_counts": {name: len(values) for name, values in val_ids.items()},
        "pair_id_intersection_counts": pair_counts,
        "global_train_val_id_intersection_count": len(global_train_val_intersection),
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "train_val_image_path_intersection_count": len(image_intersection),
        "requirement": "train/val ID intersection must be exactly 0",
    }
    if any(pair_counts.values()) or len(global_train_val_intersection) != 0:
        raise AssertionError(f"train/val ID intersection is not zero: {audit}")
    return audit


def load_split_records(entry: Mapping[str, Any], *, expected_n: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly validate one split (all eight fields, images, GT geometry)."""

    path = resolve_project_path(entry["path"])
    rows = load_dataset(
        path,
        project_root=PROJECT_ROOT,
        expected_source=entry["source"],
        split=entry["split"],
        expected_n=expected_n,
        limit=entry.get("limit"),
        check_images=True,
    )
    return rows, {
        "path": project_relative(path),
        "source": entry["source"],
        "split": entry["split"],
        "limit": entry.get("limit"),
        "rows_loaded": len(rows),
        "sha256": sha256_file(path),
    }


@dataclass
class SftExample:
    record_id: str
    image: str
    question: str
    answer_text: str
    answer_gt: list[float]
    input_ids: Any
    attention_mask: Any
    labels: Any
    pixel_values: Any
    audit: dict[str, Any] = field(default_factory=dict)


def build_sft_example(
    record: Mapping[str, Any],
    processor: Any,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    image_size: int = 384,
    patch_size: int = 16,
    scale_factor: int = 4,
    answer_precision: int = 4,
    answer_format: str = DEFAULT_ANSWER_FORMAT,
    representation: str = "normalized_decimal",
    max_length: int = DEFAULT_MAX_LENGTH,
    pad_token_id: int = PAD_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> SftExample:
    """Build one masked SFT example and all supporting audit evidence."""

    import torch

    image_path = resolve_project_path(record["image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    representation = validate_representation(representation)
    if representation == "normalized_decimal":
        answer_text, answer_audit = format_answer(
            record["answer"], precision=answer_precision, template=answer_format
        )
    else:
        answer_text, answer_audit = encode_variant_answer(record, representation)
    image_seq_len = compute_image_seq_len(image_size, patch_size, scale_factor)

    user_messages = build_variant_messages(record["question"], representation)
    assistant_message = {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    prompt = processor.apply_chat_template(user_messages, add_generation_prompt=True)
    full_conversation = processor.apply_chat_template(
        user_messages + [assistant_message], add_generation_prompt=False
    )

    # The runtime visual-token count is restored to the shipped default only in
    # the sense that the on-disk processor_config.json is never touched.  Every
    # call sets the value for this invocation and re-proves the default.
    processor_config_path = Path(model_path) / "processor_config.json"
    runtime_audit = set_runtime_image_seq_len(
        processor, image_seq_len, processor_config_path=processor_config_path
    )

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        original_size = list(image.size)
    prompt_processed = processor(text=prompt, images=[image], return_tensors="pt")
    prompt_contract = inspect_visual_token_contract(
        prompt_processed,
        patch_size=patch_size,
        scale_factor=scale_factor,
        image_size=image_size,
        image_token_id=image_token_id,
    )
    # The full conversation must go through the processor too: only the
    # processor expands ONE <image> token into the image_seq_len=36 placeholder
    # run (plus surrounding fake tokens). Tokenizing the raw template would give
    # one <image> token and misaligned labels.
    full_processed = processor(text=full_conversation, images=[image], return_tensors="pt")
    full_contract = inspect_visual_token_contract(
        full_processed,
        patch_size=patch_size,
        scale_factor=scale_factor,
        image_size=image_size,
        image_token_id=image_token_id,
    )
    contract = full_contract
    if full_contract.placeholder_count != prompt_contract.placeholder_count:
        raise RuntimeError(
            "generation prompt and full conversation disagree on visual placeholders: "
            f"{prompt_contract.placeholder_count} vs {full_contract.placeholder_count}"
        )

    tokenizer = processor.tokenizer
    prompt_ids = [int(value) for value in prompt_processed["input_ids"][0]]
    input_ids = [int(value) for value in full_processed["input_ids"][0]]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "the expanded full conversation is not a token-level prefix extension of the expanded "
            "generation prompt; train/eval prompt contract would differ"
        )
    if not full_conversation.startswith(prompt):
        raise RuntimeError("chat-template prefix contract violated: full conversation does not start with the prompt")
    continuation = full_conversation[len(prompt) :]
    expected_continuation_prefix = " " + answer_text
    if not continuation.startswith(expected_continuation_prefix):
        raise RuntimeError(
            f"assistant continuation does not start with the expected answer prefix: {continuation[:80]!r}"
        )

    # Locate the label span from the actual expanded input_ids.  The chat template
    # closes the user turn with <end_of_utterance> as well, so the prompt contains
    # one EOS token; the answer EOS is the single EOS at/after the prompt boundary.
    answer_ids = [int(value) for value in tokenizer(expected_continuation_prefix, add_special_tokens=False)["input_ids"]]
    all_eos_positions = [index for index, token_id in enumerate(input_ids) if token_id == eos_token_id]
    prompt_eos_positions = [index for index in all_eos_positions if index < len(prompt_ids)]
    answer_eos_positions = [index for index in all_eos_positions if index >= len(prompt_ids)]
    if not prompt_eos_positions:
        raise RuntimeError("generation prompt does not contain the user-turn <end_of_utterance> token")
    if len(answer_eos_positions) != 1:
        raise RuntimeError(
            f"expected exactly one explicit answer EOS token {eos_token_id} after the prompt, "
            f"got {answer_eos_positions}"
        )
    eos_position = answer_eos_positions[0]
    label_start = eos_position - len(answer_ids)
    if label_start != len(prompt_ids):
        raise RuntimeError(
            "answer does not begin exactly at the generation-prompt boundary: "
            f"label_start={label_start} prompt_tokens={len(prompt_ids)}"
        )
    if input_ids[label_start:eos_position] != answer_ids:
        raise RuntimeError(
            "the expanded input_ids answer span differs from the standalone continuation tokenization; "
            "labels must align with the trained sequence, so no silent fallback is allowed"
        )

    # Independent character-offset cross-check: the unexpanded tokenizer sees one
    # <image> token; the processor replaces that single image placeholder run
    # with image_seq_len image tokens plus fake delimiters, all before the answer.
    # The empirically measured expansion delta from the prompt is therefore also
    # the offset between the raw answer token index and the expanded index.
    raw_prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
    image_expansion_delta = len(prompt_ids) - len(raw_prompt_ids)
    encoded = tokenizer(full_conversation, add_special_tokens=False, return_offsets_mapping=True)
    raw_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    raw_placeholder_count = sum(token_id == image_token_id for token_id in raw_ids)
    if raw_placeholder_count != 1:
        raise RuntimeError(
            f"unescaped conversation tokenization must carry exactly one <image> token, got {raw_placeholder_count}"
        )
    answer_character_start = len(prompt) + 1
    if full_conversation[answer_character_start : answer_character_start + len(answer_text)] != answer_text:
        raise RuntimeError("rendered answer text is not at the expected chat-template offset")
    offset_label_start_raw = next(
        (index for index, (_, end) in enumerate(offsets) if end > answer_character_start), None
    )
    if offset_label_start_raw is None:
        raise RuntimeError("could not locate the first answer token from character offsets")
    expanded_label_start_from_offsets = offset_label_start_raw + image_expansion_delta
    if expanded_label_start_from_offsets != label_start:
        raise RuntimeError(
            "offset-derived and input_ids-derived answer starts disagree: "
            f"{expanded_label_start_from_offsets} != {label_start}"
        )

    label_end = eos_position + 1
    if len(input_ids) > max_length:
        raise ValueError(
            f"sequence length {len(input_ids)} exceeds max_length={max_length}; refusing to truncate"
        )
    labels = [-100] * label_start + input_ids[label_start:label_end] + [-100] * (len(input_ids) - label_end)
    label_tokens = input_ids[label_start:label_end]
    decoded_labels = tokenizer.decode(label_tokens, skip_special_tokens=False)
    eos_text = tokenizer.decode([eos_token_id], skip_special_tokens=False)
    if not decoded_labels.strip().endswith(eos_text):
        raise RuntimeError(f"labeled span does not end with the explicit EOS text: {decoded_labels!r}")
    answer_decoded = decoded_labels.strip()[: -len(eos_text)].strip()
    if answer_decoded != answer_text:
        raise RuntimeError(f"labeled answer text mismatch: {answer_decoded!r} != {answer_text!r}")
    if parse_box(decoded_labels) is None:
        raise RuntimeError(f"labeled answer is not parseable by the frozen evaluator: {decoded_labels!r}")
    if label_end == len(input_ids):
        if labels[-1] != eos_token_id:
            raise RuntimeError("answer EOS is not the final labeled token")
    elif any(value != -100 for value in labels[label_end:]):
        raise RuntimeError("tokens after the explicit answer EOS must not contribute to the loss")
    if any(value != -100 for value in labels[:label_start]):
        raise RuntimeError("prompt tokens must not contribute to the loss")

    audit = {
        "record_id": record["id"],
        "image": record["image"],
        "original_size_wh": original_size,
        "image_size": image_size,
        "image_seq_len": image_seq_len,
        "answer": answer_audit,
        "representation": representation,
        "representation_prompt_id": variant_prompt_id(representation),
        "representation_prompt_sha256": VARIANT_PROMPT_SHA256[representation],
        "runtime_image_seq_len": runtime_audit,
        "visual_token_contract": contract.as_dict(),
        "generation_prompt_visual_contract": prompt_contract.as_dict(),
        "raw_conversation_placeholder_count": raw_placeholder_count,
        "raw_conversation_token_count": len(raw_ids),
        "raw_prompt_token_count": len(raw_prompt_ids),
        "expanded_prompt_token_count": len(prompt_ids),
        "expanded_conversation_token_count": len(input_ids),
        "image_expansion_delta": image_expansion_delta,
        "offset_label_start_raw": offset_label_start_raw,
        "expanded_label_start_from_offsets": expanded_label_start_from_offsets,
        "prompt_tokens": label_start,
        "answer_tokens": label_end - label_start,
        "masked_tokens": len(input_ids) - (label_end - label_start),
        "sequence_tokens": len(input_ids),
        "tokens_after_eos": len(input_ids) - label_end,
        "eos_token_id": eos_token_id,
        "eos_position": eos_position,
        "prompt_eos_positions": prompt_eos_positions,
        "answer_eos_positions": answer_eos_positions,
        "pad_token_id": pad_token_id,
        "label_rule": (
            "input_ids come from the processor-expanded full conversation; labels=-100 for all "
            "positions before the label_start derived from EOS minus answer tokens; answer+EOS labeled; "
            "tokens after EOS masked"
        ),
        "first_labeled_token_text": tokenizer.decode([input_ids[label_start]], skip_special_tokens=False),
        "labeled_span_decoded": decoded_labels,
        "prefix_extension_verified": True,
        "offset_cross_check_verified": True,
    }
    return SftExample(
        record_id=record["id"],
        image=record["image"],
        question=record["question"],
        answer_text=answer_text,
        answer_gt=list(record["answer"]),
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.ones(len(input_ids), dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        pixel_values=full_processed["pixel_values"][0],
        audit=audit,
    )


def collate_examples(
    examples: Sequence[SftExample],
    *,
    pad_token_id: int = PAD_TOKEN_ID,
) -> dict[str, Any]:
    """Right-pad text tensors; never pad or truncate visual features."""

    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")
    pixel_shapes = {tuple(example.pixel_values.shape) for example in examples}
    if len(pixel_shapes) != 1:
        raise ValueError(
            "refusing to pad/truncate visual feature maps; all views must share one shape, got "
            f"{sorted(pixel_shapes)}"
        )
    max_length = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = example.input_ids
        attention_mask[row, :length] = example.attention_mask
        labels[row, :length] = example.labels
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.stack([example.pixel_values for example in examples], dim=0),
    }


def load_train_records(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the configured training split(s) with strict data validation."""

    records: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for entry in config["data"]["train"]:
        rows, audit = load_split_records(entry)
        records.extend(rows)
        evidence.append(audit)
    if not records:
        raise ValueError("training split selection is empty")
    return records, evidence


class SftDataset:
    """On-demand example builder with an optional small-set example cache."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        processor: Any,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        image_size: int = 384,
        patch_size: int = 16,
        scale_factor: int = 4,
        answer_precision: int = 4,
        answer_format: str = DEFAULT_ANSWER_FORMAT,
        representation: str = "normalized_decimal",
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_max: int = 200,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.answer_precision = answer_precision
        self.answer_format = answer_format
        self.representation = validate_representation(representation)
        self.max_length = max_length
        self._cache: dict[int, SftExample] = {}
        self._precompute = len(self.records) <= cache_max

    def __len__(self) -> int:
        return len(self.records)

    def build_example(self, index: int) -> SftExample:
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        example = build_sft_example(
            self.records[index],
            self.processor,
            model_path=self.model_path,
            image_size=self.image_size,
            patch_size=self.patch_size,
            scale_factor=self.scale_factor,
            answer_precision=self.answer_precision,
            answer_format=self.answer_format,
            representation=self.representation,
            max_length=self.max_length,
        )
        if self._precompute:
            self._cache[index] = example
        return example

    def precompute(self) -> None:
        if self._precompute and len(self._cache) < len(self.records):
            for index in range(len(self.records)):
                self.build_example(index)


def build_probe_inputs(
    record: Mapping[str, Any],
    processor: Any,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    image_size: int = 384,
    patch_size: int = 16,
    scale_factor: int = 4,
    image_token_id: int = IMAGE_TOKEN_ID,
    representation: str = "normalized_decimal",
) -> dict[str, Any]:
    """Build generation-only inputs (prompt without labels) for the post-train probe."""

    import torch

    image_path = resolve_project_path(record["image"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image_seq_len = compute_image_seq_len(image_size, patch_size, scale_factor)
    set_runtime_image_seq_len(
        processor, image_seq_len, processor_config_path=Path(model_path) / "processor_config.json"
    )
    representation = validate_representation(representation)
    prompt = processor.apply_chat_template(
        build_variant_messages(record["question"], representation), add_generation_prompt=True
    )
    processed = processor(text=prompt, images=[image], return_tensors="pt")
    contract = inspect_visual_token_contract(
        processed,
        patch_size=patch_size,
        scale_factor=scale_factor,
        image_size=image_size,
        image_token_id=image_token_id,
    )
    return {
        "prompt": prompt,
        "input_ids": processed["input_ids"],
        "attention_mask": processed["attention_mask"],
        "pixel_values": processed["pixel_values"],
        "prompt_tokens": int(processed["input_ids"].shape[1]),
        "visual_token_contract": contract.as_dict(),
    }
