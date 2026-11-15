#!/usr/bin/env python3
"""Download pinned bevaya/ScreenSpot and convert its test set to unified JSONL.

The source bbox is already normalized xyxy (verified from the pinned Parquet),
so conversion clips to [0,1] and derives pixel xyxy without a second normalize.
ScreenSpot is evaluation-only and is never written to a training JSONL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(_PROJECT_ROOT / "data" / ".cache" / "huggingface" / "hub"))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = _PROJECT_ROOT
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" / "screenspot"
LOCAL_REPO = RAW_ROOT / "bevaya"
REPO_ID = "bevaya/ScreenSpot"
REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
EXPECTED_FILES = {
    "data/test-00000-of-00003.parquet": (134512659, "ff06d312270eecc9d9ac968a51ceb0bc54f80cfda691ccccb8ebf4b4e5faa8fb"),
    "data/test-00001-of-00003.parquet": (198971508, "d48b8275f9dcff56f9c1bfedafbeb018cc200cbb649f8a92299cba0a9130891a"),
    "data/test-00002-of-00003.parquet": (268832649, "a28a1e9f027003730c1375550bfdd3b212d0c0b5c24b338140309a2af3674e5b"),
}
FINAL_KEYS = ("id", "image", "question", "answer", "raw_bbox", "width", "height", "source")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def convert_bbox(bbox: list[float], width: int, height: int) -> tuple[list[float], list[float], bool]:
    """Convert source normalized xyxy to clipped answer and unclipped pixel xyxy."""
    if len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
        raise ValueError("bbox must contain four finite numbers")
    clipped = [clip01(float(v)) for v in bbox]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox is empty after clipping")
    answer = [round(v, 8) for v in clipped]
    raw_bbox = [round(float(bbox[0]) * width, 6), round(float(bbox[1]) * height, 6),
                round(float(bbox[2]) * width, 6), round(float(bbox[3]) * height, 6)]
    return answer, raw_bbox, clipped != list(bbox)


def safe_file_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/")
    base = normalized.rsplit("/", 1)[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip(".")
    if not base or base in (".", ".."):
        raise ValueError(f"unsafe or empty file_name: {file_name!r}")
    return base


def exact_content_key(image_hash: str, instruction: str, bbox: list[float], data_type: str, data_source: str) -> str:
    return json.dumps([image_hash, instruction, bbox, data_type, data_source], ensure_ascii=False, separators=(",", ":"))


def download_shards(cached_only: bool = False) -> list[dict[str, Any]]:
    logs = []
    for name, (expected_bytes, expected_hash) in EXPECTED_FILES.items():
        local_path = LOCAL_REPO / name
        preexisting_valid = local_path.is_file() and local_path.stat().st_size == expected_bytes and sha256_file(local_path) == expected_hash
        log: dict[str, Any] = {
            "file": name, "expected_bytes": expected_bytes, "expected_sha256": expected_hash,
            "status": "cached_validated" if preexisting_valid else "download_required",
            "attempts": 0, "errors": [],
        }
        started = time.monotonic()
        if not preexisting_valid:
            if cached_only:
                raise FileNotFoundError(f"valid pinned shard is not cached: {local_path}")
            for attempt in range(1, 4):
                log["attempts"] = attempt
                try:
                    downloaded = hf_hub_download(
                        repo_id=REPO_ID, repo_type="dataset", filename=name,
                        revision=REVISION, local_dir=str(LOCAL_REPO),
                    )
                    local_path = Path(downloaded)
                    log["status"] = "downloaded_or_resumed"
                    break
                except Exception as exc:
                    log["errors"].append(f"attempt={attempt}: {type(exc).__name__}: {exc}")
            else:
                write_json(RAW_ROOT / "download_log.json", logs + [log])
                raise RuntimeError(f"download failed after retries: {name}: {log['errors']}")
        actual_hash = sha256_file(local_path)
        actual_bytes = local_path.stat().st_size
        if actual_hash != expected_hash or actual_bytes != expected_bytes:
            raise RuntimeError(f"pinned shard hash/size mismatch: {name}: {actual_bytes} {actual_hash}")
        parquet = pq.ParquetFile(local_path)
        log.update({
            "local_path": local_path.relative_to(PROJECT_ROOT).as_posix(),
            "actual_bytes": actual_bytes, "actual_sha256": actual_hash,
            "rows": parquet.metadata.num_rows, "row_groups": parquet.metadata.num_row_groups,
            "arrow_schema": str(parquet.schema_arrow),
            "elapsed_seconds": time.monotonic() - started,
        })
        logs.append(log)
    write_json(RAW_ROOT / "download_log.json", logs)
    return logs


def training_image_hashes() -> set[str]:
    hashes: set[str] = set()
    for dataset, split in (("refcoco", "train"), ("synthetic", "train")):
        manifest_path = DATA_ROOT / "manifests" / f"{dataset}_{split}.jsonl"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                digest = row.get("image_sha256")
                if digest:
                    hashes.add(digest)
                else:
                    hashes.add(sha256_file(PROJECT_ROOT / row["image"]))
    return hashes


def isolation_checks(manifests: list[dict[str, Any]]) -> dict[str, int]:
    train_ids: set[str] = set()
    train_paths: set[str] = set()
    for name in ("refcoco_train.jsonl", "synthetic_train.jsonl"):
        path = DATA_ROOT / name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                train_ids.add(row["id"])
                train_paths.add(row["image"])
    screenspot_hashes = {m["image_sha256"] for m in manifests}
    return {
        "screenspot_train_id_intersection_count": len({m["id"] for m in manifests} & train_ids),
        "screenspot_train_image_path_intersection_count": len({m["image"] for m in manifests} & train_paths),
        "screenspot_train_image_sha256_intersection_count": len(screenspot_hashes & training_image_hashes()),
    }


def convert_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}
    seen_question: dict[tuple[str, str], str] = {}
    name_hashes: dict[str, set[str]] = defaultdict(set)
    path_hashes: dict[str, str] = {}
    source_rows = 0
    sanitized_name_count = 0
    name_collision_different_image_count = 0

    for shard_index, name in enumerate(sorted(EXPECTED_FILES)):
        path = LOCAL_REPO / name
        parquet = pq.ParquetFile(path)
        shard_row_index = 0
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                source_row_id = f"{Path(name).name}:{shard_row_index:06d}"
                shard_row_index += 1
                source_rows += 1
                try:
                    file_name = row["file_name"]
                    instruction = row["instruction"].strip()
                    if not instruction:
                        raise ValueError("empty instruction")
                    data_type = row["data_type"]
                    data_source = row["data_source"]
                    if not data_type or not data_source:
                        raise ValueError("empty data_type/data_source")
                    payload = bytes(row["image"]["bytes"])
                    if not payload:
                        raise ValueError("empty image bytes")
                    with Image.open(io.BytesIO(payload)) as opened:
                        opened.load()
                        width, height = opened.size
                        image_format = opened.format
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid image dimensions")
                    answer, raw_bbox, was_clipped = convert_bbox(list(row["bbox"]), width, height)
                    image_hash = sha256_bytes(payload)
                    content_key = exact_content_key(image_hash, instruction, list(row["bbox"]), data_type, data_source)
                    if content_key in seen_content:
                        duplicates.append({
                            "source_row_id": source_row_id, "kept_id": seen_content[content_key],
                            "file_name": file_name, "instruction": instruction, "bbox": row["bbox"],
                            "image_sha256": image_hash,
                        })
                        continue
                    question_key = (image_hash, instruction)
                    if question_key in seen_question:
                        conflicts.append({
                            "source_row_id": source_row_id, "other_id": seen_question[question_key],
                            "file_name": file_name, "instruction": instruction, "bbox": row["bbox"],
                        })
                    record_id = f"screenspot_test_{len(records) + 1:06d}"
                    seen_content[content_key] = record_id
                    seen_question.setdefault(question_key, record_id)
                    base = safe_file_name(file_name)
                    if base != file_name:
                        sanitized_name_count += 1
                    output_name = base
                    if base in name_hashes and image_hash not in name_hashes[base]:
                        suffix = Path(base).suffix
                        stem = base[:-len(suffix)] if suffix else base
                        output_name = f"{stem}__{image_hash[:12]}{suffix}"
                        name_collision_different_image_count += 1
                    name_hashes[base].add(image_hash)
                    image_rel = f"data/images/screenspot/{output_name}"
                    previous_hash = path_hashes.get(image_rel)
                    if previous_hash is not None and previous_hash != image_hash:
                        raise RuntimeError(f"output path collision remains: {image_rel}")
                    path_hashes[image_rel] = image_hash
                    image_path = PROJECT_ROOT / image_rel
                    if not image_path.exists() or sha256_file(image_path) != image_hash:
                        atomic_write_bytes(image_path, payload)
                    record = {
                        "id": record_id, "image": image_rel, "question": instruction,
                        "answer": answer, "raw_bbox": raw_bbox,
                        "width": width, "height": height, "source": "screenspot",
                    }
                    manifest = {
                        **record,
                        "source_repo": REPO_ID, "source_revision": REVISION,
                        "source_split": "test", "source_shard": name,
                        "source_row_id": source_row_id, "source_file_name": file_name,
                        "source_image_struct_path": row["image"].get("path"),
                        "source_bbox_normalized_xyxy": [float(v) for v in row["bbox"]],
                        "data_type": data_type, "data_source": data_source,
                        "image_sha256": image_hash, "image_bytes": len(payload), "image_format": image_format,
                        "bbox_clipped": was_clipped, "exact_content_key": content_key,
                    }
                    records.append(record)
                    manifests.append(manifest)
                except Exception as exc:
                    invalid.append({
                        "source_row_id": source_row_id, "source_shard": name,
                        "file_name": row.get("file_name"), "instruction": row.get("instruction"),
                        "bbox": row.get("bbox"), "error": f"{type(exc).__name__}: {exc}",
                    })
    audit = {
        "source_rows_read": source_rows,
        "final_records": len(records),
        "invalid_rows_skipped": len(invalid),
        "duplicate_exact_content_records_removed": len(duplicates),
        "conflicting_same_image_instruction_count": len(conflicts),
        "sanitized_file_name_count": sanitized_name_count,
        "same_name_different_image_collision_count": name_collision_different_image_count,
        "unique_images": len({r["image"] for r in records}),
        "unique_image_hashes": len({m["image_sha256"] for m in manifests}),
    }
    write_jsonl(RAW_ROOT / "invalid_rows.jsonl", invalid)
    write_jsonl(RAW_ROOT / "duplicates_removed.jsonl", duplicates)
    write_jsonl(RAW_ROOT / "conflicting_same_image_instruction.jsonl", conflicts)
    return records, manifests, audit


def distribution(records: list[dict[str, Any]], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    areas = [(r["answer"][2] - r["answer"][0]) * (r["answer"][3] - r["answer"][1]) for r in records]
    lengths = [len(r["question"].split()) for r in records]
    resolutions = Counter(f"{r['width']}x{r['height']}" for r in records)
    return {
        "records": len(records),
        "unique_images": len({r["image"] for r in records}),
        "data_source_counts": dict(sorted(Counter(m["data_source"] for m in manifests).items())),
        "data_type_counts": dict(sorted(Counter(m["data_type"] for m in manifests).items())),
        "resolution_counts": dict(sorted(resolutions.items())),
        "image_width": {"min": min(r["width"] for r in records), "max": max(r["width"] for r in records),
                        "mean": statistics.fmean(r["width"] for r in records), "median": statistics.median(r["width"] for r in records)},
        "image_height": {"min": min(r["height"] for r in records), "max": max(r["height"] for r in records),
                         "mean": statistics.fmean(r["height"] for r in records), "median": statistics.median(r["height"] for r in records)},
        "normalized_bbox_area": {"min": min(areas), "max": max(areas), "mean": statistics.fmean(areas), "median": statistics.median(areas)},
        "expression_word_count": {"min": min(lengths), "max": max(lengths), "mean": statistics.fmean(lengths), "median": statistics.median(lengths)},
        "bbox_clipped_record_count": sum(m["bbox_clipped"] for m in manifests),
        "image_format_counts": dict(sorted(Counter(m["image_format"] for m in manifests).items())),
        "image_bytes_total": sum(m["image_bytes"] for m in manifests),
    }


def draw_sample(record: dict[str, Any], destination: Path) -> None:
    with Image.open(PROJECT_ROOT / record["image"]) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = record["answer"]
    box = [x1 * record["width"], y1 * record["height"], x2 * record["width"], y2 * record["height"]]
    draw.rectangle(box, outline=(255, 0, 0), width=max(3, min(record["width"], record["height"]) // 180))
    label = f"{record['id']} | {record['question']}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    label_height = text_box[3] - text_box[1] + 8
    label_y = max(0, box[1] - label_height)
    draw.rectangle([0, label_y, min(record["width"], text_box[2] + 8), label_y + label_height], fill="white")
    draw.text((4, label_y + 4), label, fill=(180, 0, 0), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def select_qa_samples(records: list[dict[str, Any]], manifests: list[dict[str, Any]], seed: int = 42, count: int = 10) -> list[dict[str, Any]]:
    """One deterministic sample per platform first, then fill without replacement."""
    rng = random.Random(seed)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, manifest in enumerate(manifests):
        by_source[manifest["data_source"]].append(index)
    selected: list[int] = []
    for source in sorted(by_source):
        if len(selected) < count:
            selected.append(rng.choice(by_source[source]))
    remaining = [i for i in range(len(records)) if i not in set(selected)]
    if len(selected) < count:
        selected.extend(rng.sample(remaining, count - len(selected)))
    return [records[i] for i in selected[:count]]


def generate_qa(records: list[dict[str, Any]], manifests: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    chosen = select_qa_samples(records, manifests, seed, 10)
    samples = []
    for record in chosen:
        qa_path = DATA_ROOT / "qa" / "screenspot" / "samples" / f"{record['id']}.png"
        draw_sample(record, qa_path)
        samples.append({**record, "qa_image": qa_path.relative_to(PROJECT_ROOT).as_posix()})
    sheet = Image.new("RGB", (960, 2000), "white")
    draw = ImageDraw.Draw(sheet)
    for index, sample in enumerate(samples):
        with Image.open(PROJECT_ROOT / sample["qa_image"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((480, 366))
        x, y = (index % 2) * 480, (index // 2) * 400
        draw.text((x + 5, y + 8), f"{index + 1}: {sample['id']}", fill="black", font=ImageFont.load_default())
        sheet.paste(image, (x, y + 34))
    sheet_path = DATA_ROOT / "qa" / "screenspot" / "contact_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    manifest = {
        "generated_at": utc_now(), "seed": seed,
        "sampling": "one random sample per data_source, then random fill without replacement",
        "sample_count": len(samples), "samples": samples, "visual_review": "pending",
    }
    write_json(DATA_ROOT / "qa" / "screenspot" / "sample_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.monotonic()
    downloads = download_shards(args.cached_only)
    records, manifests, audit = convert_all()
    expected_source_rows = sum(item["rows"] for item in downloads)
    if audit["source_rows_read"] != expected_source_rows:
        raise RuntimeError(f"source row count mismatch: {audit['source_rows_read']} != {expected_source_rows}")
    if not records:
        raise RuntimeError("no valid ScreenSpot records")
    write_jsonl(DATA_ROOT / "screenspot.jsonl", records)
    write_jsonl(DATA_ROOT / "manifests" / "screenspot.jsonl", manifests)
    isolation = isolation_checks(manifests)
    if any(isolation.values()):
        raise RuntimeError(f"ScreenSpot/training isolation failed: {isolation}")
    quality = {
        "duplicate_id_count": len(records) - len({r["id"] for r in records}),
        "duplicate_exact_content_count_after_cleaning": len(records) - len({m["exact_content_key"] for m in manifests}),
        "invalid_answer_count": sum(not (0 <= r["answer"][0] < r["answer"][2] <= 1 and 0 <= r["answer"][1] < r["answer"][3] <= 1) for r in records),
        "missing_image_count": sum(not (PROJECT_ROOT / r["image"]).is_file() for r in records),
        **isolation,
    }
    if any(quality.values()):
        raise RuntimeError(f"quality checks failed: {quality}")
    qa = generate_qa(records, manifests, args.seed)
    stats = {
        "dataset": "screenspot", "generated_at": utc_now(), "elapsed_seconds": time.monotonic() - started,
        "source": {
            "repo_id": REPO_ID, "revision": REVISION, "split": "test_only",
            "repo_url": "https://huggingface.co/datasets/bevaya/ScreenSpot",
            "mirror_url": "https://hf-mirror.com/datasets/bevaya/ScreenSpot",
            "bbox_semantics": "source bbox is normalized [x1,y1,x2,y2]; verified on pinned Parquet and probe images",
            "files": downloads,
        },
        "config": {"seed": args.seed, "role": "transfer_evaluation_only", "raw_bbox": "unclipped source normalized xyxy times actual image dimensions, rounded to 6 decimals"},
        "processing_audit": audit,
        "distribution": distribution(records, manifests),
        "quality_checks": quality,
        "outputs": {"screenspot.jsonl": {"bytes": (DATA_ROOT / "screenspot.jsonl").stat().st_size, "sha256": sha256_file(DATA_ROOT / "screenspot.jsonl")}},
        "qa": {
            "sample_count": len(qa["samples"]),
            "sample_manifest": "data/qa/screenspot/sample_manifest.json",
            "contact_sheet": "data/qa/screenspot/contact_sheet.png",
            "visual_review": "pending",
        },
    }
    write_json(DATA_ROOT / "screenspot_stats.json", stats)
    shared_path = DATA_ROOT / "stats.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    shared["screenspot"] = stats
    write_json(shared_path, shared)
    source_record = {
        "repo_id": REPO_ID, "revision": REVISION, "readme": "data/raw/screenspot/bevaya/README.md",
        "source_metadata": "data/raw/screenspot/source_metadata.json", "probe_result": "data/raw/screenspot/probe_result.json",
        "files": downloads, "processing_audit": audit,
    }
    write_json(RAW_ROOT / "source.json", source_record)
    print(json.dumps({"status": "complete_pending_visual_review", **audit, "quality_checks": quality,
                      "output": stats["outputs"], "elapsed_seconds": stats["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
