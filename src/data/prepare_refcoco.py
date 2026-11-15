#!/usr/bin/env python3
"""Prepare an image-disjoint RefCOCO subset with on-demand COCO downloads.

Only writes below final-project/data/. No GPU libraries are imported.
The source parquet files contain annotations only; images are fetched one by one.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" / "refcoco"
SOURCE_ROOT = RAW_ROOT / "jxu124"
IMAGE_ROOT = DATA_ROOT / "images" / "refcoco"
MANIFEST_ROOT = DATA_ROOT / "manifests"
QA_ROOT = DATA_ROOT / "qa" / "refcoco"

REPO_ID = "jxu124/refcoco"
REVISION = "9fba8200c5326e996f789191f095bd464ef1d09e"
SOURCE_FILES = {
    "train": "data/train-00000-of-00001-94431d5f4bd5b93f.parquet",
    "val": "data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
}
COCO_BASE_HTTP = "http://images.cocodataset.org"
COCO_BASE_HTTPS = "https://images.cocodataset.org"
USER_AGENT = "final-project-refcoco-preparer/1.0"
SELECTION_VERSION = "refcoco-image-disjoint-v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, split: str, *parts: Any) -> str:
    text = "|".join([SELECTION_VERSION, str(seed), split, *(str(p) for p in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def ensure_environment() -> None:
    # Keep every cache written by this task inside its permitted data/ scope.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = str(RAW_ROOT / "hf_cache")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for path in (RAW_ROOT, SOURCE_ROOT, IMAGE_ROOT, MANIFEST_ROOT, QA_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def ensure_annotation(split: str) -> Path:
    ensure_environment()
    from huggingface_hub import hf_hub_download

    filename = SOURCE_FILES[split]
    result = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        revision=REVISION,
        local_dir=str(SOURCE_ROOT),
    )
    return Path(result)


def load_rows(path: Path) -> list[dict[str, Any]]:
    # Annotation-only tables are small enough to load (about 81 MB decoded train).
    return pq.read_table(path).to_pylist()


def coco_url(file_name: str, base: str = COCO_BASE_HTTP) -> str:
    if Path(file_name).name != file_name:
        raise ValueError(f"COCO file name must not contain a directory: {file_name}")
    if file_name.startswith("COCO_train2014_"):
        folder = "train2014"
    elif file_name.startswith("COCO_val2014_"):
        folder = "val2014"
    else:
        raise ValueError(f"Unrecognized COCO file name: {file_name}")
    return f"{base}/{folder}/{file_name}"


def inspect_image(path: Path) -> tuple[tuple[int, int], str | None]:
    with Image.open(path) as image:
        image.load()
        return image.size, image.format


def download_one_image(
    file_name: str,
    expected_size: tuple[int, int],
    timeout: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    destination = IMAGE_ROOT / file_name
    relative = destination.relative_to(PROJECT_ROOT).as_posix()
    primary_url = coco_url(file_name)
    if destination.exists():
        try:
            actual_size, image_format = inspect_image(destination)
            if actual_size != expected_size:
                raise ValueError(f"cached dimensions {actual_size} != annotation {expected_size}")
            return {
                "file_name": file_name,
                "image": relative,
                "url": primary_url,
                "status": "cached",
                "bytes_downloaded": 0,
                "file_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "width": actual_size[0],
                "height": actual_size[1],
                "format": image_format,
                "attempt_count": 0,
                "errors": [],
                "timestamp": utc_now(),
            }
        except Exception as exc:
            cached_error = f"invalid cached image: {type(exc).__name__}: {exc}"
        else:
            cached_error = ""
    else:
        cached_error = ""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    errors: list[str] = []
    if cached_error:
        errors.append(cached_error)
    for attempt in range(1, retries + 1):
        base = COCO_BASE_HTTP if attempt % 2 else COCO_BASE_HTTPS
        url = coco_url(file_name, base)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
                status = getattr(response, "status", None)
            if status != 200:
                raise ValueError(f"HTTP status {status}")
            if not content:
                raise ValueError("empty response body")
            with partial.open("wb") as handle:
                handle.write(content)
            actual_size, image_format = inspect_image(partial)
            if actual_size != expected_size:
                raise ValueError(f"downloaded dimensions {actual_size} != annotation {expected_size}")
            os.replace(partial, destination)
            return {
                "file_name": file_name,
                "image": relative,
                "url": url,
                "status": "downloaded",
                "bytes_downloaded": len(content),
                "file_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "width": actual_size[0],
                "height": actual_size[1],
                "format": image_format,
                "attempt_count": attempt,
                "errors": errors,
                "timestamp": utc_now(),
            }
        except Exception as exc:
            errors.append(
                f"attempt={attempt} url={url} error={type(exc).__name__}: {exc}"
            )
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                time.sleep(0.5 * attempt)
    return {
        "file_name": file_name,
        "image": relative,
        "url": primary_url,
        "status": "failed",
        "bytes_downloaded": 0,
        "file_bytes": 0,
        "attempt_count": retries,
        "errors": errors,
        "timestamp": utc_now(),
    }


def parse_candidate(
    row: dict[str, Any],
    output_split: str,
    seed: int,
    invalid: Counter,
) -> dict[str, Any] | None:
    try:
        ann = json.loads(row["raw_anns"])
        image_info = json.loads(row["raw_image_info"])
        sentences = row.get("sentences") or []
        valid_sentences = []
        for sentence in sentences:
            question = str(sentence.get("raw") or sentence.get("sent") or "").strip()
            sent_id = sentence.get("sent_id")
            if question and sent_id is not None:
                valid_sentences.append((question, int(sent_id), sentence.get("sent")))
        if not valid_sentences:
            invalid["empty_sentences"] += 1
            return None
        if int(ann.get("iscrowd", 0)) != 0:
            invalid["crowd_annotation"] += 1
            return None
        width = int(image_info["width"])
        height = int(image_info["height"])
        if width <= 0 or height <= 0:
            invalid["invalid_image_dimensions"] += 1
            return None
        file_name = str(image_info["file_name"])
        coco_url(file_name)  # Validate the real COCO file name and folder.
        xywh = [float(v) for v in ann["bbox"]]
        if len(xywh) != 4 or not all(math.isfinite(v) for v in xywh):
            invalid["invalid_raw_bbox"] += 1
            return None
        x, y, box_width, box_height = xywh
        raw_xyxy = [x, y, x + box_width, y + box_height]
        clipped = [
            max(0.0, min(float(width), raw_xyxy[0])),
            max(0.0, min(float(height), raw_xyxy[1])),
            max(0.0, min(float(width), raw_xyxy[2])),
            max(0.0, min(float(height), raw_xyxy[3])),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            invalid["nonpositive_clipped_bbox"] += 1
            return None
        answer = [
            clipped[0] / width,
            clipped[1] / height,
            clipped[2] / width,
            clipped[3] / height,
        ]
        answer = [round(v, 8) for v in answer]
        if (
            not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in answer)
            or answer[2] <= answer[0]
            or answer[3] <= answer[1]
        ):
            invalid["invalid_normalized_answer"] += 1
            return None
        top_bbox = [float(v) for v in (row.get("bbox") or [])]
        bbox_diff = None
        if len(top_bbox) == 4:
            bbox_diff = max(abs(a - b) for a, b in zip(top_bbox, clipped))
        ref_id = int(row["ref_id"])
        ann_id = int(row["ann_id"])
        image_id = int(row["image_id"])
        question, sent_id, normalized_sentence = min(
            valid_sentences,
            key=lambda item: stable_key(seed, output_split, ref_id, item[1], "sentence"),
        )
        record_id = f"refcoco_{output_split}_r{ref_id:06d}_s{sent_id:06d}"
        selection_key = stable_key(seed, output_split, ref_id, ann_id, image_id)
        was_clipped = any(abs(a - b) > 1e-7 for a, b in zip(raw_xyxy, clipped))
        return {
            "id": record_id,
            "image": f"data/images/refcoco/{file_name}",
            "question": question,
            "answer": answer,
            "raw_bbox": [round(v, 5) for v in raw_xyxy],
            "width": width,
            "height": height,
            "source": "refcoco",
            "_meta": {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "source_file": SOURCE_FILES[output_split],
                "source_row_split": row.get("split"),
                "ref_id": ref_id,
                "ann_id": ann_id,
                "image_id": image_id,
                "sent_id": sent_id,
                "normalized_sentence": normalized_sentence,
                "raw_xywh": xywh,
                "clipped_xyxy": [round(v, 5) for v in clipped],
                "bbox_was_clipped": was_clipped,
                "top_level_bbox_xyxy": top_bbox,
                "top_level_bbox_max_abs_diff_to_clipped": bbox_diff,
                "selection_key": selection_key,
                "download_url": coco_url(file_name),
            },
        }
    except Exception as exc:
        invalid[f"parse_exception_{type(exc).__name__}"] += 1
        return None


def build_candidates(
    rows: list[dict[str, Any]], output_split: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int], set[int]]:
    invalid: Counter = Counter()
    candidates: list[dict[str, Any]] = []
    all_image_ids: set[int] = set()
    for row in rows:
        try:
            image_info = json.loads(row["raw_image_info"])
            all_image_ids.add(int(image_info["id"]))
        except Exception:
            try:
                all_image_ids.add(int(row["image_id"]))
            except Exception:
                pass
        candidate = parse_candidate(row, output_split, seed, invalid)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: (c["_meta"]["selection_key"], c["id"]))
    return candidates, dict(sorted(invalid.items())), all_image_ids


def manifest_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {**{k: v for k, v in candidate.items() if k != "_meta"}, **candidate["_meta"]}


def final_row(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "image", "question", "answer", "raw_bbox", "width", "height", "source")
    return {key: candidate[key] for key in keys}


def select_candidates(
    candidates: list[dict[str, Any]], size: int, failed_images: set[str]
) -> list[dict[str, Any]]:
    available = [
        candidate
        for candidate in candidates
        if Path(candidate["image"]).name not in failed_images
    ]
    if len(available) < size:
        raise RuntimeError(
            f"Insufficient candidates after failed-image exclusion: need {size}, have {len(available)}"
        )
    return available[:size]


def requirements_for(candidates: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    requirements: dict[str, tuple[int, int]] = {}
    for candidate in candidates:
        file_name = Path(candidate["image"]).name
        size = (int(candidate["width"]), int(candidate["height"]))
        previous = requirements.setdefault(file_name, size)
        if previous != size:
            raise ValueError(f"Conflicting dimensions for {file_name}: {previous} vs {size}")
    return requirements


def download_selected(
    train: list[dict[str, Any]], val: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    requirements = requirements_for([*train, *val])
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(download_one_image, name, size): name
            for name, size in sorted(requirements.items())
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_name):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == len(requirements):
                failures = sum(r["status"] == "failed" for r in results)
                print(
                    f"images {completed}/{len(requirements)}; failures_so_far={failures}",
                    flush=True,
                )
    return sorted(results, key=lambda r: r["file_name"])


def validate_final(
    train: list[dict[str, Any]], val: list[dict[str, Any]]
) -> dict[str, Any]:
    train_ids = [c["id"] for c in train]
    val_ids = [c["id"] for c in val]
    train_image_ids = {c["_meta"]["image_id"] for c in train}
    val_image_ids = {c["_meta"]["image_id"] for c in val}
    train_sent_ids = {c["_meta"]["sent_id"] for c in train}
    val_sent_ids = {c["_meta"]["sent_id"] for c in val}
    train_ref_ids = {c["_meta"]["ref_id"] for c in train}
    val_ref_ids = {c["_meta"]["ref_id"] for c in val}
    train_paths = {c["image"] for c in train}
    val_paths = {c["image"] for c in val}
    train_expression_keys = [
        (c["_meta"]["image_id"], c["_meta"]["ann_id"], c["question"].strip().casefold())
        for c in train
    ]
    val_expression_keys = [
        (c["_meta"]["image_id"], c["_meta"]["ann_id"], c["question"].strip().casefold())
        for c in val
    ]
    train_source_keys = [
        (c["_meta"]["ref_id"], c["_meta"]["sent_id"], c["_meta"]["ann_id"])
        for c in train
    ]
    val_source_keys = [
        (c["_meta"]["ref_id"], c["_meta"]["sent_id"], c["_meta"]["ann_id"])
        for c in val
    ]
    quality = {
        "train_duplicate_id_count": len(train_ids) - len(set(train_ids)),
        "val_duplicate_id_count": len(val_ids) - len(set(val_ids)),
        "train_duplicate_source_record_count": len(train_source_keys) - len(set(train_source_keys)),
        "val_duplicate_source_record_count": len(val_source_keys) - len(set(val_source_keys)),
        "train_duplicate_image_ann_question_count": len(train_expression_keys) - len(set(train_expression_keys)),
        "val_duplicate_image_ann_question_count": len(val_expression_keys) - len(set(val_expression_keys)),
        "train_val_id_intersection_count": len(set(train_ids) & set(val_ids)),
        "train_val_sent_id_intersection_count": len(train_sent_ids & val_sent_ids),
        "train_val_ref_id_intersection_count": len(train_ref_ids & val_ref_ids),
        "train_val_image_id_intersection_count": len(train_image_ids & val_image_ids),
        "train_val_image_path_intersection_count": len(train_paths & val_paths),
        "train_val_exact_image_ann_question_intersection_count": len(
            set(train_expression_keys) & set(val_expression_keys)
        ),
        "train_missing_image_count": 0,
        "val_missing_image_count": 0,
        "train_image_dimension_mismatch_count": 0,
        "val_image_dimension_mismatch_count": 0,
        "train_invalid_answer_count": 0,
        "val_invalid_answer_count": 0,
    }
    for split_name, candidates in (("train", train), ("val", val)):
        for candidate in candidates:
            path = PROJECT_ROOT / candidate["image"]
            if not path.exists():
                quality[f"{split_name}_missing_image_count"] += 1
                continue
            actual_size, _ = inspect_image(path)
            if actual_size != (candidate["width"], candidate["height"]):
                quality[f"{split_name}_image_dimension_mismatch_count"] += 1
            answer = candidate["answer"]
            if (
                len(answer) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in answer)
                or answer[2] <= answer[0]
                or answer[3] <= answer[1]
            ):
                quality[f"{split_name}_invalid_answer_count"] += 1
    if any(quality.values()):
        raise RuntimeError(f"Final data quality validation failed: {quality}")
    return quality


def descriptive_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    expression_words = [len(c["question"].split()) for c in candidates]
    expression_chars = [len(c["question"]) for c in candidates]
    areas = [
        max(0.0, c["answer"][2] - c["answer"][0])
        * max(0.0, c["answer"][3] - c["answer"][1])
        for c in candidates
    ]
    widths = [c["answer"][2] - c["answer"][0] for c in candidates]
    heights = [c["answer"][3] - c["answer"][1] for c in candidates]
    edges = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.00000001]
    histogram = [0] * (len(edges) - 1)
    for area in areas:
        for index in range(len(histogram)):
            if edges[index] <= area < edges[index + 1]:
                histogram[index] += 1
                break
    summary = lambda values: {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    return {
        "records": len(candidates),
        "unique_images": len({c["image"] for c in candidates}),
        "unique_ref_ids": len({c["_meta"]["ref_id"] for c in candidates}),
        "unique_sent_ids": len({c["_meta"]["sent_id"] for c in candidates}),
        "unique_ann_ids": len({c["_meta"]["ann_id"] for c in candidates}),
        "expression_word_count": summary(expression_words),
        "expression_char_count": summary(expression_chars),
        "duplicate_question_text_records": len(candidates)
        - len({c["question"].strip().casefold() for c in candidates}),
        "normalized_bbox_area": summary(areas),
        "normalized_bbox_width": summary(widths),
        "normalized_bbox_height": summary(heights),
        "normalized_bbox_area_histogram": {
            "edges": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
            "counts": histogram,
        },
        "bbox_clipped_record_count": sum(c["_meta"]["bbox_was_clipped"] for c in candidates),
        "top_level_bbox_mismatch_over_0_05px_count": sum(
            c["_meta"]["top_level_bbox_max_abs_diff_to_clipped"] is not None
            and c["_meta"]["top_level_bbox_max_abs_diff_to_clipped"] > 0.05
            for c in candidates
        ),
    }


def draw_sample(candidate: dict[str, Any], destination: Path) -> None:
    source = PROJECT_ROOT / candidate["image"]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x1, y1, x2, y2 = candidate["answer"]
    box = [x1 * width, y1 * height, x2 * width, y2 * height]
    line_width = max(2, round(min(width, height) / 150))
    draw.rectangle(box, outline=(255, 40, 40), width=line_width)
    label = f"{candidate['id']} | {candidate['question']}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_y = max(0, box[1] - text_height - 8)
    draw.rectangle(
        [max(0, box[0]), label_y, min(width, box[0] + text_width + 8), label_y + text_height + 8],
        fill=(255, 255, 255),
    )
    draw.text((max(0, box[0]) + 4, label_y + 4), label, fill=(180, 0, 0), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def make_contact_sheet(sample_rows: list[dict[str, Any]], destination: Path) -> None:
    cell_width, cell_height = 480, 360
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 5), "white")
    for index, item in enumerate(sample_rows):
        with Image.open(PROJECT_ROOT / item["qa_image"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((cell_width, cell_height - 32))
        x = (index % 2) * cell_width
        y = (index // 2) * cell_height
        sheet.paste(image, (x, y + 32))
        draw = ImageDraw.Draw(sheet)
        label = f"{item['split']} {index + 1}: {item['id']}"
        draw.text((x + 6, y + 8), label, fill="black", font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def generate_qa(train: list[dict[str, Any]], val: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for split_name, candidates in (("train", train), ("val", val)):
        chosen = random.Random(seed + (0 if split_name == "train" else 1)).sample(candidates, 10)
        split_samples = []
        for candidate in chosen:
            relative = Path("data") / "qa" / "refcoco" / split_name / f"{candidate['id']}.png"
            draw_sample(candidate, PROJECT_ROOT / relative)
            split_samples.append(
                {
                    "split": split_name,
                    "id": candidate["id"],
                    "image": candidate["image"],
                    "question": candidate["question"],
                    "answer": candidate["answer"],
                    "raw_bbox": candidate["raw_bbox"],
                    "qa_image": relative.as_posix(),
                }
            )
        make_contact_sheet(split_samples, QA_ROOT / f"{split_name}_contact_sheet.png")
        samples.extend(split_samples)
    manifest = {
        "generated_at": utc_now(),
        "sampling": "random.Random(seed + 0 for train, seed + 1 for val).sample(..., 10)",
        "seed": seed,
        "samples": samples,
        "visual_review": "pending; generated overlays are not self-certified as visually correct",
    }
    write_json(QA_ROOT / "sample_manifest.json", manifest)
    return manifest


def run_probe(seed: int) -> dict[str, Any]:
    ensure_environment()
    started = time.monotonic()
    result: dict[str, Any] = {
        "started_at": utc_now(),
        "status": "started",
        "source": {"repo_id": REPO_ID, "revision": REVISION, "file": SOURCE_FILES["val"]},
    }
    try:
        annotation = ensure_annotation("val")
        parquet = pq.ParquetFile(annotation)
        row = load_rows(annotation)[0]
        invalid: Counter = Counter()
        candidate = parse_candidate(row, "val", seed, invalid)
        if candidate is None:
            raise RuntimeError(f"First validation row could not be parsed: {dict(invalid)}")
        probe_dir = RAW_ROOT / "probe"
        probe_image = probe_dir / Path(candidate["image"]).name
        url = candidate["_meta"]["download_url"]
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            status = getattr(response, "status", None)
        if status != 200 or not content:
            raise RuntimeError(f"Probe image download failed: HTTP {status}, {len(content)} bytes")
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_image.write_bytes(content)
        actual_size, image_format = inspect_image(probe_image)
        expected_size = (candidate["width"], candidate["height"])
        if actual_size != expected_size:
            raise RuntimeError(f"Probe image dimensions {actual_size} != annotation {expected_size}")
        probe_candidate = dict(candidate)
        probe_candidate["image"] = probe_image.relative_to(PROJECT_ROOT).as_posix()
        overlay = probe_dir / "probe_overlay.png"
        draw_sample(probe_candidate, overlay)
        result.update(
            {
                "status": "passed",
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "annotation": {
                    "path": annotation.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": annotation.stat().st_size,
                    "sha256": sha256_file(annotation),
                    "num_rows": parquet.metadata.num_rows,
                    "num_row_groups": parquet.metadata.num_row_groups,
                    "schema_fields": list(parquet.schema_arrow.names),
                },
                "image_download": {
                    "url": url,
                    "http_status": status,
                    "path": probe_image.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": len(content),
                    "sha256": sha256_file(probe_image),
                    "width": actual_size[0],
                    "height": actual_size[1],
                    "format": image_format,
                },
                "sample": final_row(candidate),
                "source_metadata": candidate["_meta"],
                "overlay": overlay.relative_to(PROJECT_ROOT).as_posix(),
                "notes": [
                    "raw_anns.bbox is xywh; final raw_bbox is derived pixel xyxy before clipping",
                    "answer is clipped normalized xyxy",
                    "the COCO train2014/val2014 folder is derived from raw_image_info.file_name, not the RefCOCO split",
                ],
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(RAW_ROOT / "probe_result.json", result)
        raise
    write_json(RAW_ROOT / "probe_result.json", result)
    return result


def run_prepare(train_size: int, val_size: int, seed: int, workers: int) -> dict[str, Any]:
    ensure_environment()
    started = time.monotonic()
    annotations = {split: ensure_annotation(split) for split in ("train", "val")}
    rows = {split: load_rows(path) for split, path in annotations.items()}
    candidates: dict[str, list[dict[str, Any]]] = {}
    invalid_stats: dict[str, dict[str, int]] = {}
    all_image_ids: dict[str, set[int]] = {}
    for split in ("train", "val"):
        candidates[split], invalid_stats[split], all_image_ids[split] = build_candidates(
            rows[split], split, seed
        )
        print(
            f"{split}: source_rows={len(rows[split])}, valid_one_expression_per_ref_candidates={len(candidates[split])}",
            flush=True,
        )
        del rows[split]  # Release decoded raw annotations before image downloads.

    # Stronger than merely checking record IDs: no training image may occur anywhere
    # in the official RefCOCO validation annotations used by this source.
    before = len(candidates["train"])
    candidates["train"] = [
        c for c in candidates["train"] if c["_meta"]["image_id"] not in all_image_ids["val"]
    ]
    excluded_train = before - len(candidates["train"])
    print(
        f"train candidates excluded because their image occurs in official val: {excluded_train}; remaining={len(candidates['train'])}",
        flush=True,
    )
    if len(candidates["train"]) < train_size or len(candidates["val"]) < val_size:
        raise RuntimeError(
            f"Insufficient candidates: train={len(candidates['train'])}/{train_size}, val={len(candidates['val'])}/{val_size}"
        )

    failed_images: set[str] = set()
    all_download_results: list[dict[str, Any]] = []
    selection_rounds = 0
    selected: dict[str, list[dict[str, Any]]] = {}
    for round_index in range(1, 4):
        selection_rounds = round_index
        print(f"selection/download round {round_index}; excluded_failed_images={len(failed_images)}", flush=True)
        selected = {
            "train": select_candidates(candidates["train"], train_size, failed_images),
            "val": select_candidates(candidates["val"], val_size, failed_images),
        }
        write_jsonl(MANIFEST_ROOT / "refcoco_train.jsonl", [manifest_row(c) for c in selected["train"]])
        write_jsonl(MANIFEST_ROOT / "refcoco_val.jsonl", [manifest_row(c) for c in selected["val"]])
        results = download_selected(selected["train"], selected["val"], workers)
        all_download_results.extend(results)
        write_jsonl(
            RAW_ROOT / "download_failures.jsonl",
            [r for r in all_download_results if r["status"] == "failed"],
        )
        new_failures = {r["file_name"] for r in results if r["status"] == "failed"}
        if not new_failures:
            failed_images = set()
            break
        failed_images.update(new_failures)
    else:
        raise RuntimeError(
            f"Image downloads still failing after {selection_rounds} selection rounds: {sorted(failed_images)}"
        )

    quality = validate_final(selected["train"], selected["val"])
    write_jsonl(DATA_ROOT / "refcoco_train.jsonl", [final_row(c) for c in selected["train"]])
    write_jsonl(DATA_ROOT / "refcoco_val.jsonl", [final_row(c) for c in selected["val"]])
    write_jsonl(RAW_ROOT / "download_log.jsonl", all_download_results)
    qa = generate_qa(selected["train"], selected["val"], seed)

    selected_image_names = {
        Path(c["image"]).name for split_candidates in selected.values() for c in split_candidates
    }
    final_download_results = [
        r for r in all_download_results
        if r["status"] != "failed" and r["file_name"] in selected_image_names
    ]
    unique_success = {r["file_name"]: r for r in final_download_results}
    source_info = {
        "repo_id": REPO_ID,
        "repo_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "mirror_url": f"https://hf-mirror.com/datasets/{REPO_ID}",
        "revision": REVISION,
        "files": {
            split: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_file": SOURCE_FILES[split],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for split, path in annotations.items()
        },
        "image_url_pattern": "http://images.cocodataset.org/{train2014|val2014}/{raw_image_info.file_name}",
    }
    stats = {
        "dataset": "refcoco",
        "generated_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "source": source_info,
        "config": {
            "train_size": train_size,
            "val_size": val_size,
            "seed": seed,
            "workers": workers,
            "selection_algorithm": SELECTION_VERSION,
            "one_expression_per_ref": True,
            "train_excludes_all_official_val_image_ids": True,
            "selection_download_rounds": selection_rounds,
        },
        "candidate_processing": {
            "train_source_rows": source_info["files"]["train"]["rows"],
            "val_source_rows": source_info["files"]["val"]["rows"],
            "train_valid_candidates_before_val_image_exclusion": before,
            "train_candidates_excluded_by_official_val_images": excluded_train,
            "train_valid_candidates_after_val_image_exclusion": len(candidates["train"]),
            "val_valid_candidates": len(candidates["val"]),
            "train_invalid_or_skipped": invalid_stats["train"],
            "val_invalid_or_skipped": invalid_stats["val"],
        },
        "splits": {
            "train": descriptive_stats(selected["train"]),
            "val": descriptive_stats(selected["val"]),
        },
        "quality_checks": quality,
        "downloads": {
            "final_unique_images": len(unique_success),
            "network_download_result_count": sum(
                r["status"] == "downloaded" for r in all_download_results
            ),
            "cached_result_count": sum(r["status"] == "cached" for r in all_download_results),
            "network_bytes_downloaded": sum(r["bytes_downloaded"] for r in all_download_results),
            "failed_image_result_count": sum(r["status"] == "failed" for r in all_download_results),
            "download_log": "data/raw/refcoco/download_log.jsonl",
            "download_failures": "data/raw/refcoco/download_failures.jsonl",
        },
        "outputs": {},
        "qa": {
            "samples_per_split": 10,
            "sample_manifest": "data/qa/refcoco/sample_manifest.json",
            "train_contact_sheet": "data/qa/refcoco/train_contact_sheet.png",
            "val_contact_sheet": "data/qa/refcoco/val_contact_sheet.png",
            "sample_count": len(qa["samples"]),
            "visual_review": "pending",
        },
    }
    for name in ("refcoco_train.jsonl", "refcoco_val.jsonl"):
        path = DATA_ROOT / name
        stats["outputs"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(DATA_ROOT / "stats.json", stats)
    write_json(RAW_ROOT / "source.json", source_info)
    print(json.dumps({
        "status": "complete_pending_visual_review",
        "train_records": len(selected["train"]),
        "val_records": len(selected["val"]),
        "unique_images": len(unique_success),
        "network_bytes_downloaded": stats["downloads"]["network_bytes_downloaded"],
        "failed_image_result_count": stats["downloads"]["failed_image_result_count"],
        "quality_checks": quality,
        "elapsed_seconds": stats["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true", help="Validate one annotation and one COCO image")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.train_size <= 0 or args.val_size <= 0:
        parser.error("subset sizes must be positive")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if args.probe_only:
        result = run_probe(args.seed)
        print(json.dumps({
            "status": result["status"],
            "annotation_rows": result["annotation"]["num_rows"],
            "image_bytes": result["image_download"]["bytes"],
            "image_dimensions": [result["image_download"]["width"], result["image_download"]["height"]],
            "sample_id": result["sample"]["id"],
            "answer": result["sample"]["answer"],
        }, ensure_ascii=False, indent=2), flush=True)
    else:
        run_prepare(args.train_size, args.val_size, args.seed, args.workers)


if __name__ == "__main__":
    main()
