#!/usr/bin/env python3
"""Generate deterministic 5k/1k synthetic referring-expression grounding data.

Derived from hw6_rsi_localize.py's draw_scene/render/make_dataset idea, extended
with more shapes/colors, multiple resolutions, filled-object occlusion, exact
visibility checks, unique referring expressions and unified JSONL. CPU/PIL only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
GENERATOR_VERSION = "synthetic-grounding-v2"
COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (40, 170, 70),
    "yellow": (230, 190, 40),
    "purple": (150, 70, 180),
    "orange": (240, 135, 35),
}
SHAPES = ("circle", "square", "triangle", "diamond", "pentagon")
COMBOS = list(itertools.product(COLORS, SHAPES))
RESOLUTIONS = (
    (256, 256), (320, 320), (384, 384), (448, 448), (512, 512),
    (320, 256), (384, 256), (448, 320), (512, 384),
    (256, 320), (320, 384), (384, 512),
)
BACKGROUNDS = ((255, 255, 255), (250, 250, 250), (245, 248, 250), (250, 247, 242))
FINAL_KEYS = ("id", "image", "question", "answer", "raw_bbox", "width", "height", "source")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def split_seed(seed: int, split: str) -> int:
    return seed if split == "train" else seed + 1


def attempt_seed(seed: int, split: str, index: int, attempt: int) -> int:
    text = f"{GENERATOR_VERSION}|{split_seed(seed, split)}|{split}|{index}|{attempt}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def random_box(rng: random.Random, width: int, height: int) -> list[int]:
    minimum = min(width, height)
    lower = max(28, minimum // 8)
    upper = max(lower + 8, minimum // 3)
    size = rng.randint(lower, upper)
    margin = max(6, minimum // 32)
    x = rng.randint(margin, width - size - margin)
    y = rng.randint(margin, height - size - margin)
    return [x, y, x + size, y + size]


def overlapping_box(rng: random.Random, target: dict[str, Any], width: int, height: int) -> list[int]:
    target_box = target["box"]
    target_size = target_box[2] - target_box[0]
    minimum = min(width, height)
    size = rng.randint(max(28, minimum // 9), max(40, minimum // 3))
    dx = rnd_sign(rng) * rnd_uniform(rng, 0.15, 0.58) * target_size
    dy = rnd_sign(rng) * rnd_uniform(rng, 0.0, 0.42) * target_size
    x = round(target_box[0] + dx)
    y = round(target_box[1] + dy)
    margin = max(4, minimum // 64)
    x = max(margin, min(width - size - margin, x))
    y = max(margin, min(height - size - margin, y))
    return [x, y, x + size, y + size]


def rnd_sign(rng: random.Random) -> int:
    return -1 if rng.random() < 0.5 else 1


def rnd_uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def polygon_points(box: list[int], shape: str) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    x2 -= 1
    y2 -= 1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if shape == "triangle":
        return [(cx, y1), (x2, y2), (x1, y2)]
    if shape == "diamond":
        return [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
    if shape == "pentagon":
        rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
        return [
            (cx + rx * math.cos(-math.pi / 2 + i * 2 * math.pi / 5),
             cy + ry * math.sin(-math.pi / 2 + i * 2 * math.pi / 5))
            for i in range(5)
        ]
    raise ValueError(f"Not a polygon shape: {shape}")


def draw_object(draw: ImageDraw.ImageDraw, obj: dict[str, Any], mask: bool, line_width: int) -> None:
    box = obj["box"]
    fill = 255 if mask else COLORS[obj["color"]]
    outline = 255 if mask else tuple(round(channel * 0.55) for channel in COLORS[obj["color"]])
    shape = obj["shape"]
    if shape == "circle":
        draw.ellipse([box[0], box[1], box[2] - 1, box[3] - 1], fill=fill, outline=outline, width=line_width)
    elif shape == "square":
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=fill, outline=outline, width=line_width)
    else:
        draw.polygon(polygon_points(box, shape), fill=fill, outline=outline, width=line_width)


def object_mask(obj: dict[str, Any], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw_object(draw, obj, True, max(2, min(width, height) // 128))
    return np.asarray(image, dtype=np.uint8) > 0


def visibility(scene: dict[str, Any]) -> list[float]:
    width, height = scene["width"], scene["height"]
    masks = [object_mask(obj, width, height) for obj in scene["objects"]]
    fractions = []
    for index, mask in enumerate(masks):
        original = int(mask.sum())
        if original <= 0:
            fractions.append(0.0)
            continue
        if index + 1 < len(masks):
            later = np.logical_or.reduce(masks[index + 1:])
            visible = mask & ~later
        else:
            visible = mask
        fractions.append(int(visible.sum()) / original)
    return fractions


def render_scene(scene: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (scene["width"], scene["height"]), tuple(scene["background"]))
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(scene["width"], scene["height"]) // 128)
    for obj in scene["objects"]:
        draw_object(draw, obj, False, line_width)
    return image


def scene_signature(scene: dict[str, Any]) -> str:
    value = {
        "width": scene["width"],
        "height": scene["height"],
        "background": scene["background"],
        "objects": scene["objects"],
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def center(obj: dict[str, Any]) -> tuple[float, float]:
    box = obj["box"]
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def make_question(
    rng: random.Random,
    scene: dict[str, Any],
    target_index: int,
    occluder_index: int | None,
) -> tuple[str, dict[str, Any]]:
    objects = scene["objects"]
    target = objects[target_index]
    base = f"the {target['color']} {target['shape']}"
    choice = rng.random()
    if choice < 0.50:
        return base, {"template": "color_shape"}
    if choice < 0.70:
        cx, cy = center(target)
        vertical = "top" if cy < scene["height"] / 2 else "bottom"
        horizontal = "left" if cx < scene["width"] / 2 else "right"
        return f"{base} in the {vertical} {horizontal}", {
            "template": "quadrant", "vertical": vertical, "horizontal": horizontal,
        }
    if choice < 0.82:
        sizes = sorted(obj["box"][2] - obj["box"][0] for obj in objects)
        median = statistics.median(sizes)
        target_size = target["box"][2] - target["box"][0]
        if target_size != median:
            adjective = "large" if target_size > median else "small"
            return f"the {adjective} {target['color']} {target['shape']}", {
                "template": "relative_size", "adjective": adjective, "scene_median_size": median,
            }
    if len(objects) == 1:
        return base, {"template": "color_shape", "fallback": "single_object"}
    if choice < 0.94 or occluder_index is None:
        other_index = rng.choice([i for i in range(len(objects)) if i != target_index])
        other = objects[other_index]
        tx, ty = center(target)
        ox, oy = center(other)
        dx, dy = ox - tx, oy - ty
        if abs(dx) >= abs(dy):
            relation = "to the left of" if dx > 0 else "to the right of"
        else:
            relation = "above" if dy > 0 else "below"
        return f"{base} {relation} the {other['color']} {other['shape']}", {
            "template": "relation", "reference_index": other_index, "relation": relation,
        }
    occluder = objects[occluder_index]
    return f"{base} behind the {occluder['color']} {occluder['shape']}", {
        "template": "occlusion", "reference_index": occluder_index, "relation": "behind",
    }


def generate_candidate(
    seed: int, split: str, index: int, attempt: int
) -> dict[str, Any]:
    rng = random.Random(attempt_seed(seed, split, index, attempt))
    width, height = rng.choice(RESOLUTIONS)
    needs_occlusion = index % 3 == 0
    object_count = 1 if (not needs_occlusion and rng.random() < 0.08) else rng.randint(2, 6)
    combos = rng.sample(COMBOS, object_count)
    objects = []
    for combo_index, (color, shape) in enumerate(combos):
        obj = {"color": color, "shape": shape, "box": []}
        if combo_index == 0:
            obj["box"] = random_box(rng, width, height)
        elif combo_index == 1 and needs_occlusion:
            obj["box"] = overlapping_box(rng, objects[0], width, height)
        else:
            obj["box"] = random_box(rng, width, height)
        objects.append(obj)
    scene = {
        "width": width,
        "height": height,
        "background": rng.choice(BACKGROUNDS),
        "objects": objects,
    }
    fractions = visibility(scene)
    if needs_occlusion:
        target_index = 0
        if not 0.55 <= fractions[target_index] <= 0.90:
            raise ValueError("requested target occlusion was not achieved")
        occluder_index = 1 if fractions[target_index] < 1.0 else None
    else:
        eligible = [i for i, fraction in enumerate(fractions) if fraction >= 0.55]
        if not eligible:
            raise ValueError("no target with at least 55 percent visible area")
        target_index = rng.choice(eligible)
        occluder_index = None
    question, question_metadata = make_question(rng, scene, target_index, occluder_index)
    scene["target_index"] = target_index
    scene["occluder_index"] = occluder_index
    scene["visible_fractions"] = fractions
    scene["question"] = question
    scene["question_metadata"] = question_metadata
    scene["generation_seed"] = attempt_seed(seed, split, index, attempt)
    scene["generation_attempt"] = attempt
    scene["requested_target_occlusion"] = needs_occlusion
    return scene


def generate_record(
    seed: int,
    split: str,
    index: int,
    output_root: Path,
    forbidden_signatures: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    counters: Counter = Counter()
    for attempt in range(1, 201):
        counters["attempts"] += 1
        try:
            scene = generate_candidate(seed, split, index, attempt)
        except ValueError:
            counters["occlusion_or_visibility_rejections"] += 1
            continue
        signature = scene_signature(scene)
        if signature in forbidden_signatures:
            counters["scene_signature_collisions_avoided"] += 1
            continue
        record_id = f"synthetic_{split}_{index + 1:06d}"
        image_path = output_root / "images" / "synthetic" / split / f"{record_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = image_path.with_suffix(".tmp.png")
        render_scene(scene).save(temporary, format="PNG")
        os.replace(temporary, image_path)
        target = scene["objects"][scene["target_index"]]
        raw_bbox = [int(v) for v in target["box"]]
        answer = [
            round(raw_bbox[0] / scene["width"], 8),
            round(raw_bbox[1] / scene["height"], 8),
            round(raw_bbox[2] / scene["width"], 8),
            round(raw_bbox[3] / scene["height"], 8),
        ]
        record = {
            "id": record_id,
            "image": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "question": scene["question"],
            "answer": answer,
            "raw_bbox": raw_bbox,
            "width": scene["width"],
            "height": scene["height"],
            "source": "synthetic",
        }
        manifest = {
            **record,
            "generator_version": GENERATOR_VERSION,
            "split_seed": split_seed(seed, split),
            "generation_seed": scene["generation_seed"],
            "generation_attempt": scene["generation_attempt"],
            "resolution": [scene["width"], scene["height"]],
            "background": scene["background"],
            "objects": scene["objects"],
            "target_index": scene["target_index"],
            "occluder_index": scene["occluder_index"],
            "visible_fractions": scene["visible_fractions"],
            "target_visible_fraction": scene["visible_fractions"][scene["target_index"]],
            "requested_target_occlusion": scene["requested_target_occlusion"],
            "question_metadata": scene["question_metadata"],
            "scene_signature": signature,
            "image_sha256": sha256_file(image_path),
            "image_bytes": image_path.stat().st_size,
        }
        counters["generated"] = 1
        return record, manifest, dict(counters)
    raise RuntimeError(f"Could not generate {split} index {index} after 200 attempts")


def validate_scene_metadata(manifest: dict[str, Any]) -> None:
    objects = manifest["objects"]
    combos = [(obj["color"], obj["shape"]) for obj in objects]
    if len(set(combos)) != len(combos):
        raise ValueError(f"duplicate color/shape combinations in {manifest['id']}")
    target = objects[manifest["target_index"]]
    if target["box"] != manifest["raw_bbox"]:
        raise ValueError(f"target/raw bbox mismatch in {manifest['id']}")
    fractions = visibility(manifest)
    if max(abs(a - b) for a, b in zip(fractions, manifest["visible_fractions"])) > 1e-12:
        raise ValueError(f"visibility mismatch in {manifest['id']}")
    if manifest["requested_target_occlusion"]:
        if not 0.55 <= fractions[manifest["target_index"]] <= 0.90:
            raise ValueError(f"requested occlusion missing in {manifest['id']}")


def summarize(records: list[dict[str, Any]], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    areas = [
        (r["answer"][2] - r["answer"][0]) * (r["answer"][3] - r["answer"][1])
        for r in records
    ]
    visible = [m["target_visible_fraction"] for m in manifests]
    expression_lengths = [len(r["question"].split()) for r in records]
    return {
        "records": len(records),
        "unique_images": len({r["image"] for r in records}),
        "resolution_counts": dict(sorted(Counter(f"{r['width']}x{r['height']}" for r in records).items())),
        "object_count_distribution": dict(sorted(Counter(len(m["objects"]) for m in manifests).items())),
        "target_shape_counts": dict(sorted(Counter(m["objects"][m["target_index"]]["shape"] for m in manifests).items())),
        "target_color_counts": dict(sorted(Counter(m["objects"][m["target_index"]]["color"] for m in manifests).items())),
        "question_template_counts": dict(sorted(Counter(m["question_metadata"]["template"] for m in manifests).items())),
        "requested_target_occlusion_count": sum(m["requested_target_occlusion"] for m in manifests),
        "target_at_least_10pct_occluded_count": sum(v <= 0.90 for v in visible),
        "target_fully_visible_count": sum(abs(v - 1.0) < 1e-12 for v in visible),
        "target_visible_fraction": {
            "min": min(visible), "max": max(visible),
            "mean": statistics.fmean(visible), "median": statistics.median(visible),
        },
        "normalized_bbox_area": {
            "min": min(areas), "max": max(areas),
            "mean": statistics.fmean(areas), "median": statistics.median(areas),
        },
        "expression_word_count": {
            "min": min(expression_lengths), "max": max(expression_lengths),
            "mean": statistics.fmean(expression_lengths), "median": statistics.median(expression_lengths),
        },
        "image_bytes_total": sum(m["image_bytes"] for m in manifests),
    }


def draw_sample(record: dict[str, Any], destination: Path) -> None:
    with Image.open(PROJECT_ROOT / record["image"]) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = record["answer"]
    box = [x1 * record["width"], y1 * record["height"], x2 * record["width"], y2 * record["height"]]
    draw.rectangle(box, outline=(255, 0, 0), width=max(3, min(record["width"], record["height"]) // 100))
    label = f"{record['id']} | {record['question']}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    label_height = text_box[3] - text_box[1] + 8
    label_y = max(0, box[1] - label_height)
    draw.rectangle([0, label_y, min(record["width"], text_box[2] + 8), label_y + label_height], fill="white")
    draw.text((4, label_y + 4), label, fill=(180, 0, 0), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def make_contact_sheet(samples: list[dict[str, Any]], destination: Path) -> None:
    cell_width, cell_height = 480, 400
    sheet = Image.new("RGB", (960, 2000), "white")
    draw = ImageDraw.Draw(sheet)
    for index, sample in enumerate(samples):
        with Image.open(PROJECT_ROOT / sample["qa_image"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((cell_width, cell_height - 34))
        x, y = (index % 2) * cell_width, (index // 2) * cell_height
        draw.text((x + 5, y + 8), f"{sample['split']} {index + 1}: {sample['id']}", fill="black", font=ImageFont.load_default())
        sheet.paste(image, (x, y + 34))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def generate_qa(
    records: dict[str, list[dict[str, Any]]], output_root: Path, seed: int
) -> dict[str, Any]:
    samples = []
    for split in ("train", "eval"):
        chosen = random.Random(seed + (0 if split == "train" else 1)).sample(records[split], 10)
        split_samples = []
        for record in chosen:
            qa_path = output_root / "qa" / "synthetic" / split / f"{record['id']}.png"
            draw_sample(record, qa_path)
            split_samples.append({
                "split": split,
                "id": record["id"],
                "image": record["image"],
                "question": record["question"],
                "answer": record["answer"],
                "raw_bbox": record["raw_bbox"],
                "qa_image": qa_path.relative_to(PROJECT_ROOT).as_posix(),
            })
        make_contact_sheet(split_samples, output_root / "qa" / "synthetic" / f"{split}_contact_sheet.png")
        samples.extend(split_samples)
    manifest = {
        "generated_at": utc_now(),
        "seed": seed,
        "samples_per_split": 10,
        "samples": samples,
        "visual_review": "pending",
    }
    write_json(output_root / "qa" / "synthetic" / "sample_manifest.json", manifest)
    return manifest


def cross_split_checks(
    manifests: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    checks = {}
    for field in ("id", "image", "scene_signature", "image_sha256"):
        train_values = {m[field] for m in manifests["train"]}
        eval_values = {m[field] for m in manifests["eval"]}
        checks[f"train_eval_{field}_intersection_count"] = len(train_values & eval_values)
    checks["train_duplicate_id_count"] = len(manifests["train"]) - len({m["id"] for m in manifests["train"]})
    checks["eval_duplicate_id_count"] = len(manifests["eval"]) - len({m["id"] for m in manifests["eval"]})
    checks["train_duplicate_scene_signature_count"] = len(manifests["train"]) - len({m["scene_signature"] for m in manifests["train"]})
    checks["eval_duplicate_scene_signature_count"] = len(manifests["eval"]) - len({m["scene_signature"] for m in manifests["eval"]})
    return checks


def run_generation(train_size: int, eval_size: int, seed: int, output_root: Path, update_shared: bool) -> dict[str, Any]:
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    generation_counters: dict[str, Counter] = {"train": Counter(), "eval": Counter()}
    forbidden: set[str] = set()
    for split, size in (("train", train_size), ("eval", eval_size)):
        records[split] = []
        manifests[split] = []
        for index in range(size):
            record, manifest, counters = generate_record(seed, split, index, output_root, forbidden)
            generation_counters[split].update(counters)
            signature = manifest["scene_signature"]
            if signature in forbidden:
                raise RuntimeError(f"duplicate scene escaped generation: {signature}")
            forbidden.add(signature)
            validate_scene_metadata(manifest)
            records[split].append(record)
            manifests[split].append(manifest)
            if (index + 1) % 500 == 0 or index + 1 == size:
                print(f"{split}: generated {index + 1}/{size}", flush=True)
        write_jsonl(output_root / f"synthetic_{split}.jsonl", records[split])
        write_jsonl(output_root / "manifests" / f"synthetic_{split}.jsonl", manifests[split])

    checks = cross_split_checks(manifests)
    if any(checks.values()):
        raise RuntimeError(f"Synthetic split/duplicate checks failed: {checks}")
    for split in ("train", "eval"):
        for record, manifest in zip(records[split], manifests[split]):
            with Image.open(PROJECT_ROOT / record["image"]) as image:
                image.load()
                if image.size != (record["width"], record["height"]):
                    raise RuntimeError(f"image dimensions mismatch: {record['id']}")
            if sha256_file(PROJECT_ROOT / record["image"]) != manifest["image_sha256"]:
                raise RuntimeError(f"image hash mismatch: {record['id']}")
    qa = generate_qa(records, output_root, seed)
    stats = {
        "dataset": "synthetic",
        "generator_version": GENERATOR_VERSION,
        "derived_from": "hw6_rsi_localize.py draw_scene/render/make_dataset; extended without importing torch",
        "generated_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "config": {
            "seed": seed, "train_split_seed": split_seed(seed, "train"), "eval_split_seed": split_seed(seed, "eval"),
            "train_size": train_size, "eval_size": eval_size,
            "colors": list(COLORS), "shapes": list(SHAPES), "resolutions": list(RESOLUTIONS),
            "object_count": "1 with 8% probability in non-occlusion scenes, otherwise 2-6",
            "occlusion": "every third sample requests a target with 55%-90% visible area; exact mask visibility recorded",
            "question_uniqueness": "at most one object per color/shape combination per scene",
        },
        "splits": {split: summarize(records[split], manifests[split]) for split in ("train", "eval")},
        "generation_counters": {split: dict(sorted(generation_counters[split].items())) for split in ("train", "eval")},
        "quality_checks": checks,
        "outputs": {},
        "qa": {
            "samples_per_split": 10, "sample_count": len(qa["samples"]),
            "sample_manifest": (output_root / "qa" / "synthetic" / "sample_manifest.json").relative_to(PROJECT_ROOT).as_posix(),
            "train_contact_sheet": (output_root / "qa" / "synthetic" / "train_contact_sheet.png").relative_to(PROJECT_ROOT).as_posix(),
            "eval_contact_sheet": (output_root / "qa" / "synthetic" / "eval_contact_sheet.png").relative_to(PROJECT_ROOT).as_posix(),
            "visual_review": "pending",
        },
    }
    for split in ("train", "eval"):
        path = output_root / f"synthetic_{split}.jsonl"
        stats["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output_root / "synthetic_stats.json", stats)
    if update_shared:
        shared_path = DATA_ROOT / "stats.json"
        shared = json.loads(shared_path.read_text(encoding="utf-8")) if shared_path.exists() else {}
        shared["synthetic"] = stats
        write_json(shared_path, shared)
    print(json.dumps({
        "status": "complete_pending_visual_review",
        "train_records": len(records["train"]), "eval_records": len(records["eval"]),
        "quality_checks": checks,
        "elapsed_seconds": stats["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-only", action="store_true", help="Generate 20/10 under data/raw/synthetic_smoke")
    args = parser.parse_args()
    if args.smoke_only:
        run_generation(20, 10, args.seed, DATA_ROOT / "raw" / "synthetic_smoke", False)
    else:
        if args.train_size <= 0 or args.eval_size <= 0:
            parser.error("sizes must be positive")
        run_generation(args.train_size, args.eval_size, args.seed, DATA_ROOT, True)


if __name__ == "__main__":
    main()
