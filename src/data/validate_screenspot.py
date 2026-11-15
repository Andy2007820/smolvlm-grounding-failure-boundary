#!/usr/bin/env python3
"""Independent full reread validation of final ScreenSpot JSONL and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw" / "screenspot"
REPO = "bevaya/ScreenSpot"
REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
KEYS = ["id", "image", "question", "answer", "raw_bbox", "width", "height", "source"]
EXPECTED = {
    "data/test-00000-of-00003.parquet": (134512659, "ff06d312270eecc9d9ac968a51ceb0bc54f80cfda691ccccb8ebf4b4e5faa8fb", 424),
    "data/test-00001-of-00003.parquet": (198971508, "d48b8275f9dcff56f9c1bfedafbeb018cc200cbb649f8a92299cba0a9130891a", 424),
    "data/test-00002-of-00003.parquet": (268832649, "a28a1e9f027003730c1375550bfdd3b212d0c0b5c24b338140309a2af3674e5b", 424),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    errors: list[str] = []
    checks: Counter[str] = Counter()

    def check(condition: bool, message: str, category: str = "general") -> None:
        checks[category] += 1
        if not condition:
            errors.append(message)

    records = load_jsonl(DATA / "screenspot.jsonl")
    manifests_list = load_jsonl(DATA / "manifests" / "screenspot.jsonl")
    manifests = {m["id"]: m for m in manifests_list}
    stats = load_json(DATA / "screenspot_stats.json")
    shared = load_json(DATA / "stats.json")
    check(len(records) == 1272, f"record count {len(records)} != 1272", "schema")
    check(len(manifests_list) == len(records) == len(manifests), "manifest count/IDs differ", "schema")
    check(shared.get("screenspot") == stats, "shared stats screenspot differs from standalone", "statistics")
    check("synthetic" in shared, "existing synthetic statistics were lost", "statistics")

    source: dict[str, dict[str, Any]] = {}
    source_exact: Counter[str] = Counter()
    for name, (expected_bytes, expected_hash, expected_rows) in EXPECTED.items():
        path = RAW / "bevaya" / name
        check(path.is_file(), f"missing source shard {name}", "source")
        if not path.is_file():
            continue
        check(path.stat().st_size == expected_bytes, f"source byte size mismatch {name}", "source")
        check(sha(path) == expected_hash, f"source SHA256 mismatch {name}", "source")
        parquet = pq.ParquetFile(path)
        check(parquet.metadata.num_rows == expected_rows, f"Parquet row count mismatch {name}", "source")
        index = 0
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                source_id = f"{Path(name).name}:{index:06d}"
                index += 1
                payload = bytes(row["image"]["bytes"])
                metadata = {
                    "file_name": row["file_name"], "bbox": [float(v) for v in row["bbox"]],
                    "instruction": row["instruction"], "data_type": row["data_type"],
                    "data_source": row["data_source"], "image_struct_path": row["image"].get("path"),
                    "image_sha256": hashlib.sha256(payload).hexdigest(), "image_bytes": len(payload),
                    "source_shard": name,
                }
                source[source_id] = metadata
                key = json.dumps([metadata["image_sha256"], metadata["instruction"].strip(), metadata["bbox"],
                                  metadata["data_type"], metadata["data_source"]], ensure_ascii=False)
                source_exact[key] += 1
        check(index == expected_rows, f"iterated source rows mismatch {name}: {index}", "source")
    check(len(source) == 1272, f"source map count {len(source)} != 1272", "source")
    source_duplicate_rows = sum(count - 1 for count in source_exact.values() if count > 1)
    check(source_duplicate_rows == stats["processing_audit"]["duplicate_exact_content_records_removed"],
          "source duplicate count differs from removal audit", "duplicates")

    final_exact: Counter[str] = Counter()
    question_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    path_hashes: dict[str, str] = {}
    image_hashes: set[str] = set()
    for index, record in enumerate(records, 1):
        rid = record.get("id", f"row_{index}")
        check(list(record) == KEYS, f"{rid}: final key order/schema mismatch", "schema")
        check(rid == f"screenspot_test_{index:06d}", f"{rid}: non-sequential ID", "schema")
        check(record.get("source") == "screenspot", f"{rid}: wrong source", "schema")
        check(isinstance(record.get("question"), str) and bool(record.get("question", "").strip()), f"{rid}: empty question", "schema")
        answer = record.get("answer", [])
        check(isinstance(answer, list) and len(answer) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in answer),
              f"{rid}: invalid answer shape", "coordinates")
        if len(answer) == 4:
            check(0 <= answer[0] < answer[2] <= 1 and 0 <= answer[1] < answer[3] <= 1,
                  f"{rid}: answer is not normalized nonempty xyxy", "coordinates")
        check(rid in manifests, f"{rid}: manifest missing", "source")
        if rid not in manifests:
            continue
        manifest = manifests[rid]
        for key in KEYS:
            check(manifest.get(key) == record.get(key), f"{rid}: manifest mismatch {key}", "source")
        source_id = manifest.get("source_row_id")
        check(source_id in source, f"{rid}: unknown source_row_id {source_id}", "source")
        if source_id not in source:
            continue
        original = source[source_id]
        check(manifest.get("source_repo") == REPO and manifest.get("source_revision") == REVISION and manifest.get("source_split") == "test",
              f"{rid}: source repo/revision/split mismatch", "source")
        check(manifest.get("source_shard") == original["source_shard"], f"{rid}: shard mismatch", "source")
        check(manifest.get("source_file_name") == original["file_name"], f"{rid}: file_name mismatch", "source")
        check(manifest.get("source_image_struct_path") == original["image_struct_path"], f"{rid}: image struct path mismatch", "source")
        check(record["question"] == original["instruction"].strip(), f"{rid}: instruction mismatch", "source")
        check(manifest.get("source_bbox_normalized_xyxy") == original["bbox"], f"{rid}: source bbox mismatch", "coordinates")
        check(manifest.get("data_type") == original["data_type"] and manifest.get("data_source") == original["data_source"],
              f"{rid}: data type/source mismatch", "source")
        check(manifest.get("image_sha256") == original["image_sha256"] and manifest.get("image_bytes") == original["image_bytes"],
              f"{rid}: source image hash/bytes mismatch", "images")
        image_path = ROOT / record["image"]
        check(record["image"].startswith("data/images/screenspot/") and ".." not in Path(record["image"]).parts,
              f"{rid}: unsafe/wrong image path", "images")
        check(image_path.is_file(), f"{rid}: image missing", "images")
        if not image_path.is_file():
            continue
        output_hash = sha(image_path)
        check(output_hash == original["image_sha256"] == manifest["image_sha256"], f"{rid}: output image hash mismatch", "images")
        check(image_path.stat().st_size == manifest["image_bytes"], f"{rid}: output image byte size mismatch", "images")
        with Image.open(image_path) as opened:
            opened.load()
            check(list(opened.size) == [record["width"], record["height"]], f"{rid}: actual dimensions mismatch", "images")
            check(opened.format == manifest["image_format"], f"{rid}: image format mismatch", "images")
        bbox = original["bbox"]
        clipped = [max(0.0, min(1.0, value)) for value in bbox]
        expected_answer = [round(value, 8) for value in clipped]
        expected_raw = [round(bbox[0] * record["width"], 6), round(bbox[1] * record["height"], 6),
                        round(bbox[2] * record["width"], 6), round(bbox[3] * record["height"], 6)]
        check(record["answer"] == expected_answer, f"{rid}: normalized conversion mismatch", "coordinates")
        check(record["raw_bbox"] == expected_raw, f"{rid}: raw pixel conversion mismatch", "coordinates")
        check(manifest["bbox_clipped"] == (clipped != bbox), f"{rid}: clipping flag mismatch", "coordinates")
        content_key = json.dumps([output_hash, record["question"], bbox, manifest["data_type"], manifest["data_source"]], ensure_ascii=False)
        final_exact[content_key] += 1
        question_groups[(output_hash, record["question"])].append(content_key)
        if record["image"] in path_hashes:
            check(path_hashes[record["image"]] == output_hash, f"{rid}: same path has different image hashes", "images")
        path_hashes[record["image"]] = output_hash
        image_hashes.add(output_hash)

    duplicate_final = sum(count - 1 for count in final_exact.values() if count > 1)
    variation_groups = {key: values for key, values in question_groups.items() if len(set(values)) > 1}
    variation_extra_rows = sum(len(values) - 1 for values in variation_groups.values())
    check(duplicate_final == 0, f"final exact duplicate rows {duplicate_final}", "duplicates")
    check(variation_extra_rows == stats["processing_audit"]["conflicting_same_image_instruction_count"],
          "same-image/instruction variation count mismatch", "duplicates")
    conflict = load_json(DATA / "qa" / "screenspot" / "conflict" / "pair.json")
    check(set(conflict["ids"]) == {"screenspot_test_000559", "screenspot_test_000740"}, "unexpected disclosed conflict IDs", "duplicates")
    check(conflict["visual_review"]["status"] == "passed_with_source_annotation_variation", "conflict visual review missing", "qa")

    train_ids: set[str] = set()
    train_paths: set[str] = set()
    train_hashes: set[str] = set()
    for dataset in ("refcoco", "synthetic"):
        for record in load_jsonl(DATA / f"{dataset}_train.jsonl"):
            check(record.get("source") != "screenspot", f"ScreenSpot source leaked into {dataset}_train", "isolation")
            check(not record["image"].startswith("data/images/screenspot/"), f"ScreenSpot path leaked into {dataset}_train", "isolation")
            train_ids.add(record["id"])
            train_paths.add(record["image"])
        for manifest in load_jsonl(DATA / "manifests" / f"{dataset}_train.jsonl"):
            train_hashes.add(manifest.get("image_sha256") or sha(ROOT / manifest["image"]))
    isolation = {
        "id_intersection_count": len({r["id"] for r in records} & train_ids),
        "image_path_intersection_count": len({r["image"] for r in records} & train_paths),
        "image_sha256_intersection_count": len(image_hashes & train_hashes),
    }
    check(all(value == 0 for value in isolation.values()), f"training isolation failure {isolation}", "isolation")
    for key, value in isolation.items():
        check(stats["quality_checks"][f"screenspot_train_{key}"] == value, f"stats isolation mismatch {key}", "isolation")

    distribution = stats["distribution"]
    check(distribution["records"] == len(records), "stats record count mismatch", "statistics")
    check(distribution["unique_images"] == len(path_hashes), "stats unique image path count mismatch", "statistics")
    check(stats["processing_audit"]["unique_image_hashes"] == len(image_hashes), "stats unique image hash count mismatch", "statistics")
    check(distribution["data_source_counts"] == dict(sorted(Counter(m["data_source"] for m in manifests_list).items())),
          "platform distribution mismatch", "statistics")
    check(distribution["data_type_counts"] == dict(sorted(Counter(m["data_type"] for m in manifests_list).items())),
          "data type distribution mismatch", "statistics")
    check(distribution["image_format_counts"] == dict(sorted(Counter(m["image_format"] for m in manifests_list).items())),
          "image format distribution mismatch", "statistics")
    check(distribution["image_bytes_total"] == sum(m["image_bytes"] for m in manifests_list), "image byte total mismatch", "statistics")
    output_hash = sha(DATA / "screenspot.jsonl")
    check(stats["outputs"]["screenspot.jsonl"] == {"bytes": (DATA / "screenspot.jsonl").stat().st_size, "sha256": output_hash},
          "output byte/hash audit mismatch", "statistics")

    qa = load_json(DATA / "qa" / "screenspot" / "sample_manifest.json")
    check(len(qa["samples"]) == 10 and len({s["id"] for s in qa["samples"]}) == 10, "QA sample count/uniqueness mismatch", "qa")
    check(qa["visual_review"]["status"] == "passed" and qa["visual_review"]["full_resolution_samples_reviewed"] == 10,
          "visual review not passed for 10 full-resolution samples", "qa")
    for sample in qa["samples"]:
        check((ROOT / sample["qa_image"]).is_file(), f"missing QA overlay {sample['id']}", "qa")
    check((DATA / "qa" / "screenspot" / "contact_sheet.png").is_file(), "contact sheet missing", "qa")
    finalization = load_json(DATA / "qa" / "screenspot" / "finalization.json")
    check(finalization["status"] == "passed" and finalization["screenspot_jsonl_sha256"] == output_hash,
          "finalization status/hash mismatch", "qa")
    unit_tests = load_json(DATA / "qa" / "screenspot" / "unit_tests.json")
    check(unit_tests["status"] == "passed" and unit_tests["tests_run"] == 8 and unit_tests["failures"] == unit_tests["errors"] == 0,
          "unit test audit mismatch", "qa")
    check(stats["qa"]["visual_review"] == "passed", "stats visual review is not passed", "qa")

    report = {
        "status": "passed" if not errors else "failed",
        "records": len(records), "source_rows_reread": len(source),
        "unique_image_paths": len(path_hashes), "unique_image_hashes": len(image_hashes),
        "source_exact_duplicate_rows": source_duplicate_rows, "final_exact_duplicate_rows": duplicate_final,
        "source_annotation_variation_groups": len(variation_groups),
        "source_annotation_variation_extra_rows": variation_extra_rows,
        "variation_disposition": "both Done annotations retained; same target, IoU 0.7792577287012425; no relabeling",
        "train_isolation": isolation, "screenspot_jsonl_sha256": output_hash,
        "checks_by_category": dict(sorted(checks.items())), "check_total": sum(checks.values()),
        "error_count": len(errors), "errors": errors,
        "warnings": ["one same-image/instruction source annotation variation group retained and disclosed"] if variation_groups else [],
    }
    output = DATA / "qa" / "screenspot" / "independent_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
