#!/usr/bin/env python3
"""Independent verifier for the GGUF (llama.cpp) official-protocol evaluation.

The GGUF run cannot be checked by the frozen transformers validator, so this script
performs the equivalent checks against the frozen parser/scorer:

1. summary/dataset/prompt/representation contract;
2. details rows == n == expected-n, unique ids, exact id set == dataset id set;
3. per-row **recomputation** of the score from ``raw_output`` with the frozen
   ``grounding.score_prediction`` and comparison with the stored iou/correct/parse;
4. metrics recomputed from the details == summary.metrics (exact, 1e-12);
5. details sha256 == summary.details_sha256;
6. model / mmproj / dataset sha256 == the files on disk;
7. every row's process exited 0 (no silent CLI failures);
8. invalid counts consistent with parse failures.

Writes results/verify_gguf_eval_<dataset>.json and exits 0 when passed.

Usage: python -m src.eval.verify_gguf_eval --dataset synthetic [--no-write]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval import grounding  # noqa: E402
from eval.prompts import representation_for, prompt_sha256_for  # noqa: E402

TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(check_id: str, description: str, problems: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check_id,
        "description": description,
        "status": "failed" if problems else "passed",
        "problems": problems,
        "details": details,
    }


def resolve(dataset: str) -> dict[str, Any]:
    run_id = f"gguf_q4km_{dataset}_v1"
    summary_path = PROJECT_ROOT / f"results/zeroshot_{run_id}.json"
    details_path = PROJECT_ROOT / f"results/eval_{run_id}_details.jsonl"
    return {"run_id": run_id, "summary": summary_path, "details": details_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GGUF official-protocol evaluation verifier")
    parser.add_argument("--dataset", required=True, choices=("synthetic", "refcoco"))
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    paths = resolve(args.dataset)
    summary_path, details_path = paths["summary"], paths["details"]
    checks: list[dict[str, Any]] = []
    if not summary_path.is_file() or not details_path.is_file():
        report = {
            "status": "pending",
            "dataset": args.dataset,
            "missing": [str(path.relative_to(PROJECT_ROOT)) for path in (summary_path, details_path) if not path.is_file()],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 1. summary contract
    data_rel = summary.get("config", {}).get("dataset_path")
    data_path = PROJECT_ROOT / data_rel if data_rel else None
    data_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()] if data_path and data_path.is_file() else []
    problems: list[str] = []
    if summary.get("dataset") != args.dataset:
        problems.append(f"summary dataset={summary.get('dataset')!r}")
    if summary.get("prompt_id") != "p1":
        problems.append(f"prompt_id={summary.get('prompt_id')!r} (expected p1)")
    if summary.get("representation") != representation_for(summary.get("prompt_id", "p1")):
        problems.append(f"representation={summary.get('representation')!r}")
    if summary.get("config", {}).get("prompt_sha256") != prompt_sha256_for(summary.get("prompt_id", "p1")):
        problems.append("prompt_sha256 mismatch")
    if not data_rows:
        problems.append("dataset rows not readable")
    checks.append(_check("summary_contract", "summary 的 dataset/prompt/representation 契约", problems, {"run_id": summary.get("run_id")}))

    # 2. row counts / unique ids / id set matches dataset
    problems = []
    details_rows = summary.get("details_rows")
    if len(rows) != args.expected_n:
        problems.append(f"details rows {len(rows)} != expected-n {args.expected_n}")
    if details_rows != len(rows):
        problems.append(f"summary.details_rows={details_rows} != len(details)={len(rows)}")
    if summary.get("n") != len(rows):
        problems.append(f"summary.n={summary.get('n')} != {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids in details")
    dataset_ids = {row["id"] for row in data_rows}
    if set(ids) != dataset_ids:
        missing = len(dataset_ids - set(ids))
        extra = len(set(ids) - dataset_ids)
        problems.append(f"id set mismatch vs dataset (missing {missing}, extra {extra})")
    checks.append(_check("details_rows_ids", "details 行数 / 唯一 id / 与数据集 id 集合一致", problems, {"rows": len(rows)}))

    # 3. per-row recomputation from raw_output
    problems = []
    recomputed: list[grounding.Score] = []
    mismatches = 0
    exit_failures = 0
    for row in rows:
        score = grounding.score_prediction(row.get("raw_output", ""), row.get("ground_truth"))
        recomputed.append(score)
        if abs(score.iou - float(row["iou"])) > TOLERANCE:
            mismatches += 1
        if bool(score.correct_at_0_5) != bool(row["correct@0.5"]):
            mismatches += 1
        if (score.predicted_box is None) != (row.get("predicted_box") is None):
            mismatches += 1
        if row.get("process_exit") not in (0, None):
            exit_failures += 1
    if mismatches:
        problems.append(f"{mismatches} per-row score mismatches vs stored values")
    if exit_failures:
        problems.append(f"{exit_failures} rows had non-zero llama.cpp exit codes")
    if rows and all(row.get("backend") != "llama.cpp/llama-mtmd-cli" for row in rows):
        problems.append("backend field missing (expected llama.cpp/llama-mtmd-cli)")
    checks.append(
        _check(
            "row_recompute",
            "逐行用冻结 scorer 从 raw_output 复算 iou/correct/parse，并核对 CLI 退出码",
            problems,
            {"rows": len(rows), "mismatches": mismatches, "non_zero_exit": exit_failures},
        )
    )

    # 4. metrics recomputation
    problems = []
    metrics = grounding.metrics_from_scores(recomputed)
    for key, value in metrics.items():
        recorded = summary.get("metrics", {}).get(key)
        if isinstance(value, float):
            if not isinstance(recorded, (int, float)) or abs(float(recorded) - value) > TOLERANCE:
                problems.append(f"metric {key}: summary={recorded!r} recomputed={value!r}")
        elif recorded != value:
            problems.append(f"metric {key}: summary={recorded!r} recomputed={value!r}")
    checks.append(_check("metrics_recompute", "summary.metrics 与逐行复算完全一致", problems, {"metrics": metrics}))

    # 5. details sha256
    problems = []
    actual_sha = hashlib.sha256(details_path.read_bytes()).hexdigest()
    if actual_sha != summary.get("details_sha256"):
        problems.append(f"details sha256 {actual_sha[:12]} != summary {str(summary.get('details_sha256'))[:12]}")
    checks.append(_check("details_sha256", "details 文件 sha256 与 summary 记录一致", problems, {"sha256": actual_sha}))

    # 6. artifact shas
    problems = []
    model = summary.get("model", {})
    model_path = PROJECT_ROOT / str(model.get("path", ""))
    mmproj_path = PROJECT_ROOT / str(model.get("mmproj", {}).get("path", ""))
    if not model_path.is_file() or sha256_file(model_path) != model.get("sha256"):
        problems.append("text GGUF sha256 mismatch or missing")
    if not mmproj_path.is_file() or sha256_file(mmproj_path) != model.get("mmproj", {}).get("sha256"):
        problems.append("mmproj sha256 mismatch or missing")
    if data_path is not None and data_path.is_file() and sha256_file(data_path) != summary.get("config", {}).get("dataset_sha256"):
        problems.append("dataset sha256 mismatch")
    checks.append(
        _check(
            "artifact_shas",
            "GGUF / mmproj / dataset 的 sha256 与 summary 记录一致",
            problems,
            {"model": model_path.name, "mmproj": mmproj_path.name},
        )
    )

    # 7. invalid consistency
    problems = []
    invalid = summary.get("metrics", {}).get("invalid")
    parse_failures = sum(1 for score in recomputed if score.predicted_box is None)
    if invalid != parse_failures:
        problems.append(f"invalid={invalid} != parse failures={parse_failures}")
    parse_rate = summary.get("metrics", {}).get("parse_rate")
    if isinstance(parse_rate, (int, float)) and abs(float(parse_rate) - (1.0 - parse_failures / len(rows))) > TOLERANCE:
        problems.append(f"parse_rate={parse_rate} inconsistent with parse failures")
    checks.append(_check("invalid_consistency", "invalid / parse_rate 与逐行 parse 失败数一致", problems, {"invalid": invalid}))

    failed = [check["id"] for check in checks if check["status"] == "failed"]
    report = {
        "schema": "verify_gguf_eval/v1",
        "dataset": args.dataset,
        "run_id": summary.get("run_id"),
        "status": "failed" if failed else "passed",
        "failed_checks": failed,
        "checks": checks,
        "inputs": {
            "summary": str(summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "details": str(details_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "n": len(rows),
        },
        "note": (
            "GGUF backend verification with the frozen grounding parser/scorer. It reproduces the frozen validator's "
            "substance (counts, recomputation, hashes, contract) but is NOT the transformers validator itself; the run is "
            "still llama.cpp, so wording must stay 'GGUF (llama.cpp) evaluation' and the Table 1 row must be annotated."
        ),
    }
    if not args.no_write:
        output = PROJECT_ROOT / f"results/verify_gguf_eval_{args.dataset}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["report_path"] = str(output.relative_to(PROJECT_ROOT)).replace("\\", "/")
    print(json.dumps({"status": report["status"], "dataset": args.dataset, "failed_checks": failed, "metrics": metrics}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
