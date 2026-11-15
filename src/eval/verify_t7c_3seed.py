"""Independent verification for the T7-C (p1_pixel) 3-seed row after the
seed43 retraining fix.

Read-only: recomputes every C per-sample score from the details JSONL with the
pixel conversion layer, re-checks summary/validation/details contracts, training
hard counts/adapter hashes, and the 3-seed aggregates in the G4 summary. Writes
``results/verify_t7c_3seed.json``; nothing else is modified.

Run: ``python -m src.eval.verify_t7c_3seed``
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .grounding import metrics_from_scores
from .pixel_convert import score_prediction_pixel
from .prompts import PROMPT_VARIANTS, prompt_sha256_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
EXPECTED_N = 1000
DATASETS = {"synthetic": ("eval", "data/synthetic_eval.jsonl"), "refcoco": ("val", "data/refcoco_val.jsonl")}
SEEDS = (42, 43, 44)


def run_id_for(dataset: str, seed: int) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"t7_c{suffix}_{dataset}_v1"


def adapter_for(seed: int) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"adapters/sft_t7_c_pixel{suffix}/adapter"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"id": name, "status": "passed" if ok else "failed", "detail": detail})
        if not ok:
            problems.append(name)

    validations = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = RESULTS / f"eval_{run_id_for(dataset, seed)}_validation.json"
            payload = read_json(path) if path.is_file() else {}
            validations[f"{dataset}_s{seed}"] = {"path": str(path.relative_to(PROJECT_ROOT)),
                                                 "status": payload.get("status"), "errors": payload.get("errors")}
    check("validations_all_passed",
          all(item["status"] == "passed" and item["errors"] == [] for item in validations.values()), validations)

    per_run: dict[str, dict[str, Any]] = {}
    contract_ok = details_ok = rescore_ok = True
    for dataset, (split, data_rel) in DATASETS.items():
        dataset_ids = [row["id"] for row in read_jsonl(PROJECT_ROOT / data_rel)[:EXPECTED_N]]
        for seed in SEEDS:
            run_id = run_id_for(dataset, seed)
            summary = read_json(RESULTS / f"zeroshot_{run_id}.json")
            config = summary["config"]
            validation = read_json(RESULTS / f"eval_{run_id}_validation.json")
            contract = {
                "n": summary.get("n") == EXPECTED_N and summary.get("details_rows") == EXPECTED_N,
                "prompt": summary.get("prompt_id") == "p1_pixel" and config.get("prompt_id") == "p1_pixel",
                "representation": config.get("representation") == "pixel_xyxy",
                "conversion_applied": config.get("coordinate_conversion", {}).get("applied") is True,
                "prompt_sha": config.get("prompt_sha256") == prompt_sha256_for("p1_pixel")
                and PROMPT_VARIANTS["p1_pixel"]["sha256"] == prompt_sha256_for("p1_pixel"),
                "batch_size": summary.get("batch_size") == 1,
                "adapter": summary.get("adapter") == adapter_for(seed),
                "validation_passed": validation.get("status") == "passed" and validation.get("errors") == [],
            }
            contract_ok = contract_ok and all(contract.values())
            if not all(contract.values()):
                problems.append(f"run_contract_{dataset}_s{seed}")
            recorded = config.get("source_sha256", {})
            sources_ok = all(
                Path(__file__).with_name(name).is_file() and sha256_file(Path(__file__).with_name(name)) == value
                for name, value in recorded.items()
            )
            if seed == 43:
                # Newly produced run: evaluator sources must still match exactly.
                if not sources_ok or "pixel_convert.py" not in recorded:
                    problems.append(f"source_sha_{dataset}_s{seed}")
            else:
                # Historical runs (s42/s44) were produced before the T7-D evaluator
                # extension (grid_convert added, run_zeroshot/validate_results edited),
                # so their recorded source hashes intentionally differ from the
                # current tree.  Details rescoring below is representation-only and
                # does not depend on the changed dispatch code.
                per_run_sources_legacy = not sources_ok

            details_path = PROJECT_ROOT / summary["details_path"]
            rows = read_jsonl(details_path)
            digest_ok = sha256_file(details_path) == summary["details_sha256"]
            rows_ok = len(rows) == EXPECTED_N and [row["id"] for row in rows] == dataset_ids
            details_ok = details_ok and digest_ok and rows_ok

            scores = []
            for row in rows:
                score, _audit = score_prediction_pixel(
                    row["output_text"],
                    {"answer": row["ground_truth"], "width": row["width"], "height": row["height"]},
                )
                expected_box = list(score.predicted_box) if score.predicted_box is not None else None
                if (
                    row.get("predicted_box") != expected_box
                    or row.get("parse_error") != score.parse_error
                    or not close(row.get("iou"), score.iou)
                    or row.get("correct@0.5") != score.correct_at_0_5
                ):
                    rescore_ok = False
                    problems.append(f"row_mismatch_{dataset}_s{seed}_{row['id']}")
                    break
                scores.append(score)
            recomputed = metrics_from_scores(scores)
            metrics_ok = all(
                close(recomputed[key], summary["metrics"][key]) for key in ("acc@0.5", "mIoU", "parse_rate")
            ) and recomputed["invalid"] == summary["metrics"]["invalid"]
            rescore_ok = rescore_ok and metrics_ok
            per_run[f"{dataset}_s{seed}"] = {
                "run_id": run_id, "metrics": summary["metrics"],
                "details_sha256": summary["details_sha256"],
                "adapter_model_sha256": config["adapter"]["adapter_model_sha256"],
            }
    check("run_contracts", contract_ok, {"runs": 6})
    check("details_sha_and_rows", details_ok, {"runs": 6, "n": EXPECTED_N})
    check("row_rescoring_matches_summary", rescore_ok, {"runs": 6})

    training = {}
    training_ok = adapter_ok = True
    for seed in SEEDS:
        adapter_dir = PROJECT_ROOT / adapter_for(seed)
        run_root = adapter_dir.parent
        summary_path = run_root / "training_summary.json"
        payload = read_json(summary_path)
        hard = {
            "complete": payload.get("complete") is True,
            "step": payload.get("step") == 1250,
            "micro_batches_seen": payload.get("micro_batches_seen") == 10000,
            "forward_calls": payload.get("forward_calls") == 10000,
            "backward_calls": payload.get("backward_calls") == 10000,
            "stop_reason": payload.get("stop_reason") == "max_steps",
            "partial_final_step": payload.get("partial_final_step") is False,
            "resume_count_zero": int(payload.get("resume_count") or 0) == 0,
        }
        adapter_sha = payload.get("adapter_files", {}).get("adapter_model.safetensors", {}).get("sha256")
        file_sha = sha256_file(adapter_dir / "adapter_model.safetensors")
        training_ok = training_ok and all(hard.values())
        adapter_ok = adapter_ok and adapter_sha == file_sha
        training[f"s{seed}"] = {"hard": hard, "adapter_model_sha256": adapter_sha,
                                "loss_first": payload.get("loss_history", [None])[0],
                                "loss_last": payload.get("loss_history", [None])[-1],
                                "training_seconds": payload.get("training_seconds"),
                                "training_summary": str(summary_path.relative_to(PROJECT_ROOT))}
        if not all(hard.values()):
            problems.append(f"training_hard_counts_s{seed}")
        if adapter_sha != file_sha:
            problems.append(f"adapter_sha_mismatch_s{seed}")
    check("training_hard_counts", training_ok, training)
    check("adapter_sha_matches_training_summary", adapter_ok, {"seeds": list(SEEDS)})

    g4 = read_json(RESULTS / "eval_t7_g4_summary_3seed.json")
    aggregate_ok = True
    aggregates = {}
    for dataset in DATASETS:
        entry = g4["table"]["C"][dataset]
        if entry.get("seeds") != [42, 43, 44] or entry["aggregate"]["status"] != "three_seed_mean_std_ddof1":
            aggregate_ok = False
            problems.append(f"g4_c_seeds_{dataset}")
            continue
        metrics = {}
        for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
            values = [float(per_run[f"{dataset}_s{seed}"]["metrics"][key]) for seed in SEEDS]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            recorded = entry["aggregate"]["metrics"][key]
            if not (close(recorded["mean"], mean) and close(recorded["std_ddof1"], std)):
                aggregate_ok = False
                problems.append(f"g4_c_aggregate_{dataset}_{key}")
            metrics[key] = {"mean": mean, "std_ddof1": std}
        aggregates[dataset] = metrics
    check("g4_three_seed_aggregates", aggregate_ok, aggregates)

    report = {
        "schema": "verify_t7c_3seed/v2",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "T7-C p1_pixel 3-seed after seed43 retraining fix (old --resume did not restore LoRA weights); read-only reimplementation from details",
        "status": "passed" if not problems else "failed",
        "failed_checks": problems,
        "checks": checks,
        "runs": per_run,
        "training": training,
        "aggregates": aggregates,
    }
    (RESULTS / "verify_t7c_3seed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failed": problems, "checks": len(checks), "runs": len(per_run)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
