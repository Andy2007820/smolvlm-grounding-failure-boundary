"""Independent verification for the T7-D (p1_grid100) 3-seed run.

Read-only: recomputes every per-sample score from the details JSONL with the
grid conversion layer, re-checks summary/validation/details contracts, training
hard counts/adapter hashes, and the 3-seed aggregates in the G4 summary.  Writes
``results/verify_t7d_3seed.json``; nothing else is modified.

Run: ``python -m src.eval.verify_t7d_3seed``
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .grid_convert import score_prediction_grid
from .grounding import metrics_from_scores
from .prompts import PROMPT_VARIANTS, prompt_sha256_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
EXPECTED_N = 1000
DATASETS = {"synthetic": ("eval", "data/synthetic_eval.jsonl"), "refcoco": ("val", "data/refcoco_val.jsonl")}
SEEDS = (42, 43, 44)


def run_id_for(dataset: str, seed: int) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"t7_d{suffix}_{dataset}_v1"


def adapter_for(seed: int) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"adapters/sft_t7_d_grid100{suffix}/adapter"


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

    # 1. every D validation passed
    validations = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = RESULTS / f"eval_{run_id_for(dataset, seed)}_validation.json"
            payload = read_json(path) if path.is_file() else {}
            validations[f"{dataset}_s{seed}"] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "status": payload.get("status"),
                "errors": payload.get("errors"),
            }
    passed = all(item["status"] == "passed" and item["errors"] == [] for item in validations.values())
    check("validations_all_passed", passed, validations)

    # 2. summary / details / adapter contract + independent rescoring
    per_run: dict[str, dict[str, Any]] = {}
    recompute_ok = True
    contract_ok = True
    details_ok = True
    for dataset, (split, data_rel) in DATASETS.items():
        dataset_ids = [row["id"] for row in read_jsonl(PROJECT_ROOT / data_rel)[:EXPECTED_N]]
        for seed in SEEDS:
            run_id = run_id_for(dataset, seed)
            summary = read_json(RESULTS / f"zeroshot_{run_id}.json")
            validation = read_json(RESULTS / f"eval_{run_id}_validation.json")
            config = summary["config"]
            details_path = PROJECT_ROOT / summary["details_path"]
            run_contract = {
                "n": summary.get("n") == EXPECTED_N and summary.get("details_rows") == EXPECTED_N,
                "prompt": summary.get("prompt_id") == "p1_grid100" and config.get("prompt_id") == "p1_grid100",
                "representation": config.get("representation") == "grid100_integer",
                "conversion_applied": config.get("coordinate_conversion", {}).get("applied") is True,
                "prompt_sha": config.get("prompt_sha256") == prompt_sha256_for("p1_grid100")
                and PROMPT_VARIANTS["p1_grid100"]["sha256"] == prompt_sha256_for("p1_grid100"),
                "batch_size": summary.get("batch_size") == 1,
                "adapter": summary.get("adapter") == adapter_for(seed),
                "validation_passed": validation.get("status") == "passed" and validation.get("errors") == [],
            }
            contract_ok = contract_ok and all(run_contract.values())
            if not all(run_contract.values()):
                problems.append(f"run_contract_{dataset}_s{seed}")

            recorded_sources = config.get("source_sha256", {})
            sources_ok = True
            for name, recorded in recorded_sources.items():
                current_path = Path(__file__).with_name(name)
                if not current_path.is_file() or sha256_file(current_path) != recorded:
                    sources_ok = False
                    problems.append(f"source_sha_{dataset}_s{seed}_{name}")
            has_grid_source = "grid_convert.py" in recorded_sources
            if not has_grid_source or not sources_ok:
                problems.append(f"source_sha_missing_or_stale_{dataset}_s{seed}")
            run_contract["source_sha_current"] = sources_ok and has_grid_source

            rows = read_jsonl(details_path)
            digest_ok = sha256_file(details_path) == summary["details_sha256"]
            rows_ok = len(rows) == EXPECTED_N and [row["id"] for row in rows] == dataset_ids
            details_ok = details_ok and digest_ok and rows_ok
            scores = []
            row_ok = True
            for row in rows:
                score, _audit = score_prediction_grid(
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
                    row_ok = False
                    problems.append(f"row_mismatch_{dataset}_s{seed}_{row['id']}")
                    break
                scores.append(score)
            recomputed = metrics_from_scores(scores)
            metrics_ok = all(
                close(recomputed[key], summary["metrics"][key]) for key in ("acc@0.5", "mIoU", "parse_rate")
            ) and recomputed["invalid"] == summary["metrics"]["invalid"]
            recompute_ok = recompute_ok and row_ok and metrics_ok
            per_run[f"{dataset}_s{seed}"] = {
                "run_id": run_id,
                "metrics": summary["metrics"],
                "details_sha256": summary["details_sha256"],
                "adapter_model_sha256": config["adapter"]["adapter_model_sha256"],
            }
    check("run_contracts", contract_ok, {"runs": 6})
    check("details_sha_and_rows", details_ok, {"runs": 6, "n": EXPECTED_N})
    check("row_rescoring_matches_summary", recompute_ok, {"runs": 6})

    # 3. training hard counts + adapter hashes
    training = {}
    training_ok = True
    adapter_ok = True
    for seed in SEEDS:
        adapter_dir = PROJECT_ROOT / adapter_for(seed)
        run_root = adapter_dir.parent
        summary_path = run_root / "training_summary.json"
        payload = read_json(summary_path)
        resume_count = int(payload.get("resume_count") or 0)
        hard = {
            "complete": payload.get("complete") is True,
            "step": payload.get("step") == 1250,
            "stop_reason": payload.get("stop_reason") == "max_steps",
            "partial_final_step": payload.get("partial_final_step") is False,
        }
        if resume_count == 0:
            hard["micro_batches_seen"] = payload.get("micro_batches_seen") == 10000
            hard["forward_calls"] = payload.get("forward_calls") == 10000
            hard["backward_calls"] = payload.get("backward_calls") == 10000
        else:
            # A resumed run records only the resumed segment in its summary.
            # T7-D s42 resumed from checkpoint-1200 (50 steps = 400 micro); the
            # pre-resume segment (1,250 steps / 10,000 micro) is kept in
            # results/train_sft_t7_d_grid100_s42_attempt1_summary.json and must
            # describe the same adapter bytes.
            attempt1_path = RESULTS / f"train_sft_t7_d_grid100_s{seed}_attempt1_summary.json"
            attempt1 = read_json(attempt1_path) if attempt1_path.is_file() else {}
            hard["resumed_segment_micro"] = payload.get("micro_batches_seen") == 400
            hard["attempt1_1250_steps"] = attempt1.get("step") == 1250
            hard["attempt1_10000_micro"] = attempt1.get("micro_batches_seen") == 10000
            hard["resume_adapter_load"] = (
                payload.get("resume_adapter_load", {}).get("tensors", 0) > 0
                and not payload.get("resume_adapter_load", {}).get("missing_keys")
                and not payload.get("resume_adapter_load", {}).get("unexpected_keys")
            )
        adapter_sha = payload.get("adapter_files", {}).get("adapter_model.safetensors", {}).get("sha256")
        file_sha = sha256_file(adapter_dir / "adapter_model.safetensors")
        sha_ok = adapter_sha is not None and adapter_sha == file_sha
        results_summary_path = RESULTS / f"train_sft_t7_d_grid100_s{seed}_training.json"
        results_payload = read_json(results_summary_path) if results_summary_path.is_file() else {}
        results_sha = (
            results_payload.get("final_adapter_summary", {})
            .get("adapter_files", {})
            .get("adapter_model.safetensors", {})
            .get("sha256")
        )
        sha_ok = sha_ok and results_sha == file_sha
        training_ok = training_ok and all(hard.values())
        adapter_ok = adapter_ok and sha_ok
        training[f"s{seed}"] = {
            "hard": hard,
            "adapter_model_sha256": adapter_sha,
            "results_summary_sha_matches": results_sha == file_sha,
            "loss_first": payload.get("loss_history", [None])[0],
            "loss_last": payload.get("loss_history", [None])[-1],
            "training_seconds": payload.get("training_seconds"),
            "resume_count": payload.get("resume_count"),
            "training_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        }
        if not all(hard.values()):
            problems.append(f"training_hard_counts_s{seed}")
        if not sha_ok:
            problems.append(f"adapter_sha_mismatch_s{seed}")
    check("training_hard_counts", training_ok, training)
    check("adapter_sha_matches_training_summary", adapter_ok, {"seeds": list(SEEDS)})

    # 4. 3-seed aggregates in the G4 summary
    g4 = read_json(RESULTS / "eval_t7_g4_summary_3seed.json")
    aggregate_ok = True
    aggregates = {}
    for dataset in DATASETS:
        entry = g4["table"]["D"][dataset]
        if entry.get("seeds") != [42, 43, 44] or entry["aggregate"]["status"] != "three_seed_mean_std_ddof1":
            aggregate_ok = False
            problems.append(f"g4_d_seeds_{dataset}")
            continue
        metrics = {}
        for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
            values = [float(per_run[f"{dataset}_s{seed}"]["metrics"][key]) for seed in SEEDS]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            recorded = entry["aggregate"]["metrics"][key]
            if not (close(recorded["mean"], mean) and close(recorded["std_ddof1"], std)):
                aggregate_ok = False
                problems.append(f"g4_d_aggregate_{dataset}_{key}")
            metrics[key] = {"mean": mean, "std_ddof1": std}
        aggregates[dataset] = metrics
    check("g4_three_seed_aggregates", aggregate_ok, aggregates)

    report = {
        "schema": "verify_t7d_3seed/v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "T7-D p1_grid100 3-seed (42/43/44) independent verification; read-only reimplementation from details",
        "status": "passed" if not problems else "failed",
        "failed_checks": problems,
        "checks": checks,
        "runs": per_run,
        "training": training,
        "aggregates": aggregates,
    }
    (RESULTS / "verify_t7d_3seed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failed": problems, "checks": len(checks), "runs": len(per_run)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
