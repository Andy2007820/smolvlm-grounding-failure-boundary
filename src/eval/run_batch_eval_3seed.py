"""Execute the approved 3-seed batch-8 GPU evaluation plan after its A/B gate.

Safety properties:
- reads the exact commands from results/eval_3seed_batch8_plan.json;
- refuses to overwrite any existing run artifact;
- records nvidia-smi before/after each run and the real python exit code;
- aborts on non-zero exit, >7500 MiB, or a competing python GPU process;
- stops before the lease deadline instead of overrunning it;
- only after a 20/20 batch-1 vs batch-8 raw-output match does it run the six
  production runs and their validations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
PLAN_PATH = RESULTS / "eval_3seed_batch8_plan.json"
AB_EVIDENCE = RESULTS / "eval_ab_equivalence.json"
RUNS_EVIDENCE = RESULTS / "eval_3seed_batch1_runs.json"
SUMMARY_PATH = RESULTS / "eval_sft_v1_3seed_summary.json"
TABLE_PATH = PROJECT_ROOT / "report" / "tables" / "t6_sft_3seed_metrics.csv"
MEMORY_LIMIT_MIB = 7500
COMPARE_FIELDS = (
    "raw_output",
    "output_text",
    "predicted_box",
    "parse_error",
    "iou",
    "correct@0.5",
    "generated_tokens",
    "input_tokens",
    "hit_max_new_tokens",
)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gpu_stats() -> tuple[int, list[str]]:
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return int(memory.stdout.strip()), [line.strip() for line in apps.stdout.splitlines() if line.strip()]


def append_gpu_section(
    log_path: Path,
    *,
    lease_id: str,
    run_id: str,
    phase: str,
    memory_mib: int,
    apps: list[str],
) -> None:
    lines = [
        f"lease={lease_id}",
        f"run_id={run_id}",
        f"phase={phase}",
        "memory.used [MiB]",
        f"{memory_mib} MiB",
        "compute_processes:",
        "pid, process_name, used_gpu_memory [MiB]",
        *apps,
    ]
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def run_gpu_command(
    *,
    lease_id: str,
    run_id: str,
    command: str,
    gpu_log: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    memory_before, apps_before = gpu_stats()
    if memory_before > MEMORY_LIMIT_MIB:
        raise RuntimeError(f"GPU memory before {run_id} is {memory_before} MiB > {MEMORY_LIMIT_MIB}")
    competing = [line for line in apps_before if "python" in line.lower()]
    if competing:
        raise RuntimeError(f"competing python GPU process before {run_id}: {competing}")
    for path in (gpu_log, stdout_log, stderr_log):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing log: {path}")
    append_gpu_section(
        gpu_log,
        lease_id=lease_id,
        run_id=run_id,
        phase="before_process",
        memory_mib=memory_before,
        apps=apps_before,
    )
    argv = shlex.split(command)
    argv[0] = sys.executable
    started = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8", newline="\n") as out_handle, stderr_log.open(
        "w", encoding="utf-8", newline="\n"
    ) as err_handle:
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, stdout=out_handle, stderr=err_handle, timeout=2400)
    elapsed = time.perf_counter() - started
    memory_after, apps_after = gpu_stats()
    append_gpu_section(
        gpu_log,
        lease_id=lease_id,
        run_id=run_id,
        phase="after_process_exit",
        memory_mib=memory_after,
        apps=apps_after,
    )
    with gpu_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"python_exit_code={completed.returncode}\nwall_seconds={elapsed:.3f}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"run {run_id} exited {completed.returncode}; see {stderr_log}")
    if memory_after > MEMORY_LIMIT_MIB:
        raise RuntimeError(f"GPU memory after {run_id} is {memory_after} MiB > {MEMORY_LIMIT_MIB}")
    return {
        "run_id": run_id,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "memory_before_mib": memory_before,
        "memory_after_mib": memory_after,
    }


def run_validation(command: str, report_path: Path) -> dict[str, Any]:
    argv = shlex.split(command)
    argv[0] = sys.executable
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    elapsed = time.perf_counter() - started
    report = read_json(report_path) if report_path.is_file() else {}
    return {
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "status": report.get("status"),
        "errors": report.get("errors"),
        "checks_details_rows": report.get("checks", {}).get("details_rows"),
        "recomputed_metrics": report.get("checks", {}).get("recomputed_metrics"),
    }


def compare_details(details_a: Path, details_b: Path) -> dict[str, Any]:
    left = read_jsonl(details_a)
    right = read_jsonl(details_b)
    counts: dict[str, int] = {}
    ids_equal = len(left) == len(right) and all(
        a.get("id") == b.get("id") for a, b in zip(left, right)
    )
    for field in COMPARE_FIELDS:
        counts[field] = sum(1 for a, b in zip(left, right) if a.get(field) == b.get(field))
    return {
        "rows_a": len(left),
        "rows_b": len(right),
        "ids_equal": ids_equal,
        "field_equal_counts": counts,
        "all_equal": ids_equal and len(left) == len(right) and all(v == len(left) for v in counts.values()),
        "sha256_a": sha256_file(details_a),
        "sha256_b": sha256_file(details_b),
    }


def recompute_from_details(summary: dict[str, Any]) -> dict[str, Any]:
    details_path = PROJECT_ROOT / summary["details_path"]
    rows = read_jsonl(details_path)
    n = len(rows)
    invalid = sum(1 for row in rows if row.get("predicted_box") is None)
    correct = sum(1 for row in rows if row.get("correct@0.5"))
    return {
        "n": n,
        "acc@0.5": correct / n,
        "mIoU": math.fsum(float(row["iou"]) for row in rows) / n,
        "parse_rate": (n - invalid) / n,
        "invalid": invalid,
        "details_rows": n,
        "details_sha256": sha256_file(details_path),
        "unique_ids": len({row["id"] for row in rows}),
        "hit_max_new_tokens": sum(1 for row in rows if row.get("hit_max_new_tokens")),
        "correct_count": correct,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "std_ddof1": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return {"mean": mean, "std_ddof1": math.sqrt(variance)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--lease-end", default="2026-11-09T21:32:00+08:00")
    parser.add_argument("--safety-minutes", type=float, default=5.0)
    parser.add_argument("--plan", default=None, help="plan JSON path; defaults to the batch8 plan")
    parser.add_argument(
        "--skip-ab",
        action="store_true",
        help="skip the A/B gate because it already failed and batch1 fallback is mandated",
    )
    args = parser.parse_args()
    lease_end = dt.datetime.fromisoformat(args.lease_end)
    plan_path = Path(args.plan).resolve() if args.plan else PLAN_PATH
    plan = read_json(plan_path)
    started_at = now_iso()

    outputs = []
    if args.skip_ab:
        for run in plan["production_runs"]:
            outputs.append(PROJECT_ROOT / run["output"])
    else:
        for run in [plan["ab_gate"]["batch1"], plan["ab_gate"]["batch8"], *plan["production_runs"]]:
            outputs.append(PROJECT_ROOT / run["output"])
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"planned outputs already exist: {existing}")

    evidence: dict[str, Any] = {
        "format": "eval-3seed-execution-v1",
        "task": plan["task"],
        "lease_id": args.lease_id,
        "lease_window_local": "2026-11-09 20:52-22:30 UTC+8",
        "started_at": started_at,
        "plan_path": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": sha256_file(plan_path),
        "ab": {},
        "production_runs": [],
        "status": "running",
    }

    # ---- A/B gate -------------------------------------------------------
    if not args.skip_ab:
        ab = plan["ab_gate"]
        ab_results = []
        for key in ("batch1", "batch8"):
            record = ab[key]
            print(json.dumps({"phase": "ab", "run": record["run_id"], "started_at": now_iso()}), flush=True)
            result = run_gpu_command(
                lease_id=args.lease_id,
                run_id=record["run_id"],
                command=record["command"],
                gpu_log=PROJECT_ROOT / record["gpu_log"],
                stdout_log=PROJECT_ROOT / record["stdout_log"],
                stderr_log=PROJECT_ROOT / record["stderr_log"],
            )
            ab_results.append(result)
            print(json.dumps({"phase": "ab", "run": record["run_id"], "result": result}), flush=True)

        comparison = compare_details(
            PROJECT_ROOT / ab["batch1"]["details"],
            PROJECT_ROOT / ab["batch8"]["details"],
        )
        old_batch1 = PROJECT_ROOT / "results/eval_sft_v1_s42_synthetic_v1_details.jsonl"
        old_compare = None
        if old_batch1.is_file():
            old_rows = read_jsonl(old_batch1)[:20]
            new_rows = read_jsonl(PROJECT_ROOT / ab["batch1"]["details"])
            old_compare = {
                "reference": "results/eval_sft_v1_s42_synthetic_v1_details.jsonl (old official code batch1, first 20 rows)",
                "rows": len(new_rows),
                "raw_output_equal": sum(1 for a, b in zip(old_rows, new_rows) if a.get("raw_output") == b.get("raw_output")),
                "output_text_equal": sum(1 for a, b in zip(old_rows, new_rows) if a.get("output_text") == b.get("output_text")),
                "note": "advisory only; the batch1-vs-batch8 comparison governs",
            }
        ab_evidence = {
            "format": "eval-ab-equivalence-v1",
            "timestamp": now_iso(),
            "lease_id": args.lease_id,
            "adapter": "adapters/sft_v1/adapter",
            "subset": "synthetic_eval first 20 records (--limit 20)",
            "commands": {"batch1": ab["batch1"]["command"], "batch8": ab["batch8"]["command"]},
            "runs": ab_results,
            "comparison": comparison,
            "old_batch1_advisory_comparison": old_compare,
            "pass_rule": "ids equal and every listed field equal for all 20 rows",
            "passed": bool(comparison["all_equal"]),
            "gpu_logs": [ab["batch1"]["gpu_log"], ab["batch8"]["gpu_log"]],
        }
        AB_EVIDENCE.write_text(json.dumps(ab_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        evidence["ab"] = ab_evidence
        print(json.dumps({"phase": "ab_gate", "passed": comparison["all_equal"], "counts": comparison["field_equal_counts"]}), flush=True)

        if not comparison["all_equal"]:
            evidence["status"] = "ab_failed_stopped"
            evidence["finished_at"] = now_iso()
            RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": "ab_failed_stopped", "evidence": str(RUNS_EVIDENCE.relative_to(PROJECT_ROOT))}), flush=True)
            raise SystemExit(3)
    else:
        evidence["ab"] = {
            "skipped": True,
            "reason": "GPU batch1-vs-batch8 A/B failed (5/20 raw_output); Lead mandated permanent batch1 fallback",
            "failure_record": "results/eval_batch8_ab_failure.json",
            "equivalence_report": "results/eval_ab_equivalence.json",
        }
        print(json.dumps({"phase": "ab_gate", "skipped": True, "fallback": "batch1"}), flush=True)

    # ---- six production runs -------------------------------------------
    for index, run in enumerate(plan["production_runs"], start=1):
        remaining_minutes = (lease_end - dt.datetime.now().astimezone()).total_seconds() / 60.0
        if remaining_minutes < args.safety_minutes:
            evidence["status"] = "stopped_for_lease_deadline"
            evidence["stopped_before_run"] = run["run_id"]
            evidence["remaining_minutes_at_stop"] = remaining_minutes
            print(json.dumps({"status": "stopped_for_lease_deadline", "before": run["run_id"]}), flush=True)
            break
        print(json.dumps({"phase": "production", "index": index, "run": run["run_id"], "remaining_minutes": round(remaining_minutes, 2)}), flush=True)
        result = run_gpu_command(
            lease_id=args.lease_id,
            run_id=run["run_id"],
            command=run["command"],
            gpu_log=PROJECT_ROOT / run["gpu_log"],
            stdout_log=PROJECT_ROOT / run["stdout_log"],
            stderr_log=PROJECT_ROOT / run["stderr_log"],
        )
        validation = run_validation(run["validation_command"], PROJECT_ROOT / run["validation"])
        summary = read_json(PROJECT_ROOT / run["output"])
        recomputed = recompute_from_details(summary)
        metrics_match = all(
            math.isclose(recomputed[key], summary["metrics"][key], rel_tol=0.0, abs_tol=1e-12)
            if isinstance(recomputed[key], float)
            else recomputed[key] == summary["metrics"][key]
            for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
        )
        record = {
            "index": index,
            "seed": run["seed"],
            "dataset": run["dataset"],
            "run_id": run["run_id"],
            "adapter": run["adapter"],
            "adapter_model_sha256": summary["config"]["adapter"]["adapter_model_sha256"],
            "execution": result,
            "summary_metrics": summary["metrics"],
            "summary": run["output"],
            "details": run["details"],
            "validation": run["validation"],
            "validation_result": validation,
            "recomputed": recomputed,
            "metrics_match": metrics_match,
            "batch_size": summary.get("batch_size"),
            "n": summary.get("n"),
        }
        evidence["production_runs"].append(record)
        RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": "production", "run": run["run_id"], "metrics": summary["metrics"], "validation": validation["status"], "wall_seconds": round(result["wall_seconds"], 1)}), flush=True)
        if validation.get("status") != "passed" or validation.get("exit_code") != 0 or not metrics_match:
            evidence["status"] = "validation_failed_stopped"
            RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(4)

    evidence["finished_at"] = now_iso()
    expected_runs = len(plan["production_runs"])
    if len(evidence["production_runs"]) == expected_runs:
        evidence["status"] = "completed_all_planned"
    elif evidence["status"] != "stopped_for_lease_deadline":
        evidence["status"] = "completed_partial"
    RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 3-seed summary -------------------------------------------------
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    batch1_reference: dict[str, dict[str, Any]] = {}
    batch1_paths = {
        "synthetic": "results/zeroshot_sft_v1_s42_synthetic_v1.json",
        "refcoco": "results/zeroshot_sft_v1_s42_refcoco_v1.json",
    }
    for dataset, path in batch1_paths.items():
        batch1 = read_json(PROJECT_ROOT / path)
        validation_rel = f"results/{Path(path).stem.replace('zeroshot_', 'eval_')}_validation.json"
        validation_report = read_json(PROJECT_ROOT / validation_rel) if (PROJECT_ROOT / validation_rel).is_file() else {}
        batch1_reference[dataset] = {
            "run_id": batch1["run_id"],
            "n": batch1["n"],
            "metrics": batch1["metrics"],
            "summary": path,
            "validation": validation_rel,
            "validation_status": validation_report.get("status"),
            "role": "batch1 reference; seed42 G3 weak-positive; preserved and not overwritten",
        }
        summaries.setdefault(dataset, {})["seed42"] = {
            "seed": "seed42",
            "dataset": dataset,
            "run_id": batch1["run_id"],
            "adapter_model_sha256": batch1["config"]["adapter"]["adapter_model_sha256"],
            "summary_metrics": batch1["metrics"],
            "summary": path,
            "validation": validation_rel,
            "validation_result": {"status": validation_report.get("status")},
            "batch1_reference": True,
        }
    for record in evidence["production_runs"]:
        summaries.setdefault(record["dataset"], {})[record["seed"]] = record

    aggregate: dict[str, Any] = {}
    for dataset, per_seed in summaries.items():
        if len(per_seed) != 3:
            aggregate[dataset] = {
                "status": "incomplete",
                "seeds_present": sorted(per_seed),
                "note": "mean/std not computed for an incomplete seed set",
            }
            continue
        metrics = {
            key: mean_std([float(per_seed[seed]["summary_metrics"][key]) for seed in ("seed42", "seed43", "seed44")])
            for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
        }
        aggregate[dataset] = {
            "status": "complete",
            "seeds": ["seed42", "seed43", "seed44"],
            "per_seed": {
                seed: {
                    "run_id": per_seed[seed]["run_id"],
                    "adapter_model_sha256": per_seed[seed]["adapter_model_sha256"],
                    "metrics": per_seed[seed]["summary_metrics"],
                    "validation": per_seed[seed]["validation"],
                    "validation_status": per_seed[seed]["validation_result"]["status"],
                }
                for seed in ("seed42", "seed43", "seed44")
            },
            "mean_std_ddof1": metrics,
        }

    pooled_records = [record for per_seed in summaries.values() for record in per_seed.values()]
    pooled_values = [float(record["summary_metrics"]["acc@0.5"]) for record in pooled_records]
    pooled_metrics = {
        key: mean_std([float(record["summary_metrics"][key]) for record in pooled_records])
        for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
    }
    summary_payload = {
        "format": "eval-3seed-summary-v1",
        "task": plan["task"],
        "created_at": now_iso(),
        "method": "independent evaluator, n=1000 per dataset per seed, batch_size=1 (permanent fallback after the GPU batch8 A/B gate failed 5/20)",
        "std_definition": "sample standard deviation (ddof=1) across the three seeds (or six seed-dataset runs for pooled)",
        "adapter_hashes": {
            "seed42": "5050ec8d9d7667fd6aada392e0722ac0463902d9fa05960febba82559235db0d",
            "seed43": "9132f7be7561b0adb840d072395bad900bcf1857d35e12021037dbb713d9e9e2",
            "seed44": "f04049b0002e16d50faeabfd3c8fb3bd5b89bc428e844820243467b5347b27a0",
        },
        "batch1_seed42_reference": batch1_reference,
        "per_dataset": aggregate,
        "pooled_across_six_runs": {
            "n_runs": len(pooled_values),
            "mean_std_ddof1": pooled_metrics,
        },
        "g3_note": "seed42 batch1 G3 remains a weak positive (synthetic acc 0.011, RefCOCO acc 0.095 vs B0 0.0); the three-seed table reports all seeds, never only the best seed",
        "status": "complete" if len(evidence["production_runs"]) == expected_runs else evidence["status"],
        "execution_evidence": str(RUNS_EVIDENCE.relative_to(PROJECT_ROOT)),
        "ab_evidence": str(AB_EVIDENCE.relative_to(PROJECT_ROOT)),
        "ab_skipped_for_batch1_fallback": bool(args.skip_ab),
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dataset,seed,run_id,adapter_model_sha256,n,acc@0.5,mIoU,parse_rate,invalid,validation_status"]
    for dataset in ("synthetic", "refcoco"):
        if dataset not in aggregate or aggregate[dataset].get("status") != "complete":
            continue
        for seed in ("seed42", "seed43", "seed44"):
            entry = aggregate[dataset]["per_seed"][seed]
            m = entry["metrics"]
            lines.append(
                f"{dataset},{seed},{entry['run_id']},{entry['adapter_model_sha256']},1000,"
                f"{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},{m['invalid']},{entry['validation_status']}"
            )
        mean_std = aggregate[dataset]["mean_std_ddof1"]
        lines.append(
            f"{dataset},mean±std,--,--,1000,"
            f"{mean_std['acc@0.5']['mean']}±{mean_std['acc@0.5']['std_ddof1']},"
            f"{mean_std['mIoU']['mean']}±{mean_std['mIoU']['std_ddof1']},"
            f"{mean_std['parse_rate']['mean']}±{mean_std['parse_rate']['std_ddof1']},"
            f"{mean_std['invalid']['mean']}±{mean_std['invalid']['std_ddof1']},--"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": evidence["status"],
        "runs_completed": len(evidence["production_runs"]),
        "runs_evidence": str(RUNS_EVIDENCE.relative_to(PROJECT_ROOT)),
        "three_seed_summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "table": str(TABLE_PATH.relative_to(PROJECT_ROOT)),
        "ab_passed": ab_evidence["passed"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
