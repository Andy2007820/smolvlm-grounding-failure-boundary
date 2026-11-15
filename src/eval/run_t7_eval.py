"""Execute the approved T7 B/C four-run batch-1 evaluation under a GPU lease."""

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
PLAN_PATH = RESULTS / "eval_t7_batch1_plan.json"
RUNS_EVIDENCE = RESULTS / "eval_t7_runs.json"
MEMORY_LIMIT_MIB = 7500


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_stats() -> tuple[int, list[str]]:
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    return int(memory.stdout.strip()), [line.strip() for line in apps.stdout.splitlines() if line.strip()]


def append_gpu_section(log_path: Path, *, lease_id: str, run_id: str, phase: str, memory: int, apps: list[str]) -> None:
    lines = [
        f"lease={lease_id}", f"run_id={run_id}", f"phase={phase}",
        "memory.used [MiB]", f"{memory} MiB",
        "compute_processes:", "pid, process_name, used_gpu_memory [MiB]", *apps,
    ]
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def run_gpu(lease_id: str, run: dict[str, Any]) -> dict[str, Any]:
    run_id = run["run_id"]
    memory_before, apps_before = gpu_stats()
    if memory_before > MEMORY_LIMIT_MIB:
        raise RuntimeError(f"GPU memory before {run_id} is {memory_before} MiB")
    competing = [line for line in apps_before if "python" in line.lower()]
    if competing:
        raise RuntimeError(f"competing python GPU process before {run_id}: {competing}")
    for rel in (run["gpu_log"], run["stdout_log"], run["stderr_log"]):
        path = PROJECT_ROOT / rel
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing log: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    gpu_log = PROJECT_ROOT / run["gpu_log"]
    append_gpu_section(gpu_log, lease_id=lease_id, run_id=run_id, phase="before_process", memory=memory_before, apps=apps_before)
    argv = shlex.split(run["command"])
    argv[0] = sys.executable
    started = time.perf_counter()
    with (PROJECT_ROOT / run["stdout_log"]).open("w", encoding="utf-8", newline="\n") as out, (
        PROJECT_ROOT / run["stderr_log"]
    ).open("w", encoding="utf-8", newline="\n") as err:
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, stdout=out, stderr=err, timeout=3600)
    elapsed = time.perf_counter() - started
    memory_after, apps_after = gpu_stats()
    append_gpu_section(gpu_log, lease_id=lease_id, run_id=run_id, phase="after_process_exit", memory=memory_after, apps=apps_after)
    with gpu_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"python_exit_code={completed.returncode}\nwall_seconds={elapsed:.3f}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"run {run_id} exited {completed.returncode}")
    if memory_after > MEMORY_LIMIT_MIB:
        raise RuntimeError(f"GPU memory after {run_id} is {memory_after} MiB")
    return {"run_id": run_id, "exit_code": completed.returncode, "wall_seconds": elapsed,
            "memory_before_mib": memory_before, "memory_after_mib": memory_after}


def run_validation(command: str, report_path: Path) -> dict[str, Any]:
    argv = shlex.split(command)
    argv[0] = sys.executable
    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1800)
    elapsed = time.perf_counter() - started
    report = read_json(report_path) if report_path.is_file() else {}
    return {"report": str(report_path.relative_to(PROJECT_ROOT)), "exit_code": completed.returncode,
            "wall_seconds": elapsed, "status": report.get("status"), "errors": report.get("errors"),
            "details_rows": report.get("checks", {}).get("details_rows"),
            "recomputed_metrics": report.get("checks", {}).get("recomputed_metrics")}


def verify_run(run: dict[str, Any], summary_path: Path, prompt_id: str) -> dict[str, Any]:
    summary = read_json(summary_path)
    if summary.get("n") != 1000 or summary.get("details_rows") != 1000:
        raise ValueError(f"{run['run_id']}: incomplete rows")
    if summary.get("prompt_id") != prompt_id or summary.get("batch_size") != 1:
        raise ValueError(f"{run['run_id']}: prompt/batch mismatch")
    rows = read_jsonl(PROJECT_ROOT / summary["details_path"])
    return {"n": len(rows), "details_sha256": summary["details_sha256"], "metrics": summary["metrics"],
            "prompt_id": summary["prompt_id"], "adapter": summary.get("adapter"),
            "hit_max_new_tokens": sum(1 for row in rows if row.get("hit_max_new_tokens"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--lease-end", default="2026-11-10T06:20:00+08:00")
    parser.add_argument("--safety-minutes", type=float, default=6.0)
    args = parser.parse_args()
    lease_end = dt.datetime.fromisoformat(args.lease_end)
    plan = read_json(PLAN_PATH)
    cases = [case for letter in ("B", "C") for dataset in ("synthetic", "refcoco")
             for case in [plan["variants"][letter]["runs"][dataset]]]
    prompt_ids = {"t7_b_": "p1_bin1000", "t7_c_": "p1_pixel"}
    existing = [str(PROJECT_ROOT / case["output"]) for case in cases if (PROJECT_ROOT / case["output"]).exists()]
    if existing:
        raise FileExistsError(f"planned outputs already exist: {existing}")

    evidence: dict[str, Any] = {
        "format": "eval-t7-execution-v1", "task": plan["task"], "lease_id": args.lease_id,
        "lease_window_local": "2026-11-10 04:05-06:20 UTC+8", "started_at": now_iso(),
        "plan": str(PLAN_PATH.relative_to(PROJECT_ROOT)), "plan_sha256": sha256_file(PLAN_PATH),
        "runs": [], "status": "running",
    }
    try:
        for index, case in enumerate(cases, start=1):
            remaining = (lease_end - dt.datetime.now().astimezone()).total_seconds() / 60.0
            if remaining < args.safety_minutes:
                evidence["status"] = "stopped_for_lease_deadline"
                evidence["stopped_before"] = case["run_id"]
                evidence["remaining_minutes"] = remaining
                print(json.dumps({"status": "stopped_for_lease_deadline", "before": case["run_id"]}), flush=True)
                break
            prompt_id = prompt_ids[case["run_id"][:5]]
            print(json.dumps({"phase": "gpu", "index": index, "run": case["run_id"], "remaining_minutes": round(remaining, 2)}), flush=True)
            execution = run_gpu(args.lease_id, case)
            validation = run_validation(case["validation_command"], PROJECT_ROOT / case["validation"])
            summary_verify = verify_run(case, PROJECT_ROOT / case["output"], prompt_id)
            record = {"index": index, "run_id": case["run_id"], "prompt_id": prompt_id,
                      "execution": execution, "validation": validation, "summary": summary_verify}
            evidence["runs"].append(record)
            RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "gpu", "run": case["run_id"], "metrics": summary_verify["metrics"],
                              "validation": validation.get("status"), "wall_seconds": round(execution["wall_seconds"], 1)}), flush=True)
            if validation.get("status") != "passed" or validation.get("exit_code") != 0:
                evidence["status"] = "validation_failed_stopped"
                RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(4)
        evidence["finished_at"] = now_iso()
        evidence["status"] = "completed_all_four" if len(evidence["runs"]) == 4 else evidence["status"]
        RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        RUNS_EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if len(evidence["runs"]) == 4:
        build = subprocess.run([sys.executable, "-m", "src.eval.build_g4_summary"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        if build.returncode != 0:
            raise SystemExit(f"G4 build failed: {build.stdout[-2000:]}{build.stderr[-2000:]}")
        print(build.stdout, flush=True)
    print(json.dumps({"status": evidence["status"], "runs": [r["run_id"] for r in evidence["runs"]],
                      "evidence": str(RUNS_EVIDENCE.relative_to(PROJECT_ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
