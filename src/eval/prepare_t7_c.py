"""CPU-only preparation for the T7 C (p1_pixel) evaluation, run after training."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
ADAPTER = PROJECT_ROOT / "adapters" / "sft_t7_c_pixel" / "adapter"
TRAINING_SUMMARY = PROJECT_ROOT / "adapters" / "sft_t7_c_pixel" / "training_summary.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    import os

    child_env = os.environ.copy()
    child_env.update(env)
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env,
    )


def dry_run(dataset: str, split: str, data: str) -> dict:
    suffix = "synthetic" if dataset == "synthetic" else "refcoco"
    args = [
        sys.executable, "-m", "src.eval.run_zeroshot",
        "--run-id", f"t7_cpu_dryrun_c_{suffix}",
        "--dataset", dataset, "--split", split, "--data", data,
        "--output", f"results/zeroshot_t7_cpu_dryrun_c_{suffix}.json",
        "--details-output", f"results/eval_t7_cpu_dryrun_c_{suffix}_details.jsonl",
        "--model-path", "models/SmolVLM-256M-Instruct",
        "--adapter", "adapters/sft_t7_c_pixel/adapter",
        "--prompt-id", "p1_pixel",
        "--expected-n", "1000", "--limit", "5",
        "--image-size", "384", "--max-new-tokens", "64", "--seed", "42",
        "--device", "cuda", "--batch-size", "1", "--dry-run",
    ]
    started = time.perf_counter()
    completed = run(args)
    return {
        "command": " ".join(args),
        "exit_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "preflight": json.loads(completed.stdout) if completed.returncode == 0 else None,
        "stderr_tail": completed.stderr[-500:],
    }


def main() -> None:
    from src.eval.run_zeroshot import load_adapter_metadata

    if not (ADAPTER / "adapter_model.safetensors").is_file() or not TRAINING_SUMMARY.is_file():
        raise SystemExit("C adapter/training_summary not present")
    training = read_json(TRAINING_SUMMARY)
    if not training.get("complete"):
        raise SystemExit(f"C training summary is not complete: {training.get('step')}")
    meta = load_adapter_metadata(ADAPTER, model_path=PROJECT_ROOT / "models" / "SmolVLM-256M-Instruct")

    dry = {
        "synthetic": dry_run("synthetic", "eval", "data/synthetic_eval.jsonl"),
        "refcoco": dry_run("refcoco", "val", "data/refcoco_val.jsonl"),
    }
    for key, value in dry.items():
        if value["exit_code"] != 0:
            raise SystemExit(f"C dry-run failed for {key}: {value['stderr_tail']}")
        preflight = value["preflight"]
        if preflight["prompt_id"] != "p1_pixel" or preflight["representation"] != "pixel_xyxy":
            raise SystemExit(f"C dry-run prompt contract mismatch: {preflight}")
        if preflight["coordinate_conversion_required"] is not True:
            raise SystemExit("C dry-run conversion flag missing")
        if preflight["adapter_model_sha256"] != meta["adapter_model_sha256"]:
            raise SystemExit("C dry-run adapter sha mismatch")

    # real CPU end-to-end (20 samples) + validation for the C path
    args = [
        sys.executable, "-m", "src.eval.run_zeroshot",
        "--run-id", "t7_c_cpu_e2e",
        "--dataset", "synthetic", "--split", "eval", "--data", "data/synthetic_eval.jsonl",
        "--output", "results/zeroshot_t7_c_cpu_e2e.json",
        "--details-output", "results/eval_t7_c_cpu_e2e_details.jsonl",
        "--model-path", "models/SmolVLM-256M-Instruct",
        "--adapter", "adapters/sft_t7_c_pixel/adapter",
        "--prompt-id", "p1_pixel",
        "--expected-n", "1000", "--limit", "20",
        "--image-size", "384", "--max-new-tokens", "64", "--seed", "42",
        "--device", "cpu", "--batch-size", "1",
    ]
    completed = run(args)
    if completed.returncode != 0:
        raise SystemExit(f"C CPU e2e run failed: {completed.stderr[-1000:]}")
    validation = run([
        sys.executable, "-m", "src.eval.validate_results",
        "--result", "results/zeroshot_t7_c_cpu_e2e.json",
        "--report", "results/eval_t7_c_cpu_e2e_validation.json",
        "--expected-run-id", "t7_c_cpu_e2e",
        "--expected-dataset", "synthetic", "--expected-split", "eval", "--expected-n", "20",
        "--expected-adapter", "adapters/sft_t7_c_pixel/adapter",
        "--expected-prompt-id", "p1_pixel",
    ])
    if validation.returncode != 0:
        raise SystemExit(f"C CPU e2e validation failed: {validation.stdout[-2000:]}{validation.stderr[-2000:]}")
    validation_report = read_json(RESULTS / "eval_t7_c_cpu_e2e_validation.json")
    e2e_summary = read_json(RESULTS / "zeroshot_t7_c_cpu_e2e.json")
    e2e_rows = [json.loads(line) for line in (RESULTS / "eval_t7_c_cpu_e2e_details.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    evidence_path = RESULTS / "eval_t7_cpu_dryrun.json"
    evidence = read_json(evidence_path)
    evidence["dry_runs_with_adapter_C"] = dry
    evidence["dry_runs_pending_adapters"].pop("C", None)
    evidence.setdefault("adapter_hashes", {})["C"] = {
        "path": "adapters/sft_t7_c_pixel/adapter",
        "adapter_model_sha256": meta["adapter_model_sha256"],
        "adapter_config_sha256": meta["adapter_config_sha256"],
        "training_summary": {
            "run_id": training.get("run_id"),
            "complete": training.get("complete"),
            "step": training.get("step"),
            "samples_seen": training.get("samples_seen"),
            "seed": training.get("seed"),
            "training_seconds": training.get("training_seconds"),
        },
    }
    evidence["cpu_end_to_end"]["pixel_with_adapter_C"] = {
        "summary": "results/zeroshot_t7_c_cpu_e2e.json",
        "details": "results/eval_t7_c_cpu_e2e_details.jsonl",
        "validation": "results/eval_t7_c_cpu_e2e_validation.json",
        "validation_status": validation_report["status"],
        "validation_errors": validation_report["errors"],
        "rows": e2e_summary["n"],
        "metrics": e2e_summary["metrics"],
        "parse_rate_all_rows": sum(1 for row in e2e_rows if row.get("predicted_box") is not None) / len(e2e_rows),
        "note": "CPU float32 plumbing test with the real T7 C adapter; not an official GPU metric",
    }
    import datetime as dt

    evidence["timestamp"] = dt.datetime.now().astimezone().isoformat()
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan_path = RESULTS / "eval_t7_batch1_plan.json"
    plan = read_json(plan_path)
    plan["variants"]["C"]["adapter_model_sha256"] = meta["adapter_model_sha256"]
    plan["variants"]["C"]["adapter_config_sha256"] = meta["adapter_config_sha256"]
    plan["variants"]["C"]["training"] = {
        "run_id": training.get("run_id"),
        "complete": training.get("complete"),
        "step": training.get("step"),
        "samples_seen": training.get("samples_seen"),
        "seed": training.get("seed"),
        "training_seconds": training.get("training_seconds"),
    }
    plan["variants"]["C"]["cpu_evidence"] = {
        "dryrun": "results/eval_t7_cpu_dryrun.json",
        "e2e_validation": "results/eval_t7_c_cpu_e2e_validation.json",
        "cpu_e2e_metrics": e2e_summary["metrics"],
    }
    plan["updated_at"] = dt.datetime.now().astimezone().isoformat()
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "adapter": meta,
        "training": {"run_id": training.get("run_id"), "complete": training.get("complete"), "step": training.get("step"), "samples_seen": training.get("samples_seen"), "training_seconds": training.get("training_seconds")},
        "dry_runs": {k: {"exit": v["exit_code"], "prompt_id": v["preflight"]["prompt_id"] if v["preflight"] else None, "adapter": v["preflight"]["adapter"] if v["preflight"] else None} for k, v in dry.items()},
        "cpu_e2e_validation": validation_report["status"],
        "cpu_e2e_metrics": e2e_summary["metrics"],
        "evidence": str(evidence_path.relative_to(PROJECT_ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
