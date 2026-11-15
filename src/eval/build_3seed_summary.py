"""Build the compliant 3-seed SFT table from the six independent batch-1 runs.

Reads only results/*.json and the validated details JSONL, recomputes every
metric, refuses to omit a seed, and writes:
- results/eval_sft_v1_3seed_summary.json (per-seed, per-dataset mean/std ddof=1,
  pooled 合计, adapter hashes, validation status, evidence references)
- report/tables/table1_sft_3seed.csv
- report/tables/table1_sft_3seed.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

ADAPTERS = {
    42: ("adapters/sft_v1/adapter", "5050ec8d9d7667fd6aada392e0722ac0463902d9fa05960febba82559235db0d"),
    43: ("adapters/sft_v1_s43/adapter", "9132f7be7561b0adb840d072395bad900bcf1857d35e12021037dbb713d9e9e2"),
    44: ("adapters/sft_v1_s44/adapter", "f04049b0002e16d50faeabfd3c8fb3bd5b89bc428e844820243467b5347b27a0"),
}
RUNS = {
    ("synthetic", 42): "sft_v1_s42_synthetic_v1",
    ("synthetic", 43): "sft_v1_s43_synthetic_v1",
    ("synthetic", 44): "sft_v1_s44_synthetic_v1",
    ("refcoco", 42): "sft_v1_s42_refcoco_v1",
    ("refcoco", 43): "sft_v1_s43_refcoco_v1",
    ("refcoco", 44): "sft_v1_s44_refcoco_v1",
}
B0 = {
    "synthetic": {"acc@0.5": 0.0, "mIoU": 0.00010114397437500003, "parse_rate": 0.004, "invalid": 996},
    "refcoco": {"acc@0.5": 0.0, "mIoU": 0.0002031602971321504, "parse_rate": 0.002, "invalid": 998},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "std_ddof1": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return {"mean": mean, "std_ddof1": math.sqrt(variance)}


def load_run(dataset: str, split: str, seed: int) -> dict[str, Any]:
    run_id = RUNS[(dataset, seed)]
    adapter, adapter_sha = ADAPTERS[seed]
    summary_path = RESULTS / f"zeroshot_{run_id}.json"
    validation_path = RESULTS / f"eval_{run_id}_validation.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"missing validation: {validation_path}")
    summary = read_json(summary_path)
    validation = read_json(validation_path)
    details = read_rows = read_jsonl(PROJECT_ROOT / summary["details_path"])
    n = len(read_rows)
    invalid = sum(1 for row in read_rows if row.get("predicted_box") is None)
    correct = sum(1 for row in read_rows if row.get("correct@0.5"))
    recomputed = {
        "n": n,
        "acc@0.5": correct / n,
        "mIoU": math.fsum(float(row["iou"]) for row in read_rows) / n,
        "parse_rate": (n - invalid) / n,
        "invalid": invalid,
    }
    if n != 1000 or summary.get("n") != 1000 or summary.get("details_rows") != 1000:
        raise ValueError(f"{run_id}: incomplete rows {n}/{summary.get('details_rows')}")
    for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
        if isinstance(recomputed[key], float):
            if not math.isclose(recomputed[key], summary["metrics"][key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{run_id}: {key} recompute mismatch")
        elif recomputed[key] != summary["metrics"][key]:
            raise ValueError(f"{run_id}: {key} recompute mismatch")
    if validation.get("status") != "passed" or validation.get("errors"):
        raise ValueError(f"{run_id}: validation not passed: {validation.get('errors')}")
    if summary.get("adapter") != adapter:
        raise ValueError(f"{run_id}: adapter {summary.get('adapter')} != {adapter}")
    if summary["config"]["adapter"]["adapter_model_sha256"] != adapter_sha:
        raise ValueError(f"{run_id}: adapter model sha mismatch")
    if summary.get("batch_size") not in (None, 1) or summary["config"].get("batch_size") not in (None, 1):
        raise ValueError(f"{run_id}: batch_size is not 1")
    legacy_batch1 = summary.get("batch_size") is None
    return {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "run_id": run_id,
        "adapter": adapter,
        "adapter_model_sha256": adapter_sha,
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "validation": str(validation_path.relative_to(PROJECT_ROOT)),
        "validation_status": validation["status"],
        "details_rows": n,
        "details_sha256": summary["details_sha256"],
        "metrics": summary["metrics"],
        "recomputed": recomputed,
        "elapsed_seconds_including_load": summary["runtime"]["elapsed_seconds_including_load"],
        "hit_max_new_tokens": sum(1 for row in read_rows if row.get("hit_max_new_tokens")),
        "generated_tokens_mean": sum(int(row["generated_tokens"]) for row in read_rows) / n,
        "legacy_batch1_summary_without_batch_size_field": legacy_batch1,
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS / "eval_sft_v1_3seed_summary.json")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table1_sft_3seed.csv")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table1_sft_3seed.md")
    args = parser.parse_args()

    records = []
    for dataset, split in (("synthetic", "eval"), ("refcoco", "val")):
        for seed in (42, 43, 44):
            records.append(load_run(dataset, split, seed))

    per_dataset: dict[str, Any] = {}
    for dataset in ("synthetic", "refcoco"):
        subset = [r for r in records if r["dataset"] == dataset]
        metrics = {
            key: mean_std([float(r["metrics"][key]) for r in subset])
            for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
        }
        per_dataset[dataset] = {
            "seeds": [r["seed"] for r in subset],
            "per_seed": {
                str(r["seed"]): {
                    "run_id": r["run_id"],
                    "adapter_model_sha256": r["adapter_model_sha256"],
                    "metrics": r["metrics"],
                    "validation": r["validation"],
                    "validation_status": r["validation_status"],
                    "details_rows": r["details_rows"],
                    "details_sha256": r["details_sha256"],
                }
                for r in subset
            },
            "mean_std_ddof1": metrics,
            "b0": B0[dataset],
        }
    pooled_metrics = {
        key: mean_std([float(r["metrics"][key]) for r in records])
        for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
    }
    payload = {
        "format": "eval-3seed-summary-v2",
        "task": {"id": "task-20", "subject": "T6c · 3-seed 独立评测与主表"},
        "method": "independent frozen evaluator; n=1000 per dataset per seed; batch_size=1 (batch8 permanently rejected by the GPU A/B gate); greedy p1 384/no-split/seq36/max64/seed42",
        "std_definition": "sample standard deviation (ddof=1) across the three seeds; pooled row uses the six seed-dataset runs",
        "seed44_status": "newly evaluated in this lease (sft_v1_s44_{synthetic,refcoco}_v1)",
        "adapter_hashes": {str(seed): {"path": ADAPTERS[seed][0], "adapter_model_sha256": ADAPTERS[seed][1]} for seed in (42, 43, 44)},
        "runs": records,
        "per_dataset": per_dataset,
        "pooled_across_six_runs": {"n_runs": len(records), "mean_std_ddof1": pooled_metrics},
        "g3_note": "seed42 batch1 G3 remains a weak positive (synthetic acc 0.011, RefCOCO acc 0.095 vs B0 0.0); all three seeds are reported, none is selected or dropped",
        "hit_max_new_tokens_total": sum(r["hit_max_new_tokens"] for r in records),
        "status": "complete_3seed",
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dataset,seed,run_id,adapter_model_sha256,n,acc@0.5,mIoU,parse_rate,invalid,validation_status"]
    for record in records:
        m = record["metrics"]
        lines.append(
            f"{record['dataset']},{record['seed']},{record['run_id']},{record['adapter_model_sha256']},1000,"
            f"{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},{m['invalid']},{record['validation_status']}"
        )
    for dataset in ("synthetic", "refcoco"):
        ms = per_dataset[dataset]["mean_std_ddof1"]
        lines.append(
            f"{dataset},mean±std,--,--,1000,"
            f"{ms['acc@0.5']['mean']}±{ms['acc@0.5']['std_ddof1']},"
            f"{ms['mIoU']['mean']}±{ms['mIoU']['std_ddof1']},"
            f"{ms['parse_rate']['mean']}±{ms['parse_rate']['std_ddof1']},"
            f"{ms['invalid']['mean']}±{ms['invalid']['std_ddof1']},--"
        )
    lines.append(
        "合计,mean±std,--,--,6000,"
        f"{pooled_metrics['acc@0.5']['mean']}±{pooled_metrics['acc@0.5']['std_ddof1']},"
        f"{pooled_metrics['mIoU']['mean']}±{pooled_metrics['mIoU']['std_ddof1']},"
        f"{pooled_metrics['parse_rate']['mean']}±{pooled_metrics['parse_rate']['std_ddof1']},"
        f"{pooled_metrics['invalid']['mean']}±{pooled_metrics['invalid']['std_ddof1']},--"
    )
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Table 1 · SFT v1 3-seed grounding results",
        "",
        "Independent frozen evaluator; n=1000 per dataset per seed; batch_size=1; "
        "p1 / 384 / no-split / image_seq_len=36 / max64 / greedy / seed42. "
        "Values are mean ± sample std (ddof=1) across seeds 42/43/44.",
        "",
        "| Dataset | n | Acc@0.5 | mIoU | parse_rate | invalid |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, label in (("synthetic", "synthetic eval"), ("refcoco", "RefCOCO val")):
        ms = per_dataset[dataset]["mean_std_ddof1"]
        md.append(
            f"| {label} | 3×1000 | {fmt(ms['acc@0.5']['mean'])} ± {fmt(ms['acc@0.5']['std_ddof1'])} | "
            f"{fmt(ms['mIoU']['mean'])} ± {fmt(ms['mIoU']['std_ddof1'])} | "
            f"{fmt(ms['parse_rate']['mean'])} ± {fmt(ms['parse_rate']['std_ddof1'])} | "
            f"{fmt(ms['invalid']['mean'])} ± {fmt(ms['invalid']['std_ddof1'])} |"
        )
    md.append(
        f"| 合计 (6 runs) | 6×1000 | {fmt(pooled_metrics['acc@0.5']['mean'])} ± {fmt(pooled_metrics['acc@0.5']['std_ddof1'])} | "
        f"{fmt(pooled_metrics['mIoU']['mean'])} ± {fmt(pooled_metrics['mIoU']['std_ddof1'])} | "
        f"{fmt(pooled_metrics['parse_rate']['mean'])} ± {fmt(pooled_metrics['parse_rate']['std_ddof1'])} | "
        f"{fmt(pooled_metrics['invalid']['mean'])} ± {fmt(pooled_metrics['invalid']['std_ddof1'])} |"
    )
    md.extend(["", "Per-seed detail:", "", "| Dataset | seed | run_id | Acc@0.5 | mIoU | parse_rate | invalid |", "|---|---|---|---|---|---|---|"])
    for record in records:
        m = record["metrics"]
        md.append(
            f"| {record['dataset']} | {record['seed']} | {record['run_id']} | "
            f"{m['acc@0.5']:.6f} | {m['mIoU']:.6f} | {m['parse_rate']:.6f} | {m['invalid']} |"
        )
    md.extend(
        [
            "",
            "B0 zero-shot reference: synthetic Acc@0.5=0.0 / mIoU=0.000101 / parse=0.004; "
            "RefCOCO Acc@0.5=0.0 / mIoU=0.000203 / parse=0.002.",
            "",
            "Notes: batch8 was rejected by the GPU A/B gate (5/20 raw_output identical) and is not used. "
            "seed42 remains the weak-positive G3 result; all three seeds are reported, none selected or dropped. "
            f"All six validations passed; hit_max_new_tokens total = {payload['hit_max_new_tokens_total']}.",
            "",
            f"Evidence: `results/eval_sft_v1_3seed_summary.json`; CSV: `report/tables/table1_sft_3seed.csv`.",
        ]
    )
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.summary.relative_to(PROJECT_ROOT)),
        "csv": str(args.csv.relative_to(PROJECT_ROOT)),
        "markdown": str(args.markdown.relative_to(PROJECT_ROOT)),
        "per_dataset": {dataset: per_dataset[dataset]["mean_std_ddof1"] for dataset in per_dataset},
        "pooled": pooled_metrics,
        "hit_max_new_tokens_total": payload["hit_max_new_tokens_total"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
