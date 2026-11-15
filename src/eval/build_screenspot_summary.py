"""Build the ScreenSpot transfer results and the Table-1 ScreenSpot column.

Reads the four independent batch-1 runs (B0 base + SFT seeds 42/43/44), verifies
each from its details JSONL, and writes:
- results/eval_screenspot_summary.json
- report/tables/table1_screenspot_column.csv
- report/tables/table1_screenspot_column.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .grounding import score_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

RUNS = [
    {"key": "b0_base", "label": "B0 base (zero-shot)", "run_id": "screenspot_b0_base_v1", "adapter": None, "seed": None},
    {"key": "sft_s42", "label": "SFT v1 seed42", "run_id": "screenspot_sft_s42_v1", "adapter": "adapters/sft_v1/adapter", "seed": 42},
    {"key": "sft_s43", "label": "SFT v1 seed43", "run_id": "screenspot_sft_s43_v1", "adapter": "adapters/sft_v1_s43/adapter", "seed": 43},
    {"key": "sft_s44", "label": "SFT v1 seed44", "run_id": "screenspot_sft_s44_v1", "adapter": "adapters/sft_v1_s44/adapter", "seed": 44},
]
EXPECTED_N = 1272


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


def verify(run: dict[str, Any]) -> dict[str, Any]:
    run_id = run["run_id"]
    summary_path = RESULTS / f"zeroshot_{run_id}.json"
    validation_path = RESULTS / f"eval_{run_id}_validation.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"missing validation: {validation_path}")
    summary = read_json(summary_path)
    validation = read_json(validation_path)
    if summary.get("n") != EXPECTED_N or summary.get("details_rows") != EXPECTED_N:
        raise ValueError(f"{run_id}: incomplete rows")
    if summary.get("dataset") != "screenspot" or summary.get("split") != "test":
        raise ValueError(f"{run_id}: dataset/split mismatch")
    if summary.get("prompt_id") != "p1" or summary.get("batch_size") != 1:
        raise ValueError(f"{run_id}: prompt/batch contract mismatch")
    if summary.get("adapter") != run["adapter"]:
        raise ValueError(f"{run_id}: adapter {summary.get('adapter')} != {run['adapter']}")
    if validation.get("status") != "passed" or validation.get("errors"):
        raise ValueError(f"{run_id}: validation failed: {validation.get('errors')}")
    rows = read_jsonl(PROJECT_ROOT / summary["details_path"])
    invalid = 0
    correct = 0
    iou_values = []
    truncated = 0
    for row in rows:
        score = score_prediction(row["output_text"], row["ground_truth"])
        if score.predicted_box is None:
            invalid += 1
        if score.correct_at_0_5:
            correct += 1
        iou_values.append(score.iou)
        truncated += int(bool(row.get("hit_max_new_tokens")))
        if row.get("predicted_box") != (list(score.predicted_box) if score.predicted_box is not None else None):
            raise ValueError(f"{run_id}: predicted_box mismatch at {row['id']}")
    recomputed = {
        "n": len(rows),
        "acc@0.5": correct / len(rows),
        "mIoU": math.fsum(iou_values) / len(rows),
        "parse_rate": (len(rows) - invalid) / len(rows),
        "invalid": invalid,
    }
    for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
        if isinstance(recomputed[key], float):
            if not math.isclose(recomputed[key], summary["metrics"][key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{run_id}: {key} mismatch")
        elif recomputed[key] != summary["metrics"][key]:
            raise ValueError(f"{run_id}: {key} mismatch")
    return {
        **run,
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "details": summary["details_path"],
        "details_sha256": summary["details_sha256"],
        "validation": str(validation_path.relative_to(PROJECT_ROOT)),
        "validation_status": validation["status"],
        "metrics": summary["metrics"],
        "recomputed": recomputed,
        "hit_max_new_tokens": truncated,
        "elapsed_seconds_including_load": summary["runtime"]["elapsed_seconds_including_load"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS / "eval_screenspot_summary.json")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table1_screenspot_column.csv")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table1_screenspot_column.md")
    args = parser.parse_args()

    records = [verify(run) for run in RUNS]
    b0 = next(r for r in records if r["key"] == "b0_base")
    sft = [r for r in records if r["key"] != "b0_base"]
    sft_metrics = {
        key: mean_std([float(r["metrics"][key]) for r in sft])
        for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
    }
    for record in records:
        record["delta_vs_b0"] = {
            key: record["metrics"][key] - b0["metrics"][key]
            for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")
        }
    payload = {
        "format": "eval-screenspot-summary-v1",
        "task": {"id": "task-25", "subject": "T3b · ScreenSpot 迁移独立评测"},
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "dataset": {"path": "data/screenspot.jsonl", "split": "test", "n": EXPECTED_N, "source": "screenspot"},
        "method": "independent frozen evaluator; batch_size=1; p1 prompt; 384/no-split/seq36/max64/seed42/greedy; source bbox already normalized once and used unchanged",
        "runs": records,
        "b0_base": b0,
        "sft_three_seed_mean_std_ddof1": sft_metrics,
        "notes": [
            "B0 is the base SmolVLM-256M-Instruct with no adapter.",
            "SFT seeds 42/43/44 are the T6 adapters evaluated on ScreenSpot test; 3-seed mean/std is shown for the transfer column.",
            "No GT was modified; the disclosed same-image/same-instruction annotation pair is kept as-is.",
            "All four validations passed; hit_max_new_tokens counts are recorded per run.",
        ],
        "status": "complete" if len(records) == 4 else "incomplete",
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["variant,run_id,adapter,n,acc@0.5,mIoU,parse_rate,invalid,delta_acc_vs_b0,validation_status"]
    for record in records:
        m = record["metrics"]
        lines.append(
            f"{record['label']},{record['run_id']},{record['adapter'] or 'none'},1272,"
            f"{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},{m['invalid']},{record['delta_vs_b0']['acc@0.5']},{record['validation_status']}"
        )
    lines.append(
        "SFT 3-seed mean±std,--,--,1272,"
        f"{sft_metrics['acc@0.5']['mean']}±{sft_metrics['acc@0.5']['std_ddof1']},"
        f"{sft_metrics['mIoU']['mean']}±{sft_metrics['mIoU']['std_ddof1']},"
        f"{sft_metrics['parse_rate']['mean']}±{sft_metrics['parse_rate']['std_ddof1']},"
        f"{sft_metrics['invalid']['mean']}±{sft_metrics['invalid']['std_ddof1']},--,--"
    )
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Table 1 · ScreenSpot test transfer column (n=1272)",
        "",
        "Independent frozen evaluator; batch_size=1; p1 prompt; 384/no-split/seq36/max64/seed42. "
        "ScreenSpot source bboxes are already normalized once and are used unchanged.",
        "",
        "| Variant | run_id | Acc@0.5 | mIoU | parse_rate | invalid | ΔAcc vs B0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        m = record["metrics"]
        md.append(
            f"| {record['label']} | {record['run_id']} | {m['acc@0.5']:.6f} | {m['mIoU']:.6f} | "
            f"{m['parse_rate']:.6f} | {m['invalid']} | {record['delta_vs_b0']['acc@0.5']:+.6f} |"
        )
    md.append(
        f"| SFT 3-seed mean±std | -- | {sft_metrics['acc@0.5']['mean']:.6f} ± {sft_metrics['acc@0.5']['std_ddof1']:.6f} | "
        f"{sft_metrics['mIoU']['mean']:.6f} ± {sft_metrics['mIoU']['std_ddof1']:.6f} | "
        f"{sft_metrics['parse_rate']['mean']:.6f} ± {sft_metrics['parse_rate']['std_ddof1']:.6f} | "
        f"{sft_metrics['invalid']['mean']:.3f} ± {sft_metrics['invalid']['std_ddof1']:.3f} | -- |"
    )
    md.extend(
        [
            "",
            "Evidence: `results/eval_screenspot_summary.json`; data audit: `results/eval_screenspot_data_audit.json`.",
        ]
    )
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.summary.relative_to(PROJECT_ROOT)),
        "csv": str(args.csv.relative_to(PROJECT_ROOT)),
        "markdown": str(args.markdown.relative_to(PROJECT_ROOT)),
        "runs": {record["key"]: record["metrics"] for record in records},
        "sft_three_seed_mean_std": sft_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
