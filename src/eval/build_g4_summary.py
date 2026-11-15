"""Build the T7 G4 coordinate-representation table (A/B/C, single-seed pilot).

A reuses the already-validated T6 seed42 batch1 results. B and C are the new T7
runs. Every run is re-verified from its details JSONL before it is allowed into
the table; C is re-scored through the independent pixel conversion layer.

Outputs:
- results/eval_t7_g4_summary.json
- report/tables/table_g4_coordinate_variants.csv
- report/tables/table_g4_coordinate_variants.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .grounding import score_prediction
from .pixel_convert import score_prediction_pixel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

VARIANTS = {
    "A": {
        "representation": "normalized_decimal",
        "prompt_id": "p1",
        "adapter": "adapters/sft_v1/adapter",
        "runs": {
            "synthetic": {"run_id": "sft_v1_s42_synthetic_v1", "reuse": "T6 seed42 batch1 official result"},
            "refcoco": {"run_id": "sft_v1_s42_refcoco_v1", "reuse": "T6 seed42 batch1 official result"},
        },
    },
    "B": {
        "representation": "bin1000_integer",
        "prompt_id": "p1_bin1000",
        "adapter": "adapters/sft_t7_b_bin1000/adapter",
        "runs": {
            "synthetic": {"run_id": "t7_b_synthetic_v1"},
            "refcoco": {"run_id": "t7_b_refcoco_v1"},
        },
    },
    "C": {
        "representation": "pixel_xyxy",
        "prompt_id": "p1_pixel",
        "adapter": "adapters/sft_t7_c_pixel/adapter",
        "runs": {
            "synthetic": {"run_id": "t7_c_synthetic_v1"},
            "refcoco": {"run_id": "t7_c_refcoco_v1"},
        },
    },
}
DATASETS = ("synthetic", "refcoco")
SPLITS = {"synthetic": "eval", "refcoco": "val"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(variant: str, dataset: str, run: dict[str, Any]) -> dict[str, Any]:
    run_id = run["run_id"]
    summary_path = RESULTS / f"zeroshot_{run_id}.json"
    validation_path = RESULTS / f"eval_{run_id}_validation.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing T7 summary: {summary_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"missing T7 validation: {validation_path}")
    summary = read_json(summary_path)
    validation = read_json(validation_path)
    config = summary["config"]
    spec = VARIANTS[variant]
    if summary.get("n") != 1000 or summary.get("details_rows") != 1000:
        raise ValueError(f"{run_id}: incomplete rows")
    if variant == "A":
        # A reuses the T6 seed42 batch1 summaries, which predate the top-level
        # prompt_id/batch_size fields of the variant-aware runner.
        if config.get("prompt_id") != spec["prompt_id"]:
            raise ValueError(f"{run_id}: config prompt_id {config.get('prompt_id')} != {spec['prompt_id']}")
        if summary.get("batch_size") not in (None, 1) or config.get("batch_size") not in (None, 1):
            raise ValueError(f"{run_id}: batch_size is not 1")
    else:
        if summary.get("prompt_id") != spec["prompt_id"]:
            raise ValueError(f"{run_id}: prompt_id {summary.get('prompt_id')} != {spec['prompt_id']}")
        if config.get("representation") != spec["representation"]:
            raise ValueError(f"{run_id}: representation mismatch")
        if config.get("batch_size") != 1 or summary.get("batch_size") != 1:
            raise ValueError(f"{run_id}: batch_size is not 1")
    if summary.get("adapter") != spec["adapter"]:
        raise ValueError(f"{run_id}: adapter {summary.get('adapter')} != {spec['adapter']}")
    if validation.get("status") != "passed" or validation.get("errors"):
        raise ValueError(f"{run_id}: validation not passed: {validation.get('errors')}")
    rows = read_jsonl(PROJECT_ROOT / summary["details_path"])
    n = len(rows)
    invalid = 0
    correct = 0
    iou_values = []
    truncated = 0
    for row in rows:
        if variant == "C":
            score, _audit = score_prediction_pixel(
                row["output_text"],
                {"answer": row["ground_truth"], "width": row["width"], "height": row["height"]},
            )
        else:
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
        "n": n,
        "acc@0.5": correct / n,
        "mIoU": math.fsum(iou_values) / n,
        "parse_rate": (n - invalid) / n,
        "invalid": invalid,
    }
    for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
        if isinstance(recomputed[key], float):
            if not math.isclose(recomputed[key], summary["metrics"][key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{run_id}: {key} mismatch")
        elif recomputed[key] != summary["metrics"][key]:
            raise ValueError(f"{run_id}: {key} mismatch")
    return {
        "variant": variant,
        "dataset": dataset,
        "split": SPLITS[dataset],
        "run_id": run_id,
        "reused": run.get("reuse"),
        "adapter": spec["adapter"],
        "adapter_model_sha256": config["adapter"]["adapter_model_sha256"],
        "prompt_id": spec["prompt_id"],
        "representation": spec["representation"],
        "prompt_sha256": config["prompt_sha256"],
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
    parser.add_argument("--summary", type=Path, default=RESULTS / "eval_t7_g4_summary.json")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table_g4_coordinate_variants.csv")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table_g4_coordinate_variants.md")
    args = parser.parse_args()

    records = []
    for variant in ("A", "B", "C"):
        for dataset in DATASETS:
            records.append(verify_run(variant, dataset, VARIANTS[variant]["runs"][dataset]))

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        table.setdefault(record["variant"], {})[record["dataset"]] = record

    macro = {}
    for variant in ("A", "B", "C"):
        macro[variant] = {
            "acc@0.5": sum(table[variant][d]["metrics"]["acc@0.5"] for d in DATASETS) / 2.0,
            "mIoU": sum(table[variant][d]["metrics"]["mIoU"] for d in DATASETS) / 2.0,
            "parse_rate": sum(table[variant][d]["metrics"]["parse_rate"] for d in DATASETS) / 2.0,
            "note": "macro average of the two dataset metrics (not a pooled-n metric)",
        }

    payload = {
        "format": "eval-t7-g4-summary-v1",
        "task": {"id": "task-24", "subject": "T7-eval · B/C 坐标表示独立评测"},
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "scope": "single-seed pilot (seed42); NOT a 3-seed main table",
        "method": "independent frozen evaluator; n=1000 per dataset; batch_size=1; greedy p1-family prompt 384/no-split/seq36/max64; A reused from T6 seed42",
        "variants": {variant: {"representation": VARIANTS[variant]["representation"], "prompt_id": VARIANTS[variant]["prompt_id"], "adapter": VARIANTS[variant]["adapter"]} for variant in ("A", "B", "C")},
        "runs": records,
        "per_variant": table,
        "macro_average_two_datasets": macro,
        "notes": [
            "B uses the frozen parse_box 1000-bin branch; C uses the independent pixel conversion layer then frozen IoU/Acc/metrics.",
            "All six validations passed; hit_max_new_tokens totals are recorded per run.",
            "A/B/C were trained under the same budget/seed as a single-seed pilot; main table still needs 3 seeds.",
        ],
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["variant,dataset,run_id,representation,prompt_id,adapter_model_sha256,n,acc@0.5,mIoU,parse_rate,invalid,validation_status"]
    for variant in ("A", "B", "C"):
        for dataset in DATASETS:
            record = table[variant][dataset]
            m = record["metrics"]
            lines.append(
                f"{variant},{dataset},{record['run_id']},{record['representation']},{record['prompt_id']},"
                f"{record['adapter_model_sha256']},1000,{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},{m['invalid']},{record['validation_status']}"
            )
    for variant in ("A", "B", "C"):
        m = macro[variant]
        lines.append(
            f"{variant},macro,--,{VARIANTS[variant]['representation']},{VARIANTS[variant]['prompt_id']},--,2000,"
            f"{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},--,--"
        )
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# G4 · T7 coordinate-representation ablation (single-seed pilot, seed42)",
        "",
        "Independent frozen evaluator; n=1000 per dataset; batch_size=1; greedy p1-family prompt "
        "(384/no-split/seq36/max64). A reused from the validated T6 seed42 run. **Single-seed pilot, not a 3-seed main table.**",
        "",
        "| Variant | Representation | Dataset | Acc@0.5 | mIoU | parse_rate | invalid |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant in ("A", "B", "C"):
        for dataset in DATASETS:
            m = table[variant][dataset]["metrics"]
            md.append(
                f"| {variant} | {VARIANTS[variant]['representation']} | {dataset} | "
                f"{m['acc@0.5']:.6f} | {m['mIoU']:.6f} | {m['parse_rate']:.6f} | {m['invalid']} |"
            )
    md.extend(["", "| Variant | macro Acc@0.5 | macro mIoU | macro parse_rate |", "|---|---|---|---|"])
    for variant in ("A", "B", "C"):
        m = macro[variant]
        md.append(f"| {variant} | {m['acc@0.5']:.6f} | {m['mIoU']:.6f} | {m['parse_rate']:.6f} |")
    md.extend(
        [
            "",
            "Notes: B uses the frozen 1000-bin parse branch; C parses pixel xyxy, validates bounds/order, normalizes and then uses the frozen IoU/Acc metrics. "
            "All six validations passed. A was not rerun; it is the existing T6 seed42 batch1 result.",
            "",
            "Evidence: `results/eval_t7_g4_summary.json`; CSV: `report/tables/table_g4_coordinate_variants.csv`.",
        ]
    )
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.summary.relative_to(PROJECT_ROOT)),
        "csv": str(args.csv.relative_to(PROJECT_ROOT)),
        "markdown": str(args.markdown.relative_to(PROJECT_ROOT)),
        "macro": macro,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
