"""Build the updated G4 table with 3-seed A/B/C/D coordinate-representation rows.

A: T6 seed42/43/44 batch-1 results (p1).
B: T7 B seed42/43/44 (p1_bin1000).
C: T7 C seed42/43/44 (p1_pixel, independent pixel conversion layer; seed43 was
   retrained with the fixed resume loader after the old --resume was found not to
   restore LoRA weights — old artifacts archived).
D: T7 D seed42/43/44 (p1_grid100, independent 0-100 integer-grid conversion
   layer; answer = round(normalized*100)).

Every run is re-verified from its details JSONL before it is allowed into the
table; C is re-scored through ``pixel_convert`` and D through ``grid_convert``,
then both reuse the frozen IoU/Acc@0.5/metrics code.

Outputs:
- results/eval_t7_g4_summary_3seed.json
- report/tables/table_g4_coordinate_variants_3seed.csv
- report/tables/table_g4_coordinate_variants_3seed.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .grid_convert import score_prediction_grid
from .grounding import score_prediction
from .pixel_convert import score_prediction_pixel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
DATASETS = ("synthetic", "refcoco")
SPLITS = {"synthetic": "eval", "refcoco": "val"}
EXPECTED_N = 1000


def runs_spec() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """variant -> dataset -> seed -> run metadata."""

    a_adapter = "adapters/sft_v1/adapter"
    return {
        "A": {
            "synthetic": {
                42: {"run_id": "sft_v1_s42_synthetic_v1", "adapter": a_adapter, "prompt_id": "p1", "legacy_t6": True},
                43: {"run_id": "sft_v1_s43_synthetic_v1", "adapter": "adapters/sft_v1_s43/adapter", "prompt_id": "p1", "legacy_t6": True},
                44: {"run_id": "sft_v1_s44_synthetic_v1", "adapter": "adapters/sft_v1_s44/adapter", "prompt_id": "p1", "legacy_t6": True},
            },
            "refcoco": {
                42: {"run_id": "sft_v1_s42_refcoco_v1", "adapter": a_adapter, "prompt_id": "p1", "legacy_t6": True},
                43: {"run_id": "sft_v1_s43_refcoco_v1", "adapter": "adapters/sft_v1_s43/adapter", "prompt_id": "p1", "legacy_t6": True},
                44: {"run_id": "sft_v1_s44_refcoco_v1", "adapter": "adapters/sft_v1_s44/adapter", "prompt_id": "p1", "legacy_t6": True},
            },
        },
        "B": {
            "synthetic": {
                42: {"run_id": "t7_b_synthetic_v1", "adapter": "adapters/sft_t7_b_bin1000/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
                43: {"run_id": "t7_b_s43_synthetic_v1", "adapter": "adapters/sft_t7_b_bin1000_s43/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
                44: {"run_id": "t7_b_s44_synthetic_v1", "adapter": "adapters/sft_t7_b_bin1000_s44/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
            },
            "refcoco": {
                42: {"run_id": "t7_b_refcoco_v1", "adapter": "adapters/sft_t7_b_bin1000/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
                43: {"run_id": "t7_b_s43_refcoco_v1", "adapter": "adapters/sft_t7_b_bin1000_s43/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
                44: {"run_id": "t7_b_s44_refcoco_v1", "adapter": "adapters/sft_t7_b_bin1000_s44/adapter", "prompt_id": "p1_bin1000", "representation": "bin1000_integer"},
            },
        },
        "C": {
            "synthetic": {
                42: {"run_id": "t7_c_synthetic_v1", "adapter": "adapters/sft_t7_c_pixel/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
                43: {"run_id": "t7_c_s43_synthetic_v1", "adapter": "adapters/sft_t7_c_pixel_s43/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
                44: {"run_id": "t7_c_s44_synthetic_v1", "adapter": "adapters/sft_t7_c_pixel_s44/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
            },
            "refcoco": {
                42: {"run_id": "t7_c_refcoco_v1", "adapter": "adapters/sft_t7_c_pixel/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
                43: {"run_id": "t7_c_s43_refcoco_v1", "adapter": "adapters/sft_t7_c_pixel_s43/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
                44: {"run_id": "t7_c_s44_refcoco_v1", "adapter": "adapters/sft_t7_c_pixel_s44/adapter", "prompt_id": "p1_pixel", "representation": "pixel_xyxy"},
            },
        },
        "D": {
            "synthetic": {
                42: {"run_id": "t7_d_synthetic_v1", "adapter": "adapters/sft_t7_d_grid100/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
                43: {"run_id": "t7_d_s43_synthetic_v1", "adapter": "adapters/sft_t7_d_grid100_s43/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
                44: {"run_id": "t7_d_s44_synthetic_v1", "adapter": "adapters/sft_t7_d_grid100_s44/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
            },
            "refcoco": {
                42: {"run_id": "t7_d_refcoco_v1", "adapter": "adapters/sft_t7_d_grid100/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
                43: {"run_id": "t7_d_s43_refcoco_v1", "adapter": "adapters/sft_t7_d_grid100_s43/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
                44: {"run_id": "t7_d_s44_refcoco_v1", "adapter": "adapters/sft_t7_d_grid100_s44/adapter", "prompt_id": "p1_grid100", "representation": "grid100_integer"},
            },
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "std_ddof1": 0.0, "n": n}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return {"mean": mean, "std_ddof1": math.sqrt(variance), "n": n}


def score_row(variant: str, row: dict[str, Any]) -> Any:
    record = {"answer": row["ground_truth"], "width": row["width"], "height": row["height"]}
    if variant == "C":
        return score_prediction_pixel(row["output_text"], record)[0]
    if variant == "D":
        return score_prediction_grid(row["output_text"], record)[0]
    return score_prediction(row["output_text"], row["ground_truth"])


def verify(variant: str, dataset: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    run_id = spec["run_id"]
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
    config = summary["config"]
    if spec.get("legacy_t6"):
        if config.get("prompt_id") != "p1":
            raise ValueError(f"{run_id}: legacy prompt_id mismatch")
    else:
        if summary.get("prompt_id") != spec["prompt_id"] or summary.get("batch_size") != 1:
            raise ValueError(f"{run_id}: prompt/batch mismatch")
        if config.get("prompt_id") != spec["prompt_id"]:
            raise ValueError(f"{run_id}: config prompt_id mismatch")
        if config.get("representation") != spec["representation"]:
            raise ValueError(f"{run_id}: config representation mismatch")
    if summary.get("adapter") != spec["adapter"]:
        raise ValueError(f"{run_id}: adapter {summary.get('adapter')} != {spec['adapter']}")
    if validation.get("status") != "passed" or validation.get("errors"):
        raise ValueError(f"{run_id}: validation not passed")
    rows = read_jsonl(PROJECT_ROOT / summary["details_path"])
    invalid = 0
    correct = 0
    iou_values = []
    for row in rows:
        score = score_row(variant, row)
        if score.predicted_box is None:
            invalid += 1
        if score.correct_at_0_5:
            correct += 1
        iou_values.append(score.iou)
        if row.get("predicted_box") != (list(score.predicted_box) if score.predicted_box is not None else None):
            raise ValueError(f"{run_id}: predicted_box mismatch at {row['id']}")
    recomputed = {"n": len(rows), "acc@0.5": correct / len(rows), "mIoU": math.fsum(iou_values) / len(rows),
                  "parse_rate": (len(rows) - invalid) / len(rows), "invalid": invalid}
    for key in ("acc@0.5", "mIoU", "parse_rate", "invalid"):
        if isinstance(recomputed[key], float):
            if not math.isclose(recomputed[key], summary["metrics"][key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{run_id}: {key} mismatch")
        elif recomputed[key] != summary["metrics"][key]:
            raise ValueError(f"{run_id}: {key} mismatch")
    return {"variant": variant, "dataset": dataset, "seed": seed, "run_id": run_id, "adapter": spec["adapter"],
            "prompt_id": spec["prompt_id"], "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "details": summary["details_path"], "details_sha256": summary["details_sha256"],
            "validation": str(validation_path.relative_to(PROJECT_ROOT)), "validation_status": validation["status"],
            "metrics": summary["metrics"], "recomputed": recomputed,
            "adapter_model_sha256": config["adapter"]["adapter_model_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS / "eval_t7_g4_summary_3seed.json")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table_g4_coordinate_variants_3seed.csv")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "report" / "tables" / "table_g4_coordinate_variants_3seed.md")
    args = parser.parse_args()

    records = []
    for variant, per_dataset in runs_spec().items():
        for dataset, per_seed in per_dataset.items():
            for seed, spec in per_seed.items():
                records.append(verify(variant, dataset, seed, spec))

    variants = ("A", "B", "C", "D")
    table: dict[str, dict[str, Any]] = {}
    for variant in variants:
        table[variant] = {}
        for dataset in DATASETS:
            subset = [r for r in records if r["variant"] == variant and r["dataset"] == dataset]
            if len(subset) == 1:
                record = subset[0]
                table[variant][dataset] = {"seeds": [record["seed"]], "runs": {str(record["seed"]): record},
                                           "aggregate": {"status": "single_seed_pilot", "metrics": record["metrics"]}}
            else:
                metrics = {key: mean_std([float(r["metrics"][key]) for r in subset]) for key in ("acc@0.5", "mIoU", "parse_rate", "invalid")}
                table[variant][dataset] = {"seeds": sorted(r["seed"] for r in subset),
                                           "runs": {str(r["seed"]): r for r in subset},
                                           "aggregate": {"status": "three_seed_mean_std_ddof1", "metrics": metrics}}
    payload = {
        "format": "eval-t7-g4-summary-3seed-v1",
        "task": {"id": "task-27", "subject": "T7b/T7-C/T7-D · G4 坐标表示 3-seed 表"},
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "scope": (
            "A, B, C and D are 3-seed mean±std (seed42/43/44); C seed43 was retrained with the fixed resume "
            "loader (the old --resume did not restore LoRA weights; old artifacts archived); D (text grid 0-100) is the T7-D addition"
        ),
        "runs": records,
        "table": table,
        "notes": [
            "A reuses the T6 batch-1 results (p1); B uses p1_bin1000; C uses p1_pixel with the independent pixel conversion layer; D uses p1_grid100 with the independent 0-100 integer-grid conversion layer.",
            "All runs n=1000, batch_size=1, independent validation passed; batch8 is permanently forbidden.",
            "B seed42 and C/D seed42 are the already-evaluated runs; the other seeds are the replication runs.",
            "D reporting is neutral: the grid representation is coarser (1% resolution) than A/B by construction.",
        ],
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["variant,dataset,seed,run_id,adapter_model_sha256,n,acc@0.5,mIoU,parse_rate,invalid,validation_status"]
    for variant in variants:
        for dataset in DATASETS:
            for seed, record in sorted(table[variant][dataset]["runs"].items(), key=lambda item: int(item[0])):
                m = record["metrics"]
                lines.append(f"{variant},{dataset},{seed},{record['run_id']},{record['adapter_model_sha256']},1000,{m['acc@0.5']},{m['mIoU']},{m['parse_rate']},{m['invalid']},{record['validation_status']}")
            aggregate = table[variant][dataset]["aggregate"]
            if aggregate["status"].startswith("three_seed"):
                ms = aggregate["metrics"]
                lines.append(f"{variant},{dataset},mean±std,--,--,1000,{ms['acc@0.5']['mean']}±{ms['acc@0.5']['std_ddof1']},{ms['mIoU']['mean']}±{ms['mIoU']['std_ddof1']},{ms['parse_rate']['mean']}±{ms['parse_rate']['std_ddof1']},{ms['invalid']['mean']}±{ms['invalid']['std_ddof1']},--")
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["# G4 · T7 coordinate-representation ablation (A/B/C/D 3-seed)", "",
          "Independent frozen evaluator; n=1000 per dataset per seed; batch_size=1; greedy p1-family prompts.",
          "A (p1), B (p1_bin1000), C (p1_pixel) and D (p1_grid100) report 3-seed mean ± sample std (ddof=1) over seeds 42/43/44.",
          "", "| Variant | Dataset | Acc@0.5 | mIoU | parse_rate | invalid | Seeds |", "|---|---|---|---|---|---|---|"]
    for variant in variants:
        for dataset in DATASETS:
            aggregate = table[variant][dataset]["aggregate"]
            if aggregate["status"].startswith("three_seed"):
                ms = aggregate["metrics"]
                md.append(f"| {variant} | {dataset} | {ms['acc@0.5']['mean']:.6f} ± {ms['acc@0.5']['std_ddof1']:.6f} | {ms['mIoU']['mean']:.6f} ± {ms['mIoU']['std_ddof1']:.6f} | {ms['parse_rate']['mean']:.6f} ± {ms['parse_rate']['std_ddof1']:.6f} | {ms['invalid']['mean']:.3f} ± {ms['invalid']['std_ddof1']:.3f} | 42/43/44 |")
            else:
                m = aggregate["metrics"]
                md.append(f"| {variant} | {dataset} | {m['acc@0.5']:.6f} | {m['mIoU']:.6f} | {m['parse_rate']:.6f} | {m['invalid']} | 42 only |")
    md.extend(["", "Per-seed runs and hashes: `results/eval_t7_g4_summary_3seed.json`; CSV: `report/tables/table_g4_coordinate_variants_3seed.csv`.",
               "", "D note: the 0-100 grid caps achievable IoU by construction (1% resolution); D numbers are reported neutrally, not as a claim of improvement."])
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.summary.relative_to(PROJECT_ROOT)), "csv": str(args.csv.relative_to(PROJECT_ROOT)),
                      "markdown": str(args.markdown.relative_to(PROJECT_ROOT)),
                      "aggregates": {variant: {dataset: table[variant][dataset]["aggregate"]["metrics"] for dataset in DATASETS} for variant in variants}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
