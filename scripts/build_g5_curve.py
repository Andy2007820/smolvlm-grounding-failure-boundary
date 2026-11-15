#!/usr/bin/env python3
"""Build the G5 three-point ReST curve: initial SFT -> ReST round 1 -> round 2.

Reads only finished artefacts (never runs a model):
  * results/zeroshot_sft_v1_s42_{dataset}_v1.json            (initial SFT point)
  * results/zeroshot_rest_r1_s42_{dataset}_v1.json           (round 1)
  * results/zeroshot_rest_r2_s42_{dataset}_v1.json           (round 2)
  * results/train_rest_sample_r{1,2}.json, results/train_rest_pool_r{1,2}.json
plus the matching *_validation.json status of every point.

Writes results/g5_rest_curve.json.  The curve is the G5 evidence:  "G5 过"
is decided by this script's ``gate`` block, and every claim it makes is
single-seed pilot evidence (seed 42) — never a significance claim.

Usage: python -m scripts.build_g5_curve [--no-write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("synthetic", "refcoco")
STAGES = (
    ("sft_s42", "初始 SFT（seed42 基线）", "results/zeroshot_sft_v1_s42_{dataset}_v1.json"),
    ("rest_r1", "ReST 第 1 轮", "results/zeroshot_rest_r1_s42_{dataset}_v1.json"),
    ("rest_r2", "ReST 第 2 轮", "results/zeroshot_rest_r2_s42_{dataset}_v1.json"),
)
METRICS = ("acc@0.5", "mIoU", "parse_rate", "invalid")
SAMPLE_FILES = {
    "rest_r1": ("results/train_rest_sample_r1.json", "results/train_rest_pool_r1.json"),
    "rest_r2": ("results/train_rest_sample_r2.json", "results/train_rest_pool_r2.json"),
}


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_stage_point(root: Path, template: str, dataset: str) -> dict[str, Any] | None:
    summary_path = root / template.format(dataset=dataset)
    if not summary_path.is_file():
        return None
    summary = _read(summary_path)
    # validation files are named eval_<run>_validation.json while summaries are
    # zeroshot_<run>.json (same run id, different prefix + suffix).
    validation_name = summary_path.name
    if validation_name.startswith("zeroshot_"):
        validation_name = "eval_" + validation_name[len("zeroshot_") :]
    validation_path = summary_path.with_name(validation_name.replace(".json", "_validation.json"))
    validation: dict[str, Any] | None = None
    if validation_path.is_file():
        validation = _read(validation_path)
    metrics = dict(summary.get("metrics") or {})
    point: dict[str, Any] = {
        "summary": summary_path.relative_to(root).as_posix(),
        "run_id": summary.get("run_id"),
        "n": summary.get("n"),
        "status": summary.get("status"),
        "metrics": {key: metrics.get(key) for key in METRICS},
        "validation": None,
    }
    if validation is not None:
        point["validation"] = {
            "path": validation_path.relative_to(root).as_posix(),
            "status": validation.get("status"),
            "errors": validation.get("errors"),
            "n": validation.get("n"),
        }
    return point


def _survival(root: Path, round_tag: str) -> dict[str, Any] | None:
    sample_rel, pool_rel = SAMPLE_FILES[round_tag]
    sample_path, pool_path = root / sample_rel, root / pool_rel
    if not sample_path.is_file():
        return None
    sample = _read(sample_path)
    survival = sample.get("survival") or {}
    out: dict[str, Any] = {
        "sample_summary": sample_rel,
        "pool": sample.get("pool") if isinstance(sample.get("pool"), dict) else None,
        "generations": survival.get("generations"),
        "accepted": survival.get("accepted"),
        "survival_rate": survival.get("survival_rate"),
        "prompts_total": survival.get("prompts_total"),
        "prompts_with_accept": survival.get("prompts_with_accept"),
        "accepted_by_dataset": survival.get("accepted_by_dataset"),
        "parse_failures": survival.get("parse_failures"),
        "hit_max_new_tokens": survival.get("hit_max_new_tokens"),
    }
    if pool_path.is_file():
        manifest = _read(pool_path)
        out["pool_manifest"] = pool_rel
        out["pool_manifest_records"] = manifest.get("pool_records")
        out["tau_sweep"] = manifest.get("tau_sweep")
        out["pool_validation"] = manifest.get("pool_validation")
    return out


def build(root: Path) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for stage_id, label, template in STAGES:
        entry: dict[str, Any] = {
            "stage": stage_id,
            "label": label,
            "datasets": {dataset: _load_stage_point(root, template, dataset) for dataset in DATASETS},
        }
        entry["complete"] = all(entry["datasets"][dataset] is not None for dataset in DATASETS)
        stages.append(entry)

    present = [stage for stage in stages if stage["complete"]]

    # Deltas between consecutive complete points.
    for previous, current in zip(present, present[1:]):
        deltas: dict[str, Any] = {}
        for dataset in DATASETS:
            prev_metrics = previous["datasets"][dataset]["metrics"]
            cur_metrics = current["datasets"][dataset]["metrics"]
            dataset_delta: dict[str, Any] = {}
            for key in ("acc@0.5", "mIoU"):
                prev_value, cur_value = prev_metrics.get(key), cur_metrics.get(key)
                if isinstance(prev_value, (int, float)) and isinstance(cur_value, (int, float)):
                    dataset_delta[key] = {
                        "previous": prev_value,
                        "current": cur_value,
                        "delta": cur_value - prev_value,
                    }
            prev_n, cur_n = prev_metrics.get("n"), cur_metrics.get("n")
            prev_acc, cur_acc = prev_metrics.get("acc@0.5"), cur_metrics.get("acc@0.5")
            if isinstance(prev_n, int) and isinstance(cur_n, int) and isinstance(prev_acc, (int, float)) and isinstance(cur_acc, (int, float)):
                dataset_delta["hits"] = {
                    "previous": round(prev_acc * prev_n),
                    "current": round(cur_acc * cur_n),
                    "delta": round(cur_acc * cur_n) - round(prev_acc * prev_n),
                }
            deltas[dataset] = dataset_delta
        current["delta_vs_previous"] = deltas

    # Also delta vs the initial SFT point for every round.
    baseline = present[0] if present else None
    if baseline is not None:
        for stage in present[1:]:
            deltas_vs_baseline: dict[str, Any] = {}
            for dataset in DATASETS:
                base_metrics = baseline["datasets"][dataset]["metrics"]
                cur_metrics = stage["datasets"][dataset]["metrics"]
                dataset_delta: dict[str, Any] = {}
                for key in ("acc@0.5", "mIoU"):
                    base_value, cur_value = base_metrics.get(key), cur_metrics.get(key)
                    if isinstance(base_value, (int, float)) and isinstance(cur_value, (int, float)):
                        dataset_delta[key] = {"baseline": base_value, "current": cur_value, "delta": cur_value - base_value}
                deltas_vs_baseline[dataset] = dataset_delta
            stage["delta_vs_baseline"] = deltas_vs_baseline

    # Gate evaluation (objective, no significance wording anywhere).
    complete_r2 = any(stage["stage"] == "rest_r2" and stage["complete"] for stage in stages)
    validations_ok = True
    for stage in present:
        for dataset in DATASETS:
            validation = stage["datasets"][dataset].get("validation") or {}
            if validation.get("status") != "passed" or (validation.get("errors") or []) != []:
                validations_ok = False

    verdict = None
    if complete_r2:
        r1 = next(stage for stage in stages if stage["stage"] == "rest_r1" and stage["complete"])
        r2 = next(stage for stage in stages if stage["stage"] == "rest_r2" and stage["complete"])
        signs: list[int] = []
        detail: dict[str, Any] = {}
        for dataset in DATASETS:
            for key in ("acc@0.5", "mIoU"):
                a = r1["datasets"][dataset]["metrics"].get(key)
                b = r2["datasets"][dataset]["metrics"].get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    sign = 1 if b > a else (-1 if b < a else 0)
                    signs.append(sign)
                    detail[f"{dataset}.{key}"] = {"r1": a, "r2": b, "delta": b - a, "sign": sign}
        if all(sign >= 0 for sign in signs) and any(sign > 0 for sign in signs):
            verdict = "两集一致非降（至少一项上升）"
        elif all(sign <= 0 for sign in signs) and any(sign < 0 for sign in signs):
            verdict = "两集一致下降（负结果，照实报）"
        elif all(sign == 0 for sign in signs):
            verdict = "持平（负结果，照实报）"
        else:
            verdict = "混合（有升有降，按数据集逐个报）"
    else:
        detail = {}

    gate = {
        "g5_requires_three_points": True,
        "points_present": [stage["stage"] for stage in present],
        "r2_present": complete_r2,
        "all_validations_passed": validations_ok,
        "verdict_r1_to_r2": verdict,
        "verdict_detail": detail,
        "g5_passed": bool(complete_r2 and validations_ok),
        "significance_claim": False,
        "scope": "single-seed pilot (seed 42); never a 3-seed main-table row and never a significance claim",
    }

    return {
        "schema": "g5_rest_curve/v1",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage_order": [stage_id for stage_id, _, _ in STAGES],
        "datasets": list(DATASETS),
        "metric_keys": list(METRICS),
        "stages": stages,
        "survival": {tag: _survival(root, tag) for tag in ("rest_r1", "rest_r2")},
        "gate": gate,
        "notes": [
            "三点曲线 = 初始 SFT(seed42) -> ReST 第1轮 -> ReST 第2轮；每点均来自官方评测器 + validation，非训练内 probe。",
            "single-seed pilot：所有差值均不得表述为“ReST 有效/显著提升”。",
            "存活率与 τ sweep 来自采样 summary / 池 manifest，仅描述数据池构造，不是精度指标。",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 G5 三点 ReST 曲线")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    curve = build(root)
    output_path = root / "results/g5_rest_curve.json"
    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "output": None if args.no_write else output_path.relative_to(root).as_posix(),
        "points": curve["gate"]["points_present"],
        "r2_present": curve["gate"]["r2_present"],
        "all_validations_passed": curve["gate"]["all_validations_passed"],
        "verdict_r1_to_r2": curve["gate"]["verdict_r1_to_r2"],
        "g5_passed": curve["gate"]["g5_passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
