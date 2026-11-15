#!/usr/bin/env python3
"""Build the G6 KD-vs-SFT comparison: initial SFT (seed42) -> KD (T12).

Reads only finished artefacts (never runs a model):
  * results/zeroshot_sft_v1_s42_{dataset}_v1.json     (baseline, reused not re-run)
  * results/zeroshot_kd_v1_s42_{dataset}_v1.json      (KD run)
plus validation status, the teacher summary (results/train_kd_teacher_v1.json) and
the KD training summary (results/train_kd_v1_s42_training.json).

Writes results/g6_kd_vs_sft.json.  Both runs are single-seed pilot (seed 42); the
script never emits a significance claim.

Usage: python -m scripts.build_g6_curve [--no-write]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("synthetic", "refcoco")
STAGES = (
    ("sft_s42", "初始 SFT（seed42 基线，复用不重跑）", "results/zeroshot_sft_v1_s42_{dataset}_v1.json"),
    ("kd_v1", "KD（3B 教师响应蒸馏，seed42）", "results/zeroshot_kd_v1_s42_{dataset}_v1.json"),
)
METRICS = ("acc@0.5", "mIoU", "parse_rate", "invalid")
KD_TRAINING = "results/train_kd_v1_s42_training.json"
TEACHER_SUMMARY = "results/train_kd_teacher_v1.json"


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_point(root: Path, template: str, dataset: str) -> dict[str, Any] | None:
    summary_path = root / template.format(dataset=dataset)
    if not summary_path.is_file():
        return None
    summary = _read(summary_path)
    validation_name = summary_path.name
    if validation_name.startswith("zeroshot_"):
        validation_name = "eval_" + validation_name[len("zeroshot_") :]
    validation_path = summary_path.with_name(validation_name.replace(".json", "_validation.json"))
    metrics = dict(summary.get("metrics") or {})
    point: dict[str, Any] = {
        "summary": summary_path.relative_to(root).as_posix(),
        "run_id": summary.get("run_id"),
        "n": summary.get("n"),
        "status": summary.get("status"),
        "metrics": {key: metrics.get(key) for key in METRICS},
        "validation": None,
    }
    if validation_path.is_file():
        validation = _read(validation_path)
        point["validation"] = {
            "path": validation_path.relative_to(root).as_posix(),
            "status": validation.get("status"),
            "errors": validation.get("errors"),
            "n": validation.get("n"),
        }
    return point


def _extra(root: Path, rel: str) -> dict[str, Any] | None:
    path = root / rel
    if not path.is_file():
        return None
    payload = _read(path)
    return payload


def build(root: Path) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for stage_id, label, template in STAGES:
        entry: dict[str, Any] = {
            "stage": stage_id,
            "label": label,
            "datasets": {dataset: _load_point(root, template, dataset) for dataset in DATASETS},
        }
        entry["complete"] = all(entry["datasets"][dataset] is not None for dataset in DATASETS)
        stages.append(entry)

    baseline, kd = stages
    validations_ok = True
    for stage in stages:
        for dataset in DATASETS:
            point = stage["datasets"][dataset]
            if point is None:
                validations_ok = False
                continue
            validation = point.get("validation") or {}
            if validation.get("status") != "passed" or (validation.get("errors") or []) != []:
                validations_ok = False

    comparison: dict[str, Any] = {}
    signs: list[int] = []
    if baseline["complete"] and kd["complete"]:
        for dataset in DATASETS:
            base_metrics = baseline["datasets"][dataset]["metrics"]
            kd_metrics = kd["datasets"][dataset]["metrics"]
            dataset_comparison: dict[str, Any] = {}
            for key in ("acc@0.5", "mIoU"):
                a, b = base_metrics.get(key), kd_metrics.get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    sign = 1 if b > a else (-1 if b < a else 0)
                    signs.append(sign)
                    dataset_comparison[key] = {"sft": a, "kd": b, "delta": b - a, "sign": sign}
            n = kd_metrics.get("n") or 1000
            acc_pairs = (base_metrics.get("acc@0.5"), kd_metrics.get("acc@0.5"))
            if all(isinstance(value, (int, float)) for value in acc_pairs):
                dataset_comparison["hits"] = {
                    "sft": round(acc_pairs[0] * n),
                    "kd": round(acc_pairs[1] * n),
                    "delta": round(acc_pairs[1] * n) - round(acc_pairs[0] * n),
                }
            comparison[dataset] = dataset_comparison

    if baseline["complete"] and kd["complete"]:
        if all(sign >= 0 for sign in signs) and any(sign > 0 for sign in signs):
            verdict = "两集一致非降（至少一项上升）"
        elif all(sign <= 0 for sign in signs) and any(sign < 0 for sign in signs):
            verdict = "两集一致下降（负结果，照实报）"
        elif all(sign == 0 for sign in signs):
            verdict = "持平（负结果，照实报）"
        else:
            verdict = "混合（有升有降，按数据集逐个报）"
    else:
        verdict = None

    teacher = _extra(root, TEACHER_SUMMARY)
    training = _extra(root, KD_TRAINING)
    gate = {
        "g6_requires_kd_run_and_two_datasets": True,
        "stage_order": [stage_id for stage_id, _, _ in STAGES],
        "points_present": [stage["stage"] for stage in stages if stage["complete"]],
        "kd_present": kd["complete"],
        "all_validations_passed": validations_ok,
        "verdict_kd_vs_sft": verdict,
        "comparison": comparison,
        "g6_passed": bool(kd["complete"] and validations_ok),
        "significance_claim": False,
        "scope": "single-seed pilot (seed 42); never a significance claim",
        "distillation_kind": "response distillation (teacher text answer -> student SFT); NOT logit KD (different vocabularies, paper §6.5)",
    }

    return {
        "schema": "g6_kd_vs_sft/v1",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": list(DATASETS),
        "metric_keys": list(METRICS),
        "stages": stages,
        "teacher": {
            "path": TEACHER_SUMMARY,
            "prompts_total": (teacher or {}).get("prompts_total"),
            "accepted": (teacher or {}).get("accepted"),
            "survival_rate": (teacher or {}).get("survival_rate"),
            "accepted_by_dataset": (teacher or {}).get("accepted_by_dataset"),
            "tau": (teacher or {}).get("tau"),
            "load_mode": ((teacher or {}).get("teacher") or {}).get("load_mode"),
            "memory": (teacher or {}).get("memory"),
            "data_manifest": (teacher or {}).get("data_manifest"),
            "pool_validation": (teacher or {}).get("pool_validation"),
        },
        "kd_training": {
            "path": KD_TRAINING,
            "train_records": ((training or {}).get("plan") or {}).get("train_records"),
            "steps": ((training or {}).get("plan") or {}).get("total_optimizer_steps"),
            "micro_batches": ((training or {}).get("plan") or {}).get("total_micro_batches"),
            "training_seconds": (training or {}).get("training_seconds"),
            "loss_first": ((training or {}).get("loss_history") or [None])[0],
            "loss_last": ((training or {}).get("loss_history") or [None])[-1],
            "adapter_sha256": (((training or {}).get("final_adapter_summary") or {}).get("adapter_files") or {}),
        },
        "gate": gate,
        "notes": [
            "G6 = KD 与同 seed SFT 基线对比；基线复用不重跑，KD 为 single-seed pilot。",
            "教师 Qwen2.5-VL-3B-Instruct，IoU>=τ 过滤后作为额外 SFT 数据；学生 Qwen 与 SmolVLM 词表不同 → 只能做响应蒸馏。",
            "所有差值均不得表述为“KD 有效/显著提升”。",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 G6 KD vs SFT 对比")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    curve = build(root)
    output_path = root / "results/g6_kd_vs_sft.json"
    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": None if args.no_write else output_path.relative_to(root).as_posix(),
                "points": curve["gate"]["points_present"],
                "kd_present": curve["gate"]["kd_present"],
                "all_validations_passed": curve["gate"]["all_validations_passed"],
                "verdict_kd_vs_sft": curve["gate"]["verdict_kd_vs_sft"],
                "g6_passed": curve["gate"]["g6_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
