#!/usr/bin/env python3
"""T9 round verifier: independent re-check of a ReST round's evaluation artefacts.

CPU only, read-only, no torch model is loaded.  The frozen round-1 verifier
(:mod:`src.eval.verify_rest_r1`) is imported as a library: every *check
implementation* (details counts / metrics re-scoring / details sha256 /
validation report / adapter sha256 / report literals) is reused verbatim, so the
round-N numbers are recomputed with exactly the same semantics as round 1.
Only what is genuinely round-specific is parameterised here:

* input resolution (``results/zeroshot_<tag>_s42_<dataset>_v1.json`` etc.);
* the report-literal scope (which lines of Table 1 / the paper mention this
  round) — the check itself stays frozen;
* one extra check that round 1 got from an external gpu-runner: the training
  hard counters (forward == backward == micro-batches, 1,250 optimizer steps,
  ``stop_reason=max_steps``, complete training summary, adapter bytes/sha).

Usage::

    python src/eval/verify_rest_round.py --tag r2                 # 6 checks, no report literals
    python src/eval/verify_rest_round.py --tag r2 --check-report-literals   # after docs sync
    python src/eval/verify_rest_round.py --tag r2 --no-write     # do not write the report

Exit codes: 0 = passed/pending, 1 = failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

# ``verify_rest_r1`` is a sibling module and imports nothing heavy at module
# level, so it can be imported directly when this file is executed as a script.
for _candidate in (str(_HERE), str(PROJECT_ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import verify_rest_r1 as base  # noqa: E402

EXPECTED_N = 1000
TOLERANCE = 1e-12
DATASETS = ("synthetic", "refcoco")
EXPECTED_STEPS = 1250
EXPECTED_MICRO = 10_000

ROUNDS: dict[str, dict[str, Any]] = {
    "r2": {
        "tag": "r2",
        "run_id_round": "rest_r2",
        "summary_templates": ("results/zeroshot_rest_r2_s42_{dataset}_v1.json",),
        "summary_globs": (
            "results/zeroshot_rest_r2*s42*{dataset}*.json",
            "results/zeroshot_rest_r2*{dataset}*.json",
        ),
        "details_globs": ("results/eval_rest_r2*{dataset}*details.jsonl",),
        "validation_globs": ("results/eval_rest_r2*{dataset}*validation.json",),
        "training_candidates": (
            "results/train_rest_r2_s42_training.json",
            "results/train_rest_r2_training.json",
            "results/train_t9_rest_r2.json",
            "results/train_rest_r2.json",
        ),
        "training_glob": "results/train_*rest_r2*.json",
        "training_exclude": ("probe", "dryrun", "pool", "sample", "manifest"),
        "adapter_candidates": (
            "adapters/rest_r2/adapter/adapter_model.safetensors",
            "adapters/rest_r2/adapter_model.safetensors",
        ),
        "adapter_glob": "adapters/rest_r2/**/adapter_model.safetensors",
        # Lines of Table 1 / the paper that belong to this round.
        "report_re": (
            r"rest_r2|ReST\s*[（(]?\s*第\s*[2２二]\s*轮|ReST[\s\-_]*r2|ReST\s*round\s*2|T9\b"
        ),
        # ReST pools that must appear in the training summary's dataset list.
        "required_pools": ("data/rest/r1_pool.jsonl", "data/rest/r2_pool.jsonl"),
    },
    "kd_v1": {
        "tag": "kd_v1",
        "run_id_round": "kd_v1",
        "summary_templates": ("results/zeroshot_kd_v1_s42_{dataset}_v1.json",),
        "summary_globs": (
            "results/zeroshot_kd_v1*s42*{dataset}*.json",
            "results/zeroshot_kd_v1*{dataset}*.json",
        ),
        "details_globs": ("results/eval_kd_v1*{dataset}*details.jsonl",),
        "validation_globs": ("results/eval_kd_v1*{dataset}*validation.json",),
        "training_candidates": (
            "results/train_kd_v1_s42_training.json",
            "results/train_kd_v1_training.json",
        ),
        "training_glob": "results/train_*kd_v1*.json",
        "training_exclude": ("probe", "dryrun", "pool", "sample", "manifest", "teacher"),
        "adapter_candidates": (
            "adapters/kd_v1/adapter/adapter_model.safetensors",
            "adapters/kd_v1/adapter_model.safetensors",
        ),
        "adapter_glob": "adapters/kd_v1/**/adapter_model.safetensors",
        "report_re": r"kd_v1|\bKD\b|T11|T12|响应蒸馏",
        # KD training summary must list the teacher-generated pool.
        "required_pools": ("data/kd/kd_train.jsonl",),
    },
}


# --------------------------------------------------------------------------- #
# round-specific input resolution (mirrors the frozen round-1 resolver)
# --------------------------------------------------------------------------- #
def resolve_inputs(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    resolution: dict[str, Any] = {"datasets": {}, "training_json": None, "adapter": None, "notes": []}

    for dataset in DATASETS:
        candidates = [root / template.format(dataset=dataset) for template in spec["summary_templates"]]
        for pattern in spec["summary_globs"]:
            candidates.extend(base._glob(root, pattern.format(dataset=dataset)))
        summary = base._first_existing(candidates)

        run_id: str | None = None
        details_path: str | None = None
        if summary is not None:
            try:
                payload = base._read_json(summary)
                run_id = payload.get("run_id") if isinstance(payload.get("run_id"), str) else None
                details_path = payload.get("details_path") if isinstance(payload.get("details_path"), str) else None
            except (OSError, ValueError):
                run_id, details_path = None, None

        detail_candidates: list[Path] = []
        if details_path:
            detail_candidates.append(root / details_path)
        if run_id:
            detail_candidates.append(root / f"results/eval_{run_id}_details.jsonl")
        detail_candidates.append(root / f"results/eval_rest_r2_s42_{dataset}_v1_details.jsonl")
        for pattern in spec["details_globs"]:
            detail_candidates.extend(base._glob(root, pattern.format(dataset=dataset)))
        details = base._first_existing(detail_candidates)

        validation_candidates: list[Path] = []
        if run_id:
            validation_candidates.append(root / f"results/eval_{run_id}_validation.json")
        validation_candidates.append(root / f"results/eval_rest_r2_s42_{dataset}_v1_validation.json")
        for pattern in spec["validation_globs"]:
            validation_candidates.extend(base._glob(root, pattern.format(dataset=dataset)))
        validation = base._first_existing(validation_candidates)

        dataset_missing: list[str] = []
        if summary is None:
            dataset_missing.append(base._rel(root, root / f"results/zeroshot_rest_r2_s42_{dataset}_v1.json"))
        if details is None:
            dataset_missing.append(f"results/eval_rest_r2_s42_{dataset}_v1_details.jsonl")
        if validation is None:
            dataset_missing.append(f"results/eval_rest_r2_s42_{dataset}_v1_validation.json")
        resolution["datasets"][dataset] = {
            "run_id": run_id,
            "summary": summary,
            "details": details,
            "validation": validation,
            "ready": not dataset_missing,
            "missing": dataset_missing,
        }
        missing.extend(dataset_missing)

    training = base._first_existing([root / candidate for candidate in spec["training_candidates"]])
    if training is None:
        globbed = [
            path
            for path in base._glob(root, spec["training_glob"])
            if not any(tag in path.name for tag in spec["training_exclude"])
        ]
        training = base._first_existing(globbed)
    if training is None:
        training = base._first_existing([root / "adapters/rest_r2/training_summary.json"])
    resolution["training_json"] = training
    if training is None:
        missing.append("results/train_rest_r2_s42_training.json")

    adapter: Path | None = None
    if training is not None:
        try:
            training_payload = base._read_json(training)
        except (OSError, ValueError):
            training_payload = None
        if training_payload is not None:
            adapter = base._first_existing(base._adapter_paths_from_training(root, training_payload))
    if adapter is None:
        adapter = base._first_existing([root / candidate for candidate in spec["adapter_candidates"]])
    if adapter is None:
        globbed = [path for path in base._glob(root, spec["adapter_glob"]) if path.is_file()]
        globbed.sort(key=lambda path: ("checkpoint" in path.as_posix().lower(), len(path.as_posix())))
        adapter = base._first_existing(globbed)
    resolution["adapter"] = adapter
    if adapter is None:
        missing.append("adapters/rest_r2/adapter/adapter_model.safetensors")

    resolution["training_ready"] = training is not None
    resolution["adapter_ready"] = adapter is not None
    resolution["missing"] = missing
    return resolution


# --------------------------------------------------------------------------- #
# round-specific extra check: training hard counters + integrity
# --------------------------------------------------------------------------- #
def check_training_hardcounts(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    training_path: Path | None = ctx["resolution"].get("training_json")
    problems: list[str] = []
    evidence: dict[str, Any] = {}

    if training_path is None:
        return base._make_check(
            "training_hardcounts",
            "训练硬计数/完整性（training summary）",
            ["training summary 缺失"],
            {},
        )
    try:
        payload = base._read_json(training_path)
    except (OSError, ValueError) as exc:
        return base._make_check(
            "training_hardcounts", "训练硬计数/完整性（training summary）", [f"读取失败: {exc}"], {}
        )

    stats = payload.get("accumulation_stats") or {}
    plan = payload.get("plan") or {}

    def expect(name: str, observed: Any, expected: Any) -> None:
        evidence[name] = observed
        if observed != expected:
            problems.append(f"{name}: 期望 {expected!r}，实测 {observed!r}")

    # NOTE: the top-level ``complete`` flag is ``false`` in every historical run
    # summary (T6/T7/r1) — it is a closing-bookkeeping flag, not the success
    # marker.  The success marker is ``final_adapter_summary.complete``; the run
    # level facts are ``stop_reason`` + ``partial_final_step``.
    final_adapter = payload.get("final_adapter_summary") or {}
    expect("final_adapter_summary.complete", final_adapter.get("complete"), True)
    evidence["top_level_complete_informational"] = payload.get("complete")
    expect("stop_reason", payload.get("stop_reason"), "max_steps")
    expect("partial_final_step", payload.get("partial_final_step"), False)

    micro_expected = plan.get("total_micro_batches", EXPECTED_MICRO)
    steps_expected = plan.get("total_optimizer_steps", EXPECTED_STEPS)
    evidence["micro_batches_expected"] = micro_expected
    evidence["optimizer_steps_expected"] = steps_expected

    for field in ("micro_batches_seen", "forward_calls", "backward_calls"):
        expect(f"top.{field}", payload.get(field), micro_expected)
    expect("stats.micro_batches_seen", stats.get("micro_batches_seen"), micro_expected)
    expect("stats.forward_calls", stats.get("forward_calls"), micro_expected)
    expect("stats.backward_calls", stats.get("backward_calls"), micro_expected)
    for field in ("optimizer_steps", "clip_calls", "optimizer_step_calls", "scheduler_step_calls", "zero_grad_calls"):
        expect(f"stats.{field}", stats.get(field), steps_expected)
    expect("stats.partial_final_step", stats.get("partial_final_step"), False)
    expect("stats.stop_reason", stats.get("stop_reason"), "max_steps")

    loss_history = payload.get("loss_history") or []
    evidence["loss_history_len"] = len(loss_history)
    if len(loss_history) != steps_expected:
        problems.append(f"loss_history 长度 {len(loss_history)} != {steps_expected}")
    if loss_history:
        try:
            first, last = float(loss_history[0]), float(loss_history[-1])
            evidence["loss_first"], evidence["loss_last"] = first, last
            if not (first == first and last == last):
                problems.append("loss_history 含 NaN")
        except (TypeError, ValueError):
            problems.append("loss_history 元素不是数字")

    seconds = payload.get("training_seconds")
    evidence["training_seconds"] = seconds
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        problems.append(f"training_seconds 非法: {seconds!r}")

    datasets = payload.get("datasets") or []
    evidence["datasets"] = [
        {"path": entry.get("path"), "rows_loaded": entry.get("rows_loaded"), "sha256": (entry.get("sha256") or "")[:12]}
        for entry in datasets
        if isinstance(entry, dict)
    ]
    train_paths = [str(entry.get("path")) for entry in datasets if isinstance(entry, dict)]
    required_pools = ctx.get("spec", {}).get(
        "required_pools", ("data/rest/r1_pool.jsonl", "data/rest/r2_pool.jsonl")
    )
    for required in required_pools:
        if required not in train_paths:
            problems.append(f"训练集缺少 {required}")
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        rows = entry.get("rows_loaded")
        sha = entry.get("sha256")
        if not path_value or not isinstance(rows, int) or not isinstance(sha, str):
            problems.append(f"dataset 条目字段不全: {entry!r}")
            continue
        file_path = root / str(path_value)
        if not file_path.is_file():
            problems.append(f"{path_value} 不存在")
            continue
        actual_rows = sum(1 for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if actual_rows != rows:
            problems.append(f"{path_value} 行数 {actual_rows} != rows_loaded {rows}")
        if base._sha256_file(file_path) != sha:
            problems.append(f"{path_value} sha256 不匹配")

    id_intersection = payload.get("id_intersection")
    evidence["id_intersection"] = id_intersection
    if isinstance(id_intersection, dict):
        pairs = id_intersection.get("pair_id_intersection_counts") or {}
        if not isinstance(pairs, dict) or not pairs:
            problems.append("id_intersection.pair_id_intersection_counts 缺失")
        else:
            offenders = {name: value for name, value in pairs.items() if value != 0}
            if offenders:
                problems.append(f"train/val ID 交集非零: {offenders}")
        for key in ("global_train_val_id_intersection_count", "train_val_image_path_intersection_count"):
            value = id_intersection.get(key)
            evidence[key] = value
            if value not in (0, None):
                problems.append(f"{key} 非零: {value!r}")

    return base._make_check(
        "training_hardcounts",
        "训练硬计数/完整性：forward=backward=micro-batches、1,250 optimizer steps、stop_reason、数据集行数/sha、id 交集",
        problems,
        evidence,
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_verification(
    root: Path,
    tag: str,
    grounding: Any,
    pixel_convert: Any,
    expected_n: int,
    tolerance: float,
    check_report_literals: bool,
) -> dict[str, Any]:
    spec = ROUNDS[tag]
    resolution = resolve_inputs(root, spec)
    ready_datasets = [dataset for dataset in DATASETS if resolution["datasets"][dataset]["ready"]]
    pending_datasets = [dataset for dataset in DATASETS if not resolution["datasets"][dataset]["ready"]]
    all_ready = not pending_datasets
    adapter_ready = bool(resolution.get("training_ready") and resolution.get("adapter_ready"))

    report: dict[str, Any] = {
        "run_id": f"verify_rest_{tag}",
        "script": f"src/eval/verify_rest_round.py --tag {tag}",
        "generated_at": base._now(),
        "expected_n": expected_n,
        "tolerance": tolerance,
        "grounding_module": getattr(grounding, "__file__", None),
        "verifier_basis": "src/eval/verify_rest_r1.py (frozen check implementations, reused as a library)",
        "inputs": {
            "datasets": {
                dataset: {
                    "run_id": entry["run_id"],
                    "summary": base._rel(root, entry["summary"]) if entry["summary"] else None,
                    "details": base._rel(root, entry["details"]) if entry["details"] else None,
                    "validation": base._rel(root, entry["validation"]) if entry["validation"] else None,
                }
                for dataset, entry in resolution["datasets"].items()
            },
            "training_json": base._rel(root, resolution["training_json"]) if resolution["training_json"] else None,
            "adapter": base._rel(root, resolution["adapter"]) if resolution["adapter"] else None,
        },
        "datasets": {
            dataset: {
                "status": "checked" if entry["ready"] else "pending",
                "run_id": entry["run_id"],
                "missing": entry["missing"],
            }
            for dataset, entry in resolution["datasets"].items()
        },
        "pending": {"datasets": pending_datasets, "items": [], "missing_paths": resolution["missing"]},
        "deferred_checks": [],
    }

    if not ready_datasets:
        report.update(
            {
                "status": "pending",
                "missing": resolution["missing"],
                "reason": "真实评测/训练产物尚未落盘；脚本按 pending 语义退出（exit 0），不做任何检查",
                "checks": [],
            }
        )
        return report

    ctx: dict[str, Any] = {
        "root": root,
        "grounding": grounding,
        "pixel_convert": pixel_convert,
        "expected_n": expected_n,
        "tolerance": tolerance,
        "resolution": resolution,
        "datasets": ready_datasets,
        "spec": spec,
        "_cache": {},
    }

    builders = [
        base.check_details_counts,
        base.check_metrics,
        base.check_details_sha256,
        base.check_validation_report,
        check_training_hardcounts,
    ]
    if adapter_ready:
        builders.append(base.check_adapter_sha256)
    else:
        report["deferred_checks"].append("adapter_sha256：training JSON 或 adapter_model.safetensors 缺失")
        report["pending"]["items"].append("adapter_sha256")

    if check_report_literals and all_ready and adapter_ready:
        builders.append(base.check_report_literals)
    else:
        report["deferred_checks"].append(
            "report_literals：本轮未启用（需要先同步 Table 1 / 论文后加 --check-report-literals）"
            if not check_report_literals
            else "report_literals：产物未全部就绪，推迟"
        )
        report["pending"]["items"].append("report_literals")

    checks: list[dict[str, Any]] = []
    original_relevant = base.RELEVANT_RE
    if check_report_literals:
        base.RELEVANT_RE = re.compile(spec["report_re"], re.IGNORECASE)
    try:
        for builder in builders:
            try:
                checks.append(builder(ctx))
            except Exception as exc:  # keep verifying the remaining items
                checks.append(
                    {
                        "id": builder.__name__.removeprefix("check_"),
                        "description": "检查执行时抛出异常",
                        "status": "failed",
                        "problems": [f"{type(exc).__name__}: {exc}"],
                        "details": {},
                    }
                )
    finally:
        base.RELEVANT_RE = original_relevant

    failures = [check for check in checks if check["status"] == "failed"]
    if failures:
        status = "failed"
    elif not all_ready or not adapter_ready:
        status = "partial"
    else:
        status = "passed"
    report.update(
        {
            "status": status,
            "missing": resolution["missing"],
            "checks": checks,
            "failed_checks": [check["id"] for check in failures],
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T9 各轮独立复核（CPU、只读、不加载 torch 模型）")
    parser.add_argument("--tag", required=True, choices=sorted(ROUNDS), help="轮次标签")
    parser.add_argument("--root", default=None, help="项目根目录（默认按脚本位置推断）")
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--check-report-literals", action="store_true", help="同时核对 Table 1 / 论文里的本轮数字")
    parser.add_argument("--no-write", action="store_true", help="不写 results/verify_rest_<tag>_report.json")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    src_root = root if (root / "src/eval/grounding.py").is_file() else PROJECT_ROOT
    src_dir = str(src_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from eval import grounding  # noqa: PLC0415
        from eval import pixel_convert  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"status": "error", "error": f"无法导入 src/eval/grounding.py: {exc}"}, ensure_ascii=False))
        return 2

    try:
        report = run_verification(
            root,
            args.tag,
            grounding,
            pixel_convert,
            expected_n=args.expected_n,
            tolerance=args.tolerance,
            check_report_literals=args.check_report_literals,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    output_path = root / f"results/verify_rest_{args.tag}_report.json"
    base._write_report(report, output_path, args.no_write)

    payload = {
        "status": report["status"],
        "tag": args.tag,
        "datasets": {dataset: info["status"] for dataset, info in report.get("datasets", {}).items()},
        "checks": {check["id"]: check["status"] for check in report.get("checks", [])},
        "failed_checks": report.get("failed_checks", []),
        "pending": report.get("pending", {}).get("items", []),
        "deferred_checks": report.get("deferred_checks", []),
        "problems": {
            check["id"]: check["problems"] for check in report.get("checks", []) if check["status"] == "failed"
        },
        "report_path": base._rel(root, output_path) if not args.no_write else None,
        "generated_at": report["generated_at"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
