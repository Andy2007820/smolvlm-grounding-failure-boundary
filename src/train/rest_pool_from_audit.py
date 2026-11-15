"""从 ReST 采样的 audit 文件离线重切数据池（CPU-only，不占 GPU）。

为什么要有这一步：采样是最贵的一环（1,000 prompt × K=4 ≈ 4,000 次生成）。audit 文件把
**每一次生成的 IoU 都记下来了**，所以 τ 的选择、τ 的敏感性扫描都不必重采样 ——
只要在 CPU 上重切一遍。这让「一次采样」同时服务：

* r1 数据池（τ=0.5，与 Acc@0.5 同判据）
* τ 敏感性小表（0.1/0.25/0.3/0.5 各有多少条），论文里作为「验证器严格度」的免费对照
* 若 τ=0.5 池子过小，直接降到 τ=0.25 再切，不需要任何 GPU

输出池是 trainer 可读的八字段格式（见 ``eval.dataset.REQUIRED_KEYS`` 与
``sft_data.build_sft_example``）。``raw_bbox`` 由预测框按图像宽高换算成像素坐标，
不取整、不抄 GT —— 它是预测框的像素表示，必须与归一化 ``answer`` 自洽。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT, project_relative, resolve_project_path
from ..eval.dataset import validate_record

DATA_ROOT = (PROJECT_ROOT / "data").resolve()
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
DEFAULT_SWEEP = (0.1, 0.25, 0.3, 0.5, 0.75)


def now_iso() -> str:
    import datetime

    return datetime.datetime.now().astimezone().isoformat()


def sha256_file_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def passes(record: dict[str, Any], *, tau: float) -> bool:
    if record.get("predicted_box") is None or record.get("parse_error"):
        return False
    iou = record.get("iou")
    return isinstance(iou, (int, float)) and float(iou) >= float(tau)


def to_pool_row(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """池行必须由**训练集原始记录**回填 image/question/width/height/source。

    width/height/source 不是装饰：``train_sft.py`` 的视觉 token 预检读 width/height，
    split 校验读 source，缺任何一项训练会在加载阶段崩掉。
    """

    missing = [name for name in ("image", "question", "width", "height", "source") if name not in source]
    if missing:
        raise SystemExit(f"source record {source.get('id')!r} lacks fields: {missing}")
    answer = [float(value) for value in record["predicted_box"]]
    width = int(source["width"])
    height = int(source["height"])
    # raw_bbox is the predicted box expressed in pixel coordinates. Do NOT round:
    # dataset.py re-derives answer = raw_bbox / (width, height) and requires a
    # 1.1e-8 agreement, so integer rounding would fail the contract. The GT
    # raw_bbox cannot be reused here: the supervised answer is the prediction,
    # and substituting GT would both break consistency and leak GT into the pool.
    raw_bbox = [answer[0] * width, answer[1] * height, answer[2] * width, answer[3] * height]
    return {
        "id": record["id"],
        "image": source["image"],
        "question": source["question"],
        "answer": answer,
        "raw_bbox": raw_bbox,
        "width": width,
        "height": height,
        # The frozen loader requires every row's source to equal the config's
        # data.train[*].source. configs/rest_r1.json declares source="rest" for the
        # mixed pool, so the pipeline label wins over the origin dataset name;
        # per-dataset provenance stays in the audit file and manifest by_dataset.
        "source": "rest",
    }


def dataset_counts(records: Iterable[dict[str, Any]], *, tau: float) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, int]] = {}
    prompts: dict[str, set[str]] = {}
    for record in records:
        name = record.get("dataset", "unknown")
        bucket = per_dataset.setdefault(name, {"generations": 0, "accepted": 0})
        bucket["generations"] += 1
        if passes(record, tau=tau):
            bucket["accepted"] += 1
            prompts.setdefault(name, set()).add(record.get("source_id", ""))
    for name, bucket in per_dataset.items():
        bucket["survival_rate"] = bucket["accepted"] / bucket["generations"] if bucket["generations"] else 0.0
        bucket["prompts_with_accept"] = len(prompts.get(name, set()))
    return per_dataset


def load_source_index(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"source train file missing: {path}")
        for record in read_jsonl(path):
            index[str(record["id"])] = record
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--output-pool", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--sweep", default=",".join(str(value) for value in DEFAULT_SWEEP))
    parser.add_argument(
        "--source-data",
        default="data/synthetic_train.jsonl,data/refcoco_train.jsonl",
        help="用训练集原始记录回填 image/question（逗号分隔）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印 τ 扫描与统计，不写任何文件")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit_path = resolve_project_path(args.audit)
    pool_path = resolve_project_path(args.output_pool)
    manifest_path = resolve_project_path(args.output_manifest)
    if pool_path.parent != (DATA_ROOT / "rest").resolve():
        raise SystemExit(f"pool must be directly under data/rest/: {pool_path}")
    if args.tag not in pool_path.name:
        raise SystemExit(f"pool filename must contain tag {args.tag!r}: {pool_path}")
    if manifest_path.parent != RESULTS_ROOT or not manifest_path.name.startswith("train_"):
        raise SystemExit(f"manifest must be under results/ and start with 'train_': {manifest_path}")

    records = read_jsonl(audit_path)
    if not records:
        raise SystemExit(f"audit file is empty: {audit_path}")

    sweep_values = [float(value) for value in args.sweep.split(",") if value.strip()]
    sweep = {
        str(tau): {
            "pool_records": sum(1 for record in records if passes(record, tau=tau)),
            "by_dataset": {
                name: bucket["accepted"] for name, bucket in dataset_counts(records, tau=tau).items()
            },
        }
        for tau in sweep_values
    }
    selected = [record for record in records if passes(record, tau=args.tau)]
    counters = dataset_counts(records, tau=args.tau)
    summary = {
        "audit": project_relative(audit_path),
        "audit_sha256": sha256_file_hex(audit_path),
        "audit_rows": len(records),
        "tau": args.tau,
        "tau_sweep": sweep,
        "pool_records": len(selected),
        "by_dataset": counters,
        "prompts_total": len({record.get("source_id") for record in records}),
        "prompts_with_accept": len({record.get("source_id") for record in selected}),
        "generated_at": now_iso(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if pool_path.exists():
        raise SystemExit(f"refusing to overwrite existing pool: {pool_path}")
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {manifest_path}")

    pool_path.parent.mkdir(parents=True, exist_ok=True)
    source_index = load_source_index(
        [resolve_project_path(item.strip()) for item in args.source_data.split(",") if item.strip()]
    )
    missing = [record["id"] for record in selected if str(record.get("source_id")) not in source_index]
    if missing:
        raise SystemExit(f"{len(missing)} accepted rows have no source record, e.g. {missing[:3]}")

    pool_rows = [to_pool_row(record, source_index[str(record["source_id"])]) for record in selected]

    # Prove every produced row satisfies the frozen eval/train JSONL contract
    # (eight fields, answer == raw_bbox / (width, height), images present) before
    # any file is written. expected_source is checked per row.
    validation = {
        "validator": "src.eval.dataset.validate_record",
        "check_image": True,
        "rows": len(pool_rows),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }
    for row in pool_rows:
        try:
            validate_record(row, project_root=PROJECT_ROOT, expected_source=row["source"], check_image=True)
        except Exception as exc:  # noqa: BLE001 - report every failure verbatim
            validation["failures"].append({"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        else:
            validation["passed"] += 1
    validation["failed"] = len(validation["failures"])
    validation["all_passed"] = validation["failed"] == 0
    if not validation["all_passed"]:
        raise SystemExit(
            f"pool schema validation failed for {validation['failed']} rows; first: {validation['failures'][0]}"
        )

    with pool_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in pool_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "completed",
        "stage": "rest_pool_from_audit",
        "tag": args.tag,
        "pool": {
            "path": project_relative(pool_path),
            "records": len(pool_rows),
            "sha256": sha256_file_hex(pool_path),
        },
        "pool_schema": {
            "fields": 8,
            "raw_bbox": "derived_from_predicted_box_pixels_float",
            "source": "rest (pipeline label required by configs/rest_r1.json data.train; origin datasets kept in by_dataset/audit)",
        },
        "pool_validation": validation,
        "design_deviation": [
            {
                "field": "raw_bbox",
                "rule": "raw_bbox = predicted_box * [width, height] as float pixel coordinates",
                "reason": (
                    "ReST pool rows are model predictions, not GT. The frozen contract requires "
                    "answer == raw_bbox / (width, height); copying the GT raw_bbox would conflict with "
                    "the supervised prediction and leak GT into the pool."
                ),
            }
        ],
        "note": (
            "Pool rebuilt offline from a sampling audit; no GPU used. Records are the model's own "
            "predictions (normalised box) that passed the tau filter. Trainer-ready eight-field rows."
        ),
        **summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
