#!/usr/bin/env python3
"""T8 refactor: independent verifier for the ReST round-1 (rest_r1) evaluation artefacts.

CPU only, read-only, no torch model is loaded.  The script reuses the frozen
parser/scorer in :mod:`src.eval.grounding` so that its numbers are recomputed
with exactly the same semantics as the evaluation pipeline, then compares them
against what the result files claim.  For rows produced by the ``p1_pixel``
prompt variant the verifier dispatches to the frozen
:mod:`src.eval.pixel_convert` scorer, exactly like ``run_zeroshot.py`` does;
all other rows use ``grounding.score_prediction``.

Checked items (all required for ``status=passed``):

1. ``details`` row count == expected_n (default 1000) for both datasets, and the
   same ``n`` appears in the summary / validation JSON.
2. Every details row is re-scored from ``output_text`` + ``ground_truth`` and the
   per-row ``predicted_box`` / ``parse_error`` / ``iou`` / ``correct@0.5`` must
   match; the aggregate n / parse_rate / invalid / acc@0.5 / mIoU must match the
   summary ``metrics`` (tolerance 1e-12 by default).
3. ``validation`` report's ``details_sha256`` equals the real sha256 of the
   details file, and the validation report itself is ``status=passed`` with no
   errors.
4. sha256 of the trained ``adapter_model.safetensors`` equals the sha recorded in
   the training JSON (``train_rest_r1_training.json`` and friends), and also the
   adapter sha embedded in the evaluation summaries when present.
5. Numbers that appear in ``report/tables/table1_draft.md`` and
   ``report/paper.md`` next to rest_r1 mentions must be literal-identical to the
   JSON values (no rounding drift).  Short exact metrics (acc@0.5, parse_rate,
   invalid, n) are accepted when numerically equal (zero-padding like ``0.020``
   for ``0.02`` is not a drift); mIoU must match the JSON literal character for
   character.
6. Everything above is only attempted when the inputs exist.  While the run is
   still in progress the script prints ``{"status": "pending", "missing": [...]}``
   and exits 0 instead of crashing.  When only some datasets are complete the
   report status is ``partial``: the finished inputs are fully verified, the
   missing dataset(s) are marked ``pending``, and report-literal checking is
   deferred to the final full run.

Exit codes: 0 = pending / partial / passed; 1 = failed.

Usage::

    python src/eval/verify_rest_r1.py
    python src/eval/verify_rest_r1.py --root <fixture-dir> --no-write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_N = 1000
TOLERANCE = 1e-12
DATASETS = ("synthetic", "refcoco")
REPORT_FILES = ("report/tables/table1_draft.md", "report/paper.md")

TRAINING_JSON_CANDIDATES = (
    "results/train_rest_r1_training.json",
    "results/train_rest_r1_s42_training.json",
    "results/train_t8_rest_r1.json",
    "results/train_rest_r1.json",
)
ADAPTER_CANDIDATES = (
    "adapters/rest_r1/adapter/adapter_model.safetensors",
    "adapters/rest_r1/adapter_model.safetensors",
)
# Metrics whose JSON literals live in the reports.  ``n``/``parse_rate``/``invalid``
# are checked for presence too, but only acc@0.5 and mIoU are hard requirements.
METRIC_KEYS = ("n", "acc@0.5", "mIoU", "parse_rate", "invalid")
REQUIRED_PRESENCE = ("acc@0.5", "mIoU")

RELEVANT_RE = re.compile(
    r"rest_r1|\bReST\b|ReST\s*[（(]?\s*第\s*[1１一]\s*轮|ReST[\s\-_]*r1|ReST\s*round\s*1|T8\b",
    re.IGNORECASE,
)
# Remove threshold notation before number tokenisation ("Acc@0.5", "IoU>=0.5", ...).
STRIP_RE = re.compile(r"Acc@0\.5|@\s*0\.5|(?:>=|<=|≥|≤|>|<|＝|=)\s*0\.5", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?![A-Za-z0-9_])"
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_details_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _parse_details(data: bytes, path: Path) -> list[dict[str, Any]]:
    text = data.decode("utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: 第 {lineno} 行不是合法 JSON（文件可能仍在写入）: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: 第 {lineno} 行不是 JSON 对象")
        rows.append(value)
    return rows


def _first_existing(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _glob(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(root.glob(pattern))
    except OSError:
        return []


def _metric_literal(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return json.dumps(value)  # shortest repr that round-trips, same as JSON
    return str(value)


def _is_sha256(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", text))


# --------------------------------------------------------------------------- #
# input resolution (must never crash while the run is still in progress)
# --------------------------------------------------------------------------- #
def _expected_details_name(run_id: str | None, dataset: str) -> str:
    if run_id:
        return f"results/eval_{run_id}_details.jsonl"
    return f"results/eval_rest_r1_s42_{dataset}_v1_details.jsonl"


def _expected_validation_name(run_id: str | None, dataset: str) -> str:
    if run_id:
        return f"results/eval_{run_id}_validation.json"
    return f"results/eval_rest_r1_s42_{dataset}_v1_validation.json"


def _adapter_paths_from_training(root: Path, training: Any) -> list[Path]:
    strings: list[tuple[tuple[str, ...], str]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    strings.append((path + (str(key),), value))
                else:
                    walk(value, path + (str(key),))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(training, ())
    candidates: list[Path] = []
    for key_path, value in strings:
        tail = key_path[-1].lower() if key_path else ""
        joined = ".".join(key_path).lower()
        looks_like_adapter_dir = (
            tail in {"dir", "output_dir", "adapter_path", "adapter_dir"}
            and ("adapter" in joined or "output" in joined)
        ) or (tail in {"path", "adapter"} and "adapter" in joined)
        looks_like_adapter_file = value.lower().endswith("adapter_model.safetensors")
        if looks_like_adapter_file:
            candidates.append(root / value)
        if looks_like_adapter_dir:
            base = root / value
            candidates.append(base / "adapter" / "adapter_model.safetensors")
            candidates.append(base / "adapter_model.safetensors")
            candidates.append(base)
    return candidates


def resolve_inputs(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    resolution: dict[str, Any] = {"datasets": {}, "training_json": None, "adapter": None, "notes": []}

    for dataset in DATASETS:
        primary = root / f"results/zeroshot_rest_r1_s42_{dataset}_v1.json"
        summary = _first_existing(
            [primary]
            + _glob(root, f"results/zeroshot_rest_r1*s42*{dataset}*.json")
            + _glob(root, f"results/zeroshot_rest_r1*{dataset}*.json")
        )
        run_id: str | None = None
        details_path: str | None = None
        if summary is not None:
            try:
                payload = _read_json(summary)
                run_id = payload.get("run_id") if isinstance(payload.get("run_id"), str) else None
                details_path = payload.get("details_path") if isinstance(payload.get("details_path"), str) else None
            except (OSError, ValueError):
                run_id, details_path = None, None

        detail_candidates: list[Path] = []
        if details_path:
            detail_candidates.append(root / details_path)
        if run_id:
            detail_candidates.append(root / f"results/eval_{run_id}_details.jsonl")
        detail_candidates.extend(
            [
                root / f"results/eval_rest_r1_s42_{dataset}_v1_details.jsonl",
                root / f"results/eval_rest_r1_{dataset}_v1_details.jsonl",
            ]
        )
        detail_candidates.extend(_glob(root, f"results/eval_rest_r1*{dataset}*details.jsonl"))
        details = _first_existing(detail_candidates)

        validation_candidates: list[Path] = []
        if run_id:
            validation_candidates.append(root / f"results/eval_{run_id}_validation.json")
        validation_candidates.extend(
            [
                root / f"results/eval_rest_r1_s42_{dataset}_v1_validation.json",
                root / f"results/eval_rest_r1_{dataset}_v1_validation.json",
            ]
        )
        validation_candidates.extend(_glob(root, f"results/eval_rest_r1*{dataset}*validation.json"))
        validation = _first_existing(validation_candidates)

        dataset_missing: list[str] = []
        if summary is None:
            dataset_missing.append(_rel(root, primary))
        if details is None:
            dataset_missing.append(_expected_details_name(run_id, dataset))
        if validation is None:
            dataset_missing.append(_expected_validation_name(run_id, dataset))
        resolution["datasets"][dataset] = {
            "run_id": run_id,
            "summary": summary,
            "details": details,
            "validation": validation,
            "ready": not dataset_missing,
            "missing": dataset_missing,
        }
        missing.extend(dataset_missing)

    training = _first_existing([root / candidate for candidate in TRAINING_JSON_CANDIDATES])
    if training is None:
        training = _first_existing(
            [
                path
                for path in _glob(root, "results/train_*rest_r1*.json")
                if not any(tag in path.name for tag in ("probe", "dryrun", "pool", "sample", "manifest"))
            ]
        )
    if training is None:
        training = _first_existing(_glob(root, "adapters/rest_r1/training_summary.json"))
    resolution["training_json"] = training
    if training is None:
        missing.append("results/train_rest_r1_training.json")

    adapter: Path | None = None
    if training is not None:
        try:
            training_payload = _read_json(training)
        except (OSError, ValueError):
            training_payload = None
        if training_payload is not None:
            adapter = _first_existing(_adapter_paths_from_training(root, training_payload))
    if adapter is None:
        adapter = _first_existing([root / candidate for candidate in ADAPTER_CANDIDATES])
    if adapter is None:
        globbed = [path for path in _glob(root, "adapters/rest_r1/**/adapter_model.safetensors") if path.is_file()]
        globbed.sort(key=lambda path: ("checkpoint" in path.as_posix().lower(), len(path.as_posix())))
        adapter = _first_existing(globbed)
    resolution["adapter"] = adapter
    if adapter is None:
        missing.append("adapters/rest_r1/adapter/adapter_model.safetensors")

    resolution["training_ready"] = training is not None
    resolution["adapter_ready"] = adapter is not None
    resolution["missing"] = missing
    return resolution


# --------------------------------------------------------------------------- #
# shared loading cache
# --------------------------------------------------------------------------- #
def _cached_json(ctx: dict[str, Any], key: str, path: Path) -> Any:
    cache = ctx["_cache"]
    if key not in cache:
        cache[key] = _read_json(path)
    return cache[key]


def _cached_details(ctx: dict[str, Any], dataset: str) -> tuple[bytes, list[dict[str, Any]]]:
    cache = ctx["_cache"]
    cache_key = f"details:{dataset}"
    if cache_key not in cache:
        path = ctx["resolution"]["datasets"][dataset]["details"]
        data = _read_details_bytes(path)
        cache[cache_key] = (data, _parse_details(data, path))
    return cache[cache_key]


def _make_check(check_id: str, description: str, problems: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check_id,
        "description": description,
        "status": "failed" if problems else "passed",
        "problems": problems,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_details_counts(ctx: dict[str, Any]) -> dict[str, Any]:
    expected = ctx["expected_n"]
    problems: list[str] = []
    observed: dict[str, Any] = {}
    for dataset in ctx["datasets"]:
        entry = ctx["resolution"]["datasets"][dataset]
        _, rows = _cached_details(ctx, dataset)
        summary = _cached_json(ctx, f"summary:{dataset}", entry["summary"])
        validation = _cached_json(ctx, f"validation:{dataset}", entry["validation"])
        record = {
            "details_rows": len(rows),
            "summary_n": summary.get("n"),
            "summary_details_rows": summary.get("details_rows"),
            "validation_n": validation.get("n"),
        }
        observed[dataset] = record
        for field, value in record.items():
            if value != expected:
                problems.append(f"{dataset}: {field}={value!r}，期望 {expected}")
        ids = [row.get("id") for row in rows]
        if len(set(ids)) != len(ids):
            problems.append(f"{dataset}: details 中存在重复 id")
        indices = [row.get("index") for row in rows]
        if indices and indices != list(range(len(indices))):
            problems.append(f"{dataset}: details 的 index 不是从 0 开始的连续整数")
    return _make_check(
        "details_counts",
        f"details 行数 = n = {expected}（两数据集）",
        problems,
        {"expected_n": expected, "observed": observed},
    )


def _score_row(ctx: dict[str, Any], row: dict[str, Any], default_prompt_id: str | None) -> Any:
    """Reproduce the evaluation-time scoring path for one details row."""

    prompt_id = row.get("prompt_id") or default_prompt_id
    pixel_conversion = row.get("pixel_conversion")
    if prompt_id is None and isinstance(pixel_conversion, dict) and pixel_conversion.get("conversion_applied"):
        prompt_id = "p1_pixel"
    output_text = row.get("output_text")
    if output_text is None:
        output_text = row.get("raw_output")
    if prompt_id == "p1_pixel":
        pixel_convert = ctx.get("pixel_convert")
        if pixel_convert is None:
            raise ValueError("p1_pixel 行需要 src/eval/pixel_convert.py，但该模块不可用")
        record = {
            "answer": row.get("ground_truth"),
            "width": row.get("width"),
            "height": row.get("height"),
        }
        score, _audit = pixel_convert.score_prediction_pixel(output_text, record)
        return score
    return ctx["grounding"].score_prediction(output_text, row.get("ground_truth"))


def check_metrics(ctx: dict[str, Any]) -> dict[str, Any]:
    grounding = ctx["grounding"]
    tolerance = ctx["tolerance"]
    problems: list[str] = []
    per_dataset: dict[str, Any] = {}

    for dataset in ctx["datasets"]:
        entry = ctx["resolution"]["datasets"][dataset]
        _, rows = _cached_details(ctx, dataset)
        summary = _cached_json(ctx, f"summary:{dataset}", entry["summary"])
        validation = _cached_json(ctx, f"validation:{dataset}", entry["validation"])
        default_prompt_id = summary.get("prompt_id") or (summary.get("config") or {}).get("prompt_id")

        scores: list[Any] = []
        mismatches: list[dict[str, Any]] = []
        for row in rows:
            try:
                score = _score_row(ctx, row, default_prompt_id)
            except (TypeError, ValueError, KeyError) as exc:
                scores.append(grounding.Score(None, 0.0, False, "recompute_error"))
                mismatches.append(
                    {"index": row.get("index"), "id": row.get("id"), "field": "recompute", "error": str(exc)}
                )
                continue

            stored_pred = row.get("predicted_box")
            if stored_pred is None:
                if score.predicted_box is not None:
                    mismatches.append(
                        {
                            "index": row.get("index"),
                            "id": row.get("id"),
                            "field": "predicted_box",
                            "stored": None,
                            "recomputed": list(score.predicted_box),
                        }
                    )
            else:
                stored = tuple(float(value) for value in stored_pred)
                if (
                    score.predicted_box is None
                    or len(stored) != 4
                    or any(abs(a - b) > tolerance for a, b in zip(stored, score.predicted_box))
                ):
                    mismatches.append(
                        {
                            "index": row.get("index"),
                            "id": row.get("id"),
                            "field": "predicted_box",
                            "stored": stored_pred,
                            "recomputed": list(score.predicted_box) if score.predicted_box else None,
                        }
                    )
            if row.get("parse_error") != score.parse_error:
                mismatches.append(
                    {
                        "index": row.get("index"),
                        "id": row.get("id"),
                        "field": "parse_error",
                        "stored": row.get("parse_error"),
                        "recomputed": score.parse_error,
                    }
                )
            stored_iou = row.get("iou")
            if stored_iou is None or not isinstance(stored_iou, (int, float)) or abs(float(stored_iou) - score.iou) > tolerance:
                mismatches.append(
                    {
                        "index": row.get("index"),
                        "id": row.get("id"),
                        "field": "iou",
                        "stored": stored_iou,
                        "recomputed": score.iou,
                    }
                )
            stored_correct = row.get("correct@0.5")
            if isinstance(stored_correct, bool) and stored_correct != score.correct_at_0_5:
                mismatches.append(
                    {
                        "index": row.get("index"),
                        "id": row.get("id"),
                        "field": "correct@0.5",
                        "stored": stored_correct,
                        "recomputed": score.correct_at_0_5,
                    }
                )
            scores.append(score)

        if scores:
            metrics = grounding.metrics_from_scores(scores)
        else:
            metrics = {"acc@0.5": 0.0, "mIoU": 0.0, "parse_rate": 0.0, "invalid": 0}
            problems.append(f"{dataset}: details 为空，无法重算指标")
        recomputed = {
            "n": len(rows),
            "acc@0.5": metrics["acc@0.5"],
            "mIoU": metrics["mIoU"],
            "parse_rate": metrics["parse_rate"],
            "invalid": metrics["invalid"],
        }

        summary_metrics = summary.get("metrics") or {}
        metric_diffs: list[dict[str, Any]] = []
        for metric in ("acc@0.5", "mIoU", "parse_rate"):
            recorded = summary_metrics.get(metric)
            if not isinstance(recorded, (int, float)):
                problems.append(f"{dataset}: summary.metrics 缺少 {metric}")
                continue
            diff = float(recomputed[metric]) - float(recorded)
            if abs(diff) > tolerance:
                metric_diffs.append(
                    {"metric": metric, "recomputed": recomputed[metric], "summary": recorded, "abs_diff": abs(diff)}
                )
        invalid_recorded = summary_metrics.get("invalid")
        if invalid_recorded != recomputed["invalid"]:
            metric_diffs.append(
                {"metric": "invalid", "recomputed": recomputed["invalid"], "summary": invalid_recorded, "abs_diff": None}
            )

        validation_metrics = (validation.get("checks") or {}).get("recomputed_metrics")
        validation_diffs: list[dict[str, Any]] = []
        if isinstance(validation_metrics, dict):
            for metric in ("acc@0.5", "mIoU", "parse_rate"):
                recorded = validation_metrics.get(metric)
                if isinstance(recorded, (int, float)) and abs(float(recomputed[metric]) - float(recorded)) > tolerance:
                    validation_diffs.append(
                        {
                            "metric": metric,
                            "recomputed": recomputed[metric],
                            "validation": recorded,
                            "abs_diff": abs(float(recomputed[metric]) - float(recorded)),
                        }
                    )
        else:
            problems.append(f"{dataset}: validation 缺少 checks.recomputed_metrics，无法做第二源比对")

        if mismatches:
            problems.append(f"{dataset}: {len(mismatches)} 行逐条复算结果与 details 记录不一致")
        if metric_diffs:
            problems.append(f"{dataset}: 汇总指标与 summary.metrics 不一致（容差 {tolerance:g}）")
        if validation_diffs:
            problems.append(f"{dataset}: 汇总指标与 validation.recomputed_metrics 不一致")

        per_dataset[dataset] = {
            "prompt_id": default_prompt_id,
            "details_rows": len(rows),
            "recomputed": recomputed,
            "summary_metrics": summary_metrics,
            "validation_recomputed_metrics": validation_metrics,
            "metric_diffs_vs_summary": metric_diffs,
            "metric_diffs_vs_validation": validation_diffs,
            "row_mismatch_count": len(mismatches),
            "row_mismatch_samples": mismatches[:20],
        }

    return _make_check(
        "metrics_recompute",
        "用 src/eval/grounding.py 逐条重算 n/parse_rate/invalid/acc@0.5/mIoU 并与 summary 比对",
        problems,
        {"tolerance": tolerance, "per_dataset": per_dataset},
    )


def check_details_sha256(ctx: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    per_dataset: dict[str, Any] = {}
    for dataset in ctx["datasets"]:
        entry = ctx["resolution"]["datasets"][dataset]
        data, rows = _cached_details(ctx, dataset)
        actual = _sha256_bytes(data)
        summary = _cached_json(ctx, f"summary:{dataset}", entry["summary"])
        validation = _cached_json(ctx, f"validation:{dataset}", entry["validation"])
        summary_sha = summary.get("details_sha256")
        validation_sha = (validation.get("checks") or {}).get("details_sha256")
        if validation_sha is None:
            validation_sha = validation.get("details_sha256")
        validation_rows = (validation.get("checks") or {}).get("details_rows")
        if summary_sha is None:
            problems.append(f"{dataset}: summary 未记录 details_sha256")
        elif str(summary_sha).lower() != actual:
            problems.append(f"{dataset}: summary.details_sha256 与 details 实际 sha256 不一致")
        if validation_sha is None:
            problems.append(f"{dataset}: validation 未记录 details_sha256")
        elif str(validation_sha).lower() != actual:
            problems.append(f"{dataset}: validation.checks.details_sha256 与 details 实际 sha256 不一致")
        if validation_rows is not None and validation_rows != len(rows):
            problems.append(f"{dataset}: validation.checks.details_rows={validation_rows} 与 details 实际行数 {len(rows)} 不一致")
        per_dataset[dataset] = {
            "details_path": _rel(ctx["root"], entry["details"]),
            "actual_sha256": actual,
            "summary_details_sha256": summary_sha,
            "validation_details_sha256": validation_sha,
            "validation_details_rows": validation_rows,
            "actual_rows": len(rows),
        }
    return _make_check(
        "details_sha256",
        "validation 报告的 details_sha256 与 details 文件实际 sha256 一致（两数据集）",
        problems,
        per_dataset,
    )


def check_validation_report(ctx: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    per_dataset: dict[str, Any] = {}
    for dataset in ctx["datasets"]:
        entry = ctx["resolution"]["datasets"][dataset]
        validation = _cached_json(ctx, f"validation:{dataset}", entry["validation"])
        summary = _cached_json(ctx, f"summary:{dataset}", entry["summary"])
        note: dict[str, Any] = {
            "validation_path": _rel(ctx["root"], entry["validation"]),
            "summary_run_id": summary.get("run_id"),
            "validated_run_id": validation.get("validated_run_id"),
            "status": validation.get("status"),
            "errors": validation.get("errors"),
            "result_path": validation.get("result_path"),
        }
        if validation.get("status") != "passed":
            problems.append(f"{dataset}: validation.status={validation.get('status')!r}，期望 'passed'")
        errors = validation.get("errors")
        if isinstance(errors, list) and errors:
            problems.append(f"{dataset}: validation.errors 非空（{len(errors)} 条）")
        if validation.get("validated_run_id") not in (None, summary.get("run_id")):
            problems.append(f"{dataset}: validation.validated_run_id 与 summary.run_id 不一致")
        expected_result = _rel(ctx["root"], entry["summary"])
        if validation.get("result_path") not in (None, expected_result):
            problems.append(f"{dataset}: validation.result_path={validation.get('result_path')!r}，期望 {expected_result!r}")
        checks = validation.get("checks") or {}
        if checks.get("invalid_in_n_verified") is False:
            problems.append(f"{dataset}: validation.checks.invalid_in_n_verified=false")
        per_dataset[dataset] = note
    return _make_check(
        "validation_report",
        "validation 报告本身 status=passed、errors 为空、run_id/result_path 自洽",
        problems,
        per_dataset,
    )


def _find_adapter_sha256(training: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                if isinstance(value, str) and _is_sha256(value):
                    last = key_text.lower()
                    previous = path[-1].lower() if path else ""
                    if (last == "sha256" and "adapter_model" in previous) or last in {
                        "adapter_model_sha256",
                        "adapter_sha256",
                    }:
                        found.append({"path": ".".join(path + (key_text,)), "sha256": value.lower()})
                walk(value, path + (key_text,))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(training, ())
    found.sort(
        key=lambda record: (
            0 if "final_adapter_summary" in record["path"] else 1,
            0 if "adapter_files" in record["path"] else 1,
            record["path"],
        )
    )
    return found


def _find_adapter_bytes(training: Any) -> int | None:
    found: list[int] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                if key_text.lower() == "bytes" and isinstance(value, int):
                    previous = path[-1].lower() if path else ""
                    if "adapter_model" in previous:
                        found.append(value)
                walk(value, path + (key_text,))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(training, ())
    return found[0] if found else None


def check_adapter_sha256(ctx: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    training_path = ctx["resolution"]["training_json"]
    adapter_path = ctx["resolution"]["adapter"]
    actual = _sha256_file(adapter_path)
    actual_size = adapter_path.stat().st_size
    training = _cached_json(ctx, "training", training_path)
    recorded = _find_adapter_sha256(training)
    recorded_bytes = _find_adapter_bytes(training)
    details: dict[str, Any] = {
        "training_json": _rel(ctx["root"], training_path),
        "adapter_path": _rel(ctx["root"], adapter_path),
        "actual_sha256": actual,
        "actual_bytes": actual_size,
        "recorded_sha256_entries": recorded,
        "recorded_bytes": recorded_bytes,
        "summary_adapter_sha256": {},
    }

    if not recorded:
        problems.append("training JSON 中找不到 adapter_model.safetensors 的 sha256 记录")
    else:
        matches = [entry for entry in recorded if entry["sha256"] == actual]
        if not matches:
            problems.append(
                "adapter_model.safetensors 实际 sha256 与 training JSON 记录不一致"
                f"（实际 {actual}，记录 {recorded[0]['sha256']}）"
            )
    if recorded_bytes is not None and recorded_bytes != actual_size:
        problems.append(f"adapter_model.safetensors 实际字节数 {actual_size} 与 training JSON 记录 {recorded_bytes} 不一致")

    for dataset in ctx["datasets"]:
        summary = _cached_json(ctx, f"summary:{dataset}", ctx["resolution"]["datasets"][dataset]["summary"])
        summary_sha = ((summary.get("config") or {}).get("adapter") or {}).get("adapter_model_sha256")
        if summary_sha is None:
            summary_sha = (summary.get("adapter_load") or {}).get("adapter_metadata", {}).get("adapter_model_sha256")
        details["summary_adapter_sha256"][dataset] = summary_sha
        if summary_sha is not None and str(summary_sha).lower() != actual:
            problems.append(f"{dataset}: summary 里的 adapter_model_sha256 与实际文件不一致")

    return _make_check(
        "adapter_sha256",
        "adapter_model.safetensors sha256 与 training JSON 记录一致",
        problems,
        details,
    )


def _token_value(text: str) -> tuple[float, bool]:
    percent = text.endswith("%")
    core = text.rstrip("%").replace(",", "")
    return float(core), percent


def _line_tokens(line: str) -> list[dict[str, Any]]:
    cleaned = STRIP_RE.sub(" ", line)
    tokens: list[dict[str, Any]] = []
    for match in NUMBER_RE.finditer(cleaned):
        raw = match.group(0)
        try:
            value, percent = _token_value(raw)
        except ValueError:
            continue
        tokens.append({"text": raw, "normalized": raw.rstrip("%").replace(",", ""), "value": value, "percent": percent})
    return tokens


def _numeric_equal(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def check_report_literals(ctx: dict[str, Any]) -> dict[str, Any]:
    root = ctx["root"]
    problems: list[str] = []
    per_file: dict[str, Any] = {}

    canonicals: list[dict[str, Any]] = []
    for dataset in ctx["datasets"]:
        summary = _cached_json(ctx, f"summary:{dataset}", ctx["resolution"]["datasets"][dataset]["summary"])
        metrics = summary.get("metrics") or {}
        values = {
            "n": summary.get("n"),
            "acc@0.5": metrics.get("acc@0.5"),
            "mIoU": metrics.get("mIoU"),
            "parse_rate": metrics.get("parse_rate"),
            "invalid": metrics.get("invalid"),
        }
        for metric in METRIC_KEYS:
            value = values[metric]
            if value is None:
                problems.append(f"{dataset}: summary 缺少 {metric}，无法校对其在报告中的字面")
                continue
            canonicals.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "literal": _metric_literal(value),
                }
            )

    for relative in REPORT_FILES:
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative} 不存在")
            per_file[relative] = {"status": "failed", "problems": ["文件不存在"]}
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        relevant = [
            (lineno, line)
            for lineno, line in enumerate(text.splitlines(), 1)
            if RELEVANT_RE.search(line)
        ]
        tokens: list[dict[str, Any]] = []
        for lineno, line in relevant:
            for token in _line_tokens(line):
                token["line"] = lineno
                tokens.append(token)
        token_texts = {token["text"] for token in tokens} | {token["normalized"] for token in tokens}
        numeric_tokens = [token for token in tokens if not token["percent"]]

        missing_presence: list[dict[str, Any]] = []
        for canonical in canonicals:
            metric = canonical["metric"]
            if metric not in REQUIRED_PRESENCE:
                continue
            if metric == "mIoU":
                present = canonical["literal"] in token_texts
            else:
                present = any(_numeric_equal(token["value"], float(canonical["value"])) for token in numeric_tokens)
            if not present:
                missing_presence.append(
                    {
                        "dataset": canonical["dataset"],
                        "metric": metric,
                        "expected_literal": canonical["literal"],
                        "expected_value": canonical["value"],
                    }
                )

        suspect_rounding: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for token in tokens:
            if token["percent"]:
                for canonical in canonicals:
                    value = float(canonical["value"])
                    if abs(token["value"] / 100.0 - value) <= max(1e-9, 1e-3 * abs(value)):
                        warnings.append(
                            {
                                "line": token["line"],
                                "token": token["text"],
                                "metric": canonical["metric"],
                                "dataset": canonical["dataset"],
                                "note": "以百分数形式出现，无法逐字比对",
                            }
                        )
                continue
            for canonical in canonicals:
                if canonical["metric"] != "mIoU":
                    continue
                value = float(canonical["value"])
                if abs(token["value"] - value) <= max(1e-9, 1e-3 * abs(value)) and token["text"] != canonical["literal"]:
                    suspect_rounding.append(
                        {
                            "line": token["line"],
                            "token": token["text"],
                            "dataset": canonical["dataset"],
                            "metric": "mIoU",
                            "expected_literal": canonical["literal"],
                            "expected_value": value,
                        }
                    )

        file_problems: list[str] = []
        if not relevant:
            file_problems.append(f"{relative} 中没有出现 rest_r1 / ReST 第1轮 / T8 字样（报告可能尚未更新）")
        if missing_presence:
            file_problems.append(f"{relative} 缺少 rest_r1 关键数字的 JSON 字面（{len(missing_presence)} 项）")
        if file_problems:
            problems.extend(file_problems)

        per_file[relative] = {
            "status": "failed" if file_problems else "passed",
            "relevant_line_count": len(relevant),
            "relevant_lines": [lineno for lineno, _ in relevant][:50],
            "token_count": len(tokens),
            "missing_presence": missing_presence,
            "suspect_rounding": suspect_rounding[:50],
            "decimal_format_warnings": warnings[:20],
            "problems": file_problems,
        }

    return _make_check(
        "report_literals",
        "report/tables/table1_draft.md 与 report/paper.md 中 rest_r1 数字与 JSON 字面一致",
        problems,
        {"canonical_metrics": canonicals, "per_file": per_file},
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_verification(root: Path, grounding: Any, pixel_convert: Any, expected_n: int, tolerance: float) -> dict[str, Any]:
    resolution = resolve_inputs(root)
    ready_datasets = [dataset for dataset in DATASETS if resolution["datasets"][dataset]["ready"]]
    pending_datasets = [dataset for dataset in DATASETS if not resolution["datasets"][dataset]["ready"]]
    all_ready = not pending_datasets
    adapter_ready = bool(resolution.get("training_ready") and resolution.get("adapter_ready"))

    base: dict[str, Any] = {
        "run_id": "verify_rest_r1",
        "script": "src/eval/verify_rest_r1.py",
        "generated_at": _now(),
        "expected_n": expected_n,
        "tolerance": tolerance,
        "grounding_module": getattr(grounding, "__file__", None),
        "inputs": {
            "datasets": {
                dataset: {
                    "run_id": entry["run_id"],
                    "summary": _rel(root, entry["summary"]) if entry["summary"] else None,
                    "details": _rel(root, entry["details"]) if entry["details"] else None,
                    "validation": _rel(root, entry["validation"]) if entry["validation"] else None,
                }
                for dataset, entry in resolution["datasets"].items()
            },
            "training_json": _rel(root, resolution["training_json"]) if resolution["training_json"] else None,
            "adapter": _rel(root, resolution["adapter"]) if resolution["adapter"] else None,
        },
        "datasets": {
            dataset: {
                "status": "checked" if entry["ready"] else "pending",
                "run_id": entry["run_id"],
                "missing": entry["missing"],
            }
            for dataset, entry in resolution["datasets"].items()
        },
        "pending": {
            "datasets": pending_datasets,
            "items": [],
            "missing_paths": resolution["missing"],
        },
        "deferred_checks": [],
        "notes": resolution["notes"],
    }

    if not ready_datasets:
        report = dict(base)
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
        "_cache": {},
    }

    builders = [
        check_details_counts,
        check_metrics,
        check_details_sha256,
        check_validation_report,
    ]
    if adapter_ready:
        builders.append(check_adapter_sha256)
    else:
        base["deferred_checks"].append("adapter_sha256：training JSON 或 adapter_model.safetensors 缺失")
        base["pending"]["items"].append("adapter_sha256")
    if all_ready and adapter_ready:
        builders.append(check_report_literals)
    else:
        base["deferred_checks"].append(
            "report_literals：等全部数据集/训练产物落盘后再执行（当前 pending: "
            + ", ".join(pending_datasets or ["-"]) + "）"
        )
        base["pending"]["items"].append("report_literals")

    checks: list[dict[str, Any]] = []
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

    failures = [check for check in checks if check["status"] == "failed"]
    if failures:
        status = "failed"
    elif not all_ready or not adapter_ready:
        status = "partial"
    else:
        status = "passed"
    report = dict(base)
    report.update(
        {
            "status": status,
            "missing": resolution["missing"],
            "checks": checks,
            "failed_checks": [check["id"] for check in failures],
        }
    )
    return report


def _write_report(report: dict[str, Any], output_path: Path, no_write: bool) -> None:
    if no_write:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T8 rest_r1 独立复核（CPU、只读、不加载 torch 模型）")
    parser.add_argument("--root", default=None, help="项目根目录（默认按脚本位置推断）")
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N, help=f"每个数据集的期望样本数（默认 {EXPECTED_N}）")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE, help=f"浮点比对容差（默认 {TOLERANCE:g}）")
    parser.add_argument("--no-write", action="store_true", help="不写 results/verify_rest_r1_report.json")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    src_root = root
    if not (src_root / "src/eval/grounding.py").is_file():
        # --root may point at a fixture/checkout without sources; the frozen
        # evaluator always comes from the checkout this script lives in.
        src_root = PROJECT_ROOT
    src_dir = str(src_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from eval import grounding  # noqa: PLC0415  (late import keeps --help torch-free)
        from eval import pixel_convert  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"status": "error", "error": f"无法导入 src/eval/grounding.py: {exc}"}, ensure_ascii=False))
        return 2

    try:
        report = run_verification(
            root, grounding, pixel_convert, expected_n=args.expected_n, tolerance=args.tolerance
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    output_path = root / "results/verify_rest_r1_report.json"
    _write_report(report, output_path, args.no_write)

    if report["status"] == "pending":
        payload = {
            "status": "pending",
            "missing": report["missing"],
            "report_path": _rel(root, output_path) if not args.no_write else None,
            "generated_at": report["generated_at"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = {
        "status": report["status"],
        "datasets": {dataset: info["status"] for dataset, info in report.get("datasets", {}).items()},
        "checks": {check["id"]: check["status"] for check in report.get("checks", [])},
        "failed_checks": report.get("failed_checks", []),
        "pending": report.get("pending", {}).get("items", []),
        "deferred_checks": report.get("deferred_checks", []),
        "problems": {
            check["id"]: check["problems"]
            for check in report.get("checks", [])
            if check["status"] == "failed"
        },
        "report_path": _rel(root, output_path) if not args.no_write else None,
        "generated_at": report["generated_at"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
