"""Independently verify result schema and recompute every B0 metric from JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .dataset import load_dataset, read_jsonl, sha256_file
from .grid_convert import score_prediction_grid
from .grounding import metrics_from_scores, score_prediction
from .pixel_convert import score_prediction_pixel
from .prompts import (
    DEFAULT_PROMPT_ID,
    PROMPT_VARIANTS,
    prompt_sha256_for,
    prompt_template_for,
    representation_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
REQUIRED_TOP_LEVEL = {
    "run_id", "model", "adapter", "dataset", "split", "n", "metrics",
    "config", "git_commit", "timestamp", "status", "details_path", "details_sha256",
}
REQUIRED_METRICS = {"acc@0.5", "mIoU", "parse_rate", "invalid"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ids_digest(records: list[dict[str, Any]]) -> str:
    value = json.dumps([row["id"] for row in records], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def close_enough(actual: Any, expected: Any) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-dataset", required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument(
        "--expected-adapter",
        default=None,
        help="expected top-level PEFT adapter path; omit to require adapter=null (B0)",
    )
    parser.add_argument(
        "--expected-prompt-id",
        default=DEFAULT_PROMPT_ID,
        choices=tuple(PROMPT_VARIANTS),
        help="expected prompt variant (p1 default; p1_bin1000 / p1_pixel / p1_grid100 for T7 B/C/D)",
    )
    args = parser.parse_args()
    result_path = resolve_path(args.result)
    report_path = resolve_path(args.report)
    if report_path.parent != RESULTS_ROOT or not report_path.name.startswith("eval_"):
        raise ValueError("validation report must be results/eval_*.json")
    errors: list[str] = []
    checks: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    expected_adapter: str | None = None
    try:
        if args.expected_adapter and args.expected_adapter.strip().lower() not in ("", "none", "null"):
            adapter_arg_path = resolve_path(args.expected_adapter)
            try:
                expected_adapter = adapter_arg_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError as exc:
                raise ValueError(f"--expected-adapter must live under the project root: {exc}") from exc
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        missing = REQUIRED_TOP_LEVEL - set(summary)
        if missing:
            raise ValueError(f"missing top-level fields: {sorted(missing)}")
        for field, expected in (
            ("run_id", args.expected_run_id),
            ("dataset", args.expected_dataset),
            ("split", args.expected_split),
            ("n", args.expected_n),
            ("status", "completed"),
            ("adapter", expected_adapter),
        ):
            if summary[field] != expected:
                errors.append(f"summary {field}={summary[field]!r}, expected {expected!r}")
        if not isinstance(summary["metrics"], dict) or set(summary["metrics"]) != REQUIRED_METRICS:
            errors.append(f"metrics keys must be exactly {sorted(REQUIRED_METRICS)}")

        config = summary["config"]
        prompt_id = args.expected_prompt_id
        fixed_config = {
            "image_size": 384,
            "max_new_tokens": 64,
            "prompt_id": prompt_id,
            "seed": 42,
            "do_sample": False,
            "num_beams": 1,
            "do_image_splitting": False,
            "image_seq_len": 36,
            "local_files_only": True,
        }
        for key, expected in fixed_config.items():
            if config.get(key) != expected:
                errors.append(f"config {key}={config.get(key)!r}, expected {expected!r}")
        expected_sha256 = prompt_sha256_for(prompt_id)
        expected_template = prompt_template_for(prompt_id)
        expected_representation = representation_for(prompt_id)
        prompt_checks = {
            "prompt_id": prompt_id,
            "prompt_sha256": expected_sha256,
            "prompt_sha256_matches": config.get("prompt_sha256") == expected_sha256,
            "prompt_template_matches": config.get("prompt_template") == expected_template,
            "representation": expected_representation,
            "representation_matches": (
                config.get("representation") == expected_representation
                if prompt_id != DEFAULT_PROMPT_ID or "representation" in config
                else True
            ),
            "conversion_required": bool(PROMPT_VARIANTS[prompt_id]["conversion_required"]),
        }
        checks["prompt_variant"] = prompt_checks
        if not prompt_checks["prompt_sha256_matches"]:
            errors.append("config prompt_sha256 differs from the expected variant template")
        if not prompt_checks["prompt_template_matches"]:
            errors.append("config prompt_template differs from the expected variant template")
        if prompt_id != DEFAULT_PROMPT_ID and not prompt_checks["representation_matches"]:
            errors.append("config representation differs from the expected variant")
        if prompt_id != DEFAULT_PROMPT_ID:
            if summary.get("prompt_id") != prompt_id:
                errors.append(f"summary prompt_id={summary.get('prompt_id')!r}, expected {prompt_id!r}")
            if summary.get("representation") != expected_representation:
                errors.append(
                    f"summary representation={summary.get('representation')!r}, expected {expected_representation!r}"
                )
        conversion_config = config.get("coordinate_conversion")
        if prompt_id in ("p1_pixel", "p1_grid100"):
            if not isinstance(conversion_config, dict) or conversion_config.get("applied") is not True:
                errors.append(
                    f"config.coordinate_conversion.applied must be true for {prompt_id}"
                )
        adapter_record = config.get("adapter")
        if expected_adapter is None:
            checks["adapter"] = None
            if adapter_record is not None:
                errors.append("summary adapter is null but config.adapter metadata is present")
        elif not isinstance(adapter_record, dict):
            errors.append("config.adapter metadata missing for adapter run")
        else:
            adapter_dir = resolve_path(adapter_record.get("path", ""))
            adapter_config_path = adapter_dir / "adapter_config.json"
            adapter_model_path = adapter_dir / "adapter_model.safetensors"
            adapter_checks: dict[str, Any] = {
                "path": adapter_record.get("path"),
                "expected_path": expected_adapter,
                "path_matches": adapter_record.get("path") == expected_adapter,
                "is_trainable": adapter_record.get("is_trainable"),
                "is_trainable_false": adapter_record.get("is_trainable") is False,
                "adapter_config_sha256": adapter_record.get("adapter_config_sha256"),
                "adapter_model_sha256": adapter_record.get("adapter_model_sha256"),
                "adapter_files_present": adapter_config_path.is_file() and adapter_model_path.is_file(),
                "adapter_config_sha256_matches": False,
                "adapter_model_sha256_matches": False,
            }
            if adapter_checks["adapter_files_present"]:
                adapter_checks["adapter_config_sha256_matches"] = (
                    sha256_file(adapter_config_path) == adapter_record.get("adapter_config_sha256")
                )
                adapter_checks["adapter_model_sha256_matches"] = (
                    sha256_file(adapter_model_path) == adapter_record.get("adapter_model_sha256")
                )
            checks["adapter"] = adapter_checks
            for key in (
                "path_matches",
                "is_trainable_false",
                "adapter_files_present",
                "adapter_config_sha256_matches",
                "adapter_model_sha256_matches",
            ):
                if not adapter_checks[key]:
                    errors.append(f"adapter check failed: {key}")
        data_path = resolve_path(config["dataset_path"])
        if sha256_file(data_path) != config["dataset_sha256"]:
            errors.append("dataset SHA256 differs from result config")
        records = load_dataset(
            data_path,
            project_root=PROJECT_ROOT,
            expected_source=args.expected_dataset,
            split=args.expected_split,
            check_images=False,
        )
        selected = records[: args.expected_n]
        if ids_digest(selected) != config["selected_ids_sha256"]:
            errors.append("selected ID sequence SHA256 mismatch")

        details_path = resolve_path(summary["details_path"])
        if details_path.parent != RESULTS_ROOT or not details_path.name.startswith("eval_"):
            errors.append("details_path must be results/eval_*")
        actual_details_hash = sha256_file(details_path)
        if actual_details_hash != summary["details_sha256"]:
            errors.append("details SHA256 mismatch")
        details = read_jsonl(details_path)
        checks["details_rows"] = len(details)
        if len(details) != args.expected_n or summary.get("details_rows") != args.expected_n:
            errors.append(f"details row count {len(details)} / summary {summary.get('details_rows')} != {args.expected_n}")
        if len(details) != len(selected):
            raise ValueError("cannot align details and selected dataset rows")

        scores = []
        seen_ids: set[str] = set()
        for index, (detail, record) in enumerate(zip(details, selected)):
            sample_id = record["id"]
            expected_fields = {
                "run_id": args.expected_run_id,
                "config_fingerprint": config["config_fingerprint"],
                "dataset": args.expected_dataset,
                "split": args.expected_split,
                "index": index,
                "id": sample_id,
                "image": record["image"],
                "question": record["question"],
                "ground_truth": record["answer"],
            }
            for key, expected in expected_fields.items():
                if detail.get(key) != expected:
                    errors.append(
                        f"dataset={args.expected_dataset} split={args.expected_split} id={sample_id} "
                        f"detail {key}={detail.get(key)!r}, expected {expected!r}"
                    )
            if sample_id in seen_ids:
                errors.append(f"duplicate detail id: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            seen_ids.add(sample_id)
            if not isinstance(detail.get("raw_output"), str):
                errors.append(f"missing raw_output: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if not isinstance(detail.get("output_text"), str):
                raise ValueError(f"missing output_text: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if prompt_id == "p1_pixel":
                score, conversion_audit = score_prediction_pixel(detail["output_text"], record)
            elif prompt_id == "p1_grid100":
                score, conversion_audit = score_prediction_grid(detail["output_text"], record)
            else:
                score = score_prediction(detail["output_text"], record["answer"])
                conversion_audit = None
            scores.append(score)
            if prompt_id != DEFAULT_PROMPT_ID:
                if detail.get("prompt_id") != prompt_id:
                    errors.append(f"detail prompt_id mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
                if detail.get("representation") != expected_representation:
                    errors.append(f"detail representation mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            expected_box = list(score.predicted_box) if score.predicted_box is not None else None
            if detail.get("predicted_box") != expected_box:
                errors.append(f"predicted_box mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if detail.get("parse_error") != score.parse_error:
                errors.append(f"parse_error mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if not close_enough(detail.get("iou"), score.iou):
                errors.append(f"IoU mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if detail.get("correct@0.5") != score.correct_at_0_5:
                errors.append(f"correct@0.5 mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            if prompt_id == "p1_pixel":
                if detail.get("pixel_conversion") != conversion_audit:
                    errors.append(f"pixel_conversion audit mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
                if detail.get("pixel_box") != (conversion_audit or {}).get("pixel_box"):
                    errors.append(f"pixel_box mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
            elif prompt_id == "p1_grid100":
                if detail.get("grid_conversion") != conversion_audit:
                    errors.append(f"grid_conversion audit mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")
                if detail.get("grid_box") != (conversion_audit or {}).get("grid_box"):
                    errors.append(f"grid_box mismatch: dataset={args.expected_dataset} split={args.expected_split} id={sample_id}")

        recomputed = metrics_from_scores(scores)
        checks["recomputed_metrics"] = recomputed
        for key in REQUIRED_METRICS:
            if key == "invalid":
                if summary["metrics"].get(key) != recomputed[key]:
                    errors.append(f"metric {key}: {summary['metrics'].get(key)} != recomputed {recomputed[key]}")
            elif not close_enough(summary["metrics"].get(key), recomputed[key]):
                errors.append(f"metric {key}: {summary['metrics'].get(key)} != recomputed {recomputed[key]}")
        checks["invalid_in_n_verified"] = len(scores) == args.expected_n
        checks["raw_outputs_present"] = all(isinstance(row.get("raw_output"), str) for row in details)
        checks["unique_detail_ids"] = len(seen_ids)
        checks["dataset_sha256"] = config["dataset_sha256"]
        checks["details_sha256"] = actual_details_hash
        checks["recorded_source_sha256"] = config.get("source_sha256")
        current_source = {}
        for name in config.get("source_sha256", {}):
            source_path = Path(__file__).with_name(name)
            if source_path.is_file():
                current_source[name] = sha256_file(source_path)
        checks["recorded_source_matches_current"] = current_source == config.get("source_sha256")
        if not checks["recorded_source_matches_current"]:
            errors.append("recorded evaluator source hashes differ from current source; use the recorded version or rerun affected results")
    except Exception as exc:
        errors.append(f"validation aborted: {type(exc).__name__}: {exc}")

    report = {
        "run_id": f"validation_{args.expected_run_id}",
        "validated_run_id": args.expected_run_id,
        "dataset": args.expected_dataset,
        "split": args.expected_split,
        "n": args.expected_n,
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "status": "passed" if not errors else "failed",
        "result_path": result_path.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
        "errors": errors,
        "gpu_used": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
