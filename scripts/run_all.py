#!/usr/bin/env python3
"""One-command reproduction driver for the final-project (T17).

``scripts/run_all.sh`` and ``scripts/run_all.ps1`` are thin wrappers around this
module; both expose the same options.  The driver is dependency-free (standard
library only, no torch/transformers import) so it can run the environment check
even when the Python environment is broken.

Safety contract
---------------
* Nothing expensive runs silently: every training/evaluation command is listed
  under an explicit stage, and GPU work additionally requires ``--gpu-lease-id``.
* ``--allow-train`` is mandatory for any training stage.  ``--stage all`` never
  retrains without it; it reports the stage as skipped with a reason instead.
* Existing artifacts are never overwritten by default: a command whose output
  already exists is skipped (pass ``--force-retrain`` / ``--force-merge`` /
  ``--force-rerun`` to opt in).  The frozen evaluator itself also refuses to
  overwrite results.
* ``--dry-run`` prints the exact commands and skips without creating logs,
  writing results or loading a model.

Examples
--------
    python scripts/run_all.py --list-stages
    python scripts/run_all.py --stage check --dry-run
    python scripts/run_all.py --stage data --stage tests
    python scripts/run_all.py --stage all --dry-run
    python scripts/run_all.py --stage train-sft --allow-train --gpu-lease-id <ID>
    python scripts/run_all.py --stage eval-sft --gpu-lease-id <ID>
    python scripts/run_all.py --stage eval-sft --cpu-eval
    python scripts/run_all.py --stage deploy
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "logs" / "run_all"
MODEL_DIR = "models/SmolVLM-256M-Instruct"
GPU_MEMORY_LIMIT_MIB = 7500
LOCAL_TZ = timezone(timedelta(hours=8))
PY = sys.executable or "python"

TRAIN_STAGES = {"train-sft", "train-t7", "train-diagnostics"}
GPU_EVAL_STAGES = {"eval-b0", "eval-sft", "eval-t7", "eval-screenspot"}
ALL_ORDER = (
    "check",
    "data",
    "tests",
    "train-sft",
    "train-t7",
    "train-diagnostics",
    "eval-b0",
    "eval-sft",
    "eval-t7",
    "eval-deploy",
    "eval-screenspot",
    "verify",
    "tables",
    "deploy",
    "deploy-int8",
    "demo",
)

STAGE_TITLES = {
    "check": "环境与资产预检（只读 + deploy_prep 快照）",
    "data": "数据验证与数据单测（CPU）",
    "tests": "全项目 CPU 单测（训练/评测/部署/demo 不加载 CUDA 模型）",
    "train-sft": "SFT v1 三 seed 训练（昂贵，需 --allow-train + GPU 租约）",
    "train-t7": "T7 A/B/C 训练（昂贵，需 --allow-train + GPU 租约）",
    "train-diagnostics": "T5 100 条过拟合诊断训练（昂贵，需 --allow-train + GPU 租约）",
    "eval-b0": "零样本 B0 两集评测（GPU；--cpu-eval 可退 CPU）",
    "eval-sft": "SFT v1 三 seed × 两集评测 + 逐 run 验证（GPU/CPU）",
    "eval-t7": "T7 A/B(3-seed)+C(pilot) × 两集评测 + 验证 + G4 3seed 表（GPU/CPU）",
    "eval-deploy": "merged CPU fp32 官方部署评测（synthetic 1000，CPU-only 无 GPU 租约）",
    "eval-screenspot": "ScreenSpot 迁移评测 B0 + 三 seed + 验证 + 表（GPU/CPU）",
    "verify": "对已存在的 summary 逐 run 复算验证（CPU，只补缺失验证报告；含 T7-B seed43/44 与 deploy 官方 run）",
    "tables": "由 results/*.json 生成 3-seed 主表 / G4 3seed 表 / ScreenSpot 表（CPU）",
    "deploy": "T14 部署：check_env / merge / verify / 等价性 / CPU fp32 延迟 / handoff（CPU）",
    "deploy-int8": "T14 负结果复现：torch INT8 dynamic 50 图延迟与掉点（CPU，慢）",
    "demo": "T15 demo smoke：fake + 真实 merged 模型 HTTP/API 往返（CPU；浏览器截图证据见 deploy_demo_browser.json）",
}

# ---------------------------------------------------------------------------
# Run metadata (single source of truth for eval stage and validation stage)
# ---------------------------------------------------------------------------
SFT_SEED_DIRS = {42: "sft_v1", 43: "sft_v1_s43", 44: "sft_v1_s44"}

B0_RUNS: tuple[dict[str, Any], ...] = (
    {
        "run_id": "zeroshot_b0_synthetic_s42_v1",
        "dataset": "synthetic",
        "split": "eval",
        "data": "data/synthetic_eval.jsonl",
        "n": 1000,
        "adapter": None,
        "prompt": "p1",
        "summary": "results/zeroshot_synthetic.json",
        "details": "results/eval_zeroshot_b0_synthetic_s42_v1_details.jsonl",
        "validation": "results/eval_zeroshot_b0_synthetic_s42_v1_validation.json",
    },
    {
        "run_id": "zeroshot_b0_refcoco_s42_v1",
        "dataset": "refcoco",
        "split": "val",
        "data": "data/refcoco_val.jsonl",
        "n": 1000,
        "adapter": None,
        "prompt": "p1",
        "summary": "results/zeroshot_refcoco.json",
        "details": "results/eval_zeroshot_b0_refcoco_s42_v1_details.jsonl",
        "validation": "results/eval_zeroshot_b0_refcoco_s42_v1_validation.json",
    },
)

SFT_RUNS: tuple[dict[str, Any], ...] = tuple(
    {
        "run_id": f"sft_v1_s{seed}_{dataset}_v1",
        "dataset": dataset,
        "split": "eval" if dataset == "synthetic" else "val",
        "data": "data/synthetic_eval.jsonl" if dataset == "synthetic" else "data/refcoco_val.jsonl",
        "n": 1000,
        "adapter": f"adapters/{SFT_SEED_DIRS[seed]}/adapter",
        "prompt": "p1",
        "summary": f"results/zeroshot_sft_v1_s{seed}_{dataset}_v1.json",
        "details": f"results/eval_sft_v1_s{seed}_{dataset}_v1_details.jsonl",
        "validation": f"results/eval_sft_v1_s{seed}_{dataset}_v1_validation.json",
    }
    for seed in (42, 43, 44)
    for dataset in ("synthetic", "refcoco")
)

T7_B_RUNS: tuple[dict[str, Any], ...] = tuple(
    {
        "run_id": f"t7_b_s{seed}_{dataset}_v1" if seed != 42 else f"t7_b_{dataset}_v1",
        "dataset": dataset,
        "split": "eval" if dataset == "synthetic" else "val",
        "data": "data/synthetic_eval.jsonl" if dataset == "synthetic" else "data/refcoco_val.jsonl",
        "n": 1000,
        "adapter": f"adapters/sft_t7_b_bin1000{'' if seed == 42 else f'_s{seed}'}/adapter",
        "prompt": "p1_bin1000",
        "summary": (
            f"results/zeroshot_t7_b_{dataset}_v1.json"
            if seed == 42
            else f"results/zeroshot_t7_b_s{seed}_{dataset}_v1.json"
        ),
        "details": (
            f"results/eval_t7_b_{dataset}_v1_details.jsonl"
            if seed == 42
            else f"results/eval_t7_b_s{seed}_{dataset}_v1_details.jsonl"
        ),
        "validation": (
            f"results/eval_t7_b_{dataset}_v1_validation.json"
            if seed == 42
            else f"results/eval_t7_b_s{seed}_{dataset}_v1_validation.json"
        ),
    }
    for seed in (42, 43, 44)
    for dataset in ("synthetic", "refcoco")
)

T7_C_RUNS: tuple[dict[str, Any], ...] = tuple(
    {
        "run_id": f"t7_c_s{seed}_{dataset}_v1" if seed != 42 else f"t7_c_{dataset}_v1",
        "dataset": dataset,
        "split": "eval" if dataset == "synthetic" else "val",
        "data": "data/synthetic_eval.jsonl" if dataset == "synthetic" else "data/refcoco_val.jsonl",
        "n": 1000,
        "adapter": f"adapters/sft_t7_c_pixel{'' if seed == 42 else f'_s{seed}'}/adapter",
        "prompt": "p1_pixel",
        "summary": (
            f"results/zeroshot_t7_c_{dataset}_v1.json"
            if seed == 42
            else f"results/zeroshot_t7_c_s{seed}_{dataset}_v1.json"
        ),
        "details": (
            f"results/eval_t7_c_{dataset}_v1_details.jsonl"
            if seed == 42
            else f"results/eval_t7_c_s{seed}_{dataset}_v1_details.jsonl"
        ),
        "validation": (
            f"results/eval_t7_c_{dataset}_v1_validation.json"
            if seed == 42
            else f"results/eval_t7_c_s{seed}_{dataset}_v1_validation.json"
        ),
    }
    for seed in (42, 43, 44)
    for dataset in ("synthetic", "refcoco")
)

T7_D_RUNS: tuple[dict[str, Any], ...] = tuple(
    {
        "run_id": f"t7_d_s{seed}_{dataset}_v1" if seed != 42 else f"t7_d_{dataset}_v1",
        "dataset": dataset,
        "split": "eval" if dataset == "synthetic" else "val",
        "data": "data/synthetic_eval.jsonl" if dataset == "synthetic" else "data/refcoco_val.jsonl",
        "n": 1000,
        "adapter": f"adapters/sft_t7_d_grid100{'' if seed == 42 else f'_s{seed}'}/adapter",
        "prompt": "p1_grid100",
        "summary": (
            f"results/zeroshot_t7_d_{dataset}_v1.json"
            if seed == 42
            else f"results/zeroshot_t7_d_s{seed}_{dataset}_v1.json"
        ),
        "details": (
            f"results/eval_t7_d_{dataset}_v1_details.jsonl"
            if seed == 42
            else f"results/eval_t7_d_s{seed}_{dataset}_v1_details.jsonl"
        ),
        "validation": (
            f"results/eval_t7_d_{dataset}_v1_validation.json"
            if seed == 42
            else f"results/eval_t7_d_s{seed}_{dataset}_v1_validation.json"
        ),
    }
    for seed in (42, 43, 44)
    for dataset in ("synthetic", "refcoco")
)

T7_RUNS: tuple[dict[str, Any], ...] = T7_B_RUNS + T7_C_RUNS + T7_D_RUNS

SCREENSPOT_RUNS: tuple[dict[str, Any], ...] = (
    {
        "run_id": "screenspot_b0_base_v1",
        "dataset": "screenspot",
        "split": "test",
        "data": "data/screenspot.jsonl",
        "n": 1272,
        "adapter": None,
        "prompt": "p1",
        "summary": "results/zeroshot_screenspot_b0_base_v1.json",
        "details": "results/eval_screenspot_b0_base_v1_details.jsonl",
        "validation": "results/eval_screenspot_b0_base_v1_validation.json",
    },
    *(
        {
            "run_id": f"screenspot_sft_s{seed}_v1",
            "dataset": "screenspot",
            "split": "test",
            "data": "data/screenspot.jsonl",
            "n": 1272,
            "adapter": f"adapters/{SFT_SEED_DIRS[seed]}/adapter",
            "prompt": "p1",
            "summary": f"results/zeroshot_screenspot_sft_s{seed}_v1.json",
            "details": f"results/eval_screenspot_sft_s{seed}_v1_details.jsonl",
            "validation": f"results/eval_screenspot_sft_s{seed}_v1_validation.json",
        }
        for seed in (42, 43, 44)
    ),
)

#: Official deployment evaluation of the merged CPU-fp32 checkpoint (task-29).
#: CPU-only (no GPU lease): it reuses the frozen evaluator through --model-path.
DEPLOY_EVAL_RUN: dict[str, Any] = {
    "run_id": "deploy_cpu_fp32_synthetic_v1",
    "dataset": "synthetic",
    "split": "eval",
    "data": "data/synthetic_eval.jsonl",
    "n": 1000,
    "adapter": None,
    "prompt": "p1",
    "model_path": "adapters/merged/sft_v1_merged",
    "device_override": "cpu",
    "summary": "results/zeroshot_deploy_cpu_fp32_synthetic_v1.json",
    "details": "results/eval_deploy_cpu_fp32_synthetic_v1_details.jsonl",
    "validation": "results/eval_deploy_cpu_fp32_synthetic_v1_validation.json",
}

ALL_EVAL_RUNS = B0_RUNS + SFT_RUNS + T7_RUNS + (DEPLOY_EVAL_RUN,) + SCREENSPOT_RUNS


def display(argv: Iterable[str]) -> str:
    return " ".join(f'"{token}"' if " " in token else token for token in argv)


def command(
    name: str,
    argv: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    requires: Iterable[str] = (),
    skip_if_exists: str | None = None,
    requires_files: Iterable[str] = (),
    est_minutes: float | None = None,
    skip_if_exists_is_dir: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "argv": [str(token) for token in argv],
        "display": display([str(token) for token in argv]),
        "cwd": cwd,
        "env": dict(env or {}),
        "requires": list(requires),
        "skip_if_exists": skip_if_exists,
        "skip_if_exists_is_dir": skip_if_exists_is_dir,
        "requires_files": list(requires_files),
        "est_minutes": est_minutes,
        "notes": notes,
    }


def eval_argv(run: dict[str, Any], *, device: str) -> list[str]:
    argv = [
        PY,
        "-m",
        "src.eval.run_zeroshot",
        "--run-id",
        run["run_id"],
        "--dataset",
        run["dataset"],
        "--split",
        run["split"],
        "--data",
        run["data"],
        "--output",
        run["summary"],
        "--details-output",
        run["details"],
        "--model-path",
        run.get("model_path", MODEL_DIR),
        "--expected-n",
        str(run["n"]),
        "--image-size",
        "384",
        "--max-new-tokens",
        "64",
        "--seed",
        "42",
        "--device",
        device,
        "--batch-size",
        "1",
    ]
    if run["adapter"]:
        argv.extend(["--adapter", run["adapter"]])
    if run["prompt"] != "p1":
        argv.extend(["--prompt-id", run["prompt"]])
    return argv


def validation_argv(run: dict[str, Any]) -> list[str]:
    argv = [
        PY,
        "-m",
        "src.eval.validate_results",
        "--result",
        run["summary"],
        "--report",
        run["validation"],
        "--expected-run-id",
        run["run_id"],
        "--expected-dataset",
        run["dataset"],
        "--expected-split",
        run["split"],
        "--expected-n",
        str(run["n"]),
    ]
    if run["adapter"]:
        argv.extend(["--expected-adapter", run["adapter"]])
    if run["prompt"] != "p1":
        argv.extend(["--expected-prompt-id", run["prompt"]])
    return argv


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------
def stage_check(_: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        command(
            "environment_and_assets",
            [PY, "-m", "src.deploy.check_env", "--force", "--print-summary"],
            est_minutes=0.2,
            notes="read-only probe; overwrites results/deploy_prep.json with the current environment snapshot",
        )
    ]


def stage_data(_: argparse.Namespace) -> list[dict[str, Any]]:
    data_dir = "src/data"
    return [
        command("validate_refcoco", [PY, "-m", "src.data.validate_refcoco"], est_minutes=1.0),
        command(
            "validate_synthetic",
            [PY, "validate_synthetic.py"],
            cwd=data_dir,
            est_minutes=1.5,
            notes="script-style module: imports make_synthetic from its own directory",
        ),
        command("validate_screenspot", [PY, "-m", "src.data.validate_screenspot"], est_minutes=1.0),
        command("test_prepare_refcoco", [PY, "test_prepare_refcoco.py"], cwd=data_dir, est_minutes=0.3),
        command("test_make_synthetic", [PY, "test_make_synthetic.py"], cwd=data_dir, est_minutes=0.5),
        command("test_prepare_screenspot", [PY, "test_prepare_screenspot.py"], cwd=data_dir, est_minutes=0.3),
    ]


def stage_tests(_: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        command("train_t5_cpu", [PY, "-m", "src.train.test_t5_cpu"], est_minutes=1.5),
        command("eval_grounding", [PY, "-m", "src.eval.test_grounding"], est_minutes=0.2),
        command("eval_batch_cpu", [PY, "-m", "src.eval.test_batch_eval_cpu"], est_minutes=1.0),
        command("eval_adapter_support", [PY, "-m", "src.eval.test_adapter_support"], est_minutes=0.5),
        command("eval_processor_forward", [PY, "-m", "src.eval.test_processor_forward"], est_minutes=0.5),
        command("eval_adapter_forward_cpu", [PY, "-m", "src.eval.test_adapter_forward_cpu"], est_minutes=1.0),
        command("eval_t7_variants_cpu", [PY, "-m", "src.eval.test_t7_variants_cpu"], est_minutes=0.3),
        command("eval_t7_grid100_cpu", [PY, "-m", "src.eval.test_grid_convert_cpu"], est_minutes=0.3),
        command("deploy_demo_cpu", [PY, "-m", "src.deploy.run_cpu_tests", "--force"], est_minutes=1.0),
    ]


def training_command(
    adapter_dir: str, config_path: str, *, max_steps: int | None
) -> dict[str, Any]:
    argv = [PY, "-m", "src.train.train_sft", "--config", config_path, "--device", "cuda"]
    if max_steps is not None:
        argv.extend(["--max-steps", str(max_steps)])
    return command(
        f"train_{Path(config_path).stem}",
        argv,
        requires=("gpu", "train"),
        skip_if_exists=adapter_dir,
        skip_if_exists_is_dir=True,
        est_minutes=65.0,
        notes="resumes are explicit: append --resume latest to the printed command; never auto-resumed here",
    )


def stage_train_sft(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = []
    for config_path, adapter_dir in (
        ("configs/sft_v1.json", "adapters/sft_v1/adapter"),
        ("configs/sft_v1_s43.json", "adapters/sft_v1_s43/adapter"),
        ("configs/sft_v1_s44.json", "adapters/sft_v1_s44/adapter"),
    ):
        commands.append(training_command(adapter_dir, config_path, max_steps=1250))
    return commands


def stage_train_t7(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = []
    for config_path, adapter_dir in (
        ("configs/t7_b_bin1000.json", "adapters/sft_t7_b_bin1000/adapter"),
        ("configs/t7_b_bin1000_s43.json", "adapters/sft_t7_b_bin1000_s43/adapter"),
        ("configs/t7_b_bin1000_s44.json", "adapters/sft_t7_b_bin1000_s44/adapter"),
        ("configs/t7_c_pixel.json", "adapters/sft_t7_c_pixel/adapter"),
        ("configs/t7_c_pixel_s43.json", "adapters/sft_t7_c_pixel_s43/adapter"),
        ("configs/t7_c_pixel_s44.json", "adapters/sft_t7_c_pixel_s44/adapter"),
        ("configs/t7_d_grid100.json", "adapters/sft_t7_d_grid100/adapter"),
        ("configs/t7_d_grid100_s43.json", "adapters/sft_t7_d_grid100_s43/adapter"),
        ("configs/t7_d_grid100_s44.json", "adapters/sft_t7_d_grid100_s44/adapter"),
    ):
        commands.append(training_command(adapter_dir, config_path, max_steps=1250))
    return commands


def stage_train_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        training_command("adapters/sft_overfit100/adapter", "configs/sft_overfit100.json", max_steps=200),
        training_command("adapters/sft_overfit100_e48/adapter", "configs/sft_overfit100_e48.json", max_steps=600),
        training_command("adapters/sft_overfit1_e1600/adapter", "configs/sft_overfit1_e1600.json", max_steps=200),
    ]


def eval_stage_commands(
    runs: Iterable[dict[str, Any]], *, cpu_eval: bool, force_rerun: bool
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for run in runs:
        # Per-run device override (the deploy official evaluation is CPU-only
        # regardless of --cpu-eval; the frozen evaluator is reused unchanged).
        device = run.get("device_override") or ("cpu" if cpu_eval else "cuda")
        requires: tuple[str, ...] = () if device == "cpu" else ("gpu",)
        generator = command(
            f"generate_{run['run_id']}",
            eval_argv(run, device=device),
            requires=requires,
            skip_if_exists=run["summary"],
            est_minutes=25.0,
            notes=f"batch_size=1, seed42, image 384 no-split, max_new_tokens 64, device={device}",
        )
        commands.append(generator)
        commands.append(
            command(
                f"validate_{run['run_id']}",
                validation_argv(run),
                requires=(),
                skip_if_exists=run["validation"],
                requires_files=(run["summary"],),
                est_minutes=0.5,
            )
        )
    return commands


def stage_eval_b0(args: argparse.Namespace) -> list[dict[str, Any]]:
    return eval_stage_commands(B0_RUNS, cpu_eval=args.cpu_eval, force_rerun=args.force_rerun)


def stage_eval_sft(args: argparse.Namespace) -> list[dict[str, Any]]:
    return eval_stage_commands(SFT_RUNS, cpu_eval=args.cpu_eval, force_rerun=args.force_rerun)


def stage_eval_t7(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = eval_stage_commands(T7_RUNS, cpu_eval=args.cpu_eval, force_rerun=args.force_rerun)
    commands.append(
        command(
            "build_g4_summary_3seed",
            [PY, "-m", "src.eval.build_g4_summary_3seed"],
            requires_files=tuple(run["summary"] for run in SFT_RUNS + T7_RUNS),
            est_minutes=0.5,
            notes="A/B/C/D all 3-seed (T6 3 seeds + T7 B/C/D 3 seeds each); writes results/eval_t7_g4_summary_3seed.json",
        )
    )
    return commands


def stage_eval_deploy(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Official CPU-fp32 deployment evaluation of the merged checkpoint (task-29)."""
    return eval_stage_commands((DEPLOY_EVAL_RUN,), cpu_eval=True, force_rerun=args.force_rerun)


def stage_eval_screenspot(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = eval_stage_commands(SCREENSPOT_RUNS, cpu_eval=args.cpu_eval, force_rerun=args.force_rerun)
    commands.append(
        command(
            "build_screenspot_summary",
            [PY, "-m", "src.eval.build_screenspot_summary"],
            requires_files=tuple(run["summary"] for run in SCREENSPOT_RUNS),
            est_minutes=0.5,
        )
    )
    return commands


def stage_verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Recompute every existing summary; only missing validation reports are written."""
    commands: list[dict[str, Any]] = []
    for run in ALL_EVAL_RUNS:
        commands.append(
            command(
                f"verify_{run['run_id']}",
                validation_argv(run),
                requires_files=(run["summary"],),
                skip_if_exists=run["validation"],
                est_minutes=0.3,
                notes="existing validation reports are not overwritten",
            )
        )
    commands.append(
        command(
            "verify_t7c_3seed",
            [PY, "-m", "src.eval.verify_t7c_3seed"],
            est_minutes=0.5,
            notes="independent recomputation for the C 3-seed row (writes results/verify_t7c_3seed.json)",
        )
    )
    commands.append(
        command(
            "verify_t7d_3seed",
            [PY, "-m", "src.eval.verify_t7d_3seed"],
            est_minutes=0.5,
            notes="independent recomputation for the D 3-seed row (writes results/verify_t7d_3seed.json)",
        )
    )
    return commands


def stage_tables(_: argparse.Namespace) -> list[dict[str, Any]]:
    g4_inputs = tuple(run["summary"] for run in SFT_RUNS + T7_RUNS)  # A (6) + B (6) + C (6) + D (6)
    return [
        command(
            "build_table1_sft_3seed",
            [PY, "-m", "src.eval.build_3seed_summary"],
            requires_files=tuple(run["summary"] for run in SFT_RUNS),
            est_minutes=0.3,
        ),
        command(
            "build_table_g4_3seed",
            [PY, "-m", "src.eval.build_g4_summary_3seed"],
            requires_files=g4_inputs,
            est_minutes=0.5,
            notes="A/B/C/D all 3-seed; supersedes the earlier single-seed G4 table",
        ),
        command(
            "build_table_screenspot",
            [PY, "-m", "src.eval.build_screenspot_summary"],
            requires_files=tuple(run["summary"] for run in SCREENSPOT_RUNS),
            est_minutes=0.3,
        ),
    ]


def stage_deploy(args: argparse.Namespace) -> list[dict[str, Any]]:
    merge_force = ["--force"] if args.force_merge else []
    return [
        command("deploy_env", [PY, "-m", "src.deploy.check_env", "--force", "--print-summary"], est_minutes=0.2),
        command(
            "merge_lora_fp32",
            [
                PY,
                "-m",
                "src.deploy.merge_lora",
                "--adapter",
                "adapters/sft_v1/adapter",
                "--output",
                "adapters/merged/sft_v1_merged",
                "--dtype",
                "float32",
                "--report",
                "results/deploy_merge_report.json",
                *merge_force,
            ],
            requires_files=(MODEL_DIR + "/model.safetensors", "adapters/sft_v1/adapter/adapter_model.safetensors"),
            skip_if_exists="adapters/merged/sft_v1_merged",
            skip_if_exists_is_dir=True,
            est_minutes=0.3,
            notes="CPU-only fp32 merge; no GPU needed",
        ),
        command(
            "merge_verify_only",
            [
                PY,
                "-m",
                "src.deploy.merge_lora",
                "--verify-only",
                "--artifact",
                "adapters/merged/sft_v1_merged",
                "--report",
                "results/deploy_merge_verify.json",
            ],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            skip_if_exists="results/deploy_merge_verify.json",
            est_minutes=0.1,
        ),
        command(
            "merge_equivalence",
            [
                PY,
                "-m",
                "src.deploy.merge_lora",
                "--verify-equivalence",
                "--artifact",
                "adapters/merged/sft_v1_merged",
                "--report",
                "results/deploy_merge_equivalence.json",
            ],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors", "adapters/sft_v1/adapter/adapter_model.safetensors"),
            skip_if_exists="results/deploy_merge_equivalence.json",
            est_minutes=0.5,
            notes="5 fixed samples; base+adapter vs merged must decode identically",
        ),
        command(
            "latency_cpu_fp32",
            [
                PY,
                "-m",
                "src.deploy.bench_latency",
                "--engine",
                "hf-cpu",
                "--artifact",
                "adapters/merged/sft_v1_merged",
                "--n",
                "50",
                "--warmup",
                "5",
                "--seed",
                "42",
                "--threads",
                "2",
                "--output",
                "results/deploy_latency_fp32.json",
                "--details-output",
                "results/deploy_latency_fp32_details.jsonl",
            ],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            skip_if_exists="results/deploy_latency_fp32.json",
            est_minutes=2.0,
            notes="CPU 2 threads; diagnostic metrics only, official deploy accuracy belongs to eval-closer",
        ),
        command(
            "quantize_gguf_plan",
            [PY, "-m", "src.deploy.quantize", "--route", "gguf", "--plan-only", "--merged", "adapters/merged/sft_v1_merged"],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            est_minutes=0.1,
            notes="plan only: GGUF is blocked in this environment (no llama.cpp); no artifact is produced",
        ),
        command(
            "quantize_onnx_plan",
            [PY, "-m", "src.deploy.quantize", "--route", "onnx", "--plan-only", "--merged", "adapters/merged/sft_v1_merged"],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            est_minutes=0.1,
            notes="plan only: ONNX INT8 is blocked by the transformers<4.58 requirement; no artifact is produced",
        ),
        command(
            "deploy_handoff",
            [
                PY,
                "-m",
                "src.deploy.quantize",
                "--emit-handoff",
                "--route",
                "fp32",
                "--merged",
                "adapters/merged/sft_v1_merged",
                "--quantized-route",
                "torch-int8-dynamic (negative, runtime-only)",
                "--quantized-artifact",
                "none",
            ],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            skip_if_exists="results/deploy_handoff.json",
            est_minutes=0.1,
            notes="hand-off document for independent evaluation of the merged checkpoint",
        ),
    ]


def stage_deploy_int8(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        command(
            "latency_cpu_torch_int8",
            [
                PY,
                "-m",
                "src.deploy.bench_latency",
                "--engine",
                "torch-int8",
                "--artifact",
                "adapters/merged/sft_v1_merged",
                "--n",
                "50",
                "--warmup",
                "5",
                "--seed",
                "42",
                "--threads",
                "2",
                "--output",
                "results/deploy_latency_torch_int8.json",
                "--details-output",
                "results/deploy_latency_torch_int8_details.jsonl",
            ],
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            skip_if_exists="results/deploy_latency_torch_int8.json",
            est_minutes=2.0,
            notes="negative-result reproduction (quality collapse); not a deployment candidate",
        )
    ]


def gradio_env() -> str | None:
    """Return a PYTHONPATH that makes the workspace-local gradio importable."""
    candidate = ROOT / ".cache" / "deploy_pkgs"
    return str(candidate) if candidate.is_dir() else None


def stage_demo(args: argparse.Namespace) -> list[dict[str, Any]]:
    python_path = gradio_env()
    env = {"PYTHONPATH": python_path} if python_path else {}
    commands = [
        command(
            "demo_fake_smoke",
            [PY, "-m", "demo.app", "--engine", "fake", "--smoke", "--smoke-output", "results/deploy_demo_smoke_fake.json"],
            env=env,
            requires=("gradio",),
            skip_if_exists="results/deploy_demo_smoke_fake.json",
            est_minutes=1.0,
            notes="UI plumbing only; the fake predictor is never a model result",
        )
    ]
    commands.append(
        command(
            "demo_real_model_smoke",
            [
                PY,
                "-m",
                "demo.app",
                "--engine",
                "hf-cpu",
                "--artifact",
                "adapters/merged/sft_v1_merged",
                "--smoke",
                "--smoke-output",
                "results/deploy_demo_smoke.json",
            ],
            env=env,
            requires=("gradio", "model"),
            skip_if_exists="results/deploy_demo_smoke.json",
            requires_files=("adapters/merged/sft_v1_merged/model.safetensors",),
            est_minutes=2.0,
            notes="requires --allow-model; browser human confirmation is separate and still pending",
        )
    )
    return commands


STAGE_BUILDERS = {
    "check": stage_check,
    "data": stage_data,
    "tests": stage_tests,
    "train-sft": stage_train_sft,
    "train-t7": stage_train_t7,
    "train-diagnostics": stage_train_diagnostics,
    "eval-b0": stage_eval_b0,
    "eval-sft": stage_eval_sft,
    "eval-t7": stage_eval_t7,
    "eval-deploy": stage_eval_deploy,
    "eval-screenspot": stage_eval_screenspot,
    "verify": stage_verify,
    "tables": stage_tables,
    "deploy": stage_deploy,
    "deploy-int8": stage_deploy_int8,
    "demo": stage_demo,
}


# ---------------------------------------------------------------------------
# Preflight / guards
# ---------------------------------------------------------------------------
def path_exists(value: str, *, is_dir: bool = False) -> bool:
    target = ROOT / value
    return target.is_dir() if is_dir else target.exists()


def nvidia_smi_memory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "memory_used_mib": None, "error": "nvidia-smi not on PATH"}
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "memory_used_mib": None, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False, "memory_used_mib": None, "error": (completed.stderr or "").strip()[:300]}
    values = [token.strip() for token in completed.stdout.splitlines()[0].split(",")]
    used = float(values[0]) if values and values[0] else None
    total = float(values[1]) if len(values) > 1 and values[1] else None
    return {
        "available": True,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "over_limit": bool(used is not None and used > GPU_MEMORY_LIMIT_MIB),
    }


def preflight(args: argparse.Namespace, stage_ids: list[str]) -> list[dict[str, Any]]:
    """Validate flags; return stage-level skip decisions (never runs commands)."""
    if args.dry_run:
        # A dry-run only prints the plan: show the authorisation skips instead of
        # failing, and do not even query the GPU.
        return [{"gpu": {"available": None, "note": "not queried during dry-run"}}]
    problems: list[str] = []
    requested_train = [stage for stage in stage_ids if stage in TRAIN_STAGES]
    requested_gpu = [stage for stage in stage_ids if stage in GPU_EVAL_STAGES and not args.cpu_eval]
    explicit_all = any(
        part.strip() == "all" for value in args.stage for part in str(value).split(",")
    )

    # Explicitly requested expensive stages must carry their authorisation.
    # `--stage all` is different: it reports unauthorised stages as skipped,
    # so a single command can show what would run without silently training.
    if requested_train and not args.allow_train and not explicit_all:
        problems.append(
            "--allow-train is required for training stages; refusing to run a training stage silently "
            f"(requested: {', '.join(requested_train)})"
        )
    if (requested_train or requested_gpu) and not args.gpu_lease_id and not explicit_all:
        problems.append(
            "a GPU lease id is required for any GPU stage; pass --gpu-lease-id <LEASE-ID> "
            f"(requested: {', '.join(requested_train + requested_gpu)})"
        )
    if args.cpu_eval and requested_train:
        problems.append("--cpu-eval only applies to evaluation stages; training must run on GPU")
    if problems:
        raise SystemExit("preflight_failed: " + " | ".join(problems))

    gpu_state = nvidia_smi_memory()
    if gpu_state.get("over_limit"):
        raise SystemExit(
            f"preflight_failed: GPU memory {gpu_state['memory_used_mib']} MiB exceeds the "
            f"{GPU_MEMORY_LIMIT_MIB} MiB safety limit; stop and report to the Lead"
        )
    return [{"gpu": gpu_state}]


def gpu_guard_after(stage_id: str, command_name: str) -> None:
    state = nvidia_smi_memory()
    if state.get("over_limit"):
        raise SystemExit(
            f"gpu_guard: {stage_id}/{command_name} left GPU memory at {state['memory_used_mib']} MiB "
            f"(>{GPU_MEMORY_LIMIT_MIB}); aborting remaining commands"
        )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def stage_specs(args: argparse.Namespace, stage_ids: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for stage_id in stage_ids:
        builder = STAGE_BUILDERS[stage_id]
        specs.append({"id": stage_id, "title": STAGE_TITLES[stage_id], "commands": builder(args)})
    return specs


def should_skip(args: argparse.Namespace, spec: dict[str, Any], item: dict[str, Any]) -> str | None:
    requires = item["requires"]
    if "train" in requires and not args.allow_train:
        return "training not authorised (missing --allow-train)"
    if "gpu" in requires and not args.gpu_lease_id:
        return "GPU work not authorised (missing --gpu-lease-id)"
    if "gpu" in requires and args.cpu_eval:
        return "GPU work skipped because --cpu-eval was requested"
    if "model" in requires and not args.allow_model:
        return "model load not authorised (missing --allow-model)"
    if "gradio" in requires:
        python_path = gradio_env()
        if python_path is None:
            return "gradio not available: workspace-local .cache/deploy_pkgs missing"
        try:
            probe = subprocess.run(
                [PY, "-c", "import gradio, gradio_client"],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYTHONPATH": python_path},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"gradio probe failed: {type(exc).__name__}: {exc}"
        if probe.returncode != 0:
            return "gradio import failed under PYTHONPATH=.cache/deploy_pkgs"
    missing = [path for path in item["requires_files"] if not path_exists(path)]
    if missing:
        return f"prerequisite artifacts missing (not_run): {', '.join(missing)}"
    target = item["skip_if_exists"]
    if target and not ((args.force_retrain and "train" in requires) or (args.force_merge and "merge_lora" in item["name"]) or args.force_rerun):
        if target.endswith("/") or path_exists(target, is_dir=item["skip_if_exists_is_dir"]):
            return f"artifact already exists, refusing to overwrite: {target}"
    return None


def run_command(item: dict[str, Any], *, log_dir: Path, stage_id: str, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": item["name"],
        "display": item["display"],
        "cwd": item["cwd"] or ".",
        "status": "pending",
        "exit_code": None,
        "seconds": None,
        "log": None,
        "reason": None,
        "gpu_before": None,
        "gpu_after": None,
        "notes": item.get("notes"),
    }
    if dry_run:
        record["status"] = "planned"
        record["log"] = (log_dir / stage_id / f"{item['name']}.log").relative_to(ROOT).as_posix()
        return record
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **item["env"]}
    cwd = ROOT / item["cwd"] if item["cwd"] else ROOT
    command_log = log_dir / stage_id / f"{item['name']}.log"
    command_log.parent.mkdir(parents=True, exist_ok=True)
    if "gpu" in item["requires"]:
        record["gpu_before"] = nvidia_smi_memory()
    started = time.perf_counter()
    with command_log.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {datetime.now(LOCAL_TZ).isoformat()} stage={stage_id} command={item['name']}\n")
        handle.write(f"# cwd={cwd}\n# {item['display']}\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                item["argv"],
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            exit_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            handle.write(f"\n# exception: {type(exc).__name__}: {exc}\n")
            exit_code = -1
    record["exit_code"] = exit_code
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["log"] = command_log.relative_to(ROOT).as_posix()
    record["status"] = "ok" if exit_code == 0 else "failed"
    if "gpu" in item["requires"]:
        record["gpu_after"] = nvidia_smi_memory()
        if exit_code == 0:
            gpu_guard_after(stage_id, item["name"])
    return record


def run_stages(args: argparse.Namespace) -> int:
    stage_ids = resolve_stages(args)
    if not stage_ids:
        raise SystemExit("no stages selected")
    specs = stage_specs(args, stage_ids)
    preflight(args, stage_ids)

    stamp = datetime.now(LOCAL_TZ).strftime("%Y%m%dT%H%M%S%z") + f"-{os.getpid()}"
    log_dir = Path(args.log_root)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir = log_dir / stamp

    manifest: dict[str, Any] = {
        "format": "run_all_manifest_v1",
        "started_at": datetime.now(LOCAL_TZ).isoformat(),
        "root": str(ROOT),
        "python": PY,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "dry_run": bool(args.dry_run),
        "flags": {
            "allow_train": bool(args.allow_train),
            "allow_model": bool(args.allow_model),
            "gpu_lease_id": args.gpu_lease_id,
            "cpu_eval": bool(args.cpu_eval),
            "force_retrain": bool(args.force_retrain),
            "force_merge": bool(args.force_merge),
            "force_rerun": bool(args.force_rerun),
            "keep_going": bool(args.keep_going),
        },
        "stages": [],
        "notes": [
            "dry-run does not execute commands and creates no logs",
            "training/eval stages are explicit; --stage all skips unauthorised expensive stages with a reason",
            "results/report artifacts are never overwritten unless an explicit force flag is used",
        ],
    }

    total_commands = 0
    ok_commands = 0
    skipped_commands = 0
    failed_commands = 0
    estimate_minutes = 0.0
    failed_any = False

    for spec in specs:
        stage_record: dict[str, Any] = {
            "stage": spec["id"],
            "title": spec["title"],
            "status": "dry_run" if args.dry_run else "running",
            "reason": None,
            "commands": [],
            "seconds": None,
        }
        stage_started = time.perf_counter()
        if not args.dry_run:
            print(f"\n== stage {spec['id']}: {spec['title']} ==", flush=True)
        for item in spec["commands"]:
            total_commands += 1
            skip_reason = should_skip(args, spec, item)
            if skip_reason:
                stage_record["commands"].append(
                    {
                        "name": item["name"],
                        "display": item["display"],
                        "status": "skipped",
                        "reason": skip_reason,
                        "notes": item.get("notes"),
                    }
                )
                skipped_commands += 1
                continue
            if args.dry_run:
                record = run_command(item, log_dir=log_dir, stage_id=spec["id"], dry_run=True)
                record["est_minutes"] = item.get("est_minutes")
                if item.get("est_minutes"):
                    estimate_minutes += float(item["est_minutes"])
            else:
                print(f"  RUN   {item['name']}: {item['display']}", flush=True)
                record = run_command(item, log_dir=log_dir, stage_id=spec["id"], dry_run=False)
            stage_record["commands"].append(record)
            if record["status"] in ("ok", "planned"):
                ok_commands += 1
            elif record["status"] == "failed":
                failed_commands += 1
                failed_any = True
                if not args.keep_going:
                    stage_record["status"] = "failed"
                    stage_record["seconds"] = round(time.perf_counter() - stage_started, 2)
                    manifest["stages"].append(stage_record)
                    manifest["finished_at"] = datetime.now(LOCAL_TZ).isoformat()
                    manifest["summary"] = summarize(args, total_commands, ok_commands, skipped_commands, failed_commands, estimate_minutes)
                    write_manifest(args, manifest, log_dir, stamp)
                    print_manifest(manifest)
                    return 1
        if stage_record["commands"] and all(command["status"] == "skipped" for command in stage_record["commands"]):
            stage_record["status"] = "skipped"
            reasons = sorted({command["reason"] for command in stage_record["commands"] if command.get("reason")})
            stage_record["reason"] = "; ".join(reason for reason in reasons if reason)
        elif failed_any and any(command["status"] == "failed" for command in stage_record["commands"]):
            stage_record["status"] = "failed"
        elif args.dry_run:
            stage_record["status"] = "dry_run"
        else:
            stage_record["status"] = "ok"
        stage_record["seconds"] = round(time.perf_counter() - stage_started, 2)
        manifest["stages"].append(stage_record)
        if failed_any and not args.keep_going:
            break

    manifest["finished_at"] = datetime.now(LOCAL_TZ).isoformat()
    manifest["summary"] = summarize(args, total_commands, ok_commands, skipped_commands, failed_commands, estimate_minutes)
    write_manifest(args, manifest, log_dir, stamp)
    print_manifest(manifest)
    return 1 if failed_commands else 0


def summarize(
    args: argparse.Namespace,
    total: int,
    ok: int,
    skipped: int,
    failed: int,
    estimate_minutes: float,
) -> dict[str, Any]:
    return {
        "total_commands": total,
        "ok_or_planned": ok,
        "skipped": skipped,
        "failed": failed,
        "estimated_minutes_planned": round(estimate_minutes, 1),
        "dry_run": bool(args.dry_run),
        "result": "dry_run" if args.dry_run else ("failed" if failed else "ok"),
    }


def write_manifest(args: argparse.Namespace, manifest: dict[str, Any], log_dir: Path, stamp: str) -> None:
    if args.dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest["log_dir"] = log_dir.relative_to(ROOT).as_posix() if log_dir.is_relative_to(ROOT) else str(log_dir)
    manifest["stamp"] = stamp
    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = Path(args.log_root)
    if not latest.is_absolute():
        latest = ROOT / latest
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "LATEST").write_text(stamp + "\n", encoding="utf-8")


def print_manifest(manifest: dict[str, Any]) -> None:
    mode = "DRY-RUN (no commands executed, no logs written)" if manifest["dry_run"] else "EXECUTION"
    print(f"== run_all {mode} ==")
    for stage in manifest["stages"]:
        print(f"\n[{stage['stage']}] {stage['title']} -> {stage['status']}" + (f" ({stage['reason']})" if stage.get("reason") else ""))
        for item in stage["commands"]:
            status = item["status"]
            if status == "skipped":
                print(f"  SKIP  {item['name']}: {item['reason']}")
            elif status == "planned":
                print(f"  PLAN  {item['name']}  (~{item.get('est_minutes')} min)  log: {item['log']}")
                print(f"        {item['display']}")
                if item.get("notes"):
                    print(f"        note: {item['notes']}")
            else:
                line = f"  {status.upper():4s}  {item['name']}  exit={item['exit_code']}  {item['seconds']}s  log: {item['log']}"
                print(line)
                if status == "failed":
                    print(f"        {item['display']}")
    summary = manifest["summary"]
    print(
        f"\nsummary: total={summary['total_commands']} ok={summary['ok_or_planned']} "
        f"skipped={summary['skipped']} failed={summary['failed']} "
        f"estimated_minutes={summary['estimated_minutes_planned']} result={summary['result']}"
    )
    if not manifest["dry_run"]:
        print(f"manifest: {Path(manifest.get('log_dir', 'logs/run_all'))}")


def resolve_stages(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    for value in args.stage:
        requested.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not requested:
        return []
    if "all" in requested:
        return list(ALL_ORDER)
    unknown = [stage for stage in requested if stage not in STAGE_BUILDERS]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}; see --list-stages")
    # Stable, documented order: the canonical ALL_ORDER, filtered by the request.
    return [stage for stage in ALL_ORDER if stage in set(requested)]


def list_stages() -> None:
    print("Available stages (order used by --stage all):")
    for stage_id in ALL_ORDER:
        flags = []
        if stage_id in TRAIN_STAGES:
            flags.append("--allow-train")
        if stage_id in GPU_EVAL_STAGES:
            flags.append("--gpu-lease-id (or --cpu-eval)")
        if stage_id == "demo":
            flags.append("--allow-model for the real-model smoke")
        suffix = ("  [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {stage_id:18s} {STAGE_TITLES[stage_id]}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", action="append", default=[], help="stage id (repeatable or comma separated); default: check")
    parser.add_argument("--list-stages", action="store_true", help="print available stages and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the exact commands and skip decisions; execute nothing")
    parser.add_argument("--allow-train", action="store_true", help="allow training stages to run (GPU lease still required)")
    parser.add_argument("--allow-model", action="store_true", help="allow the real-model demo smoke (CPU model load)")
    parser.add_argument("--gpu-lease-id", default=None, help="Lead-issued GPU lease id required by any GPU stage")
    parser.add_argument("--cpu-eval", action="store_true", help="run evaluation stages on CPU instead of CUDA (slow but no lease)")
    parser.add_argument("--force-retrain", action="store_true", help="allow a training stage to overwrite an existing adapter")
    parser.add_argument("--force-merge", action="store_true", help="allow the deploy merge stage to replace an existing merged checkpoint")
    parser.add_argument("--force-rerun", action="store_true", help="allow re-running eval/demo commands whose artifacts already exist")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed command")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT.relative_to(ROOT)), help="log root directory")
    parser.add_argument("--python", default=None, help="python executable to use for stage commands (default: current)")
    return parser


def main(argv: list[str] | None = None) -> int:
    global PY
    args = build_parser().parse_args(argv)
    if args.python:
        PY = args.python
    if args.list_stages:
        list_stages()
        return 0
    if not args.stage:
        args.stage = ["check"]
    return run_stages(args)


if __name__ == "__main__":
    raise SystemExit(main())
