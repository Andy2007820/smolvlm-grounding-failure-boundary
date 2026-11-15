#!/usr/bin/env python3
"""T10 (P6): verifier-noise sensitivity of the ReST acceptance pools — CPU replay.

Replays the *already recorded* per-generation IoU audit (data/rest/r1_pool_audit.jsonl
and data/rest/r2_pool_audit.jsonl, 4,000 generations each) under a noisy verifier:

    accept_noisy = accept_true XOR flip,  flip ~ Bernoulli(p) with p in {0, 0.1, 0.2, 0.3}

i.e. the verifier's accept/reject decision is corrupted with symmetric probability p
(false accepts and false rejects equally likely).  For each p the script reports the
pool that the noisy verifier would have kept, and the *pool-quality* consequences:

  * pool size (and per-dataset split);
  * contamination = accepted rows whose TRUE IoU < 0.5 (false accepts) — the noisy
    rows the student would be trained on;
  * lost_good = true-accepted rows that the noisy verifier rejects (false rejects);
  * pool-implied Acc@0.5 = fraction of accepted rows with true IoU >= 0.5;
  * pool-implied mIoU = mean true IoU over accepted rows.

Scope red line (recorded in the output): this is a **replay of verifier decisions, not
a retraining experiment**.  Downstream model accuracy after retraining on a corrupted
pool is NOT measured here; the paper must say so explicitly (PLAN §P6 口径).

Usage: python -m scripts.t10_verifier_noise [--trials 200] [--no-write]
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDITS = {
    "r1": "data/rest/r1_pool_audit.jsonl",
    "r2": "data/rest/r2_pool_audit.jsonl",
}
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)
TAU = 0.5


def load_audit(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty audit: {path}")
    return rows


def summarise(rows: list[dict[str, Any]], keep: list[bool]) -> dict[str, Any]:
    accepted = [row for row, flag in zip(rows, keep) if flag]
    true_accepted = [row for row in rows if row["iou"] >= TAU]
    true_ids = {row["id"] for row in true_accepted}
    accepted_ids = {row["id"] for row in accepted}
    false_accepts = [row for row in accepted if row["iou"] < TAU]
    false_rejects = [row for row in rows if row["id"] in true_ids and row["id"] not in accepted_ids]
    ious = [row["iou"] for row in accepted]
    return {
        "pool_size": len(accepted),
        "by_dataset": {
            dataset: sum(1 for row in accepted if (row.get("dataset") or row.get("source")) == dataset)
            for dataset in ("synthetic", "refcoco")
        },
        "contamination_rate": (len(false_accepts) / len(accepted)) if accepted else 0.0,
        "false_accepts": len(false_accepts),
        "false_rejects": len(false_rejects),
        "lost_good_rate": (len(false_rejects) / len(true_accepted)) if true_accepted else 0.0,
        "pool_acc_at_0_5": (sum(1 for row in accepted if row["iou"] >= TAU) / len(accepted)) if accepted else 0.0,
        "pool_miou": (statistics.fmean(ious) if ious else 0.0),
    }


def replay(rows: list[dict[str, Any]], p: float, trials: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    baseline_flag = [row["iou"] >= TAU for row in rows]
    per_trial: list[dict[str, Any]] = []
    for _ in range(trials):
        keep: list[bool] = []
        for flag in baseline_flag:
            flip = rng.random() < p
            keep.append((not flag) if flip else flag)
        per_trial.append(summarise(rows, keep))

    def agg(key: str) -> dict[str, float]:
        values = [trial[key] for trial in per_trial]
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    deterministic = summarise(rows, baseline_flag)
    return {
        "noise_p": p,
        "trials": trials,
        "true_verifier_reference": deterministic,
        "pool_size": agg("pool_size"),
        "contamination_rate": agg("contamination_rate"),
        "pool_acc_at_0_5": agg("pool_acc_at_0_5"),
        "pool_miou": agg("pool_miou"),
        "false_accepts": agg("false_accepts"),
        "false_rejects": agg("false_rejects"),
        "lost_good_rate": agg("lost_good_rate"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T10 verifier-noise replay (CPU only)")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.tau != TAU:
        raise SystemExit("this replay is frozen at tau=0.5 (the τ used by both ReST rounds)")

    rounds: dict[str, Any] = {}
    for tag, rel in AUDITS.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"missing audit: {rel}")
        rows = load_audit(path)
        rounds[tag] = {
            "audit": rel,
            "audit_rows": len(rows),
            "generations": len(rows),
            "noise_levels": [replay(rows, p, args.trials, args.seed) for p in NOISE_LEVELS],
        }

    payload = {
        "schema": "t10_verifier_noise/v1",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tau": args.tau,
        "noise_model": "symmetric accept/reject flip with probability p on each recorded generation decision",
        "trials_per_level": args.trials,
        "seed": args.seed,
        "rounds": rounds,
        "scope_red_line": (
            "Replay of verifier decisions on recorded audits (CPU only). Downstream model accuracy after "
            "retraining on a noisy pool is NOT measured; no training was run. Paper must state this."
        ),
        "license_note": "single-seed pilot context; pool-quality deltas are not significance claims",
    }

    output_path = PROJECT_ROOT / "results/t10_verifier_noise.json"
    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": None if args.no_write else str(output_path.relative_to(PROJECT_ROOT).as_posix()),
                "rounds": {
                    tag: {
                        "rows": info["audit_rows"],
                        "levels": {
                            str(level["noise_p"]): {
                                "pool_size_mean": round(level["pool_size"]["mean"], 1),
                                "contamination_mean": round(level["contamination_rate"]["mean"], 4),
                                "pool_acc_mean": round(level["pool_acc_at_0_5"]["mean"], 4),
                                "pool_miou_mean": round(level["pool_miou"]["mean"], 4),
                            }
                            for level in info["noise_levels"]
                        },
                    }
                    for tag, info in rounds.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
