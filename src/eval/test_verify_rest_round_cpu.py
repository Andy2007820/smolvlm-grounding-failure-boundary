#!/usr/bin/env python3
"""CPU tests for the generic round verifier (``src/eval/verify_rest_round.py``).

The round verifier reuses the frozen round-1 check implementations.  These tests
prove that reuse is sound *before* any round-2 artefact exists:

1. running the generic verifier against the **existing round-1 artefacts** (via an
   injected ``r1`` spec) must reproduce round 1's verdict: 6/6 checks passed, no
   problems, ``status=passed``;
2. a project root without any evaluation artefacts must resolve to
   ``status=pending`` (never crash, exit-code semantics 0);
3. the round-2 spec must resolve the round-2 file names (not the round-1 ones),
   so a finished round 2 can never silently be verified against round 1;
4. the training hard-counter check must actually fire on a broken summary.

Temp roots live under ``artifacts/tmp/`` inside the workspace (the sandbox denies
cleanup of some platform temp paths).
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]
for _candidate in (str(_HERE), str(PROJECT_ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import verify_rest_round as vr  # noqa: E402

R1_SPEC = {
    "tag": "r1",
    "run_id_round": "rest_r1",
    "summary_templates": ("results/zeroshot_rest_r1_s42_{dataset}_v1.json",),
    "summary_globs": (
        "results/zeroshot_rest_r1*s42*{dataset}*.json",
        "results/zeroshot_rest_r1*{dataset}*.json",
    ),
    "details_globs": ("results/eval_rest_r1*{dataset}*details.jsonl",),
    "validation_globs": ("results/eval_rest_r1*{dataset}*validation.json",),
    "training_candidates": (
        "results/train_rest_r1_s42_training.json",
        "results/train_rest_r1_training.json",
    ),
    "training_glob": "results/train_*rest_r1*.json",
    "training_exclude": ("probe", "dryrun", "pool", "sample", "manifest"),
    "adapter_candidates": (
        "adapters/rest_r1/adapter/adapter_model.safetensors",
        "adapters/rest_r1/adapter_model.safetensors",
    ),
    "adapter_glob": "adapters/rest_r1/**/adapter_model.safetensors",
    "required_pools": ("data/rest/r1_pool.jsonl",),
    "report_re": r"rest_r1",
}


def _workspace_tempdir(testcase: unittest.TestCase) -> Path:
    path = PROJECT_ROOT / "artifacts" / "tmp" / f"verify-round-test-{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=True)
    testcase.addCleanup(shutil.rmtree, path, True)
    return path


def _load_scorers():
    from eval import grounding, pixel_convert  # noqa: PLC0415

    return grounding, pixel_convert


class GenericVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grounding, cls.pixel_convert = _load_scorers()
        vr.ROUNDS["r1"] = R1_SPEC

    def test_round1_artifacts_reproduce_six_passed_checks(self) -> None:
        report = vr.run_verification(
            PROJECT_ROOT,
            "r1",
            self.grounding,
            self.pixel_convert,
            expected_n=1000,
            tolerance=1e-12,
            check_report_literals=False,
        )
        problems = {check["id"]: check["problems"] for check in report["checks"] if check["status"] != "passed"}
        self.assertEqual(report["status"], "passed", json.dumps(problems, ensure_ascii=False, indent=2))
        self.assertEqual(report["failed_checks"], [])
        check_ids = [check["id"] for check in report["checks"]]
        self.assertEqual(
            check_ids,
            [
                "details_counts",
                "metrics_recompute",
                "details_sha256",
                "validation_report",
                "training_hardcounts",
                "adapter_sha256",
            ],
        )
        for check in report["checks"]:
            self.assertEqual(check["problems"], [], check["id"])

    def test_empty_root_is_pending_not_crashing(self) -> None:
        root = _workspace_tempdir(self)
        report = vr.run_verification(
            root,
            "r2",
            self.grounding,
            self.pixel_convert,
            expected_n=1000,
            tolerance=1e-12,
            check_report_literals=False,
        )
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["checks"], [])
        self.assertTrue(report["missing"])

    def test_r2_spec_resolves_only_round2_names(self) -> None:
        spec = vr.ROUNDS["r2"]
        self.assertIn("rest_r2", spec["summary_templates"][0])
        self.assertIn("rest_r2", spec["training_candidates"][0])
        self.assertIn("rest_r2", spec["adapter_candidates"][0])
        self.assertIn("data/rest/r2_pool.jsonl", spec["required_pools"])
        resolution = vr.resolve_inputs(PROJECT_ROOT, spec)
        # Round 1 exists but round 2 does not yet: every dataset must be missing.
        for dataset in vr.DATASETS:
            self.assertFalse(resolution["datasets"][dataset]["ready"], dataset)
        self.assertIsNone(resolution["training_json"])
        self.assertIsNone(resolution["adapter"])

    def test_hardcounts_detects_broken_training_summary(self) -> None:
        root = _workspace_tempdir(self)
        (root / "results").mkdir()
        training = json.loads(
            (PROJECT_ROOT / "results/train_rest_r1_s42_training.json").read_text(encoding="utf-8")
        )
        broken = dict(training)
        broken["stop_reason"] = "interrupted"
        broken["final_adapter_summary"] = dict(broken.get("final_adapter_summary") or {}, complete=False)
        stats = dict(broken.get("accumulation_stats") or {})
        stats["forward_calls"] = 123
        broken["accumulation_stats"] = stats
        path = root / "results/train_rest_r1_s42_training.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        ctx = {"root": root, "resolution": {"training_json": path}, "spec": R1_SPEC}
        check = vr.check_training_hardcounts(ctx)
        self.assertEqual(check["status"], "failed")
        joined = " ".join(check["problems"])
        self.assertIn("final_adapter_summary.complete", joined)
        self.assertIn("stop_reason", joined)
        self.assertIn("forward_calls", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
