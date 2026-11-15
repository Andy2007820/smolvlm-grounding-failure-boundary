"""CPU-only unit tests for the T14/T15 deployment skeletons.

Hard constraints enforced here (and recorded in ``results/deploy_cpu_tests.json``):

* ``CUDA_VISIBLE_DEVICES=""`` is set before any import;
* no model weights are loaded, no training/benchmark runs, no network;
* ``torch`` must never be imported by the prep-facing modules.

The tests that execute subprocesses only use ``--dry-run`` / ``--plan-only``
modes, which are designed to stay CPU-light.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.deploy.config import (  # noqa: E402
    BENCH_SAMPLES,
    MODEL_PATH,
    PROJECT_ROOT,
    checked_results_path,
    expected_image_seq_len,
    latency_stats,
    prior_cpu_latency_evidence,
    read_jsonl,
    select_benchmark_records,
    selected_ids_digest,
)
from src.deploy.env import (  # noqa: E402
    adapter_facts,
    collect_environment,
    merged_facts,
    model_facts,
    onnx_asset_inventory,
    packages,
    quantized_facts,
)
from src.deploy.infer import classify_artifact  # noqa: E402
from src.deploy.merge_lora import validate_merge_inputs, verify_merged_directory  # noqa: E402
from src.deploy.plan import (  # noqa: E402
    FALLBACK_CHAIN,
    build_risks,
    build_steps,
    route_statuses,
)
from src.deploy.quantize import (  # noqa: E402
    build_fp32_plan,
    build_gguf_plan,
    build_handoff,
    build_onnx_plan,
    build_torch_int8_plan,
)

PYTHON = sys.executable or "python"
SUBPROCESS_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
# The DSH sandbox may deny writes under the OS temp directory, so tests keep
# their scratch directories inside the writable workspace.
TMP_ROOT = PROJECT_ROOT / ".cache" / "deploy_tests"


class TempDirMixin(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def make_temp_dir(self) -> Path:
        # ``tempfile.mkdtemp`` uses mode 0o700 and the DSH Windows sandbox then
        # denies writes inside that directory, so create it with the default mode.
        path = TMP_ROOT / f"case_{uuid.uuid4().hex[:10]}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = collect_environment()

    def test_detection_does_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("gradio", sys.modules)

    def test_expected_versions_are_detected(self):
        packages_payload = self.environment["packages"]
        self.assertTrue(packages_payload["torch"]["available"])
        self.assertTrue(packages_payload["transformers"]["available"])
        self.assertTrue(packages_payload["peft"]["available"])
        self.assertTrue(packages_payload["onnxruntime"]["available"])
        self.assertIsInstance(packages_payload["torch"]["version"], str)
        self.assertTrue(packages_payload["torch"]["version"].startswith("2.11"))

    def test_missing_optional_tools_are_reported_not_faked(self):
        packages_payload = self.environment["packages"]
        self.assertFalse(packages_payload["gradio"]["available"])
        self.assertFalse(packages_payload["optimum"]["available"])
        self.assertFalse(packages_payload["gguf"]["available"])
        llama = self.environment["tools"]["llama_cpp"]
        self.assertFalse(llama["available"])
        self.assertTrue(llama["missing"])

    def test_gpu_snapshot_is_read_only_metadata(self):
        gpu = self.environment["gpu"]
        self.assertIn("available", gpu)
        self.assertFalse(gpu.get("gpu_compute_started_by_this_module", False))
        self.assertEqual(self.environment["prep_policy"]["gpu_compute_used"], False)
        self.assertEqual(self.environment["prep_policy"]["model_loaded"], False)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = collect_environment()

    def test_steps_are_ordered_and_have_commands(self):
        steps = build_steps(self.environment)
        self.assertEqual([step["id"] for step in steps], [f"S{index}" for index in range(1, 11)])
        for step in steps:
            self.assertGreater(step["estimated_minutes"], 0, step["id"])
            self.assertTrue(step["acceptance"], step["id"])
            self.assertIn("command", step)
            self.assertIsInstance(step["command"]["argv"], list, step["id"])
            self.assertTrue(step["command"]["argv"], step["id"])

    def test_only_gpu_and_download_steps_declare_their_needs(self):
        steps = {step["id"]: step for step in build_steps(self.environment)}
        self.assertTrue(steps["S8"]["requires_gpu"])
        self.assertFalse(steps["S1"]["requires_gpu"])
        self.assertFalse(steps["S6"]["requires_gpu"])
        self.assertTrue(steps["S3"]["requires_download"])
        self.assertTrue(steps["S9"]["requires_download"])

    def test_fallback_chain_order_is_the_documented_one(self):
        self.assertEqual(list(FALLBACK_CHAIN), ["gguf", "onnx", "torch-int8", "hf-fp32"])

    def test_route_statuses_report_blockers(self):
        routes = route_statuses(self.environment)
        self.assertEqual(set(routes), {"gguf", "onnx", "torch-int8", "hf-fp32"})
        self.assertIn(routes["gguf"]["status"], {"ready", "needs_tooling_and_download"})
        self.assertFalse(routes["gguf"]["gpu_required"])
        self.assertIn("optimum", routes["onnx"]["missing"])
        self.assertEqual(routes["torch-int8"]["status"], "ready")
        self.assertEqual(routes["hf-fp32"]["status"], "ready")

    def test_risks_cover_integrity_and_4bit_claims(self):
        routes = route_statuses(self.environment)
        risks = build_risks(self.environment, routes)
        ids = {risk["id"] for risk in risks}
        self.assertGreaterEqual(len(ids), 8)
        joined = json.dumps(risks, ensure_ascii=False)
        self.assertIn("BASE model without our LoRA", joined)
        self.assertIn("CPU fp32 is NOT faster", joined)
        for risk in risks:
            for key in ("risk", "probability", "impact", "mitigation", "owner"):
                self.assertIn(key, risk)


class AssetTests(unittest.TestCase):
    def test_adapter_identity_and_hash(self):
        adapter = adapter_facts(with_hash=True)
        self.assertTrue(adapter["all_required_files_present"])
        self.assertEqual(adapter["r"], 16)
        self.assertEqual(adapter["lora_alpha"], 32)
        self.assertTrue(adapter["base_model_matches_project_model"])
        self.assertGreaterEqual(adapter["target_module_count"], 90)
        self.assertTrue(adapter["target_modules_include_q_proj"])
        self.assertEqual(len(adapter["adapter_weights"]["sha256"]), 64)
        self.assertGreater(adapter["adapter_weights"]["bytes"], 1_000_000)

    def test_model_facts_show_384_no_split_contract(self):
        model = model_facts(with_hash=False)
        self.assertTrue(model["all_required_files_present"])
        self.assertEqual(model["architecture"], "Idefics3ForConditionalGeneration")
        self.assertEqual(model["vision_patch_size"], 16)
        self.assertEqual(model["vision_scale_factor"], 4)
        self.assertEqual(model["processor_default_image_seq_len"], 64)
        self.assertEqual(model["expected_runtime_image_seq_len"], 36)
        self.assertFalse(model["processor_default_matches_384_no_split"])
        self.assertEqual(model["generation_eos_token_id"], 49279)

    def test_asset_facts_track_reality_not_intent(self):
        # Execution (T14) created the merged checkpoint. On 2026-11-11 the P8
        # 4-bit GGUF route produced real artifacts under adapters/quantized/gguf
        # (Q4_K_M text + Q8_0 mmproj), so the facts must report them: this test
        # tracks reality, not the earlier "quantized dir is empty" intent.
        merged = merged_facts()
        self.assertTrue(merged["exists"])
        self.assertFalse(merged["missing_required_files"])
        quantized = quantized_facts()
        self.assertEqual(quantized["status"], "has_artifacts")
        paths = sorted(item["path"] for item in quantized["artifacts"])
        self.assertEqual(
            paths,
            [
                "adapters/quantized/gguf/mmproj-sft_v1_merged-Q8_0.gguf",
                "adapters/quantized/gguf/sft_v1_merged-Q4_K_M.gguf",
            ],
        )
        for item in quantized["artifacts"]:
            self.assertEqual(item["suffix"], ".gguf")
            self.assertGreater(item["bytes"], 1_000_000)

    def test_upstream_onnx_assets_cannot_carry_the_lora(self):
        inventory = onnx_asset_inventory()
        self.assertTrue(inventory["exists"])
        self.assertGreaterEqual(len(inventory["files"]), 20)
        self.assertGreater(inventory["total_bytes"], 1_000_000_000)
        self.assertFalse(inventory["carries_project_lora"])
        self.assertTrue(inventory["has_split_pipeline"])
        self.assertIn("BASE", inventory["scientific_note"])
        self.assertTrue(all(item["carries_project_lora"] is False for item in inventory["files"]))

    def test_classify_artifact_real_paths(self):
        adapter = classify_artifact(PROJECT_ROOT / "adapters" / "sft_v1" / "adapter")
        self.assertEqual(adapter["kind"], "adapter")
        self.assertTrue(adapter["usable"])
        base = classify_artifact(MODEL_PATH)
        self.assertEqual(base["kind"], "base")
        unknown = classify_artifact(PROJECT_ROOT / "does-not-exist")
        self.assertEqual(unknown["kind"], "unknown")
        self.assertFalse(unknown["usable"])


class MergeTests(TempDirMixin):
    def test_merge_input_validation_accepts_the_real_adapter(self):
        plan = validate_merge_inputs(MODEL_PATH, PROJECT_ROOT / "adapters" / "sft_v1" / "adapter", PROJECT_ROOT / "adapters" / "merged" / "tmp_probe")
        self.assertEqual(plan["dtype"], "float32")
        self.assertGreater(plan["expected_output_bytes"], 1_000_000_000)
        self.assertTrue(plan["provisional"])

    def test_merge_input_validation_rejects_wrong_base(self):
        temporary = self.make_temp_dir()
        adapter_dir = temporary / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "base_model_name_or_path": "D:/somewhere/else",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj"],
                }
            ),
            encoding="utf-8",
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
        with self.assertRaises(ValueError):
            validate_merge_inputs(MODEL_PATH, adapter_dir, temporary / "out")

    def test_merge_input_validation_rejects_missing_adapter(self):
        temporary = self.make_temp_dir()
        with self.assertRaises(FileNotFoundError):
            validate_merge_inputs(MODEL_PATH, temporary / "nope", temporary / "out")

    def test_verify_merged_directory_reports_missing_files_then_passes(self):
        from src.deploy.config import MERGED_REQUIRED_FILES

        temporary = self.make_temp_dir()
        artifact = temporary / "merged"
        artifact.mkdir()
        (artifact / "config.json").write_text(
            json.dumps({"architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3"}),
            encoding="utf-8",
        )
        (artifact / "model.safetensors").write_bytes(b"stub")
        first = verify_merged_directory(artifact, with_hash=False)
        self.assertFalse(first["ok"])
        self.assertIn("required_files_present", {check["check"] for check in first["checks"]})
        for name in MERGED_REQUIRED_FILES:
            target = artifact / name
            if target.exists():
                continue
            if name == "processor_config.json":
                target.write_text(json.dumps({"image_seq_len": 64}), encoding="utf-8")
            else:
                target.write_text("{}", encoding="utf-8")
        second = verify_merged_directory(artifact, with_hash=False)
        self.assertTrue(second["ok"], second["checks"])

    def test_merge_dry_run_subprocess_is_cpu_light(self):
        completed = subprocess.run(
            [PYTHON, "-m", "src.deploy.merge_lora", "--dry-run"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=SUBPROCESS_ENV,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "dry_run_passed")
        self.assertEqual(payload["dtype"], "float32")
        self.assertTrue(payload["provisional"])


class QuantizePlanTests(unittest.TestCase):
    def test_gguf_plan_blocked_without_tools(self):
        llama = {"missing": ["convert_hf_to_gguf.py", "llama-quantize", "llama-mtmd-cli"], "binaries": {}, "convert_script": None, "home": None}
        plan = build_gguf_plan(PROJECT_ROOT / "adapters" / "merged" / "x", PROJECT_ROOT / "adapters" / "quantized" / "x", llama=llama)
        self.assertEqual(plan["status"], "blocked_tool_missing")
        self.assertEqual(plan["commands"], [])
        self.assertTrue(plan["missing"])

    def test_gguf_plan_commands_when_tools_present(self):
        llama = {
            "missing": [],
            "home": "C:/fake/llama.cpp",
            "convert_script": "C:/fake/llama.cpp/convert_hf_to_gguf.py",
            "binaries": {
                "llama-quantize": {"path": "C:/fake/llama.cpp/build/bin/llama-quantize.exe"},
                "llama-mtmd-cli": {"path": "C:/fake/llama.cpp/build/bin/llama-mtmd-cli.exe"},
                "llama-cli": {"path": None},
            },
        }
        fake_packages = {name: {"available": True} for name in ("gguf", "sentencepiece")}
        with mock.patch("src.deploy.quantize.packages", return_value=fake_packages):
            plan = build_gguf_plan(PROJECT_ROOT / "adapters" / "merged" / "x", PROJECT_ROOT / "adapters" / "quantized" / "x", llama=llama)
        self.assertEqual(plan["status"], "ready")
        steps = [command["step"] for command in plan["commands"]]
        self.assertEqual(steps, ["convert_f16_mmproj", "quantize_q4_k_m", "smoke_mmproj_generation"])
        convert = plan["commands"][0]["argv"]
        self.assertIn("--mmproj", convert)
        self.assertIn("--outfile", convert)
        quantize = plan["commands"][1]["argv"]
        self.assertEqual(quantize[-1], "Q4_K_M")

    def test_onnx_plan_reports_missing_tools(self):
        plan = build_onnx_plan(PROJECT_ROOT / "adapters" / "merged" / "x", PROJECT_ROOT / "adapters" / "quantized" / "onnx")
        self.assertEqual(plan["status"], "blocked_tool_missing")
        self.assertIn("optimum", plan["missing"])
        self.assertIn("onnx", plan["missing"])

    def test_export_free_routes_are_ready(self):
        torch_plan = build_torch_int8_plan(PROJECT_ROOT / "adapters" / "merged" / "x")
        fp32_plan = build_fp32_plan(PROJECT_ROOT / "adapters" / "merged" / "x")
        self.assertEqual(torch_plan["status"], "ready")
        self.assertFalse(torch_plan["requires_download"])
        self.assertIn("runtime_only", json.dumps(torch_plan))
        self.assertIn("not GGUF/Q4_K_M", " ".join(torch_plan["notes"]))
        self.assertEqual(fp32_plan["status"], "ready")
        self.assertIsNone(fp32_plan["fallback"])

    def test_handoff_contract_matches_the_evaluator(self):
        handoff = build_handoff(PROJECT_ROOT / "adapters" / "merged" / "sft_v1_merged")
        contract = handoff["inference_contract"]
        self.assertEqual(contract["prompt_id"], "p1")
        self.assertEqual(contract["image_seq_len"], 36)
        self.assertEqual(contract["max_new_tokens"], 64)
        self.assertFalse(contract["do_sample"])
        commands = " ".join(handoff["hf_checkpoint"]["evaluator_command_examples"])
        self.assertIn("--model-path", commands)
        self.assertIn("--batch-size 1", commands)
        self.assertEqual(handoff["quantized_checkpoint"]["official_metrics_status"], "not_run")


class BenchmarkPlanTests(TempDirMixin):
    def test_selection_is_deterministic_and_unique(self):
        records = read_jsonl(PROJECT_ROOT / "data" / "synthetic_eval.jsonl")
        first = select_benchmark_records(records, BENCH_SAMPLES, seed=42)
        second = select_benchmark_records(records, BENCH_SAMPLES, seed=42)
        other = select_benchmark_records(records, BENCH_SAMPLES, seed=7)
        self.assertEqual([record["id"] for record in first], [record["id"] for record in second])
        self.assertNotEqual(selected_ids_digest(first), selected_ids_digest(other))
        self.assertEqual(len({record["id"] for record in first}), BENCH_SAMPLES)

    def test_selection_rejects_oversampling(self):
        with self.assertRaises(ValueError):
            select_benchmark_records([{"id": "a"}], 2)

    def test_latency_stats_use_nearest_rank(self):
        values = [float(value) for value in range(1, 11)]
        stats = latency_stats(values)
        self.assertEqual(stats["n"], 10.0)
        self.assertEqual(stats["median_seconds"], 5.5)
        self.assertEqual(stats["mean_seconds"], 5.5)
        self.assertEqual(stats["p95_seconds"], 10.0)
        self.assertEqual(stats["p99_seconds"], 10.0)
        self.assertEqual(stats["min_seconds"], 1.0)
        self.assertEqual(stats["max_seconds"], 10.0)
        uniform = latency_stats([2.0] * 50)
        self.assertEqual(uniform["p95_seconds"], 2.0)
        self.assertEqual(uniform["stdev_seconds"], 0.0)

    def test_prior_cpu_evidence_is_recorded(self):
        prior = prior_cpu_latency_evidence()
        self.assertTrue(prior["available"])
        self.assertGreater(prior["estimated_per_sample_seconds"], 0.5)
        self.assertLess(prior["estimated_per_sample_seconds"], 3.0)
        self.assertIn("including model load", prior["caveat"])

    def test_bench_plan_only_subprocess(self):
        completed = subprocess.run(
            [PYTHON, "-m", "src.deploy.bench_latency", "--engine", "hf-cpu-adapter", "--plan-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            env=SUBPROCESS_ENV,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "plan_only")
        self.assertEqual(payload["selection"]["n"], BENCH_SAMPLES)
        self.assertEqual(payload["selection"]["warmup"], 5)
        self.assertEqual(len(payload["selection"]["selected_ids_digest"]), 64)
        self.assertEqual(payload["inference"]["image_seq_len"], 36)
        self.assertEqual(payload["inference"]["prompt_id"], "p1")
        self.assertFalse(payload["gpu"]["required"])
        self.assertIsNone(payload["gpu"]["snapshot_before"])

    def test_bench_plan_only_rejects_cuda_without_lease(self):
        checkpoint = self.make_temp_dir() / "merged"
        checkpoint.mkdir()
        from src.deploy.config import MERGED_REQUIRED_FILES

        for name in MERGED_REQUIRED_FILES:
            if name == "config.json":
                (checkpoint / name).write_text(
                    json.dumps({"architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3"}),
                    encoding="utf-8",
                )
            else:
                (checkpoint / name).write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"stub")
        from src.deploy.bench_latency import build_benchmark_plan, build_parser

        args = build_parser().parse_args(
            ["--engine", "hf-cuda", "--plan-only", "--artifact", str(checkpoint)]
        )
        with self.assertRaises(RuntimeError):
            build_benchmark_plan(args)


class ContractTests(unittest.TestCase):
    def test_image_seq_len_contract_is_36(self):
        self.assertEqual(expected_image_seq_len(), 36)
        self.assertEqual(expected_image_seq_len(512, 16, 4), 64)

    def test_results_path_guard_blocks_foreign_prefixes(self):
        self.assertEqual(checked_results_path("results/deploy_x.json", "deploy_").name, "deploy_x.json")
        with self.assertRaises(ValueError):
            checked_results_path("results/eval_x.json", "deploy_")
        with self.assertRaises(ValueError):
            checked_results_path("adapters/quantized/x.json", "deploy_")

    def test_prep_document_is_honest_if_present(self):
        path = PROJECT_ROOT / "results" / "deploy_prep.json"
        if not path.is_file():
            self.skipTest("results/deploy_prep.json not generated yet")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["phase"], "prep")
        self.assertEqual(document["status"], "prep_complete")
        self.assertEqual(document["fallback_chain"], list(FALLBACK_CHAIN))
        claims = document["claims"]
        for key in (
            "merged_model_created",
            "quantized_model_created",
            "cpu_latency_measured",
            "demo_launched",
            "official_deployment_metrics",
        ):
            self.assertTrue(claims[key].startswith("not_run"), key)
        self.assertFalse(claims["gpu_used_during_prep"])
        self.assertFalse(claims["model_loaded_during_prep"])
        step_ids = [step["id"] for step in document["steps"]]
        self.assertEqual(step_ids, [f"S{index}" for index in range(1, 11)])
        self.assertGreater(document["estimated_total_minutes"]["primary_gguf_path"], 0)
        self.assertTrue(document["gaps"])
        self.assertTrue(document["risks"])
        self.assertIn("tool_availability", document)
        self.assertIn("assets", document)

    def test_packages_helper_is_import_free(self):
        before = set(sys.modules)
        payload = packages()
        self.assertIn("torch", payload)
        added = set(sys.modules) - before
        self.assertNotIn("torch", added)
        self.assertNotIn("gradio", added)


if __name__ == "__main__":
    unittest.main(verbosity=2)
