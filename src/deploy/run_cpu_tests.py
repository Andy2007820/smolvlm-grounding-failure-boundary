"""Run the deployment/demo CPU test suites and write ``results/deploy_cpu_tests.json``.

The runner forces ``CUDA_VISIBLE_DEVICES=""`` before any test module is imported,
so no test can create a CUDA context while the GPU is leased to the evaluator.
The test modules themselves never load model weights and never touch the network.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import unittest
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from .config import now_iso, project_relative, write_json_atomic  # noqa: E402

DEFAULT_MODULES = ("src.deploy.test_deploy_cpu", "demo.test_demo_cpu")


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, Any]] = []
        self._started: dict[str, float] = {}

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._started[test.id()] = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        duration = time.perf_counter() - self._started.pop(test.id(), time.perf_counter())
        status = "passed"
        if any(test is case for case, _ in self.failures):
            status = "failed"
        elif any(test is case for case, _ in self.errors):
            status = "errored"
        elif any(test is case for case, _ in self.skipped):
            status = "skipped"
        self.records.append({"test": test.id(), "status": status, "duration_seconds": round(duration, 4)})
        super().stopTest(test)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/deploy_cpu_tests.json")
    parser.add_argument("--modules", nargs="*", default=list(DEFAULT_MODULES))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[2] / output
    if not output.name.startswith("deploy_"):
        raise ValueError(f"test result must start with deploy_: {output.name}")
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    started = time.perf_counter()
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromNames(args.modules)
    except Exception as exc:  # pragma: no cover - import failure of a test module
        payload = {
            "status": "import_failed",
            "modules": args.modules,
            "error": f"{type(exc).__name__}: {exc}",
            "captured_at": now_iso(),
        }
        write_json_atomic(output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, resultclass=RecordingResult).run(suite)

    failures = [record for record in getattr(result, "records", []) if record["status"] in ("failed", "errored")]
    payload = {
        "schema_version": 1,
        "status": "passed" if result.wasSuccessful() else "failed",
        "captured_at": now_iso(),
        "modules": args.modules,
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed": len(result.failures),
        "errored": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_seconds": time.perf_counter() - started,
        "records": getattr(result, "records", []),
        "failure_details": failures,
        "constraints": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_imported": "torch" in sys.modules,
            "model_weights_loaded": False,
            "network_used": False,
            "gpu_used": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "note": "CPU-only deployment/demo tests; model load, merge, quantization and gradio launch are not covered here",
    }
    write_json_atomic(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "total_tests", "passed", "failed", "errored", "elapsed_seconds")}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
