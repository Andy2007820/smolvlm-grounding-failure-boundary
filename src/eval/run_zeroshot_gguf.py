#!/usr/bin/env python3
"""Official-protocol evaluation of the quantized GGUF via llama.cpp **server** (one model load, many images).

Same dataset rows, frozen p1 prompt, greedy decoding and the same frozen grounding
scorer as the transformers evaluator; the model itself runs in ``llama-server.exe``
(OpenAI-compatible /v1/chat/completions with data-URI images).  Loading the model
once and streaming images over HTTP makes a 1,000-image run feasible on CPU.

Verified by ``src/eval/verify_gguf_eval.py`` (details count, metric recomputation
with the frozen scorer, details sha256, summary/dataset consistency, artifact sha).

Usage::
    python -m src.eval.run_zeroshot_gguf --run-id gguf_q4km_synthetic_v1 \
        --dataset synthetic --split eval --data data/synthetic_eval.jsonl \
        --output results/zeroshot_gguf_q4km_synthetic_v1.json \
        --details-output results/eval_gguf_q4km_synthetic_v1_details.jsonl \
        --expected-n 1000 --workers 6
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval import grounding  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402
from eval.prompts import PROMPT_ID, prompt_sha256_for, prompt_template_for, representation_for  # noqa: E402

P8_DIR = PROJECT_ROOT / "artifacts" / "p8-4bit-gguf"
DEFAULT_SERVER = P8_DIR / "llama-bin" / "llama-server.exe"
DEFAULT_MODEL = PROJECT_ROOT / "adapters" / "quantized" / "gguf" / "sft_v1_merged-Q4_K_M.gguf"
DEFAULT_MMPROJ = PROJECT_ROOT / "adapters" / "quantized" / "gguf" / "mmproj-sft_v1_merged-Q8_0.gguf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GGUF (llama.cpp server) official-protocol grounding evaluation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True, choices=("synthetic", "refcoco"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-id", default=PROMPT_ID)
    parser.add_argument("--server-exe", default=str(DEFAULT_SERVER))
    parser.add_argument("--model-gguf", default=str(DEFAULT_MODEL))
    parser.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=6, help="llama-server parallel slots")
    parser.add_argument("--ctx", type=int, default=16384, help="total llama-server context; per-slot = ctx/parallel")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--server-url", default=None, help="reuse an already running llama-server")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=("cli", "server"), default="cli", help="cli = one llama-mtmd-cli process per image (measured faster); server = llama-server HTTP")
    parser.add_argument("--cli-exe", default=str(P8_DIR / "llama-bin" / "llama-mtmd-cli.exe"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def request_one_cli(args: argparse.Namespace, record: dict[str, Any], prompt: str) -> dict[str, Any]:
    started = time.time()
    result = None
    for attempt in range(3):
        result = subprocess.run(
            [
                args.cli_exe,
                "-m",
                args.model_gguf,
                "--mmproj",
                args.mmproj,
                "--image",
                str(PROJECT_ROOT / record["image"]),
                "-p",
                prompt,
                "-n",
                str(args.max_new_tokens),
                "--temp",
                "0",
                "--no-warmup",
                "--threads",
                str(args.threads),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode == 0 and (result.stdout or "").strip():
            break
        time.sleep(0.5 + attempt)
    output_text = (result.stdout or "").strip() if result is not None else ""
    score = grounding.score_prediction(output_text, record["answer"])
    return {
        "id": record["id"],
        "index": record["index"],
        "image": record["image"],
        "question": record["question"],
        "ground_truth": record["answer"],
        "raw_bbox": record.get("raw_bbox"),
        "width": record.get("width"),
        "height": record.get("height"),
        "output_text": output_text[-400:],
        "raw_output": output_text,
        "predicted_box": list(score.predicted_box) if score.predicted_box else None,
        "parse_error": score.parse_error,
        "iou": score.iou,
        "correct@0.5": score.correct_at_0_5,
        "elapsed_seconds_diagnostic": round(time.time() - started, 3),
        "process_exit": result.returncode if result is not None else None,
        "attempts": attempt + 1,
        "backend": "llama.cpp/llama-mtmd-cli",
    }


def wait_for_server(url: str, timeout: float, process: subprocess.Popen | None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise SystemExit(f"llama-server exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    raise SystemExit("llama-server did not become healthy in time")


def start_server(args: argparse.Namespace, log_path: Path) -> tuple[subprocess.Popen, str]:
    url = f"http://127.0.0.1:{args.port}"
    command = [
        args.server_exe,
        "-m",
        args.model_gguf,
        "--mmproj",
        args.mmproj,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--threads",
        str(args.threads),
        "-c",
        str(args.ctx),
        "-np",
        str(args.parallel),
        "--no-warmup",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    wait_for_server(url, 240, process)
    return process, url


def request_one(url: str, record: dict[str, Any], prompt: str, max_new_tokens: int) -> dict[str, Any]:
    image_bytes = (PROJECT_ROOT / record["image"]).read_bytes()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    started = time.time()
    output_text = ""
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            output_text = (body["choices"][0]["message"].get("content") or "").strip()
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    elapsed = time.time() - started
    score = grounding.score_prediction(output_text, record["answer"])
    return {
        "id": record["id"],
        "index": record["index"],
        "image": record["image"],
        "question": record["question"],
        "ground_truth": record["answer"],
        "raw_bbox": record.get("raw_bbox"),
        "width": record.get("width"),
        "height": record.get("height"),
        "output_text": output_text,
        "predicted_box": list(score.predicted_box) if score.predicted_box else None,
        "parse_error": score.parse_error or error,
        "iou": score.iou,
        "correct@0.5": score.correct_at_0_5,
        "elapsed_seconds_diagnostic": round(elapsed, 3),
        "attempt_error": error,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model_gguf).resolve()
    mmproj_path = Path(args.mmproj).resolve()
    for path in (model_path, mmproj_path, Path(args.server_exe)):
        if not path.is_file():
            raise SystemExit(f"missing file: {path}")

    data_path = PROJECT_ROOT / args.data
    records = load_dataset(
        data_path,
        project_root=PROJECT_ROOT,
        expected_source=args.dataset,
        split=args.split,
        expected_n=None,
        limit=args.limit,
        check_images=True,
    )
    for index, record in enumerate(records):
        record["index"] = record.get("index", index)
    if args.expected_n and len(records) < args.expected_n and not args.limit:
        raise SystemExit(f"dataset has {len(records)} rows, expected {args.expected_n}")

    model_sha = sha256_file(model_path)
    mmproj_sha = sha256_file(mmproj_path)
    dataset_sha = sha256_file(data_path)
    if args.dry_run:
        print(json.dumps({"status": "preflight_passed", "dataset_rows": len(records), "model_sha256": model_sha, "mmproj_sha256": mmproj_sha, "dataset_sha256": dataset_sha, "workers": args.workers}, ensure_ascii=False, indent=2))
        return 0

    output_path = PROJECT_ROOT / args.output
    details_path = PROJECT_ROOT / args.details_output
    for path in (output_path, details_path):
        if path.exists() and not args.resume:
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    done_ids: set[str] = set()
    if args.resume and details_path.is_file():
        with details_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    pending = [record for record in records if record["id"] not in done_ids]

    process = None
    url = None
    if args.backend == "server":
        if args.server_url:
            url = args.server_url.rstrip("/")
            wait_for_server(url, 60, None)
        else:
            process, url = start_server(args, P8_DIR / f"llama-server-{args.run_id}.log")

    template = prompt_template_for(args.prompt_id)
    mode = "a" if args.resume and details_path.is_file() else "w"
    started = time.time()
    completed = len(done_ids)
    backend_label = "cli" if args.backend == "cli" else url
    print(json.dumps({"event": "start", "pending": len(pending), "done": len(done_ids), "workers": args.workers, "backend": backend_label}), flush=True)
    try:
        with details_path.open(mode, encoding="utf-8") as details_handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                if args.backend == "cli":
                    futures = {
                        pool.submit(request_one_cli, args, record, template.format(question=record["question"])): record
                        for record in pending
                    }
                else:
                    futures = {
                        pool.submit(request_one, url, record, template.format(question=record["question"]), args.max_new_tokens): record
                        for record in pending
                    }
                for future in as_completed(futures):
                    row = future.result()
                    details_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    details_handle.flush()
                    completed += 1
                    if completed % 100 == 0 or completed == len(records):
                        elapsed = time.time() - started
                        done_now = max(completed - len(done_ids), 1)
                        remaining = len(records) - completed
                        print(
                            json.dumps(
                                {"event": "progress", "completed": completed, "of": len(records), "elapsed_s": round(elapsed, 1), "eta_s": round(elapsed / done_now * remaining, 1)},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    rows: list[dict[str, Any]] = []
    with details_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    scores = [
        grounding.Score(
            tuple(row["predicted_box"]) if row["predicted_box"] else None,
            float(row["iou"]),
            bool(row["correct@0.5"]),
            row["parse_error"],
        )
        for row in rows
    ]
    metrics = grounding.metrics_from_scores(scores)
    summary = {
        "run_id": args.run_id,
        "model": {
            "path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": model_sha,
            "format": "GGUF Q4_K_M (text) + Q8_0 (mmproj)",
            "backend": ("llama.cpp b12282 / llama-mtmd-cli.exe (one process per image)" if args.backend == "cli" else "llama.cpp b12282 / llama-server.exe (OpenAI-compatible /v1/chat/completions)"),
            "mmproj": {"path": str(mmproj_path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": mmproj_sha},
        },
        "adapter": None,
        "dataset": args.dataset,
        "split": args.split,
        "n": len(rows),
        "batch_size": 1,
        "prompt_id": args.prompt_id,
        "representation": representation_for(args.prompt_id),
        "metrics": metrics,
        "config": {
            "image_size": 384,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": 1,
            "prompt_id": args.prompt_id,
            "prompt_sha256": prompt_sha256_for(args.prompt_id),
            "representation": representation_for(args.prompt_id),
            "seed": 42,
            "do_sample": False,
            "num_beams": 1,
            "decoding": "deterministic_greedy",
            "device": "cpu",
            "backend": "llama.cpp",
            "workers": args.workers,
            "dataset_path": args.data,
            "dataset_sha256": dataset_sha,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "completed",
        "details_path": str(details_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "details_sha256": hashlib.sha256(details_path.read_bytes()).hexdigest(),
        "details_rows": len(rows),
        "runtime": {"elapsed_seconds": round(time.time() - started, 1), "workers": args.workers},
        "note": (
            "GGUF (llama.cpp server) run of the official evaluation protocol: same dataset rows, frozen p1 prompt, greedy decoding "
            "and the frozen grounding scorer. Backend is llama.cpp, so the equivalent independent verification is "
            "src/eval/verify_gguf_eval.py (not the frozen transformers validator)."
        ),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "run_id": args.run_id, "n": len(rows), "metrics": metrics, "elapsed_s": summary["runtime"]["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
