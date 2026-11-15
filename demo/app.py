"""Gradio demo for T15: upload an image + referring expression -> annotated box.

The UI is intentionally thin: all model work goes through
:mod:`src.deploy.infer` (or the GGUF CLI driver), the prompt is ``p1`` from
:mod:`src.eval.prompts`, and the box parser is the frozen
:func:`src.eval.grounding.parse_box_with_error`.  ``gradio`` is imported lazily,
so this module can be unit-tested on a machine without gradio installed.

Launch (after the Lead releases the GPU/CPU lease and gradio is installed)::

    python -m demo.app --engine hf-cpu --artifact adapters/merged/sft_v1_merged --port 7860
    python -m demo.app --engine fake --smoke          # UI plumbing only, no model

``--fake`` exists so the UI can be verified without a model; it must never be
presented as a real prediction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from src.deploy.config import (
    DEFAULT_ADAPTER,
    DEFAULT_MERGED,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PROJECT_ROOT,
    make_scratch_dir,
    now_iso,
    project_relative,
    read_jsonl,
    remove_scratch_dir,
    resolve_project_path,
    write_json_atomic,
)
from demo.draw import draw_box, format_result

DEMO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEMO_ROOT / "artifacts"
DATASET = PROJECT_ROOT / "data" / "synthetic_eval.jsonl"


class FakePredictor:
    """Deterministic placeholder used only to verify the UI wiring."""

    def __init__(self, box: list[float] | None = None) -> None:
        self.box = box or [0.25, 0.25, 0.75, 0.75]

    def load(self) -> dict[str, Any]:
        return {"status": "loaded", "engine": "fake", "is_fake": True, "note": "placeholder predictor; not a model"}

    def predict(self, image: Any, question: str) -> dict[str, Any]:
        return {
            "raw_output": f"[{', '.join(f'{value:.2f}' for value in self.box)}]",
            "output_text": f"[{', '.join(f'{value:.2f}' for value in self.box)}]",
            "predicted_box": list(self.box),
            "parse_error": None,
            "input_tokens": 0,
            "generated_tokens": 0,
            "hit_max_new_tokens": False,
            "latency_seconds": 0.001,
        }


class GroundingDemo:
    """Framework-free core so it can be unit-tested without gradio."""

    def __init__(
        self,
        predictor: Any,
        *,
        engine_label: str,
        artifact_label: str,
        is_fake: bool = False,
        examples: list[list[str]] | None = None,
    ) -> None:
        self.predictor = predictor
        self.engine_label = engine_label
        self.artifact_label = artifact_label
        self.is_fake = is_fake
        self.examples = examples or []

    def locate(self, image: Any, question: str) -> tuple[Any, str]:
        from PIL import Image

        if image is None:
            return None, "请先上传一张图片。"
        if not isinstance(image, Image.Image):
            return None, "输入必须是 PIL 图像。"
        if not isinstance(question, str) or not question.strip():
            return image, "请输入要定位物体的指代表达，例如 “the red cup”。"
        started = time.perf_counter()
        try:
            prediction = self.predictor.predict(image, question)
        except Exception as exc:  # demo must not crash on a bad sample
            return image, f"**推理失败**：`{type(exc).__name__}: {exc}`"
        latency = time.perf_counter() - started
        box = prediction.get("predicted_box")
        pixel_box = None
        if box is not None:
            try:
                from demo.draw import box_to_pixels

                pixel_box = box_to_pixels(box, image.size[0], image.size[1])
            except ValueError:
                pixel_box = None
        annotated = draw_box(image, box)
        prediction.setdefault("latency_seconds", latency)
        text = format_result(prediction, pixel_box=pixel_box)
        if self.is_fake:
            text = "> ⚠️ 当前为 `--fake` 占位预测器，仅验证界面链路，不是模型输出。\n\n" + text
        return annotated, text

    def description(self) -> str:
        return (
            "### SmolVLM-256M 指代定位演示（T15）\n"
            f"- 引擎：`{self.engine_label}`\n"
            f"- 模型：`{self.artifact_label}`\n"
            f"- 推理契约：prompt `p1`、image_size `{IMAGE_SIZE}`、no-split、"
            f"image_seq_len `36`、max_new_tokens `{MAX_NEW_TOKENS}`、greedy\n"
            "- 解析器：`src.eval.grounding.parse_box_with_error`（与独立评测器一致）\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("hf-cpu", "hf-cpu-adapter", "hf-cuda", "gguf-q4", "fake"), default="hf-cpu"
    )
    parser.add_argument("--artifact", default=str(DEFAULT_MERGED), help="merged checkpoint or GGUF directory")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio link (off by default)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="launch, verify HTTP+API round-trip, write results/deploy_demo_smoke.json, exit")
    parser.add_argument("--smoke-output", default="results/deploy_demo_smoke.json", help="smoke evidence JSON under results/deploy_*")
    parser.add_argument("--smoke-image", default=None, help="image for the --smoke round-trip")
    parser.add_argument("--smoke-question", default=None)
    return parser


def load_examples(limit: int = 3) -> list[list[str]]:
    try:
        records = read_jsonl(DATASET, limit=limit)
    except OSError:
        return []
    examples: list[list[str]] = []
    for record in records:
        image = PROJECT_ROOT / record["image"]
        if image.is_file():
            examples.append([str(image), record["question"]])
    return examples


def _extract_gguf(artifact: Path) -> tuple[str, str]:
    files = sorted(artifact.rglob("*.gguf")) if artifact.is_dir() else [artifact]
    models = [path for path in files if "mmproj" not in path.name.lower()]
    mmproj = [path for path in files if "mmproj" in path.name.lower()]
    if not models or not mmproj:
        raise FileNotFoundError(f"need a quantized .gguf and a *mmproj*.gguf under {artifact}")
    return project_relative(models[0]), project_relative(mmproj[0])


class _GgufDemoPredictor:
    """Adapts the CLI driver to the PIL-image interface used by the demo."""

    def __init__(self, artifact: Path, *, max_new_tokens: int) -> None:
        from src.deploy.bench_latency import GgufPredictor

        model, mmproj = _extract_gguf(artifact)
        self.inner = GgufPredictor(
            {"model": model, "mmproj": mmproj}, max_new_tokens=max_new_tokens
        )
        self.report = self.inner.load()

    def load(self) -> dict[str, Any]:
        return self.report

    def predict(self, image: Any, question: str) -> dict[str, Any]:
        scratch = make_scratch_dir("demo_gguf")
        try:
            path = scratch / "upload.png"
            image.save(path)
            result = self.inner.predict_path(path, question)
        finally:
            remove_scratch_dir(scratch)
        return result


def build_predictor(args: argparse.Namespace) -> tuple[Any, str, str, bool]:
    if args.engine == "fake":
        return FakePredictor(), "fake", "fake placeholder", True
    if args.engine == "gguf-q4":
        artifact = resolve_project_path(args.artifact)
        return _GgufDemoPredictor(artifact, max_new_tokens=args.max_new_tokens), "gguf-q4", project_relative(artifact), False
    from src.deploy.infer import HFPredictor

    merged = None
    adapter = None
    if args.engine == "hf-cpu-adapter":
        adapter = resolve_project_path(args.adapter)
        artifact_label = f"{project_relative(resolve_project_path(args.model_path))} + {project_relative(adapter)}"
    else:
        merged = resolve_project_path(args.artifact)
        artifact_label = project_relative(merged)
    dtype = "float32" if args.device == "cpu" else "bfloat16"
    predictor = HFPredictor(
        model_path=resolve_project_path(args.model_path),
        merged=merged,
        adapter=adapter,
        device=args.device,
        dtype=dtype,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
    )
    predictor.load()
    return predictor, args.engine, artifact_label, False


def build_ui(demo: GroundingDemo) -> Any:
    import gradio as gr

    with gr.Blocks(title="SmolVLM-256M 指代定位演示") as ui:
        gr.Markdown(demo.description())
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")
                question_input = gr.Textbox(label="指代表达（英文）", placeholder="the red cup", lines=2)
                run_button = gr.Button("定位", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="pil", label="定位结果")
                text_output = gr.Markdown(label="坐标与原始输出")
        if demo.examples:
            gr.Examples(examples=demo.examples, inputs=[image_input, question_input])
        run_button.click(fn=demo.locate, inputs=[image_input, question_input], outputs=[image_output, text_output], api_name="locate")
    return ui


def _pick_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


def _wait_http(url: str, timeout_seconds: float = 60.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return {"status": response.status, "url": url, "content_length": len(response.read())}
        except Exception as exc:  # server not ready yet
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    return {"status": None, "url": url, "error": last_error}


def run_smoke(args: argparse.Namespace) -> int:
    import gradio as gr

    predictor, engine, artifact_label, is_fake = build_predictor(args)
    demo = GroundingDemo(predictor, engine_label=engine, artifact_label=artifact_label, is_fake=is_fake, examples=load_examples())
    ui = build_ui(demo)
    port = _pick_free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    ui.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True, show_error=True, inbrowser=False)
    http_result = _wait_http(url + "/", timeout_seconds=90)
    client_result: dict[str, Any] = {"status": "not_run"}
    annotated_saved: str | None = None
    try:
        from gradio_client import Client

        try:
            from gradio_client import handle_file as _handle_file
        except ImportError:  # pragma: no cover - older gradio_client
            _handle_file = None

        records = read_jsonl(DATASET, limit=1)
        image_path = resolve_project_path(args.smoke_image) if args.smoke_image else PROJECT_ROOT / records[0]["image"]
        question = args.smoke_question or records[0]["question"]
        client = Client(url, verbose=False)
        attempt_errors: list[str] = []
        prediction = None
        candidates: list[Any] = []
        if _handle_file is not None:
            candidates.append(_handle_file(str(image_path)))
        candidates.append({"path": str(image_path)})
        for candidate in candidates:
            try:
                prediction = client.predict(candidate, question, api_name="/locate")
                break
            except Exception as exc:  # try the next input encoding
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
        if prediction is None:
            raise RuntimeError("all gradio_client input encodings failed: " + " | ".join(attempt_errors))
        client_result = {"status": "completed", "returned_types": [type(item).__name__ for item in prediction]}
        if attempt_errors:
            client_result["earlier_attempt_errors"] = attempt_errors
        if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
            client_result["text"] = prediction[1]
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            annotated_name = "deploy_demo_smoke_fake_output.png" if is_fake else "deploy_demo_smoke_output.png"
            annotated_saved = str(ARTIFACTS_DIR / annotated_name)
            returned_image = prediction[0]
            if isinstance(returned_image, str):
                shutil.copyfile(returned_image, annotated_saved)
            elif hasattr(returned_image, "save"):
                returned_image.save(annotated_saved)
        else:
            client_result["raw"] = repr(prediction)[:2000]
    except Exception as exc:  # pragma: no cover - requires gradio installed
        client_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            ui.close()
        except Exception:
            pass

    payload = {
        "schema_version": 1,
        "run_id": "deploy_demo_smoke",
        "captured_at": now_iso(),
        "engine": engine,
        "artifact": artifact_label,
        "is_fake_predictor": is_fake,
        "gradio_version": getattr(gr, "__version__", None),
        "url": url,
        "http_probe": http_result,
        "api_round_trip": client_result,
        "annotated_output": project_relative(Path(annotated_saved)) if annotated_saved else None,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_note": (
            "fake predictor = UI plumbing only, never a model result; real-model smoke requires the "
            "post-evaluation lease and a browser confirmation by the Lead for the G7 demo claim"
        ),
    }
    output = Path(args.smoke_output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not output.parent.name == "results" or not output.name.startswith("deploy_"):
        raise ValueError(f"--smoke-output must be results/deploy_*: {output}")
    write_json_atomic(output, payload)
    passed = bool(http_result.get("status") == 200 and client_result.get("status") == "completed")
    payload["status"] = "passed" if passed else "failed"
    write_json_atomic(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "url", "is_fake_predictor", "api_round_trip")}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        return run_smoke(args)
    import gradio as gr

    predictor, engine, artifact_label, is_fake = build_predictor(args)
    demo = GroundingDemo(predictor, engine_label=engine, artifact_label=artifact_label, is_fake=is_fake, examples=load_examples())
    ui = build_ui(demo)
    ui.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
