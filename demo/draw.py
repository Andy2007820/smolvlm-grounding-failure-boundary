"""Pure-PIL drawing helpers for the Gradio demo (no gradio import here)."""

from __future__ import annotations

from typing import Any, Sequence

from src.eval.grounding import validate_box

BOX_COLOR = (230, 57, 70)
BOX_WIDTH_RATIO = 0.01


def box_to_pixels(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a normalized xyxy box to integer pixel corners inside the image."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = validate_box(box, normalized=True)
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(0, min(width - 1, int(round(x2 * width))))
    bottom = max(0, min(height - 1, int(round(y2 * height))))
    if right <= left or bottom <= top:
        raise ValueError(f"box collapses to an empty pixel rectangle: {(left, top, right, bottom)}")
    return left, top, right, bottom


def draw_box(
    image: Any,
    box: Sequence[float] | None,
    *,
    color: tuple[int, int, int] = BOX_COLOR,
    width: int | None = None,
) -> Any:
    """Return a copy of a PIL image with the predicted box drawn on it."""

    from PIL import ImageDraw

    if image is None:
        raise ValueError("image must be a PIL image")
    if not hasattr(image, "size"):
        raise ValueError("image must be a PIL image")
    annotated = image.copy()
    if box is None:
        return annotated
    pixel_box = box_to_pixels(box, annotated.size[0], annotated.size[1])
    if width is None:
        width = max(2, int(round(min(annotated.size) * BOX_WIDTH_RATIO)))
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(pixel_box, outline=color, width=width)
    return annotated


def format_result(prediction: dict[str, Any], *, pixel_box: tuple[int, int, int, int] | None = None) -> str:
    """Human-readable result block shown next to the annotated image."""

    box = prediction.get("predicted_box")
    lines: list[str] = []
    if box is None:
        error = prediction.get("parse_error") or "unparseable output"
        lines.append(f"**未能解析出边界框**：`{error}`")
    else:
        rendered = ", ".join(f"{value:.4f}" for value in box)
        lines.append(f"**归一化框 [x1, y1, x2, y2]**：`[{rendered}]`")
        if pixel_box is not None:
            lines.append(f"**像素框**：`{pixel_box}`")
    latency = prediction.get("latency_seconds")
    if isinstance(latency, (int, float)):
        lines.append(f"**延迟（含图像预处理）**：`{latency:.3f} s`")
    if isinstance(latency, (int, float)) and latency < 0.01:
        lines.append("")
        lines.append("> 本结果来自 `--fake` 占位预测器，仅用于验证界面链路，不是模型输出。")
    raw = prediction.get("output_text")
    if raw is not None:
        lines.append("")
        lines.append("**模型原始输出**：")
        lines.append("```text")
        lines.append(str(raw).strip())
        lines.append("```")
    return "\n".join(lines)
