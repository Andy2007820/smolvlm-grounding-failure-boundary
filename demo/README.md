# demo/ · T15 Gradio 指代定位演示

上传图片 + 输入英文指代表达 → 模型输出归一化框 → 图上画框 + 原始输出。

> 状态（2026-11-10 01:0x，评测期间）：**demo 未启动过**，`results/deploy_prep.json` 中 `demo_launched=not_run`。当前 gradio 未安装，真实模型也需要等 Lead 放行。**不得**把未运行的界面算作 G7 通过。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | Gradio UI + 框架无关的 `GroundingDemo` 核心 + `--smoke` 自动验证 |
| `draw.py` | 纯 PIL 画框/结果文案（可单测，不 import gradio） |
| `test_demo_cpu.py` | CPU 单测：画框像素、handler、假预测器、参数解析 |
| `artifacts/` | `--smoke` 保存的标注输出图（执行后生成） |

推理契约与评测完全一致：prompt `p1`（`src.eval.prompts`）、image_size 384、no-split、`image_seq_len=36`、greedy、max_new_tokens=64、解析器 `src.eval.grounding.parse_box_with_error`。

## 运行

前提：Lead 已宣布评测结束并放行；安装 gradio（需批准）：

```powershell
python -m pip install gradio
# 真实模型（CPU；若用 GPU 需 Lead 的 GPU 租约）
python -m demo.app --engine hf-cpu --artifact adapters/merged/sft_v1_merged --port 7860
# 或未合并的 base+adapter（与评测路径一致，慢一些）
python -m demo.app --engine hf-cpu-adapter --port 7860
# 量化后
python -m demo.app --engine gguf-q4 --artifact adapters/quantized/sft_v1_q4_k_m --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，上传 `data/images/synthetic/eval/synthetic_eval_000001.png`，输入 `the purple triangle`，应看到图中三角框与坐标文本。

## 自动验证（产出证据，不替代人眼确认）

```powershell
# 只验证界面链路（假预测器，1 秒级；不是模型输出）
python -m demo.app --engine fake --smoke
# 真实模型 + 真实图片 + gradio_client 往返（需模型加载许可）
python -m demo.app --engine hf-cpu --artifact adapters/merged/sft_v1_merged --smoke
```

产物：`results/deploy_demo_smoke.json`（HTTP 200 + API 往返结果 + 引擎 + 是否 fake + 标注图路径）、`demo/artifacts/deploy_demo_smoke_output.png`。

G7 的 demo 判据需要：真实模型 smoke 通过 + Lead 在浏览器里实际看到出框（agent 不能自证"人眼可见"）；若只能跑到假预测器 smoke，demo 部分必须记 `not_run/partial`。

## 单测（准备阶段已通过）

```powershell
python -m unittest demo.test_demo_cpu -v
# 或随部署套件一起：python -m src.deploy.run_cpu_tests --force
```

## 说明

- `--fake` 是占位预测器，UI 顶部会明确标注；任何截图/报告不得用它冒充模型结果。
- 上传图在 demo 内是可选的；空图/空文本会返回中文提示，不会崩。
- 画框使用归一化框 → 像素四舍五入，框宽 = 短边 1%（最小 2 px）。
- 若 gradio 安装不获批：降级为 `python -m src.deploy.infer` 风格的 CLI 预测 + 标注 PNG，并如实写"Gradio 未跑"。
