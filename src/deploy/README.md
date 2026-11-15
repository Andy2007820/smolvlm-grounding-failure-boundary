# src/deploy · T14 部署骨架与执行手册

> 状态（2026-11-10 01:0x，GPU 评测期间）：**仅完成 CPU 只读准备**。合并、量化、延迟、demo 均未执行，所有 deployment 结论以 `results/deploy_prep.json` 的 `claims` 为准（全部 `not_run`）。
> 负责人：deploy-runner。写权限：`src/deploy/`、`demo/`、`results/deploy_*`、`adapters/quantized/`。执行阶段另需 Lead 确认 `adapters/merged`（T14 写范围）。

## 1. 模块地图

| 文件 | 作用 | 何时执行 |
|---|---|---|
| `config.py` | 常量、路径校验、sha256、50 图确定性抽样、延迟统计、时间估计 | 随时（纯 CPU） |
| `env.py` | 只读环境/工具/资产探测（**不 import torch**、不碰 GPU、不下载） | 随时 |
| `plan.py` | 由探测结果生成步骤、风险、fallback、claims | 随时 |
| `check_env.py` | 生成 `results/deploy_prep.json` | 随时 |
| `merge_lora.py` | LoRA 合并 fp32（`--dry-run`/`--verify-only`/`--verify-equivalence`） | 评测结束、Lead 放行后 |
| `quantize.py` | GGUF Q4_K_M→ONNX→torch INT8→fp32 四条路线 + handoff | 合并完成后 |
| `bench_latency.py` | 预热 5 + 50 图延迟/显存/诊断精度（`--plan-only` 可先跑） | 量化完成后 |
| `infer.py` | 共享 `HFPredictor`（merged 或 base+adapter），prompt/parser 复用 `src.eval` | 被 bench/demo 调用 |
| `test_deploy_cpu.py` | CPU 单测（不加载权重） | 随时 |
| `run_cpu_tests.py` | 跑 deploy+demo 单测并写 `results/deploy_cpu_tests.json` | 随时 |

## 2. 已实测的前置结论（不是部署指标）

- 环境：torch 2.11.0+cu128、transformers 5.19.0、peft 0.21.2、onnxruntime 1.31.0；**gradio / optimum / onnx / gguf / sentencepiece 均未安装**；无 llama.cpp、无 cmake、无 MSVC 编译器。
- 上游 ONNX：`models/SmolVLM-256M-Instruct/onnx/` 有 24 个文件（transformers.js 导出），**是 base 模型、不含本项目 LoRA**，只能做 ONNX 管线冒烟，不能当 `+量化` 行。
- CPU 先验（来自既有评测诊断 `results/zeroshot_batch_cpu_20_b1.json`）：fp32+adapter 20 图含加载 38.97 s ≈ **1.95 s/图上界**；GPU bf16+adapter 实测 ≈ 1.33 s/图（`elapsed_seconds_diagnostic` 中位数）。→ **CPU fp32 很可能不比 GPU 快**，这是要如实报告的负结果风险（R5）。

## 3. 执行顺序（Lead 放行后）

```powershell
# 0) 只读预检（随时可跑，不加载模型）
python -m src.deploy.check_env --print-summary
python -m src.deploy.merge_lora --dry-run
python -m src.deploy.bench_latency --engine hf-cpu-adapter --plan-only

# 1) 合并 LoRA -> fp32 CPU checkpoint（约 6 min，峰值内存 ~3 GB，磁盘 ~1.0 GB）
python -m src.deploy.merge_lora --adapter adapters/sft_v1/adapter ^
  --output adapters/merged/sft_v1_merged --dtype float32 ^
  --report results/deploy_merge_report.json

# 2) 合并等价性（base+adapter vs merged，5 条固定样本，必须逐字一致；约 8 min）
python -m src.deploy.merge_lora --verify-equivalence ^
  --artifact adapters/merged/sft_v1_merged ^
  --report results/deploy_merge_equivalence.json

# 3) 首选 GGUF Q4_K_M（需先获得下载/安装许可；见第 5 节）
python -m src.deploy.quantize --route gguf --plan-only ^
  --merged adapters/merged/sft_v1_merged
python -m src.deploy.quantize --route gguf ^
  --merged adapters/merged/sft_v1_merged ^
  --output adapters/quantized/sft_v1_q4_k_m ^
  --report results/deploy_quantize_gguf.json

# 4) CPU fp32 延迟（50 图/预热 5；约 2–4 min）
python -m src.deploy.bench_latency --engine hf-cpu --artifact adapters/merged/sft_v1_merged ^
  --n 50 --warmup 5 --seed 42 ^
  --output results/deploy_latency_hf_cpu_fp32.json ^
  --details-output results/deploy_latency_hf_cpu_fp32_details.jsonl

# 5) 量化引擎延迟（同样 50 个 id，保证可比）
python -m src.deploy.bench_latency --engine gguf-q4 --artifact adapters/quantized/sft_v1_q4_k_m ^
  --n 50 --warmup 5 --seed 42 --output results/deploy_latency_gguf_q4.json

# 6) GPU bf16 基线（需要 Lead 的 GPU 租约；脚本会拒绝无 lease 的 CUDA）
python -m src.deploy.bench_latency --engine hf-cuda --artifact adapters/merged/sft_v1_merged ^
  --device cuda --gpu-lease-id <LEASE-ID> --n 50 --warmup 5 ^
  --output results/deploy_latency_hf_cuda_bf16.json

# 7) Gradio demo（需先获准安装 gradio）
python -m demo.app --engine hf-cpu --artifact adapters/merged/sft_v1_merged --port 7860

# 8) handoff 给 eval-closer 做独立复测
python -m src.deploy.quantize --emit-handoff --route gguf ^
  --merged adapters/merged/sft_v1_merged ^
  --quantized-route gguf --quantized-artifact adapters/quantized/sft_v1_q4_k_m
```

任一步失败：**记录真实报错 → 按 `gguf → onnx → torch-int8 → hf-fp32` 降级**，不得把未跑通的路线写成已量化。

## 4. 单测（准备阶段已通过 50/50）

```powershell
python -m src.deploy.run_cpu_tests --force
# -> results/deploy_cpu_tests.json
```

约束：`CUDA_VISIBLE_DEVICES=""`、不加载权重、不联网、不训练。覆盖计划/工具探测/合并校验/量化命令/ONNX 归属/抽样与统计/画框与 demo handler/`deploy_prep.json` 诚实性。

## 5. GGUF 路线的前置缺口（必须 Lead/用户批准）

1. llama.cpp 工具链缺失：`convert_hf_to_gguf.py`、`llama-quantize`、`llama-mtmd-cli`。
   - 方案 A：下载 Windows 预编译 CPU release，解压到 `final-project/third_party/llama.cpp`（或设置 `LLAMA_CPP_DIR`）；release 里通常只有二进制，`convert_hf_to_gguf.py` 需从同版本的 llama.cpp 仓库取对应脚本。
   - 方案 B：源码构建（需要 cmake + MSVC，本机都没有 → 当前不可行）。
2. Python 依赖：`pip install gguf sentencepiece`（convert 脚本需要）。
3. `--mmproj` 的文件名与 `llama-mtmd-cli` 参数随版本变化；执行器先做 10 分钟冒烟（`--help` + 转换小样本），不接受"没出框就宣布 4-bit 完成"。
4. ONNX 路线：需要 `pip install optimum[onnxruntime] onnx`，且 optimum 对 Idefics3 的导出支持未验证；失败则记录 blocked，用 torch INT8。

## 6. 已知坑

- **`tempfile.mkdtemp` 在本机沙箱下不可写**（0o700 目录子项被拒），所以所有临时文件用 `config.make_scratch_dir()` 建在项目 `.cache/deploy_scratch/`。
- 384/no-split 的视觉 token 数是 **36**，不是 processor_config 默认 64；所有推理路径都必须按 `expected_image_seq_len()` 设置，否则 CUDA assert（历史事故）。
- 合并后必须验证 `processor_config.json` 仍是 64、merged 模型 config 无 peft/lora 残留；`merge_and_unload` 后 LoRA 模块数必须为 0。
- 生成配置 EOS=49279、pad=2；解码固定 greedy、max_new_tokens=64、seed 42，与评测一致。
- 上游 ONNX 资产不含 LoRA（见第 2 节）。
- 部署候选是**临时的** `adapters/sft_v1/adapter`（seed42）：T12/KD 未跑、seed44 评测未完成，Lead 冻结最终 adapter 后需重新 merge/量化。

## 7. 交付缺口（当前）

- `adapters/merged/` 未创建；`adapters/quantized/` 为空。
- 无任何真实量化/延迟/demo 数字；`results/deploy_prep.json` 的 `claims` 全部 `not_run`。
- `adapters/merged` 不在本准备任务的写范围内，执行前需 Lead 明确。
- 项目根目录残留一个空的 `probe_if4ilkub/`（本机沙箱下 `mkdtemp` 创建的目录 ACL 拒绝删除，无法清理；不是模型/结果数据）。
