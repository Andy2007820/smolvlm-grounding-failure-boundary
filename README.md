# SmolVLM-256M Grounding Experiments

本仓库只包含实验代码、配置、数据清单、评测结果与复现所需材料。

## 目录

| 路径 | 内容 |
|---|---|
| `src/` | 数据准备、训练、评测、验证与部署代码 |
| `configs/` | SFT、ReST、知识蒸馏、T7 坐标变体配置 |
| `scripts/` | `run_all` 阶段驱动、G5/G6 汇总、T10 噪声回放 |
| `data/` | RefCOCO / Synthetic / ScreenSpot 的 JSONL、清单、统计与 QA 文件（不含图片） |
| `results/` | summary、details、validation、verify、analysis 与 GPU 日志 |
| `demo/` | Gradio demo 代码 |
| `requirements.txt` | 实测环境依赖 |
| `TRAINING-FAILURE-LEDGER.md` | 训练/评测失败与负结果台账 |
| `EXPERIMENT-MANIFEST.json` | 本仓库文件的 SHA-256 与字节数清单 |

## 环境

实测环境：Windows 11、Python 3.14.8、torch 2.11.0+cu128、transformers 5.19.0、peft 0.21.2、NVIDIA RTX 5060 Laptop 8 GB。

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 固定的是实验实测版本。CUDA/CPU 评测均使用 batch1；GPU 训练需要 CUDA。

## 数据准备

仓库包含数据 JSONL 与 manifest，不包含图片。图片需按各数据准备脚本重新下载/生成：

```bash
python src/data/prepare_refcoco.py --help
python src/data/prepare_screenspot.py --help
python src/data/make_synthetic.py --help
```

数据统计、训练/评测隔离和 QA 证据在 `data/stats.json`、`data/synthetic_stats.json`、`data/screenspot_stats.json` 与 `data/qa/`。

## 快速检查

```bash
python scripts/run_all.py --list-stages
python scripts/run_all.py --stage check --dry-run
python scripts/run_all.py --stage all --dry-run
```

`--dry-run` 不加载模型、不写结果，只打印将执行的命令。训练阶段必须显式提供 `--allow-train` 与 GPU 租约；已有产物默认跳过，不会被静默覆盖。

## 训练与评测

```bash
# SFT（示例；实际租约 ID 由复现者自己的调度系统提供）
python src/train/train_sft.py --config configs/sft_v1.json --device cuda --gpu-lease-id <LEASE_ID>

# 独立评测（以 synthetic eval 为例）
python src/eval/run_zeroshot.py --run-id <run_id> --dataset synthetic --split eval \
  --data data/synthetic_eval.jsonl --output results/zeroshot_<run_id>.json \
  --details-output results/eval_<run_id>_details.jsonl \
  --adapter adapters/sft_v1/adapter --expected-n 1000 --device cuda

# 逐条重算验证
python src/eval/validate_results.py --result results/zeroshot_<run_id>.json \
  --report results/eval_<run_id>_validation.json \
  --expected-run-id <run_id> --expected-dataset synthetic --expected-split eval \
  --expected-n 1000 --expected-adapter adapters/sft_v1/adapter
```

坐标变体、ReST、教师蒸馏、ScreenSpot 与部署分别使用 `configs/` 与 `src/eval/` 下对应入口；`scripts/run_all.py` 是统一调度入口。

## 不包含的大文件

仓库不包含：

- 模型权重：`models/SmolVLM-256M-Instruct/`
- LoRA/merged/量化产物：`adapters/`
- 数据图片：`data/images/`、`data/raw/`
- 训练日志与历史归档：`logs/`、`archive/`

这些文件需要按 `src/data/`、`src/train/` 与部署代码自行生成或从对应模型/数据集页面下载。

## 结果与验证

`results/` 中保留全部已落盘 summary、details、validation、verify 与失败/负结果证据。关键入口包括：

- `results/eval_sft_v1_3seed_summary.json`
- `results/eval_t7_g4_summary_3seed.json`
- `results/g5_rest_curve.json`
- `results/g6_kd_vs_sft.json`
- `results/eval_screenspot_summary.json`
- `results/deploy_execution_summary.json`
- `results/t10_verifier_noise.json`

每个结果 JSON 记录 `run_id`、源码哈希、数据哈希、adapter 哈希与配置指纹。`EXPERIMENT-MANIFEST.json` 提供仓库内文件级 SHA-256，可用于核对 checkout 完整性。
