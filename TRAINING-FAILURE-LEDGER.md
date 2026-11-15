# 训练失败与负结果台账

> 建立时间：2026-11-09（UTC+8）
> 原则：**失败、崩溃与负结果全部保留，不删除、不覆盖、不挑好结果。**
> 证据副本：`archive/training_failures_2026-11-09/`；副本清单与逐文件 SHA-256：`archive/training_failures_2026-11-09/manifest.json`（manifest SHA256=`e8a4925758ee7216153452554590fd118a5dd65b0e40190114b07a2ea0bc506f`）。原始文件仍保留在原路径。

## 一、失败训练运行

| 编号 | 运行/阶段 | 结果 | 根因 | 是否产生有效指标 |
|---|---|---|---|---|
| T5-A1 | `sft_overfit100_s42` GPU attempt1，14:56:05–14:56:13 | exit 1，训练前崩溃 | 120 个 LoRA q/k/v/o 模块集合完全相同，但校验用有序列表比较；排序不同导致 `missing=[] extra=[]` 的假不匹配 | 否；未创建 optimizer，无 forward/backward/step/checkpoint |
| T5-A2 | 100 条过拟合，200 optimizer steps（run_id=`sft_overfit100_s42`） | 完成但定位负结果 | 格式已学会，定位未学会；预算/容量不足 | 有诊断指标：parse_rate=1.0、acc@0.5=0.0、mIoU=0.0303（20 条训练内 probe，非官方评测） |
| T5-E48 | 100 条过拟合，600 optimizer steps（run_id=`sft_overfit100_e48_s42`） | 完成但定位仍负结果 | 延长预算只提高覆盖度，未出现任何 IoU≥0.5 样本；不是数据/标签错位 | 有诊断指标：parse_rate=1.0、acc@0.5=0.0、mIoU=0.0842、max IoU=0.425（20 条训练内 probe，非官方评测） |
| T5-P16 | 首次 100 条训练后 probe，`max_new_tokens=16` | 测量失败，非模型失败 | 16 tokens 只能生成两个坐标即截断，20/20 invalid；`parse_rate=0` 是截断伪影 | 否；已作废，不作为 G3 或调参依据 |
| T5-PROBE-A1 | probe-only attempt1，15:11:28–15:11:36 | exit 1，生成前失败 | 加载 processor 后未先设置 384/no-split；默认 512 切图产生 17×64 token 与 36 placeholders 不匹配 | 否；未生成，未写正式 probe |
| T4-EVAL-A0 | B0 synthetic 首次 GPU 生成 | CUDA device-side assert，未产生指标 | 384/no-split 实际视觉 token=36，但 processor 默认 `image_seq_len=64`；视觉特征 `masked_scatter` 数量不匹配 | 否；details 0 行，未计 invalid，未启动 RefCOCO |
| T6-CPU-A1 | seed43/44 CPU dry-run 配置校验，18:11:41 | 两次 dry-run 均 exit 1，GPU/训练未开始 | `validate_config` 仍把 seed 固定为 42，sft_v1_s43/s44 被 `ConfigError: seed must be 42` 拒绝 | 否；未读数据/未建 plan/无 adapter/checkpoint/训练指标 |
| T6-S43-A1 | seed43 训练，18:53:45 step800 后 | Python 进程消失，无 training summary；最近完整 checkpoint=600 | 疑似执行 wrapper/会话结束误杀子进程；stderr 无 Python 异常，无 OOM/竞争/数值问题 | 否；未产生最终指标（checkpoint-600 可用，已 resume） |
| T7-C-A1 | T7-C pixel 训练 attempt1，02:57:48 | exit 1，首个 RefCOCO 样本编码阶段 | C 像素表示误用 1e-8 一致性断言；RefCOCO 整数 raw_bbox 与归一化 answer 有 ≤1px 舍入差 5.6e-4 | 否；未产生 step/checkpoint/指标 |

## 二、失败原因与后续改进

### T5-A1：LoRA target 校验误报

- **失败阶段**：模型权重加载成功（471/471）后、`build_lora_model()` 校验处；训练尚未开始。
- **根因**：发现函数按 `(layer, leaf)` 排序，配置按 q/k/v/o 顺序；两个集合相等但列表顺序不同。
- **已实施修复**：新增 `lora_targets_match()` 做集合比较；新增 reversed/sorted/缺元素回归断言；Lead 独立复跑 15/15。
- **后续改进**：所有“集合等价”检查一律比较集合或规范化排序；GPU 前先跑 CPU 单测与编译。
- **留档**：`logs/train/sft_overfit100_attempt1_failure.log`、`results/train_t5_gpu_failure_attempt1.json`。

### T5-A2 / T5-E48：100 条过拟合定位负结果

- **现象**：200 步 loss 1.5556→0.9983，600 步→0.8390；两次 probe 均能完整输出 4 个归一化坐标并到 EOS，但 acc@0.5 都是 0；600 步 mIoU 从 0.0303 提到 0.0842，仍无 IoU≥0.5。
- **已排除**：100 条答案与 raw_bbox/宽高一致 100/100；标签解码与 GT 加 EOS 一致 100/100；prompt 全 mask；36 个图像 placeholder 全匹配；xywh/平移假设更差。
- **结论**：**不是数据、图像、标签、答案对齐或 vision 特征注入 bug**；是 LoRA/优化预算/数据规模限制。不得通过删样本、改评测器或只报子集来“修复”。
- **后续改进**：
  1. 不再盲目延长 100 条步数；单样本正对照已证明优化链路可学会定位。
  2. 进入 T6 全量 SFT，用完整数据提供多样化定位信号；G3 必须由独立评测在 synthetic eval/RefCOCO val 上判定。
  3. 若全量 SFT 的 acc 仍不高于 B0，再启用 T7 坐标表示消融（文本/归一化/`<loc>`），而不是继续堆步数。
  4. 长训练预算只在有明确容量/课程假设、且经 Lead 新租约批准时使用。
- **留档**：原始训练/probe/配置/日志均在原路径；证据副本见 archive 的 `t5_100_records_200_steps_negative` 与 `t5_100_records_600_steps_negative`；分析见 `results/train_t5_e48_probe_analysis.json`、`results/train_t5_e48_alignment_audit.json`。

### T5-P16：probe token 上限测量缺陷

- **根因**：`probe.max_new_tokens=16`，完整 4 坐标答案需要约 33 tokens；输出在第二个坐标后被截断。
- **已实施修复**：保留截断文件为 `results/train_sft_overfit100_s42_probe_truncated_at_16.json`；另做 64-token probe-only；T6 配置 `sft_v1.json` 已改为 64。
- **后续改进**：正式评测前先检查 `hit_max_new_tokens`/EOS/截断率；任何 `truncated>0` 的诊断不得用于结论；训练诊断固定 `max_new_tokens=64` 并记录生成 token 数。
- **留档**：截断 probe、64-token probe、`results/train_t5_gpu_run_report.json`。

### T5-PROBE-A1：probe-only processor 配置遗漏

- **根因**：probe-only 新路径未复用训练时的 `configure_processor(384/no-split)`；默认 512 切图与 36 个 placeholder 不一致。
- **已实施修复**：加载 processor 后先配置 384/no-split，再断言 views×36==placeholders；新增 CPU AST 测试禁止 probe-only 训练/反向传播。
- **后续改进**：所有推理入口共享视觉 token helper 与 preflight；新入口先用方形+长方形各一条做 CPU 断言，再申请 GPU。
- **留档**：`results/train_t5_probe_only_attempt1_failure.json`、`logs/train/sft_overfit100_probe_stderr.log`。

### T4-EVAL-A0：B0 CUDA assert（非训练，但同属失败留档）

- **根因**：384/no-split 视觉序列实际 36 tokens，processor 默认 placeholder 64；CUDA 异步栈落在 rotary，但根因在视觉 `masked_scatter`。
- **已实施修复**：每次调用按 `image_size/patch_size/scale_factor` 计算并设置 `image_seq_len=36`；方形/长方形 CPU 前向 2/2 通过后才重跑；GPU 修复版完成两集 B0 并经验证。
- **后续改进**：训练/评测/量化共用视觉 token contract；T6 不得沿用默认 64；任何 384/no-split 任务先断言 `views × tokens_per_view == placeholders`。
- **留档**：attempt0 空 details、`gpu.log`、`blocking_stack.log`、`failure_analysis.json`、`eval_cpu_forward_tests.json`。


### T6-CPU-A1：多 seed 配置被固定 seed 校验阻塞（追加于 2026-11-09 18:20）

- **阶段**：seed43/44 CPU dry-run 的 `load_config` 校验；**未读数据、未建计划、未启动 GPU/训练**。
- **根因**：`src/train/config.py` 的 `validate_config` 仍要求 `seed == FIXED_SEED(42)`；新增的 `sft_v1_s43/sft_v1_s44` 被 `ConfigError: seed must be 42, got 43/44` 拒绝。
- **影响**：两 dry-run 立即 exit 1；无 adapter/checkpoint/训练 JSONL/results 产物，无污染；GPU 未使用。
- **修复**：`ALLOWED_SEEDS=(42,43,44)`（其他 T5/T6 固定契约校验不变）；`test_01` tamper 用例改为 seed=41；新增 `test_19` 校验 s43/s44 仅 seed/run_name/output/log 不同且继承完整契约；修复后两份 dry-run preflight_passed（10000 条、1250 步、ID 交集 0、36 placeholders），19/19 单测通过。
- **证据**：`results/train_t6_cpu_preflight_blocker.json`、`logs/train/sft_v1_s43_dryrun_attempt1_stderr.log`、`logs/train/sft_v1_s44_dryrun_attempt1_stderr.log`；archive 副本见 `archive/training_failures_2026-11-09/t6_cpu_preflight_seed_validation/`；manifest 新 SHA256=`d4b356d215fb230ae03e71af32228dacf5d0fbbb5dab4e52a88a03a72ce0c043`。
- **改进**：多 seed/多配置扩展时，先确认 fixed-contract 校验是否允许目标 seed；所有新 seed 配置必须走同一 dry-run + 单测门禁后再申请 GPU。


### T6-S43-A1：seed43 在 step800 后被 wrapper/会话终止（追加于 2026-11-09 19:04）

- **阶段**：seed43 训练到 optimizer step 800（18:53:45）后 Python 进程消失，早于 checkpoint-000800 写入。
- **现场**：最近完整 checkpoint=`adapters/sft_v1_s43/checkpoint-000600`（step600/samples4800）；无 training_summary；stderr 无 Python 异常栈；GPU 回落 1394 MiB，无 python 计算进程；无 OOM/竞争/CUDA assert/非有限 loss。
- **根因判断**：执行包装器/会话生命周期误杀子进程（基础设施问题），非模型/数据/评测器 bug。
- **影响**：无有效最终指标；无污染产物；checkpoint-600 完整可用于确定性 resume。
- **处置**：用原 lease `GPU-LEASE-T6SEEDS-20261109-08`、同 config/output/log 执行 `--resume latest`（从 checkpoint-600），并全程用 job_output 保活监控；恢复到 1250 步。
- **改进**：长时间训练必须保持本轮 turn 存活并用 job_output 轮询；保持 200-step checkpoint 作为恢复点；resume 校验 config fingerprint/seed/model/data 哈希。
- **证据**：`results/train_t6_seed43_abort_step800.json`、`logs/train/sft_v1_s43_gpu.log`、`logs/train/sft_v1_s43_stderr.log`、`logs/train/sft_v1_s43.jsonl`、`checkpoint-000600/checkpoint_meta.json`；archive 见 `archive/training_failures_2026-11-09/t6_seed43_wrapper_abort_step800/`；manifest 新 SHA256=`e8add8337e88e667a6e7884e871ad701177f42514b9d1ec8fa17717f3b835d04`。


### T7-C-A1：C 像素表示对 RefCOCO 的整数像素舍入容差过严（追加于 2026-11-09 03:02）

- **阶段**：T7-C attempt1 首个 RefCOCO 训练样本的答案编码；未进入 optimizer step，无 checkpoint/指标。
- **根因**：RefCOCO `raw_bbox` 为整数像素、归一化 answer 来自源浮点 bbox，存在 ≤1px 舍入差（本次 5.625e-4）；C 编码最初复用 1e-8 断言，过严。
- **影响**：GPU 已加载但训练未开始，无污染产物；C 的失败日志与 start 事件保留。
- **修复**：`pixel_rounding_tolerance = 1/width + 1/height + 1e-8` 并记录 `max_abs_pixel_answer_gap`；test_20 增加 50 条 RefCOCO C 编码/转换测试；20/20 CPU 通过后同 lease 重跑 C（attempt2）。
- **证据**：`results/train_t7_c_attempt1_failure.json`、`logs/train/sft_t7_c_stderr.log`、`logs/train/sft_t7_c_gpu.log`、`logs/train/sft_t7_c_pixel.jsonl`；archive 见 `archive/training_failures_2026-11-09/t7_c_attempt1_pixel_tolerance/`；manifest 新 SHA256=`e8a4925758ee7216153452554590fd118a5dd65b0e40190114b07a2ea0bc506f`。
- **改进**：像素表示上线前必须用真实 RefCOCO 样本（非仅 synthetic 方形/长方形）做编码 roundtrip；整数像素 GT 与归一化 GT 的容差按 1px 上限设计。

## 三、正对照（不是失败，保留以防误判）

**T5 单样本可学习性对照**（`sft_overfit1_e1600_s42`）：仅 1 条训练样本、200 步，loss 1.5422→0.000295；probe 预测 `[0.2891,0.7500,0.4883,0.9094]`，GT `[0.2890625,0.75,0.48828125,0.909375]`，IoU=0.99956、acc=1.0、parse=1.0、显式 EOS、无截断。

该正对照证明 vision 注入、`pixel_values` 传递、label mask 与 loss 路径有效，因此 100 条负结果不能归因于“图像没有进入模型”或标签错位；其证据同样保留，不删除。

## 四、保留与审计规则

1. 原路径文件全部保留；archive 中为逐字节副本，不以副本替代原件。
2. `archive/training_failures_2026-11-09/manifest.json` 记录每个副本的 bytes 与 SHA-256；任何后续清理不得删除这些路径。
3. 新的失败/负结果必须追加到本台账和 archive manifest，写明 run_id、阶段、根因、是否污染产物、改进动作和证据路径。
4. 诊断指标一律标注 `diagnostic_only=true, official_eval=false`；G3/正式表只使用独立评测产生的 `results/*.json`。
5. 禁止把失败运行的 checkpoint 当最终模型、禁止覆盖失败 probe、禁止删除失败日志。

## 五、T6/G3 独立评测期间的失败（2026-11-09）

### G3-EVAL-A1：synthetic GPU attempt1 启动失败（包装器缺陷，非模型/数据/评测器）

- **阶段**：2026-11-09 17:20:00（UTC+8，租约 GPU-LEASE-T6EVAL-20261109-07），synthetic 评测 attempt1 在模型加载阶段被中断，**0 样本**。
- **根因**：PowerShell 启动包装器设置 `$ErrorActionPreference='Stop'`，把 transformers `pad_token_id=128002` 的 stderr 警告转成 NativeCommandError 并终止子进程；不是模型、数据或评测器失败。
- **污染**：无。未创建 details/summary，未产生任何指标；GPU 立即释放（1505→1504 MiB，无 python 计算进程）。
- **修复**：attempt2 改用 `Start-Process` 显式重定向 stdout/stderr 并去掉 Stop；同一条已批准命令随后完成 synthetic（n=1000，validation passed）→ RefCOCO（n=1000，validation passed）。
- **留档**：`results/eval_sft_v1_s42_synthetic_launch_failure.json`、`results/eval_sft_v1_s42_synthetic_gpu_attempt1.log`、`results/eval_sft_v1_s42_synthetic_stderr_attempt1.log`；副本与逐文件 SHA-256：`archive/t6eval_launch_failure_2026-11-09/`，manifest SHA256=`f866d67ba1f2edaebdd8dcc3eb90e0b76ee82309f672676c4c2513d92dd585eb`。原始文件全部保留。
- **工具缺陷备注**：本会话 PowerShell 的 `Start-Process -PassThru` 不返回 `Process.ExitCode`（为 `$null`），故 gpu.log 的 `python_exit_code=` 为空；DSH 后台任务包装器记录 exit code=0，且两次 Python 运行均写出 `status=completed`、`details_rows=1000` 的 summary。gpu.log 已追加 `python_exit_code_observation=` 行说明，未修改历史行。


### G3-EVAL-A2：GPU batch8 等价性门禁失败（2026-11-09）

- **阶段**：20:50–20:52（UTC+8，GPU-LEASE-T6EVAL3-20261109-09），同一 adapter=`adapters/sft_v1/adapter`（seed42），synthetic_eval 前 20 条，batch1 vs batch8。
- **结果**：`raw_output` 仅 **5/20** 逐字一致（15/20 不同），predicted_box 5/20、iou 10/20、correct@0.5 18/20；passed=false。旧官方 seed42 batch1 与新 batch1 仍 **20/20** 一致。
- **根因**：CPU float32 的 batch1 vs batch8 是 20/20 一致，说明左 padding/切片/EOS 截断逻辑正确；GPU bf16 下批量矩阵乘/reduction 顺序改变 greedy argmax，**batch8 与 batch1 非逐位等价**，不满足项目“同评测方法”要求。
- **污染**：无。A/B 后立即停止，**未运行任何 batch8 正式 run**（0 个），未产生/污染官方指标；GPU 立即回到 1289MiB、无 python 进程。
- **处置**：永久回退 batch1；batch8 不得用于任何官方数字、不得与 batch1 混用。剩余 seed43/44 × 两集改用 batch1 顺序重跑（Lead 延长租约至 22:30）。
- **留档**：`results/eval_batch8_ab_failure.json`、`results/eval_ab_equivalence.json`；证据副本 15 个文件在 `archive/t6eval_batch8_ab_failure_2026-11-09/`，manifest SHA256=`{manifest_sha}`。原始文件全部保留。

- **A/B 证据 SHA256（G3-EVAL-A2 补充，2026-11-09）**：
  - `results/eval_ab_equivalence.json` = `c48dede1d7fcec9025fb84ee7a76881c5b8be13163fe806be9363216e8529878`
  - `results/eval_batch8_ab_failure.json` = `5c03eea0df1f418a3d552e3b39b09baf92e810d95c9af41f8ea1789414d22562`
  - `results/eval_ab_batch1_s42_20_details.jsonl` = `c6af24c08cc7bf3681ce3c48a7f38142b7714379b4fd480f421d7fddc7a619c6`
  - `results/eval_ab_batch8_s42_20_details.jsonl` = `a7658ecf533d51c11db2bfa9f387299a554f1834c8c58d08e23f92ff547e0706`
  - `results/zeroshot_ab_batch1_s42_20.json` = `3d55393b5ddaa04c838cfec9ee1d4f4c710916ad1b6bb0fb6cba16cc0ae17fce`
  - `results/zeroshot_ab_batch8_s42_20.json` = `aceb3a63f4de24570e131b382393fe1908740dbc5af8a0cf0f26dc50340a72bd`
  - `results/eval_ab_batch1_s42_20_gpu.log` = `9e0e112e6972a4f3ea6283ac742a869b98a89b22803567161ebf1cea22dd9f04`
  - `results/eval_ab_batch8_s42_20_gpu.log` = `c94e3b0675b8315897a562221eac4389bf1e5bdbdbaea5b0edc295ed4321b8a2`
  - `archive/t6eval_batch8_ab_failure_2026-11-09/manifest.json` = `a19f7a624c845af33a072bb352f1e5186a2f2a9a6bd8d55c2140b36baae124be`（内含 15 个副本的 bytes/SHA256）


### T6C-EVAL-A1：seed43 synthetic run 在 998/1000 被 driver 超时终止（已 resume 修复）

- **阶段**：2026-11-09 20:53:47–21:33:50（UTC+8，GPU-LEASE-T6EVAL3-20261109-09）。
- **结果**：driver 以 `subprocess.run(timeout=2400s)` 运行 run1，run1 实际约 2403s（慢速阶段 ~2.4s/条），子进程在 **998/1000** 行被终止；当时无 summary、无 validation、无任何指标产出。
- **根因**：driver 固定 40min 子进程超时接近本 run 预计耗时，不是数据/模型/评测器失败，也不是租约边界（租约已延长）。
- **修复/改进**：同一 run_id + 同 config，仅追加 `--resume`（不改 ID、不加 limit）补最后 2 条；31.1s 完成，最终 1000 行、summary `status=completed`、details sha=`b601397e2264fa82f04e759772e7200d9b6f1ca34312bc0976ee96069d2d61c9`、validation passed/errors=[]。以后短租约禁止把固定 subprocess timeout 设在预计耗时附近；任何 partial 必须以 resume + summary + validation 作为完成门禁。
- **污染**：无。partial 阶段未写 summary/metrics；最终结果覆盖完整 1000 行并通过独立验证。
- **留档**：`results/eval_sft_v1_s43_synthetic_v1_partial_abort.json`、gpu/stdout/stderr 日志、validation；副本见 archive。


## 六、T6c 3-seed 评测最终状态（2026-11-09 22:15）

- **seed42**：沿用已完成并验证的 batch1 官方结果（synthetic acc@0.5=0.011 / mIoU=0.0674376394；RefCOCO acc@0.5=0.095 / mIoU=0.2112482887）。
- **seed43 synthetic**：先被 driver 固定 subprocess timeout 在 998/1000 终止（见 T6C-EVAL-A1），同一 run_id/config `--resume` 补齐；最终 n=1000、acc@0.5=0.004、mIoU=0.0522555185、parse_rate=1.0、invalid=0、details sha=`b601397e2264fa82f04e759772e7200d9b6f1ca34312bc0976ee96069d2d61c9`、validation passed/errors=[]。
- **seed43 RefCOCO**：21:38:05–22:13:54 完成 n=1000、acc@0.5=0.052、mIoU=0.1341380139、parse_rate=0.999、invalid=1、details sha=`695fbc4204a14d4e05ca71c63f8d0ac0a08841550acda8352f6e8f91d383b5e4`、validation passed/errors=[]，exit 0，未超 22:15 截止。
- **seed44**：**未进行任何独立评测**（无 seed44 评测输出；训练 adapter 已 final 但硬截止前无 GPU 时段）。
- **2-seed 汇总（仅 seed42+43，明确非 3seed 主表）**：`results/eval_sft_v1_2seed_summary.json` sha256=`cf227e19c2d7af07612e844c8d9c414b33feea09604c154b236201abbcbc80fa`；synthetic acc 0.0075±0.00495、mIoU 0.05985±0.01074；RefCOCO acc 0.0735±0.03041、mIoU 0.17269±0.05453（ddof=1, n=2）。
- **批次策略**：batch8 被 GPU A/B 门禁否决（5/20 raw_output），已永久回退 batch1，未混用；本次 4 个 run 全部 batch_size=1，validation 全 passed，hit_max_new_tokens=0。
- **GPU 释放**：run2 于 22:13:54 退出，gpu.log after=1578 MiB、exit 0；22:15 复查 1550 MiB、无 python 计算进程，租约按时结束。
- **证据 SHA256（关键）**：seed43 synthetic details=`b601397e2264fa82f04e759772e7200d9b6f1ca34312bc0976ee96069d2d61c9`；seed43 refcoco details=`695fbc4204a14d4e05ca71c63f8d0ac0a08841550acda8352f6e8f91d383b5e4`；2seed summary=`cf227e19c2d7af07612e844c8d9c414b33feea09604c154b236201abbcbc80fa`；A/B manifest=`a19f7a624c845af33a072bb352f1e5186a2f2a9a6bd8d55c2140b36baae124be`；run1 partial manifest=`d50157254234de1b681e71b5bc72afaa463cc175d52f59825537e97e99849df2`。


## 七、T6c 3-seed 评测完成（2026-11-10 01:38）

- **GPU-LEASE-T6EVAL4-20261110-01**（01:00–02:10，独占）：仅 seed44 两集 batch1，`--batch-size 1`，p1/384/no-split/seq36/max64/greedy/seed42，n=1000。
- **seed44 新结果**：
  - `sft_v1_s44_synthetic_v1`：n=1000、acc@0.5=0.01、mIoU=0.06043332964441271、parse_rate=1.0、invalid=0、details sha=`73553084102df724c7a4237fe9ab55ae1c1e4cc729670d12edaa28acba3115d5`、validation passed、wall=1420.9s、gpu.log exit0。
  - `sft_v1_s44_refcoco_v1`：n=1000、acc@0.5=0.097、mIoU=0.20310631666903378、parse_rate=1.0、invalid=0、details sha=`fdaf35b90f5235a2bfe45273c871f2d3cb7619130f6940d0093357cc140cab45`、validation passed、wall=1364.6s、gpu.log exit0。
  - GPU 01:38 释放：gpu.log after=1210/1081 MiB，仅 Harness。
- **3-seed 主表（ddof=1，n=3 seeds）**：
  - synthetic：acc@0.5=0.008333±0.003786、mIoU=0.060042±0.007599、parse=1.0±0、invalid=0±0。
  - RefCOCO：acc@0.5=0.081333±0.025423、mIoU=0.182831±0.042365、parse=0.999667±0.000577、invalid=0.3333±0.5774。
  - 合计（6 runs）：acc@0.5=0.044833±0.043162、mIoU=0.121437±0.072554、parse=0.999833±0.000408、invalid=0.1667±0.4082。
- **产物 SHA256**：3seed summary=`8c6c16d4f064414c4136aab8fa9b0ccdce541856d6b7e487d885e67f03387183`；table1 csv=`69b9581d4ca5e7d8703a77ca4a2541ec807e314df144294d3f8307dbb125c72e`；table1 md=`71183e4e71437c37b25733e3d0f411980aa508f4565fac17b347eca3259f3f35`；seed44 synthetic details=`73553084102df724c7a4237fe9ab55ae1c1e4cc729670d12edaa28acba3115d5`；seed44 refcoco details=`fdaf35b90f5235a2bfe45273c871f2d3cb7619130f6940d0093357cc140cab45`。
- 方法合规：6/6 validation passed、全部 batch1（batch8 永久禁用）、hit_max_new_tokens=0；3 seeds 全部报告，未挑 seed；seed42 G3 弱正结论不变；2-seed 文件已标记 superseded（数值未改）。


## 八、T14-eval merged CPU fp32 与 adapter GPU 官方结果偏差（2026-11-10，诊断留档）

- **运行**：run_id=`deploy_cpu_fp32_synthetic_v1`，adapters/merged/sft_v1_merged（model safetensors sha=`5398b41e2e069e8efb87da09d607e47295e85107edf9a8b47be4a211968387d2`），CPU float32、2 线程、p1/384/seq36/max64/seed42/greedy、batch1、n=1000；validation passed/errors=[]，details sha=`9427f6c2feb056701a5767e3d6f0a1635916131bb9bc3a8fe2336ce46fae27e3`。
- **结果**：merged CPU acc@0.5=0.017、mIoU=0.0837601、parse=1.0、invalid=0；seed42 adapter 官方（GPU bf16）acc@0.5=0.011、mIoU=0.0674376。仅 111/1000 raw_output 逐字相同、16/1000 correct@0.5 判定不同、IoU 平均绝对差 0.04998（最大 0.666）。
- **定性**：这是 CPU fp32 与 GPU bf16 的数值/设备差异，不是 merge 权重错误；此前 `deploy_merge_equivalence.json` 的 5/5 同设备输出逐字一致。merged 结果仅作部署诊断，**不得**作为 Table1/Table3 官方精度；官方数字仍用 adapter GPU 结果。
- **时序异常**：wall=3230.8s（53.8min），但单条样本曾停 1556.3s（主机 CPU 争用），median=1.465s/sample，与部署延迟 1.57s/sample 一致；该 wall 不可当 benchmark。
- **留档文件**：`results/eval_deploy_cpu_fp32_synthetic_summary.json`、`results/zeroshot_deploy_cpu_fp32_synthetic_v1.json`、`results/eval_deploy_cpu_fp32_synthetic_v1_details.jsonl`、`results/eval_deploy_cpu_fp32_synthetic_v1_validation.json`、`results/eval_deploy_cpu_fp32_synthetic_v1_run_info.txt`；archive 副本与 SHA 见 `archive/t14_cpu_fp32_deviation_2026-11-10/manifest.json`。

## 九、T8-R1-A1：rest_r1 首次训练在 step 25 被回收（2026-11-10，已由 Lead 重跑）

- **阶段**：2026-11-10 19:04:44–19:06:18（GPU-LEASE-REST-R1-TRAIN-20261110-01），configs/rest_r1.json、--max-steps 1250、训练数据 D0∪D+ =10,191 条、pool sha=bcf4275f204275445bdb57815217a10ab5e9f80a70f0e1c2e0b289fe7fd35494、config_fingerprint=b9a9699d6ad8c81ddbd36d69c0d223d861bada65675b450f4e4aed62e533b729。
- **结果**：训练进程在 **step 25** 被杀（samples_seen=200、elapsed=86.6s、最后 loss=1.2548169791698456）；logs/train/rest_r1.jsonl 共 26 行（start + step 1–25），无 completed / final_adapter 事件；adapters/rest_r1 为空、无 checkpoint 残留，无残留可 resume；GPU 回到 1358 MiB、无 python 计算进程。
- **根因**：作业生命周期归属错误 —— 训练由 teammate（gpu-runner）作为**自己的**后台作业启动，回合结束时该作业被回收，连带 python 训练进程被杀。与数据/模型/租约无关；未产出任何指标或 summary，不构成一次训练结果。
- **修复**：长任务（>5 分钟）一律由 Lead 启动并持有，teammate 只读日志/诊断/写 summary；Lead 已于 19:14 以完全相同的 config、参数、租约重跑（后台作业 pwsh-2183540f6c67436f），重跑 start 事件核对 train_records=10191、micro_forwards_total=10000、total_optimizer_steps=1250、id_intersection=0。
- **污染**：无。首发 attempt 未写 adapter/summary/metrics/results 文件。
- **留档**：logs/train/rest_r1_attempt1_killed_at_step25.log sha256=64dadce3556e05dc3051c90519db9aa9ccd2de37fd1010b936f7ecf776a5d8dd（10,288 B、26 行）、logs/train/rest_r1_attempt1_gpu.log sha256=a29b9d0585b4ab05aa1bc174a5c5d55a8bdc9ba17dc142699055433c5b69f69d（215 B）；副本与 archive/training_failures_2026-11-10a/manifest.json（sha256=2aeaeb90c92ccdb4db70e8d41b5446f2431d55bb5e3b7d590a35406c6febf50b）已归档。

## 十、T7-D-S42-A1：s42 训练数值完成但 finalize 写 probe 失败（路径命名冲突，已 resume 补齐）

- **阶段**：2026-11-12 00:04:41–01:03:19（GPU-LEASE-T7D-S42-20261112-01），configs/t7_d_grid100.json，--max-steps 1250。
- **结果**：训练 1,250 步 / 10,000 micro 全部完成；checkpoint-000200…001200、final adapter、adapters/sft_t7_d_grid100/training_summary.json 均已写出；训练级 probe 已算得 parse=1.0 / acc@0.5=0.05 / mIoU=0.13196229612719448。随后写 results/train_sft_t7_d_grid100_s42_training.json 之前，probe 结果写入 results/train_sft_t7_d_grid100_s42_probe.json 触发 FileExistsError（该路径已被 P1 的 dry-run --plan-out 占用；brief §5.1 模板命名与训练脚本 probe 默认名相同）。
- **根因**：路径命名冲突（brief 命令模板缺陷 + P1 证据落点与训练 probe 默认名相同），与数据/模型/数值稳定性无关。
- **修复**：3 个 dry-run plan 移到 results/train_t7_dryrun_plan_d{,_s43,_s44}.json（内容不删），从 checkpoint-001200 --resume latest 续跑最后 50 步补齐官方 finalize；并修复旧 --resume 不恢复 LoRA 权重的缺陷（见下条）。
- **污染**：attempt1 的 adapter 是完整 1,250 步权重但缺 results 训练 JSON；正式证据采用 resume 后完整链路产物，attempt1 仅作事故取证。
- **留档**：results/train_sft_t7_d_grid100_s42_attempt1_summary.json、results/train_sft_t7_d_grid100_s42_attempt1_record.json；artifacts/tmp/t7d_s42_attempt1/（adapter 副本、training summary、attempt record）、artifacts/tmp/t7d_s42_train.err、logs/train/sft_t7_d_grid100.jsonl。

## 十一、T7-C-S43-A2：旧 resume 缺陷导致 C seed43 非同质；经用户批准重训并替换（2026-11-12）

- **背景**：2026-11-11 旧 train_sft --resume 只恢复 optimizer/scheduler/RNG、**不恢复 LoRA 权重**；C seed43 首训被 job-runner 崩溃打断后从 checkpoint-400 续跑，实际以新初始化 adapter 继续 850 步。T7-D s42 恢复事故中暴露并修复该缺陷（见 T7-D-S42-A1）。
- **证据（旧 run，已归档）**：old adapter sha e3f1e3b82e87af7e54d67def0f59b990e2f2ff7d55454b3684801c1f2dfbe137；旧评测 synthetic acc 0.008 / mIoU 0.05225127496234196 / parse 0.989 / invalid 11，refcoco acc 0.004 / mIoU 0.0446142342401675 / parse 0.968 / invalid 32；整套旧产物在 archive/t7c_s43_oldresume_2026-11-12/oldrun/（含 adapter/checkpoints/training summary/eval files/log），manifest 见同目录 manifest.json。
- **授权**：用户 2026-11-12 明确允许改动 C 历史行并重训修正，同时要求记入本台账。
- **处置**：用修复后的代码从零重训 C seed43（1,250 步 / 10,000 micro，同预算），重跑两集评测与 validation，重建 C 3-seed 聚合；旧产物只移动/归档不删除。
- **新结果（2026-11-12 11:12 完成）**：adapter sha 760d81420bbda432e859d4cbc6e7d837e97cd2176432647be289cb590c3d7655；训练 1,250 步/10,000 micro、resume_count=0、stop_reason=max_steps、partial_final_step=false、loss 1.7714→1.0811；两集评测+validation passed：synthetic acc 0.021 / mIoU 0.056774 / parse 0.991 / invalid 9，RefCOCO acc 0.049 / mIoU 0.136543 / parse 0.871 / invalid 129；新 C 3-seed 聚合：synthetic 0.017667±0.003512 / mIoU 0.056593±0.001348 / parse 0.992000±0.001732 / inv 8.000±1.732，RefCOCO 0.045667±0.003055 / mIoU 0.135648±0.000775 / parse 0.899000±0.029052 / inv 101.000±29.052；results/verify_t7c_3seed.json 7/7 passed。
