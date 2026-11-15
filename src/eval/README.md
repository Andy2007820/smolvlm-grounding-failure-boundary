# Independent grounding evaluation (eval-keeper)

This directory owns the evaluator. Training code may call it, but trainers do
not modify it. Core parsing and metrics are CPU-only and do not import torch.

## Frozen metric contract

All boxes are continuous `[x1, y1, x2, y2]`. IoU uses intersection / union and
does **not** add 1 to widths or heights.

For `n` selected evaluation samples:

- `Acc@0.5 = count(valid prediction and IoU >= 0.5) / n`.
- `mIoU = sum(sample IoU) / n`; an invalid prediction contributes IoU 0.
- `parse_rate = count(valid parsed prediction) / n`.
- `invalid = n - count(valid parsed prediction)`.

Invalid predictions remain in every denominator. A data/GT error is different:
it aborts preflight with dataset, split, sample id and reason; it is never
silently reclassified as model `invalid`.

### Prediction grammar

1. A single square-bracketed candidate with exactly four numbers takes
   precedence over surrounding prose. Multiple such boxes are ambiguous and
   invalid. Without brackets, the entire output must contain exactly four
   numbers. Digits attached to labels such as `x1=` are not coordinates.
2. Four values in `[0,1]` are normalized coordinates. Otherwise all values must
   be integers in `[0,1000]` and are divided by 1000. Mixed normalized/bin
   values and fractional bin values are invalid.
3. Coordinates must be finite, satisfy `x2>x1` and `y2>y1`, and be in range.
   The evaluator never clips, swaps or otherwise repairs a prediction.

Synthetic occlusion answers are the disclosed full pre-occlusion object boxes;
evaluation uses the existing `answer` unchanged. ScreenSpot source boxes are
normalized; the data pipeline already converted them once to unified pixel
`raw_bbox` plus normalized `answer`. Evaluation performs no further GT
normalization and retains the disclosed annotation-variation pair.

## Shared p1 and image pipeline

`prompts.py` is the source of truth for training and evaluation:

```text
Locate the object described below in the image.
Description: {question}
Answer with the bounding box as normalized coordinates [x1, y1, x2, y2], each between 0 and 1.
```

The chat template inserts exactly one `<image>` item. B0 fixes seed 42,
greedy decoding (`do_sample=False`, `num_beams=1`), bf16 on CUDA,
`max_new_tokens=64`, local model files only, one image view with
`do_image_splitting=False`, and processor size/max image size longest edge 384.
The no-split Idefics3 path produces square 384x384 pixels. With patch size 16
and scale factor 4, `image_seq_len=(384/16/4)^2=36`; the processor is explicitly
set to 36 rather than its 512px default of 64. A CPU probe on the first
synthetic sample produced `pixel_values=[1,1,3,384,384]`, 36 image placeholders
and 93 total input tokens. The known tokenizer pad warning is kept visible; it
is not silently patched by the evaluator.

## Files

- `grounding.py`: parser, IoU, per-sample score and aggregates.
- `dataset.py`: read-only eight-field JSONL/image/GT preflight.
- `prompts.py`: shared p1 template and chat message construction.
- `run_zeroshot.py`: deterministic inference, resumable JSONL trace and summary.
- `validate_results.py`: independent schema/hash/detail/metric recomputation.
- `test_grounding.py`: CPU regression tests and JSON test report.

## CPU tests and preflight

Run from `final-project/`:

```powershell
python -m src.eval.test_grounding --report results/eval_unit_tests.json
```

Current result: 15 lightweight tests passed, including exact IoU=0.5 inclusion,
invalid predictions in all denominators, one-shot iterators, malformed
coordinates, invalid GT context and the 384px visual-token formula. See
`results/eval_unit_tests.json` for source hashes. A separate real-model CPU
regression also passed 2/2 tests (square `synthetic_eval_000001` and rectangular
`synthetic_eval_000005`): each had 36 image placeholders,
`pixel_values=[1,1,3,384,384]`, and completed a one-token greedy forward. See
`results/eval_cpu_forward_tests.json` and rerun with:

```powershell
python -m src.eval.test_processor_forward --report results/eval_cpu_forward_tests.json
```

Both final B0 selections passed `--dry-run` (1000 rows each, all images and GT
checked, no torch/CUDA model load):

```powershell
python -m src.eval.run_zeroshot --run-id zeroshot_b0_synthetic_s42_v1 --dataset synthetic --split eval --data data/synthetic_eval.jsonl --output results/zeroshot_synthetic.json --details-output results/eval_zeroshot_b0_synthetic_s42_v1_details.jsonl --model-path models/SmolVLM-256M-Instruct --expected-n 1000 --image-size 384 --max-new-tokens 64 --seed 42 --device cuda --dry-run
python -m src.eval.run_zeroshot --run-id zeroshot_b0_refcoco_s42_v1 --dataset refcoco --split val --data data/refcoco_val.jsonl --output results/zeroshot_refcoco.json --details-output results/eval_zeroshot_b0_refcoco_s42_v1_details.jsonl --model-path models/SmolVLM-256M-Instruct --expected-n 1000 --image-size 384 --max-new-tokens 64 --seed 42 --device cuda --dry-run
```

Data SHA256:

- synthetic eval: `2a61b1b72812c78fd62f56d429b1932b0cf2e2a370695149309aa3a3c4c71854`.
- RefCOCO val: `19d3c28100bb5ea0a8125ca498e736b28959bc350762115deed6a52eb57df316`.
- local model weights: `74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e`.

## Actual B0 (GPU lease required)

Remove `--dry-run` from the commands above only after explicit Lead approval.
The calling shell records `nvidia-smi --query-gpu=memory.used --format=csv`
before process start and after process exit in the corresponding
`results/eval_*_gpu.log`. The runner additionally records load/generation/
cleanup memory and torch peaks. More than 7500 MiB or a competing job aborts.

Every sample is appended immediately to `results/eval_*_details.jsonl` with
`run_id`, config fingerprint, id, GT, uncleaned raw decoded output, cleaned
output text, parsed box, parse error, IoU and correctness. A failure during
image/model inference aborts instead of becoming `invalid`. Resume an
interrupted run with the identical command plus `--resume`; id sequence,
dataset/model/source hashes and all generation settings must match.

The project was not a Git repository at preparation time. Results therefore
record `git_commit: null` and `git_commit_status`, plus code/data/model hashes;
no commit is invented.

## Validation

```powershell
python -m src.eval.validate_results --result results/zeroshot_synthetic.json --report results/eval_zeroshot_b0_synthetic_s42_v1_validation.json --expected-run-id zeroshot_b0_synthetic_s42_v1 --expected-dataset synthetic --expected-split eval --expected-n 1000
python -m src.eval.validate_results --result results/zeroshot_refcoco.json --report results/eval_zeroshot_b0_refcoco_s42_v1_validation.json --expected-run-id zeroshot_b0_refcoco_s42_v1 --expected-dataset refcoco --expected-split val --expected-n 1000
```

Validation checks the required JSON contract, fixed B0 config, data/detail
hashes, selected ID sequence, per-sample raw output and recomputed scores, then
recomputes every aggregate from all 1000 details. See `CHANGELOG.md` for the
pre-B0 development record and post-B0 change policy.

## Read-only adapter (SFT) evaluation

`run_zeroshot.py` accepts an optional `--adapter <dir>` for evaluating a PEFT
LoRA checkpoint (for example `adapters/sft_v1/adapter`) with the same frozen
parser, IoU, Acc@0.5, mIoU, parse_rate and `n` denominators. The adapter is
loaded with `PeftModel.from_pretrained(base, is_trainable=False,
torch_device=device)` and the runner asserts that at least one LoRA module is
attached, that no parameter is trainable and that the model is in eval mode.
Adapter config/weight SHA256, rank/alpha/dropout and PEFT version are recorded in
the result `config.adapter` and in the `adapter` summary field; `adapter_load`
records the frozen-load evidence. The adapter directory is only read.

```powershell
python -m src.eval.run_zeroshot --run-id sft_v1_s42_synthetic_v1 --dataset synthetic --split eval --data data/synthetic_eval.jsonl --output results/zeroshot_sft_v1_s42_synthetic_v1.json --details-output results/eval_sft_v1_s42_synthetic_v1_details.jsonl --model-path models/SmolVLM-256M-Instruct --adapter adapters/sft_v1/adapter --expected-n 1000 --image-size 384 --max-new-tokens 64 --seed 42 --device cuda --dry-run
python -m src.eval.validate_results --result results/zeroshot_sft_v1_s42_synthetic_v1.json --report results/eval_sft_v1_s42_synthetic_v1_validation.json --expected-run-id sft_v1_s42_synthetic_v1 --expected-dataset synthetic --expected-split eval --expected-n 1000 --expected-adapter adapters/sft_v1/adapter
```

Remove `--dry-run` and run under a Lead-issued GPU lease exactly as for B0. The
RefCOCO command is identical with `--run-id sft_v1_s42_refcoco_v1 --dataset
refcoco --split val --data data/refcoco_val.jsonl`. `validate_results` keeps its
old behaviour when `--expected-adapter` is omitted (`adapter` must be `null`).

CPU evidence for this path (`gpu_used=false`, no CUDA context created):

```powershell
python -m src.eval.test_adapter_support --report results/eval_adapter_tests.json
python -m src.eval.test_adapter_forward_cpu --report results/eval_adapter_cpu_forward.json
```

These cover read-only adapter metadata/hash checks, rejection of missing or
wrong-base/non-LoRA adapters, the `--dry-run` preflight with and without an
adapter, and a real CPU base+adapter load (`is_trainable=False`, 120 LoRA
modules, 0 trainable parameters, 36 placeholders, one-token greedy generation).

## Optional batched evaluation (`--batch-size`)

`--batch-size` defaults to `1`, which runs the historical one-sample code path
unchanged. Values above 1 batch the greedy generation calls:
left-padded `input_ids`/`attention_mask`, `pixel_values` stacked per sample,
per-row prompt length recovered from the attention mask, generation sliced from
the padded input length, and each continuation trimmed at (and including) its
first EOS so a finished row never carries generation pads. Greedy
(`do_sample=False`, `num_beams=1`), 64 tokens, parsing, IoU and every metric
denominator are unchanged. `batch_size` is part of the config fingerprint, so a
resume can never mix batch sizes.

Batched runs are only allowed after a GPU A/B gate: same adapter, first 20
synthetic-eval records, batch-1 vs batch-8, byte-identical `raw_output` on
20/20 records. Any mismatch means permanent fallback to batch-1 with no mixing;
the evidence plan is recorded in `results/eval_batch_dryrun.json`.

**The GPU gate failed on 2026-11-09 (batch-8 matched batch-1 on only 5/20
`raw_output`; CPU float32 was 20/20).** `batch_size > 1` is therefore prohibited
for official evaluation and all reported SFT numbers use `batch_size=1`.
Failure evidence: `results/eval_batch8_ab_failure.json`,
`results/eval_ab_equivalence.json` and
`archive/t6eval_batch8_ab_failure_2026-11-09/`. CPU tests:

```powershell
python -m src.eval.test_batch_eval_cpu --report results/eval_batch_cpu_tests.json
```

The CPU suite covers EOS trimming, variable-length left padding, processor/pixel
equivalence, dry-run fingerprint changes and a real CPU batch-1 vs batch-2 run
with byte-identical outputs; it validates the machinery, not GPU bf16 identity.

## T7 coordinate-representation variants (`--prompt-id`)

`--prompt-id` selects the answer representation and its exact training prompt:

| variant | prompt id | representation | metric path |
|---|---|---|---|
| A | `p1` | normalized decimal `[x1,y1,x2,y2]` in [0,1] | frozen `parse_box`/`score_prediction` |
| B | `p1_bin1000` | integers in [0,1000] | frozen `parse_box` integer/1000 branch |
| C | `p1_pixel` | integer pixel xyxy | independent `pixel_convert` -> validate bounds/order -> normalize -> frozen IoU/Acc/metrics |

Prompt text is byte-identical to `src/train/coordinate_variants.py` (asserted by
CPU test and recorded as `prompt_sha256` in the config/fingerprint). C never
calls `parse_box` (it would misread pixels as 1000-bin values); invalid pixel
outputs remain in `n` with IoU 0. `grounding.py` and all metric denominators are
untouched. Examples:

```powershell
python -m src.eval.run_zeroshot --run-id t7_b_synthetic_v1 --dataset synthetic --split eval --data data/synthetic_eval.jsonl --output results/zeroshot_t7_b_synthetic_v1.json --details-output results/eval_t7_b_synthetic_v1_details.jsonl --model-path models/SmolVLM-256M-Instruct --adapter adapters/sft_t7_b_bin1000/adapter --prompt-id p1_bin1000 --expected-n 1000 --image-size 384 --max-new-tokens 64 --seed 42 --device cuda --batch-size 1
python -m src.eval.validate_results --result results/zeroshot_t7_b_synthetic_v1.json --report results/eval_t7_b_synthetic_v1_validation.json --expected-run-id t7_b_synthetic_v1 --expected-dataset synthetic --expected-split eval --expected-n 1000 --expected-adapter adapters/sft_t7_b_bin1000/adapter --expected-prompt-id p1_bin1000
```

CPU tests / dry-run evidence:

```powershell
python -m src.eval.test_t7_variants_cpu --report results/eval_t7_variants_cpu_tests.json
```

Reports: `results/eval_t7_variants_cpu_tests.json` (16 tests),
`results/eval_t7_cpu_dryrun.json` (three-prompt dry-run with adapter A, plus
CPU end-to-end validations for B/C paths). Plans and final G4 table:
`results/eval_t7_batch1_plan.json`, `results/eval_t7_g4_summary.json`,
`report/tables/table_g4_coordinate_variants.*`.
