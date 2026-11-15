# Training pipeline (train-runner)

T5 status (2026-11-09, CPU phase): **implementation + CPU preflight complete, including
a corrected true `batch=1 x gradient_accumulation=8` loop; no GPU load and no training
started.** Next action: Lead review of the corrected accumulation evidence, then an
explicit GPU lease before the 100-record overfit run.

Evidence:

| Artifact | Path | Result |
|---|---|---|
| CPU unit tests / preflight | `results/train_t5_cpu_tests.json` | **16/16 passed**, `gpu_used=false` |
| CPU dry-run plan (true accumulation) | `results/train_t5_dryrun_plan_v4.json` | `preflight_passed`, 1600 micro-batches / 200 optimizer steps |
| GPU overfit run + probe report | `results/train_t5_gpu_run_report.json` | 200/200 steps, hard counters pass, lease released |
| Corrected probe-only (64 tokens) | `results/train_sft_overfit100_s42_probe_maxtok64.json` | parse_rate=1.0, acc@0.5=0.0, mIoU=0.0303, 0 truncated |
| GPU lease request payload | `results/train_t5_gpu_lease_request.json` | exact command, estimates, logging, stop rules |

## 1. Fixed contract (rejected by `config.py` if drifted)

| Item | Value |
|---|---|
| LoRA | `r=16`, `lora_alpha=32`, dropout `0.05`, text-decoder `q/k/v/o` only |
| Learning rate | `1e-4` |
| Batch | `1 x 8` **true** gradient accumulation = effective batch `8` |
| Image | `384` px, no image splitting, single view |
| Precision | `bfloat16` |
| Vision tower | frozen (0 trainable vision params at runtime audit) |
| Seed | `42` (`torch`, CUDA, numpy, python; `cudnn.deterministic=True`) |
| Gradient checkpointing | enabled, `use_cache=False` |
| Answer EOS | explicit `<end_of_utterance>` = token **49279** appended to every answer |
| Pad | token **2** (`<|im_end|>`) |
| Train/val ID intersection | asserted **0** before training |

Configs: `configs/sft_overfit100.json` (100-record overfit test) and
`configs/sft_v1.json` (full SFT v1 plan for T6). Hyperparameter flags are not exposed
on the CLI; the config JSON is the single source of truth.

## 2. True gradient accumulation (`batch=1 x 8`)

Implemented once in `src/train/accumulation.py:run_true_gradient_accumulation` and used
by `train_sft.run_training`:

```
for micro in range(8):            # each collated as batch=1
    forward()                     # ONE sample per forward
    (loss / 8).backward()         # only gradient scaling
after 8 micro-batches:
    clip_grad_norm_  (once)
    optimizer.step() (once)
    scheduler.step() (once)
    optimizer.zero_grad() (once)
```

* `samples_seen` increments per micro-batch; `global_step` increments per optimizer step.
* Checkpoints/resume and the stream replay skip are expressed in **micro-batches**
  (`samples_seen`), so a resume replays exactly the samples already consumed.
* The previous pseudo-accumulation (collate 8 examples into one batch-8 forward, then
  multiply the already-averaged loss by 1/8) is removed; a regression test asserts the
  legacy `pending_examples` path is absent.

Executable CPU regression evidence (`results/train_t5_cpu_tests.json`):

| Check | Measured |
|---|---|
| micro-batch sizes seen by the model | `[1,1,1,1,1,1,1,1]` |
| forward / backward calls in one optimizer step | `8 / 8` |
| clip / optimizer step / scheduler step / zero_grad calls | `1 / 1 / 1 / 1` |
| loss scale per micro-batch | `0.125` |
| accumulated gradient for micro losses 1..8 (lr=0.1, SGD) | **4.5** = mean(1..8) |
| buggy double-scaled value (rejected) | `0.5625` = 4.5/8 |
| weight after one step | `0.55` = 1 − 0.1×4.5 |
| two-step gradients / final weight | `[4.5, 12.5]` / `−0.70` |
| partial final group (5 micro-batches) | gradient `15/8 = 1.875`, flagged `partial_accumulation=True` |
| max-steps stop | exactly 8 forwards, 1 optimizer step, `stop_reason=max_steps` |
| resume replay from step 1 / 8 samples | next step gradient `12.5`, `samples_seen=16` |

## 3. Visual-token contract (B0 regression guard)

The evaluator derives the placeholder count from the image/patch/scale geometry
(`src/eval/run_zeroshot.py:compute_image_seq_len`; re-used here, not copied). At
384 px / patch 16 / scale 4:

```
image_seq_len = (384 / 16 / 4) ** 2 = 36
```

The shipped `processor_config.json` stays at its default **64**. Every image
processing call sets the instance attribute to 36 through
`vision_tokens.set_runtime_image_seq_len`, which re-reads the JSON before and after and
raises if the on-disk default changed. Values observed on CPU for square, rectangular
and real RefCOCO photos:

| Case | pixel_values | views | tokens/view | placeholders | view*view_tokens |
|---|---|---|---|---|---|
| synthetic 384x384 | `[1,1,3,384,384]` | 1 | 36 | 36 | 36 |
| synthetic 256x320 | `[1,1,3,384,384]` | 1 | 36 | 36 | 36 |
| RefCOCO real photo | `[1,1,3,384,384]` | 1 | 36 | 36 | 36 |

A deliberate negative test (runtime `image_seq_len=64`) raises, so the original
CUDA-assert mismatch cannot silently return. Visual features are never padded or
truncated; `collate_examples` raises if view shapes differ.

## 4. Label masking and explicit EOS

Training inputs are the **processor-expanded** full conversation
(`processor(text=full_conversation, images=[image])`), not a raw tokenization of the
chat template (the raw template contains one `<image>` token; the processor expands it
to 36 image tokens + fake delimiters, a delta of 38 positions at 384 px).

The template also closes the **user** turn with `<end_of_utterance>`, so there are two
EOS ids in the sequence. Only the **second** EOS (the answer terminator) is a label:

```
user turn ... <end_of_utterance>(prompt, masked)
Assistant: [x1, y1, x2, y2]<end_of_utterance>(answer, labeled)
\n (masked)
```

Concretely for `synthetic_train_000004`:

* `label_start = 93` (answer starts exactly at the prompt boundary, verified by re-processing the generation prompt)
* answer span = 33 tokens (`" ["` is the first; BPE fuses the template space with `[`)
* answer EOS at position `125`; the single user-turn EOS at `88` is masked
* labels: `-100` for positions `0..92` and `126..126`; real ids for `93..125`
* decoded labeled span = ` [0.3490, 0.0365, 0.5859, 0.2734]<end_of_utterance>`
* `prompt_eos_positions=[88]`, `answer_eos_positions=[125]`

`build_sft_example` cross-checks three independent derivations (prefix extension of the
expanded generation prompt, EOS minus re-tokenized answer length, and character-offset +
image-expansion delta) and raises instead of guessing.

## 5. `pad_token_id=128002` investigation

The warning seen during G0/B0 is emitted by transformers while constructing
`Idefics3Config`: the class default for top-level `pad_token_id` is `128_002`
(`transformers/models/idefics3/configuration_idefics3.py`), validated against a legacy
32000-entry default text vocabulary. It is **not** used by anything in this project:

* tokenizer: pad `<|im_end|>` = **2**, eos `<end_of_utterance>` = **49279**
* `generation_config.json`: pad **2**, eos **49279**
* `text_config.pad_token_id`: **2**; `image_token_id`: **49190**
* every SFT answer explicitly appends 49279

`results/train_t5_cpu_tests.json → observations.pad_eos_audit` records the exact warning
text/source and the checks; `train_sft --dry-run` repeats the audit.

## 6. Commands

CPU tests (no weights, no CUDA, no training):

```powershell
python -m src.train.test_t5_cpu --report results/train_t5_cpu_tests.json
```

CPU dry-run preflight (validates config/data/images/visual-token/label contract,
computes the true-accumulation step plan, writes nothing to adapters/):

```powershell
python -m src.train.train_sft --config configs/sft_overfit100.json --dry-run --plan-out results/train_t5_dryrun_plan_v4.json
```

CPU training without `--dry-run` is refused by design; CUDA requires an explicit lease id:

```powershell
python -m src.train.train_sft --config configs/sft_overfit100.json --device cuda --gpu-lease-id GPU-LEASE-<...>
```

Common flags: `--resume [CHECKPOINT_DIR]` (default `latest`), `--max-steps N` (lease
profiling), `--profile-steps 25` (throughput/memory telemetry), `--no-probe`.

## 7. Checkpoints, resume and audit

* Every `training.save_every_steps` (100–200) **optimizer steps**: `adapters/<run>/checkpoint-000100/...`
* `checkpoint-*/adapter/` = PEFT weights + `adapter_config.json`
* `checkpoint-*/trainer_state.pt` = optimizer, scheduler, CPU/CUDA RNG states
* `checkpoint-*/checkpoint_meta.json` is written **last and atomically**; it records config fingerprint, seed, model/data hashes, selected/order digests, zero train/val ID intersection, LoRA target list/counts, trainable/frozen parameter counts, image/answer contract, true-accumulation contract, environment versions, GPU lease + memory telemetry, source hashes, and resume lineage. A directory without this marker is not resumable.
* Resume refuses any mismatch in config fingerprint, seed, model file hashes, dataset hashes, image_seq_len, EOS or pad id; `samples_seen` is the micro-batch count and replays exactly the consumed stream.
* Final run writes `adapters/<run>/adapter/` + `training_summary.json`,
  `results/train_<run>_training.json`, `logs/train/<run>.jsonl`, and a diagnostic
  `results/train_<run>_probe.json` (official metrics remain eval-keeper's job).

## 8. GPU lease gate

`train_sft` aborts when CUDA is requested without `--gpu-lease-id GPU-LEASE-...`, when
`nvidia-smi` reports more than 7500 MiB, or when the loss is non-finite. Before/after
memory is recorded by the calling shell plus in-process checks after load, at the profile
step, after training, and after cleanup. See `results/train_t5_gpu_lease_request.json`
for the exact request sent to the Lead: 100-record overfit run, 15–30 min estimate
(hard cap 40), ~2.5–3.5 GiB total estimate including the ~1.0–1.3 GiB GUI baseline,
first-25-step measured extrapolation, and expected accumulation counters
(1600 forwards, 200 optimizer steps).

## 9. Files

| File | Purpose |
|---|---|
| `config.py` | fixed-contract validation + config fingerprint |
| `vision_tokens.py` | shared visual-token helper, default-preservation audit |
| `sft_data.py` | answer formatting, label masking, dataset/collate, probe inputs |
| `accumulation.py` | **true batch=1 x 8 accumulation loop** (executable regression evidence) |
| `model_audit.py` | processor geometry, pad/EOS/128002 audit |
| `checkpoint.py` | atomic checkpoint metadata, resume guards, RNG state |
| `train_sft.py` | CLI, dry-run, training loop, profiling, probe, GPU lease gate |
| `probe_only.py` | inference-only re-probe of a trained adapter (no loss/backward/optimizer) |
| `test_t5_cpu.py` | 16 CPU tests + preflight/accumulation/probe-only evidence report |

## 10. Research-integrity notes

* No modifications to `data/`, `src/data/`, `src/eval/`, ground truth or existing
  `results/`; eval source hashes still match `results/eval_unit_tests.json`.
* Answers are rendered at 4 decimals (worst observed rounding error 5e-5 over the 100
  overfit records), with an automatic finer-precision fallback if rounding would make the
  frozen evaluator reject a box. The GT JSON is untouched.
* The GUI's ~1.0–1.3 GiB baseline is never described as "zero GPU usage".
* Probe metrics are explicitly labelled training diagnostics, not official evaluation
  numbers.

## 11. T5 100-record overfit result (attempt 2, lease GPU-LEASE-T5-20261109-02)

Hard acceptance counts from `results/train_sft_overfit100_s42_training.json`:

* forward=1600, backward=1600, optimizer steps=200, clip=200, scheduler=200, zero_grad=200
* micro_batches=1600, samples_seen=1600, every step `8 x batch=1`, `stop_reason=max_steps`
* torch peak 651.8 MiB allocated / 768.0 MiB reserved; nvidia-smi peak 2094 MiB; after-exit 1191 MiB
* checkpoints `checkpoint-000100`, `checkpoint-000200`, final adapter (sha256
  `47da671c...52ed7`), training 548.9 s

Diagnostic probe (not G3, not official evaluation):

* First probe was configured with `max_new_tokens=16` and truncated every answer mid-box; it is
  preserved as `results/train_sft_overfit100_s42_probe_truncated_at_16.json`
  (sha256 `d4b2145c...1154`) and must not be treated as a metric.
* Corrected inference-only probe (`results/train_sft_overfit100_s42_probe_maxtok64.json`,
  64 tokens, 20 records): **parse_rate=1.0, invalid=0, 20/20 EOS reached, 0 truncated**,
  **acc@0.5=0.0, mIoU=0.0303**.
* Interpretation: the 100-record overfit learned the answer format but not target locations at
  200 optimizer steps (predictions cluster in a narrow region). Honest negative/incomplete
  result; G3 is an independent eval-keeper decision after full SFT.
## 12. T5 extension e48 (600 optimizer steps) — result

* Config `configs/sft_overfit100_e48.json` (independent run/output/log; original run untouched).
* 600/600 optimizer steps, hard counters forward=backward=4800, optimizer/clip/scheduler/zero=600,
  micro_batches=samples_seen=4800, every step 8x batch=1; 26.98 min; peak nvidia 2101 MiB,
  torch peak 651.8/768.0 MiB; after-exit 1196 MiB.
* Probe (20 train records, 64 tokens, greedy): parse_rate=1.0, invalid=0, 20/20 EOS, 0 truncated,
  **acc@0.5=0.0, mIoU=0.0842** (up from 0.0303 at 200 steps; max IoU 0.425).
* Alignment audit (`results/train_t5_e48_alignment_audit.json`): 100/100 records have consistent
  answer/raw_bbox, evaluator-parseable answer text, prompt masking, explicit EOS=49279,
  36 image placeholders, and labels decoding exactly to GT answer; no coordinate-convention
  transform explains predictions. Conclusion: no data-label misalignment; localization remains a
  model/training-signal limitation. Per Lead instruction, step scaling stopped and no further
  steps were added.
* Evidence: `results/train_t5_gpu_run_report_e48.json`, `results/train_t5_e48_probe_analysis.json`.
## 13. T5 single-sample learnability control (200 optimizer steps)

* Independent config `configs/sft_overfit1_e1600.json`: only the first synthetic_train record
  (`limit=1`), 1600 micro-batches = 200 optimizer steps, same fixed contract, probe n=1/max64.
* 200/200 steps, hard counters forward=backward=1600, optimizer/clip/scheduler/zero=200,
  micro_batches=samples_seen=1600, every step 8x batch=1; 9.13 min; peak nvidia 2021 MiB,
  torch peak 644.7/688.0 MiB; after-exit 1190 MiB.
* Loss: step1 1.5422 -> step100 0.000414 -> step200 0.000295 (single-sample memorization).
* Probe: predicted `[0.2891, 0.7500, 0.4883, 0.9094]` vs GT
  `[0.2890625, 0.75, 0.48828125, 0.909375]`; IoU **0.99956**, acc@0.5=**1.0**,
  parse_rate=1.0, explicit EOS, 33 tokens (<64, no truncation).
* Conclusion: vision/pixel/label/loss path is functional; the 100-record failure is an
  optimization/capacity/data-scale limitation, not a pipeline or alignment failure.
  Evidence: `results/train_t5_gpu_run_report_overfit1.json`.
