# Evaluator change log

## v1.0.0-preB0 · 2026-11-09

Initial independent implementation before any B0 model result existed.

- Froze normalized continuous xyxy IoU, inclusive Acc@0.5, all-sample mIoU and
  parse_rate. Invalid model outputs have IoU 0 and remain in `n`.
- Separated data/GT errors from model errors: preflight aborts with
  dataset/split/id/reason. Synthetic full pre-occlusion GT is used unchanged.
- Added deterministic bracket/prose parsing, normalized and 1000-bin inputs,
  finite/order/range checks, and no clipping/reordering repair.
- Added p1, 384 single-view processing, greedy seed-42 inference, local model
  and data hashes, raw per-sample outputs, resumable details and result validator.
- Enabled the ScreenSpot dataset argument for a later separately leased run;
  this does not run or alter that dataset and does not change metric code.

### CPU-test-driven fixes (before B0; no experimental results affected)

1. **Parser diff**: added numeric token lookarounds so the `1` in `x1=` is not
   parsed as a coordinate, while `<loc_100>` remains parseable as a 1000-bin
   value. **Reason**: the labelled-coordinate regression test exposed eight
   extracted numbers instead of four. This is a correctness fix, not tuning to
   model scores.
2. **Metric implementation diff**: `metrics_from_scores` now counts invalid
   predictions from its materialized list rather than re-consuming the input
   iterable. **Reason**: one-shot iterators would otherwise undercount invalid.
3. **Test fixture diff**: the inclusive-threshold fixture uses binary-exact
   widths 0.75/0.25 rather than a decimal case whose floating ratio can be just
   below 0.5. **Reason**: test the mathematical threshold, not binary rounding.
   Production comparison remains exactly `IoU >= 0.5`, with no epsilon.
4. **Processor infrastructure diff**: compute and set Idefics3
   `processor.image_seq_len=(image_size/patch_size/scale_factor)^2`, hence 36
   rather than the default 64 at 384px, and record patch/scale/sequence length
   in the config fingerprint. **Reason**: the first pre-B0 CUDA attempt failed
   before producing a row; a CPU forward exposed fewer visual source vectors
   than image placeholders (`masked_scatter`). No-split pixels are 384x384 and
   yield 24x24 patches, then 6x6=36 after 4x4 pixel shuffle. No model weights,
   data, prompt text, GT, metric, seed, image size or decoding setting changed.
   The empty failed-attempt details and original GPU log are preserved; the
   authorized retry uses separate blocking/stack logs.

CPU evidence after fixes: 15/15 lightweight tests passed. The separate
`eval_cpu_forward_tests.json` records 2/2 real-image CPU forwards passed
(square `synthetic_eval_000001` and rectangular `synthetic_eval_000005`), each
with 36 placeholders and `[1,1,3,384,384]` pixels. Both 1000-row B0 dry-runs
passed image/GT/schema/hash preflight with no GPU model load. No completed B0
result exists, so no experimental metric was affected by these fixes.

## v1.1.0-preG3 · 2026-11-09

Read-only PEFT adapter support for SFT evaluation (G3). This is a benign,
additive change; the frozen parser, IoU, Acc@0.5, mIoU and parse_rate code
(`grounding.py`) is untouched, and all pre-existing metric denominators are
unchanged.

- **`run_zeroshot.py` diff**: added optional `--adapter <dir>`. Adapter
  preflight (`load_adapter_metadata`) requires `adapter_config.json` +
  `adapter_model.safetensors`, `peft_type=LORA`, `task_type=CAUSAL_LM`, and an
  `base_model_name_or_path` that resolves to the `--model-path`. It records
  adapter file SHA256/bytes, LoRA rank/alpha/dropout and PEFT version in the
  config (and therefore in the config fingerprint and summary `adapter` field).
  At load time the runner calls `PeftModel.from_pretrained(base,
  is_trainable=False, torch_device=device)`, then hard-asserts that at least one
  LoRA module attached, that **no parameter is trainable**, and that the model
  is in eval mode. `torch_device` is passed explicitly so a `--device cpu` run
  (CPU test) never creates a CUDA context.
- **`validate_results.py` diff**: added optional `--expected-adapter`
  (default `None` = require `adapter: null`, i.e. unchanged B0 behaviour). When
  an adapter is expected it also re-hashes the current `adapter_config.json` /
  `adapter_model.safetensors` and checks the recorded `config.adapter` metadata
  (`path`, `is_trainable=false`, both SHA256). No metric code changed.
- **New CPU tests**: `test_adapter_support.py` (13 tests: real adapter metadata,
  read-only hashing, missing/wrong-base/non-LoRA/non-CAUSAL_LM rejection, dry-run
  preflight with/without adapter, fingerprint change, invalid adapter CLI) and
  `test_adapter_forward_cpu.py` (2 tests: real CPU base+adapter load with
  `CUDA_VISIBLE_DEVICES=""`, 120 LoRA modules, 0 trainable parameters, 36 image
  placeholders, one-token greedy generation; `gpu_used=false`,
  `cuda_initialized=false`). Reports: `results/eval_adapter_tests.json`,
  `results/eval_adapter_cpu_forward.json`.
- **Affected run IDs**: none of the B0 results (`zeroshot_b0_synthetic_s42_v1`,
  `zeroshot_b0_refcoco_s42_v1`) are recomputed or overwritten; their JSON and
  validation reports remain the authoritative B0 evidence. Because
  `run_zeroshot.py` itself is part of `source_sha256` in each summary, re-running
  `validate_results` on those *old* B0 JSON files would now report recorded
  source drift by design. That is expected after a benign support addition and
  is why the old validation reports are preserved rather than regenerated. New
  SFT runs get new run IDs and therefore fresh source hashes.
- **Reason**: G3 requires independent evaluation of the T6 SFT adapter on the
  full synthetic-eval and RefCOCO-val sets with the frozen evaluator; there was
  no way to load an adapter read-only before this change.

## v1.2.0-preG3batch · 2026-11-09

Optional batched greedy evaluation for the remaining G3 seeds. This is an
equivalence-preserving speed change: the parser, IoU, Acc@0.5, mIoU, parse_rate
and all denominators (`grounding.py`, `score_prediction`,
`metrics_from_scores`) are untouched, and the historical one-sample path is
kept in its own branch.

- **`run_zeroshot.py` diff**: added `--batch-size` (default `1`; validation
  rejects `<1`). `batch_size == 1` executes the historical per-sample
  processor/generate/decode block. `batch_size > 1` uses:
  1. the same per-sample prompts (`build_messages` + `apply_chat_template`);
  2. the same per-sample images, processed together with
     `processor(text=prompts, images=[[image], ...], padding=True)` after
     setting `tokenizer.padding_side = "left"`;
  3. per-row prompt length from `attention_mask.sum(dim=1)`, while generation
     slicing uses the padded input length (left padding puts every real prompt
     at the right edge);
  4. `max_new_tokens=64`, `do_sample=False`, `num_beams=1` unchanged;
  5. each row's continuation trimmed at (and including) its first
     `generation_config.eos_token_id`, so a row that finishes early does not
     carry the generation pad tokens that are only needed to keep the batch
     alive. Rows without EOS keep the full 64-token continuation, preserving
     the no-truncation contract;
  6. decoding/score/detail writing shared with the batch-1 path through
     `emit_sample`, so detail fields and metrics are identical.
  `batch_size` is recorded in the config (and therefore the config
  fingerprint), the preflight, the summary, `summary.batch_contract`, and each
  detail row. Mixing batch sizes across a resume is impossible because the
  fingerprint covers `batch_size`.
- **`validate_results.py` diff**: none. `batch_size` is extra config metadata;
  the frozen config checks, per-row recomputation and adapter hash checks are
  unchanged. Batched runs are validated exactly like batch-1 runs.
- **New CPU tests**: `test_batch_eval_cpu.py` (14 tests) covers inclusive
  EOS trimming (early/middle/first/absent), multiple EOS ids, simulated
  left-padded variable-length extraction, left-padding and per-row prompt
  lengths, per-sample pixel tensor equality between single and batched
  processing, batch-of-one processor equality, dry-run `batch_size` recording
  and fingerprint change, rejection of `batch_size=0`, and a real CPU end-to-end
  batch-1 vs batch-2 run with byte-identical raw outputs (2/2, 33 tokens, no
  truncation, adapter frozen with 120 LoRA modules and 0 trainable parameters).
  Report: `results/eval_batch_cpu_tests.json` (`gpu_used=false`,
  `cuda_initialized=false`). CPU float32 equivalence validates the batching and
  slicing machinery only; it does not by itself prove GPU bf16 identity.
- **Mandatory GPU gate before use**: same adapter, first 20 synthetic-eval
  records, batch-1 vs batch-8, `raw_output` byte-identical 20/20. Any mismatch
  means permanent fallback to batch-1 and no mixing. Evidence/plan:
  `results/eval_batch_dryrun.json`.
- **GPU A/B outcome (2026-11-09 20:50)**: the gate **FAILED** — batch-8 matched
  batch-1 on only 5/20 `raw_output` (CPU float32 was 20/20, so the padding and
  slicing logic is correct; GPU bf16 batched kernels change the greedy argmax).
  `batch_size > 1` is therefore **prohibited for official evaluation**; every
  reported SFT number uses `batch_size=1`. Failure record:
  `results/eval_batch8_ab_failure.json`; evidence archive:
  `archive/t6eval_batch8_ab_failure_2026-11-09/`.
- **Affected run IDs**: none. Existing batch-1 B0 and SFT results are preserved
  and remain authoritative; batched re-runs (when licensed) use new run IDs and
  new validation reports.

## v1.3.0-preT7 · 2026-11-10

Coordinate-representation variants for the T7 G4 ablation. A (`p1`, normalized
decimal) is unchanged; B (`p1_bin1000`) reuses the frozen parser; C
(`p1_pixel`) gets an independent pixel-conversion layer before the frozen
metrics. `src/eval/grounding.py` (parse_box, IoU, Acc@0.5, mIoU, parse_rate and
all denominators) is untouched.

- **`prompts.py` diff**: added `PROMPT_ID_BIN1000="p1_bin1000"`,
  `PROMPT_ID_PIXEL="p1_pixel"`, their templates, per-variant SHA256, a
  `PROMPT_VARIANTS` registry (prompt id -> template/sha256/representation/
  conversion flag), and `build_messages_for(prompt_id, question)`. Prompt text
  is byte-identical to `src/train/coordinate_variants.py`; a CPU test asserts
  templates, ids, SHA256 and chat messages are equal to the training module.
- **`pixel_convert.py` (new)**: independent C conversion. It mirrors the
  training-side bracket grammar, requires finite integer pixel values with
  `0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`, normalizes by
  width/height, validates with the frozen `validate_box`, then computes
  `Score` with frozen `iou`/`ACC_THRESHOLD`. It deliberately never calls
  `parse_box` (which would misread pixels as 1000-bin values). Invalid outputs
  become `Score(None, 0.0, False, error)` and remain in `n`.
- **`run_zeroshot.py` diff**: added `--prompt-id {p1,p1_bin1000,p1_pixel}`
  (default `p1`, so historical behaviour is unchanged). The selected prompt
  template builds the chat messages; `config` records
  `prompt_id/prompt_sha256/prompt_template/representation/coordinate_conversion`
  (all inside the config fingerprint); summary records
  `prompt_id/representation/coordinate_contract`; detail rows record
  `prompt_id/representation`, and for C also `pixel_conversion` audit and
  `pixel_box`. `pixel_convert.py` was added to `source_sha256`. B uses the
  frozen `score_prediction` path unchanged.
- **`validate_results.py` diff**: added `--expected-prompt-id` (default `p1` =
  old behaviour). It checks the recorded prompt SHA256/template/representation
  against the eval-side registry, recomputes per-record scores with the
  representation-appropriate path (frozen for A/B, `score_prediction_pixel` for
  C), and for C also cross-checks the recorded `pixel_conversion` audit and
  `pixel_box`. The fixed image/decoding contract and all metric aggregation
  code are unchanged.
- **CPU evidence**: `test_t7_variants_cpu.py` 16/16 passed (prompt equality with
  training, B frozen-parser roundtrip, C roundtrip/bounds/reversed/
  non-integer/multi-box invalidity, invalid-in-n, dry-run plumbing). Two CPU
  end-to-end runs (20 rows each, adapter A) validated with the variant prompt
  ids. Dry-runs with adapter A for all three prompt ids pass; B/C adapter
  dry-runs are pending because those adapters do not exist yet.
  Reports: `results/eval_t7_variants_cpu_tests.json`,
  `results/eval_t7_cpu_dryrun.json`,
  `results/eval_t7_{bin1000,pixel}_cpu_e2e_validation.json`.
- **Affected run IDs**: no A/B0/T6 result is modified or recomputed; T7 runs use
  new run IDs (`t7_b_*`, `t7_c_*`). Because `run_zeroshot.py` changed, re-running
  the validator on *old* summaries would report recorded source drift by design
  (same policy as v1.1.0); old validation reports remain authoritative.
- **Reason**: T7 G4 requires an independent evaluator for the bin1000 and pixel
  answer representations without changing the frozen metric contract.

## Required post-B0 policy

Every subsequent evaluator change must record the concrete diff, reason and
affected run IDs here. Changes that can alter parsing, scores, preprocessing or
aggregation require rerunning all affected results under new run IDs; old JSON
is preserved. A change must never be justified by making a desired experimental
outcome look better. Benign support additions must explain why old scores are
unaffected.
