"""True ``batch=1 x gradient_accumulation=8`` training loop.

This helper is the single implementation used by ``train_sft.run_training``.  It
performs *true* gradient accumulation:

* exactly ``gradient_accumulation_steps`` separate micro-batch forward/backward
  passes (each collated as batch=1), never one batch-of-8 forward;
* each micro loss is divided by ``gradient_accumulation_steps`` before
  ``backward()``, so the accumulated gradient is the mean of the micro losses and
  is not shrunk a second time;
* ``clip_grad_norm_`` runs once, then ``optimizer.step()`` once, then
  ``scheduler.step()`` once, then ``optimizer.zero_grad()`` once per optimizer
  step;
* ``samples_seen``/``micro_batches_seen`` increment per micro-batch,
  ``optimizer_step`` increments per optimizer step.

The helper is deliberately model-agnostic (any module returning an object with a
``loss`` attribute) so the exact accumulation semantics are executable in a
CPU-only regression test with a tiny module, independently of the 256M model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class OptimizerStepRecord:
    """One optimizer-step observation, used for logging/checkpoints."""

    optimizer_step: int
    epoch: int
    samples_seen: int
    micro_batches_in_step: int
    micro_batch_sizes: tuple[int, ...]
    loss_mean_of_micro_losses: float
    learning_rate: float | None
    grad_norm: float
    partial_accumulation: bool


@dataclass
class AccumulationResult:
    """Counters that make the accumulation semantics auditable."""

    optimizer_steps: int = 0
    micro_batches_seen: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    clip_calls: int = 0
    optimizer_step_calls: int = 0
    scheduler_step_calls: int = 0
    zero_grad_calls: int = 0
    partial_final_step: bool = False
    stop_reason: str = "completed"
    loss_scale: float = 0.0
    gradient_accumulation_steps: int = 0
    micro_batch_size: int = 1
    records: list[OptimizerStepRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "micro_batches_seen": self.micro_batches_seen,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "clip_calls": self.clip_calls,
            "optimizer_step_calls": self.optimizer_step_calls,
            "scheduler_step_calls": self.scheduler_step_calls,
            "zero_grad_calls": self.zero_grad_calls,
            "partial_final_step": self.partial_final_step,
            "stop_reason": self.stop_reason,
            "loss_scale": self.loss_scale,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "micro_batch_size": self.micro_batch_size,
            "elapsed_seconds": self.elapsed_seconds,
            "step_records": [
                {
                    "optimizer_step": record.optimizer_step,
                    "epoch": record.epoch,
                    "samples_seen": record.samples_seen,
                    "micro_batches_in_step": record.micro_batches_in_step,
                    "micro_batch_sizes": list(record.micro_batch_sizes),
                    "loss_mean_of_micro_losses": record.loss_mean_of_micro_losses,
                    "learning_rate": record.learning_rate,
                    "grad_norm": record.grad_norm,
                    "partial_accumulation": record.partial_accumulation,
                }
                for record in self.records
            ],
        }


def _move_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("input_ids", "labels", "attention_mask"):
        value = batch.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    # Generic fallback for the tiny CPU regression module: any tensor-like value.
    for value in batch.values():
        if hasattr(value, "shape") and len(getattr(value, "shape")) >= 1:
            return int(value.shape[0])
    raise ValueError("cannot determine micro-batch size: batch has no tensor-like value")


def run_true_gradient_accumulation(
    *,
    micro_batches: Iterable[tuple[int, Mapping[str, Any]]],
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    device: str = "cpu",
    gradient_accumulation_steps: int = 8,
    expected_micro_batch_size: int = 1,
    max_grad_norm: float = 1.0,
    parameters: Sequence[Any] | None = None,
    initial_optimizer_step: int = 0,
    initial_samples_seen: int = 0,
    on_optimizer_step: Callable[[OptimizerStepRecord], None] | None = None,
    should_stop: Callable[[int], bool] | None = None,
    loss_scale: float | None = None,
) -> AccumulationResult:
    """Run true micro-batch accumulation over ``micro_batches``.

    ``micro_batches`` yields ``(epoch, batch)`` where each ``batch`` is already
    collated with exactly ``expected_micro_batch_size`` rows (the project uses 1).
    The loop performs one forward/backward per yielded micro-batch, and every
    ``gradient_accumulation_steps`` micro-batches performs exactly one
    clip/step/scheduler/zero sequence.
    """

    import torch

    if isinstance(gradient_accumulation_steps, bool) or gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps!r}")
    if expected_micro_batch_size <= 0:
        raise ValueError(f"expected_micro_batch_size must be positive, got {expected_micro_batch_size!r}")
    if loss_scale is None:
        loss_scale = 1.0 / float(gradient_accumulation_steps)
    if loss_scale <= 0 or loss_scale > 1.0:
        raise ValueError(f"loss_scale must be in (0, 1], got {loss_scale!r}")

    trainable_parameters = list(parameters) if parameters is not None else [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("no trainable parameters were provided")

    result = AccumulationResult(
        loss_scale=float(loss_scale),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        micro_batch_size=int(expected_micro_batch_size),
    )
    pending_losses: list[float] = []
    pending_batch_sizes: list[int] = []
    start = time.perf_counter()
    internal_steps = 0
    current_epoch = 0
    stop_requested = False

    def finish_optimizer_step(*, partial: bool) -> None:
        nonlocal internal_steps, current_epoch
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, float(max_grad_norm))
        result.clip_calls += 1
        optimizer.step()
        result.optimizer_step_calls += 1
        if scheduler is not None:
            scheduler.step()
            result.scheduler_step_calls += 1
        optimizer.zero_grad(set_to_none=True)
        result.zero_grad_calls += 1
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            learning_rate: float | None = float(scheduler.get_last_lr()[0])
        else:
            learning_rate = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
        internal_steps += 1
        result.optimizer_steps += 1
        record = OptimizerStepRecord(
            optimizer_step=initial_optimizer_step + internal_steps,
            epoch=current_epoch,
            samples_seen=initial_samples_seen + result.micro_batches_seen,
            micro_batches_in_step=len(pending_losses),
            micro_batch_sizes=tuple(pending_batch_sizes),
            loss_mean_of_micro_losses=sum(pending_losses) / len(pending_losses),
            learning_rate=learning_rate,
            grad_norm=float(grad_norm),
            partial_accumulation=partial,
        )
        result.records.append(record)
        pending_losses.clear()
        pending_batch_sizes.clear()
        if on_optimizer_step is not None:
            on_optimizer_step(record)

    for epoch, raw_batch in micro_batches:
        batch = _move_to_device(raw_batch, device)
        batch_size = _batch_size(batch)
        if batch_size != int(expected_micro_batch_size):
            raise AssertionError(
                "true accumulation requires one sample per forward: "
                f"expected batch={expected_micro_batch_size}, got {batch_size}"
            )
        current_epoch = int(epoch)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss at optimizer step {initial_optimizer_step + internal_steps + 1}")
        result.forward_calls += 1
        # Divide each micro loss by the accumulation factor; this is the only
        # gradient scaling and yields the mean of the micro losses after summing.
        scaled_loss = loss * result.loss_scale
        scaled_loss.backward()
        result.backward_calls += 1
        result.micro_batches_seen += 1
        pending_losses.append(float(loss.item()))
        pending_batch_sizes.append(batch_size)
        if len(pending_losses) == int(gradient_accumulation_steps):
            finish_optimizer_step(partial=False)
            if should_stop is not None and should_stop(initial_optimizer_step + internal_steps):
                result.stop_reason = "max_steps"
                stop_requested = True
                break

    if pending_losses and not stop_requested:
        # End-of-stream partial group (only when total micro-batches are not a
        # multiple of the accumulation factor). Losses keep the constant /8
        # denominator, which is recorded as partial_accumulation=True.
        finish_optimizer_step(partial=True)
        result.partial_final_step = True
        result.stop_reason = "stream_exhausted"
    elif not result.stop_reason or result.stop_reason == "completed":
        result.stop_reason = "completed"

    result.elapsed_seconds = time.perf_counter() - start
    return result
