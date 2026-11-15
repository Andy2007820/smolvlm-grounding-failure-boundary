"""Deployment package for T14 (merge / quantize / CPU latency) and T15 (demo).

Hard rules for every module in this package:

* **Import-time is CPU-light.**  Nothing here may import ``torch``,
  ``transformers``, ``onnxruntime`` or ``gradio`` at module import time, and
  nothing may allocate CUDA memory.  Heavy imports happen inside the
  ``load()`` / ``run()`` functions that the Lead's GPU/CPU lease authorises.
* **No fabricated deployment numbers.**  A route that has not been executed is
  reported as ``not_run``; only commands that actually ran may appear under
  ``artifacts``.
* **The evaluator is read-only.**  Box parsing / IoU / metrics are imported from
  :mod:`src.eval.grounding` so the deployment path cannot drift from the frozen
  official contract.  This package never edits ``src/eval``.
* **One shared prompt.**  The demo, the benchmark and the evaluator all call
  :func:`src.eval.prompts.build_messages` so the prompt id ``p1`` is identical
  everywhere.

The planned execution order is documented in ``src/deploy/README.md`` and
machine-readable in ``results/deploy_prep.json`` (produced by
``python -m src.deploy.check_env``).
"""
