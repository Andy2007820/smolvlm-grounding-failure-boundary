"""Training pipeline for the SmolVLM-256M-Instruct grounding project.

T5 introduces the CPU-reproducible SFT implementation and its preflight tests.
No module in this package loads a CUDA model or starts training at import time;
``train_sft`` refuses CUDA execution unless an explicit GPU lease id is supplied.
"""
