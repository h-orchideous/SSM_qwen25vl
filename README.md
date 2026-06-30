# SSM_qwen25vl

This repository keeps the evaluation-related code for running VSR experiments with Qwen2.5-VL under three modes:

- `vl`: standard Qwen2.5-VL video evaluation.
- `sw`: sliding-window / streaming-frame evaluation.
- `sw_ssm`: sliding-window evaluation with an SSM memory mechanism attached to the native Qwen2.5-VL model.

The repository is intentionally trimmed to evaluation code. Cambrian training code and Cambrian model definitions are not included. The remaining `cambrian/ssm/` package is kept because the Qwen2.5-VL SSM evaluation path imports its SSM compressor and streaming cache utilities.

## Main Files

- `lmms-eval/scripts/vsr_qwen25vl.sh`: unified launch script for VSR evaluation.
- `lmms-eval/docs/vsr_qwen25vl.md`: detailed usage notes for `vl`, `sw`, and `sw_ssm`.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window.py`: Qwen2.5-VL sliding-window evaluation model.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window_ssm.py`: Qwen2.5-VL sliding-window + SSM evaluation model.
- `lmms-eval/lmms_eval/models/model_utils/qwen_ssm_patch.py`: runtime patch for injecting SSM memory into Qwen2.5-VL layers.
- `cambrian/ssm/`: SSM memory modules reused by `sw_ssm`.

## Quick Start

Run from the `lmms-eval` directory:

```bash
cd lmms-eval

MODEL_VARIANT=sw_ssm \
CHECKPOINT=Qwen/Qwen2.5-VL-7B-Instruct \
NUM_FRAMES=150 \
VIDEO_FPS=1.0 \
STREAM_QUERY_MODE=chunk \
bash scripts/vsr_qwen25vl.sh
```

`STREAM_QUERY_MODE=chunk` streams all frames in each chunk into KV/SSM memory and only runs the text query once at the chunk end. This keeps `sw_ssm` aligned with the `sw` evaluation flow while adding SSM history readback.

For more configuration options, see `lmms-eval/docs/vsr_qwen25vl.md`.

## Git Scope

This repo should only track files needed for evaluation and the Qwen2.5-VL SSM mechanism. Avoid committing unrelated model training code, generated results, caches, checkpoints, or local environment files.
