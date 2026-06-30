# SSM_qwen25vl

This repository is trimmed to the files needed by `lmms-eval/scripts/vsr_qwen25vl.sh`.

The launcher supports three VSR evaluation modes:

- `MODEL_VARIANT=vl`: native Qwen2.5-VL video evaluation.
- `MODEL_VARIANT=sw`: Qwen2.5-VL sliding-window video evaluation.
- `MODEL_VARIANT=sw_ssm`: Qwen2.5-VL sliding-window evaluation with SSM memory fusion.

## Kept Runtime Scope

- `lmms-eval/scripts/vsr_qwen25vl.sh`: main evaluation launcher.
- `lmms-eval/scripts/vsr_chunk_settings.sh`: chunk defaults sourced by the launcher.
- `lmms-eval/scripts/prepare_vsr_chunks.py`: optional pre-chunk helper when streaming chunk mode is disabled.
- `lmms-eval/lmms_eval/`: minimal lmms-eval runtime needed by this launcher.
- `lmms-eval/lmms_eval/tasks/cambrians_vsr_local/vsr_local_10mins.yaml`: task hard-coded by the launcher.
- `cambrian/ssm/`: SSM compressor and streaming cache used by `MODEL_VARIANT=sw_ssm`.

## Run

```bash
cd lmms-eval

MODEL_VARIANT=sw_ssm \
CHECKPOINT=/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct \
NUM_FRAMES=-1 \
VIDEO_FPS=1 \
STREAM_QUERY_MODE=chunk \
bash scripts/vsr_qwen25vl.sh
```

The default task is `vsr_local_10mins`. Other lmms-eval models, tasks, examples, docs, tools, and generated efficiency spreadsheets are intentionally not tracked.
