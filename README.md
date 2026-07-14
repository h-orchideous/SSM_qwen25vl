# SSM_qwen25vl

This repository is trimmed to the files needed for Qwen2.5-VL VSR evaluation with three long-video handling modes.

## Evaluation Modes

| Mode | lmms-eval model | Mechanism |
| ---- | --------------- | --------- |
| `vl` | `qwen2_5_vl` | Native Qwen2.5-VL video inference. Long videos are split into chunks before inference. |
| `sw` | `qwen_vsr_sliding_window` | SimpleStream-style frame streaming. The model keeps only the latest `SENSORY_WINDOW_SIZE` raw frames and queries once at the end. No KV cache is retained. |
| `ssm` | `qwen_vsr_sliding_window_ssm` | SimpleStream sliding window plus V-Mamba-style 4-direction CSMS SSM. Frames evicted from the window are encoded as 2D visual hidden tokens and absorbed into SSM memory. Final query uses the current window and SSM fusion. |

## Kept Runtime Scope

- `lmms-eval/scripts/vsr_qwen25vl.sh`: main VSR evaluation launcher.
- `lmms-eval/scripts/vsr_chunk_settings.sh`: shared chunk defaults sourced by the launcher.
- `lmms-eval/scripts/prepare_vsr_chunks.py`: optional pre-chunk helper.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window.py`: clean SimpleStream sliding-window baseline.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window_ssm.py`: SimpleStream + CSMS SSM evaluation path.
- `lmms-eval/lmms_eval/models/model_utils/qwen_ssm_patch.py`: SSM fusion patch and 4-direction hidden-latent compressor.
- `lmms-eval/lmms_eval/models/model_utils/qwen_chunk_utils.py`: robust video chunk materialization.
- `lmms-eval/lmms_eval/tasks/cambrians_vsr_local/vsr_local_10mins.yaml`: default local VSR task.
- `cambrian/ssm/`: selective SSM and CUDA selective-scan integration used by `ssm`.

Other upstream lmms-eval tasks, examples, docs, tools, generated spreadsheets, and unrelated model files are intentionally outside the clean VSR evaluation scope.

## Run

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval

MODEL_VARIANT=vl bash scripts/vsr_qwen25vl.sh
MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
MODEL_VARIANT=ssm bash scripts/vsr_qwen25vl.sh
```

The default task is `vsr_local_10mins`. The default checkpoint is:

```text
/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct
```

Use explicit GPUs when the shared machine is busy:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
```

## Chunk Settings

The launcher sources:

```bash
lmms-eval/scripts/vsr_chunk_settings.sh
```

Current defaults:

```bash
CHUNK_SECONDS=200
CHUNK_OVERLAP_SECONDS=0
CHUNK_MAX_NUM_FRAMES=200
CHUNK_ENCODE_MODE=reencode
CHUNK_PER_VIDEO_FLOW=1
CHUNK_TEMP_MODE=1
KEEP_CHUNKS=0
```

Environment variables passed on the command line override these defaults.

## Current VSR Efficiency Records

The following records come from:

```text
/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval/scripts/logs/vsr_eval_runs
```

Only complete runs with `samples=60` and `exit_status=0` are listed. Incomplete `samples=0` runs are ignored.

| Mode | run_id | Samples | Total time(s) | sec/sample | Throughput(samples/s) | Peak single-GPU memory(GiB) | Avg per-GPU peak memory(GiB) | 8-GPU peak sum(GiB) | Avg GPU util(%) |
| ---- | ------ | ------- | ------------- | ---------- | --------------------- | --------------------------- | ---------------------------- | ------------------- | --------------- |
| `vl` | `20260714_144732` | 60 | 2061 | 34.350 | 0.029112 | 41.868 | 30.800 | 246.403 | 60.01 |
| `sw` | `20260714_152358` | 60 | 563 | 9.383 | 0.106572 | 35.069 | 25.319 | 202.556 | 56.47 |

`ssm` is not listed here because there is no complete `samples=60` run in the current `scripts/logs/vsr_eval_runs` directory after the CSMS SSM rewrite.

Metric meanings:

- `Total time(s)`: wall-clock runtime from `summary.txt`.
- `sec/sample`: `Total time / Samples`.
- `Throughput(samples/s)`: `Samples / Total time`.
- `Peak single-GPU memory(GiB)`: maximum observed `memory.used` across all GPUs.
- `Avg per-GPU peak memory(GiB)`: average of each GPU's own peak memory.
- `8-GPU peak sum(GiB)`: sum of per-GPU peak memory values.
- `Avg GPU util(%)`: average `utilization.gpu` from `gpu.csv`.

## Detailed Documentation

See:

```text
lmms-eval/docs/vsr_qwen25vl.md
```
