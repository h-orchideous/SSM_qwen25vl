# SSM_qwen25vl

This repository is trimmed to the files needed for Qwen2.5-VL VSR evaluation with three long-video handling modes.

## Evaluation Modes

| Mode | lmms-eval model | Mechanism |
| ---- | --------------- | --------- |
| `vl` | `qwen2_5_vl` | Native Qwen2.5-VL video inference. Long videos are split into chunks before inference. |
| `sw` | `qwen_vsr_sliding_window` | SimpleStream-style frame streaming. The model keeps only the latest `SENSORY_WINDOW_SIZE` raw frames and queries once at the end. No KV cache is retained. |
| `ssm` | `qwen_vsr_sliding_window_ssm` | Temporal KV-SSM streaming. Each frame is prefetched through Qwen2.5-VL to produce decoder KV; KV outside the recent frame window is absorbed in time order into per-layer SSM hidden states. Final query uses the current KV window and SSM fusion. |

## Kept Runtime Scope

- `lmms-eval/scripts/vsr_qwen25vl.sh`: main VSR evaluation launcher.
- `lmms-eval/scripts/vsr_chunk_settings.sh`: shared chunk defaults sourced by the launcher.
- `lmms-eval/scripts/prepare_vsr_chunks.py`: optional pre-chunk helper.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window.py`: clean SimpleStream sliding-window baseline.
- `lmms-eval/lmms_eval/models/simple/qwen_vsr_sliding_window_ssm.py`: temporal KV-SSM evaluation path.
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

## Training Module

The training path supports Qwen2.5-VL raw-frame sliding-window SFT. It streams video frames in time order, keeps only the most recent `STREAM_FRAME_WINDOW` frames, drops older raw frames, and computes loss only on assistant answer tokens.

Tracked training files:

- `cambrian/model/language_model/qwen2_5_ssm.py`: Qwen2.5-VL wrapper with raw-window training forward and optional SSM/SW paths.
- `cambrian/scripts/train_qwen25vl_ssm.py`: JSON/JSONL SFT trainer and frame-streaming data collator.
- `cambrian/scripts/train_qwen25vl_ssm.sh`: default SW raw-window training launcher.
- `cambrian/scripts/merge_fsdp_qwen25vl_checkpoint.py`: converts Accelerate FSDP `.distcp` checkpoints into a loadable HuggingFace checkpoint.

Default SW training settings in `train_qwen25vl_ssm.sh`:

```bash
TRAIN_STAGE=sw
SSM_SLIDING_WINDOW=0
SW_RAW_FRAME_WINDOW=100
STREAM_FRAME_WINDOW=100
TRAINABLE_MODULES=model,lm_head
GRADIENT_CHECKPOINTING=true
```

Example training command:

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTHON_BIN=/home/ZhangHuayu/miniconda3/envs/cambrians_train/bin/python \
DATA_PATH=/data1/ZhangHuayu/datasets/VSI-Train-10k/vsi_train_10k.jsonl \
VIDEO_ROOT=/data1/ZhangHuayu/datasets/VSI-Train-10k \
OUTPUT_DIR=/home/ZhangHuayu/Workspace/cambrian-s/checkpoints/qwen25vl_sw_raw_window_100_lr5e-6_pix100352 \
FPS=1 \
MIN_PIXELS=100352 \
MAX_PIXELS=100352 \
LEARNING_RATE=5e-6 \
LOGGING_STEPS=1 \
bash cambrian/scripts/train_qwen25vl_ssm.sh
```

Merge the final FSDP checkpoint for evaluation:

```bash
PYTHONNOUSERSITE=1 /home/ZhangHuayu/miniconda3/envs/cambrians_train/bin/python \
  cambrian/scripts/merge_fsdp_qwen25vl_checkpoint.py \
  --checkpoint_dir /home/ZhangHuayu/Workspace/cambrian-s/checkpoints/qwen25vl_sw_raw_window_100_lr5e-6_pix100352/checkpoint-312 \
  --output_dir /home/ZhangHuayu/Workspace/cambrian-s/checkpoints/qwen25vl_sw_raw_window_100_lr5e-6_pix100352_hf \
  --base_model_dir /data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct
```

Current merged training checkpoint for evaluation:

```text
/data1/ZhangHuayu/models/qwen25vl_sw_raw_window_100_lr5e-6_pix100352_hf
```

## Current VSR Efficiency Records

The following records come from:

```text
/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval/scripts/logs/vsr_eval_runs
```

Only complete runs with `samples=60` and `exit_status=0` are listed. Incomplete `samples=0` runs are ignored.

| Mode | run_id | Samples | Total time(s) | sec/sample | Throughput(samples/s) | Peak single-GPU memory(GiB) | Avg per-GPU peak memory(GiB) | 8-GPU peak sum(GiB) | Avg GPU util(%) |
| ---- | ------ | ------- | ------------- | ---------- | --------------------- | --------------------------- | ---------------------------- | ------------------- | --------------- |
| `vl` | `20260714_144732` | 60 | 2061 | 34.350 | 0.029112 | 41.868 | 30.800 | 246.403 | 60.01 |
| `sw` | `20260714_172805` | 60 | 1118 | 18.633 | 0.053667 | 43.778 | 26.217 | 209.736 | 49.61 |
| `ssm` | `20260714_185813` | 60 | 1775 | 29.583 | 0.033803 | 39.513 | 28.223 | 225.785 | 33.75 |

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
