#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODEL_PATH="${MODEL_PATH:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a JSON/JSONL SFT file}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen25vl_ssm_sw}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_STAGE="${TRAIN_STAGE:-ssm}"
if [[ -z "${TRAIN_SSM_ONLY:-}" ]]; then
  if [[ "${TRAIN_STAGE}" == "sw" ]]; then
    TRAIN_SSM_ONLY=false
  else
    TRAIN_SSM_ONLY=true
  fi
fi

extra_args=()
if [[ -n "${MIN_PIXELS:-}" ]]; then
  extra_args+=(--min_pixels "${MIN_PIXELS}")
fi
if [[ -n "${MAX_PIXELS:-}" ]]; then
  extra_args+=(--max_pixels "${MAX_PIXELS}")
fi

"${PYTHON_BIN}" cambrian/scripts/train_qwen25vl_ssm.py \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --image_root "${IMAGE_ROOT:-}" \
  --video_root "${VIDEO_ROOT:-}" \
  --mask_prompt_labels "${MASK_PROMPT_LABELS:-true}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --bf16 "${BF16:-true}" \
  --gradient_checkpointing false \
  --remove_unused_columns false \
  --report_to "${REPORT_TO:-none}" \
  --train_stage "${TRAIN_STAGE}" \
  --train_ssm_only "${TRAIN_SSM_ONLY}" \
  --trainable_modules "${TRAINABLE_MODULES:-}" \
  --ssm_sliding_window "${SSM_SLIDING_WINDOW:-128}" \
  --ssm_frame_window "${SSM_FRAME_WINDOW:-${SENSORY_WINDOW_SIZE:-128}}" \
  --ssm_training_step_size "${SSM_TRAINING_STEP_SIZE:-${SSM_TRAINING_CHUNK_SIZE:-128}}" \
  --ssm_prefix_len "${SSM_PREFIX_LEN:-0}" \
  --ssm_d_state "${SSM_D_STATE:-64}" \
  --ssm_max_memory_len "${SSM_MAX_MEMORY_LEN:-256}" \
  --ssm_fusion_num_heads "${SSM_FUSION_NUM_HEADS:-8}" \
  --ssm_fusion_bottleneck "${SSM_FUSION_BOTTLENECK:-256}" \
  --ssm_layer_sharing "${SSM_LAYER_SHARING:-group4}" \
  --ssm_visual_encode_chunk_size "${SSM_VISUAL_ENCODE_CHUNK_SIZE:-1}" \
  --stream_video_as_images "${STREAM_VIDEO_AS_IMAGES:-true}" \
  --stream_frame_stride "${STREAM_FRAME_STRIDE:-1}" \
  "${extra_args[@]}" \
  "$@"
