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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

MODEL_PATH="${MODEL_PATH:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a JSON/JSONL SFT file}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen25vl_ssm_sw}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_STAGE="${TRAIN_STAGE:-sw}"
SW_RAW_FRAME_WINDOW="${SW_RAW_FRAME_WINDOW:-100}"
STREAM_FRAME_WINDOW="${STREAM_FRAME_WINDOW:-${SW_RAW_FRAME_WINDOW}}"
SSM_FRAME_WINDOW="${SSM_FRAME_WINDOW:-${STREAM_FRAME_WINDOW}}"
if [[ -z "${TRAIN_SSM_ONLY:-}" ]]; then
  if [[ "${TRAIN_STAGE}" == "sw" ]]; then
    TRAIN_SSM_ONLY=false
  else
    TRAIN_SSM_ONLY=true
  fi
fi
if [[ -z "${TRAINABLE_MODULES:-}" && "${TRAIN_STAGE}" == "sw" ]]; then
  TRAINABLE_MODULES=model,lm_head
fi
if [[ -z "${SSM_SLIDING_WINDOW:-}" ]]; then
  if [[ "${TRAIN_STAGE}" == "sw" ]]; then
    SSM_SLIDING_WINDOW=0
  else
    SSM_SLIDING_WINDOW=128
  fi
fi
if [[ -z "${GRADIENT_CHECKPOINTING:-}" ]]; then
  if [[ "${TRAIN_STAGE}" == "sw" && "${SSM_SLIDING_WINDOW}" == "0" ]]; then
    GRADIENT_CHECKPOINTING=true
  else
    GRADIENT_CHECKPOINTING=false
  fi
fi

check_visible_gpu_processes() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "${visible}" ]]; then
    return 0
  fi

  local gpu_csv
  gpu_csv="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${gpu_csv}" ]]; then
    return 0
  fi

  local gpu_map
  gpu_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${gpu_map}" ]]; then
    return 0
  fi

  local current_user
  current_user="$(id -un)"
  local found=0
  local killable_pids=()

  IFS=',' read -ra visible_indices <<< "${visible}"
  for raw_idx in "${visible_indices[@]}"; do
    local gpu_idx
    gpu_idx="$(echo "${raw_idx}" | xargs)"
    [[ -z "${gpu_idx}" ]] && continue

    local gpu_uuid
    gpu_uuid="$(awk -F, -v idx="${gpu_idx}" '$1 ~ "^[[:space:]]*"idx"[[:space:]]*$" {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "${gpu_map}")"
    [[ -z "${gpu_uuid}" ]] && continue

    while IFS=',' read -r app_uuid app_pid app_name app_mem; do
      app_uuid="$(echo "${app_uuid}" | xargs)"
      app_pid="$(echo "${app_pid}" | xargs)"
      app_name="$(echo "${app_name}" | xargs)"
      app_mem="$(echo "${app_mem}" | xargs)"
      [[ "${app_uuid}" != "${gpu_uuid}" || -z "${app_pid}" ]] && continue

      local owner
      owner="$(ps -o user= -p "${app_pid}" 2>/dev/null | xargs || true)"
      [[ -z "${owner}" ]] && owner="unknown"
      found=1
      echo "[train_qwen25vl_ssm] GPU ${gpu_idx} is occupied: pid=${app_pid} user=${owner} mem=${app_mem}MiB cmd=${app_name}" >&2
      if [[ "${owner}" == "${current_user}" ]]; then
        killable_pids+=("${app_pid}")
      fi
    done <<< "${gpu_csv}"
  done

  if [[ "${found}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${KILL_OWN_GPU_PROCS:-0}" == "1" && "${#killable_pids[@]}" -gt 0 ]]; then
    echo "[train_qwen25vl_ssm] KILL_OWN_GPU_PROCS=1, killing current-user GPU processes: ${killable_pids[*]}" >&2
    kill "${killable_pids[@]}" 2>/dev/null || true
    sleep 3
    return 0
  fi

  echo "[train_qwen25vl_ssm] ERROR: requested CUDA_VISIBLE_DEVICES=${visible} has running processes." >&2
  echo "[train_qwen25vl_ssm] Clear them first, or set KILL_OWN_GPU_PROCS=1 to kill only your own processes." >&2
  exit 3
}

check_visible_gpu_processes

echo "[train_qwen25vl_ssm] train_stage=${TRAIN_STAGE}"
echo "[train_qwen25vl_ssm] train_ssm_only=${TRAIN_SSM_ONLY}"
echo "[train_qwen25vl_ssm] trainable_modules=${TRAINABLE_MODULES:-all}"
echo "[train_qwen25vl_ssm] ssm_sliding_window=${SSM_SLIDING_WINDOW}"
echo "[train_qwen25vl_ssm] gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "[train_qwen25vl_ssm] stream_video_as_images=${STREAM_VIDEO_AS_IMAGES:-true}"
echo "[train_qwen25vl_ssm] stream_frame_window=${STREAM_FRAME_WINDOW:-${SSM_FRAME_WINDOW:-unset}}"
echo "[train_qwen25vl_ssm] fps=${FPS:-unset}"

visible_gpu_count() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "${visible}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | xargs
    else
      echo 1
    fi
    return
  fi

  local count=0
  IFS=',' read -ra visible_indices <<< "${visible}"
  for raw_idx in "${visible_indices[@]}"; do
    local gpu_idx
    gpu_idx="$(echo "${raw_idx}" | xargs)"
    [[ -n "${gpu_idx}" ]] && count=$((count + 1))
  done
  echo "${count}"
}

find_free_port() {
  "${PYTHON_BIN}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

extra_args=()
if [[ -n "${MIN_PIXELS:-}" ]]; then
  extra_args+=(--min_pixels "${MIN_PIXELS}")
fi
if [[ -n "${MAX_PIXELS:-}" ]]; then
  extra_args+=(--max_pixels "${MAX_PIXELS}")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  extra_args+=(--max_frames "${MAX_FRAMES}")
fi
if [[ -n "${STREAM_FRAME_WINDOW:-${SSM_FRAME_WINDOW:-}}" ]]; then
  extra_args+=(--stream_frame_window "${STREAM_FRAME_WINDOW:-${SSM_FRAME_WINDOW}}")
fi
if [[ -n "${FPS:-}" ]]; then
  extra_args+=(--fps "${FPS}")
fi

train_args=(
  cambrian/scripts/train_qwen25vl_ssm.py
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
  --bf16 "${BF16:-false}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --gradient_checkpointing_kwargs "${GRADIENT_CHECKPOINTING_KWARGS:-{\"use_reentrant\":false}}" \
  --remove_unused_columns false \
  --report_to "${REPORT_TO:-none}" \
  --train_stage "${TRAIN_STAGE}" \
  --train_ssm_only "${TRAIN_SSM_ONLY}" \
  --trainable_modules "${TRAINABLE_MODULES:-}" \
  --ssm_sliding_window "${SSM_SLIDING_WINDOW}" \
  --ssm_frame_window "${SSM_FRAME_WINDOW}" \
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
)

NUM_PROCESSES="${NUM_PROCESSES:-$(visible_gpu_count)}"
LAUNCHER="${LAUNCHER:-auto}"
if [[ "${LAUNCHER}" == "auto" ]]; then
  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    LAUNCHER=accelerate
  else
    LAUNCHER=python
  fi
fi

echo "[train_qwen25vl_ssm] launcher=${LAUNCHER}"
echo "[train_qwen25vl_ssm] num_processes=${NUM_PROCESSES}"

if [[ "${LAUNCHER}" == "accelerate" ]]; then
  ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}"
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    ACCELERATE_BIN="accelerate"
  fi
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$(find_free_port)}"
  echo "[train_qwen25vl_ssm] main_process_port=${MAIN_PROCESS_PORT}"
  "${ACCELERATE_BIN}" launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES:-1}" \
    --mixed_precision "${MIXED_PRECISION:-no}" \
    --dynamo_backend "${DYNAMO_BACKEND:-no}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLDecoderLayer \
    --fsdp_sharding_strategy "${FSDP_SHARDING_STRATEGY:-FULL_SHARD}" \
    --fsdp_state_dict_type "${FSDP_STATE_DICT_TYPE:-SHARDED_STATE_DICT}" \
    --fsdp_use_orig_params "${FSDP_USE_ORIG_PARAMS:-true}" \
    "${train_args[@]}"
elif [[ "${LAUNCHER}" == "python" ]]; then
  "${PYTHON_BIN}" "${train_args[@]}"
else
  echo "[train_qwen25vl_ssm] ERROR: unsupported LAUNCHER=${LAUNCHER}; use auto, accelerate, or python." >&2
  exit 4
fi
