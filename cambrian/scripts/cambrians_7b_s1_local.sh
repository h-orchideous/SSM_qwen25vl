#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Fixed paths (set to your machine locations)
DATA_PATH="/data1/ZhangHuayu/datasets/VSI-Train-10k/vsi_train_10k.jsonl"
IMAGE_FOLDER="/data1/ZhangHuayu/datasets/VSI-Train-10k"
MODEL_NAME_OR_PATH="/data1/ZhangHuayu/models/Qwen2.5-7B-Instruct"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] DATA_PATH file not found: $DATA_PATH"
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "[ERROR] IMAGE_FOLDER directory not found: $IMAGE_FOLDER"
    exit 1
fi

# Local-friendly defaults (all can be overridden by env vars)
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/cambrians_7b_s1}"
RUN_NAME="${RUN_NAME:-cambrians_7b_s1_local}"
REPORT_TO="${REPORT_TO:-none}"
LAUNCHER="${LAUNCHER:-fsdp}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
BF16="${BF16:-False}"
USE_FSDP="${USE_FSDP:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SSM_NUM_KV_HEADS="${SSM_NUM_KV_HEADS:-}"
SSM_HEAD_DIM="${SSM_HEAD_DIM:-}"
SSM_HIDDEN_DIM="${SSM_HIDDEN_DIM:-}"
SSM_D_STATE="${SSM_D_STATE:-64}"
SSM_MAX_MEMORY_LEN="${SSM_MAX_MEMORY_LEN:-}"
SSM_FUSION_NUM_HEADS="${SSM_FUSION_NUM_HEADS:-8}"
SSM_FUSION_BOTTLENECK="${SSM_FUSION_BOTTLENECK:-23328}"
SSM_LAYER_SHARING="${SSM_LAYER_SHARING:-group4}"
SSM_USE_FAST_PATH="${SSM_USE_FAST_PATH:-1}"
LOAD_WEIGHTS="${LOAD_WEIGHTS:-}"
VIDEO_FOLDER="${VIDEO_FOLDER:-}"
VIDEO_FPS="${VIDEO_FPS:-1}"
# Recommended: 2-4 frames per step; default to 4.
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-4}"
VIDEO_FORCE_SAMPLE="${VIDEO_FORCE_SAMPLE:-True}"
# Optional: auto-calc SSM size from desired frames. If DESIRED_SSM_FRAMES is set
# or SSM_AUTO_CALC=1, we compute SSM_MAX_MEMORY_LEN = DESIRED_SSM_FRAMES * SI_TOKEN_LEN
SSM_AUTO_CALC="${SSM_AUTO_CALC:-1}"
DESIRED_SSM_FRAMES="${DESIRED_SSM_FRAMES:-64}"

# Per-frame token length (matches --si_token_len passed to the training script)
SI_TOKEN_LEN="${SI_TOKEN_LEN:-729}"

# Support lowercase env vars commonly used in s4 scripts
per_device_train_batch_size="${per_device_train_batch_size:-${PER_DEVICE_TRAIN_BATCH_SIZE:-8}}"
dataloader_num_workers="${dataloader_num_workers:-${DATALOADER_NUM_WORKERS:-4}}"
max_seq_len="${max_seq_len:-8192}"
report_to="${report_to:-${REPORT_TO:-wandb}}"
load_weights_env="${load_weights:-${LOAD_WEIGHTS:-}}"
# Use GPUs 0,1,2,3 by default
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# Resolve python executable for environments where `python` is unavailable.
PYTHON_BIN="${PYTHON_BIN:-}"
# If running inside a conda environment, prefer that python executable
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "[ERROR] Neither 'python' nor 'python3' found in PATH."
        exit 1
    fi
fi

# If requested, auto-calc SSM_MAX_MEMORY_LEN from desired frames
if [[ "${SSM_AUTO_CALC}" == "1" && -n "${DESIRED_SSM_FRAMES}" ]]; then
    # compute ssm size = frames * tokens_per_frame
    calc=$(( DESIRED_SSM_FRAMES * SI_TOKEN_LEN ))
    echo "[INFO] Auto-calculated SSM_MAX_MEMORY_LEN=${calc} (frames=${DESIRED_SSM_FRAMES} * si_token_len=${SI_TOKEN_LEN})"
    SSM_MAX_MEMORY_LEN="${calc}"
fi

if [[ -z "${SSM_MAX_MEMORY_LEN}" ]]; then
    SSM_MAX_MEMORY_LEN="23328"
    echo "[INFO] SSM_MAX_MEMORY_LEN not set and auto-calc not applied; fallback to ${SSM_MAX_MEMORY_LEN}."
fi

mkdir -p "$OUTPUT_DIR"

# training args
TRAIN_ARGS="
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version qwen_2 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower_aux_list [\"google/siglip2-so400m-patch14-384\"] \
    --vision_tower_aux_token_len_list [729] \
    --image_position 14 \
    --vision_hidden_size 1152 \
    --video_folder ${VIDEO_FOLDER:-} \
    --video_fps ${VIDEO_FPS:-1} \
    --video_max_frames $VIDEO_MAX_FRAMES \
    --video_force_sample ${VIDEO_FORCE_SAMPLE:-True} \
    --connector_only True \
    --unfreeze_mm_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --mm_projector_lr 1e-3 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_im_newline_token True \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 $BF16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size:-$PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $max_seq_len \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers:-$DATALOADER_NUM_WORKERS} \
    --lazy_preprocess True \
    --report_to ${report_to:-$REPORT_TO} \
    --run_name $RUN_NAME \
    --max_images_per_sample 128 \
    --anyres_max_subimages 9 \
    --si_token_len 729 \
    --miv_token_len 64 \
    --ssm_num_kv_heads ${SSM_NUM_KV_HEADS:-} \
    --ssm_head_dim ${SSM_HEAD_DIM:-} \
    --ssm_hidden_dim ${SSM_HIDDEN_DIM:-} \
    --ssm_d_state ${SSM_D_STATE:-64} \
    --ssm_max_memory_len $SSM_MAX_MEMORY_LEN \
    --ssm_fusion_num_heads ${SSM_FUSION_NUM_HEADS:-8} \
    --ssm_fusion_bottleneck $SSM_FUSION_BOTTLENECK \
    --ssm_layer_sharing ${SSM_LAYER_SHARING:-group4} \
    --ssm_use_fast_path ${SSM_USE_FAST_PATH:-1} \
    --load_weights ${load_weights_env:-$LOAD_WEIGHTS} \
"

if [[ "$USE_FSDP" == "1" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS \
        --fsdp full_shard \
        --fsdp_config fsdp_config.json \
    "
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS \
        --train_continue True \
        --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    "
fi

echo "Training arguments:"
echo "$TRAIN_ARGS"

echo "[INFO] Using GPUs: ${GPU_IDS}"
echo "[INFO] Using Python: ${PYTHON_BIN}"
echo "[INFO] Launcher: ${LAUNCHER} -> invoking train_${LAUNCHER}.py"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" cambrian/train/train_${LAUNCHER}.py $TRAIN_ARGS
