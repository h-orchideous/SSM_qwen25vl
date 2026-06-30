#!/usr/bin/env bash
set -euo pipefail

# Unified VSC benchmark launcher for Qwen2.5-VL.
# MODEL_VARIANT=vl      -> qwen2_5_vl_vsc chunked-video baseline
# MODEL_VARIANT=sw      -> qwen_vsc_sliding_window
# MODEL_VARIANT=sw_ssm  -> qwen_vsc_sliding_window_ssm with SSM long-memory fusion

MODEL_VARIANT=${MODEL_VARIANT:-vl}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}
TASK_SUFFIX=${TASK_SUFFIX:-10mins}
TASK=${TASK:-qwen_vsc_streaming_local_${TASK_SUFFIX}}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export DECORD_EOF_RETRY_MAX=20480
export VSI_SUPER_COUNT_ROOT=${VSI_SUPER_COUNT_ROOT:-/data1/ZhangHuayu/datasets/VSI-SUPER-Count}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}
export USE_TF=${USE_TF:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=${PYTHONPATH:-$PROJECT_ROOT:$REPO_ROOT}

if [[ ":${PYTHONPATH}:" != *":${PROJECT_ROOT}:"* ]]; then
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi
if [[ ":${PYTHONPATH}:" != *":${REPO_ROOT}:"* ]]; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi

PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(command -v python)
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "[vsc_qwen25] ERROR: python executable not found"
    exit 2
fi

chunk_config_script=${CHUNK_CONFIG_SCRIPT:-$SCRIPT_DIR/vsc_chunk_settings.sh}
if [ -f "$chunk_config_script" ]; then
    # shellcheck source=/dev/null
    source "$chunk_config_script"
else
    echo "[vsc_qwen25] WARNING: chunk config script not found at $chunk_config_script; using inline fallback defaults."
fi

case "$MODEL_VARIANT" in
    vl|qwen25vl)
        model_family_default="qwen2_5_vl_vsc"
        log_root_default="logs/vsc/qwen/perf_qwen25vl"
        output_root_default="logs/vsc/qwen/output_qwen25vl"
        log_suffix_default="qwen25vl_vsc"
        main_process_port_default=29520
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=1
        ;;
    sw|qwen25_sw)
        model_family_default="qwen_vsc_sliding_window"
        log_root_default="logs/vsc/qwen/perf_sw"
        output_root_default="logs/vsc/qwen/output_sw"
        log_suffix_default="qwen_sw_vsc"
        main_process_port_default=29521
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=0
        ;;
    sw_ssm|qwen25_sw_ssm)
        model_family_default="qwen_vsc_sliding_window_ssm"
        log_root_default="logs/vsc/qwen/perf_sw_ssm"
        output_root_default="logs/vsc/qwen/output_sw_ssm"
        log_suffix_default="qwen_sw_ssm_vsc"
        main_process_port_default=29522
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=0
        ;;
    *)
        echo "[vsc_qwen25] ERROR: unsupported MODEL_VARIANT=$MODEL_VARIANT (expected: vl, sw, sw_ssm)"
        exit 2
        ;;
esac

LOG_ROOT=${LOG_ROOT:-$log_root_default}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$output_root_default}
if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$REPO_ROOT/$LOG_ROOT"
fi
if [[ "$EVAL_OUTPUT_ROOT" != /* ]]; then
    EVAL_OUTPUT_ROOT="$REPO_ROOT/$EVAL_OUTPUT_ROOT"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi
num_processes=${NUM_PROCESSES:-$num_processes}

checkpoint=${CHECKPOINT:-$checkpoint_default}
model_family=${MODEL_FAMILY:-$model_family_default}
log_suffix=${LOG_SUFFIX:-$log_suffix_default}
main_process_port=${MAIN_PROCESS_PORT:-$main_process_port_default}
force_simple=${FORCE_SIMPLE:-$force_simple_default}

# VL baseline: one model.generate() on a capped number of frames.
max_pixels=${MAX_PIXELS:-175616}
min_pixels=${MIN_PIXELS:-$max_pixels}
max_num_frames=${MAX_NUM_FRAMES:-150}
video_fps=${VIDEO_FPS:-1}
if [ "$MODEL_VARIANT" = "vl" ] || [ "$MODEL_VARIANT" = "qwen25vl" ]; then
    use_custom_video_loader=${USE_CUSTOM_VIDEO_LOADER:-False}
else
    use_custom_video_loader=${USE_CUSTOM_VIDEO_LOADER:-True}
fi
max_image_size=${MAX_IMAGE_SIZE:-448}
attn_impl=${ATTN_IMPLEMENTATION:-sdpa}
interleave_visuals=${INTERLEAVE_VISUALS:-False}
use_fast_processor=${USE_FAST_PROCESSOR:-True}

# SW / SW+SSM streaming params.
conv_template=${CONV_TEMPLATE:-qwen_2}
num_frames=${NUM_FRAMES:-}
miv_token_len=${MIV_TOKEN_LEN:-64}
si_token_len=${SI_TOKEN_LEN:-729}
sensory_window_size=${SENSORY_WINDOW_SIZE:-140}
sliding_window_stride=${SLIDING_WINDOW_STRIDE:-}
enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING:-True}
sensory_window_max_tokens=${SENSORY_WINDOW_MAX_TOKENS:-0}
stream_visual_micro_batch_size=${STREAM_VISUAL_MICRO_BATCH_SIZE:-1}
stream_query_mode=${STREAM_QUERY_MODE:-chunk}
ssm_d_state=${SSM_D_STATE:-64}
ssm_max_memory_len=${SSM_MAX_MEMORY_LEN:-256}
ssm_fusion_num_heads=${SSM_FUSION_NUM_HEADS:-8}
ssm_fusion_bottleneck=${SSM_FUSION_BOTTLENECK:-256}
ssm_layer_sharing=${SSM_LAYER_SHARING:-group4}

chunk_seconds=${CHUNK_SECONDS:-150}
chunk_overlap_seconds=${CHUNK_OVERLAP_SECONDS:-0}
chunk_max_num_frames=${CHUNK_MAX_NUM_FRAMES:-150}
chunk_encode_mode=${CHUNK_ENCODE_MODE:-reencode}
chunk_per_video_flow=${CHUNK_PER_VIDEO_FLOW:-1}
chunk_temp_mode=${CHUNK_TEMP_MODE:-1}
keep_chunks=${KEEP_CHUNKS:-0}
single_video_path=${SINGLE_VIDEO_PATH:-}

if [ -z "${num_frames}" ]; then
    if [ "$MODEL_VARIANT" = "sw" ] || [ "$MODEL_VARIANT" = "qwen25_sw" ] || [ "$MODEL_VARIANT" = "sw_ssm" ] || [ "$MODEL_VARIANT" = "qwen25_sw_ssm" ]; then
        num_frames=-1
        echo "[vsc_qwen25] INFO: NUM_FRAMES not set for sliding-window mode; defaulting to -1 (no per-chunk frame cap)."
    else
        num_frames=1
    fi
fi

if [ "$MODEL_VARIANT" = "vl" ] || [ "$MODEL_VARIANT" = "qwen25vl" ]; then
    model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},max_num_frames=${max_num_frames},fps=${video_fps},use_custom_video_loader=${use_custom_video_loader},max_image_size=${max_image_size},attn_implementation=${attn_impl},interleave_visuals=${interleave_visuals},use_fast_processor=${use_fast_processor}"
    if [ "$use_custom_video_loader" = "False" ] || [ "$use_custom_video_loader" = "false" ] || [ "$use_custom_video_loader" = "0" ]; then
        model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},max_num_frames=${max_num_frames},fps=${video_fps},use_custom_video_loader=${use_custom_video_loader},attn_implementation=${attn_impl},interleave_visuals=${interleave_visuals},use_fast_processor=${use_fast_processor}"
    fi
else
    model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},video_max_frames=${num_frames},video_fps=${video_fps},miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},sensory_window_max_tokens=${sensory_window_max_tokens},stream_visual_micro_batch_size=${stream_visual_micro_batch_size},stream_query_mode=${stream_query_mode},use_custom_video_loader=${use_custom_video_loader},max_image_size=${max_image_size},attn_implementation=${attn_impl},use_fast_processor=${use_fast_processor}"
    if [ -n "$sliding_window_stride" ]; then
        model_args_default+=",sliding_window_stride=${sliding_window_stride}"
    fi
    if [ "$MODEL_VARIANT" = "sw_ssm" ] || [ "$MODEL_VARIANT" = "qwen25_sw_ssm" ]; then
        model_args_default+=",ssm_d_state=${ssm_d_state},ssm_max_memory_len=${ssm_max_memory_len},ssm_fusion_num_heads=${ssm_fusion_num_heads},ssm_fusion_bottleneck=${ssm_fusion_bottleneck},ssm_layer_sharing=${ssm_layer_sharing}"
    fi
fi
model_args=${MODEL_ARGS:-$model_args_default}

check_environment() {
    MODEL_VARIANT="$MODEL_VARIANT" CHECKPOINT="$checkpoint" "$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import sys

variant = os.environ.get("MODEL_VARIANT", "vl")
if importlib.util.find_spec("lmms_eval") is None:
    print("[vsc_qwen25] ERROR: lmms_eval is not importable in current PYTHON_BIN")
    sys.exit(2)

import transformers
print(f"[vsc_qwen25] transformers_version={transformers.__version__}")
if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
    print("[vsc_qwen25] ERROR: current transformers lacks Qwen2_5_VLForConditionalGeneration")
    sys.exit(2)

if variant in {"sw_ssm", "qwen25_sw_ssm"} and importlib.util.find_spec("cambrian.ssm.ssm_compressor") is None:
    print("[vsc_qwen25] ERROR: project SSM module cambrian.ssm.ssm_compressor is not importable in current PYTHONPATH")
    print("[vsc_qwen25] NOTE: this is the SSM implementation package, not the Cambrian model path.")
    sys.exit(2)

print("[vsc_qwen25] environment check passed")
PY
}

check_environment

cd "$REPO_ROOT"
mkdir -p "$LOG_ROOT"
METRICS_CSV="$LOG_ROOT/metrics.csv"
if [ ! -f "$METRICS_CSV" ]; then
    echo "run_id,benchmark,seconds,samples,throughput_sps,sec_per_sample,output_path,log_file,gpu_log" > "$METRICS_CSV"
fi

TEMP_CHUNK_ROOT_TO_CLEAN=""
cleanup_temp_chunks() {
    if [ -z "${TEMP_CHUNK_ROOT_TO_CLEAN:-}" ]; then
        return
    fi
    if [ "$keep_chunks" = "1" ] || [ "$keep_chunks" = "true" ] || [ "$keep_chunks" = "True" ]; then
        echo "[vsc_qwen25] kept temporary chunks at $TEMP_CHUNK_ROOT_TO_CLEAN"
    else
        rm -rf "$TEMP_CHUNK_ROOT_TO_CLEAN"
        echo "[vsc_qwen25] cleaned temporary chunks at $TEMP_CHUNK_ROOT_TO_CLEAN"
    fi
}
trap cleanup_temp_chunks EXIT

run_eval() {
    local benchmark="$1"
    local run_id
    run_id=$(date +%Y%m%d_%H%M%S)
    local run_dir="$LOG_ROOT/$benchmark/$run_id"
    mkdir -p "$run_dir"

    local output_path="$EVAL_OUTPUT_ROOT/$benchmark/$run_id"
    local start_marker="$run_dir/.start"
    local gpu_log="$run_dir/gpu.csv"
    local eval_log="$run_dir/eval.log"
    local summary_file="$run_dir/summary.txt"
    local chunk_manifest=""
    local chunk_output_root=""
    local temp_chunk_root=""
    local chunk_cleanup_status="not_requested"

    date +%s > "$start_marker"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l "$GPU_LOG_INTERVAL" > "$gpu_log" &
    local gpu_pid=$!

    local start_ts
    start_ts=$(date +%s)

    echo "[vsc_qwen25] variant=$MODEL_VARIANT"
    echo "[vsc_qwen25] task=$benchmark"
    echo "[vsc_qwen25] repo_root=$REPO_ROOT"
    echo "[vsc_qwen25] dataset_root=$VSI_SUPER_COUNT_ROOT"
    echo "[vsc_qwen25] num_processes=$num_processes"
    echo "[vsc_qwen25] model_family=$model_family"
    echo "[vsc_qwen25] checkpoint=$checkpoint"
    echo "[vsc_qwen25] model_args=$model_args"

    if [ -n "$single_video_path" ]; then
        export VSC_SINGLE_VIDEO_PATH="$single_video_path"
        echo "[vsc_qwen25] single_video_path=$single_video_path"
    else
        unset VSC_SINGLE_VIDEO_PATH
    fi

    if [[ "$chunk_seconds" != "0" && "$chunk_seconds" != "0.0" ]]; then
        local task_yaml="$REPO_ROOT/lmms_eval/tasks/qwen_vsc_streaming_local/${benchmark}.yaml"
        local chunk_output_base="${VSC_CHUNK_OUTPUT_ROOT:-$REPO_ROOT/.cache/vsc_chunks}"
        if [ "$chunk_temp_mode" = "1" ] || [ "$chunk_temp_mode" = "true" ] || [ "$chunk_temp_mode" = "True" ]; then
            mkdir -p "$chunk_output_base"
            temp_chunk_root=$(mktemp -d "$chunk_output_base/${benchmark}_${run_id}_XXXXXX")
            chunk_output_root="$temp_chunk_root"
            chunk_cleanup_status="pending"
            TEMP_CHUNK_ROOT_TO_CLEAN="$temp_chunk_root"
            echo "[vsc_qwen25] prechunk_storage=temporary"
            echo "[vsc_qwen25] prechunk_temp_root=$temp_chunk_root"
        else
            chunk_output_root="$chunk_output_base"
            chunk_cleanup_status="disabled"
            TEMP_CHUNK_ROOT_TO_CLEAN=""
            echo "[vsc_qwen25] prechunk_storage=persistent"
            echo "[vsc_qwen25] prechunk_output_root=$chunk_output_root"
        fi

        export VSC_CHUNK_OUTPUT_ROOT="$chunk_output_root"
        if [ "$chunk_per_video_flow" = "1" ] || [ "$chunk_per_video_flow" = "true" ] || [ "$chunk_per_video_flow" = "True" ]; then
            export VSC_STREAMING_CHUNK_MODE=1
            unset VSC_PRECHUNK_MANIFEST
            chunk_manifest="per_video_streaming"
            echo "[vsc_qwen25] chunk_prepare_mode=per_video_streaming"
        else
            export VSC_STREAMING_CHUNK_MODE=0
            local prepare_chunk_cmd=(
                "$PYTHON_BIN" "$SCRIPT_DIR/prepare_vsc_chunks.py"
                --task-yaml "$task_yaml"
                --output-root "$chunk_output_root"
                --dataset-root "$VSI_SUPER_COUNT_ROOT"
                --chunk-seconds "$chunk_seconds"
                --chunk-overlap-seconds "$chunk_overlap_seconds"
                --chunk-max-num-frames "$chunk_max_num_frames"
                --chunk-encode-mode "$chunk_encode_mode"
            )
            if [ -n "$single_video_path" ]; then
                prepare_chunk_cmd+=(--only-video "$single_video_path")
            fi
            chunk_manifest=$("${prepare_chunk_cmd[@]}")
            export VSC_PRECHUNK_MANIFEST="$chunk_manifest"
            echo "[vsc_qwen25] prechunk_manifest=$chunk_manifest"
        fi
    else
        export VSC_STREAMING_CHUNK_MODE=0
        unset VSC_PRECHUNK_MANIFEST
        echo "[vsc_qwen25] chunk_prepare_mode=disabled"
    fi

    local force_simple_flag=""
    if [ "$force_simple" = "1" ] || [ "$force_simple" = "True" ] || [ "$force_simple" = "true" ]; then
        force_simple_flag="--force_simple"
    fi

    set +e
    "$PYTHON_BIN" -m accelerate.commands.launch --num_processes "${num_processes:-1}" --main_process_port "$main_process_port" -m lmms_eval \
        --model "$model_family" \
        $force_simple_flag \
        --model_args "$model_args" \
        --tasks "$benchmark" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "$log_suffix" \
        --output_path "$output_path" \
        2>&1 | tee "$eval_log"
    local eval_status=${PIPESTATUS[0]}
    set -e

    local end_ts
    end_ts=$(date +%s)
    local seconds=$((end_ts - start_ts))
    kill "$gpu_pid" >/dev/null 2>&1 || true

    local samples_file
    samples_file=$(find "$output_path" -name "*samples*.jsonl" -newer "$start_marker" 2>/dev/null | sort | tail -n 1 || true)

    local samples=0
    if [ -n "$samples_file" ]; then
        samples=$(wc -l < "$samples_file")
    fi

    local throughput="0"
    local sec_per_sample="0"
    if [ "$seconds" -gt 0 ] && [ "$samples" -gt 0 ]; then
        read -r throughput sec_per_sample <<EOF
$(python3 - <<PY
samples=$samples
seconds=$seconds
print(f"{samples/seconds:.6f} {seconds/samples:.6f}")
PY
)
EOF
    fi

    if [ -n "$temp_chunk_root" ]; then
        if [ "$keep_chunks" = "1" ] || [ "$keep_chunks" = "true" ] || [ "$keep_chunks" = "True" ]; then
            chunk_cleanup_status="kept"
            echo "[vsc_qwen25] kept temporary chunks at $temp_chunk_root"
        else
            rm -rf "$temp_chunk_root"
            chunk_cleanup_status="cleaned"
            echo "[vsc_qwen25] cleaned temporary chunks at $temp_chunk_root"
        fi
        TEMP_CHUNK_ROOT_TO_CLEAN=""
    fi

    {
        echo "benchmark: $benchmark"
        echo "run_id: $run_id"
        echo "seconds: $seconds"
        echo "samples: $samples"
        echo "throughput_sps: $throughput"
        echo "sec_per_sample: $sec_per_sample"
        echo "output_path: $output_path"
        echo "eval_log: $eval_log"
        echo "gpu_log: $gpu_log"
        echo "prechunk_manifest: ${chunk_manifest:-none}"
        echo "prechunk_output_root: ${chunk_output_root:-none}"
        echo "prechunk_cleanup: $chunk_cleanup_status"
        echo "exit_status: $eval_status"
    } > "$summary_file"

    echo "$run_id,$benchmark,$seconds,$samples,$throughput,$sec_per_sample,$output_path,$eval_log,$gpu_log" >> "$METRICS_CSV"
    echo "[vsc_qwen25] summary_file=$summary_file"
    echo "[vsc_qwen25] output_path=$output_path"
    echo "[vsc_qwen25] eval_log=$eval_log"
    echo "[vsc_qwen25] exit_status=$eval_status"

    return "$eval_status"
}

if ! run_eval "$TASK"; then
    echo "[vsc_qwen25] ERROR: evaluation failed. See the printed summary_file and eval_log above."
    exit 1
fi
