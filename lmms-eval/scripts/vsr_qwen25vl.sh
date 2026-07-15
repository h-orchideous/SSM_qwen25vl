#!/usr/bin/env bash
set -euo pipefail

# Unified VSR benchmark launcher.
# MODEL_VARIANT=vl   -> qwen2_5_vl, chunked native Qwen2.5-VL inference
# MODEL_VARIANT=sw   -> qwen_vsr_sliding_window, SimpleStream recent-N raw-frame window
# MODEL_VARIANT=ssm  -> qwen_vsr_sliding_window_ssm, SimpleStream window + SSM-compressed evicted frames
# Logs: GPU utilization (nvidia-smi), wall time, throughput, avg time per sample.

MODEL_VARIANT=${MODEL_VARIANT:-vl}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}

LAUNCH_CWD=$(pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi

export DECORD_EOF_RETRY_MAX=20480
export VSI_SUPER_RECALL_ROOT="/data1/ZhangHuayu/datasets/VSI-SUPER-Recall"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}
export USE_TF=${USE_TF:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
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
    echo "[vsr_qwen25] ERROR: python executable not found"
    exit 2
fi

chunk_config_script=${CHUNK_CONFIG_SCRIPT:-$SCRIPT_DIR/vsr_chunk_settings.sh}
if [ -f "$chunk_config_script" ]; then
    # shellcheck source=/dev/null
    source "$chunk_config_script"
else
    echo "[vsr_qwen25] WARNING: chunk config script not found at $chunk_config_script; using inline fallback defaults."
fi

case "$MODEL_VARIANT" in
    vl)
        model_family_default="qwen2_5_vl"
        log_root_default="logs/vsr_eval_runs/perf_vl"
        output_root_default="logs/vsr_eval_runs/output_vl"
        log_suffix_default="qwen25vl"
        main_process_port_default=29510
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=1
        ;;
    sw)
        model_family_default="qwen_vsr_sliding_window"
        log_root_default="logs/vsr_eval_runs/perf_sw"
        output_root_default="logs/vsr_eval_runs/output_sw"
        log_suffix_default="qwen_sw"
        main_process_port_default=29500
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=0
        ;;
    ssm)
        model_family_default="qwen_vsr_sliding_window_ssm"
        log_root_default="logs/vsr_eval_runs/perf_ssm"
        output_root_default="logs/vsr_eval_runs/output_ssm"
        log_suffix_default="qwen_ssm"
        main_process_port_default=29501
        checkpoint_default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct"
        force_simple_default=0
        ;;
    *)
        echo "[vsr_qwen25] ERROR: unsupported MODEL_VARIANT=$MODEL_VARIANT (expected: vl, sw, ssm)"
        exit 2
        ;;
esac

LOG_ROOT=${LOG_ROOT:-$log_root_default}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$output_root_default}

if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$LAUNCH_CWD/$LOG_ROOT"
fi
if [[ "$EVAL_OUTPUT_ROOT" != /* ]]; then
    EVAL_OUTPUT_ROOT="$LAUNCH_CWD/$EVAL_OUTPUT_ROOT"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

if [ "$MODEL_VARIANT" = "vl" ] && [ "$num_processes" -gt 8 ]; then
    num_processes=8
fi
num_processes=${NUM_PROCESSES:-$num_processes}

checkpoint=${CHECKPOINT:-$checkpoint_default}
model_family=${MODEL_FAMILY:-$model_family_default}
log_suffix=${LOG_SUFFIX:-$log_suffix_default}
main_process_port=${MAIN_PROCESS_PORT:-$main_process_port_default}
force_simple=${FORCE_SIMPLE:-$force_simple_default}

# ---------------------------
# Qwen2.5-VL mode params (MODEL_VARIANT=vl)
# ---------------------------
max_pixels=${MAX_PIXELS:-175616}
min_pixels=${MIN_PIXELS:-$max_pixels}
max_num_frames=${MAX_NUM_FRAMES:-768}
video_fps=${VIDEO_FPS:-1}
use_custom_video_loader=${USE_CUSTOM_VIDEO_LOADER:-True}
max_image_size=${MAX_IMAGE_SIZE:-448}
attn_impl=${ATTN_IMPLEMENTATION:-sdpa}
interleave_visuals=${INTERLEAVE_VISUALS:-False}
use_fast_processor=${USE_FAST_PROCESSOR:-True}

# ---------------------------
# Sliding-window mode params (MODEL_VARIANT=sw/ssm)
# ---------------------------
conv_template=${CONV_TEMPLATE:-qwen_2}
num_frames=${NUM_FRAMES:-}
miv_token_len=${MIV_TOKEN_LEN:-64}
si_token_len=${SI_TOKEN_LEN:-729}
sensory_window_size=${SENSORY_WINDOW_SIZE:-140}
sliding_window_stride=${SLIDING_WINDOW_STRIDE:-1}
enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING:-True}
sensory_window_max_tokens=${SENSORY_WINDOW_MAX_TOKENS:-0}
stream_visual_micro_batch_size=${STREAM_VISUAL_MICRO_BATCH_SIZE:-1}
stream_query_mode=${STREAM_QUERY_MODE:-chunk}
ssm_d_state=${SSM_D_STATE:-64}
ssm_max_memory_len=${SSM_MAX_MEMORY_LEN:-256}
ssm_fusion_num_heads=${SSM_FUSION_NUM_HEADS:-8}
ssm_fusion_bottleneck=${SSM_FUSION_BOTTLENECK:-256}
ssm_layer_sharing=${SSM_LAYER_SHARING:-group4}
ssm_fusion_policy=${SSM_FUSION_POLICY:-query_only}

# ---------------------------
# Shared chunk params (all modes)
# Defined in scripts/vsr_chunk_settings.sh
# ---------------------------
chunk_seconds=${CHUNK_SECONDS:-200}
chunk_overlap_seconds=${CHUNK_OVERLAP_SECONDS:-0}
chunk_max_num_frames=${CHUNK_MAX_NUM_FRAMES:-200}
chunk_encode_mode=${CHUNK_ENCODE_MODE:-copy} #不进行重编码，reencode
chunk_per_video_flow=${CHUNK_PER_VIDEO_FLOW:-1}
chunk_temp_mode=${CHUNK_TEMP_MODE:-1}
keep_chunks=${KEEP_CHUNKS:-0}
single_video_path=${SINGLE_VIDEO_PATH:-}

if [ -z "${num_frames}" ]; then
    if [ "$MODEL_VARIANT" = "sw" ] || [ "$MODEL_VARIANT" = "ssm" ]; then
        num_frames=-1
        echo "[vsr_qwen25] INFO: NUM_FRAMES not set for sliding-window mode; defaulting to -1 (no per-chunk frame cap)."
    else
        num_frames=1
    fi
fi

if { [ "$MODEL_VARIANT" = "sw" ] || [ "$MODEL_VARIANT" = "ssm" ]; } && [ "${num_frames}" -eq 1 ] && [ "${sensory_window_size}" -gt 1 ]; then
    echo "[vsr_qwen25] WARNING: NUM_FRAMES=1 means each chunk contributes only 1 frame to SW runtime cache."
    echo "[vsr_qwen25] WARNING: with SENSORY_WINDOW_SIZE=${sensory_window_size}, frame-drop sliding is unlikely to trigger unless chunks per video exceed the window size."
    echo "[vsr_qwen25] HINT: for ${chunk_seconds}s chunks and 140-frame window, try NUM_FRAMES=-1 (or >=140 with VIDEO_FPS=1)."
fi

TEMP_CHUNK_ROOT_TO_CLEAN=""
TEMP_CHUNK_KEEP="$keep_chunks"

cleanup_temp_chunks() {
    if [ -z "${TEMP_CHUNK_ROOT_TO_CLEAN:-}" ]; then
        return
    fi

    if [ "$TEMP_CHUNK_KEEP" = "1" ] || [ "$TEMP_CHUNK_KEEP" = "true" ] || [ "$TEMP_CHUNK_KEEP" = "True" ]; then
        echo "[vsr_qwen25] kept temporary chunks at $TEMP_CHUNK_ROOT_TO_CLEAN"
    else
        rm -rf "$TEMP_CHUNK_ROOT_TO_CLEAN"
        echo "[vsr_qwen25] cleaned temporary chunks at $TEMP_CHUNK_ROOT_TO_CLEAN"
    fi

    TEMP_CHUNK_ROOT_TO_CLEAN=""
}

trap cleanup_temp_chunks EXIT

if [ "$MODEL_VARIANT" = "vl" ]; then
    # vl: direct video path into qwen2_5_vl with chunk/fps controls.
    model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},max_num_frames=${max_num_frames},fps=${video_fps},use_custom_video_loader=${use_custom_video_loader},max_image_size=${max_image_size},attn_implementation=${attn_impl},interleave_visuals=${interleave_visuals},use_fast_processor=${use_fast_processor}"
elif [ "$MODEL_VARIANT" = "sw" ]; then
    # sw: SimpleStream recent-N raw-frame sliding-window path.
    model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},video_max_frames=${num_frames},video_fps=${video_fps},miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},sensory_window_max_tokens=${sensory_window_max_tokens},stream_visual_micro_batch_size=${stream_visual_micro_batch_size},stream_query_mode=${stream_query_mode},use_custom_video_loader=${use_custom_video_loader},max_image_size=${max_image_size},attn_implementation=${attn_impl},use_fast_processor=${use_fast_processor}"
    if [ -n "$sliding_window_stride" ]; then
        model_args_default+=",sliding_window_stride=${sliding_window_stride}"
    fi
else
    # ssm: temporal KV-SSM. Stream frames through decoder KV, absorb evicted KV
    # into per-layer recurrent hidden states, and fuse at final query.
    model_args_default="pretrained=${checkpoint},min_pixels=${min_pixels},max_pixels=${max_pixels},video_max_frames=${num_frames},video_fps=${video_fps},miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},sensory_window_max_tokens=${sensory_window_max_tokens},stream_visual_micro_batch_size=${stream_visual_micro_batch_size},stream_query_mode=${stream_query_mode},use_custom_video_loader=${use_custom_video_loader},max_image_size=${max_image_size},attn_implementation=${attn_impl},use_fast_processor=${use_fast_processor},ssm_d_state=${ssm_d_state},ssm_max_memory_len=${ssm_max_memory_len},ssm_fusion_num_heads=${ssm_fusion_num_heads},ssm_fusion_bottleneck=${ssm_fusion_bottleneck},ssm_layer_sharing=${ssm_layer_sharing},ssm_fusion_policy=${ssm_fusion_policy}"
    if [ -n "$sliding_window_stride" ]; then
        model_args_default+=",sliding_window_stride=${sliding_window_stride}"
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
    print("[vsr_qwen25] ERROR: lmms_eval is not importable in current PYTHON_BIN")
    sys.exit(2)

import transformers
print(f"[vsr_qwen25] transformers_version={transformers.__version__}")

is_vl = variant == "vl"
is_sw_raw = variant == "sw"
is_ssm = variant == "ssm"
is_sw = is_sw_raw or is_ssm

if is_vl or is_sw:
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        print("[vsr_qwen25] ERROR: current transformers lacks Qwen2_5_VLForConditionalGeneration")
        print("[vsr_qwen25] HINT: upgrade transformers in this env, then retry")
        sys.exit(2)

if is_ssm and importlib.util.find_spec("cambrian.ssm.ssm_compressor") is None:
    print("[vsr_qwen25] ERROR: cambrian.ssm.ssm_compressor is not importable in current PYTHONPATH")
    sys.exit(2)

print("[vsr_qwen25] environment check passed")
PY
}

check_environment

cd "$REPO_ROOT"

if [ "$MODEL_VARIANT" = "vl" ] && [ "$use_custom_video_loader" = "True" ] && [ "$video_fps" = "1" ] && [ "$max_num_frames" -le 0 ]; then
    if [ "${chunk_seconds}" = "0" ] || [ "${chunk_seconds}" = "0.0" ]; then
        echo "[vsr_qwen25] ERROR: uncapped 1 FPS evaluation is not feasible for qwen2_5_vl without chunking."
        echo "[vsr_qwen25] REASON: without chunking, the full video goes into one Qwen2.5-VL forward pass and vision attention memory grows quadratically with sequence length."
        echo "[vsr_qwen25] ACTION 1: set MAX_NUM_FRAMES to a positive cap, e.g. 128 or 256."
        echo "[vsr_qwen25] ACTION 2: enable time-window chunking with overlap, e.g. CHUNK_SECONDS=24 CHUNK_OVERLAP_SECONDS=12."
        echo "[vsr_qwen25] EXAMPLE: CHUNK_SECONDS=24 CHUNK_OVERLAP_SECONDS=12 CHUNK_MAX_NUM_FRAMES=32 bash vsr_qwen25vl.sh"
        exit 2
    fi
fi

mkdir -p "$LOG_ROOT"
METRICS_CSV="$LOG_ROOT/metrics.csv"
if [ ! -f "$METRICS_CSV" ]; then
    echo "run_id,benchmark,seconds,samples,throughput_sps,sec_per_sample,output_path,log_file,gpu_log" > "$METRICS_CSV"
fi

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

    # Common run metadata (all modes)
    echo "[vsr_qwen25] variant=$MODEL_VARIANT"
    echo "[vsr_qwen25] launch_cwd=$LAUNCH_CWD"
    echo "[vsr_qwen25] repo_root=$REPO_ROOT"
    echo "[vsr_qwen25] log_root=$LOG_ROOT"
    echo "[vsr_qwen25] eval_output_root=$EVAL_OUTPUT_ROOT"
    echo "[vsr_qwen25] num_processes=$num_processes"
    echo "[vsr_qwen25] model_family=$model_family"
    echo "[vsr_qwen25] checkpoint=$checkpoint"
    echo "[vsr_qwen25] chunk_config_script=$chunk_config_script"
    if [ -n "$single_video_path" ]; then
        echo "[vsr_qwen25] single_video_mode=enabled"
        echo "[vsr_qwen25] single_video_path=$single_video_path"
        export VSR_SINGLE_VIDEO_PATH="$single_video_path"
    else
        echo "[vsr_qwen25] single_video_mode=disabled"
        unset VSR_SINGLE_VIDEO_PATH
    fi

    # Mode-specific run metadata
    if [ "$MODEL_VARIANT" = "vl" ]; then
        echo "[vsr_qwen25] mode=direct_video_start_end_windows"
        echo "[vsr_qwen25] use_custom_video_loader=$use_custom_video_loader"
        echo "[vsr_qwen25] min_pixels=$min_pixels"
        echo "[vsr_qwen25] video_fps=$video_fps"
        echo "[vsr_qwen25] max_num_frames=$max_num_frames"
        echo "[vsr_qwen25] chunk_params_source=vsr_chunk_settings.sh"
        echo "[vsr_qwen25] chunk_seconds=$chunk_seconds"
        echo "[vsr_qwen25] chunk_overlap_seconds=$chunk_overlap_seconds"
        echo "[vsr_qwen25] chunk_max_num_frames=$chunk_max_num_frames"
        echo "[vsr_qwen25] chunk_encode_mode=$chunk_encode_mode"
        echo "[vsr_qwen25] chunk_per_video_flow=$chunk_per_video_flow"
        echo "[vsr_qwen25] chunk_temp_mode=$chunk_temp_mode"
        echo "[vsr_qwen25] keep_chunks=$keep_chunks"
        if [ "$max_num_frames" -le 0 ]; then
            echo "[vsr_qwen25] frame_cap=disabled (no explicit cap)"
        fi
    else
        echo "[vsr_qwen25] mode=sliding_window"
        if [ "$MODEL_VARIANT" = "sw" ]; then
            echo "[vsr_qwen25] sliding_backend=simplestream_raw_frame_window"
        else
            echo "[vsr_qwen25] sliding_backend=native_qwen25vl_ssm"
            echo "[vsr_qwen25] ssm_d_state=$ssm_d_state"
            echo "[vsr_qwen25] ssm_max_memory_len=$ssm_max_memory_len (ignored by current temporal state-only compressor)"
            echo "[vsr_qwen25] ssm_fusion_num_heads=$ssm_fusion_num_heads"
            echo "[vsr_qwen25] ssm_fusion_bottleneck=$ssm_fusion_bottleneck"
            echo "[vsr_qwen25] ssm_layer_sharing=$ssm_layer_sharing"
            echo "[vsr_qwen25] ssm_fusion_policy=$ssm_fusion_policy"
        fi
        echo "[vsr_qwen25] chunk_params_source=vsr_chunk_settings.sh"
        echo "[vsr_qwen25] num_frames=$num_frames"
        echo "[vsr_qwen25] miv_token_len=$miv_token_len"
        echo "[vsr_qwen25] si_token_len=$si_token_len"
        echo "[vsr_qwen25] sensory_window_size=$sensory_window_size"
        echo "[vsr_qwen25] chunk_seconds=$chunk_seconds"
        echo "[vsr_qwen25] chunk_overlap_seconds=$chunk_overlap_seconds"
        echo "[vsr_qwen25] chunk_max_num_frames=$chunk_max_num_frames"
        echo "[vsr_qwen25] sliding_window_stride=${sliding_window_stride:-auto}"
        echo "[vsr_qwen25] stream_query_mode=$stream_query_mode"
        echo "[vsr_qwen25] chunk_per_video_flow=$chunk_per_video_flow"
        echo "[vsr_qwen25] chunk_temp_mode=$chunk_temp_mode"
        echo "[vsr_qwen25] keep_chunks=$keep_chunks"
        echo "[vsr_qwen25] enable_visual_feature_caching=$enable_visual_feature_caching"
    fi

    if [[ "$chunk_seconds" != "0" && "$chunk_seconds" != "0.0" ]]; then
        local task_yaml="$REPO_ROOT/lmms_eval/tasks/cambrians_vsr_local/${benchmark}.yaml"
        local chunk_output_base="${VSR_CHUNK_OUTPUT_ROOT:-$REPO_ROOT/.cache/vsr_chunks}"
        if [ "$chunk_temp_mode" = "1" ] || [ "$chunk_temp_mode" = "true" ] || [ "$chunk_temp_mode" = "True" ]; then
            mkdir -p "$chunk_output_base"
            temp_chunk_root=$(mktemp -d "$chunk_output_base/${benchmark}_${run_id}_XXXXXX")
            chunk_output_root="$temp_chunk_root"
            chunk_cleanup_status="pending"
            TEMP_CHUNK_ROOT_TO_CLEAN="$temp_chunk_root"
            TEMP_CHUNK_KEEP="$keep_chunks"
            echo "[vsr_qwen25] prechunk_storage=temporary"
            echo "[vsr_qwen25] prechunk_temp_root=$temp_chunk_root"
        else
            chunk_output_root="$chunk_output_base"
            chunk_cleanup_status="disabled"
            TEMP_CHUNK_ROOT_TO_CLEAN=""
            echo "[vsr_qwen25] prechunk_storage=persistent"
            echo "[vsr_qwen25] prechunk_output_root=$chunk_output_root"
        fi

        export VSR_CHUNK_OUTPUT_ROOT="$chunk_output_root"
        if [ "$chunk_per_video_flow" = "1" ] || [ "$chunk_per_video_flow" = "true" ] || [ "$chunk_per_video_flow" = "True" ]; then
            export VSR_STREAMING_CHUNK_MODE=1
            unset VSR_PRECHUNK_MANIFEST
            chunk_manifest="per_video_streaming"
            echo "[vsr_qwen25] chunk_prepare_mode=per_video_streaming"
        else
            export VSR_STREAMING_CHUNK_MODE=0
            local prepare_chunk_cmd=(
                "$PYTHON_BIN" "$SCRIPT_DIR/prepare_vsr_chunks.py"
                --task-yaml "$task_yaml"
                --output-root "$chunk_output_root"
                --dataset-root "$VSI_SUPER_RECALL_ROOT"
                --chunk-seconds "$chunk_seconds"
                --chunk-overlap-seconds "$chunk_overlap_seconds"
                --chunk-max-num-frames "$chunk_max_num_frames"
                --chunk-encode-mode "$chunk_encode_mode"
            )
            if [ -n "$single_video_path" ]; then
                prepare_chunk_cmd+=(--only-video "$single_video_path")
            fi
            chunk_manifest=$("${prepare_chunk_cmd[@]}")
            export VSR_PRECHUNK_MANIFEST="$chunk_manifest"
            echo "[vsr_qwen25] prechunk_manifest=$chunk_manifest"
        fi
    else
        export VSR_STREAMING_CHUNK_MODE=0
        unset VSR_PRECHUNK_MANIFEST
    fi

    # --force_simple is only needed by qwen2_5_vl simple path (enabled by default for vl).
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
throughput=samples/seconds
sec_per_sample=seconds/samples
print(f"{throughput:.6f} {sec_per_sample:.6f}")
PY
)
EOF
    fi

    if [ -n "$temp_chunk_root" ]; then
        if [ "$keep_chunks" = "1" ] || [ "$keep_chunks" = "true" ] || [ "$keep_chunks" = "True" ]; then
            chunk_cleanup_status="kept"
            echo "[vsr_qwen25] kept temporary chunks at $temp_chunk_root"
        else
            rm -rf "$temp_chunk_root"
            chunk_cleanup_status="cleaned"
            echo "[vsr_qwen25] cleaned temporary chunks at $temp_chunk_root"
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

    echo "[vsr_qwen25] summary_file=$summary_file"
    echo "[vsr_qwen25] output_path=$output_path"
    echo "[vsr_qwen25] eval_log=$eval_log"
    echo "[vsr_qwen25] exit_status=$eval_status"

    return "$eval_status"
}

# Manual single-task run: edit the task name below before launching.
if ! run_eval "vsr_local_10mins"; then
    echo "[vsr_qwen25] ERROR: evaluation failed. See the printed summary_file and eval_log above."
    exit 1
fi
