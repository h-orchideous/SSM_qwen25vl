#!/usr/bin/env bash
set -euo pipefail

# Run VSR (sliding-window baseline) with Qwen-adapted Cambrian model path.
# Logs: GPU utilization (nvidia-smi), wall time, throughput, avg time per video.

MODEL_VARIANT=${MODEL_VARIANT:-qwen}
TASK=${TASK:-vsr_local_10mins}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)

LOG_ROOT=${LOG_ROOT:-logs/vsr/qwen/perf_sw}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/vsr/qwen/output_sw}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
    echo "[vsr_sw_qwen] ERROR: python executable not found"
    exit 2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS="," read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

case "$MODEL_VARIANT" in
    qwen)
        model_family_default="qwen_vsr_sliding_window"
        log_suffix_default="qwen_sw"
        main_process_port_default=29500
        export CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-0}
        ;;
    qwen_ssm)
        model_family_default="qwen_vsr_sliding_window_ssm"
        log_suffix_default="qwen_sw_ssm"
        main_process_port_default=29501
        export CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-1}
        ;;
    *)
        echo "[vsr_sw_qwen] ERROR: unsupported MODEL_VARIANT=$MODEL_VARIANT (expected: qwen or qwen_ssm)"
        exit 2
        ;;
esac

if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$REPO_ROOT/$LOG_ROOT"
fi
if [[ "$EVAL_OUTPUT_ROOT" != /* ]]; then
    EVAL_OUTPUT_ROOT="$REPO_ROOT/$EVAL_OUTPUT_ROOT"
fi

checkpoint=${CHECKPOINT:-/data1/ZhangHuayu/models/Cambrian-S-7B}
num_frames=${NUM_FRAMES:-128}
miv_token_len=${MIV_TOKEN_LEN:-64}
si_token_len=${SI_TOKEN_LEN:-729}
sensory_window_size=${SENSORY_WINDOW_SIZE:-512}
enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING:-True}
conv_template=${CONV_TEMPLATE:-qwen_2}
log_suffix=${LOG_SUFFIX:-$log_suffix_default}
main_process_port=${MAIN_PROCESS_PORT:-$main_process_port_default}
model_family=${MODEL_FAMILY:-$model_family_default}

check_environment() {
    CHECKPOINT="$checkpoint" "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys

checkpoint = os.environ.get("CHECKPOINT", "")

if importlib.util.find_spec("lmms_eval") is None:
    print("[vsr_sw_qwen] ERROR: lmms_eval is not importable in current PYTHONPATH/PYTHON_BIN")
    sys.exit(2)

if importlib.util.find_spec("cambrian") is None:
    print("[vsr_sw_qwen] ERROR: cambrian is not importable in current PYTHONPATH")
    sys.exit(2)

cfg_path = os.path.join(checkpoint, "config.json")
if os.path.exists(cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("mm_vision_tower_aux_list") is None:
            print("[vsr_sw_qwen] ERROR: checkpoint appears non-Cambrian multimodal (missing mm_vision_tower_aux_list)")
            print("[vsr_sw_qwen] HINT: this sliding-window script expects a Cambrian-Qwen checkpoint, not raw Qwen2.5-VL.")
            sys.exit(2)
    except Exception as exc:
        print(f"[vsr_sw_qwen] WARNING: failed to parse checkpoint config: {exc}")

print("[vsr_sw_qwen] environment check passed")
PY
}

check_environment

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

    date +%s > "$start_marker"

    # GPU logging (timestamp, utilization, memory, power)
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l "$GPU_LOG_INTERVAL" > "$gpu_log" &
    local gpu_pid=$!

    local start_ts
    start_ts=$(date +%s)

    echo "[vsr_sw_qwen] variant=$MODEL_VARIANT"
    echo "[vsr_sw_qwen] model_family=$model_family"
    echo "[vsr_sw_qwen] checkpoint=$checkpoint"
    echo "[vsr_sw_qwen] conv_template=$conv_template"
    echo "[vsr_sw_qwen] num_frames=$num_frames"
    echo "[vsr_sw_qwen] miv_token_len=$miv_token_len"
    echo "[vsr_sw_qwen] si_token_len=$si_token_len"
    echo "[vsr_sw_qwen] sensory_window_size=$sensory_window_size"
    echo "[vsr_sw_qwen] enable_visual_feature_caching=$enable_visual_feature_caching"
    echo "[vsr_sw_qwen] CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-0}"

    set +e
    "$PYTHON_BIN" -m accelerate.commands.launch --num_processes "${num_processes:-1}" --main_process_port "$main_process_port" -m lmms_eval \
        --model "$model_family" \
        --model_args "pretrained=${checkpoint},conv_template=${conv_template},video_max_frames=${num_frames},miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},enable_visual_feature_caching=${enable_visual_feature_caching}" \
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
        echo "exit_status: $eval_status"
    } > "$summary_file"

    echo "$run_id,$benchmark,$seconds,$samples,$throughput,$sec_per_sample,$output_path,$eval_log,$gpu_log" >> "$METRICS_CSV"

    return "$eval_status"
}

if ! run_eval "$TASK"; then
    echo "[vsr_sw_qwen] ERROR: evaluation failed. Check eval.log and summary.txt under LOG_ROOT."
    exit 1
fi
