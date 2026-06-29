#!/usr/bin/env bash
set -euo pipefail

# Run VSC streaming sliding-window with performance logging.
# Logs: GPU utilization (nvidia-smi), wall time, throughput, avg time per sample.

LOG_ROOT=${LOG_ROOT:-logs/vsc/perf_7B_sw_test_256}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/vsc/output_7B_sw_test_256}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

export DECORD_EOF_RETRY_MAX=20480
export VSI_SUPER_COUNT_ROOT="/data1/ZhangHuayu/datasets/VSI-SUPER-Count"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

IFS="," read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
num_processes=${#devices[@]}

checkpoint=${CHECKPOINT:-/data1/ZhangHuayu/models/Cambrian-S-7B}
num_frames=${NUM_FRAMES:--1}
miv_token_len=${MIV_TOKEN_LEN:-64}
si_token_len=${SI_TOKEN_LEN:-729}
sliding_window_size=${SLIDING_WINDOW_SIZE:-32}
enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING:-True}
log_suffix=${LOG_SUFFIX:-vsc_streaming_sw}
main_process_port=${MAIN_PROCESS_PORT:-29500}


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

    set +e
    accelerate launch --num_processes "$num_processes" --main_process_port "$main_process_port" -m lmms_eval \
        --model cambrians_vsc_streaming_sliding_window \
        --model_args "pretrained=${checkpoint},conv_template=qwen_2,video_max_frames=${num_frames},miv_token_len=${miv_token_len},si_token_len=${si_token_len},sliding_window_size=${sliding_window_size},enable_visual_feature_caching=${enable_visual_feature_caching}" \
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

run_eval "cambrians_vsc_streaming_local_10mins"
#run_eval "cambrians_vsc_streaming_local_30mins"
#run_eval "cambrians_vsc_streaming_local_60mins"
#run_eval "cambrians_vsc_streaming_local_120mins"
#run_eval "cambrians_vsc_streaming_local_240mins"
