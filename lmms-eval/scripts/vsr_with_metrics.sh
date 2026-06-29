#!/usr/bin/env bash
set -euo pipefail

# Run VSR benchmarks with performance logging.
# Logs: GPU utilization (nvidia-smi), wall time, throughput, avg time per video.

LOG_ROOT=${LOG_ROOT:-logs/perf_VL}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/output_VL}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

export DECORD_EOF_RETRY_MAX=20480
export VSI_SUPER_RECALL_ROOT="/data1/ZhangHuayu/datasets/VSI-SUPER-Recall"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
# Evaluate the modified SSM language model on the original `cambrians` eval path.
export CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-1}
export CAMBRIAN_RANDOMIZE_TRAINABLE_PARAMS=${CAMBRIAN_RANDOMIZE_TRAINABLE_PARAMS:-0}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

# VSR requires a multimodal Cambrian checkpoint (contains vision towers/projector).
checkpoint=${CHECKPOINT:-/data1/ZhangHuayu/models/Cambrian-S-7B}

mkdir -p "$LOG_ROOT"
METRICS_CSV="$LOG_ROOT/metrics.csv"
if [ ! -f "$METRICS_CSV" ]; then
    echo "run_id,benchmark,seconds,samples,throughput_sps,sec_per_sample,output_path,log_file,gpu_log" > "$METRICS_CSV"
fi

run_eval() {
    local benchmark="$1"
    local extra_args="$2"

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

    echo "[vsr_with_metrics] CAMBRIAN_USE_SSM_MODEL=$CAMBRIAN_USE_SSM_MODEL"
    echo "[vsr_with_metrics] CAMBRIAN_RANDOMIZE_TRAINABLE_PARAMS=$CAMBRIAN_RANDOMIZE_TRAINABLE_PARAMS"
    echo "[vsr_with_metrics] checkpoint=$checkpoint"

    set +e
    bash evaluate_all_in_one.sh \
        --model cambrians \
        --benchmark "$benchmark" \
        --num_processes "${num_processes:-1}" \
        --num_frames "${NUM_FRAMES:-128}" \
        --pretrained "$checkpoint" \
        --miv_token_len "${MIV_TOKEN_LEN:-64}" \
        --si_token_len "${SI_TOKEN_LEN:-729}" \
        --output_path "$output_path" \
        $extra_args \
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

run_eval "vsr_local_10mins" "--sensory_window_size 32 --compression_downsample_ratio 2 --consolidation_method drop_merge --retrieval_topk 32 --enable_visual_feature_caching True --surprise_threshold 0.35 --consolidation_mem_budget 16384"
#run_eval "vsr_local_30mins" "--sensory_window_size 64 --compression_downsample_ratio 2 --consolidation_method drop --retrieval_topk 256 --enable_visual_feature_caching True --surprise_threshold 0.3 --consolidation_mem_budget 32768"
#run_eval "vsr_local_60mins" "--sensory_window_size 64 --compression_downsample_ratio 2 --consolidation_method drop --retrieval_topk 512 --enable_visual_feature_caching True --surprise_threshold 0.25 --consolidation_mem_budget 16384"
#run_eval "vsr_local_120mins" "--sensory_window_size 32 --compression_downsample_ratio 2 --consolidation_method drop_merge --retrieval_topk 128 --enable_visual_feature_caching True --surprise_threshold 0.25 --consolidation_mem_budget 32768"
#run_eval "vsr_local_240mins" "--sensory_window_size 32 --compression_downsample_ratio 2 --consolidation_method drop --retrieval_topk 32 --enable_visual_feature_caching True --surprise_threshold 0.35 --consolidation_mem_budget 16384"
