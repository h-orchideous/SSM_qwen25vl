#!/usr/bin/env bash
set -euo pipefail

# Run VSR benchmark with switchable model variant.
# MODEL_VARIANT=vl   -> qwen2_5_vl (native Qwen chat route by default, no extra strategy)
# MODEL_VARIANT=ssm  -> cambrians_vsr_sliding_window_ssm (SSM path)
# Logs: GPU utilization (nvidia-smi), wall time, throughput, avg time per sample.

MODEL_VARIANT=${MODEL_VARIANT:-vl}
GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-5}
TASK=${TASK:-vsr_local_10mins}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

export DECORD_EOF_RETRY_MAX=20480
export VSI_SUPER_RECALL_ROOT="/data1/ZhangHuayu/datasets/VSI-SUPER-Recall"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(command -v python)
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "[vsr_qwen25] ERROR: python executable not found"
    exit 2
fi

case "$MODEL_VARIANT" in
    vl)
        model_family_default="qwen2_5_vl"
        log_root_default="logs/vsr/perf_qwen25vl"
        output_root_default="logs/vsr/output_qwen25vl"
        log_suffix_default="qwen25vl"
        main_process_port_default=29510
        ;;
    ssm)
        model_family_default="cambrians_vsr_sliding_window_ssm"
        log_root_default="logs/vsr/perf_qwen25vl_ssm"
        output_root_default="logs/vsr/output_qwen25vl_ssm"
        log_suffix_default="qwen25vl_ssm"
        main_process_port_default=29511
        export CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-1}
        ;;
    *)
        echo "[vsr_qwen25] ERROR: unsupported MODEL_VARIANT=$MODEL_VARIANT (expected: vl or ssm)"
        exit 2
        ;;
esac

LOG_ROOT=${LOG_ROOT:-$log_root_default}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$output_root_default}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

checkpoint=${CHECKPOINT:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
model_family=${MODEL_FAMILY:-$model_family_default}
log_suffix=${LOG_SUFFIX:-$log_suffix_default}
main_process_port=${MAIN_PROCESS_PORT:-$main_process_port_default}

attn_impl=${ATTN_IMPLEMENTATION:-sdpa}
interleave_visuals=${INTERLEAVE_VISUALS:-False}

if [ "$MODEL_VARIANT" = "vl" ]; then
    # Keep only pretrained so all video processing follows Qwen defaults.
    model_args_default="pretrained=${checkpoint}"
else
    model_args_default="pretrained=${checkpoint},conv_template=qwen_1_5"
fi
model_args=${MODEL_ARGS:-$model_args_default}

check_environment() {
    MODEL_VARIANT="$MODEL_VARIANT" CHECKPOINT="$checkpoint" ALLOW_INCOMPATIBLE_CHECKPOINT="${ALLOW_INCOMPATIBLE_CHECKPOINT:-0}" "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys

variant = os.environ.get("MODEL_VARIANT", "vl")
checkpoint = os.environ.get("CHECKPOINT", "")
allow_incompatible = os.environ.get("ALLOW_INCOMPATIBLE_CHECKPOINT", "0") == "1"

if importlib.util.find_spec("lmms_eval") is None:
    print("[vsr_qwen25] ERROR: lmms_eval is not importable in current PYTHON_BIN")
    sys.exit(2)

import transformers
print(f"[vsr_qwen25] transformers_version={transformers.__version__}")

if variant == "vl":
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        print("[vsr_qwen25] ERROR: current transformers lacks Qwen2_5_VLForConditionalGeneration")
        print("[vsr_qwen25] HINT: upgrade transformers in this env, then retry")
        sys.exit(2)

if variant == "ssm":
    cfg_path = os.path.join(checkpoint, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            has_aux = cfg.get("mm_vision_tower_aux_list") is not None
            if not has_aux and not allow_incompatible:
                print("[vsr_qwen25] ERROR: checkpoint appears non-Cambrian multimodal (missing mm_vision_tower_aux_list)")
                print("[vsr_qwen25] HINT: use a Cambrian multimodal checkpoint, or set ALLOW_INCOMPATIBLE_CHECKPOINT=1 to force run")
                sys.exit(2)
        except Exception as exc:
            print(f"[vsr_qwen25] WARNING: failed to parse checkpoint config: {exc}")

print("[vsr_qwen25] environment check passed")
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

    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l "$GPU_LOG_INTERVAL" > "$gpu_log" &
    local gpu_pid=$!

    local start_ts
    start_ts=$(date +%s)

    echo "[vsr_qwen25] variant=$MODEL_VARIANT"
    echo "[vsr_qwen25] model_family=$model_family"
    echo "[vsr_qwen25] checkpoint=$checkpoint"
    if [ "$MODEL_VARIANT" = "vl" ]; then
        echo "[vsr_qwen25] mode=native_qwen_chat_no_extra_strategy"
        echo "[vsr_qwen25] model_args=$model_args"
    fi
    if [ "$MODEL_VARIANT" = "ssm" ]; then
        echo "[vsr_qwen25] CAMBRIAN_USE_SSM_MODEL=${CAMBRIAN_USE_SSM_MODEL:-0}"
    fi

    set +e
    "$PYTHON_BIN" -m accelerate.commands.launch --num_processes "${num_processes:-1}" --main_process_port "$main_process_port" -m lmms_eval \
        --model "$model_family" \
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

# Run tasks separately for easier manual editing.
# Keep one active line or uncomment multiple lines to run sequentially.
run_eval "vsr_local_10mins"
# run_eval "vsr_local_30mins"
# run_eval "vsr_local_60mins"
# run_eval "vsr_local_120mins"
# run_eval "vsr_local_240mins"
