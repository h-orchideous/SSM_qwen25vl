#!/usr/bin/env bash
set -euo pipefail

# SSM streaming wrapper.
# This script keeps Qwen script in Gemini-style reference mode while
# enabling streaming-style settings for SSM evaluation.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/models/Cambrian-S-7B}
VIDEO_FPS=${VIDEO_FPS:-1}
VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:--1}
VIDEO_FORCE_SAMPLE=${VIDEO_FORCE_SAMPLE:-False}
SENSORY_WINDOW_SIZE=${SENSORY_WINDOW_SIZE:-512}
ENABLE_VISUAL_FEATURE_CACHING=${ENABLE_VISUAL_FEATURE_CACHING:-True}
MIV_TOKEN_LEN=${MIV_TOKEN_LEN:-64}
SI_TOKEN_LEN=${SI_TOKEN_LEN:-729}

if [ -z "${MODEL_ARGS:-}" ]; then
	MODEL_ARGS="pretrained=${CHECKPOINT},conv_template=qwen_2,video_fps=${VIDEO_FPS},video_max_frames=${VIDEO_MAX_FRAMES},video_force_sample=${VIDEO_FORCE_SAMPLE},miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN},sensory_window_size=${SENSORY_WINDOW_SIZE},enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING}"
fi

echo "[vsr_qwen25_ssm] checkpoint=$CHECKPOINT"
echo "[vsr_qwen25_ssm] video_fps=$VIDEO_FPS"
echo "[vsr_qwen25_ssm] video_max_frames=$VIDEO_MAX_FRAMES"
echo "[vsr_qwen25_ssm] sensory_window_size=$SENSORY_WINDOW_SIZE"
echo "[vsr_qwen25_ssm] enable_visual_feature_caching=$ENABLE_VISUAL_FEATURE_CACHING"

MODEL_VARIANT=ssm \
CHECKPOINT="$CHECKPOINT" \
MODEL_ARGS="$MODEL_ARGS" \
bash "$SCRIPT_DIR/vsr_with_metrics_qwen25vl.sh"
