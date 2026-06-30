#!/usr/bin/env bash

# Shared VSC chunk settings for vl / sw / sw_ssm.
# Edit these values before each evaluation run, then launch scripts/vsc_qwen25vl.sh.
# This file is auto-sourced by scripts/vsc_qwen25vl.sh.
#
# Meaning:
# - CHUNK_SECONDS: time span of each chunk
# - CHUNK_OVERLAP_SECONDS: overlap between adjacent chunks
# - CHUNK_MAX_NUM_FRAMES: per-chunk frame cap
# - CHUNK_ENCODE_MODE: ffmpeg chunk materialization mode (reencode/copy/auto)
# - CHUNK_PER_VIDEO_FLOW: per-video stream flow (slice -> infer -> aggregate -> cleanup)
# - CHUNK_TEMP_MODE: write chunks into a per-run temporary directory when set to 1
# - KEEP_CHUNKS: keep the temporary chunk directory after the run when set to 1

export CHUNK_SECONDS=${CHUNK_SECONDS:-150}
export CHUNK_OVERLAP_SECONDS=${CHUNK_OVERLAP_SECONDS:-0}
export CHUNK_MAX_NUM_FRAMES=${CHUNK_MAX_NUM_FRAMES:-150}
export CHUNK_ENCODE_MODE=${CHUNK_ENCODE_MODE:-reencode}  # reencode / copy / auto
export CHUNK_PER_VIDEO_FLOW=${CHUNK_PER_VIDEO_FLOW:-1}
export CHUNK_TEMP_MODE=${CHUNK_TEMP_MODE:-1} #不保存，临时切片
export KEEP_CHUNKS=${KEEP_CHUNKS:-0} #不保存

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[vsc_chunk] CHUNK_SECONDS=$CHUNK_SECONDS"
    echo "[vsc_chunk] CHUNK_OVERLAP_SECONDS=$CHUNK_OVERLAP_SECONDS"
    echo "[vsc_chunk] CHUNK_MAX_NUM_FRAMES=$CHUNK_MAX_NUM_FRAMES"
    echo "[vsc_chunk] CHUNK_ENCODE_MODE=$CHUNK_ENCODE_MODE"
    echo "[vsc_chunk] CHUNK_PER_VIDEO_FLOW=$CHUNK_PER_VIDEO_FLOW"
    echo "[vsc_chunk] CHUNK_TEMP_MODE=$CHUNK_TEMP_MODE"
    echo "[vsc_chunk] KEEP_CHUNKS=$KEEP_CHUNKS"
    echo "[vsc_chunk] This script is auto-sourced by scripts/vsc_qwen25vl.sh."
fi
