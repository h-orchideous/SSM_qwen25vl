#!/usr/bin/env bash
set -euo pipefail

# Randomized-parameter VSR benchmark wrapper.
# It reuses the standard VSR metrics pipeline, but enables the Cambrian SSM model
# and randomizes the newly trained modules for inference-only sanity checks.

export PYTHONNOUSERSITE=1
export CAMBRIAN_USE_SSM_MODEL=1
export CAMBRIAN_RANDOMIZE_TRAINABLE_PARAMS=1
export CAMBRIAN_RANDOMIZE_MODULES=${CAMBRIAN_RANDOMIZE_MODULES:-ssm_compressor,mm_projector,image_newline}
export CAMBRIAN_RANDOMIZE_STD=${CAMBRIAN_RANDOMIZE_STD:-0.02}
export MODEL_FAMILY=${MODEL_FAMILY:-cambrians_vsr_sliding_window_ssm}

exec bash "$(dirname "$0")/vsr_with_metrics_sw.sh" "$@"
