#!/usr/bin/env python3
"""Wrapper script to run sliding-window VSR (Recall) on local 10mins dataset.

This script sets sensible env vars and invokes `python -m lmms_eval` with the
`cambrians_vsc_streaming_sliding_window` model and the `cambrians_vsr_local_10mins`
task. It is intentionally minimal — for advanced GPU selection use
`run_sw_vsc_baseline.py`.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path


def get_gpu_free_mem_list() -> list[tuple[int, int]]:
    """Return [(gpu_index, free_mem_mib), ...] sorted by free memory desc."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if not out:
            raise RuntimeError("empty nvidia-smi output")

        pairs = []
        for line in out.splitlines():
            idx_str, free_str = [x.strip() for x in line.split(",", 1)]
            pairs.append((int(idx_str), int(free_str)))

        if not pairs:
            raise RuntimeError("cannot parse gpu status")
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    except Exception:
        return []


def pick_least_used_gpu(default_gpu: str = "0") -> str:
    """Pick GPU with most free memory (best-effort)."""
    pairs = get_gpu_free_mem_list()
    if not pairs:
        return default_gpu
    return str(pairs[0][0])


def main():
    project_root = Path(__file__).resolve().parent
    python_bin = sys.executable

    model_args = (
        "pretrained=/home/ZhangHuayu/Workspace/models/Cambrian-S-0.5B,"
        "conv_template=qwen_2,miv_token_len=64,si_token_len=729,"
        "sliding_window_size=128,enable_visual_feature_caching=True"
    )

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("DECORD_EOF_RETRY_MAX", "20480")
    # Point to local Recall dataset root
    env.setdefault("VSI_SUPER_RECALL_ROOT", "/data1/ZhangHuayu/datasets/VSI-SUPER-Recall")

    cmd = [
        python_bin,
        "-m",
        "lmms_eval",
        "--model",
        "cambrians_vsc_streaming_sliding_window",
        "--model_args",
        model_args,
        "--tasks",
        "cambrians_vsr_local_10mins",
        "--batch_size",
        "1",
        "--log_samples",
        "--log_samples_suffix",
        "sw_baseline_vsr_10mins",
        "--output_path",
        "logs/baseline_compare",
    ]

    printable = " ".join(shlex.quote(p) for p in cmd)
    print("[INFO] project_root:", project_root)
    print("[INFO] python_bin:", python_bin)
    print("[RUN]", printable)

    # Respect CUDA_VISIBLE_DEVICES set by caller; otherwise auto-select GPU
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = pick_least_used_gpu(default_gpu="0")

    return subprocess.call(cmd, cwd=project_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
