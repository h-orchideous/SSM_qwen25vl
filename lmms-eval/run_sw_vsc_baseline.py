#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path


def build_model_args(args: argparse.Namespace) -> str:
    return (
        f"pretrained={args.pretrained},"
        f"conv_template={args.conv_template},"
        f"miv_token_len={args.miv_token_len},"
        f"si_token_len={args.si_token_len},"
        f"sliding_window_size={args.sliding_window_size},"
        f"enable_visual_feature_caching={str(args.enable_visual_feature_caching)}"
    )


def get_gpu_free_mem_list() -> list[tuple[int, int]]:
    """Return [(gpu_index, free_mem_mib), ...] sorted by free memory descending."""
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
            idx = int(idx_str)
            free_mem = int(free_str)
            pairs.append((idx, free_mem))

        if not pairs:
            raise RuntimeError("cannot parse gpu status")
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    except Exception:
        return []


def auto_select_gpus(min_free_gb: float) -> list[str]:
    """Pick candidate GPUs by free memory (desc), filtered by min_free_gb when possible."""
    pairs = get_gpu_free_mem_list()
    min_free_mib = int(min_free_gb * 1024)

    if not pairs:
        env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        return [env_gpu]

    filtered = [idx for idx, free_mem in pairs if free_mem >= min_free_mib]
    if filtered:
        return [str(i) for i in filtered]

    # If all GPUs are below threshold, still return best-effort order.
    return [str(i) for i, _ in pairs]


def find_local_vsc_label_file(vsc_local_root: str) -> Path | None:
    candidates = [
        Path(vsc_local_root) / "test_10mins.parquet",
        Path(vsc_local_root) / "annotations" / "test_10mins.parquet",
        Path(vsc_local_root) / "labels" / "test_10mins.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    root = Path(vsc_local_root)
    if root.exists():
        for p in root.rglob("test_10mins.parquet"):
            if p.is_file():
                return p.resolve()

    for cache_root in [
        Path("/home/ZhangHuayu/.cache/huggingface"),
        Path("/data1/ZhangHuayu/datasets"),
    ]:
        if cache_root.exists():
            for p in cache_root.rglob("test_10mins.parquet"):
                if "VSI-SUPER-Count" in str(p):
                    return p.resolve()

    return None


def sync_local_task_yaml(project_root: Path, label_file: Path) -> None:
    task_yaml = (
        project_root
        / "lmms_eval"
        / "tasks"
        / "cambrians_vsc_streaming_local"
        / "cambrians_vsc_streaming_local_10mins.yaml"
    )
    text = task_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()

    new_lines = []
    replaced = False
    for line in lines:
        if line.strip().startswith("test:"):
            new_lines.append(f"    test: {label_file}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError(f"cannot find test path field in {task_yaml}")

    task_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def ensure_local_videos(task: str, vsc_local_root: str, auto_extract: bool = True) -> None:
    """Ensure local videos referenced by VSC tasks exist. Auto-extract if requested."""
    root = Path(vsc_local_root)

    if task.endswith("10mins"):
        subdir = "10mins"
        zip_candidates = [root / "10mins.zip"]
    elif task.endswith("30mins"):
        subdir = "30mins"
        zip_candidates = [root / "30mins.zip"]
    elif task.endswith("60mins"):
        subdir = "60mins"
        zip_candidates = [root / "60mins.zip"]
    elif task.endswith("120mins"):
        subdir = "120mins"
        zip_candidates = sorted(root.glob("120mins*.zip"))
    else:
        return

    video_dir = root / subdir
    has_videos = video_dir.exists() and any(video_dir.glob("*.mp4"))
    if has_videos:
        return

    if not auto_extract:
        raise RuntimeError(
            f"Video directory missing or empty: {video_dir}. "
            "Enable auto extract or manually unzip dataset videos."
        )

    if not zip_candidates:
        raise RuntimeError(f"No zip files found for task videos under: {root}")

    print(f"[INFO] Video directory not ready: {video_dir}")
    print("[INFO] Auto-extracting local VSC videos...")
    root.mkdir(parents=True, exist_ok=True)
    for zp in zip_candidates:
        if not zp.exists():
            continue
        print(f"[INFO] extracting {zp}")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(root)

    has_videos = video_dir.exists() and any(video_dir.glob("*.mp4"))
    if not has_videos:
        raise RuntimeError(f"Extraction finished but no videos found in: {video_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Cambrian sliding-window VSC benchmark with editable arguments.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval"),
        help="Path to lmms-eval root directory.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run lmms_eval.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="CUDA_VISIBLE_DEVICES value. Use 'auto' to pick the most free GPU.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Minimum free GPU memory (GB) required for auto selection preference.",
    )
    parser.add_argument(
        "--max-gpu-retries",
        type=int,
        default=3,
        help="When OOM occurs, retry on next candidate GPU up to this many attempts.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/home/ZhangHuayu/Workspace/models/Cambrian-S-0.5B",
        help="Local model path or HF model id.",
    )
    parser.add_argument("--conv-template", type=str, default="qwen_2")
    parser.add_argument("--miv-token-len", type=int, default=64)
    parser.add_argument("--si-token-len", type=int, default=729)
    parser.add_argument("--sliding-window-size", type=int, default=128)
    parser.add_argument(
        "--enable-visual-feature-caching",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Whether to cache visual features.",
    )
    parser.add_argument("--task", type=str, default="cambrians_vsc_streaming_local_10mins")
    parser.add_argument(
        "--vsc-local-root",
        type=str,
        default="/data1/ZhangHuayu/datasets/VSI-SUPER-Count",
        help="Local root directory for VSI-SUPER-Count videos.",
    )
    parser.add_argument(
        "--vsc-label-file",
        type=str,
        default="",
        help="Optional explicit local label parquet path (e.g., test_10mins.parquet).",
    )
    parser.add_argument(
        "--auto-extract-videos",
        action="store_true",
        default=True,
        help="Auto-extract local VSC zip videos when target folder is missing.",
    )
    parser.add_argument(
        "--no-auto-extract-videos",
        action="store_false",
        dest="auto_extract_videos",
        help="Disable automatic local video extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=-1, help="Use > 0 for quick smoke test.")
    parser.add_argument("--log-samples-suffix", type=str, default="sw_baseline_vsc_10mins")
    parser.add_argument("--output-path", type=str, default="logs/baseline_compare")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and env only, do not execute.",
    )

    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if not project_root.exists():
        print(f"[ERROR] project root not found: {project_root}")
        return 1

    if args.task == "cambrians_vsc_streaming_local_10mins":
        if args.vsc_label_file:
            label_file = Path(args.vsc_label_file).expanduser().resolve()
            if not label_file.exists():
                print(f"[ERROR] vsc label file not found: {label_file}")
                return 1
        else:
            found = find_local_vsc_label_file(args.vsc_local_root)
            if found is None:
                print("[ERROR] Cannot find local VSC label file: test_10mins.parquet")
                print(f"[HINT] Put it under {args.vsc_local_root} or pass --vsc-label-file /abs/path/test_10mins.parquet")
                return 1
            label_file = found
        sync_local_task_yaml(project_root, label_file)

    if args.task.startswith("cambrians_vsc_streaming_local_"):
        ensure_local_videos(args.task, args.vsc_local_root, auto_extract=args.auto_extract_videos)

    env = os.environ.copy()
    candidate_gpus = auto_select_gpus(args.min_free_gb) if args.gpu.lower() == "auto" else [args.gpu]
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("DECORD_EOF_RETRY_MAX", "20480")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env["VSI_SUPER_COUNT_ROOT"] = args.vsc_local_root

    cmd = [
        args.python_bin,
        "-m",
        "lmms_eval",
        "--model",
        "cambrians_vsc_streaming_sliding_window",
        "--model_args",
        build_model_args(args),
        "--tasks",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--log_samples_suffix",
        args.log_samples_suffix,
        "--output_path",
        args.output_path,
    ]

    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    print("[INFO] project_root:", project_root)
    print("[INFO] python_bin:", args.python_bin)
    print("[INFO] gpu_selection_mode:", args.gpu)
    print("[INFO] candidate_gpus:", ",".join(candidate_gpus))
    print("[INFO] min_free_gb:", args.min_free_gb)
    print("[INFO] max_gpu_retries:", args.max_gpu_retries)
    print("[INFO] task:", args.task)
    print("[INFO] VSI_SUPER_COUNT_ROOT:", env["VSI_SUPER_COUNT_ROOT"])
    print("[INFO] auto_extract_videos:", args.auto_extract_videos)
    if args.task == "cambrians_vsc_streaming_local_10mins":
        print("[INFO] VSC label file:", str(label_file))
    print("[INFO] HF_HUB_OFFLINE:", env["HF_HUB_OFFLINE"])
    print("[INFO] HF_DATASETS_OFFLINE:", env["HF_DATASETS_OFFLINE"])
    print("[RUN]", printable_cmd)

    if args.dry_run:
        return 0

    max_attempts = min(max(args.max_gpu_retries, 1), len(candidate_gpus))
    last_code = 1
    for attempt_idx, gpu in enumerate(candidate_gpus[:max_attempts], start=1):
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[INFO] attempt {attempt_idx}/{max_attempts} on GPU {gpu}")
        completed = subprocess.run(cmd, cwd=project_root, env=env)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode == 0:
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            return 0

        stderr_lower = (completed.stderr or "").lower()
        is_oom = "outofmemoryerror" in stderr_lower or "cuda out of memory" in stderr_lower
        last_code = completed.returncode

        if is_oom and attempt_idx < max_attempts:
            print(f"[WARN] OOM on GPU {gpu}, retrying on next candidate...")
            continue

        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode

    return last_code


if __name__ == "__main__":
    raise SystemExit(main())
