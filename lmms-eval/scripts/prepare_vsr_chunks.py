#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import decord
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-chunk VSR videos for evaluation")
    parser.add_argument("--task-yaml", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chunk-seconds", type=float, required=True)
    parser.add_argument("--chunk-overlap-seconds", type=float, required=True)
    parser.add_argument("--chunk-max-num-frames", type=int, required=True)
    parser.add_argument("--chunk-encode-mode", required=True)
    parser.add_argument("--only-video", default="", help="optional relative or absolute path of a single target video")
    return parser.parse_args()


def read_test_parquet(task_yaml_path: str) -> str:
    test_path = None
    with open(task_yaml_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^\s*test:\s*(.+?)\s*$", line)
            if match:
                test_path = match.group(1)
                break
    if not test_path:
        raise ValueError(f"Failed to find test parquet path in {task_yaml_path}")
    return test_path


def materialize_chunk(video_path: str, chunk_path: str, start_sec: float, end_sec: float, encode_mode: str):
    clip_duration = max(0.001, end_sec - start_sec)
    copy_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}", "-t", f"{clip_duration:.3f}",
        "-i", video_path,
        "-c", "copy",
        chunk_path,
    ]
    reencode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}", "-t", f"{clip_duration:.3f}",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        chunk_path,
    ]
    primary_cmd = reencode_cmd if encode_mode == "reencode" else copy_cmd
    fallback_cmd = copy_cmd if encode_mode == "reencode" else reencode_cmd
    try:
        subprocess.run(primary_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def probe_frame_count(video_path: str) -> int:
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_path,
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    return int((probe.stdout or "").strip() or "0")


def build_chunks_for_video(video_path: str, output_root: Path, chunk_seconds: float, chunk_overlap_seconds: float, chunk_max_num_frames: int, chunk_encode_mode: str):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) if total_frames > 0 else 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    if duration <= 0 or chunk_seconds <= 0:
        return [video_path]

    rel_safe = video_path.lstrip(os.sep).replace("/", "__")
    chunk_dir = output_root / rel_safe
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    step = chunk_seconds - chunk_overlap_seconds
    start_sec = 0.0
    chunk_index = 0
    while start_sec < duration:
        end_sec = min(start_sec + chunk_seconds, duration)
        chunk_name = f"chunk_{chunk_index:04d}_{int(start_sec * 1000):010d}_{int(end_sec * 1000):010d}.mp4"
        chunk_path = chunk_dir / chunk_name
        if not chunk_path.exists():
            materialize_chunk(video_path, str(chunk_path), start_sec, end_sec, chunk_encode_mode)
        try:
            frame_count = probe_frame_count(str(chunk_path))
        except Exception:
            frame_count = 0
        if frame_count >= 2:
            chunk_paths.append(str(chunk_path))
        if end_sec >= duration:
            break
        start_sec += step
        chunk_index += 1

    return chunk_paths or [video_path]


def main():
    args = parse_args()
    parquet_path = read_test_parquet(args.task_yaml)
    dataset = load_dataset("parquet", data_files={"test": parquet_path}, split="test")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    single_target = ""
    if args.only_video:
        if os.path.isabs(args.only_video):
            single_target = os.path.normpath(args.only_video)
        else:
            single_target = os.path.normpath(os.path.join(args.dataset_root, args.only_video))

    manifest = {}
    for doc in dataset:
        rel_video_path = doc["video_path"]
        full_video_path = os.path.join(args.dataset_root, rel_video_path)
        full_video_path = os.path.normpath(full_video_path)
        if single_target and full_video_path != single_target:
            continue
        manifest[full_video_path] = build_chunks_for_video(
            full_video_path,
            output_root=output_root,
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            chunk_max_num_frames=args.chunk_max_num_frames,
            chunk_encode_mode=args.chunk_encode_mode,
        )

    task_name = Path(args.task_yaml).stem
    manifest_path = output_root / f"{task_name}_chunk_manifest.json"
    if single_target and not manifest:
        raise ValueError(f"--only-video target not found in task dataset: {single_target}")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(str(manifest_path))


if __name__ == "__main__":
    main()
