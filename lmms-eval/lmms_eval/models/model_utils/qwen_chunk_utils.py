import math
import os
import subprocess
import fcntl
from dataclasses import dataclass
from typing import Optional

import decord
from loguru import logger as eval_logger


@dataclass
class ChunkConfig:
    seconds: float
    overlap_seconds: float
    max_num_frames: int
    encode_mode: str

    @property
    def enabled(self) -> bool:
        return self.seconds > 0


def load_chunk_config(max_num_frames_fallback: int = -1) -> ChunkConfig:
    seconds = float(os.getenv("CHUNK_SECONDS", "0") or 0)
    overlap_seconds = float(os.getenv("CHUNK_OVERLAP_SECONDS", "0") or 0)
    max_num_frames = int(os.getenv("CHUNK_MAX_NUM_FRAMES", "-1") or -1)
    encode_mode = str(os.getenv("CHUNK_ENCODE_MODE", "reencode") or "reencode").lower()

    if seconds < 0 or overlap_seconds < 0:
        raise ValueError("chunk_seconds and chunk_overlap_seconds must be non-negative")
    if seconds > 0 and overlap_seconds >= seconds:
        raise ValueError("chunk_overlap_seconds must be smaller than chunk_seconds")
    if encode_mode not in {"auto", "copy", "reencode"}:
        raise ValueError("chunk_encode_mode must be one of {'auto', 'copy', 'reencode'}")

    if max_num_frames <= 0 and max_num_frames_fallback > 0:
        max_num_frames = int(max_num_frames_fallback)

    return ChunkConfig(
        seconds=seconds,
        overlap_seconds=overlap_seconds,
        max_num_frames=max_num_frames,
        encode_mode=encode_mode,
    )


def build_chunk_specs(video_path: str, chunk_config: ChunkConfig, video_reader_cache: Optional[dict] = None):
    if not chunk_config.enabled:
        return []

    if video_reader_cache is not None and video_path in video_reader_cache:
        vr = video_reader_cache[video_path]
    else:
        vr = decord.VideoReader(video_path)
        if video_reader_cache is not None:
            video_reader_cache[video_path] = vr

    total_frames = len(vr)
    raw_fps = float(vr.get_avg_fps())
    duration = total_frames / raw_fps if raw_fps > 0 else 0.0
    if duration <= 0:
        return []

    step = chunk_config.seconds - chunk_config.overlap_seconds
    nframes_per_chunk = chunk_config.max_num_frames if chunk_config.max_num_frames > 0 else 32

    chunked_visuals = []
    start_sec = 0.0
    chunk_index = 0
    while start_sec < duration:
        end_sec = min(start_sec + chunk_config.seconds, duration)
        chunked_visuals.append(
            {
                "type": "video_chunk_spec",
                "video_path": video_path,
                "chunk_index": chunk_index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "nframes_per_chunk": nframes_per_chunk,
            }
        )
        if end_sec >= duration:
            break
        start_sec += step
        chunk_index += 1

    return chunked_visuals


def materialize_chunk_visual(
    chunk_spec: dict,
    *,
    max_pixels: int,
    min_pixels: int,
    fps: Optional[float],
    encode_mode: str,
    log_prefix: str,
    chunk_dir: str,
):
    video_path = chunk_spec["video_path"]
    start_sec = chunk_spec["start_sec"]
    end_sec = chunk_spec["end_sec"]
    nframes_per_chunk = int(chunk_spec["nframes_per_chunk"])
    clip_duration = max(0.001, end_sec - start_sec)

    chunk_index = int(chunk_spec.get("chunk_index", 0))
    chunk_name = f"chunk_{chunk_index:04d}_{int(start_sec * 1000):010d}_{int(end_sec * 1000):010d}.mp4"
    chunk_path = os.path.join(chunk_dir, chunk_name)

    def _is_decodable(path: str) -> bool:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,codec_name,width,height",
            "-of",
            "csv=p=0",
            path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe.returncode != 0:
            return False
        fields = [part.strip() for part in (probe.stdout or "").strip().split(",")]
        if len(fields) < 4 or "unknown" in fields:
            return False
        decode_cmd = ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "null", "-"]
        return subprocess.run(decode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode == 0

    lock_path = f"{chunk_path}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if os.path.exists(chunk_path) and not _is_decodable(chunk_path):
            eval_logger.warning(
                f"[{log_prefix} chunk] remove invalid chunk before rematerializing. clip={os.path.basename(chunk_path)}"
            )
            try:
                os.remove(chunk_path)
            except OSError:
                pass

        if os.path.exists(chunk_path):
            pass
        else:
            tmp_chunk_path = f"{chunk_path}.tmp.{os.getpid()}.mp4"
            copy_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_sec:.3f}",
                "-t",
                f"{clip_duration:.3f}",
                "-i",
                video_path,
                "-c",
                "copy",
                tmp_chunk_path,
            ]
            reencode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_sec:.3f}",
                "-t",
                f"{clip_duration:.3f}",
                "-i",
                video_path,
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp_chunk_path,
            ]

            primary_cmd = reencode_cmd if encode_mode == "reencode" else copy_cmd
            fallback_mode = None
            fallback_cmd = None
            if encode_mode == "reencode":
                fallback_mode, fallback_cmd = "copy", copy_cmd
            elif encode_mode in {"copy", "auto"}:
                fallback_mode, fallback_cmd = "reencode", reencode_cmd

            try:
                subprocess.run(primary_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            except subprocess.CalledProcessError as exc:
                stderr_msg = exc.stderr.strip() if exc.stderr else ""
                if fallback_cmd is None:
                    raise
                eval_logger.warning(
                    f"[{log_prefix} chunk] ffmpeg {encode_mode} failed, fallback to {fallback_mode}. video={os.path.basename(video_path)} start={start_sec:.2f}s end={end_sec:.2f}s err={stderr_msg}"
                )
                try:
                    subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as fallback_exc:
                    fallback_stderr = fallback_exc.stderr.strip() if fallback_exc.stderr else ""
                    eval_logger.warning(
                        f"[{log_prefix} chunk] ffmpeg fallback {fallback_mode} failed. video={os.path.basename(video_path)} start={start_sec:.2f}s end={end_sec:.2f}s err={fallback_stderr}"
                    )
                    try:
                        os.remove(tmp_chunk_path)
                    except OSError:
                        pass
                    return None

            if not _is_decodable(tmp_chunk_path):
                eval_logger.warning(
                    f"[{log_prefix} chunk] materialized chunk is not decodable; skip. video={os.path.basename(video_path)} start={start_sec:.2f}s end={end_sec:.2f}s clip={os.path.basename(chunk_path)}"
                )
                try:
                    os.remove(tmp_chunk_path)
                except OSError:
                    pass
                return None

            os.replace(tmp_chunk_path, chunk_path)

    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            chunk_path,
        ]
        probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        frame_count = int((probe.stdout or "").strip() or "0")
        if frame_count < 2:
            eval_logger.warning(
                f"[{log_prefix} chunk] skip short chunk (<2 frames). video={os.path.basename(video_path)} start={start_sec:.2f}s end={end_sec:.2f}s clip={os.path.basename(chunk_path)} frames={frame_count}"
            )
            return None
    except Exception as exc:
        eval_logger.warning(
            f"[{log_prefix} chunk] failed to probe frame count; continue without short-chunk filter. clip={os.path.basename(chunk_path)} err={exc}"
        )

    chunk = {
        "type": "video",
        "video": chunk_path,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
    }
    if fps is not None:
        effective_fps = float(fps)
        if nframes_per_chunk > 0:
            max_fps_by_chunk_cap = float(nframes_per_chunk) / clip_duration
            effective_fps = min(effective_fps, max_fps_by_chunk_cap)
        chunk["fps"] = max(effective_fps, 1e-3)
        if nframes_per_chunk > 0:
            chunk["max_frames"] = nframes_per_chunk
    elif nframes_per_chunk > 0:
        chunk["nframes"] = nframes_per_chunk

    chunk["chunk_debug"] = (
        f"video={os.path.basename(video_path)} start={start_sec:.2f}s end={end_sec:.2f}s clip={os.path.basename(chunk_path)}"
    )
    return chunk


def build_chunked_frame_indices(total_frames: int, avg_fps: float, chunk_config: ChunkConfig):
    frame_idx = list(range(total_frames))
    if not chunk_config.enabled or total_frames <= 0 or avg_fps <= 0:
        return frame_idx

    step_seconds = chunk_config.seconds - chunk_config.overlap_seconds
    start_sec = 0.0
    duration = total_frames / avg_fps
    merged_idx = []
    while start_sec < duration:
        end_sec = min(start_sec + chunk_config.seconds, duration)
        start_frame = int(math.floor(start_sec * avg_fps))
        end_frame = int(math.ceil(end_sec * avg_fps))
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        chunk_idx = list(range(start_frame, end_frame))
        if chunk_config.max_num_frames > 0 and len(chunk_idx) > chunk_config.max_num_frames:
            pick = [
                int(round(i))
                for i in __import__("numpy").linspace(0, len(chunk_idx) - 1, chunk_config.max_num_frames)
            ]
            chunk_idx = [chunk_idx[p] for p in pick]

        merged_idx.extend(chunk_idx)
        if end_sec >= duration:
            break
        start_sec += step_seconds

    return list(dict.fromkeys(merged_idx))
