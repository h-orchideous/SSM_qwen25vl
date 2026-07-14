import time
from collections import deque
from typing import List

import decord
import torch
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen_ssm_patch import (
    absorb_encoded_frame_to_ssm,
    build_qwen_ssm_compressor,
    install_qwen_ssm_fusion,
)
from lmms_eval.models.simple.qwen_vsr_sliding_window import Qwen_VSR_SlidingWindow


@register_model("qwen_vsr_sliding_window_ssm")
class Qwen_VSR_SlidingWindow_SSM(Qwen_VSR_SlidingWindow):
    """
    Native Qwen2.5-VL sliding-window path with CSMS SSM long-memory fusion.

    This class keeps the official Qwen2_5_VLForConditionalGeneration model and
    patches its decoder layers after loading. Encoded frames that leave the
    recent visual window are absorbed into an SSM compressor and read back
    through a small fusion module during the final answer.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        ssm_d_state: int = 64,
        ssm_max_memory_len: int = 256,
        ssm_fusion_num_heads: int = 8,
        ssm_fusion_bottleneck: int = 256,
        ssm_layer_sharing: str = "group4",
        ssm_fusion_policy: str = "query_only",
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, log_prefix="qwen_vsr_sw_ssm", **kwargs)
        self.ssm_fusion_policy = str(ssm_fusion_policy).lower()
        if self.ssm_fusion_policy not in {"query_only", "always"}:
            raise ValueError(f"ssm_fusion_policy must be 'query_only' or 'always', got {ssm_fusion_policy}")

        self.ssm_compressor = build_qwen_ssm_compressor(
            self.model,
            d_state=ssm_d_state,
            max_memory_len=ssm_max_memory_len,
            fusion_num_heads=ssm_fusion_num_heads,
            fusion_bottleneck=ssm_fusion_bottleneck,
            layer_sharing=ssm_layer_sharing,
        )
        self.model.ssm_compressor = self.ssm_compressor
        self.model.ssm_fusion_enabled = self.ssm_fusion_policy == "always"
        patched_layers = install_qwen_ssm_fusion(self.model)

        eval_logger.info(
            f"[qwen_vsr_sw_ssm init] backend=native_qwen25vl patched_layers={patched_layers} "
            f"ssm_d_state={ssm_d_state} ssm_max_memory_len={ssm_max_memory_len} "
            f"ssm_layer_sharing={ssm_layer_sharing} fusion_policy={self.ssm_fusion_policy}"
        )

    def _absorb_evicted_encoded_frame(self, encoded_frame):
        absorb_encoded_frame_to_ssm(
            getattr(self, "ssm_compressor", None),
            self.model,
            encoded_frame.vision_emb,
            encoded_frame.grid_thw,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_streaming_video(
        self,
        context: str,
        static_visuals: List[dict],
        video_paths: List[str],
        current_gen_kwargs: dict,
        until: List[str],
    ):
        if getattr(self, "ssm_compressor", None) is not None:
            self.ssm_compressor.reset()

        if static_visuals:
            raise NotImplementedError("[qwen_vsr_sw_ssm] streaming ssm currently supports video-only requests")

        if not video_paths:
            return ""

        window_size = max(1, int(self.sensory_window_size))
        encoded_window = deque()
        total_sampled = 0
        total_evicted = 0
        chunk_count = 0
        stream_start = time.perf_counter()
        previous_fusion = getattr(self.model, "ssm_fusion_enabled", False)

        # Build SSM memory from encoded frames that leave the same visual window used by SW.
        # Fusion is disabled during ingestion so memory construction is not recursively conditioned on itself.
        self.model.ssm_fusion_enabled = False
        with torch.inference_mode():
            for chunk_idx, chunk_path in enumerate(video_paths):
                try:
                    vr = decord.VideoReader(chunk_path)
                except Exception as exc:
                    eval_logger.warning(f"[qwen_vsr_sw_ssm chunk] failed to open video={chunk_path} err={exc}")
                    continue

                total_frames = len(vr)
                if total_frames <= 0:
                    continue

                sample_indices = self._build_sample_indices(vr, total_frames, target_fps=self.fps)
                if not sample_indices:
                    continue

                chunk_sampled = 0
                chunk_evicted = 0
                for frame_idx in sample_indices:
                    try:
                        frame = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
                    except Exception as exc:
                        eval_logger.warning(
                            f"[qwen_vsr_sw_ssm chunk] skip unreadable frame. video={chunk_path} frame_idx={frame_idx} err={exc}"
                        )
                        continue

                    if self.max_image_size is not None:
                        width, height = frame.size
                        longest_edge = max(width, height)
                        if longest_edge > self.max_image_size:
                            scale = self.max_image_size / float(longest_edge)
                            frame = frame.resize(
                                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                                Image.Resampling.BICUBIC,
                            )

                    encoded_frame = self._encode_vision_frames([frame])
                    encoded_frame.frame_index = int(frame_idx)
                    encoded_frame.chunk_index = int(chunk_idx)

                    if len(encoded_window) == window_size:
                        evicted_frame = encoded_window.popleft()
                        self._absorb_evicted_encoded_frame(evicted_frame)
                        total_evicted += 1
                        chunk_evicted += 1
                    encoded_window.append(encoded_frame)

                    total_sampled += 1
                    chunk_sampled += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if chunk_sampled > 0:
                    chunk_count += 1
                    eval_logger.info(
                        f"[qwen_vsr_sw_ssm chunk] chunk_index={chunk_idx + 1}/{len(video_paths)} chunk_frames={chunk_sampled} kept_frames={len(encoded_window)} ssm_absorbed_frames={chunk_evicted} queries=0 query_mode=final input_fps={self.fps} window_mode=simplestream_encoded_csms_ssm"
                    )

        if not encoded_window:
            self.model.ssm_fusion_enabled = previous_fusion
            return ""

        generate_start = time.perf_counter()
        self.model.ssm_fusion_enabled = True
        try:
            answer = self._generate_with_encoded_window(context, list(encoded_window), current_gen_kwargs)
        finally:
            self.model.ssm_fusion_enabled = previous_fusion
        generate_elapsed = time.perf_counter() - generate_start

        for term in until:
            if len(term) > 0:
                answer = answer.split(term)[0]

        total_elapsed = time.perf_counter() - stream_start
        eval_logger.info(
            f"[qwen_vsr_sw_ssm stream] chunks={chunk_count} sampled_frames={total_sampled} kept_frames={len(encoded_window)} ssm_absorbed_frames={total_evicted} queries=1 query_mode=final window_mode=simplestream_encoded_csms_ssm generate_s={generate_elapsed:.3f} total_s={total_elapsed:.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer
