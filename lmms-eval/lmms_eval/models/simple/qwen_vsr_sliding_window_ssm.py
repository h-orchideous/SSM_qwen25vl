import time
from typing import List

import decord
import torch
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen_vsr_sliding_window import Qwen_VSR_SlidingWindow
from cambrian.model.language_model.qwen2_5_ssm import Qwen2_5SSMConfig, Qwen2_5SSMForConditionalGeneration
from cambrian.ssm.ssm_compressor import SSMCacheCompressor


@register_model("qwen_vsr_sliding_window_ssm")
class Qwen_VSR_SlidingWindow_SSM(Qwen_VSR_SlidingWindow):
    """
    Native Qwen2.5-VL sliding-window path with KV-SSM long-memory fusion.

    This class keeps the official Qwen2_5_VLForConditionalGeneration model and
    uses the Qwen2.5-VL SSM wrapper to trim decoder past_key_values. KV that
    falls outside the recent frame window is absorbed into an SSM compressor and
    read back through a small fusion module during the final answer.
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
        text_config = getattr(self.model.config, "text_config", self.model.config)
        hidden_size = int(getattr(text_config, "hidden_size"))
        num_heads = int(getattr(text_config, "num_attention_heads"))
        num_kv_heads = int(getattr(text_config, "num_key_value_heads", num_heads))
        head_dim = hidden_size // num_heads
        num_layers = int(getattr(text_config, "num_hidden_layers"))

        self.ssm_compressor = SSMCacheCompressor(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_size,
            d_state=ssm_d_state,
            max_memory_len=ssm_max_memory_len,
            fusion_num_heads=ssm_fusion_num_heads,
            fusion_bottleneck=ssm_fusion_bottleneck,
            layer_sharing=ssm_layer_sharing,
        )
        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            self.ssm_compressor = self.ssm_compressor.to(device=first_param.device, dtype=first_param.dtype)
        self.model.ssm_compressor = self.ssm_compressor
        self.model.config.ssm_enabled = True
        self.model.config.ssm_training_stage = "ssm"
        self.model.config.ssm_frame_window = int(self.sensory_window_size)
        self.model.config.ssm_sliding_window = int(self.sensory_window_size)

        eval_logger.info(
            f"[qwen_vsr_sw_ssm init] backend=native_qwen25vl_temporal_kv_ssm num_layers={num_layers} "
            f"num_kv_heads={num_kv_heads} head_dim={head_dim} "
            f"ssm_d_state={ssm_d_state} ssm_max_memory_len_ignored={ssm_max_memory_len} "
            f"ssm_layer_sharing={ssm_layer_sharing} fusion_policy={self.ssm_fusion_policy}"
        )

    def _load_pretrained_model(self, pretrained: str, model_kwargs: dict):
        config = Qwen2_5SSMConfig.from_pretrained(pretrained)
        config.ssm_enabled = True
        return Qwen2_5SSMForConditionalGeneration.from_pretrained(pretrained, config=config, **model_kwargs)

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
        sampled_frames = []
        total_sampled = 0
        chunk_count = 0
        stream_start = time.perf_counter()

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

                    sampled_frames.append(frame)
                    total_sampled += 1
                    chunk_sampled += 1
                    if total_sampled > window_size:
                        chunk_evicted += 1

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if chunk_sampled > 0:
                    chunk_count += 1
                    eval_logger.info(
                        f"[qwen_vsr_sw_ssm chunk] chunk_index={chunk_idx + 1}/{len(video_paths)} chunk_frames={chunk_sampled} kept_frames={min(len(sampled_frames), window_size)} ssm_absorbed_frames={chunk_evicted} queries=0 query_mode=final input_fps={self.fps} window_mode=model_internal_kv_ssm"
                    )

        if not sampled_frames:
            return ""

        total_evicted = max(0, len(sampled_frames) - window_size)
        eval_logger.info(
            f"[qwen_vsr_sw_ssm final] sampled_frames={len(sampled_frames)} kept_frames={min(len(sampled_frames), window_size)} "
            f"ssm_absorbed_frames={total_evicted} window_mode=model_internal_kv_ssm"
        )
        generate_start = time.perf_counter()
        try:
            answer = self._generate_with_recent_frames(context, sampled_frames, current_gen_kwargs)
        finally:
            self.model._active_ssm_compressor = None
        generate_elapsed = time.perf_counter() - generate_start

        for term in until:
            if len(term) > 0:
                answer = answer.split(term)[0]

        total_elapsed = time.perf_counter() - stream_start
        eval_logger.info(
            f"[qwen_vsr_sw_ssm stream] chunks={chunk_count} sampled_frames={total_sampled} kept_frames={min(len(sampled_frames), window_size)} ssm_absorbed_frames={total_evicted} queries=1 query_mode=final window_mode=model_internal_kv_ssm generate_s={generate_elapsed:.3f} total_s={total_elapsed:.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    def _generate_with_recent_frames(self, context: str, recent_frames: List[Image.Image], current_gen_kwargs: dict) -> str:
        from qwen_vl_utils import process_vision_info

        video_visual = {
            "type": "video",
            "video": [frame.convert("RGB") for frame in recent_frames],
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        if self.fps is not None:
            video_visual["sample_fps"] = float(self.fps)

        message = self._build_message(context, [video_visual])
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([message])
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        for key in ("input_ids", "attention_mask", "image_grid_thw", "video_grid_thw"):
            if key in inputs and inputs[key] is not None:
                inputs[key] = inputs[key].to(self.device)

        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=current_gen_kwargs["do_sample"],
            temperature=current_gen_kwargs["temperature"],
            top_p=current_gen_kwargs["top_p"],
            num_beams=current_gen_kwargs["num_beams"],
            max_new_tokens=current_gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            ssm_compressor=self.ssm_compressor,
            ssm_sliding_window=self.sensory_window_size,
        )

        generated_ids_trimmed = cont[:, inputs.input_ids.shape[-1] :]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
