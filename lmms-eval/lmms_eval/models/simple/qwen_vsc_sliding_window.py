import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


LOG_PREFIX = "qwen_vsc_sw"


@register_model("qwen_vsc_sliding_window")
class Qwen_VSC_SlidingWindow(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        video_max_frames: int = 32,
        video_fps: Optional[float] = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        miv_token_len: int = 64,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        use_custom_video_loader: Optional[bool] = False,
        sensory_window_size: int = 140,
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        enable_visual_feature_caching: bool = False,
        sensory_window_max_tokens: Optional[int] = 0,
        stream_visual_micro_batch_size: int = 1,
        stream_query_mode: str = "chunk",
        conv_template: Optional[str] = "qwen_2",
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        use_fast_processor: Optional[bool] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        log_prefix: str = LOG_PREFIX,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.log_prefix = str(log_prefix)
        self.video_force_sample = bool(video_force_sample)
        self.add_time_instruction = bool(add_time_instruction)
        self.miv_token_len = int(miv_token_len)
        self.si_token_len = int(si_token_len)
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_max_subimages = int(anyres_max_subimages)
        self.enable_visual_feature_caching = bool(enable_visual_feature_caching)
        self.conv_template = conv_template
        self.sensory_window_size = int(sliding_window_size) if sliding_window_size is not None else int(sensory_window_size)
        self.sliding_window_stride = int(sliding_window_stride) if sliding_window_stride is not None else max(1, self.sensory_window_size // 2)
        self.sensory_window_max_tokens = None if sensory_window_max_tokens is None or int(sensory_window_max_tokens) <= 0 else int(sensory_window_max_tokens)
        self.stream_visual_micro_batch_size = max(1, int(stream_visual_micro_batch_size))
        self.stream_query_mode = str(stream_query_mode).lower()
        if self.stream_query_mode not in {"chunk", "frame"}:
            raise ValueError(f"stream_query_mode must be 'chunk' or 'frame', got {stream_query_mode}")
        self.fps = None if video_fps is None else float(video_fps)
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = int(video_max_frames)
        eval_logger.info(f"video_max_frames: {self.max_num_frames}")
        eval_logger.info(f"video_fps: {self.fps}")
        eval_logger.info(f"video_force_sample: {self.video_force_sample}")
        eval_logger.info(f"add_time_instruction: {self.add_time_instruction}")
        eval_logger.info(f"miv_token_len: {self.miv_token_len}")
        eval_logger.info(f"si_token_len: {self.si_token_len}")
        eval_logger.info(f"image_aspect_ratio: {self.image_aspect_ratio}")
        eval_logger.info(f"anyres_max_subimages: {self.anyres_max_subimages}")
        eval_logger.info(f"sensory_window_size: {self.sensory_window_size}")
        eval_logger.info(f"sliding_window_stride: {self.sliding_window_stride}")
        eval_logger.info(f"sensory_window_max_tokens: {self.sensory_window_max_tokens}")
        eval_logger.info(f"stream_visual_micro_batch_size: {self.stream_visual_micro_batch_size}")
        eval_logger.info(f"stream_query_mode: {self.stream_query_mode}")
        eval_logger.info(
            f"[{self.log_prefix} init] backend=raw_qwen25vl video_max_frames={self.max_num_frames} video_fps={self.fps} sensory_window_size={self.sensory_window_size} sliding_window_stride={self.sliding_window_stride} sensory_window_max_tokens={self.sensory_window_max_tokens} stream_visual_micro_batch_size={self.stream_visual_micro_batch_size} stream_query_mode={self.stream_query_mode}"
        )
        if self.max_num_frames == 1 and self.sensory_window_size > 1:
            eval_logger.warning(
                f"[{self.log_prefix} init] video_max_frames=1: each chunk contributes one frame only; frame-drop sliding may not trigger for large windows"
            )

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        processor_kwargs = {"max_pixels": max_pixels, "min_pixels": min_pixels}
        if use_fast_processor is not None:
            processor_kwargs["use_fast"] = bool(use_fast_processor)
        self.processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen_VSC_SlidingWindow")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_content(self, video_path: str, video_reader_cache: Optional[dict] = None):
        video_content = {
            "type": "video",
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }

        if self.use_custom_video_loader:
            # max_num_frames <= 0 means no explicit frame cap.
            effective_max_num_frames = int(self.max_num_frames) if int(self.max_num_frames) > 0 else 10**9
            sampled_frames = read_video_pyav_pil(
                video_path,
                num_frm=effective_max_num_frames,
                fps=self.fps,
                max_image_size=self.max_image_size,
                force_include_last_frame=True,
            )
            raw_fps = None
            try:
                raw_fps = float(decord.VideoReader(video_path).get_avg_fps())
            except Exception as exc:
                eval_logger.warning(f"Failed to read raw FPS for {video_path}: {exc}")

            sample_fps = self.fps
            if sample_fps is None and raw_fps and len(sampled_frames) > 0:
                try:
                    duration = len(decord.VideoReader(video_path)) / raw_fps
                    if duration > 0:
                        sample_fps = len(sampled_frames) / duration
                except Exception:
                    sample_fps = None

            video_content["video"] = sampled_frames
            if sample_fps is not None:
                video_content["sample_fps"] = float(sample_fps)
            if raw_fps is not None:
                video_content["raw_fps"] = raw_fps
        else:
            video_content["video"] = video_path
            if self.fps is not None:
                video_content["fps"] = self.fps
            else:
                video_content["nframes"] = self.max_num_frames

        return video_content

    def _sample_video_frames(self, video_path: str):
        if self.use_custom_video_loader:
            sampled_frames = read_video_pyav_pil(
                video_path,
                num_frm=(int(self.max_num_frames) if int(self.max_num_frames) > 0 else 10**9),
                fps=self.fps,
                max_image_size=self.max_image_size,
                force_include_last_frame=True,
            )
            return sampled_frames

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        if total_frames <= 0:
            return []

        frame_idx = self._build_sample_indices(vr, total_frames)

        frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def _build_sample_indices(self, vr: decord.VideoReader, total_frames: int, target_fps: Optional[float] = None) -> List[int]:
        if total_frames <= 0:
            return []

        effective_fps = self.fps if target_fps is None else float(target_fps)
        if effective_fps is not None:
            raw_fps = float(vr.get_avg_fps())
            if raw_fps > 0:
                sample_total = max(1, int(np.ceil(total_frames / raw_fps * effective_fps)))
                sample_total = min(sample_total, total_frames)
                if self.max_num_frames > 0:
                    sample_total = min(sample_total, self.max_num_frames)
                return np.linspace(0, total_frames - 1, sample_total, dtype=int).tolist()
            return list(range(total_frames))

        if self.max_num_frames > 0 and total_frames > self.max_num_frames:
            return np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int).tolist()

        return list(range(total_frames))

    def _normalize_past_key_values(self, past_key_values):
        if past_key_values is None:
            return []

        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        normalized = []
        for layer_cache in past_key_values:
            if hasattr(layer_cache, "to_legacy_cache"):
                layer_cache = layer_cache.to_legacy_cache()

            if isinstance(layer_cache, tuple) and len(layer_cache) == 2 and torch.is_tensor(layer_cache[0]):
                normalized.append(layer_cache)
                continue

            if isinstance(layer_cache, (list, tuple)) and len(layer_cache) == 1 and isinstance(layer_cache[0], tuple):
                normalized.append(layer_cache[0])
                continue

            raise ValueError(f"Unexpected layer cache structure: type={type(layer_cache)}")

        return normalized

    def _prepare_stream_prompt_tokens(self, context: str, reference_frame: Image.Image):
        probe_visual = {
            "type": "video",
            "video": [reference_frame],
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        probe_message = self._build_message(context, [probe_visual])
        probe_text = self.processor.apply_chat_template(probe_message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([probe_message])
        probe_inputs = self.processor(text=[probe_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        input_ids = probe_inputs["input_ids"][0]

        # Qwen2.5-VL special token ids from tokenizer added vocab.
        video_token_id = 151656  # <|video_pad|>
        video_positions = (input_ids == video_token_id).nonzero(as_tuple=False).flatten()
        if video_positions.numel() == 0:
            raise ValueError("Failed to locate <|video_pad|> region for streaming prompt split")

        first_pad = int(video_positions[0].item())
        last_pad = int(video_positions[-1].item())
        pre_tokens = input_ids[:first_pad].unsqueeze(0)
        post_tokens = input_ids[last_pad + 1 :].unsqueeze(0)
        return pre_tokens, post_tokens

    def _encode_single_frame_embedding(self, frame: Image.Image):
        batch = self.processor(text=[""], images=[frame.convert("RGB")], padding=True, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.model.visual.dtype)
        image_grid_thw = batch["image_grid_thw"].to(self.device)

        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        grid = image_grid_thw[0].tolist()
        token_count = int(grid[0] * grid[1] * grid[2])
        return image_embeds[:token_count].unsqueeze(0)

    def _cache_seq_len(self, cache_obj) -> int:
        if hasattr(cache_obj, "get_seq_length"):
            return int(cache_obj.get_seq_length())
        legacy = self._normalize_past_key_values(cache_obj)
        if not legacy:
            return 0
        return int(legacy[0][0].size(2))

    def _trim_dynamic_cache_visual_prefix(self, cache_obj, prefix_len: int, drop_tokens: int):
        if drop_tokens <= 0:
            return cache_obj

        legacy = self._normalize_past_key_values(cache_obj)
        trimmed = []
        for key_states, value_states in legacy:
            seq_len = key_states.size(2)
            drop_end = min(seq_len, prefix_len + drop_tokens)
            if drop_end <= prefix_len:
                trimmed.append((key_states, value_states))
                continue

            trimmed_key = torch.cat([key_states[..., :prefix_len, :], key_states[..., drop_end:, :]], dim=2)
            trimmed_value = torch.cat([value_states[..., :prefix_len, :], value_states[..., drop_end:, :]], dim=2)
            trimmed.append((trimmed_key, trimmed_value))

        return DynamicCache.from_legacy_cache(trimmed)

    def _clone_cache(self, cache_obj):
        legacy = self._normalize_past_key_values(cache_obj)
        if not legacy:
            return cache_obj

        cloned = []
        for key_states, value_states in legacy:
            cloned.append((key_states.clone(), value_states.clone()))

        return DynamicCache.from_legacy_cache(cloned)

    def _decode_with_past_key_values(self, past_key_values, post_tokens: torch.Tensor, current_gen_kwargs: dict):
        if post_tokens.size(1) == 0:
            eval_logger.warning(f"[{self.log_prefix} kv] empty post_tokens encountered; fallback to one pad token for decoding bootstrap")
            post_tokens = torch.full(
                (1, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )

        def _sample_next_token(logits: torch.Tensor) -> torch.Tensor:
            if not current_gen_kwargs["do_sample"]:
                return torch.argmax(logits, dim=-1)

            temperature = current_gen_kwargs.get("temperature")
            top_p = current_gen_kwargs.get("top_p")

            if temperature is not None and temperature > 0:
                logits = logits / float(temperature)

            probs = torch.softmax(logits, dim=-1)
            if top_p is not None and 0 < float(top_p) < 1:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs > float(top_p)
                sorted_mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                return sorted_indices.gather(-1, sampled_idx).squeeze(-1)

            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        if int(current_gen_kwargs.get("num_beams", 1)) != 1:
            eval_logger.warning(f"[{self.log_prefix} kv] num_beams>1 is not supported in incremental manual decode, falling back to greedy/sampling with num_beams=1")

        out = self.model(
            input_ids=post_tokens,
            inputs_embeds=None,
            attention_mask=None,
            position_ids=None,
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values,
            output_attentions=False,
            output_hidden_states=False,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
        pred = _sample_next_token(logits)
        generated = [pred]

        max_new_tokens = int(current_gen_kwargs["max_new_tokens"])
        for _ in range(max_new_tokens - 1):
            if int(pred.item()) == int(self.tokenizer.eos_token_id):
                break
            out = self.model(
                input_ids=pred[:, None],
                inputs_embeds=None,
                attention_mask=None,
                position_ids=None,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            pred = _sample_next_token(logits)
            generated.append(pred)

        output_ids = torch.stack(generated, dim=1)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _generate_with_incremental_kv(
        self,
        context: str,
        sampled_frames: List[Image.Image],
        current_gen_kwargs: dict,
        chunk_idx: int,
        query_frame_indices: Optional[List[int]] = None,
    ):
        if not sampled_frames:
            return "", 0, 0, 0

        with torch.inference_mode():
            pre_tokens, post_tokens = self._prepare_stream_prompt_tokens(context, sampled_frames[0])
            pre_tokens = pre_tokens.to(self.device)
            post_tokens = post_tokens.to(self.device)

            prefix_out = self.model(
                input_ids=pre_tokens,
                inputs_embeds=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            prefix_cache = self._normalize_past_key_values(prefix_out.past_key_values)
            prefix_len = int(prefix_cache[0][0].size(2)) if prefix_cache else 0

            past_key_values = prefix_out.past_key_values
            frame_token_lengths: deque = deque()
            evicted_frames = 0
            query_count = 0
            window_answers: List[str] = []
            streaming_answers: List[str] = []
            query_index_set = set(int(idx) for idx in (query_frame_indices or []) if int(idx) >= 0)
            window_size = max(1, int(self.sensory_window_size))

            for frame_idx, frame in enumerate(sampled_frames):
                prev_seq_len = self._cache_seq_len(past_key_values)
                frame_embeds = self._encode_single_frame_embedding(frame)
                frame_input_ids = torch.full(
                    (frame_embeds.size(0), frame_embeds.size(1)),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )

                frame_out = self.model(
                    input_ids=frame_input_ids,
                    inputs_embeds=frame_embeds,
                    attention_mask=None,
                    position_ids=None,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                past_key_values = frame_out.past_key_values
                new_seq_len = self._cache_seq_len(past_key_values)
                frame_tokens = max(0, int(new_seq_len - prev_seq_len))
                frame_token_lengths.append(frame_tokens)

                # FIFO sliding: when the window overflows, drop only the oldest frame's KV.
                if len(frame_token_lengths) > window_size:
                    drop_tokens = frame_token_lengths.popleft()
                    evicted_frames += 1
                    if drop_tokens > 0:
                        past_key_values = self._trim_dynamic_cache_visual_prefix(
                            past_key_values,
                            prefix_len=prefix_len,
                            drop_tokens=drop_tokens,
                        )

                if query_frame_indices is not None and (frame_idx in query_index_set or frame_idx == len(sampled_frames) - 1):
                    query_cache = self._clone_cache(past_key_values)
                    streaming_answer = self._decode_with_past_key_values(query_cache, post_tokens, current_gen_kwargs)
                    streaming_answers.append(streaming_answer)
                    query_count += 1
                elif self.stream_query_mode == "frame":
                    query_cache = self._clone_cache(past_key_values)
                    window_answer = self._decode_with_past_key_values(query_cache, post_tokens, current_gen_kwargs)
                    window_answers.append(window_answer)
                    query_count += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if len(frame_token_lengths) == 0:
                return "", 0, 0, 0

            if query_frame_indices is not None:
                chunk_ans = self._format_streaming_count_outputs(streaming_answers)
            elif self.stream_query_mode == "chunk":
                query_cache = self._clone_cache(past_key_values)
                chunk_ans = self._decode_with_past_key_values(query_cache, post_tokens, current_gen_kwargs)
                query_count = 1
            else:
                chunk_ans = self._aggregate_window_answers(window_answers)
            kept_frames = len(frame_token_lengths)
            return chunk_ans, kept_frames, evicted_frames, query_count

    def _format_streaming_count_outputs(self, answers: List[str]) -> str:
        values = []
        for answer in answers:
            clean = parse_reasoning_model_answer(answer).strip()
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", clean)
            if match:
                try:
                    values.append(float(match.group(0)))
                    continue
                except ValueError:
                    pass
            values.append(0.0)
        return json.dumps(values)

    def _encode_stream_video_features(self, video_paths: List[str]):
        sampled_frames = []
        chunk_frame_counts = []
        for video_path in video_paths:
            chunk_frames = self._sample_video_frames(video_path)
            sampled_frames.extend(chunk_frames)
            chunk_frame_counts.append(len(chunk_frames))

        if not sampled_frames:
            return None, [], []

        if self.max_image_size is not None:
            resized_frames = []
            for frame in sampled_frames:
                width, height = frame.size
                longest_edge = max(width, height)
                if longest_edge > self.max_image_size:
                    scale = self.max_image_size / float(longest_edge)
                    resized_frames.append(
                        frame.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.BICUBIC)
                    )
                else:
                    resized_frames.append(frame)
            sampled_frames = resized_frames

        frame_features = []
        token_lengths = []

        sampled_frames_rgb = [frame.convert("RGB") for frame in sampled_frames]
        with torch.inference_mode():
            for batch_start in range(0, len(sampled_frames_rgb), self.stream_visual_micro_batch_size):
                frame_batch = sampled_frames_rgb[batch_start : batch_start + self.stream_visual_micro_batch_size]
                batch = self.processor(text=[""] * len(frame_batch), images=frame_batch, padding=True, return_tensors="pt")
                pixel_values = batch["pixel_values"].to(self.device, dtype=self.model.visual.dtype)
                image_grid_thw = batch["image_grid_thw"].to(self.device)

                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)

                cursor = 0
                for grid in image_grid_thw.tolist():
                    token_count = int(grid[0] * grid[1] * grid[2])
                    # Keep encoded frame features on CPU to avoid accumulating all visual embeddings on GPU.
                    frame_features.append(image_embeds[cursor : cursor + token_count].unsqueeze(0).to("cpu"))
                    token_lengths.append(token_count)
                    cursor += token_count

                del batch, pixel_values, image_grid_thw, image_embeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if token_lengths:
            eval_logger.info(
                f"[{self.log_prefix} visual] frames={len(token_lengths)} max_frame_tokens={max(token_lengths)} avg_frame_tokens={(sum(token_lengths) / len(token_lengths)):.1f} micro_batch={self.stream_visual_micro_batch_size}"
            )

        return frame_features, token_lengths, chunk_frame_counts

    def _run_streaming_video(
        self,
        context: str,
        static_visuals: List[dict],
        video_paths: List[str],
        current_gen_kwargs: dict,
        until: List[str],
        query_frame_indices: Optional[List[int]] = None,
    ):
        if static_visuals:
            raise NotImplementedError(f"[{self.log_prefix}] streaming sw currently supports video-only requests")

        if not video_paths:
            return ""

        total_generate_elapsed = 0.0
        chunk_answers = []

        if query_frame_indices is not None:
            sampled_frames = []
            chunk_frame_counts = []
            for chunk_path in video_paths:
                try:
                    vr = decord.VideoReader(chunk_path)
                except Exception as exc:
                    eval_logger.warning(f"[{self.log_prefix} chunk] failed to open video={chunk_path} err={exc}")
                    continue

                total_frames = len(vr)
                if total_frames <= 0:
                    continue

                sample_indices = self._build_sample_indices(vr, total_frames, target_fps=self.fps)
                chunk_sampled = 0
                for frame_idx in sample_indices:
                    try:
                        frame = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
                    except Exception as exc:
                        eval_logger.warning(f"[{self.log_prefix} chunk] skip unreadable frame. video={chunk_path} frame_idx={frame_idx} err={exc}")
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
                    chunk_sampled += 1
                chunk_frame_counts.append(chunk_sampled)

            if not sampled_frames:
                return ""

            generate_start = time.perf_counter()
            answer, kept_frames, evicted_frames, query_count = self._generate_with_incremental_kv(
                context,
                sampled_frames,
                current_gen_kwargs,
                chunk_idx=0,
                query_frame_indices=query_frame_indices,
            )
            total_generate_elapsed = time.perf_counter() - generate_start
            eval_logger.info(
                f"[{self.log_prefix} stream] chunks={len(video_paths)} chunk_frames={chunk_frame_counts} total_frames={len(sampled_frames)} kept_frames={kept_frames} dropped_frames={evicted_frames} queries={query_count} query_mode=vsc_query_times input_fps={self.fps} kv_mode=frame_stream generate_s={total_generate_elapsed:.3f}"
            )
            return answer

        for chunk_idx, chunk_path in enumerate(video_paths):
            try:
                vr = decord.VideoReader(chunk_path)
            except Exception as exc:
                eval_logger.warning(f"[{self.log_prefix} chunk] failed to open video={chunk_path} err={exc}")
                continue

            total_frames = len(vr)
            if total_frames <= 0:
                continue

            # Strictly follow configured video_fps for streaming sampling.
            sample_indices = self._build_sample_indices(vr, total_frames, target_fps=self.fps)
            if not sample_indices:
                continue

            chunk_generate_elapsed = 0.0
            sampled_count = 0
            sampled_frames = []

            for frame_idx in sample_indices:
                try:
                    frame = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
                except Exception as exc:
                    eval_logger.warning(
                        f"[{self.log_prefix} chunk] skip unreadable frame. video={chunk_path} frame_idx={frame_idx} err={exc}"
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
                sampled_count += 1

            if sampled_count == 0:
                continue

            generate_start = time.perf_counter()
            try:
                chunk_ans, kept_frames, evicted_frames, query_count = self._generate_with_incremental_kv(
                    context,
                    sampled_frames,
                    current_gen_kwargs,
                    chunk_idx,
                )
            except ValueError as exc:
                if "nframes should in interval" in str(exc):
                    eval_logger.warning(
                        f"[{self.log_prefix} chunk] skip invalid frame-stream chunk. video={chunk_path} err={exc}"
                    )
                    continue
                raise

            generate_elapsed = time.perf_counter() - generate_start
            chunk_generate_elapsed += generate_elapsed
            total_generate_elapsed += generate_elapsed

            for term in until:
                if len(term) > 0:
                    chunk_ans = chunk_ans.split(term)[0]

            chunk_answers.append(chunk_ans)
            eval_logger.info(
                f"[{self.log_prefix} chunk] chunk_index={chunk_idx + 1}/{len(video_paths)} chunk_frames={sampled_count} kept_frames={kept_frames} dropped_frames={evicted_frames} queries={query_count} query_mode={self.stream_query_mode} input_fps={self.fps} kv_mode=frame_stream generate_s={chunk_generate_elapsed:.3f}"
            )

        if not chunk_answers:
            return ""
        answer = self._aggregate_chunk_answers(chunk_answers)
        eval_logger.info(
            f"[{self.log_prefix} stream] chunks={len(chunk_answers)} total_generate_s={total_generate_elapsed:.3f} avg_generate_s={(total_generate_elapsed / len(chunk_answers)):.3f}"
            if chunk_answers
            else f"[{self.log_prefix} stream] chunks=0 total_generate_s={total_generate_elapsed:.3f}"
        )
        return answer

    def _build_message(self, context: str, visuals: List[dict]):
        message = [{"role": "system", "content": self.system_prompt}]
        if self.interleave_visuals is False:
            message.append(
                {
                    "role": "user",
                    "content": visuals + [{"type": "text", "text": context}],
                }
            )
        else:
            image_placeholders = re.findall(r"<image \d+>", context)
            content_parts = []
            text_parts = re.split(r"<image \d+>", context)
            if text_parts[0]:
                content_parts.append({"type": "text", "text": text_parts[0]})

            for i, placeholder in enumerate(image_placeholders):
                img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                image_idx = min(img_idx, len(visuals) - 1) if visuals else 0
                if visuals and image_idx < len(visuals):
                    content_parts.append(visuals[image_idx])
                if i + 1 < len(text_parts) and text_parts[i + 1]:
                    content_parts.append({"type": "text", "text": text_parts[i + 1]})

            message.append(
                {
                    "role": "user",
                    "content": content_parts,
                }
            )

        return message

    def _generate_for_messages(self, messages: List[list], current_gen_kwargs: dict, debug_label: Optional[str] = None):
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

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
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _aggregate_chunk_answers(self, answers: List[str]) -> str:
        votes = {}
        for ans in answers:
            key = self._extract_choice_label(ans)
            if key is not None:
                votes[key] = votes.get(key, 0) + 1

        if votes:
            return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        return ""

    def _extract_choice_label(self, answer: str) -> Optional[str]:
        clean = parse_reasoning_model_answer(answer).strip()
        m = re.match(r"^\s*([A-D])\b", clean, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1).upper()

    def _aggregate_window_answers(self, answers: List[str]) -> str:
        # Later windows usually include more complete context in streaming mode.
        weighted_votes = {}
        for idx, ans in enumerate(answers):
            key = self._extract_choice_label(ans)
            if key is None:
                continue

            weight = idx + 1
            weighted_votes[key] = weighted_votes.get(key, 0) + weight

        if weighted_votes:
            return sorted(weighted_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        return ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": int(os.getenv("VSC_MAX_NEW_TOKENS", "64") or "64"),
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            answers = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                static_visuals = []
                stream_video_paths = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                            stream_video_paths.append(visual)
                        elif isinstance(visual, Image.Image):
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            static_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                sample = self.task_dict[task][split][doc_id[i]]
                use_vsc_query_times = str(os.getenv("VSC_USE_QUERY_TIMES", "0") or "0").lower() in {"1", "true", "yes"}
                query_frame_indices = (
                    sample.get("query_times", None)
                    if use_vsc_query_times and hasattr(sample, "get") and sample.get("answers", None) is not None
                    else None
                )

                if stream_video_paths:
                    answers.append(self._run_streaming_video(context, static_visuals, stream_video_paths, current_gen_kwargs, until, query_frame_indices=query_frame_indices))
                else:
                    message = self._build_message(context, static_visuals)
                    ans = self._generate_for_messages([message], current_gen_kwargs, debug_label="non_chunk_visual")[0]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers.append(ans)

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
