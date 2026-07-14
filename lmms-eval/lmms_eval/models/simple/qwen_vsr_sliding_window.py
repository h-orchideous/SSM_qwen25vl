import base64
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
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


LOG_PREFIX = "qwen_vsr_sw"


@dataclass
class EncodedFrame:
    vision_emb: torch.Tensor
    grid_thw: torch.Tensor
    frame_index: int
    chunk_index: int


@register_model("qwen_vsr_sliding_window")
class Qwen_VSR_SlidingWindow(lmms):
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
        self.sliding_window_stride = int(sliding_window_stride) if sliding_window_stride is not None else 1
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen_VSR_SlidingWindow")

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

    def _special_token_id(self, token: str, fallback: Optional[int] = None) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            if fallback is None:
                raise ValueError(f"Failed to resolve special token id for {token}")
            return int(fallback)
        return int(token_id)

    def _get_text_embedding_layer(self):
        text_model = getattr(self.model, "model", self.model)
        if hasattr(text_model, "embed_tokens"):
            return text_model.embed_tokens
        return self.model.get_input_embeddings()

    @torch.inference_mode()
    def _encode_vision_frames(self, frames: List[Image.Image]) -> EncodedFrame:
        if not frames:
            raise ValueError("_encode_vision_frames requires at least one frame")

        batch = self.processor(
            text=[""],
            images=[frame.convert("RGB") for frame in frames],
            padding=True,
            return_tensors="pt",
        )
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.model.visual.dtype)
        image_grid_thw = batch["image_grid_thw"].to(self.device)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        encoded = EncodedFrame(
            vision_emb=image_embeds.detach(),
            grid_thw=image_grid_thw.detach(),
            frame_index=-1,
            chunk_index=-1,
        )
        del batch, pixel_values, image_grid_thw, image_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return encoded

    def _combine_encoded_window(self, encoded_window: List[EncodedFrame]):
        combined_embeds = torch.cat([item.vision_emb.to(self.device) for item in encoded_window], dim=0)
        combined_grid_thw = torch.cat([item.grid_thw.to(self.device) for item in encoded_window], dim=0)
        return combined_embeds, combined_grid_thw

    def _build_cached_vision_input_ids(self, context: str, num_vision_tokens: int) -> torch.Tensor:
        im_start_id = self._special_token_id("<|im_start|>", 151644)
        im_end_id = self._special_token_id("<|im_end|>", 151645)
        vision_start_id = self._special_token_id("<|vision_start|>", getattr(self.model.config, "vision_start_token_id", 151652))
        vision_end_id = self._special_token_id("<|vision_end|>", getattr(self.model.config, "vision_end_token_id", 151653))
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))

        input_ids_list: List[int] = []
        if self.system_prompt:
            input_ids_list.append(im_start_id)
            input_ids_list.extend(self.tokenizer.encode("system\n", add_special_tokens=False))
            input_ids_list.extend(self.tokenizer.encode(self.system_prompt, add_special_tokens=False))
            input_ids_list.append(im_end_id)
            input_ids_list.extend(self.tokenizer.encode("\n", add_special_tokens=False))

        input_ids_list.append(im_start_id)
        input_ids_list.extend(self.tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(vision_start_id)
        input_ids_list.extend([image_token_id] * int(num_vision_tokens))
        input_ids_list.append(vision_end_id)
        input_ids_list.extend(self.tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(self.tokenizer.encode(context, add_special_tokens=False))
        input_ids_list.append(im_end_id)
        input_ids_list.extend(self.tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.append(im_start_id)
        input_ids_list.extend(self.tokenizer.encode("assistant\n", add_special_tokens=False))
        return torch.tensor([input_ids_list], dtype=torch.long, device=self.device)

    def _get_rope_index_for_cached_vision(self, input_ids, image_grid_thw, attention_mask):
        try:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        except TypeError:
            try:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    None,
                    attention_mask,
                )
            except TypeError:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    attention_mask,
                )
        self.model.rope_deltas = rope_deltas
        return position_ids

    @torch.inference_mode()
    def _generate_with_cached_vision(self, context: str, cached_embeds: torch.Tensor, cached_grid_thw: torch.Tensor, current_gen_kwargs: dict) -> str:
        if cached_embeds.numel() == 0:
            return ""

        input_ids = self._build_cached_vision_input_ids(context, int(cached_embeds.shape[0]))
        attention_mask = torch.ones_like(input_ids)
        embed_layer = self._get_text_embedding_layer()
        inputs_embeds = embed_layer(input_ids)

        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        expected = int(image_mask[..., 0].sum().item())
        if expected != int(cached_embeds.shape[0]):
            raise ValueError(f"cached vision token count mismatch: prompt={expected} cached={cached_embeds.shape[0]}")
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask,
            cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )
        position_ids = self._get_rope_index_for_cached_vision(input_ids, cached_grid_thw.to(inputs_embeds.device), attention_mask)

        cont = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=current_gen_kwargs["do_sample"],
            temperature=current_gen_kwargs["temperature"],
            top_p=current_gen_kwargs["top_p"],
            num_beams=current_gen_kwargs["num_beams"],
            max_new_tokens=current_gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
        )

        prompt_len = int(input_ids.shape[1])
        if cont.shape[1] > prompt_len:
            cont = cont[:, prompt_len:]
        return self.processor.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _generate_with_encoded_window(self, context: str, encoded_window: List[EncodedFrame], current_gen_kwargs: dict) -> str:
        if not encoded_window:
            return ""
        cached_embeds, cached_grid_thw = self._combine_encoded_window(encoded_window)
        return self._generate_with_cached_vision(context, cached_embeds, cached_grid_thw, current_gen_kwargs)

    def _generate_with_recent_frames(self, context: str, recent_frames: List[Image.Image], current_gen_kwargs: dict) -> str:
        if not recent_frames:
            return ""
        encoded = self._encode_vision_frames(recent_frames)
        return self._generate_with_cached_vision(context, encoded.vision_emb, encoded.grid_thw, current_gen_kwargs)

    def _run_streaming_video(self, context: str, static_visuals: List[dict], video_paths: List[str], current_gen_kwargs: dict, until: List[str]):
        if static_visuals:
            raise NotImplementedError(f"[{self.log_prefix}] simplestream sw currently supports video-only requests")

        if not video_paths:
            return ""

        window_size = max(1, int(self.sensory_window_size))
        encoded_window = deque(maxlen=window_size)
        sampled_count = 0
        evicted_frames = 0
        chunk_count = 0

        for chunk_idx, chunk_path in enumerate(video_paths):
            try:
                vr = decord.VideoReader(chunk_path)
            except Exception as exc:
                eval_logger.warning(f"[{self.log_prefix} chunk] failed to open video={chunk_path} err={exc}")
                continue

            total_frames = len(vr)
            if total_frames <= 0:
                continue

            sample_indices = self._build_sample_indices(vr, total_frames, target_fps=self.fps)
            if not sample_indices:
                continue

            chunk_sampled = 0
            chunk_dropped = 0
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

                encoded_frame = self._encode_vision_frames([frame])
                encoded_frame.frame_index = int(frame_idx)
                encoded_frame.chunk_index = int(chunk_idx)

                if len(encoded_window) == window_size:
                    evicted_frames += 1
                    chunk_dropped += 1
                encoded_window.append(encoded_frame)
                sampled_count += 1
                chunk_sampled += 1

            if chunk_sampled > 0:
                chunk_count += 1
                eval_logger.info(
                    f"[{self.log_prefix} chunk] chunk_index={chunk_idx + 1}/{len(video_paths)} chunk_frames={chunk_sampled} kept_frames={len(encoded_window)} dropped_frames={chunk_dropped} queries=0 query_mode=final input_fps={self.fps} window_mode=simplestream_encoded"
                )

        if not encoded_window:
            return ""

        generate_start = time.perf_counter()
        answer = self._generate_with_encoded_window(context, list(encoded_window), current_gen_kwargs)
        generate_elapsed = time.perf_counter() - generate_start

        for term in until:
            if len(term) > 0:
                answer = answer.split(term)[0]

        eval_logger.info(
            f"[{self.log_prefix} stream] chunks={chunk_count} sampled_frames={sampled_count} kept_frames={len(encoded_window)} dropped_frames={evicted_frames} queries=1 window_mode=simplestream_encoded generate_s={generate_elapsed:.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
                "max_new_tokens": 32768,
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

                if stream_video_paths:
                    answers.append(self._run_streaming_video(context, static_visuals, stream_video_paths, current_gen_kwargs, until))
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
