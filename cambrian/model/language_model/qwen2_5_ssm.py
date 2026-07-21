import types
import weakref
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
except Exception:  # pragma: no cover - compatibility across transformers versions
    Qwen2_5_VLCausalLMOutputWithPast = CausalLMOutputWithPast

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional across transformers versions
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - optional across transformers versions
    AutoModelForVision2Seq = None


class Qwen2_5SSMConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_ssm"
    ssm_enabled: bool = True
    ssm_sliding_window: Optional[int] = None
    ssm_prefix_len: int = 0
    ssm_training_step_size: Optional[int] = None
    ssm_training_chunk_size: Optional[int] = None
    ssm_frame_window: Optional[int] = None
    ssm_training_stage: str = "ssm"
    ssm_visual_encode_chunk_size: int = 1

    def __init__(self, *args, **kwargs):
        hidden_size = kwargs.get("hidden_size", 8192)
        num_attention_heads = kwargs.get("num_attention_heads", 64)
        head_dim = hidden_size // num_attention_heads

        kwargs.setdefault("vision_start_token_id", 151652)
        kwargs.setdefault("vision_end_token_id", 151653)
        kwargs.setdefault("image_token_id", 151655)
        kwargs.setdefault("video_token_id", 151656)
        kwargs.setdefault("rope_scaling", {"type": "mrope", "mrope_section": self._default_mrope_section(head_dim)})
        self.ssm_sliding_window = kwargs.pop("ssm_sliding_window", self.ssm_sliding_window)
        self.ssm_prefix_len = kwargs.pop("ssm_prefix_len", self.ssm_prefix_len)
        self.ssm_training_step_size = kwargs.pop("ssm_training_step_size", self.ssm_training_step_size)
        self.ssm_training_chunk_size = kwargs.pop("ssm_training_chunk_size", self.ssm_training_chunk_size)
        self.ssm_frame_window = kwargs.pop("ssm_frame_window", self.ssm_frame_window)
        self.ssm_training_stage = kwargs.pop("ssm_training_stage", self.ssm_training_stage)
        self.ssm_visual_encode_chunk_size = kwargs.pop(
            "ssm_visual_encode_chunk_size",
            self.ssm_visual_encode_chunk_size,
        )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _default_mrope_section(head_dim: int):
        half_head_dim = max(1, int(head_dim) // 2)
        temporal = max(1, half_head_dim // 4)
        height = max(1, (half_head_dim - temporal) // 2)
        width = max(1, half_head_dim - temporal - height)

        total = temporal + height + width
        if total != half_head_dim:
            width = max(1, width + half_head_dim - total)

        return [temporal, height, width]


class Qwen2_5SSMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Native Qwen2.5-VL + SSM memory fusion.

    This class keeps the official HuggingFace Qwen2_5_VLForConditionalGeneration
    implementation for vision encoding, multimodal token replacement, RoPE,
    cache handling, and generation. It only wraps the language decoder layers
    (`self.model.layers`) so an attached `ssm_compressor` can fuse compressed
    long-term memory after each decoder self-attention.
    """

    config_class = Qwen2_5SSMConfig

    def __init__(self, config):
        super().__init__(config)
        self._active_ssm_compressor = None
        self._install_ssm_fusion_wrappers()

    def _install_ssm_fusion_wrappers(self) -> int:
        layers = getattr(getattr(self, "model", None), "layers", None)
        if layers is None:
            raise AttributeError("Could not find Qwen2.5-VL decoder layers at self.model.layers")

        patched = 0
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "_qwen25vl_ssm_original_forward"):
                continue

            layer._qwen25vl_ssm_original_forward = layer.forward
            layer._qwen25vl_ssm_parent_ref = weakref.ref(self)
            layer._qwen25vl_ssm_layer_idx = layer_idx

            def forward_with_ssm(self, *args, **kwargs):
                hidden_states = kwargs.get("hidden_states", args[0] if args else None)
                outputs = self._qwen25vl_ssm_original_forward(*args, **kwargs)

                parent_ref = getattr(self, "_qwen25vl_ssm_parent_ref", None)
                parent = parent_ref() if parent_ref is not None else None
                if parent is None:
                    return outputs

                compressor = getattr(parent, "_active_ssm_compressor", None)
                if (
                    hidden_states is None
                    or compressor is None
                    or not getattr(parent.config, "ssm_enabled", False)
                    or not compressor.has_memory(self._qwen25vl_ssm_layer_idx)
                ):
                    return outputs

                fused_hidden = compressor.fuse(
                    layer_idx=self._qwen25vl_ssm_layer_idx,
                    hidden_states=hidden_states,
                    attn_output=outputs[0],
                )
                return (fused_hidden,) + tuple(outputs[1:])

            layer.forward = types.MethodType(forward_with_ssm, layer)
            patched += 1

        return patched

    def forward(
        self,
        *args,
        ssm_compressor: Optional[object] = None,
        ssm_sliding_window: Optional[int] = None,
        ssm_prefix_len: Optional[int] = None,
        **kwargs,
    ):
        kwargs.pop("num_items_in_batch", None)
        if ssm_compressor is None:
            ssm_compressor = getattr(self, "ssm_compressor", None)

        previous_compressor = self._active_ssm_compressor
        previous_window = getattr(self, "_active_ssm_sliding_window", None)
        previous_prefix_len = getattr(self, "_active_ssm_prefix_len", None)
        previous_trainable_memory = getattr(self, "_active_ssm_trainable_memory", False)
        self._active_ssm_compressor = ssm_compressor
        self._active_ssm_sliding_window = ssm_sliding_window
        self._active_ssm_prefix_len = ssm_prefix_len
        self._maybe_init_generation_frame_window(kwargs)
        try:
            if (
                not args
                and self._should_use_ssm_training_forward(kwargs, ssm_compressor, ssm_sliding_window)
            ):
                return self._forward_ssm_sliding_window_train(
                    ssm_compressor=ssm_compressor,
                    ssm_sliding_window=ssm_sliding_window,
                    ssm_prefix_len=ssm_prefix_len,
                    **kwargs,
                )
            if (
                not args
                and self._should_use_ssm_generation_prefill_forward(kwargs, ssm_compressor, ssm_sliding_window)
            ):
                return self._forward_ssm_sliding_window_generation_prefill(
                    ssm_compressor=ssm_compressor,
                    ssm_sliding_window=ssm_sliding_window,
                    ssm_prefix_len=ssm_prefix_len,
                    **kwargs,
                )
            if (
                not args
                and self.training
                and kwargs.get("labels", None) is not None
                and kwargs.get("inputs_embeds", None) is None
                and kwargs.get("input_ids", None) is not None
                and (kwargs.get("pixel_values", None) is not None or kwargs.get("pixel_values_videos", None) is not None)
            ):
                return self._forward_raw_window_train(**kwargs)
            return super().forward(*args, **kwargs)
        finally:
            self._active_ssm_compressor = previous_compressor
            self._active_ssm_sliding_window = previous_window
            self._active_ssm_prefix_len = previous_prefix_len
            self._active_ssm_trainable_memory = previous_trainable_memory

    def _forward_raw_window_train(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids.device != self.model.embed_tokens.weight.device:
            input_ids = input_ids.to(self.model.embed_tokens.weight.device)
        embed_device = input_ids.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(embed_device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(embed_device)

        inputs_embeds = self._build_qwen25vl_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        loss = None
        logits = None
        if labels is not None:
            shift_labels = labels[..., 1:].to(hidden_states.device)
            valid_mask = shift_labels != -100
            if bool(valid_mask.any()):
                selected_hidden = hidden_states[..., :-1, :][valid_mask]
                selected_logits = self.lm_head(selected_hidden).float()
                loss = CrossEntropyLoss()(selected_logits, shift_labels[valid_mask])
            else:
                loss = hidden_states.sum() * 0.0
        else:
            logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def _maybe_init_generation_frame_window(self, kwargs):
        if self.training:
            return
        input_ids = kwargs.get("input_ids", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        frame_window = getattr(self.config, "ssm_frame_window", None)
        if input_ids is None or video_grid_thw is None or frame_window is None or int(frame_window) <= 0:
            if input_ids is not None:
                self._clear_generation_frame_window()
            return

        frame_spans = self._build_training_frame_spans(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            frame_window=int(frame_window),
        )
        if not frame_spans:
            return
        self._generation_frame_spans = frame_spans
        self._generation_frame_window = int(frame_window)
        self._generation_frame_prefill_len = int(input_ids.shape[-1])
        self._generation_frame_trimmed = False
        self._generation_tail_abs_start = int(frame_spans[0][0])

    def _clear_generation_frame_window(self):
        self._generation_frame_spans = None
        self._generation_frame_window = None
        self._generation_frame_prefill_len = None
        self._generation_frame_trimmed = False
        self._generation_tail_abs_start = None
        self._generation_trimmed_prefix_len = None
        self._generation_kept_len = None

    def _should_use_ssm_training_forward(self, kwargs, compressor, ssm_sliding_window) -> bool:
        if not self.training or kwargs.get("labels", None) is None:
            return False
        stage = str(getattr(self.config, "ssm_training_stage", "ssm")).lower()
        if stage not in {"sw", "ssm"}:
            return False
        if stage == "ssm" and (compressor is None or not getattr(self.config, "ssm_enabled", False)):
            return False
        keep_tokens = ssm_sliding_window
        if keep_tokens is None:
            keep_tokens = getattr(self.config, "ssm_sliding_window", None)
        return keep_tokens is not None and int(keep_tokens) > 0

    def _should_use_ssm_generation_prefill_forward(self, kwargs, compressor, ssm_sliding_window) -> bool:
        if self.training or kwargs.get("labels", None) is not None:
            return False
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is not None and (self._cache_seq_len(past_key_values) or 0) > 0:
            return False
        if kwargs.get("inputs_embeds", None) is not None:
            return False
        if kwargs.get("input_ids", None) is None:
            return False
        if kwargs.get("use_cache", True) is False:
            return False
        if kwargs.get("video_grid_thw", None) is None and kwargs.get("image_grid_thw", None) is None:
            return False

        frame_window = getattr(self.config, "ssm_frame_window", None)
        if frame_window is None or int(frame_window) <= 0:
            return False

        keep_tokens = ssm_sliding_window
        if keep_tokens is None:
            keep_tokens = getattr(self.config, "ssm_sliding_window", None)
        if keep_tokens is None or int(keep_tokens) <= 0:
            return False

        stage = str(getattr(self.config, "ssm_training_stage", "ssm")).lower()
        if stage not in {"sw", "ssm"}:
            return False
        if stage == "ssm" and (compressor is None or not getattr(self.config, "ssm_enabled", False)):
            return False
        return True

    def _build_qwen25vl_inputs_embeds(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self._encode_visual_in_chunks(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device),
                image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self._encode_visual_in_chunks(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device),
                video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        return inputs_embeds

    def _visual_requires_grad(self) -> bool:
        if getattr(self.config, "ssm_freeze_visual", False):
            return False
        return any(param.requires_grad for param in self.visual.parameters())

    def _encode_visual(self, pixel_values, grid_thw):
        if self._visual_requires_grad():
            return self.visual(pixel_values, grid_thw=grid_thw)

        with torch.no_grad():
            embeds = self.visual(pixel_values, grid_thw=grid_thw)
        return embeds.detach()

    def _encode_visual_in_chunks(self, pixel_values, grid_thw):
        chunk_size = int(getattr(self.config, "ssm_visual_encode_chunk_size", 1) or 1)
        if grid_thw is None or chunk_size <= 0 or int(grid_thw.shape[0]) <= chunk_size:
            return self._encode_visual(pixel_values, grid_thw)

        embeds = []
        offsets = [0]
        for grid in grid_thw:
            offsets.append(offsets[-1] + int(grid.prod().item()))

        for start in range(0, int(grid_thw.shape[0]), chunk_size):
            end = min(start + chunk_size, int(grid_thw.shape[0]))
            pixel_start = offsets[start]
            pixel_end = offsets[end]
            embeds.append(self._encode_visual(pixel_values[pixel_start:pixel_end], grid_thw[start:end]))
        return torch.cat(embeds, dim=0)

    @staticmethod
    def _grid_offsets(grid_thw):
        offsets = [0]
        if grid_thw is None:
            return offsets
        for grid in grid_thw:
            offsets.append(offsets[-1] + int(grid.prod().item()))
        return offsets

    def _build_stream_visual_segments(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        if input_ids is None or input_ids.shape[0] != 1:
            return [], []

        token_ids = input_ids[0]
        image_segments = []
        image_positions = torch.nonzero(token_ids == self.config.image_token_id, as_tuple=False).flatten().tolist()
        image_offsets = self._grid_offsets(image_grid_thw)
        cursor = 0
        if image_grid_thw is not None and image_positions:
            for image_idx in range(int(image_grid_thw.shape[0])):
                token_count = self._image_token_count(image_grid_thw[image_idx])
                positions = image_positions[cursor : cursor + token_count]
                cursor += token_count
                if len(positions) != token_count:
                    break
                image_segments.append(
                    {
                        "token_start": int(positions[0]),
                        "token_end": int(positions[-1]) + 1,
                        "grid_start": image_idx,
                        "grid_end": image_idx + 1,
                        "grid": image_grid_thw[image_idx : image_idx + 1],
                        "pixel_start": image_offsets[image_idx],
                        "pixel_end": image_offsets[image_idx + 1],
                    }
                )

        video_segments = []
        video_positions = torch.nonzero(token_ids == self.config.video_token_id, as_tuple=False).flatten().tolist()
        video_offsets = self._grid_offsets(video_grid_thw)
        cursor = 0
        if video_grid_thw is not None and video_positions:
            for video_idx in range(int(video_grid_thw.shape[0])):
                grid = video_grid_thw[video_idx]
                frame_tokens = self._video_frame_token_count(grid)
                num_frames = int(grid[0].item())
                video_token_count = frame_tokens * num_frames
                positions = video_positions[cursor : cursor + video_token_count]
                cursor += video_token_count
                if len(positions) != video_token_count or frame_tokens <= 0 or num_frames <= 0:
                    break
                pixel_base = video_offsets[video_idx]
                for frame_idx in range(num_frames):
                    frame_token_start = frame_idx * frame_tokens
                    token_start = int(positions[frame_token_start])
                    token_end = int(positions[frame_token_start + frame_tokens - 1]) + 1
                    pixel_start = pixel_base + frame_idx * frame_tokens
                    pixel_end = pixel_start + frame_tokens
                    video_segments.append(
                        {
                            "token_start": token_start,
                            "token_end": token_end,
                            "grid": torch.tensor(
                                [[1, int(grid[1].item()), int(grid[2].item())]],
                                dtype=grid.dtype,
                                device=grid.device,
                            ),
                            "pixel_start": pixel_start,
                            "pixel_end": pixel_end,
                        }
                    )
        return image_segments, video_segments

    def _build_stream_segment_inputs_embeds(
        self,
        input_ids,
        start: int,
        end: int,
        pixel_values=None,
        pixel_values_videos=None,
        image_segments=None,
        video_segments=None,
    ):
        segment_ids = input_ids[:, start:end]
        inputs_embeds = self.model.embed_tokens(segment_ids)
        embed_device = inputs_embeds.device
        embed_dtype = inputs_embeds.dtype

        for segment in image_segments or []:
            token_start = max(start, int(segment["token_start"]))
            token_end = min(end, int(segment["token_end"]))
            if token_end <= token_start or pixel_values is None:
                continue
            grid = segment.get("grid", None)
            if grid is None:
                grid = torch.empty(0)
            grid = grid.to(embed_device) if hasattr(grid, "to") else grid
            pixels = pixel_values[int(segment["pixel_start"]) : int(segment["pixel_end"])].to(embed_device)
            pixels = pixels.type(self.visual.dtype)
            image_embeds = self._encode_visual(pixels, grid).to(embed_device, embed_dtype)
            local_start = token_start - start
            local_end = token_end - start
            inputs_embeds[:, local_start:local_end, :] = image_embeds[: local_end - local_start].unsqueeze(0)

        for segment in video_segments or []:
            token_start = max(start, int(segment["token_start"]))
            token_end = min(end, int(segment["token_end"]))
            if token_end <= token_start or pixel_values_videos is None:
                continue
            grid = segment["grid"].to(embed_device)
            pixels = pixel_values_videos[int(segment["pixel_start"]) : int(segment["pixel_end"])].to(embed_device)
            pixels = pixels.type(self.visual.dtype)
            video_embeds = self._encode_visual(pixels, grid).to(embed_device, embed_dtype)
            local_start = token_start - start
            local_end = token_end - start
            inputs_embeds[:, local_start:local_end, :] = video_embeds[: local_end - local_start].unsqueeze(0)

        return inputs_embeds

    def _forward_ssm_sliding_window_train(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        ssm_compressor=None,
        ssm_sliding_window=None,
        ssm_prefix_len=None,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("SSM training forward requires input_ids or inputs_embeds")
        if past_key_values is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if getattr(self.model, "gradient_checkpointing", False):
            raise ValueError(
                "SSM sliding-window training requires use_cache=True, which is incompatible with "
                "gradient checkpointing. Disable gradient_checkpointing for this training mode."
            )
        keep_tokens = int(ssm_sliding_window if ssm_sliding_window is not None else self.config.ssm_sliding_window)
        prefix_len = int(ssm_prefix_len if ssm_prefix_len is not None else getattr(self.config, "ssm_prefix_len", 0))
        frame_window = getattr(self.config, "ssm_frame_window", None)
        frame_window = int(frame_window) if frame_window is not None else None
        step_size = (
            getattr(self.config, "ssm_training_step_size", None)
            or getattr(self.config, "ssm_training_chunk_size", None)
            or keep_tokens
        )
        step_size = max(1, int(step_size))
        stage = str(getattr(self.config, "ssm_training_stage", "ssm")).lower()
        use_ssm_memory = stage == "ssm" and ssm_compressor is not None and getattr(self.config, "ssm_enabled", False)

        if use_ssm_memory and hasattr(ssm_compressor, "reset"):
            ssm_compressor.reset()
        self._active_ssm_trainable_memory = use_ssm_memory

        if input_ids.device != self.model.embed_tokens.weight.device:
            input_ids = input_ids.to(self.model.embed_tokens.weight.device)
        embed_device = input_ids.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(embed_device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(embed_device)

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas

        seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        past = past_key_values if past_key_values is not None else DynamicCache()
        logits_chunks = [] if labels is None else None
        loss_sum = None
        loss_count = 0
        loss_fct = CrossEntropyLoss(reduction="sum") if labels is not None else None
        hidden_chunks = [] if output_hidden_states else None
        attn_chunks = [] if output_attentions else None
        frame_spans = self._build_training_frame_spans(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            image_grid_thw=image_grid_thw,
            frame_window=frame_window,
        )
        if frame_spans:
            prefix_len = max(prefix_len, int(frame_spans[0][0]))
        tail_abs_start = prefix_len
        image_segments, video_segments = self._build_stream_visual_segments(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        try:
            for start, end in self._iter_training_segments(seq_len, frame_spans, step_size):
                if inputs_embeds is not None:
                    chunk_embeds = inputs_embeds[:, start:end, :]
                else:
                    chunk_embeds = self._build_stream_segment_inputs_embeds(
                        input_ids=input_ids,
                        start=start,
                        end=end,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        image_segments=image_segments,
                        video_segments=video_segments,
                    )
                chunk_position_ids = position_ids[..., start:end] if position_ids is not None else None
                past_len = self._cache_seq_len(past) or 0
                local_cache_position = torch.arange(
                    past_len,
                    past_len + (end - start),
                    device=chunk_embeds.device,
                )
                local_attention_mask = None
                if attention_mask is not None:
                    local_attention_mask = torch.ones(
                        attention_mask.shape[0],
                        past_len + (end - start),
                        dtype=attention_mask.dtype,
                        device=chunk_embeds.device,
                    )

                outputs = self.model(
                    input_ids=None,
                    position_ids=chunk_position_ids,
                    attention_mask=local_attention_mask,
                    past_key_values=past,
                    inputs_embeds=chunk_embeds,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    cache_position=local_cache_position,
                )
                past = outputs.past_key_values
                logits = self.lm_head(outputs.last_hidden_state)
                if labels is None:
                    logits_chunks.append(logits)
                else:
                    shift_logits_parts = []
                    shift_label_parts = []
                    if end - start > 1:
                        shift_logits_parts.append(logits[:, :-1, :])
                        shift_label_parts.append(labels[:, start + 1 : end])
                    if end < seq_len:
                        shift_logits_parts.append(logits[:, -1:, :])
                        shift_label_parts.append(labels[:, end : end + 1])

                    if shift_logits_parts:
                        chunk_shift_logits = torch.cat(shift_logits_parts, dim=1)
                        chunk_shift_labels = torch.cat(shift_label_parts, dim=1).to(chunk_shift_logits.device)
                        valid_count = int((chunk_shift_labels != -100).sum().item())
                        if valid_count > 0:
                            chunk_loss = loss_fct(
                                chunk_shift_logits.float().reshape(-1, self.config.vocab_size),
                                chunk_shift_labels.reshape(-1),
                            )
                            loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
                            loss_count += valid_count
                if output_hidden_states:
                    hidden_chunks.append(outputs.hidden_states)
                if output_attentions:
                    attn_chunks.append(outputs.attentions)

                past, _, tail_abs_start = self.absorb_and_trim_past_key_values(
                    past_key_values=past,
                    compressor=ssm_compressor if use_ssm_memory else None,
                    keep_tokens=keep_tokens,
                    prefix_len=prefix_len,
                    current_seq_end=end,
                    frame_spans=frame_spans,
                    frame_window=frame_window,
                    tail_abs_start=tail_abs_start,
                )

            loss = None
            if labels is not None:
                if loss_sum is None:
                    loss = self.model.embed_tokens.weight.sum() * 0.0
                else:
                    loss = loss_sum / max(1, loss_count)
                logits = None
            else:
                logits = torch.cat(logits_chunks, dim=1)

            if not return_dict:
                output = (logits, past)
                return (loss,) + output if loss is not None else output

            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past,
                hidden_states=tuple(hidden_chunks) if hidden_chunks is not None else None,
                attentions=tuple(attn_chunks) if attn_chunks is not None else None,
                rope_deltas=self.rope_deltas,
            )
        finally:
            self._active_ssm_trainable_memory = False
            if use_ssm_memory and hasattr(ssm_compressor, "reset"):
                ssm_compressor.reset()

    def _forward_ssm_sliding_window_generation_prefill(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        ssm_compressor=None,
        ssm_sliding_window=None,
        ssm_prefix_len=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        keep_tokens = int(ssm_sliding_window if ssm_sliding_window is not None else self.config.ssm_sliding_window)
        prefix_len = int(ssm_prefix_len if ssm_prefix_len is not None else getattr(self.config, "ssm_prefix_len", 0))
        frame_window = int(getattr(self.config, "ssm_frame_window"))
        stage = str(getattr(self.config, "ssm_training_stage", "ssm")).lower()
        use_ssm_memory = stage == "ssm" and ssm_compressor is not None and getattr(self.config, "ssm_enabled", False)

        if input_ids.device != self.model.embed_tokens.weight.device:
            input_ids = input_ids.to(self.model.embed_tokens.weight.device)
        embed_device = input_ids.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(embed_device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(embed_device)

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas

        seq_len = input_ids.shape[1]
        past = DynamicCache()
        frame_spans = self._build_training_frame_spans(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            image_grid_thw=image_grid_thw,
            frame_window=frame_window,
        )
        if frame_spans:
            prefix_len = max(prefix_len, int(frame_spans[0][0]))
        tail_abs_start = prefix_len

        last_hidden_state = None
        hidden_chunks = [] if output_hidden_states else None
        attn_chunks = [] if output_attentions else None
        image_segments, video_segments = self._build_stream_visual_segments(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        for start, end in self._iter_training_segments(seq_len, frame_spans, keep_tokens):
            chunk_embeds = self._build_stream_segment_inputs_embeds(
                input_ids=input_ids,
                start=start,
                end=end,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_segments=image_segments,
                video_segments=video_segments,
            )
            chunk_position_ids = position_ids[..., start:end] if position_ids is not None else None
            past_len = self._cache_seq_len(past) or 0
            local_cache_position = torch.arange(
                past_len,
                past_len + (end - start),
                device=chunk_embeds.device,
            )
            local_attention_mask = None
            if attention_mask is not None:
                local_attention_mask = torch.ones(
                    attention_mask.shape[0],
                    past_len + (end - start),
                    dtype=attention_mask.dtype,
                    device=chunk_embeds.device,
                )

            outputs = self.model(
                input_ids=None,
                position_ids=chunk_position_ids,
                attention_mask=local_attention_mask,
                past_key_values=past,
                inputs_embeds=chunk_embeds,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=local_cache_position,
            )
            past = outputs.past_key_values
            last_hidden_state = outputs.last_hidden_state
            if output_hidden_states:
                hidden_chunks.append(outputs.hidden_states)
            if output_attentions:
                attn_chunks.append(outputs.attentions)

            past, _, tail_abs_start = self.absorb_and_trim_past_key_values(
                past_key_values=past,
                compressor=ssm_compressor if use_ssm_memory else None,
                keep_tokens=keep_tokens,
                prefix_len=prefix_len,
                current_seq_end=end,
                frame_spans=frame_spans,
                frame_window=frame_window,
                tail_abs_start=tail_abs_start,
            )

        if last_hidden_state is None:
            last_hidden_state = self.model.embed_tokens(input_ids[:, -1:])
        logits = self.lm_head(last_hidden_state[:, -1:, :])

        kept_len = self._cache_seq_len(past) or seq_len
        self._generation_frame_trimmed = True
        self._generation_tail_abs_start = tail_abs_start
        self._generation_trimmed_prefix_len = int(prefix_len)
        self._generation_kept_len = int(kept_len)

        if not return_dict:
            output = (logits, past)
            return output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past,
            hidden_states=tuple(hidden_chunks) if hidden_chunks is not None else None,
            attentions=tuple(attn_chunks) if attn_chunks is not None else None,
            rope_deltas=self.rope_deltas,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        stage = str(getattr(self.config, "ssm_training_stage", "ssm")).lower()
        compressor = getattr(self, "_active_ssm_compressor", None) or getattr(self, "ssm_compressor", None)
        keep_tokens = getattr(self, "_active_ssm_sliding_window", None)
        if keep_tokens is None:
            keep_tokens = getattr(self.config, "ssm_sliding_window", None)
        if keep_tokens is None or int(keep_tokens) <= 0:
            return model_kwargs
        if stage == "ssm" and compressor is None:
            return model_kwargs

        prefix_len = getattr(self, "_active_ssm_prefix_len", None)
        if prefix_len is None:
            prefix_len = getattr(self.config, "ssm_prefix_len", 0)

        past_key_values = model_kwargs.get("past_key_values", None)
        if past_key_values is None:
            return model_kwargs

        frame_spans = getattr(self, "_generation_frame_spans", None)
        frame_window = getattr(self, "_generation_frame_window", None)
        prefill_len = getattr(self, "_generation_frame_prefill_len", None)
        already_frame_trimmed = getattr(self, "_generation_frame_trimmed", False)
        cache_seq_len = self._cache_seq_len(past_key_values)
        if frame_spans is not None and already_frame_trimmed:
            attention_mask = model_kwargs.get("attention_mask", None)
            if attention_mask is not None and cache_seq_len is not None:
                target_len = int(cache_seq_len) + int(num_new_tokens)
                if attention_mask.shape[-1] > target_len:
                    generation_prefix_len = int(getattr(self, "_generation_trimmed_prefix_len", 0) or 0)
                    tail_len = max(0, target_len - generation_prefix_len)
                    if generation_prefix_len > 0 and tail_len > 0:
                        model_kwargs["attention_mask"] = torch.cat(
                            [attention_mask[..., :generation_prefix_len], attention_mask[..., -tail_len:]],
                            dim=-1,
                        )
                    else:
                        model_kwargs["attention_mask"] = attention_mask[..., -target_len:]
                cache_position = model_kwargs.get("cache_position", None)
                if cache_position is not None:
                    model_kwargs["cache_position"] = torch.arange(
                        int(cache_seq_len),
                        int(cache_seq_len) + int(num_new_tokens),
                        device=cache_position.device,
                    )
            return model_kwargs

        use_frame_trim = (
            frame_spans is not None
            and frame_window is not None
            and prefill_len is not None
            and cache_seq_len is not None
            and not already_frame_trimmed
            and cache_seq_len >= prefill_len
        )
        if use_frame_trim:
            generation_prefix_len = int(frame_spans[0][0])
            past_key_values, kept_len, tail_abs_start = self.absorb_and_trim_past_key_values(
                past_key_values=past_key_values,
                compressor=compressor if stage == "ssm" else None,
                keep_tokens=int(keep_tokens),
                prefix_len=generation_prefix_len,
                current_seq_end=int(prefill_len),
                frame_spans=frame_spans,
                frame_window=int(frame_window),
                tail_abs_start=getattr(self, "_generation_tail_abs_start", generation_prefix_len),
            )
            self._generation_frame_trimmed = True
            self._generation_tail_abs_start = tail_abs_start
            model_kwargs["past_key_values"] = past_key_values
            attention_mask = model_kwargs.get("attention_mask", None)
            if attention_mask is not None and kept_len is not None and attention_mask.shape[-1] > kept_len:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask[..., :generation_prefix_len], attention_mask[..., -(kept_len - generation_prefix_len) :]],
                    dim=-1,
                )
            return model_kwargs

        if frame_spans is not None:
            return model_kwargs

        past_key_values, kept_len, _ = self.absorb_and_trim_past_key_values(
            past_key_values=past_key_values,
            compressor=compressor if stage == "ssm" else None,
            keep_tokens=int(keep_tokens),
            prefix_len=int(prefix_len),
        )
        model_kwargs["past_key_values"] = past_key_values

        # Keep the mask length aligned with the manually trimmed cache.
        attention_mask = model_kwargs.get("attention_mask", None)
        if attention_mask is not None and kept_len is not None and attention_mask.shape[-1] > kept_len:
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask[..., : int(prefix_len)], attention_mask[..., -int(keep_tokens) :]],
                dim=-1,
            )[..., -kept_len:]

        return model_kwargs

    @staticmethod
    def _cache_seq_len(past_key_values) -> Optional[int]:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        if len(past_key_values) == 0:
            return None
        first_layer = past_key_values[0]
        if first_layer is None or len(first_layer) < 2:
            return None
        return int(first_layer[0].shape[-2])

    @staticmethod
    def _keep_prefix_and_recent(tensor, prefix_len: int, keep_tokens: int):
        pieces = []
        if prefix_len > 0:
            pieces.append(tensor[..., :prefix_len, :])
        if keep_tokens > 0:
            pieces.append(tensor[..., -keep_tokens:, :])
        if not pieces:
            return tensor[..., :0, :]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)

    def _visual_spatial_merge_area(self) -> int:
        vision_config = getattr(self.config, "vision_config", None)
        merge_size = int(getattr(vision_config, "spatial_merge_size", 1) or 1)
        return max(1, merge_size * merge_size)

    def _video_frame_token_count(self, grid_thw: torch.Tensor) -> int:
        return int(grid_thw[1].item() * grid_thw[2].item()) // self._visual_spatial_merge_area()

    def _image_token_count(self, grid_thw: torch.Tensor) -> int:
        return int(grid_thw[0].item() * grid_thw[1].item() * grid_thw[2].item()) // self._visual_spatial_merge_area()

    def _build_training_frame_spans(
        self,
        input_ids,
        video_grid_thw=None,
        image_grid_thw=None,
        frame_window: Optional[int] = None,
    ):
        if frame_window is None or frame_window <= 0 or input_ids is None:
            return None
        if input_ids.shape[0] != 1:
            return None

        token_ids = input_ids[0]
        frame_spans = []
        image_token_positions = torch.nonzero(token_ids == self.config.image_token_id, as_tuple=False).flatten().tolist()
        if image_grid_thw is not None and image_token_positions:
            cursor = 0
            for image_idx in range(int(image_grid_thw.shape[0])):
                grid = image_grid_thw[image_idx]
                token_count = self._image_token_count(grid)
                image_positions = image_token_positions[cursor : cursor + token_count]
                cursor += token_count
                if len(image_positions) != token_count:
                    break
                frame_spans.append((image_positions[0], image_positions[-1] + 1))

        video_token_positions = torch.nonzero(token_ids == self.config.video_token_id, as_tuple=False).flatten().tolist()
        if video_grid_thw is not None and video_token_positions:
            cursor = 0
            for video_idx in range(int(video_grid_thw.shape[0])):
                grid = video_grid_thw[video_idx]
                frame_tokens = self._video_frame_token_count(grid)
                num_frames = int(grid[0].item())
                if frame_tokens <= 0 or num_frames <= 0:
                    continue
                video_token_count = frame_tokens * num_frames
                video_positions = video_token_positions[cursor : cursor + video_token_count]
                cursor += video_token_count
                if len(video_positions) != video_token_count:
                    break

                for frame_idx in range(num_frames):
                    frame_start = frame_idx * frame_tokens
                    start = video_positions[frame_start]
                    end = video_positions[frame_start + frame_tokens - 1] + 1
                    frame_spans.append((start, end))

        return sorted(set(frame_spans)) or None

    @staticmethod
    def _iter_training_segments(seq_len: int, frame_spans, step_size: int):
        if frame_spans:
            start = 0
            for _, end in frame_spans:
                end = max(start + 1, min(int(end), seq_len))
                if end > start:
                    yield start, end
                    start = end
            if start < seq_len:
                yield start, seq_len
            return

        for start in range(0, seq_len, step_size):
            yield start, min(seq_len, start + step_size)

    @staticmethod
    def _frame_window_drop_range(
        current_seq_end: int,
        frame_spans,
        frame_window: Optional[int],
        prefix_len: int,
        tail_abs_start: int,
        cache_seq_len: int,
    ):
        if frame_window is None or frame_window <= 0 or not frame_spans:
            return None
        visible_frames = [(start, end) for start, end in frame_spans if end <= current_seq_end]
        if len(visible_frames) <= frame_window:
            return None

        keep_tail_abs_start = int(visible_frames[-int(frame_window)][0])
        if keep_tail_abs_start <= tail_abs_start:
            return None

        drop_start = prefix_len
        drop_end = prefix_len + (keep_tail_abs_start - int(tail_abs_start))
        drop_end = min(max(drop_start, drop_end), cache_seq_len)
        if drop_end <= drop_start:
            return None
        return drop_start, drop_end, keep_tail_abs_start

    def absorb_and_trim_past_key_values(
        self,
        past_key_values,
        compressor,
        keep_tokens: int,
        prefix_len: int = 0,
        current_seq_end: Optional[int] = None,
        frame_spans=None,
        frame_window: Optional[int] = None,
        tail_abs_start: Optional[int] = None,
    ):
        seq_len = self._cache_seq_len(past_key_values)
        if seq_len is None:
            return past_key_values, None, tail_abs_start

        prefix_len = max(0, min(int(prefix_len), seq_len))
        keep_tokens = max(0, int(keep_tokens))
        new_tail_abs_start = tail_abs_start
        if current_seq_end is not None and frame_spans is not None and frame_window is not None:
            drop_range = self._frame_window_drop_range(
                current_seq_end=current_seq_end,
                frame_spans=frame_spans,
                frame_window=frame_window,
                prefix_len=prefix_len,
                tail_abs_start=prefix_len if tail_abs_start is None else tail_abs_start,
                cache_seq_len=seq_len,
            )
            if drop_range is None:
                return past_key_values, seq_len, new_tail_abs_start
            drop_start, drop_end, new_tail_abs_start = drop_range
        else:
            drop_tokens = seq_len - prefix_len - keep_tokens
            if drop_tokens <= 0:
                return past_key_values, seq_len, new_tail_abs_start
            drop_start = prefix_len
            drop_end = prefix_len + drop_tokens

        kept_len = seq_len - (drop_end - drop_start)

        if isinstance(past_key_values, DynamicCache):
            for layer_idx, (key_states, value_states) in enumerate(
                zip(past_key_values.key_cache, past_key_values.value_cache)
            ):
                if key_states.numel() == 0:
                    continue
                if compressor is not None:
                    self._absorb_evicted_kv(
                        compressor,
                        layer_idx,
                        key_states[..., drop_start:drop_end, :],
                        value_states[..., drop_start:drop_end, :],
                    )
                keep_indices = [torch.arange(0, drop_start, device=key_states.device)]
                if drop_end < key_states.shape[-2]:
                    keep_indices.append(torch.arange(drop_end, key_states.shape[-2], device=key_states.device))
                keep_indices = torch.cat(keep_indices) if keep_indices else torch.empty(0, device=key_states.device, dtype=torch.long)
                past_key_values.key_cache[layer_idx] = torch.index_select(key_states, -2, keep_indices)
                past_key_values.value_cache[layer_idx] = torch.index_select(value_states, -2, keep_indices)
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = kept_len
            return past_key_values, kept_len, new_tail_abs_start

        trimmed_layers = []
        for layer_idx, layer_cache in enumerate(past_key_values):
            key_states, value_states = layer_cache[:2]
            if compressor is not None:
                self._absorb_evicted_kv(
                    compressor,
                    layer_idx,
                    key_states[..., drop_start:drop_end, :],
                    value_states[..., drop_start:drop_end, :],
                )
            keep_indices = [torch.arange(0, drop_start, device=key_states.device)]
            if drop_end < key_states.shape[-2]:
                keep_indices.append(torch.arange(drop_end, key_states.shape[-2], device=key_states.device))
            keep_indices = torch.cat(keep_indices) if keep_indices else torch.empty(0, device=key_states.device, dtype=torch.long)
            trimmed_key = torch.index_select(key_states, -2, keep_indices)
            trimmed_value = torch.index_select(value_states, -2, keep_indices)
            trimmed_layers.append((trimmed_key, trimmed_value) + tuple(layer_cache[2:]))

        return tuple(trimmed_layers), kept_len, new_tail_abs_start

    def _absorb_evicted_kv(self, compressor, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        if getattr(self, "_active_ssm_trainable_memory", False) and hasattr(compressor, "absorb_trainable"):
            compressor.absorb_trainable(layer_idx, key_states, value_states)
        else:
            compressor.absorb(layer_idx, key_states.detach().clone(), value_states.detach().clone())

    # Compatibility helpers for older Cambrian-oriented launch code.
    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return getattr(self, "visual", None)

    def get_vision_tower_aux_list(self):
        visual = self.get_vision_tower()
        return [] if visual is None else [visual]


# Backward-compatible names. Existing scripts import Qwen2_5SSMForCausalLM or
# CambrianQwenForCausalLM from this file; both now point to the native
# Qwen2.5-VL implementation above.
Qwen2_5SSMForCausalLM = Qwen2_5SSMForConditionalGeneration
Qwen2_5SSMModel = Qwen2_5SSMForConditionalGeneration
CambrianQwenConfig = Qwen2_5SSMConfig
CambrianQwenModel = Qwen2_5SSMModel
CambrianQwenForCausalLM = Qwen2_5SSMForConditionalGeneration


AutoConfig.register("qwen2_5_ssm", Qwen2_5SSMConfig)
AutoModelForCausalLM.register(Qwen2_5SSMConfig, Qwen2_5SSMForConditionalGeneration)
if AutoModelForImageTextToText is not None:
    AutoModelForImageTextToText.register(Qwen2_5SSMConfig, Qwen2_5SSMForConditionalGeneration)
if AutoModelForVision2Seq is not None:
    AutoModelForVision2Seq.register(Qwen2_5SSMConfig, Qwen2_5SSMForConditionalGeneration)
