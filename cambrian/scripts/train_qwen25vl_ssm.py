#!/usr/bin/env python
"""Minimal SFT entrypoint for native Qwen2.5-VL + SW/SSM training."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from cambrian.model.language_model.qwen2_5_ssm import Qwen2_5SSMConfig, Qwen2_5SSMForConditionalGeneration
from cambrian.ssm.ssm_compressor import SSMCacheCompressor

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover - runtime dependency check
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    _QWEN_VL_UTILS_IMPORT_ERROR = None


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct")
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="sdpa")
    train_ssm_only: bool = field(default=True)
    trainable_modules: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default="")
    image_root: Optional[str] = field(default=None)
    video_root: Optional[str] = field(default=None)
    max_length: int = field(default=8192)
    mask_prompt_labels: bool = field(default=True)
    min_pixels: Optional[int] = field(default=None)
    max_pixels: Optional[int] = field(default=None)
    fps: Optional[float] = field(default=None)
    max_frames: Optional[int] = field(default=None)
    stream_video_as_images: bool = field(default=True)
    stream_frame_stride: int = field(default=1)


@dataclass
class SSMArguments:
    train_stage: str = field(default="ssm")
    ssm_sliding_window: int = field(default=128)
    ssm_frame_window: Optional[int] = field(default=None)
    ssm_prefix_len: int = field(default=0)
    ssm_training_step_size: Optional[int] = field(default=None)
    ssm_training_chunk_size: Optional[int] = field(default=None)
    ssm_d_state: int = field(default=64)
    ssm_max_memory_len: int = field(default=256)
    ssm_fusion_num_heads: int = field(default=8)
    ssm_fusion_bottleneck: int = field(default=256)
    ssm_layer_sharing: str = field(default="group4")
    ssm_use_fast_path: bool = field(default=True)
    ssm_visual_encode_chunk_size: int = field(default=1)


class JsonSftDataset(Dataset):
    def __init__(self, data_path: str):
        if not data_path:
            raise ValueError("--data_path is required")
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        if path.suffix.lower() == ".jsonl":
            self.records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.records = data if isinstance(data, list) else data.get("data", [])

        if not self.records:
            raise ValueError(f"No training records loaded from {data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def _dtype_from_name(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"auto", "none"}:
        return "auto"
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _resolve_path(value: str, root: Optional[str]) -> str:
    if not root or not isinstance(value, str) or "://" in value or os.path.isabs(value):
        return value
    return str(Path(root) / value)


def _resolve_media_paths(messages: Sequence[Dict[str, Any]], image_root: Optional[str], video_root: Optional[str]):
    resolved = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                item = dict(item)
                item_type = item.get("type")
                if item_type == "image":
                    key = "image" if "image" in item else "path" if "path" in item else None
                    if key is not None:
                        item[key] = _resolve_path(item[key], image_root)
                elif item_type == "video":
                    key = "video" if "video" in item else "path" if "path" in item else None
                    if key is not None:
                        item[key] = _resolve_path(item[key], video_root)
                new_content.append(item)
            message["content"] = new_content
        resolved.append(message)
    return resolved


def _decode_video_frames(video_path: str, frame_stride: int = 1):
    from PIL import Image

    frame_stride = max(1, int(frame_stride))
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        indices = list(range(0, len(vr), frame_stride))
        return [Image.fromarray(frame.asnumpy()).convert("RGB") for frame in vr.get_batch(indices)]
    except Exception as decord_exc:
        try:
            import av

            frames = []
            container = av.open(video_path)
            for idx, frame in enumerate(container.decode(video=0)):
                if idx % frame_stride == 0:
                    frames.append(frame.to_image().convert("RGB"))
            container.close()
            if frames:
                return frames
        except Exception as av_exc:
            raise RuntimeError(
                f"Failed to decode streaming video frames: {video_path}; "
                f"decord_err={decord_exc}; av_err={av_exc}"
            ) from av_exc
        raise RuntimeError(f"Failed to decode streaming video frames: {video_path}; decord_err={decord_exc}") from decord_exc


def _stream_videos_as_images(messages: Sequence[Dict[str, Any]], frame_stride: int = 1) -> List[Dict[str, Any]]:
    streamed = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if not isinstance(content, list):
            streamed.append(message)
            continue

        new_content = []
        for item in content:
            item = dict(item)
            if item.get("type") != "video":
                new_content.append(item)
                continue

            video_path = item.get("video") or item.get("path")
            if not video_path:
                continue
            for frame in _decode_video_frames(video_path, frame_stride=frame_stride):
                new_content.append({"type": "image", "image": frame})
        message["content"] = new_content
        streamed.append(message)
    return streamed


def _strip_visual_marker(text: str) -> str:
    return str(text).replace("<image>", "").replace("<video>", "").strip()


def _convert_conversations_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = []
    video = record.get("video")
    image = record.get("image")
    visual_attached = False

    for turn in record.get("conversations", []):
        speaker = turn.get("from")
        value = turn.get("value", "")
        if speaker in {"human", "user"}:
            content = []
            if not visual_attached:
                if video:
                    content.append({"type": "video", "video": video})
                    visual_attached = True
                elif image:
                    content.append({"type": "image", "image": image})
                    visual_attached = True
            content.append({"type": "text", "text": _strip_visual_marker(value)})
            messages.append({"role": "user", "content": content})
        elif speaker in {"gpt", "assistant"}:
            messages.append({"role": "assistant", "content": str(value)})
        else:
            role = "user" if not messages or messages[-1].get("role") == "assistant" else "assistant"
            messages.append({"role": role, "content": _strip_visual_marker(value)})

    return messages


def _messages_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in record:
        return record["messages"]
    if "conversations" in record:
        return _convert_conversations_record(record)
    if "prompt" in record and "response" in record:
        return [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    raise ValueError("Each record must contain messages/conversations or prompt+response")


def _prompt_messages(messages: Sequence[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not messages or messages[-1].get("role") != "assistant":
        return None
    return [dict(message) for message in messages[:-1]]


def _contains_vision(messages_batch: Sequence[Sequence[Dict[str, Any]]]) -> bool:
    for messages in messages_batch:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                        return True
    return False


def _process_vision(messages: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[Any, Any]:
    if process_vision_info is None:
        raise ImportError("qwen-vl-utils is required for multimodal SFT") from _QWEN_VL_UTILS_IMPORT_ERROR
    result = process_vision_info(messages)
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    raise RuntimeError("Unexpected qwen_vl_utils.process_vision_info return value")


class Qwen25VLSsmDataCollator:
    def __init__(self, processor, data_args: DataArguments):
        self.processor = processor
        self.data_args = data_args
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        prompt_lengths = []
        batch_messages = []

        for record in features:
            messages = _messages_from_record(record)
            messages = _resolve_media_paths(messages, self.data_args.image_root, self.data_args.video_root)
            if self.data_args.stream_video_as_images:
                messages = _stream_videos_as_images(messages, frame_stride=self.data_args.stream_frame_stride)
            batch_messages.append(messages)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

            if self.data_args.mask_prompt_labels:
                prompt_messages = _prompt_messages(messages)
                prompt_lengths.append(self._prompt_length(prompt_messages) if prompt_messages is not None else None)
            else:
                prompt_lengths.append(None)

        image_inputs, video_inputs = _process_vision(batch_messages) if _contains_vision(batch_messages) else (None, None)

        processor_kwargs = {
            "text": texts,
            "padding": True,
            "truncation": True,
            "max_length": self.data_args.max_length,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        if self.data_args.fps is not None:
            processor_kwargs["fps"] = self.data_args.fps
        if self.data_args.max_frames is not None:
            processor_kwargs["max_frames"] = self.data_args.max_frames

        batch = self.processor(**processor_kwargs)
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        if self.data_args.mask_prompt_labels:
            for row_idx, prompt_len in enumerate(prompt_lengths):
                if prompt_len is None:
                    continue
                prompt_len = min(prompt_len, labels.shape[1])
                labels[row_idx, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    def _prompt_length(self, prompt_messages: Sequence[Dict[str, Any]]) -> int:
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        kwargs = {
            "text": [prompt_text],
            "padding": False,
            "truncation": True,
            "max_length": self.data_args.max_length,
            "return_tensors": "pt",
        }
        if _contains_vision([prompt_messages]):
            image_inputs, video_inputs = _process_vision([prompt_messages])
            if image_inputs:
                kwargs["images"] = image_inputs
            if video_inputs:
                kwargs["videos"] = video_inputs
        if self.data_args.fps is not None:
            kwargs["fps"] = self.data_args.fps
        if self.data_args.max_frames is not None:
            kwargs["max_frames"] = self.data_args.max_frames
        return int(self.processor(**kwargs)["input_ids"].shape[1])


def _build_ssm_compressor(model, ssm_args: SSMArguments) -> SSMCacheCompressor:
    config = model.config
    num_layers = int(getattr(config, "num_hidden_layers"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(config, "hidden_size")) // max(1, num_attention_heads)

    compressor = SSMCacheCompressor(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim=int(getattr(config, "hidden_size")),
        d_state=ssm_args.ssm_d_state,
        max_memory_len=ssm_args.ssm_max_memory_len,
        fusion_num_heads=ssm_args.ssm_fusion_num_heads,
        fusion_bottleneck=ssm_args.ssm_fusion_bottleneck,
        layer_sharing=ssm_args.ssm_layer_sharing,
    )
    for module in compressor.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = bool(ssm_args.ssm_use_fast_path and module.use_fast_path)
    return compressor


def _configure_trainable_parameters(model, model_args: ModelArguments, train_stage: str):
    train_stage = train_stage.lower()
    if model_args.trainable_modules:
        module_names = [name.strip() for name in model_args.trainable_modules.split(",") if name.strip()]
        if not module_names:
            raise ValueError("--trainable_modules was provided but no module names were parsed")
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in module_names):
                param.requires_grad = True
        return

    if train_stage == "ssm":
        if model_args.train_ssm_only:
            model.requires_grad_(False)
            if getattr(model, "ssm_compressor", None) is None:
                raise ValueError("SSM stage requires model.ssm_compressor")
            model.ssm_compressor.requires_grad_(True)
        return

    if train_stage == "sw" and model_args.train_ssm_only:
        raise ValueError(
            "TRAIN_STAGE=sw has no SSM parameters to train. Set TRAIN_SSM_ONLY=false for backbone training, "
            "or set TRAINABLE_MODULES to a comma-separated list such as 'lm_head'."
        )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SSMArguments, TrainingArguments))
    model_args, data_args, ssm_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        raise ValueError("Disable gradient_checkpointing: SW/SSM training needs use_cache=True to collect evicted KV.")
    train_stage = ssm_args.train_stage.lower()
    if train_stage not in {"sw", "ssm"}:
        raise ValueError("--train_stage must be 'sw' or 'ssm'")

    config = Qwen2_5SSMConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.ssm_enabled = True
    config.ssm_sliding_window = ssm_args.ssm_sliding_window
    config.ssm_frame_window = ssm_args.ssm_frame_window
    config.ssm_prefix_len = ssm_args.ssm_prefix_len
    config.ssm_training_step_size = (
        ssm_args.ssm_training_step_size
        or ssm_args.ssm_training_chunk_size
        or ssm_args.ssm_sliding_window
    )
    config.ssm_training_chunk_size = config.ssm_training_step_size
    config.ssm_training_stage = train_stage
    config.ssm_visual_encode_chunk_size = ssm_args.ssm_visual_encode_chunk_size
    config.ssm_compressor_config = {
        "train_stage": train_stage,
        "d_state": ssm_args.ssm_d_state,
        "max_memory_len": ssm_args.ssm_max_memory_len,
        "fusion_num_heads": ssm_args.ssm_fusion_num_heads,
        "fusion_bottleneck": ssm_args.ssm_fusion_bottleneck,
        "layer_sharing": ssm_args.ssm_layer_sharing,
        "use_fast_path": ssm_args.ssm_use_fast_path,
        "visual_encode_chunk_size": ssm_args.ssm_visual_encode_chunk_size,
    }

    model = Qwen2_5SSMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=_dtype_from_name(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )
    if train_stage == "ssm":
        model.ssm_compressor = _build_ssm_compressor(model, ssm_args)
    else:
        model.ssm_compressor = None
    model.config.use_cache = True

    _configure_trainable_parameters(model, model_args, train_stage)

    processor_kwargs = {"trust_remote_code": model_args.trust_remote_code}
    if data_args.min_pixels is not None:
        processor_kwargs["min_pixels"] = data_args.min_pixels
    if data_args.max_pixels is not None:
        processor_kwargs["max_pixels"] = data_args.max_pixels
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)

    train_dataset = JsonSftDataset(data_args.data_path)
    data_collator = Qwen25VLSsmDataCollator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if getattr(model, "ssm_compressor", None) is not None:
        torch.save(model.ssm_compressor.state_dict(), os.path.join(training_args.output_dir, "ssm_compressor.bin"))


if __name__ == "__main__":
    main()
