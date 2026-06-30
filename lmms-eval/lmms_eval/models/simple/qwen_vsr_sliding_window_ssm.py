from typing import List

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen_ssm_patch import (
    absorb_trimmed_kv_to_ssm,
    build_qwen_ssm_compressor,
    install_qwen_ssm_fusion,
)
from lmms_eval.models.simple.qwen_vsr_sliding_window import Qwen_VSR_SlidingWindow


@register_model("qwen_vsr_sliding_window_ssm")
class Qwen_VSR_SlidingWindow_SSM(Qwen_VSR_SlidingWindow):
    """
    Native Qwen2.5-VL sliding-window path with SSM long-memory fusion.

    This class keeps the official Qwen2_5_VLForConditionalGeneration model and
    patches its decoder layers after loading. Evicted visual KV is absorbed into
    an SSM compressor, then read back through a small fusion module after each
    decoder self-attention.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        ssm_d_state: int = 64,
        ssm_max_memory_len: int = 256,
        ssm_fusion_num_heads: int = 8,
        ssm_fusion_bottleneck: int = 256,
        ssm_layer_sharing: str = "group4",
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, log_prefix="qwen_vsr_sw_ssm", **kwargs)

        self.ssm_compressor = build_qwen_ssm_compressor(
            self.model,
            d_state=ssm_d_state,
            max_memory_len=ssm_max_memory_len,
            fusion_num_heads=ssm_fusion_num_heads,
            fusion_bottleneck=ssm_fusion_bottleneck,
            layer_sharing=ssm_layer_sharing,
        )
        self.model.ssm_compressor = self.ssm_compressor
        patched_layers = install_qwen_ssm_fusion(self.model)

        eval_logger.info(
            f"[qwen_vsr_sw_ssm init] backend=native_qwen25vl patched_layers={patched_layers} "
            f"ssm_d_state={ssm_d_state} ssm_max_memory_len={ssm_max_memory_len} "
            f"ssm_layer_sharing={ssm_layer_sharing}"
        )

    def _trim_dynamic_cache_visual_prefix(self, cache_obj, prefix_len: int, drop_tokens: int):
        if drop_tokens <= 0:
            return cache_obj

        legacy = self._normalize_past_key_values(cache_obj)
        absorb_trimmed_kv_to_ssm(
            getattr(self, "ssm_compressor", None),
            legacy,
            prefix_len=prefix_len,
            drop_tokens=drop_tokens,
        )
        return super()._trim_dynamic_cache_visual_prefix(cache_obj, prefix_len, drop_tokens)

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
        return super()._run_streaming_video(context, static_visuals, video_paths, current_gen_kwargs, until)
