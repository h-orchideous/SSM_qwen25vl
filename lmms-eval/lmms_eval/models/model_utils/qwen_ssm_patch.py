import os
import sys
import types
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from cambrian.ssm.ssm_compressor import MemoryFusionLayer, SelectiveSSMLayer
except Exception as exc:  # pragma: no cover - exercised only when optional package is absent
    MemoryFusionLayer = None
    SelectiveSSMLayer = None
    _SSM_IMPORT_ERROR = exc
else:
    _SSM_IMPORT_ERROR = None

_VMAMBA_CSM_IMPORT_ERROR = None
_VMAMBA_CSM_LOGGED = False
try:
    from lmms_eval.models.model_utils.csm_triton import cross_merge_fn as _vmamba_cross_merge_fn
    from lmms_eval.models.model_utils.csm_triton import cross_scan_fn as _vmamba_cross_scan_fn
except Exception as exc:
    try:
        from cambrian.ssm.csm_triton import cross_merge_fn as _vmamba_cross_merge_fn
        from cambrian.ssm.csm_triton import cross_scan_fn as _vmamba_cross_scan_fn
    except Exception:
        vmamba_models_path = os.environ.get(
            "VMAMBA_MODELS_PATH",
            "/home/ZhangHuayu/Workspace/VMamba/classification/models",
        )
        if vmamba_models_path and os.path.isdir(vmamba_models_path) and vmamba_models_path not in sys.path:
            sys.path.append(vmamba_models_path)
        try:
            from csm_triton import cross_merge_fn as _vmamba_cross_merge_fn
            from csm_triton import cross_scan_fn as _vmamba_cross_scan_fn
        except Exception as retry_exc:  # pragma: no cover - depends on optional VMamba checkout
            _vmamba_cross_scan_fn = None
            _vmamba_cross_merge_fn = None
            _VMAMBA_CSM_IMPORT_ERROR = retry_exc
        else:
            _VMAMBA_CSM_IMPORT_ERROR = None
    else:
        _VMAMBA_CSM_IMPORT_ERROR = None
else:
    _VMAMBA_CSM_IMPORT_ERROR = None


def _get_text_config(model):
    config = getattr(model, "config", None)
    return getattr(config, "text_config", config)


def _infer_2d_shape(num_tokens: int, preferred_shape=None):
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if preferred_shape is not None:
        preferred_h, preferred_w = int(preferred_shape[0]), int(preferred_shape[1])
        if preferred_h > 0 and preferred_w > 0 and preferred_h * preferred_w == num_tokens:
            return preferred_h, preferred_w
        target_ratio = preferred_h / max(1, preferred_w)
    else:
        target_ratio = 1.0

    best_h, best_w = 1, num_tokens
    best_score = float("inf")
    limit = int(num_tokens**0.5)
    for h in range(1, limit + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        for cand_h, cand_w in ((h, w), (w, h)):
            score = abs((cand_h / max(1, cand_w)) - target_ratio)
            if score < best_score:
                best_h, best_w = cand_h, cand_w
                best_score = score
    return best_h, best_w


def _build_cross_scan_sequences(hidden_latents: torch.Tensor, map_h: int, map_w: int):
    global _VMAMBA_CSM_LOGGED
    if not _VMAMBA_CSM_LOGGED:
        message = (
            "[qwen_ssm_patch] "
            f"vmamba_csm_available={_vmamba_cross_scan_fn is not None} "
            f"hidden_latents_is_cuda={hidden_latents.is_cuda} "
            f"import_error={_VMAMBA_CSM_IMPORT_ERROR}"
        )
        logger.info(message)
        print(message, flush=True)
        _VMAMBA_CSM_LOGGED = True

    hidden_map = hidden_latents.reshape(hidden_latents.size(0), int(map_h), int(map_w), hidden_latents.size(-1))
    if _vmamba_cross_scan_fn is not None and hidden_latents.is_cuda:
        # VMamba cross2d order is horizontal, vertical, horizontal_reverse, vertical_reverse.
        # Keep the local compressor order as horizontal, horizontal_reverse, vertical, vertical_reverse.
        scanned = _vmamba_cross_scan_fn(
            hidden_map,
            in_channel_first=False,
            out_channel_first=False,
            scans=0,
            force_torch=False,
        )
        return (
            scanned[:, :, 0, :].contiguous(),
            scanned[:, :, 2, :].contiguous(),
            scanned[:, :, 1, :].contiguous(),
            scanned[:, :, 3, :].contiguous(),
        )

    horizontal = hidden_map.reshape(hidden_latents.size(0), int(map_h) * int(map_w), hidden_latents.size(-1))
    vertical = hidden_map.transpose(1, 2).contiguous().reshape(
        hidden_latents.size(0),
        int(map_h) * int(map_w),
        hidden_latents.size(-1),
    )
    return (
        horizontal,
        torch.flip(horizontal, dims=[1]),
        vertical,
        torch.flip(vertical, dims=[1]),
    )


def _merge_cross_scan_outputs(directional_outputs, map_h: int, map_w: int):
    if _vmamba_cross_merge_fn is not None and directional_outputs[0].is_cuda:
        # Convert local order back to VMamba order: horizontal, vertical, horizontal_reverse, vertical_reverse.
        merged_input = torch.stack(
            [
                directional_outputs[0],
                directional_outputs[2],
                directional_outputs[1],
                directional_outputs[3],
            ],
            dim=2,
        ).contiguous()
        merged_input = merged_input.view(
            directional_outputs[0].size(0),
            int(map_h),
            int(map_w),
            4,
            directional_outputs[0].size(-1),
        )
        return _vmamba_cross_merge_fn(
            merged_input,
            in_channel_first=False,
            out_channel_first=False,
            scans=0,
            force_torch=False,
        )

    horizontal = directional_outputs[0]
    horizontal_rev = torch.flip(directional_outputs[1], dims=[1])
    vertical = directional_outputs[2].reshape(
        horizontal.size(0),
        int(map_w),
        int(map_h),
        horizontal.size(-1),
    ).transpose(1, 2).contiguous().reshape_as(horizontal)
    vertical_rev = torch.flip(directional_outputs[3], dims=[1]).reshape(
        horizontal.size(0),
        int(map_w),
        int(map_h),
        horizontal.size(-1),
    ).transpose(1, 2).contiguous().reshape_as(horizontal)
    return horizontal + horizontal_rev + vertical + vertical_rev


def _get_visual_merge_size(model) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    merge_size = getattr(vision_config, "spatial_merge_size", None)
    if merge_size is None:
        merge_size = getattr(config, "spatial_merge_size", 2)
    return max(1, int(merge_size))


def _spatial_shape_from_grid(model, grid_thw: torch.Tensor, num_tokens: int):
    grid_t, grid_h, grid_w = [int(v) for v in grid_thw.reshape(-1, 3)[0].tolist()]
    merge_size = _get_visual_merge_size(model)
    preferred_h = max(1, grid_t * (grid_h // merge_size))
    preferred_w = max(1, grid_w // merge_size)
    return _infer_2d_shape(
        int(num_tokens),
        preferred_shape=(preferred_h, preferred_w),
    )


class SSMHiddenCompressor(nn.Module):
    """
    4-direction 2D hidden-latent SSM memory for SimpleStream-style evaluation.

    Encoded frames that leave the recent visual window are restored as a 2D
    hidden-token map and absorbed with V-Mamba-style cross-scan directions.
    No rolling KV cache is maintained by the evaluation model.
    """

    NUM_DIRECTIONS = 4

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        d_state: int = 64,
        max_memory_len: int = 256,
        fusion_num_heads: int = 8,
        fusion_bottleneck: int = 256,
        layer_sharing: str = "group4",
    ):
        super().__init__()
        if SelectiveSSMLayer is None or MemoryFusionLayer is None:
            raise ImportError(f"Failed to import hidden SSM components: {_SSM_IMPORT_ERROR}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for SelectiveSSMLayer reuse, got {hidden_dim}")

        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.max_memory_len = int(max_memory_len)
        self._ssm_input_half_dim = self.hidden_dim // 2

        if layer_sharing == "none":
            n_unique = self.num_layers
            self._layer_map = list(range(self.num_layers))
        elif layer_sharing == "group4":
            n_unique = (self.num_layers + 3) // 4
            self._layer_map = [i // 4 for i in range(self.num_layers)]
        elif layer_sharing == "all":
            n_unique = 1
            self._layer_map = [0] * self.num_layers
        else:
            raise ValueError(f"Unknown layer_sharing: {layer_sharing}")

        self.ssm_layers = nn.ModuleList()
        for _ in range(n_unique):
            self.ssm_layers.append(
                nn.ModuleList(
                    [
                        SelectiveSSMLayer(
                            d_kv=self._ssm_input_half_dim,
                            d_state=int(d_state),
                        )
                        for _ in range(self.NUM_DIRECTIONS)
                    ]
                )
            )
        self.fusion_layers = nn.ModuleList(
            [
                MemoryFusionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(fusion_num_heads),
                    bottleneck_dim=int(fusion_bottleneck),
                )
                for _ in range(n_unique)
            ]
        )
        self._ssm_states = [[None] * self.NUM_DIRECTIONS for _ in range(self.num_layers)]
        self._ssm_memories = [None] * self.num_layers

    @torch.no_grad()
    def absorb_hidden(self, hidden_latents: torch.Tensor, spatial_shape=None):
        if spatial_shape is None:
            if hidden_latents.ndim != 3:
                raise ValueError(f"hidden_latents must be (B, L, H), got {tuple(hidden_latents.shape)}")
            spatial_shape = _infer_2d_shape(hidden_latents.size(1))
        return self.absorb_spatial_hidden(hidden_latents, spatial_shape)

    @torch.no_grad()
    def absorb_spatial_hidden(self, hidden_latents: torch.Tensor, spatial_shape):
        if hidden_latents.ndim != 3:
            raise ValueError(f"hidden_latents must be (B, L, H), got {tuple(hidden_latents.shape)}")
        if hidden_latents.size(-1) != self.hidden_dim:
            raise ValueError(f"hidden_latents hidden dim mismatch: expected {self.hidden_dim}, got {hidden_latents.size(-1)}")

        map_h, map_w = int(spatial_shape[0]), int(spatial_shape[1])
        if map_h <= 0 or map_w <= 0:
            raise ValueError(f"Invalid spatial_shape: {spatial_shape}")
        token_count = map_h * map_w
        if hidden_latents.size(1) < token_count:
            map_h, map_w = _infer_2d_shape(hidden_latents.size(1), preferred_shape=(map_h, map_w))
            token_count = map_h * map_w
        hidden_latents = hidden_latents[:, :token_count, :].detach()

        for layer_idx in range(self.num_layers):
            ssm_idx = self._layer_map[layer_idx]
            directional_inputs = _build_cross_scan_sequences(hidden_latents, map_h, map_w)
            directional_outputs = []
            for direction_idx, direction_input in enumerate(directional_inputs):
                ssm_output, new_state = self.ssm_layers[ssm_idx][direction_idx](
                    direction_input,
                    self._ssm_states[layer_idx][direction_idx],
                )
                self._ssm_states[layer_idx][direction_idx] = new_state.detach()
                directional_outputs.append(ssm_output.detach())

            ssm_output = _merge_cross_scan_outputs(directional_outputs, map_h, map_w)
            if self._ssm_memories[layer_idx] is None:
                self._ssm_memories[layer_idx] = ssm_output
            else:
                self._ssm_memories[layer_idx] = torch.cat(
                    [self._ssm_memories[layer_idx], ssm_output],
                    dim=1,
                )
            if self._ssm_memories[layer_idx].shape[1] > self.max_memory_len:
                self._ssm_memories[layer_idx] = self._ssm_memories[layer_idx][:, -self.max_memory_len :]

    def get_memory_for_fusion(self, layer_idx: int):
        return self._ssm_memories[layer_idx]

    def fuse(self, layer_idx: int, hidden_states: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        memory = self.get_memory_for_fusion(layer_idx)
        if memory is None:
            return attn_output
        ssm_idx = self._layer_map[layer_idx]
        memory_context = self.fusion_layers[ssm_idx](hidden_states, memory)
        return attn_output + memory_context

    def has_memory(self, layer_idx: int) -> bool:
        return self._ssm_memories[layer_idx] is not None

    def reset(self):
        self._ssm_states = [[None] * self.NUM_DIRECTIONS for _ in range(self.num_layers)]
        self._ssm_memories = [None] * self.num_layers


def build_qwen_ssm_compressor(
    model,
    d_state: int = 64,
    max_memory_len: int = 256,
    fusion_num_heads: int = 8,
    fusion_bottleneck: int = 256,
    layer_sharing: str = "group4",
):
    text_config = _get_text_config(model)
    hidden_dim = int(getattr(text_config, "hidden_size"))
    num_layers = int(getattr(text_config, "num_hidden_layers"))

    compressor = SSMHiddenCompressor(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        d_state=int(d_state),
        max_memory_len=int(max_memory_len),
        fusion_num_heads=int(fusion_num_heads),
        fusion_bottleneck=int(fusion_bottleneck),
        layer_sharing=layer_sharing,
    )

    first_param = next(model.parameters(), None)
    if first_param is not None:
        compressor = compressor.to(device=first_param.device, dtype=first_param.dtype)
    return compressor


def install_qwen_ssm_fusion(model, compressor_attr: str = "ssm_compressor") -> int:
    text_model = getattr(model, "model", model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder layers at model.model.layers")

    patched = 0
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, "_qwen_ssm_original_forward"):
            continue

        layer._qwen_ssm_original_forward = layer.forward
        layer._qwen_ssm_parent_model = model
        layer._qwen_ssm_layer_idx = layer_idx

        def forward_with_ssm(self, *args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            outputs = self._qwen_ssm_original_forward(*args, **kwargs)
            if not getattr(self._qwen_ssm_parent_model, "ssm_fusion_enabled", True):
                return outputs

            compressor = getattr(self._qwen_ssm_parent_model, compressor_attr, None)
            if hidden_states is None or compressor is None or not compressor.has_memory(self._qwen_ssm_layer_idx):
                return outputs

            fused_hidden = compressor.fuse(
                layer_idx=self._qwen_ssm_layer_idx,
                hidden_states=hidden_states,
                attn_output=outputs[0],
            )
            return (fused_hidden,) + tuple(outputs[1:])

        layer.forward = types.MethodType(forward_with_ssm, layer)
        patched += 1

    return patched


@torch.no_grad()
def absorb_encoded_frame_to_ssm(compressor, model, vision_emb: torch.Tensor, grid_thw: torch.Tensor):
    if compressor is None:
        return

    if vision_emb.ndim != 2:
        raise ValueError(f"vision_emb must be (L, H), got {tuple(vision_emb.shape)}")
    if grid_thw.ndim == 1:
        grid_thw = grid_thw.unsqueeze(0)

    map_h, map_w = _spatial_shape_from_grid(model, grid_thw, vision_emb.size(0))
    visual_tokens = vision_emb[: map_h * map_w].unsqueeze(0)
    compressor.absorb_spatial_hidden(visual_tokens, spatial_shape=(map_h, map_w))


@torch.no_grad()
def absorb_frame_to_ssm(compressor, model, processor, frame, device):
    if compressor is None:
        return

    frame_batch = processor(text=[""], images=[frame.convert("RGB")], padding=True, return_tensors="pt")
    pixel_values = frame_batch["pixel_values"].to(device, dtype=model.visual.dtype)
    image_grid_thw = frame_batch["image_grid_thw"].to(device)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    absorb_encoded_frame_to_ssm(compressor, model, image_embeds, image_grid_thw)
