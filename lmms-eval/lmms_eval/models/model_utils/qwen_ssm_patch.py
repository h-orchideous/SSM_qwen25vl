import types

import torch

try:
    from cambrian.ssm.ssm_compressor import SSMCacheCompressor
except Exception as exc:  # pragma: no cover - exercised only when optional package is absent
    SSMCacheCompressor = None
    _SSM_IMPORT_ERROR = exc
else:
    _SSM_IMPORT_ERROR = None


def _get_text_config(model):
    config = getattr(model, "config", None)
    return getattr(config, "text_config", config)


def build_qwen_ssm_compressor(
    model,
    d_state: int = 64,
    max_memory_len: int = 256,
    fusion_num_heads: int = 8,
    fusion_bottleneck: int = 256,
    layer_sharing: str = "group4",
):
    if SSMCacheCompressor is None:
        raise ImportError(f"Failed to import SSMCacheCompressor: {_SSM_IMPORT_ERROR}")

    text_config = _get_text_config(model)
    hidden_dim = int(getattr(text_config, "hidden_size"))
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    num_attention_heads = int(getattr(text_config, "num_attention_heads"))
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(text_config, "head_dim", hidden_dim // num_attention_heads))

    compressor = SSMCacheCompressor(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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


def absorb_trimmed_kv_to_ssm(compressor, legacy_cache, prefix_len: int, drop_tokens: int):
    if compressor is None or drop_tokens <= 0:
        return

    for layer_idx, (key_states, value_states) in enumerate(legacy_cache):
        seq_len = key_states.size(2)
        drop_start = min(seq_len, int(prefix_len))
        drop_end = min(seq_len, int(prefix_len) + int(drop_tokens))
        if drop_end <= drop_start:
            continue

        evicted_keys = key_states[:, :, drop_start:drop_end, :].detach().clone()
        evicted_values = value_states[:, :, drop_start:drop_end, :].detach().clone()
        compressor.absorb(layer_idx, evicted_keys, evicted_values)
