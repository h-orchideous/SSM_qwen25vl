"""
streaming_vlm/inference/ssm/ssm_streaming_cache.py

SSM 版 StreamingCache
因为采用融合方式而非 prepend，这个类比之前简单很多
"""

try:
    from .streaming_cache import StreamingCache
except Exception:
    try:
        # try top-level package fallback if placed elsewhere on PYTHONPATH
        from streaming_cache import StreamingCache
    except Exception:
        # Fallback minimal StreamingCache for environments without streaming_vlm
        class StreamingCache:
            def __init__(self, _distributed_cache_data=None):
                # key_cache/value_cache expected to be lists of tensors per layer
                self.key_cache = []
                self.value_cache = []
from typing import Optional, Iterable
import os
import torch


class SSMStreamingCache(StreamingCache):
    """
    StreamingCache + SSM 长期记忆。
    
    ★ 与 prepend 方案的关键区别:
    - KV cache 中只有 recent raw KV（和原始一样）
    - SSM memory 不在 KV cache 中，而是通过 MemoryFusionLayer 融合
    - 因此 get_seq_length() 不需要任何修改
    - causal mask 不需要任何修改
    - RoPE 不需要任何修改
    
    唯一的新增功能: evict_to_ssm() 方法
    """
    
    def __init__(self, ssm_compressor, _distributed_cache_data=None):
        super().__init__(_distributed_cache_data)
        self.ssm_compressor = ssm_compressor
    
    def evict_to_ssm(self, layer_idx: int, start_index: int, end_index: int):
        """
        将 [start_index, end_index] 的 KV 送入 SSM 后从 cache 中删除。
        
        这是对原始 prune 操作的增强：
        原始: 直接删除 → 信息丢失
        增强: 先送入 SSM → 再删除 → 信息压缩保留
        """
        if len(self.key_cache) <= layer_idx or not self.key_cache[layer_idx].numel():
            if os.getenv("SSM_DEBUG"):
                print(f"[SSM_DEBUG] evict_to_ssm called but key_cache empty for layer={layer_idx}")
            return
        
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        # Detailed debug print
        if os.getenv("SSM_DEBUG"):
            seq_len = k.shape[2]
            device = k.device
            print(
                f"[SSM_DEBUG] evict_to_ssm layer={layer_idx} start={start_index} end={end_index} seq_len_before={seq_len} device={device} k_dtype={k.dtype} v_dtype={v.dtype}"
            )
        
        # 提取要 evict 的部分
        evicted_k = k[:, :, start_index:end_index + 1, :].clone()
        evicted_v = v[:, :, start_index:end_index + 1, :].clone()
        
        # 送入 SSM
        try:
            self.ssm_compressor.absorb(layer_idx, evicted_k, evicted_v)
            if os.getenv("SSM_DEBUG"):
                print(f"[SSM_DEBUG] absorb called for layer={layer_idx} evicted_len={(end_index-start_index+1)}")
        except Exception as e:
            if os.getenv("SSM_DEBUG"):
                print(f"[SSM_DEBUG] absorb exception layer={layer_idx} err={e}")
        
        # 从 cache 中删除（和原始 prune 逻辑相同）
        seq_len = k.shape[2]
        indices = list(range(start_index)) + list(range(end_index + 1, seq_len))
        if indices:
            idx_t = torch.tensor(indices, device=k.device)
            self.key_cache[layer_idx] = torch.index_select(k, 2, idx_t)
            self.value_cache[layer_idx] = torch.index_select(v, 2, idx_t)
        else:
            self.key_cache[layer_idx] = k[:, :, :0, :]
            self.value_cache[layer_idx] = v[:, :, :0, :]

        if os.getenv("SSM_DEBUG"):
            new_seq_len = self.key_cache[layer_idx].shape[2]
            print(f"[SSM_DEBUG] evict_to_ssm layer={layer_idx} seq_len_after={new_seq_len}")
    
    # ★ 不需要 override get_seq_length()
    # ★ 不需要修改 update()
    # ★ 因为 SSM memory 不在 KV cache 中