try:
    from transformers import DynamicCache
except Exception:
    # Minimal fallback if `transformers` is not installed in the environment
    class DynamicCache:
        def __init__(self, *args, **kwargs):
            # simple placeholders expected by the StreamingCache implementation
            self._seen_tokens = 0
            self.key_cache = []
            self.value_cache = []
from typing import Optional, Iterable, List, Dict, Any, Tuple
import os
import torch

# Inherits from DynamicCache but adds position cache
class StreamingCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__()
        self.position_ids_cache: List[torch.Tensor] = []
    def update_position_ids(self, position_ids: torch.Tensor, layer_idx: int = 0):
        """
        Parameters:
            position_ids (`torch.Tensor`):
                The new position ids to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
        """
        # Update the cache
        if position_ids is not None:
            if len(self.position_ids_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.position_ids_cache), layer_idx):
                    self.position_ids_cache.append(torch.tensor([]))
                self.position_ids_cache.append(position_ids)
            else:  # fills previously skipped layers; checking for tensor causes errors
                self.position_ids_cache[layer_idx] = position_ids
            
        return self.position_ids_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional debug print to inspect KV shapes (controlled by env var SSM_DEBUG=1)
        if os.getenv("SSM_DEBUG") == "1":
            try:
                ks = key_states.shape if hasattr(key_states, 'shape') else None
                vs = value_states.shape if hasattr(value_states, 'shape') else None
            except Exception:
                ks = None
                vs = None
            print(f"[SSM_DEBUG] update layer={layer_idx} key_states.shape={ks} value_states.shape={vs}")

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]