"""
SSM 长期记忆压缩器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

try:
    from .selective_scan_interface import selective_scan_fn, HAS_CUDA_KERNEL
    print(f"SSM Compressor: CUDA kernel available = {HAS_CUDA_KERNEL}")
except ImportError:
    HAS_CUDA_KERNEL = False
    selective_scan_fn = None
    print("SSM Compressor: CUDA kernel not found, will use PyTorch fallback")

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except Exception:
    selective_state_update = None


class SelectiveSSMLayer(nn.Module):
    """
    Mamba-style Selective SSM for KV cache compression.
    使用 selective_scan_cuda_oflex CUDA kernel。
    
    功能：接收被 evict 的 KV pairs，增量更新隐状态 h，
    同时返回更新后的 recurrent state。
    
    思路：
    - 输入是 concat(K, V)，因为 K 和 V 的压缩应该是联合的
      （同一个 token 的 key 和 value 语义相关，分开压缩会丢失对应关系）
    - SSM 的 selective mechanism（输入依赖的 B, C, dt）让模型学会
      "什么信息值得记住，什么可以遗忘"
    - 输出经过 residual connection，保留输入 KV 的细节信息
    """
    
    def __init__(
        self,
        d_kv: int,              # num_kv_heads * head_dim
        d_state: int = 64,      # SSM 隐状态维度（越大记忆容量越大，计算越多）
        d_conv: int = 4,        # 局部卷积核大小（捕捉相邻 token 的模式）
        expand: int = 2,        # 扩展因子（内部维度 = d_kv * expand）
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.d_kv = d_kv
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_kv * expand
        self.dt_rank = max(1, d_kv // 16)
        self.use_fast_path = use_fast_path and HAS_CUDA_KERNEL
        
        # 输入投影: concat(K,V) = 2*d_kv → 2*d_inner (分出 x 和 gate z)
        self.in_proj = nn.Linear(2 * d_kv, 2 * self.d_inner, bias=False)
        
        # 1D causal conv: 捕捉局部时序模式
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        
        # SSM 参数
        # dt: 控制"记忆更新速度"（大 dt = 更多吸收新输入，小 dt = 更多保留旧状态）
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        with torch.no_grad():
            # 初始化让 softplus(dt_bias) ≈ 0.1，即默认中等更新速度
            self.dt_proj.bias.fill_(math.log(math.exp(0.1) - 1))
        
        # A: 状态衰减矩阵（对角，log 空间存储保证正定）
        # A[i,j] 越大 → 该维度上的记忆衰减越快
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.A_log._no_weight_decay = True  # 不加 weight decay
        
        # B: 输入矩阵（input-dependent → selective）
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        # C: 输出矩阵（input-dependent → selective）
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        # D: 跳跃连接（直接从输入到输出）
        self.D = nn.Parameter(torch.ones(self.d_inner,dtype=torch.float32))
        self.D._no_weight_decay = True
        
        # 输出投影: d_inner → 2*d_kv (回到 K,V 空间)
        self.out_proj = nn.Linear(self.d_inner, 2 * d_kv, bias=False)
        self.norm = nn.RMSNorm(2 * d_kv)
        
    
    def forward(
        self,
        kv_input: torch.Tensor,                     # (batch, seq_len, 2*d_kv)
        ssm_state: Optional[torch.Tensor] = None,   # (batch, d_inner, d_state) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with CUDA kernel acceleration.
        """
        if not self.use_fast_path:
            return self._forward_pytorch(kv_input, ssm_state)
        
        try:
            return self._forward_cuda(kv_input, ssm_state)
        except Exception as e:
            print(f"⚠️  CUDA kernel failed: {e}")
            print(f"   Falling back to PyTorch implementation")
            self.use_fast_path = False
            return self._forward_pytorch(kv_input, ssm_state)
    
    def _forward_cuda(
        self,
        kv_input: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 selective_scan_cuda_oflex 的快速实现。
        
        selective_scan_fn 接口:
            selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, 
                            delta_softplus, return_last_state, nrows)
        
        参数形状:
            u:     (batch, d_inner, seqlen)
            delta: (batch, d_inner, seqlen)
            A:     (d_inner, d_state)
            B:     (batch, d_state, seqlen)
            C:     (batch, d_state, seqlen)
            D:     (d_inner,)
            z:     None (oflex 不支持)
        """
        if ssm_state is not None:
            if (
                selective_state_update is not None
                and kv_input.is_cuda
                and not torch.is_grad_enabled()
            ):
                return self._forward_state_update_cuda(kv_input, ssm_state)
            return self._forward_pytorch(kv_input, ssm_state)

        batch, seq_len, _ = kv_input.shape
        residual = kv_input
        input_dtype = kv_input.dtype
        
        kv_input = self.norm(kv_input)
        
        # 1. 投影 + 分出 x 和 gate z
        xz = self.in_proj(kv_input)              # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)               # 各 (B, L, d_inner)
        
        # 2. Conv
        x = x.transpose(1, 2).contiguous()       # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]       # (B, d_inner, L)
        x = F.silu(x)
        
        # ========== 准备 selective_scan_fn 的参数 ==========
        
        # u: 直接用 conv 后的 x
        u = x  # (B, d_inner, L)
        
        # delta: 从 x 计算
        # 方法：对每个时间步，用 x 的前 dt_rank 维度通过 dt_proj
        x_transposed = x.transpose(1, 2)                          # (B, L, d_inner)
        delta = self.dt_proj(x_transposed[:, :, :self.dt_rank])  # (B, L, d_inner)
        delta = delta.transpose(1, 2).contiguous()                # (B, d_inner, L)
        
        # A: 状态矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # B, C: 从 x 计算（input-dependent）
        B = self.B_proj(x_transposed).transpose(1, 2).contiguous()  # (B, d_state, L)
        C = self.C_proj(x_transposed).transpose(1, 2).contiguous()  # (B, d_state, L)
        
        # D: 跳跃连接
        D = self.D.float()   # (d_inner,)
        
        # 调用 CUDA kernel
        out, last_state = selective_scan_fn(
            u, delta, A, B, C, D,
            z=None,              # oflex 不支持 z，gate 在后面手动应用
            delta_bias=None,
            delta_softplus=True,
            return_last_state=True,
            nrows=1,             # 并行度，1-4 之间
        )
        # out: (B, d_inner, L)
        # last_state: (B, d_inner, d_state)
        
        # 3. 转回 (B, L, d_inner) + 应用 gate
        out = out.transpose(1, 2).contiguous()   # (B, L, d_inner)
        out = out * F.silu(z)                    # 手动应用 SiLU gate
        
        # 4. 输出投影
        output = self.out_proj(out)              # (B, L, 2*d_kv)
        output = output + residual
        
        last_state = last_state.to(input_dtype)
        
        return output, last_state

    def _forward_state_update_cuda(
        self,
        kv_input: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stateful inference path using Mamba's Triton selective_state_update.

        selective_scan_cuda/oflex can scan a sequence fast but cannot accept an
        initial state. This path updates the carried state token by token on GPU,
        preserving temporal history across absorbed segments.
        """
        batch, seq_len, _ = kv_input.shape
        residual = kv_input
        input_dtype = kv_input.dtype

        kv_input = self.norm(kv_input)
        xz = self.in_proj(kv_input)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)[:, :, :seq_len]
        x = F.silu(x).transpose(1, 2).contiguous()

        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        state = ssm_state.contiguous()
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t].contiguous()
            dt_t = self.dt_proj(x_t[:, : self.dt_rank]).contiguous()
            B_t = self.B_proj(x_t).contiguous()
            C_t = self.C_proj(x_t).contiguous()
            z_t = z[:, t].contiguous()
            y_t = selective_state_update(
                state,
                x_t,
                dt_t,
                A,
                B_t,
                C_t,
                D=D,
                z=z_t,
                dt_softplus=True,
            )
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        output = self.out_proj(y) + residual
        return output, state.to(input_dtype)

    def update_state_only(
        self,
        kv_input: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Update recurrent state without materializing SSM outputs.

        The compressor only reads the final hidden state for fusion, so this
        computes the recurrence in vectorized chunks instead of running a
        Python loop over every evicted token.
        """
        batch, seq_len, _ = kv_input.shape
        input_dtype = kv_input.dtype

        kv_input = self.norm(kv_input)
        xz = self.in_proj(kv_input)
        x, _ = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)[:, :, :seq_len]
        x = F.silu(x).transpose(1, 2).contiguous()

        if ssm_state is None:
            state = torch.zeros(
                batch,
                self.d_inner,
                self.d_state,
                device=kv_input.device,
                dtype=torch.float32,
            )
        else:
            state = ssm_state.float()

        A = -torch.exp(self.A_log.float())
        chunk_size = max(1, int(chunk_size))
        for start in range(0, seq_len, chunk_size):
            x_chunk = x[:, start : start + chunk_size]
            dt_input = x_chunk[..., : self.dt_rank].to(self.dt_proj.weight.dtype)
            b_input = x_chunk.to(self.B_proj.weight.dtype)
            dt = F.softplus(self.dt_proj(dt_input).float())
            B = self.B_proj(b_input).float()
            x_chunk = x_chunk.float()

            delta_a = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            delta_b = dt.unsqueeze(-1) * B.unsqueeze(2) * x_chunk.unsqueeze(-1)

            prod_all = delta_a.prod(dim=1)
            suffix_prod = torch.cumprod(torch.flip(delta_a, dims=[1]), dim=1)
            suffix_prod = torch.flip(suffix_prod, dims=[1])
            ones = torch.ones_like(suffix_prod[:, :1])
            suffix_after = torch.cat([suffix_prod[:, 1:], ones], dim=1)
            state = prod_all * state + (delta_b * suffix_after).sum(dim=1)

        return state.to(input_dtype)
    
    def _forward_pytorch(
        self,
        kv_input: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch 参考实现（慢速 fallback）。
        """
        batch, seq_len, _ = kv_input.shape
        residual = kv_input
        kv_input = self.norm(kv_input)
        
        xz = self.in_proj(kv_input)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        # SSM 循环
        if ssm_state is None:
            ssm_state = torch.zeros(
                batch, self.d_inner, self.d_state,
                device=kv_input.device, dtype=kv_input.dtype
            )
        
        A = -torch.exp(self.A_log)
        outputs = []
        
        for t in range(seq_len):
            # 计算 B, C, dt
            B = self.B_proj(x[:, t])              # (B, d_state)
            C = self.C_proj(x[:, t])              # (B, d_state)
            dt = F.softplus(self.dt_proj(x[:, t, :self.dt_rank]))  # (B, d_inner)
            
            # 离散化
            dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))       # (B, d_inner, d_state)
            dB = dt.unsqueeze(-1) * B.unsqueeze(1)                   # (B, d_inner, d_state)
            
            # 状态更新
            ssm_state = dA * ssm_state + dB * x[:, t].unsqueeze(-1)
            
            # 输出
            y = (ssm_state * C.unsqueeze(1)).sum(-1) + self.D * x[:, t]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)          # (B, L, d_inner)
        y = y * F.silu(z)                        # gate
        output = self.out_proj(y) + residual
        
        return output, ssm_state


class MemoryFusionLayer(nn.Module):
    """
    在每个 Transformer layer 中，将 SSM memory 融合到 attention 输出中。
    
    结构:
    1. Cross-Attention: Q = hidden_states, K/V = ssm_memory
    2. Gating: gate = σ(linear(hidden_states))
    3. Output: attn_output + gate * cross_attn_output
    
    设计思路：
    - Cross-attention 让当前 query 从 SSM memory 中检索相关历史信息
    - Gating 让模型学会"什么时候需要参考长期记忆"
    - 初始化 gate ≈ 0，这样未训练时模型行为 = 原始模型（安全）
    - 训练后 gate 会学到在需要时打开（如回顾性引用、长程依赖）
    
    
    优化版: 使用低秩投影减少参数量。
    
    原始: cross_attn 使用 full hidden_dim
    优化: 使用 bottleneck 投影到较低维度再做 cross-attention
    """
    def __init__(
        self,
        hidden_dim: int,         # 3584
        bottleneck_dim: int = 256,  #  低秩瓶颈
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 下投影: hidden_dim → bottleneck_dim
        self.q_down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.mem_down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        
        # Cross-attention 在低维空间
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        # 上投影: bottleneck_dim → hidden_dim
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # Norms
        self.norm_q = nn.RMSNorm(bottleneck_dim)
        self.norm_mem = nn.RMSNorm(bottleneck_dim)
        
        # Gate
        self.gate_proj = nn.Linear(hidden_dim, 1, bias=True)
        with torch.no_grad():
            self.gate_proj.bias.fill_(-5.0)
            nn.init.zeros_(self.gate_proj.weight)
    
    def forward(self, hidden_states, memory_states):
        """
        hidden_states: (B, q_len, hidden_dim) = 3584
        memory_states: (B, mem_len, hidden_dim) = 3584
        从 SSM memory 中检索信息并通过 gating 融合。
        
        Returns:
            memory_context: (B, q_len, hidden_dim) 要加到 attn_output 上的增量
        """
        #下投影
        q = self.norm_q(self.q_down(hidden_states))        # (B, q_len, 256)
        m = self.norm_mem(self.mem_down(memory_states))     # (B, mem_len, 256)
        
        context, _ = self.cross_attn(q, m, m)               # (B, q_len, 256)
        # 上投影
        context = self.up_proj(context)                      # (B, q_len, 3584)
        # Gate
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # (B, q_len, 1)
        
        return gate * context


class SSMCacheCompressor(nn.Module):
    """
    State-only temporal SSM 长期记忆系统。
    
    架构总结:
    ┌──────────────────────────────────────────────────────────────┐
    │ Per Transformer Layer:                                        │
    │                                                                │
    │   evicted frame KV ──→ SelectiveSSMLayer ──→ h_t              │
    │                                              │                │
    │   hidden_states ──→ MemoryFusionLayer ←──────┘                │
    │                         │                                      │
    │                         └──→ memory_context (加到 attn 输出)   │
    └──────────────────────────────────────────────────────────────┘

    这个版本按视频时间顺序递推，只保留每层一个 recurrent hidden state。
    不再保存 max_memory_len 个历史 memory tokens，因此历史存储不随视频长度增长。
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_dim: int,          # model hidden size，用于 MemoryFusionLayer
        d_state: int = 64,
        max_memory_len: int = 256,
        fusion_num_heads: int = 8,
        fusion_bottleneck: int = 256,
        layer_sharing: str = "group4",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.d_kv = num_kv_heads * head_dim
        # Kept for CLI/checkpoint compatibility. State-only temporal SSM does
        # not keep a rolling memory-token buffer.
        self.max_memory_len = max_memory_len
        
        # Layer sharing
        if layer_sharing == "none":
            n_unique = num_layers
            self._layer_map = list(range(num_layers))
        elif layer_sharing == "group4":
            n_unique = (num_layers + 3) // 4
            self._layer_map = [i // 4 for i in range(num_layers)]
        elif layer_sharing == "all":
            n_unique = 1
            self._layer_map = [0] * num_layers
        else:
            raise ValueError(f"Unknown layer_sharing: {layer_sharing}")
        
        # SSM layers: 处理被 evict 的 KV，更新长期记忆
        self.ssm_layers = nn.ModuleList([
            SelectiveSSMLayer(d_kv=self.d_kv, d_state=d_state,use_fast_path=HAS_CUDA_KERNEL)
            for _ in range(n_unique)
        ])
        
        # Memory → hidden_dim 投影
        # SSM 输出维度是 2*d_kv，需要投影到 hidden_dim 以便 cross-attention
        self.memory_up_projs = nn.ModuleList([
            nn.Linear(2 * self.d_kv, hidden_dim, bias=False)
            for _ in range(n_unique)
        ])
        
        # Fusion layers: 在 attention 后将 memory 融合到输出
        self.fusion_layers = nn.ModuleList([
            MemoryFusionLayer(
                hidden_dim=hidden_dim,
                num_heads=fusion_num_heads,
                bottleneck_dim=fusion_bottleneck,
            )
            for _ in range(n_unique)
        ])
        
        # Runtime states（非参数，不保存到 checkpoint）
        self._ssm_states: List[Optional[torch.Tensor]] = [None] * num_layers
        # 缓存投影后的 memory（避免重复计算）
        self._projected_memories: List[Optional[torch.Tensor]] = [None] * num_layers
        self._dirty: List[bool] = [True] * num_layers
    
    @torch.no_grad()
    def absorb(
        self, 
        layer_idx: int,
        evicted_keys: torch.Tensor,    # (B, num_kv_heads, evict_len, head_dim)
        evicted_values: torch.Tensor,  # (B, num_kv_heads, evict_len, head_dim)
    ):
        """
        将被 evict 的 KV 增量写入 SSM state。
        
        调用时机: prune_id_and_kv_cache 中，在实际删除 KV 之前调用。
        
        处理流程:
        1. reshape KV: (B,H,L,D) → (B,L,H*D) → concat → (B,L,2*H*D)
        2. SSM forward: 按时间顺序扫描 evicted tokens，更新 ssm_state
        3. fusion 时只从当前 ssm_state 投影出一个 memory token
        """
        batch, _, evict_len, _ = evicted_keys.shape
        
        # Reshape: (B, H, L, D) → (B, L, H*D)
        k_flat = evicted_keys.transpose(1, 2).reshape(batch, evict_len, -1)
        v_flat = evicted_values.transpose(1, 2).reshape(batch, evict_len, -1)
        kv_input = torch.cat([k_flat, v_flat], dim=-1)    # (B, L, 2*d_kv)
        
        ssm_idx = self._layer_map[layer_idx]
        
        new_state = self.ssm_layers[ssm_idx].update_state_only(
            kv_input,
            self._ssm_states[layer_idx],
        )
        
        # 更新隐状态（detach 断开计算图，避免内存泄漏）
        self._ssm_states[layer_idx] = new_state.detach()

        # 标记缓存失效
        self._dirty[layer_idx] = True
        self._projected_memories[layer_idx] = None

    def absorb_trainable(
        self,
        layer_idx: int,
        evicted_keys: torch.Tensor,
        evicted_values: torch.Tensor,
    ):
        """
        Training-time state update. Unlike absorb(), this keeps the computation
        graph so SelectiveSSMLayer, memory projection, and fusion parameters can
        be optimized from the language-model loss.
        """
        batch, _, evict_len, _ = evicted_keys.shape
        k_flat = evicted_keys.transpose(1, 2).reshape(batch, evict_len, -1)
        v_flat = evicted_values.transpose(1, 2).reshape(batch, evict_len, -1)
        kv_input = torch.cat([k_flat, v_flat], dim=-1)

        ssm_idx = self._layer_map[layer_idx]
        new_state = self.ssm_layers[ssm_idx].update_state_only(
            kv_input,
            self._ssm_states[layer_idx],
        )
        self._ssm_states[layer_idx] = new_state
        self._dirty[layer_idx] = True
        self._projected_memories[layer_idx] = None
    
    def get_memory_for_fusion(
        self,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        获取投影到 hidden_dim 的 memory states，用于 MemoryFusionLayer。
        
        Returns:
            memory: (B, mem_len, hidden_dim) or None
        """
        if self._ssm_states[layer_idx] is None:
            return None
        
        # 使用缓存避免重复投影
        if not self._dirty[layer_idx] and self._projected_memories[layer_idx] is not None:
            return self._projected_memories[layer_idx]
        
        ssm_idx = self._layer_map[layer_idx]
        # SelectiveSSMLayer state shape: (B, d_inner, d_state).
        # d_inner == 2*d_kv with the current default expand=2, so mean over
        # d_state gives one temporal memory token in the KV feature space.
        raw_memory = self._ssm_states[layer_idx].mean(dim=-1).unsqueeze(1)
        projected = self.memory_up_projs[ssm_idx](raw_memory)  # (B, mem_len, hidden_dim)

        if projected.requires_grad:
            return projected
        
        self._projected_memories[layer_idx] = projected.detach()
        self._dirty[layer_idx] = False
        
        return projected
    
    def fuse(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,    # (B, q_len, hidden_dim)
        attn_output: torch.Tensor,      # (B, q_len, hidden_dim)
    ) -> torch.Tensor:
        """
        将 SSM memory 融合到 attention 输出中。
        
        调用时机: 每个 decoder layer 的 self-attention 之后。
        
        如果没有 memory（还没有 evict 过），直接返回 attn_output。
        
        Returns:
            fused_output: (B, q_len, hidden_dim)
        """
        memory = self.get_memory_for_fusion(layer_idx)
        if memory is None:
            return attn_output
        
        ssm_idx = self._layer_map[layer_idx]
        memory_context = self.fusion_layers[ssm_idx](hidden_states, memory)
        
        return attn_output + memory_context
    
    def has_memory(self, layer_idx: int) -> bool:
        """检查指定层是否有 SSM memory。"""
        return self._ssm_states[layer_idx] is not None
    
    def reset(self):
        """新视频时重置所有状态。"""
        self._ssm_states = [None] * self.num_layers
        self._projected_memories = [None] * self.num_layers
        self._dirty = [True] * self.num_layers
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def memory_stats(self):
        """返回当前 memory 使用情况。"""
        stats = {}
        for i in range(self.num_layers):
            if self._ssm_states[i] is not None:
                stats[f"layer_{i}"] = {
                    "memory_len": 1,
                    "ssm_state_norm": self._ssm_states[i].norm().item() if self._ssm_states[i] is not None else 0,
                }
        return stats
