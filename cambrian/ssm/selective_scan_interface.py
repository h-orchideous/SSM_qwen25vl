"""
streaming_vlm/inference/ssm/selective_scan_interface.py

selective_scan_cuda_oflex 的 PyTorch 接口
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import os
import ctypes


def build_selective_scan_fn(selective_scan_cuda, mode="ssoflex"):
    """构建 selective_scan 函数"""
    SSOFLEX_FLOAT = True

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None,
                    delta_bias=None, delta_softplus=False,
                    return_last_state=False, nrows=1, backnrows=-1):
            """
            Args:
                u:      (batch, d_inner, seqlen)
                delta:  (batch, d_inner, seqlen)
                A:      (d_inner, d_state)
                B:      (batch, d_state, seqlen) or (d_inner, d_state)
                C:      (batch, d_state, seqlen) or (d_inner, d_state)
                D:      (d_inner,) or None
                z:      (batch, d_inner, seqlen) or None (oflex mode 不支持)
                delta_bias: (d_inner,) or None
                delta_softplus: bool
                return_last_state: bool
                nrows:  int (1-4)
                backnrows: int
            """
            # 确保内存连续
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()

            # B, C 维度处理：3D → 4D
            ctx.squeeze_B = False
            ctx.squeeze_C = False
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            # dtype 处理
            if D is not None and (D.dtype != torch.float):
                ctx._d_dtype = D.dtype
                D = D.float()
            if delta_bias is not None and (delta_bias.dtype != torch.float):
                ctx._delta_bias_dtype = delta_bias.dtype
                delta_bias = delta_bias.float()

            # nrows 参数验证
            assert nrows in [1, 2, 3, 4], f"nrows must be in [1,2,3,4], got {nrows}"
            backnrows = nrows if backnrows <= 0 else backnrows
            ctx.backnrows = backnrows
            ctx.delta_softplus = delta_softplus
            ctx.has_z = False  # oflex mode 不支持 z

            # 调用 CUDA kernel forward
            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D,
                delta_bias, delta_softplus,
                nrows, SSOFLEX_FLOAT
            )

            # 提取 last_state
            last_state = x[:, :, -1, 1::2]  # (batch, d_inner, d_state)
            
            # 保存用于反向传播
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            
            if not return_last_state:
                return out
            else:
                return out, last_state

        @staticmethod
        def backward(ctx, dout, *args):
            """
            反向传播
            
            注意：如果 return_last_state=True，会有两个梯度输入 (dout, dlast_state)
            但我们只使用 dout，因为 last_state 被 detach 了
            """
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors

            if dout.stride(-1) != 1:
                dout = dout.contiguous()

            # 调用 CUDA kernel backward
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias,
                dout, x, ctx.delta_softplus, ctx.backnrows
            )

            # 恢复 B, C 的维度
            if ctx.squeeze_B:
                dB = dB.squeeze(1)
            if ctx.squeeze_C:
                dC = dC.squeeze(1)

            # 恢复 D 的 dtype
            _dD = None
            if D is not None:
                if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                    _dD = dD.to(ctx._d_dtype)
                else:
                    _dD = dD

            # 恢复 delta_bias 的 dtype
            _ddelta_bias = None
            if delta_bias is not None:
                if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                    _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
                else:
                    _ddelta_bias = ddelta_bias

            # 返回 12 个梯度（对应 forward 的 12 个参数）
            return (
                du,              # u
                ddelta,          # delta
                dA,              # A
                dB,              # B
                dC,              # C
                _dD,             # D
                None,            # z (oflex 不支持)
                _ddelta_bias,    # delta_bias
                None,            # delta_softplus
                None,            # return_last_state
                None,            # nrows
                None,            # backnrows
            )

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                          delta_bias=None, delta_softplus=False,
                          return_last_state=False, nrows=1, backnrows=-1):
        """
        Selective scan 用户接口
        
        Args:
            u:      (batch, d_inner, seqlen)
            delta:  (batch, d_inner, seqlen)
            A:      (d_inner, d_state)
            B:      (batch, d_state, seqlen)
            C:      (batch, d_state, seqlen)
            D:      (d_inner,) or None
            z:      None (oflex 不支持，保留参数只是为了接口兼容)
            delta_bias: (d_inner,) or None
            delta_softplus: bool
            return_last_state: bool
            nrows:  int (1-4) - 并行度
            backnrows: int - backward 并行度
        
        Returns:
            out: (batch, d_inner, seqlen)
            last_state (optional): (batch, d_inner, d_state)
        """
        outs = SelectiveScanFn.apply(
            u, delta, A, B, C, D, z, delta_bias,
            delta_softplus, return_last_state, nrows, backnrows
        )
        
        if not return_last_state:
            return outs.to(u.dtype)
        else:
            return outs[0].to(u.dtype), outs[1]

    return selective_scan_fn


def _preload_torch_libs_for_extensions():
    """Preload torch shared libraries so custom CUDA extensions can resolve symbols."""
    try:
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        candidates = [
            "libc10.so",
            "libtorch_cpu.so",
            "libc10_cuda.so",
            "libtorch_cuda.so",
        ]
        for libname in candidates:
            libpath = os.path.join(torch_lib_dir, libname)
            if os.path.exists(libpath):
                ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        # Best effort preload; keep graceful fallback behavior.
        pass


def build_selective_scan_fn_standard(selective_scan_cuda):
    """Build selective_scan function for `selective_scan_cuda` (mamba-style kernel)."""

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None,
                    delta_bias=None, delta_softplus=False,
                    return_last_state=False, nrows=1, backnrows=-1):
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()

            ctx.squeeze_B = False
            ctx.squeeze_C = False
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            out_list = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            out = out_list[0]
            x = out_list[1]
            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None

            last_state = x[:, :, -1, 1::2]

            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                out_z = out_list[2]
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                return out_z if not return_last_state else (out_z, last_state)

        @staticmethod
        def backward(ctx, dout, *args):
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

            if dout.stride(-1) != 1:
                dout = dout.contiguous()

            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, z, delta_bias,
                dout, x, out, None, ctx.delta_softplus,
                False,
            )
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

            return (
                du,
                ddelta,
                dA,
                dB,
                dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None,
                None,
            )

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                          delta_bias=None, delta_softplus=False,
                          return_last_state=False, nrows=1, backnrows=-1):
        outs = SelectiveScanFn.apply(
            u, delta, A, B, C, D, z, delta_bias,
            delta_softplus, return_last_state, nrows, backnrows,
        )
        if not return_last_state:
            return outs.to(u.dtype)
        else:
            return outs[0].to(u.dtype), outs[1]

    return selective_scan_fn


# ========== 自动初始化 ==========
try:
    import selective_scan_cuda_oflex
    selective_scan_fn = build_selective_scan_fn(
        selective_scan_cuda_oflex,
        mode="ssoflex"
    )
    HAS_CUDA_KERNEL = True
    print("✓ selective_scan_interface: using selective_scan_cuda_oflex (CUDA, fast)")
except ImportError as e:
    try:
        _preload_torch_libs_for_extensions()
        import selective_scan_cuda
        selective_scan_fn = build_selective_scan_fn_standard(selective_scan_cuda)
        HAS_CUDA_KERNEL = True
        print("✓ selective_scan_interface: using selective_scan_cuda (CUDA, fast)")
    except Exception as e2:
        selective_scan_fn = None
        HAS_CUDA_KERNEL = False
        print(f"✗ selective_scan_interface: failed to load CUDA kernel")
        print(f"  Error (oflex): {e}")
        print(f"  Error (standard): {e2}")
        print(f"  Fallback: use PyTorch implementation (slow)")


__all__ = ['selective_scan_fn', 'HAS_CUDA_KERNEL']