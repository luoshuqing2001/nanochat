"""Single-query softplus attention with fixed-size, independent KV tasks.

Each program reduces one KV chunk, then accumulates a D-vector with atomics or a separate reduction.
There is no query padding, host work table, or synchronization between programs.
This is an inference-only path; prefill and backward use the existing kernels.
"""

import torch
import triton
import triton.language as tl
from nanochat.softplus_math import softplus_pair


@triton.jit
def _decode_kernel(
    Q, K, V, O,
    qb, qh, qd, kb, kt, kh, kd, vb, vt, vh, vd,
    T, START, SCALE, ROW_SCALE,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BD: tl.constexpr, BV: tl.constexpr, CHUNK: tl.constexpr,
    MODE: tl.constexpr, SPLITS: tl.constexpr,
):
    chunk, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    n = START + chunk * CHUNK + tl.arange(0, CHUNK)
    d = tl.arange(0, BD)
    q = tl.load(Q + b * qb + h * qh + d * qd, d < D, 0).to(tl.float32)
    k = tl.load(K + b * kb + h * kh + n[:, None] * kt + d[None, :] * kd,
                (n[:, None] < T) & (d[None, :] < D), 0).to(tl.float32)
    s = tl.sum(k * q[None, :], 1) * SCALE
    p, _ = softplus_pair(s)
    dv = tl.arange(0, BV)
    v = tl.load(V + b * vb + h * vh + n[:, None] * vt + dv[None, :] * vd,
                (n[:, None] < T) & (dv[None, :] < DV), 0).to(tl.float32)
    out = tl.sum(p[:, None] * v, 0) * ROW_SCALE
    ptr = O + (bh * SPLITS + chunk if MODE == 2 else bh) * DV + dv
    if MODE == 1:
        tl.atomic_add(ptr, out, dv < DV, sem="relaxed")
    else:
        tl.store(ptr, out, dv < DV)


@triton.jit
def _combine(P, O, S:tl.constexpr, D:tl.constexpr, BS:tl.constexpr, BD:tl.constexpr):
    h=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,BD)
    p=tl.load(P+(h*S+s[:,None])*D+d[None,:],(s[:,None]<S)&(d[None,:]<D),0.)
    tl.store(O+h*D+d,tl.sum(p,0),d<D)


def softplus_decode(q, k, v, window_left=None, alpha=1.0, scale=None, chunk_size=None,
                    reduction="auto", num_warps=None):
    """Bottom-right causal attention for Tq=1, with equal Q/K/V head counts.

    Supports strided cache views, unequal QK/V dimensions, and arbitrary positive
    cache lengths. Only the final KV chunk is shorter. FP32 accumulation retains
    accuracy across many independent chunks; a single chunk writes output directly.
    """
    b, tq, h, d = q.shape
    t, dv = k.shape[1], v.shape[-1]
    if tq != 1 or t <= 0 or k.shape != (b, t, h, d) or v.shape[:3] != (b, t, h):
        raise ValueError("softplus_decode requires Tq=1 and matching batch/head/cache sizes")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, v must be on the same CUDA device")
    start = max(0, t - window_left - 1) if window_left is not None and window_left >= 0 else 0
    n = t - start
    if chunk_size is None:
        # Short windows need small tasks to expose enough parallelism even under
        # CUDA Graph replay. Tiny caches fit one task and skip zero/cast entirely.
        chunk_size = min(64, max(32, triton.next_power_of_2(n))) if n <= 1024 else 256
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of two")
    splits = triton.cdiv(n, chunk_size)
    if reduction not in ("auto","atomic","partial"):
        raise ValueError("reduction must be auto, atomic, or partial")
    if reduction == "auto":
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        reduction = "partial" if splits <= 32 and b*h <= max(1, sms//4) else "atomic"
    mode = 0 if splits == 1 else (1 if reduction == "atomic" else 2)
    shape = (b, 1, h, dv)
    if mode == 2:
        out = torch.empty((b*h,splits,dv),device=q.device,dtype=torch.float32)
    else:
        out = (torch.zeros(shape, device=q.device, dtype=torch.float32) if mode == 1
               else torch.empty(shape, device=q.device, dtype=q.dtype))
    _decode_kernel[(splits, b * h)](
        q, k, v, out,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        t, start, d ** -0.5 if scale is None else scale, n ** -alpha,
        H=h, D=d, DV=dv, BD=triton.next_power_of_2(d), BV=triton.next_power_of_2(dv),
        CHUNK=chunk_size, MODE=mode, SPLITS=splits,
        num_warps=num_warps or (8 if chunk_size >= 256 else 4),
    )
    if mode == 2:
        final=torch.empty(shape,device=q.device,dtype=q.dtype)
        _combine[(b*h,)](out,final,splits,dv,triton.next_power_of_2(splits),
                         triton.next_power_of_2(dv),num_warps=4 if splits*dv<=8192 else 8)
        return final
    return out.to(q.dtype)
