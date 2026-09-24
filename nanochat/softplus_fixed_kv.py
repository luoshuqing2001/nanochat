"""Fixed-length KV tasks for causal softplus prefill and training.

Tasks cover only the causal/window band. Every task owns a query tile and an
absolute, fixed-length KV interval; outputs from different intervals add directly.
Only multiply-owned forward rows use FP32 atomics. Backward recomputes each score
tile once for all three gradients, accumulating partial gradients atomically.
"""
from functools import lru_cache
import math

import torch
import triton
import triton.language as tl


def auto_fixed_kv_chunk(q, k, v, window=-1):
    """Few-head full causal tasks, sized continuously from length and SM parallelism."""
    if (q.shape[1] < 1024 or q.shape[1] != k.shape[1] or window >= 0
            or q.shape[-1] != 128 or q.dtype not in (torch.bfloat16,torch.float16)
            or q.shape != k.shape or k.shape != v.shape
            or any(x.stride(-1) != 1 or x.dtype != q.dtype or x.device != q.device for x in (q, k, v))):
        return None
    props = torch.cuda.get_device_properties(q.device)
    bh=q.shape[0]*q.shape[2]
    if props.major != 12 or bh > max(1,props.multi_processor_count//24):
        return None
    target=q.shape[1]/8*math.sqrt(bh*48/props.multi_processor_count)
    return min(triton.next_power_of_2(q.shape[1]),triton.next_power_of_2(max(256,math.ceil(target))))


@lru_cache(maxsize=128)
def _plan(tq, tk, window, bm, chunk, device):
    tasks, multi = [], []
    for m, begin in enumerate(range(0, tq, bm)):
        hi = min(begin + bm + tk - tq, tk)
        lo = max(0, begin + tk - tq - window) if window >= 0 else 0
        starts = list(range(lo // chunk * chunk, hi, chunk))
        acc_row = len(multi) * bm if len(starts) > 1 else -1
        if acc_row >= 0:
            multi.append(m)
        tasks.extend((m, n, acc_row) for n in starts)
    # Visit successive diagonals. Adjacent tasks mostly write different Q and KV
    # rows rather than immediately contending for the same accumulator cache lines.
    tasks.sort(key=lambda x: (x[0] * bm // chunk - x[1] // chunk, x[0]))
    return (torch.tensor(tasks, device=device, dtype=torch.int32),
            torch.tensor(multi, device=device, dtype=torch.int32))


@lru_cache(maxsize=128)
def _backward_plan(tq, tk, window, bn, chunk, device):
    tasks, multi = [], []
    for ni, begin in enumerate(range(0, tk, bn)):
        lo = max(0, begin - (tk - tq))
        hi = min(tq, begin + bn + window - (tk - tq)) if window >= 0 else tq
        starts = list(range(lo // chunk * chunk, hi, chunk))
        ai = len(multi) * bn if len(starts) > 1 else -1
        if ai >= 0:
            multi.append(ni)
        tasks.extend((ni, m, ai) for m in starts or [0])
    tasks.sort(key=lambda x: (x[1] // chunk - x[0] * bn // chunk, x[0]))
    return (torch.tensor(tasks, device=device, dtype=torch.int32),
            torch.tensor(multi, device=device, dtype=torch.int32))


@triton.jit
def _map(s):
    y = tl.exp2(-tl.abs(s) * 1.4426950408889634)
    # Same log1p polynomial as the existing CuTe score map (absolute error <1.4e-5).
    p = 0.10028652 + y * -0.0236890026
    p = -0.208668975 + y * p
    p = 0.324411505 + y * p
    p = -0.499187794 + y * p
    p = 0.999981869 + y * p
    sp = tl.maximum(s, 0.) + y * p
    # Reuse the exponential and avoid cancellation in 1-exp(-softplus(s)).
    sig = tl.where(s >= 0., 1. / (1. + y), y / (1. + y))
    return sp, sig


@triton.jit
def _count_scale(m, TK: tl.constexpr, TQ: tl.constexpr, W: tl.constexpr, ALPHA: tl.constexpr):
    n = tl.minimum(m + TK - TQ + 1, TK)
    if W >= 0:
        n = tl.minimum(n, W + 1)
    n = tl.maximum(n, 1).to(tl.float32)
    if ALPHA == 1.:
        return 1. / n
    elif ALPHA == 0.:
        return tl.full(n.shape, 1., tl.float32)
    else:
        return tl.exp2(tl.log2(n) * -ALPHA)


@triton.jit
def _fwd(Q, K, V, O, ACC, TASKS,
         qb, qt, qh, kb, kt, kh, vb, vt, vh,
         TQ: tl.constexpr, TK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         AR: tl.constexpr, W: tl.constexpr, ALPHA: tl.constexpr, SCALE: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr):
    task, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    mi = tl.load(TASKS + task * 3)
    begin = tl.load(TASKS + task * 3 + 1)
    ai = tl.load(TASKS + task * 3 + 2)
    m = mi * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + b * qb + h * qh + m[:, None] * qt + d[None, :], m[:, None] < TQ, 0)
    acc = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum(tl.minimum(begin + CHUNK, (mi + 1) * BM + TK - TQ), TK)
    lo = begin
    if W >= 0:
        lo = tl.maximum(lo, tl.maximum(0, (mi * BM + TK - TQ - W) // BN * BN))
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        k = tl.load(K + b * kb + h * kh + n[:, None] * kt + d[None, :], n[:, None] < TK, 0)
        v = tl.load(V + b * vb + h * vh + n[:, None] * vt + d[None, :], n[:, None] < TK, 0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        keep = (n[None, :] <= m[:, None] + TK - TQ) & (n[None, :] < TK)
        if W >= 0:
            keep &= n[None, :] >= m[:, None] + TK - TQ - W
        p, _ = _map(s)
        acc += tl.dot(tl.where(keep, p, 0.).to(v.dtype), v)
    acc *= _count_scale(m, TK, TQ, W, ALPHA)[:, None]
    if ai < 0:
        dest = O + ((b * TQ + m[:, None]) * H + h) * D + d[None, :]
        tl.store(dest, acc, m[:, None] < TQ)
    else:
        rows = ai + tl.arange(0, BM)
        acc_dest = ACC + ((b * AR + rows[:, None]) * H + h) * D + d[None, :]
        tl.atomic_add(acc_dest, acc, m[:, None] < TQ, sem="relaxed")


@triton.jit
def _finish(ACC, O, MULTI, TQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            AR: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    m = tl.load(MULTI + tile) * BM
    i = tl.arange(0, BLOCK)
    r, d = i // D, i % D
    mask = (r < BM) & (m + r < TQ)
    x = tl.load(ACC + ((b * AR + tile * BM + r) * H + h) * D + d, mask, 0.)
    tl.store(O + ((b * TQ + m + r) * H + h) * D + d, x, mask)


@triton.jit
def _bwd(Q, K, V, DO, DQ, DK, DV, TASKS,
         qb, qt, qh, kb, kt, kh, vb, vt, vh, gb, gt, gh,
         TQ: tl.constexpr, TK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         W: tl.constexpr, ALPHA: tl.constexpr, SCALE: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr):
    task, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    mi = tl.load(TASKS + task * 3)
    begin = tl.load(TASKS + task * 3 + 1)
    m, d = mi * BM + tl.arange(0, BM), tl.arange(0, D)
    q = tl.load(Q + b * qb + h * qh + m[:, None] * qt + d[None, :], m[:, None] < TQ, 0)
    do = tl.load(DO + b * gb + h * gh + m[:, None] * gt + d[None, :], m[:, None] < TQ, 0)
    c = _count_scale(m, TK, TQ, W, ALPHA)[:, None]
    dq = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum(tl.minimum(begin + CHUNK, (mi + 1) * BM + TK - TQ), TK)
    lo = begin
    if W >= 0:
        lo = tl.maximum(lo, tl.maximum(0, (mi * BM + TK - TQ - W) // BN * BN))
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        k = tl.load(K + b * kb + h * kh + n[:, None] * kt + d[None, :], n[:, None] < TK, 0)
        v = tl.load(V + b * vb + h * vh + n[:, None] * vt + d[None, :], n[:, None] < TK, 0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        keep = ((n[None, :] <= m[:, None] + TK - TQ) & (n[None, :] < TK)
                & (m[:, None] < TQ))
        if W >= 0:
            keep &= n[None, :] >= m[:, None] + TK - TQ - W
        p, sig = _map(s)
        p = tl.where(keep, p * c, 0.)
        dp = tl.dot(do, tl.trans(v))
        ds = tl.where(keep, dp * c * sig, 0.)
        dq += tl.dot(ds.to(k.dtype), k) * SCALE
        dk = tl.dot(tl.trans(ds).to(q.dtype), q) * SCALE
        dv = tl.dot(tl.trans(p).to(do.dtype), do)
        off = ((b * TK + n[:, None]) * H + h) * D + d[None, :]
        tl.atomic_add(DK + off, dk, n[:, None] < TK, sem="relaxed")
        tl.atomic_add(DV + off, dv, n[:, None] < TK, sem="relaxed")
    off = ((b * TQ + m[:, None]) * H + h) * D + d[None, :]
    tl.atomic_add(DQ + off, dq, m[:, None] < TQ, sem="relaxed")


def _validate(q, k, v, chunk, bm):
    b, tq, h, d = q.shape
    tk = k.shape[1]
    if not (0 < tq <= tk and b > 0 and h > 0 and d in (64, 128)):
        raise ValueError("fixed KV requires 0 < Tq <= Tk, nonempty batch/heads, and D64/D128")
    if k.shape != (b, tk, h, d) or v.shape != k.shape:
        raise ValueError("fixed KV requires equal Q/K/V head counts and dimensions")
    if any(x.device != q.device or not x.is_cuda or x.dtype != q.dtype or x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("Q/K/V require matching CUDA device/dtype and contiguous feature dimensions")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("fixed KV supports BF16/FP16")
    if bm not in (32, 64, 128) or chunk < 64 or chunk % 64:
        raise ValueError("query tile must be 32/64/128 and KV chunk a positive multiple of 64")


def fixed_kv_forward(q, k, v, window=-1, alpha=1., scale=None, chunk=256, bm=64):
    _validate(q, k, v, chunk, bm)
    b, tq, h, d = q.shape
    tk = k.shape[1]
    tasks, multi = _plan(tq, tk, window, bm, chunk, str(q.device))
    ar = multi.numel() * bm
    out = torch.empty((b, tq, h, d), device=q.device, dtype=q.dtype)
    acc = torch.zeros((b, ar, h, d), device=q.device, dtype=torch.float32) if ar else out
    _fwd[(tasks.shape[0], b * h)](
        q, k, v, out, acc, tasks, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        tq, tk, h, d, ar, window, alpha, d ** -.5 if scale is None else scale,
        bm, 64, chunk, num_warps=4 if bm <= 64 else 8, num_stages=3)
    if ar:
        _finish[(multi.numel(), b * h)](acc, out, multi, tq, h, d, ar, bm,
                                       triton.next_power_of_2(bm * d), num_warps=4)
    return out


def fixed_kv_backward(q, k, v, do, window=-1, alpha=1., scale=None, chunk=256, bm=64):
    _validate(q, k, v, chunk, bm)
    b, tq, h, d = q.shape
    tk = k.shape[1]
    tasks, _ = _plan(tq, tk, window, bm, chunk, str(q.device))
    do = do.contiguous()
    nq, nk = q.numel(), k.numel()
    buf = torch.zeros(nq + 2 * nk, device=q.device, dtype=torch.float32)
    dq, dk, dv = buf[:nq], buf[nq:nq + nk], buf[nq + nk:]
    _bwd[(tasks.shape[0], b * h)](
        q, k, v, do, dq, dk, dv, tasks,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3],
        tq, tk, h, d, window, alpha, d ** -.5 if scale is None else scale,
        bm, 64, chunk, num_warps=8, num_stages=1)
    buf = buf.to(q.dtype)
    return buf[:nq].view(q.shape), buf[nq:nq + nk].view(k.shape), buf[nq + nk:].view(v.shape)


@triton.jit
def _bwd_kv(Q, K, V, DO, DQ, DK, DV, AK, AV, TASKS,
            qb, qt, qh, kb, kt, kh, vb, vt, vh, gb, gt, gh,
            TQ: tl.constexpr, TK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            AR: tl.constexpr, W: tl.constexpr, ALPHA: tl.constexpr, SCALE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr):
    task, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    ni = tl.load(TASKS + task * 3)
    begin = tl.load(TASKS + task * 3 + 1)
    ai = tl.load(TASKS + task * 3 + 2)
    n, d = ni * BN + tl.arange(0, BN), tl.arange(0, D)
    k = tl.load(K + b * kb + h * kh + n[:, None] * kt + d[None, :], n[:, None] < TK, 0)
    v = tl.load(V + b * vb + h * vh + n[:, None] * vt + d[None, :], n[:, None] < TK, 0)
    dk, dv = tl.zeros((BN, D), tl.float32), tl.zeros((BN, D), tl.float32)
    lo = tl.maximum(begin, tl.maximum(0, (ni * BN - (TK - TQ)) // BM * BM))
    hi = tl.minimum(begin + CHUNK, TQ)
    if W >= 0:
        hi = tl.minimum(hi, (ni + 1) * BN + W - (TK - TQ))
    for m0 in range(lo, hi, BM):
        m = m0 + tl.arange(0, BM)
        q = tl.load(Q + b * qb + h * qh + m[:, None] * qt + d[None, :], m[:, None] < TQ, 0)
        do = tl.load(DO + b * gb + h * gh + m[:, None] * gt + d[None, :], m[:, None] < TQ, 0)
        c = _count_scale(m, TK, TQ, W, ALPHA)[:, None]
        s = tl.dot(q, tl.trans(k)) * SCALE
        keep = ((n[None, :] <= m[:, None] + TK - TQ) & (n[None, :] < TK)
                & (m[:, None] < TQ))
        if W >= 0:
            keep &= n[None, :] >= m[:, None] + TK - TQ - W
        p, sig = _map(s)
        p = tl.where(keep, p * c, 0.)
        ds = tl.where(keep, tl.dot(do, tl.trans(v)) * c * sig, 0.)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dq = tl.dot(ds.to(k.dtype), k) * SCALE
        off = ((b * TQ + m[:, None]) * H + h) * D + d[None, :]
        tl.atomic_add(DQ + off, dq, m[:, None] < TQ, sem="relaxed")
    dk *= SCALE
    if ai < 0:
        off = ((b * TK + n[:, None]) * H + h) * D + d[None, :]
        tl.store(DK + off, dk, n[:, None] < TK)
        tl.store(DV + off, dv, n[:, None] < TK)
    else:
        r = ai + tl.arange(0, BN)
        offa = ((b * AR + r[:, None]) * H + h) * D + d[None, :]
        tl.atomic_add(AK + offa, dk, n[:, None] < TK, sem="relaxed")
        tl.atomic_add(AV + offa, dv, n[:, None] < TK, sem="relaxed")


def fixed_kv_backward_grouped(q, k, v, do, window=-1, alpha=1., scale=None,
                              chunk=1024, bn=64, bm=64):
    """Group query tiles per fixed KV block; only split owners atomically write dK/dV."""
    _validate(q, k, v, chunk, bm)
    b, tq, h, d = q.shape
    tk = k.shape[1]
    tasks, multi = _backward_plan(tq, tk, window, bn, chunk, str(q.device))
    ar = multi.numel() * bn
    nq, na = q.numel(), b * ar * h * d
    accum = torch.zeros(nq + 2 * na, device=q.device, dtype=torch.float32)
    dq, ak, av = accum[:nq], accum[nq:nq + na], accum[nq + na:]
    dk, dv = [torch.empty((b, tk, h, d), device=q.device, dtype=q.dtype) for _ in range(2)]
    do = do.contiguous()
    _bwd_kv[(tasks.shape[0], b * h)](
        q, k, v, do, dq, dk, dv, ak, av, tasks,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3],
        tq, tk, h, d, ar, window, alpha, d ** -.5 if scale is None else scale,
        bm, bn, chunk, num_warps=8, num_stages=1)
    if ar:
        for a, out in ((ak, dk), (av, dv)):
            _finish[(multi.numel(), b * h)](a, out, multi, tk, h, d, ar, bn,
                                           triton.next_power_of_2(bn * d), num_warps=4)
    return dq.view(q.shape).to(q.dtype), dk, dv


@torch.library.custom_op("softplus_fixed_kv::fwd", mutates_args=(), device_types="cuda")
def _op_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
            alpha: float, scale: float, kv_chunk: int, q_chunk: int) -> torch.Tensor:
    return fixed_kv_forward(q, k, v, window, alpha, scale, kv_chunk, bm=128)


@_op_fwd.register_fake
def _(q, k, v, window, alpha, scale, kv_chunk, q_chunk):
    return torch.empty_like(q)


@torch.library.custom_op("softplus_fixed_kv::bwd", mutates_args=(), device_types="cuda")
def _op_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor,
            window: int, alpha: float, scale: float, q_chunk: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fixed_kv_backward_grouped(q, k, v, do, window, alpha, scale, q_chunk)


@_op_bwd.register_fake
def _(q, k, v, do, window, alpha, scale, q_chunk):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, ctx.window, ctx.alpha, ctx.scale, _kv_chunk, ctx.q_chunk = inputs
    ctx.save_for_backward(q, k, v)


def _backward(ctx, do):
    q, k, v = ctx.saved_tensors
    dq, dk, dv = torch.ops.softplus_fixed_kv.bwd(q, k, v, do, ctx.window,
                                              ctx.alpha, ctx.scale, ctx.q_chunk)
    return dq, dk, dv, None, None, None, None, None


torch.library.register_autograd("softplus_fixed_kv::fwd", _backward, setup_context=_setup)


def fixed_kv_attention(q, k, v, window=-1, alpha=1., scale=None, kv_chunk=1024, q_chunk=1024):
    """Differentiable, compile-compatible fixed-KV prefill/training entry point."""
    return torch.ops.softplus_fixed_kv.fwd(q, k, v, int(window), float(alpha),
                                          q.shape[-1] ** -.5 if scale is None else float(scale),
                                          int(kv_chunk), int(q_chunk))
