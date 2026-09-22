"""
Softplus attention: O_i = n_i^-alpha * sum_j softplus(q_i . k_j * scale) v_j

Softmax is two things at once: an elementwise positive map and an l1 normalisation over
the row. This drops the normalisation and keeps a positive elementwise map, following the
scaled point-wise family n^-alpha h(S) of "Replacing softmax with ReLU in Vision
Transformers" (arXiv 2309.08586) with h = softplus, as in "Softplus Attention with
Re-weighting" (arXiv 2501.13428).

Why this is cheaper to compute than softmax attention, not just different:

  softmax needs a row sum, which is data dependent, so a tiled kernel has to carry a
  running max and a running sum and rescale the output accumulator every time the max
  moves -- the whole online-softmax algorithm that FlashAttention is built around.

  Here the only row-wise quantity is n_i, the number of keys query i attends to. Under
  causal masking with an optional left window that is min(i+1, window+1): pure index
  arithmetic, known before any data is read. So the tile loop is just

      acc += softplus(Q Kj^T * scale) @ Vj

  with no reductions, no rescaling and no LSE. The backward drops softmax's
  sum(dO * O) correction term too, since d/dx softplus(x) = sigmoid(x) acts elementwise.

alpha = 1 reproduces softmax's scale behaviour (output RMS falling with position as the
average is taken over more keys); alpha = 0.5 makes the output scale position
independent. Default 1.

Interface matches nanochat's flash_attn wrapper: q, k, v are (B, T, H, D), window_size is
the FA3 (left, right) tuple with -1 meaning unlimited, and only right = 0 (causal) is
supported.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    """softplus(s) = max(s, 0) + log(1 + exp(-|s|)).

    Algebraically identical to log(1 + exp(s)), but the exponent is never positive, so
    the exponential cannot overflow for any input and no branch or clamp on magnitude is
    needed. The branchy form this replaced (return s above a threshold) also dropped the
    log term there, which this keeps."""
    return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _fwd_kernel(
    Q, K, V, O,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sob, sot, soh,
    T, scale, alpha,
    WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // tl.num_programs(2) if False else bh  # bh packs (batch, head), see grid
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # causal: keys up to the last query in this tile. window: not before i - WINDOW.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(0, pid_m * BLOCK_M - WINDOW)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :]
        v_ptrs = V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :]
        k = tl.load(k_ptrs, mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        p = tl.where(keep, _softplus(s), 0.0)
        acc += tl.dot(p.to(v.dtype), v)

    # n_i: how many keys this query saw. Index arithmetic, no reduction.
    n = offs_m + 1
    if WINDOW >= 0:
        n = tl.minimum(n, WINDOW + 1)
    acc = acc * tl.exp(-alpha * tl.log(n.to(tl.float32)))[:, None]

    o_ptrs = O + off_b * sob + off_h * soh + offs_m[:, None] * sot + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)


def _fwd(q, k, v, window, alpha, scale):
    B, T, H, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M = BLOCK_N = 64
    grid = (triton.cdiv(T, BLOCK_M), B, H)
    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        T, scale, alpha,
        WINDOW=window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return o


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, DO, DK, DV,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sdob, sdot, sdoh,
    T, scale, alpha,
    WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per key tile: dK_j = scale * sum_i dS_ij q_i, dV_j = sum_i c_i P_ij dO_i."""
    pid_n = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                mask=offs_n[:, None] < T, other=0.0)
    v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                mask=offs_n[:, None] < T, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # causal: only queries at or after this key tile; window: not past j + WINDOW
    lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    hi = T
    if WINDOW >= 0:
        hi = tl.minimum(T, (pid_n + 1) * BLOCK_N + WINDOW)

    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                    mask=offs_m[:, None] < T, other=0.0)
        do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdot + offs_d[None, :],
                     mask=offs_m[:, None] < T, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T) & (offs_m[:, None] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)

        n = offs_m + 1
        if WINDOW >= 0:
            n = tl.minimum(n, WINDOW + 1)
        c = tl.exp(-alpha * tl.log(n.to(tl.float32)))[:, None]      # (BLOCK_M, 1)

        p = tl.where(keep, _softplus(s), 0.0) * c                    # c folded in for dV
        dv += tl.dot(tl.trans(p).to(do.dtype), do)

        dp = tl.dot(do, tl.trans(v)) * c                             # (BLOCK_M, BLOCK_N)
        ds = tl.where(keep, dp * tl.sigmoid(s), 0.0)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    tl.store(DK + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < T)
    tl.store(DV + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < T)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, DO, DQ,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sdob, sdot, sdoh,
    T, scale, alpha,
    WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per query tile: dQ_i = scale * sum_j dS_ij k_j."""
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdot + offs_d[None, :],
                 mask=offs_m[:, None] < T, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    n = offs_m + 1
    if WINDOW >= 0:
        n = tl.minimum(n, WINDOW + 1)
    c = tl.exp(-alpha * tl.log(n.to(tl.float32)))[:, None]

    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(0, pid_m * BLOCK_M - WINDOW)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        dp = tl.dot(do, tl.trans(v)) * c
        ds = tl.where(keep, dp * tl.sigmoid(s), 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * scale

    tl.store(DQ + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < T)


def _bwd(q, k, v, do, window, alpha, scale, BLOCK_M=64, BLOCK_N=64):
    """Two-kernel backward: one program per key tile for dK/dV, one per query tile for dQ.
    Measured faster than the single-pass atomic variant above (0.90 vs 1.32 ms at
    B4 T1024 H8 D128 window 255 on a GB10), because causal masking makes every later
    query tile contend for the same dK/dV tile."""
    B, T, H, D = q.shape
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    args = (q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            T, scale, alpha)
    kw = dict(WINDOW=window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
              num_warps=8, num_stages=2)
    _bwd_kv_kernel[(triton.cdiv(T, BLOCK_N), B, H)](q, k, v, do, dk, dv, *args, **kw)
    _bwd_q_kernel[(triton.cdiv(T, BLOCK_M), B, H)](q, k, v, do, dq, *args, **kw)
    return dq, dk, dv


class _SoftplusAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window, alpha, scale):
        o = _fwd(q, k, v, window, alpha, scale)
        ctx.save_for_backward(q, k, v)
        ctx.window, ctx.alpha, ctx.scale = window, alpha, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        B, T, H, D = q.shape
        do = do.contiguous()
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        BLOCK_M = BLOCK_N = 64
        args = (q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                do.stride(0), do.stride(1), do.stride(2),
                T, ctx.scale, ctx.alpha)
        kw = dict(WINDOW=ctx.window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                  num_warps=8, num_stages=2)
        _bwd_kv_kernel[(triton.cdiv(T, BLOCK_N), B, H)](q, k, v, do, dk, dv, *args, **kw)
        _bwd_q_kernel[(triton.cdiv(T, BLOCK_M), B, H)](q, k, v, do, dq, *args, **kw)
        return dq, dk, dv, None, None, None


# torch.library wrappers, for the same reason FA4 needs them: called directly from a
# compiled model, dynamo traces into the launcher, breaks the graph and costs more than
# the kernel saves. As an opaque op with a shape rule, inductor keeps one graph.
@torch.library.custom_op("softplus_attn::fwd", mutates_args=(), device_types="cuda")
def _op_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            window: int, alpha: float, scale: float) -> torch.Tensor:
    return _fwd(q, k, v, window, alpha, scale)


@_op_fwd.register_fake
def _(q, k, v, window, alpha, scale):
    return torch.empty_like(q)


@torch.library.custom_op("softplus_attn::bwd", mutates_args=(), device_types="cuda")
def _op_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor,
            window: int, alpha: float, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bwd(q, k, v, do.contiguous(), window, alpha, scale)


@_op_bwd.register_fake
def _(q, k, v, do, window, alpha, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, window, alpha, scale = inputs
    ctx.save_for_backward(q, k, v)
    ctx.window, ctx.alpha, ctx.scale = window, alpha, scale


def _backward(ctx, do):
    q, k, v = ctx.saved_tensors
    dq, dk, dv = torch.ops.softplus_attn.bwd(q, k, v, do, ctx.window, ctx.alpha, ctx.scale)
    return dq, dk, dv, None, None, None


torch.library.register_autograd("softplus_attn::fwd", _backward, setup_context=_setup_context)


def softplus_attn_func(q, k, v, causal=True, window_size=(-1, 0), alpha=1.0, softmax_scale=None):
    """FA3-shaped entry point. q, k, v are (B, T, H, D); only causal (right window 0)."""
    assert causal and window_size[1] in (0, None), "softplus attention is causal-only here"
    left = window_size[0]
    window = -1 if left is None or left < 0 or left >= k.size(1) else int(left)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return torch.ops.softplus_attn.fwd(q, k, v, window, float(alpha), float(scale))


def softplus_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                               causal=True, window_size=(-1, 0), alpha=1.0, softmax_scale=None):
    """Inference path. Materialises the scores: q is one token during decode and a prompt
    during prefill, so the Tq x context matrix is small and a kernel would not pay off."""
    B, Tq, H, D = q.shape
    pos = int(cache_seqlens[0].item())
    if k is not None and v is not None:
        k_cache[:, pos:pos + Tq] = k
        v_cache[:, pos:pos + Tq] = v
    end = pos + Tq
    ks, vs = k_cache[:, :end], v_cache[:, :end]
    scale = D ** -0.5 if softmax_scale is None else softmax_scale
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), ks.float()) * scale
    row = torch.arange(end - Tq, end, device=q.device)[:, None]
    col = torch.arange(end, device=q.device)[None, :]
    keep = col <= row
    left = window_size[0]
    if left is not None and 0 <= left < end:
        keep = keep & ((row - col) <= left)
        n = torch.clamp(row + 1, max=left + 1)
    else:
        n = row + 1
    p = torch.nn.functional.softplus(s) * keep
    o = torch.einsum("bhqk,bkhd->bqhd", p, vs.float())
    return (o * n.float().pow(-alpha)[None, :, None, :].squeeze(-1).unsqueeze(-1)).to(q.dtype)


@triton.jit
def _bwd_fused_kernel(
    Q, K, V, DO, DQ, DK, DV,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sdob, sdot, sdoh,
    T, scale, alpha,
    WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Single-pass backward: one program per query tile computes S once and produces dQ in
    registers while atomically accumulating dK and dV.

    This is only possible because the output is an unnormalised sum. With softmax, a key
    tile's contribution depends on the row's global max and sum, so partial results cannot
    be added into a shared buffer -- they need the separate combine pass FlashAttention
    does. Here dK_j = scale * sum_i dS_ij q_i is a plain sum over query tiles, so atomics
    work directly.

    The trade is contention (under causal masking a key tile is touched by every later
    query tile) and non-determinism, against computing Q K^T once instead of twice.
    DK and DV must be fp32 buffers: atomic_add on bf16 is not reliable across archs.
    """
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdot + offs_d[None, :],
                 mask=offs_m[:, None] < T, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    n = offs_m + 1
    if WINDOW >= 0:
        n = tl.minimum(n, WINDOW + 1)
    c = tl.exp(-alpha * tl.log(n.to(tl.float32)))[:, None]

    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(0, pid_m * BLOCK_M - WINDOW)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kmask = offs_n[:, None] < T
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                    mask=kmask, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                    mask=kmask, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale                       # computed once, used by all three
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T) & (offs_m[:, None] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)

        p = tl.where(keep, _softplus(s), 0.0) * c
        dp = tl.dot(do, tl.trans(v)) * c
        ds = tl.where(keep, dp * tl.sigmoid(s), 0.0)

        dq += tl.dot(ds.to(k.dtype), k) * scale
        tl.atomic_add(DV + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                      tl.dot(tl.trans(p).to(do.dtype), do), mask=kmask)
        tl.atomic_add(DK + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                      tl.dot(tl.trans(ds).to(q.dtype), q) * scale, mask=kmask)

    tl.store(DQ + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < T)


def _bwd_fused(q, k, v, do, window, alpha, scale, BLOCK_M=64, BLOCK_N=64):
    B, T, H, D = q.shape
    dq = torch.empty_like(q)
    dk = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)  # atomics need fp32
    dv = torch.zeros_like(dk)
    _bwd_fused_kernel[(triton.cdiv(T, BLOCK_M), B, H)](
        q, k, v, do, dq, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        T, scale, alpha,
        WINDOW=window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return dq, dk.to(q.dtype), dv.to(q.dtype)
