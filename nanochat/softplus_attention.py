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

import os
import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    """softplus(s) = max(s, 0) + log(1 + exp(-|s|)), evaluated in base 2.

    The max/|s| form keeps the exponent non-positive, so the exponential cannot overflow
    for any input, with no branch or clamp on magnitude. Writing it with exp2/log2 maps
    onto the hardware's ex2/lg2 instructions directly; exp/log lower to the same
    instructions plus a multiply each.

    Note softplus costs two transcendentals per score against softmax's one exp, which is
    a structural cost of the score map, not of this implementation."""
    # constants inlined: a @triton.jit body cannot capture module-level floats
    return tl.maximum(x, 0.0) + tl.log2(1.0 + tl.exp2(-tl.abs(x) * 1.4426950408889634)) * 0.6931471805599453


@triton.jit
def _row_scale(offs_m, alpha, WINDOW: tl.constexpr):
    """c_i = n_i^-alpha with n_i = min(i+1, WINDOW+1). Index arithmetic, no reduction --
    this is what replaces softmax's row sum, and why no tile ever needs the others."""
    n = offs_m + 1
    if WINDOW >= 0:
        n = tl.minimum(n, WINDOW + 1)
    return tl.exp(-alpha * tl.log(n.to(tl.float32)))[:, None]


@triton.jit
def _fwd_kernel(
    Q, K, V, O,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sob, sot, soh,
    T, scale, alpha,
    REVERSE: tl.constexpr, WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if REVERSE:
        # Causal work grows with the query index, so the last tiles are the long ones;
        # issuing them first leaves only short tiles in the tail. Free -- an index
        # permutation, bit-identical results -- and worth 17% when there are too few
        # programs for the scheduler to hide the imbalance. See _reverse_tiles().
        pid_m = tl.num_programs(0) - 1 - pid_m
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    diag = pid_m * BLOCK_M                      # first key index on the diagonal tile
    hi = tl.minimum(diag + BLOCK_M, T)
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(0, diag - WINDOW)
        lo = (lo // BLOCK_N) * BLOCK_N

    # Bulk: every key in these tiles is strictly before every query in this tile, so the
    # causal comparison is known true and only the window edge can bite. Splitting it out
    # keeps the compare-and-select off the tiles that do not need it -- at T=2048 with
    # 64-wide tiles a mid-sequence query tile has ~16 key tiles and only one is diagonal.
    for start_n in range(lo, diag, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :])
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :])
        p = _softplus(tl.dot(q, tl.trans(k)) * scale)
        if WINDOW >= 0:
            p = tl.where((offs_m[:, None] - offs_n[None, :]) <= WINDOW, p, 0.0)
        acc += tl.dot(p.to(v.dtype), v)

    # Diagonal tile: the only one that needs the causal mask.
    for start_n in range(diag, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kmask = offs_n[:, None] < T
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                    mask=kmask, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                    mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        acc += tl.dot(tl.where(keep, _softplus(s), 0.0).to(v.dtype), v)

    acc = acc * _row_scale(offs_m, alpha, WINDOW)
    tl.store(O + off_b * sob + off_h * soh + offs_m[:, None] * sot + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, DO, DK, DV,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sdob, sdot, sdoh,
    T, scale, alpha,
    REVERSE: tl.constexpr, WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per key tile: dK_j = scale * sum_i dS_ij q_i, dV_j = sum_i c_i P_ij dO_i."""
    pid_n = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    kmask = offs_n[:, None] < T
    k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                mask=kmask, other=0.0)
    v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                mask=kmask, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    diag = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    hi = T
    if WINDOW >= 0:
        hi = tl.minimum(T, (pid_n + 1) * BLOCK_N + WINDOW)

    for start_m in range(diag, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mmask = offs_m[:, None] < T
        q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                    mask=mmask, other=0.0)
        do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdot + offs_d[None, :],
                     mask=mmask, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T) & mmask
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        c = _row_scale(offs_m, alpha, WINDOW)

        sp = _softplus(s)
        # sigmoid(s) = 1 - exp(-softplus(s)): reuses the softplus already computed instead
        # of a second transcendental pair. Exact, and sp >= 0 keeps the exponent negative.
        sig = 1.0 - tl.exp(-sp)

        p = tl.where(keep, sp, 0.0) * c
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v)) * c
        ds = tl.where(keep, dp * sig, 0.0)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    tl.store(DK + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=kmask)
    tl.store(DV + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=kmask)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, DO, DQ,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sdob, sdot, sdoh,
    T, scale, alpha,
    REVERSE: tl.constexpr, WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per query tile: dQ_i = scale * sum_j dS_ij k_j."""
    pid_m = tl.program_id(0)
    if REVERSE:
        pid_m = tl.num_programs(0) - 1 - pid_m
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdot + offs_d[None, :],
                 mask=offs_m[:, None] < T, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    c = _row_scale(offs_m, alpha, WINDOW)

    diag = pid_m * BLOCK_M
    hi = tl.minimum(diag + BLOCK_M, T)
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(0, diag - WINDOW)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start_n in range(lo, diag, BLOCK_N):          # bulk, causal known true
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :])
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        ds = tl.dot(do, tl.trans(v)) * c * tl.sigmoid(s)
        if WINDOW >= 0:
            ds = tl.where((offs_m[:, None] - offs_n[None, :]) <= WINDOW, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * scale

    for start_n in range(diag, hi, BLOCK_N):          # diagonal tile
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kmask = offs_n[:, None] < T
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                    mask=kmask, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                    mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        ds = tl.where(keep, tl.dot(do, tl.trans(v)) * c * tl.sigmoid(s), 0.0)
        dq += tl.dot(ds.to(k.dtype), k) * scale

    tl.store(DQ + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < T)


@triton.jit
def _fwd_splitk_kernel(
    Q, K, V, O32,
    sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sob, sot, soh,
    T, scale, H: tl.constexpr, WINDOW: tl.constexpr, SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Forward with the key range split across SPLITS programs per query tile, each
    accumulating its partial output with atomic_add into an fp32 buffer.

    Causal masking makes work per query tile grow with its index -- tile 0 reads one key
    tile, tile 31 reads 32 -- so one program per query tile is badly balanced. Splitting
    the key range evens that out, and softplus is what allows it: the output is a plain
    sum, so partial results can be added into a shared buffer. Softmax cannot, since each
    tile's contribution depends on the row's global max and sum, which is why
    FlashAttention writes per-split (O, m, l) and runs a combine pass.

    Whether it pays is a race between balance, which is bounded, and atomic traffic,
    which is SPLITS times the output volume. Measured on a GB10 (48 SMs), full context,
    forward only, against one program per query tile:

        B8  T2048  H10   2560 programs   1.82 ms   ->  4.49 (x2)  7.44 (x16)
        B1  T8192  H4     512 programs   1.51 ms   ->  1.87 (x2)  2.70 (x16)
        B1  T8192  H1     128 programs   0.57 ms   ->  0.49 (x2)  0.65 (x16)
        B1 T16384  H1     256 programs   1.69 ms   ->  1.66 (x2)  1.99 (x16)

    So it wins only when query-tile programs are within roughly 3x the SM count, and only
    at 2 to 4 splits: long-context decoding or a batch of one, not pretraining. Hence
    splits=1 by default.
    """
    pid_m = tl.program_id(0)
    sid = tl.program_id(1)
    bh = tl.program_id(2)
    off_b = bh // H
    off_h = bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    diag = pid_m * BLOCK_M
    hi_all = tl.minimum(diag + BLOCK_M, T)
    lo_all = 0
    if WINDOW >= 0:
        lo_all = tl.maximum(0, diag - WINDOW)
        lo_all = (lo_all // BLOCK_N) * BLOCK_N

    ntiles = (hi_all - lo_all + BLOCK_N - 1) // BLOCK_N
    per = (ntiles + SPLITS - 1) // SPLITS
    lo = lo_all + sid * per * BLOCK_N
    hi = tl.minimum(lo + per * BLOCK_N, hi_all)
    if lo >= hi:
        return

    q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqt + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        km = offs_n[:, None] < T
        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skt + offs_d[None, :],
                    mask=km, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svt + offs_d[None, :],
                    mask=km, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < T)
        if WINDOW >= 0:
            keep = keep & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW)
        acc += tl.dot(tl.where(keep, _softplus(s), 0.0).to(v.dtype), v)

    tl.atomic_add(O32 + off_b * sob + off_h * soh + offs_m[:, None] * sot + offs_d[None, :],
                  acc, mask=offs_m[:, None] < T)


def _fwd_splitk(q, k, v, window, alpha, scale, splits):
    """Split-K forward. The n^-alpha scaling has to wait until every split has landed,
    so it happens here rather than in the kernel."""
    B, T, H, D = q.shape
    cfg = _fwd_config(window)
    o32 = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)
    _fwd_splitk_kernel[(triton.cdiv(T, cfg["BLOCK_M"]), splits, B * H)](
        q, k, v, o32,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o32.stride(0), o32.stride(1), o32.stride(2),
        T, scale, H=H, WINDOW=window, SPLITS=splits, HEAD_DIM=D, **cfg,
    )
    i = torch.arange(T, device=q.device, dtype=torch.float32)
    n = i + 1 if window < 0 else torch.clamp(i + 1, max=window + 1)
    return (o32 * n.pow(-alpha)[None, :, None, None]).to(q.dtype)


_SM_COUNT = None


def _reverse_tiles(programs):
    """Whether to issue query tiles longest-first.

    Only matters when there are too few programs for the scheduler to interleave long and
    short tiles across waves. Measured on a GB10 (48 SMs), full-context forward:
    B1 T8192 H1 (128 programs) 0.57 -> 0.47 ms, while B8 T2048 H10 (2560 programs) and
    B64 T2048 H10 (20480) each lose about 3%, presumably L2 locality. Results are
    bit-identical either way, so this only ever trades time."""
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return programs < 3 * _SM_COUNT


def _fwd_config(window):
    """Launch configuration, swept on a GB10 at B8 T2048 H10 D128 bf16.

    The optimum moves with the window: a full-context tile spends most of its time in the
    bulk loop and wants the larger query tile, while a 512-wide window has only a handful
    of key tiles per query tile and does better with fewer warps. Measured forward times:
    full context 2.22 ms at 64x64/w8/s2 against 1.84 ms here; window 512 1.22 ms against
    1.08 ms. Deeper pipelining (3 stages) helps both."""
    if 0 <= window <= 1024:
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3)


def _fwd(q, k, v, window, alpha, scale):
    B, T, H, D = q.shape
    o = torch.empty_like(q)
    cfg = _fwd_config(window)
    grid = (triton.cdiv(T, cfg["BLOCK_M"]), B, H)
    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        T, scale, alpha,
        REVERSE=_reverse_tiles(grid[0] * B * H), WINDOW=window, HEAD_DIM=D, **cfg,
    )
    return o


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
    # 64x64 with 8 warps measured best for both window regimes; the larger query tiles
    # that help the forward do not compile here (shared memory holds q, k, v, do and two
    # accumulators at once).
    kw = dict(WINDOW=window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
              num_warps=8, num_stages=2)
    # dK/dV work shrinks with the key index -- key tile 0 is read by every query tile --
    # so that kernel is already longest-first. dQ grows like the forward.
    _bwd_kv_kernel[(triton.cdiv(T, BLOCK_N), B, H)](q, k, v, do, dk, dv, *args,
                                                    REVERSE=False, **kw)
    _bwd_q_kernel[(triton.cdiv(T, BLOCK_M), B, H)](
        q, k, v, do, dq, *args,
        REVERSE=_reverse_tiles(triton.cdiv(T, BLOCK_M) * B * H), **kw)
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


# =============================================================================
# Backend selection
#
# The kernels above are the Triton implementation. flash_attn_4/flash_fwd_softplus.py
# is a second one that runs FA4's own mainloop; it is faster on both layer types and is
# the default. NANOCHAT_SOFTPLUS_IMPL picks between them:
#
#   hybrid  (default) FA4's forward always, Triton's backward on windowed layers
#   fa4     FA4's mainloop for everything
#   triton  the kernels in this file
#   mixed   triton for windowed layers, fa4 for full-causal ones
#
# Measured on 1x GB10, B8 T2048 H12 D128, attention fwd+bwd, median of 5, against FA4
# softmax at 5.725 ms (windowed) / 8.593 ms (full causal):
#
#                     windowed        full causal
#   triton            5.209 (1.099x)  9.428 (0.911x)
#   fa4               5.338 (1.068x)  8.396 (1.023x)
#   hybrid            4.981 (1.149x)  8.396 (1.023x)
#
# FA4 has the faster forward everywhere and Triton the faster backward on windowed
# layers, so taking one from each beats either alone. The reason is not the score map:
# FA4 accumulates dQ atomically into a 96 MiB float32 buffer and converts it in a second
# pass, and both passes run at the machine's memory roof (0.528 ms against a 0.529 ms
# bare memset, 0.691 against 0.675 for a bare cast). That 1.22 ms is a fixed cost per
# call, so a 3.2 ms windowed backward feels it far more than a 5.5 ms full-causal one.
# Triton's backward keeps dQ in registers and pays recompute instead.
#
# End to end, d12/bs32, median of 5: hybrid 1266.6 ms against fa4's 1270.5, and the two
# distributions do not overlap (worst hybrid 1268.3 < best fa4 1268.6).
# =============================================================================

_SOFTPLUS_IMPL = os.environ.get("NANOCHAT_SOFTPLUS_IMPL", "hybrid")

# Forward tile splitting for the fa4 backend during *training*. Off by default; see
# softplus_attn_fa4_func. NANOCHAT_SOFTPLUS_SPLITS=4 turns it on for an A/B.
_SOFTPLUS_SPLITS = int(os.environ.get("NANOCHAT_SOFTPLUS_SPLITS", "1"))


def _load_fa4_softplus():
    try:
        from flash_attn_4.softplus_api import softplus_attn_fa4, softplus_attn_fa4_func
        return softplus_attn_fa4, softplus_attn_fa4_func
    except Exception:
        return None, None


_FA4_FWD, _FA4_FUNC = _load_fa4_softplus()
HAS_FA4_SOFTPLUS = _FA4_FUNC is not None


def _left_window(window_size, seqlen_k):
    """(left, right) as nanochat passes it -> the left bound FA4 wants, or None."""
    left = window_size[0] if window_size is not None else None
    if left is None or left < 0 or left >= seqlen_k:
        return None
    return int(left)


def softplus_impl_name(window_size=None):
    """Which backend a layer with this window actually runs on: 'fa4' | 'triton'."""
    if not HAS_FA4_SOFTPLUS:
        return "triton"
    if _SOFTPLUS_IMPL == "mixed":
        return "triton" if (window_size is not None and window_size[0] is not None
                            and window_size[0] >= 0) else "fa4"
    return "fa4" if _SOFTPLUS_IMPL in ("fa4", "hybrid") else "triton"


def softplus_attention(q, k, v, window_size=(-1, 0), alpha=1.0):
    """Training entry point. q, k, v are (B, T, H, D); causal only."""
    if softplus_impl_name(window_size) == "fa4":
        # hybrid: FA4 has the faster forward everywhere, Triton the faster backward on
        # windowed layers, because FA4's fp32 dQ accumulator is a fixed 1.22 ms and a
        # windowed backward is short enough to feel it.
        windowed = window_size is not None and window_size[0] is not None and window_size[0] >= 0
        bwd_impl = 1 if (_SOFTPLUS_IMPL == "hybrid" and windowed) else 0
        return _FA4_FUNC(q, k, v, causal=True,
                         window_size=(_left_window(window_size, k.size(1)), 0), alpha=alpha,
                         num_splits=_SOFTPLUS_SPLITS, bwd_impl=bwd_impl)
    return softplus_attn_func(q, k, v, causal=True, window_size=window_size, alpha=alpha)


def softplus_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                               causal=True, window_size=(-1, 0), alpha=1.0, softmax_scale=None):
    """Inference path.

    Uses FA4's softplus kernel with tile splitting when it is available. Decode is what
    splitting exists for: one query token against a long cache is a handful of programs
    on 48 SMs, and cutting the key range is the only parallelism left. `num_splits="auto"`
    sizes it from the shape -- 6.0x against a 4k cache, 2.8x against 64k, and 1 whenever
    the machine is already full.

    The fallback materialises the Tq x context score matrix in float32, which is fine for
    a single decode step and quadratic for a long prefill.
    """
    B, Tq, H, D = q.shape
    pos = int(cache_seqlens[0].item())
    if k is not None and v is not None:
        k_cache[:, pos:pos + Tq] = k
        v_cache[:, pos:pos + Tq] = v
    end = pos + Tq
    ks, vs = k_cache[:, :end], v_cache[:, :end]
    if HAS_FA4_SOFTPLUS and _SOFTPLUS_IMPL != "materialize":
        return _FA4_FWD(q, ks, vs, True, (_left_window(window_size, end), 0),
                        alpha, softmax_scale, num_splits="auto")
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
