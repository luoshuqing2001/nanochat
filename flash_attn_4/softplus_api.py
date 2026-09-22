"""Public entry point for the CuTe softplus attention forward.

Separate from `flash_fwd_softplus` to avoid a cycle: `interface` imports the kernel
classes from there, so the wrapper that calls `interface` has to live elsewhere.
"""

import math
import os
from typing import Optional, Tuple, Union

import torch

# A query-tile program is one CTA. Splitting pays only when there are too few of them
# to fill the machine; past roughly this much occupancy the extra programs, the atomic
# traffic and the fp32 accumulator cost more than the shortened critical path buys.
# Measured on GB10 (48 SMs): the optimum sits at 4 splits and turns over by 16.
_SPLIT_TARGET_PROGRAMS_PER_SM = 3.0
_SPLIT_MAX = 8

# Balanced scheduling has a different optimum from uniform splitting: the cost is the
# atomics, so the aim is to fill the machine and stop, not to keep subdividing.
_BALANCED_TARGET_CTAS_PER_SM = float(os.environ.get("FA4_BALANCED_CTAS_PER_SM", "2.0"))


def auto_balanced_chunk(batch, num_head, seqlen_q, seqlen_k, causal=True, window_left=None,
                        tile_m=128, tile_n=64, device=None):
    """Key blocks per CTA for the balanced scheduler, or None to leave it off.

    Two decisions. Whether to aggregate at all: only when one CTA per query tile would
    leave the machine idle, since the atomic accumulator costs a zeroing pass and fp32
    output writes -- about 1.5 ms against a 1.7 ms kernel at training shapes, far more
    than the few percent that perfect balance could recover there. Then how big a chunk:
    enough CTAs to fill the machine a few times over, and no smaller, because every extra
    CTA is another set of atomics.
    """
    sms = torch.cuda.get_device_properties(device or 0).multi_processor_count
    num_m = max(1, (seqlen_q + tile_m - 1) // tile_m)
    if batch * num_head * num_m >= _SPLIT_TARGET_PROGRAMS_PER_SM * sms:
        return None
    from flash_attn_4.balanced_scheduler import causal_block_counts

    counts = causal_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left)
    total = batch * num_head * sum(counts)
    chunk = max(1, int(total // (_BALANCED_TARGET_CTAS_PER_SM * sms)))
    return max(1, min(chunk, max(counts) if counts else 1))


def auto_bwd_m_chunk(batch, num_head, seqlen_q, seqlen_k, causal=True, window_left=None,
                     window_right=None, tile_m=64, tile_n=128, device=None):
    """Query tiles per CTA for the balanced *backward*, or None to leave it off.

    The backward parallelizes over KV blocks, so the same triangular imbalance appears
    mirrored, and the same trade applies: balancing it means dK and dV go through
    fp32 accumulators and a conversion pass, which at training shapes costs far more
    than the imbalance is worth (B8 T2048 H12: 6.894 -> 9.030 ms at an identical CTA
    count). Only worth it when one CTA per KV block leaves the machine idle.
    """
    sms = torch.cuda.get_device_properties(device or 0).multi_processor_count
    num_n = max(1, (seqlen_k + tile_n - 1) // tile_n)
    if batch * num_head * num_n >= _SPLIT_TARGET_PROGRAMS_PER_SM * sms:
        return None
    from flash_attn_4.balanced_scheduler import causal_m_block_counts

    counts = causal_m_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal,
                                   window_left, window_right)
    total = batch * num_head * sum(counts)
    chunk = max(1, int(total // (_BALANCED_TARGET_CTAS_PER_SM * sms)))
    return max(1, min(chunk, max(counts) if counts else 1))


def auto_num_splits(batch, num_head, seqlen_q, seqlen_k, tile_m=128, device=None):
    """How many splits to use, or 1 to leave the split path off.

    Long-context inference over one sequence is the case this exists for: a single
    prefill of 8k tokens with one head group is a few dozen query tiles against 48 SMs,
    and the tail of the causal triangle leaves most of them idle. Training shapes have
    thousands of query tiles and want no splitting at all.
    """
    sms = torch.cuda.get_device_properties(device or 0).multi_processor_count
    programs = batch * num_head * max(1, (seqlen_q + tile_m - 1) // tile_m)
    target = _SPLIT_TARGET_PROGRAMS_PER_SM * sms
    if programs >= target:
        return 1
    # Round up: at 128 programs against a target of 144 the useful answer is 2, not 1.
    # The measured optimum flattens out at 4-8 splits and turns over past 16, hence the
    # clamp -- reaching the target exactly matters less than not overshooting.
    splits = math.ceil(target / programs)
    key_blocks = max(1, (seqlen_k + 63) // 64)  # never more splits than key blocks
    return max(1, min(splits, _SPLIT_MAX, key_blocks))


def softplus_attn_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    alpha: float = 1.0,
    softmax_scale: Optional[float] = None,
    num_splits: Union[int, str] = 1,
    balanced_chunk: Optional[int] = None,
    atomic_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Softplus attention forward: `O_i = n_i^-alpha * sum_j softplus(s_ij) v_j`.

    q, k, v are (B, T, H, D). Matches `nanochat.softplus_attention.softplus_attn_func`
    to bf16 noise, but runs FA4's mainloop instead of the Triton one.

    `num_splits` > 1 turns on tile splitting: each query tile's key range is cut into
    that many programs, aggregated with atomic_add into an fp32 buffer that is cast back
    at the end. It shortens the critical path under a causal mask -- the longest program
    goes from `n_block_max` tiles to `ceil(n_block_max / num_splits)` -- at the cost of
    that many more programs and the atomic traffic. It is worth it only when there are
    too few query-tile programs to fill the GPU; at training shapes it is not.

    Forward only -- no autograd. Use `softplus_attn_fa4_func` to train with it.
    """
    from flash_attn_4.interface import _flash_attn_fwd

    left, right = window_size
    if left is not None and (left < 0 or left >= k.size(1)):
        left = None
    local = left is not None
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    assert causal, "softplus attention is causal-only here"
    assert right in (0, None), "only a zero right window is supported"

    if num_splits == "auto":
        # "auto" means the balanced scheduler: a constant number of key blocks per CTA,
        # which beats uniform splitting everywhere splitting is worth doing at all.
        num_splits = 1
        if balanced_chunk is None:
            balanced_chunk = auto_balanced_chunk(
                q.shape[0], q.shape[2], q.shape[1], k.shape[1],
                causal=True, window_left=left,
            )

    out = None
    if balanced_chunk is not None:
        num_splits = 1
    if num_splits > 1 or balanced_chunk is not None:
        # The CTAs accumulate atomically, so the destination has to be zeroed. float32 is
        # the safe choice; q.dtype skips the workspace, its zeroing and the cast back, at
        # the price of accumulating in bf16 across the CTAs sharing a query tile.
        acc_dtype = atomic_dtype or torch.float32
        out = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=acc_dtype, device=q.device)

    o, *_ = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=not local,
        window_size_left=left,
        window_size_right=0 if local else None,
        attn_kind="softplus",
        softplus_alpha=float(alpha),
        softplus_num_splits=int(num_splits),
        softplus_balanced_chunk=balanced_chunk,
        out=out,
    )
    return o.to(q.dtype) if out is not None and o.dtype != q.dtype else o


# Registered as torch.library custom ops rather than a plain autograd.Function. Dynamo
# does not break the graph on either, but an autograd.Function's backward stays outside
# the inductor graph, so the elementwise work around it -- the QKV split's gradient
# accumulation above all -- can no longer fuse. That cost 171 ms of extra elementwise
# time and an eager 107 ms CUDAFunctor_add on a d12/bs32 step, turning a 15% faster
# attention into an 8% slower step. FA4's softmax path (fa3_compat.py) and the Triton
# softplus kernel both use custom ops for the same reason.


@torch.library.custom_op("softplus_attn_fa4::fwd", mutates_args=(), device_types="cuda")
def _op_fwd_fa4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                window: int, alpha: float, scale: float, splits: int,
                m_chunk: int) -> torch.Tensor:
    from flash_attn_4.interface import _flash_attn_fwd

    left = None if window < 0 else window
    # splits > 1 aggregates with atomic_add into a zeroed fp32 buffer.
    out = None
    if splits > 1:
        out = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=torch.float32, device=q.device)
    o, *_ = _flash_attn_fwd(
        q, k, v, softmax_scale=scale, causal=left is None,
        window_size_left=left, window_size_right=0 if left is not None else None,
        attn_kind="softplus", softplus_alpha=alpha, softplus_num_splits=splits, out=out,
    )
    return o.to(q.dtype) if splits > 1 else o


@_op_fwd_fa4.register_fake
def _(q, k, v, window, alpha, scale, splits, m_chunk):
    return torch.empty_like(q)


@torch.library.custom_op("softplus_attn_fa4::bwd", mutates_args=(), device_types="cuda")
def _op_bwd_fa4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                do: torch.Tensor, window: int, alpha: float, scale: float,
                m_chunk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn_4.interface import _flash_attn_bwd

    left = None if window < 0 else window
    # The softplus backward reads no LSE; the signature still wants a tensor.
    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
    dq, dk, dv, *_ = _flash_attn_bwd(
        q, k, v, out, do.contiguous(), lse,
        softmax_scale=scale, causal=left is None,
        window_size_left=left, window_size_right=0 if left is not None else None,
        attn_kind="softplus", softplus_alpha=alpha,
        softplus_balanced_m_chunk=None if m_chunk <= 0 else m_chunk,
    )
    return dq, dk, dv


@_op_bwd_fa4.register_fake
def _(q, k, v, out, do, window, alpha, scale, m_chunk):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context_fa4(ctx, inputs, output):
    q, k, v, window, alpha, scale, _splits, m_chunk = inputs
    ctx.save_for_backward(q, k, v, output)
    ctx.window, ctx.alpha, ctx.scale, ctx.m_chunk = window, alpha, scale, m_chunk


def _backward_fa4(ctx, do):
    q, k, v, out = ctx.saved_tensors
    dq, dk, dv = torch.ops.softplus_attn_fa4.bwd(
        q, k, v, out, do, ctx.window, ctx.alpha, ctx.scale, ctx.m_chunk
    )
    return dq, dk, dv, None, None, None, None, None


torch.library.register_autograd(
    "softplus_attn_fa4::fwd", _backward_fa4, setup_context=_setup_context_fa4
)


def softplus_attn_fa4_func(q, k, v, causal=True, window_size=(None, None),
                           alpha=1.0, softmax_scale=None, num_splits=1, bwd_m_chunk="auto"):
    """Differentiable softplus attention, forward and backward both in CuTe.

    `num_splits` > 1 splits the forward's key range and aggregates with atomic_add; the
    backward is unaffected, since it already parallelises over KV blocks. It is off by
    default because at training shapes there is no load to balance -- d12/bs32 launches
    3072 query-tile programs onto 48 SMs, 64 waves, and the hardware scheduler evens
    that out by itself. Measure before turning it on.
    """
    left, right = window_size
    assert causal, "softplus attention is causal-only here"
    assert right in (0, None), "only a zero right window is supported"
    window = -1 if left is None or left < 0 or left >= k.size(1) else int(left)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if bwd_m_chunk == "auto":
        bwd_m_chunk = auto_bwd_m_chunk(
            q.shape[0], q.shape[2], q.shape[1], k.shape[1],
            causal=window < 0, window_left=None if window < 0 else window,
            window_right=None if window < 0 else 0,
        ) or 0
    return torch.ops.softplus_attn_fa4.fwd(q, k, v, window, float(alpha), scale,
                                           int(num_splits), int(bwd_m_chunk))
