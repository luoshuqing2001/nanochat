"""Public entry point for the CuTe softplus attention forward.

Separate from `flash_fwd_softplus` to avoid a cycle: `interface` imports the kernel
classes from there, so the wrapper that calls `interface` has to live elsewhere.
"""

import math
from typing import Optional, Tuple, Union

import torch

# A query-tile program is one CTA. Splitting pays only when there are too few of them
# to fill the machine; past roughly this much occupancy the extra programs, the atomic
# traffic and the fp32 accumulator cost more than the shortened critical path buys.
# Measured on GB10 (48 SMs): the optimum sits at 4 splits and turns over by 16.
_SPLIT_TARGET_PROGRAMS_PER_SM = 3.0
_SPLIT_MAX = 8


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
        num_splits = auto_num_splits(q.shape[0], q.shape[2], q.shape[1], k.shape[1])

    out = None
    if num_splits > 1:
        # The splits accumulate atomically, so the destination has to be fp32 and zeroed.
        out = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=torch.float32, device=q.device)

    o, *_ = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=not local,
        window_size_left=left,
        window_size_right=0 if local else None,
        attn_kind="softplus",
        softplus_alpha=float(alpha),
        softplus_num_splits=int(num_splits),
        out=out,
    )
    return o.to(q.dtype) if num_splits > 1 else o


class _SoftplusAttnFA4(torch.autograd.Function):
    """Autograd wrapper: FA4's mainloop for both directions.

    The backward carries no LSE and runs no preprocess pass, because softplus needs
    neither -- see flash_bwd_softplus.py. Splitting is forward-only, so training uses
    num_splits=1 and gets nothing from this class it would not get from the plain call.
    """

    @staticmethod
    def forward(ctx, q, k, v, left, alpha, scale):
        from flash_attn_4.interface import _flash_attn_fwd

        out, *_ = _flash_attn_fwd(
            q, k, v, softmax_scale=scale, causal=left is None,
            window_size_left=left, window_size_right=0 if left is not None else None,
            attn_kind="softplus", softplus_alpha=alpha,
        )
        ctx.save_for_backward(q, k, v, out)
        ctx.left, ctx.alpha, ctx.scale = left, alpha, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        from flash_attn_4.interface import _flash_attn_bwd

        q, k, v, out = ctx.saved_tensors
        left = ctx.left
        # `lse` is unused by the softplus backward but the signature wants a tensor.
        lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
        dq, dk, dv, *_ = _flash_attn_bwd(
            q, k, v, out, dout.contiguous(), lse,
            softmax_scale=ctx.scale, causal=left is None,
            window_size_left=left, window_size_right=0 if left is not None else None,
            attn_kind="softplus", softplus_alpha=ctx.alpha,
        )
        return dq, dk, dv, None, None, None


def softplus_attn_fa4_func(q, k, v, causal=True, window_size=(None, None),
                           alpha=1.0, softmax_scale=None):
    """Differentiable softplus attention, forward and backward both in CuTe."""
    left, right = window_size
    if left is not None and (left < 0 or left >= k.size(1)):
        left = None
    assert causal, "softplus attention is causal-only here"
    assert right in (0, None), "only a zero right window is supported"
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return _SoftplusAttnFA4.apply(q, k, v, left, float(alpha), scale)
