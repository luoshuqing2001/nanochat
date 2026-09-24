"""Softplus attention dispatch with fixed-KV CuTe and Triton implementations.

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


def _forward_tile_n(q):
    return 128 if torch.cuda.get_device_capability(q.device)[0] >= 12 and q.shape[-1] <= 64 else 64


def _forward_tile_shape(q):
    # On SM120, 64 query rows reduce register/shared-memory pressure and expose
    # more independent work without introducing atomic output aggregation.
    return (64 if torch.cuda.get_device_capability(q.device)[0] == 12 else 128,
            _forward_tile_n(q))


def kv_chunk_blocks(q, tokens):
    """Convert a token interval to the SM80/SM120 forward's actual KV tile count."""
    tile_n = _forward_tile_n(q)
    if tokens <= 0 or tokens % tile_n:
        raise ValueError(f"KV chunk must be a positive multiple of {tile_n} tokens")
    return tokens // tile_n


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
    # A short local loop does not recover the extra initialization/conversion
    # launches. Local attention also has little triangular imbalance after its
    # first window. Keep such tasks whole instead of splitting on occupancy alone.
    visible = min(seqlen_k, window_left + 1) if window_left is not None else seqlen_k
    if visible <= 8 * tile_n:
        return None
    if seqlen_k - seqlen_q >= seqlen_q and batch * num_head * num_m >= sms:
        return None
    if batch * num_head * num_m >= _SPLIT_TARGET_PROGRAMS_PER_SM * sms:
        return None
    from flash_attn_4.balanced_scheduler import causal_block_counts

    counts = causal_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left)
    total = batch * num_head * sum(counts)
    chunk = max(1, int(total // (_BALANCED_TARGET_CTAS_PER_SM * sms)))
    return max(1, min(chunk, max(counts) if counts else 1))


def auto_bwd_m_chunk(batch, num_head, seqlen_q, seqlen_k, causal=True, window_left=None,
                     window_right=None, tile_m=64, tile_n=None, device=None):
    """Query tiles per CTA for the balanced *backward*, or None to leave it off.

    The backward parallelizes over KV blocks, so the same triangular imbalance appears
    mirrored, and the same trade applies: balancing it means dK and dV go through
    fp32 accumulators and a conversion pass, which at training shapes costs far more
    than the imbalance is worth (B8 T2048 H12: 6.894 -> 9.030 ms at an identical CTA
    count). Only worth it when one CTA per KV block leaves the machine idle.
    """
    props = torch.cuda.get_device_properties(device or 0)
    sms = props.multi_processor_count
    # _flash_attn_bwd uses 64x64 on SM120, not its signature's 64x128 default.
    # Counting 128-wide tiles halved the estimated parallelism and unnecessarily
    # enabled dK/dV atomics for B1/H6/T2048, which already has 192 KV tasks.
    if tile_n is None:
        tile_n = 64 if props.major == 12 else 128
    if min(seqlen_q, window_left + 1 if window_left is not None else seqlen_q) <= 8 * tile_m:
        return None
    num_n = max(1, (seqlen_k + tile_n - 1) // tile_n)
    if batch * num_head * num_n >= _SPLIT_TARGET_PROGRAMS_PER_SM * sms:
        return None
    from flash_attn_4.balanced_scheduler import causal_m_block_counts

    counts = causal_m_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal,
                                   window_left, window_right)
    total = batch * num_head * sum(counts)
    # Four waves amortize atomic updates while reducing the long causal tail;
    # this continuous rule replaces the 4K/8K/16K single-head lookup table.
    chunk = max(1, math.ceil(total / (4 * sms)))
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


def _validate_schedule_options(num_splits, balanced_chunk, stream_waves, fragment_n, q_in_regs):
    if not isinstance(stream_waves, int) or stream_waves < 0:
        raise ValueError("stream_waves must be a nonnegative integer")
    if fragment_n not in (0, 32, 64):
        raise ValueError("fragment_n must be 0, 32, or 64")
    if stream_waves and (fragment_n or q_in_regs):
        raise ValueError("Stream-K and CuTe fragment options select different kernels")
    if (stream_waves or fragment_n or q_in_regs) and (balanced_chunk is not None or num_splits not in (1, "auto")):
        raise ValueError("experimental schedules require unsplit inputs and no balanced_chunk")


def _native_forward_options(q, num_splits, balanced_chunk, stream_waves,
                            cute_stream_waves, stream_tail, warp_overlap, poly_estrin, stream_atomic=False, stream_tiles=False, stream_global=False, stream_complete=False, whole_lpt=False, kv_split_size=0, kv_major=False, owner_pair=False, head_lpt=False):
    if head_lpt and (owner_pair or whole_lpt or kv_split_size or cute_stream_waves or stream_waves or warp_overlap or poly_estrin):
        raise ValueError("head_lpt is a separate unsplit schedule")
    if owner_pair and (whole_lpt or kv_split_size or cute_stream_waves or stream_waves or warp_overlap or poly_estrin):
        raise ValueError("owner_pair is a separate unsplit schedule")
    if kv_major and not kv_split_size:
        raise ValueError("kv_major requires a positive kv_split_size")
    if not isinstance(kv_split_size, int) or kv_split_size < 0 or kv_split_size % _forward_tile_n(q):
        raise ValueError("kv_split_size must be zero or a positive multiple of the KV tile size")
    if kv_split_size and stream_tail:
        raise ValueError("kv_split_size is incompatible with stream_tail")
    if not isinstance(cute_stream_waves, int) or cute_stream_waves < 0:
        raise ValueError("cute_stream_waves must be a nonnegative integer")
    if stream_global and (not stream_tiles or not (cute_stream_waves or kv_split_size)):
        raise ValueError("stream_global requires stream_tiles and cute_stream_waves")
    if stream_complete and (not (cute_stream_waves or kv_split_size) or stream_atomic):
        raise ValueError("stream_complete requires private CuTe stream partials")
    if whole_lpt and (kv_split_size or cute_stream_waves or stream_waves or warp_overlap or poly_estrin):
        raise ValueError("whole_lpt is a separate unsplit schedule")
    if (whole_lpt or owner_pair or head_lpt) and torch.cuda.get_device_capability(q.device)[0]!=12:
        raise ValueError("whole_lpt requires SM120")
    selected = bool(kv_split_size or cute_stream_waves or warp_overlap or poly_estrin or whole_lpt or owner_pair or head_lpt)
    if stream_tiles and (not (cute_stream_waves or kv_split_size) or stream_tail):
        raise ValueError("stream_tiles requires cute_stream_waves and no stream_tail")
    if stream_atomic and not (cute_stream_waves or kv_split_size):
        raise ValueError("stream_atomic requires cute_stream_waves")
    if stream_tail and not (cute_stream_waves or kv_split_size):
        raise ValueError("stream_tail requires cute_stream_waves")
    if selected and (stream_waves or balanced_chunk is not None or num_splits not in (1, "auto")):
        raise ValueError("native forward options require unsplit CuTe inputs")
    if selected and torch.cuda.get_device_capability(q.device)[0] not in (8, 12):
        raise ValueError("native forward options require SM80 or SM120")
    workers = (max(1, math.ceil(cute_stream_waves * torch.cuda.get_device_properties(q.device).multi_processor_count
                               / (1 if stream_global else q.shape[0] * q.shape[2]))) if cute_stream_waves else (1 if kv_split_size else 0))
    return selected, workers


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
    stream_waves: int = 0,
    fragment_n: int = 0,
    q_in_regs: bool = False,
    cute_stream_waves: int = 0,
    stream_tail: bool = False,
    warp_overlap: bool = False,
    poly_estrin: bool = False,
    stream_atomic: bool = False,
    stream_tiles: bool = False,
    stream_global: bool = False,
    stream_complete: bool = False,
    whole_lpt: bool = False,
    kv_split_size: int = 0,
    kv_major: bool = False,
    owner_pair: bool = False,
    head_lpt: bool = False,
) -> torch.Tensor:
    """Softplus attention forward: `O_i = n_i^-alpha * sum_j softplus(s_ij) v_j`.

    q, k, v are (B, T, H, D). Matches `nanochat.softplus_attention.softplus_attn_func`
    to bf16 noise. Auto uses tuned fixed-KV kernels where measurements favor them.

    `num_splits` > 1 turns on tile splitting: each query tile's key range is cut into
    that many programs, aggregated with atomic_add into an fp32 buffer that is cast back
    at the end. It shortens the critical path under a causal mask -- the longest program
    goes from `n_block_max` tiles to `ceil(n_block_max / num_splits)` -- at the cost of
    that many more programs and the atomic traffic. It is worth it only when there are
    too few query-tile programs to fill the GPU; at training shapes it is not.

    `num_splits="auto"` uses independent fixed-size KV tasks: a vector-reduction
    kernel for single-query D64/D128 inference, a tiled Triton kernel for measured
    GB10 prefill shapes, and the CuTe balanced scheduler otherwise. Explicit splits
    or balanced_chunk select the CuTe implementation.

    `cute_stream_waves=2, stream_atomic=True` uses FP32 atomic accumulation only
    for split query tiles. `stream_tiles=True` removes the persistent loop and
    splits only tiles exceeding the work budget. Neither changes default dispatch.
    Experimental schedules are opt-in: `stream_waves=2` selects persistent Stream-K;
    `fragment_n=32/64` selects CuTe score fragments; `q_in_regs=True` retains Q.
    None is automatically selected solely from occupancy.

    `kv_split_size=512, stream_atomic=True` caps each CTA at 512 KV tokens,
    accumulates locally, and atomically merges only split query tiles.
    `kv_major=True` visits query tiles sharing a KV segment consecutively.

    `owner_pair=True` sequentially pairs long/short complete query tiles per CTA.
    It is an explicit SM120 full-causal experiment; no automatic dispatch changes.

    Forward only -- no autograd. Use `softplus_attn_fa4_func` to train with it.
    """
    left, right = window_size
    if left is not None and (left < 0 or left >= k.size(1)):
        left = None
    local = left is not None
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    assert causal, "softplus attention is causal-only here"
    assert right in (0, None), "only a zero right window is supported"

    _validate_schedule_options(num_splits, balanced_chunk, stream_waves, fragment_n, q_in_regs)
    stream_tiles = bool(stream_tiles or kv_split_size)
    native_fwd, native_workers = _native_forward_options(
        q, num_splits, balanced_chunk, stream_waves, cute_stream_waves,
        stream_tail, warp_overlap, poly_estrin, stream_atomic, stream_tiles, stream_global, stream_complete, whole_lpt, kv_split_size, kv_major, owner_pair, head_lpt)
    if stream_waves:
        if atomic_dtype is not None:
            raise ValueError("Stream-K uses private FP32 partials; atomic_dtype is incompatible")
        from nanochat.softplus_stream_k import stream_k_forward
        return stream_k_forward(q, k, v, -1 if left is None else left, alpha, scale,
                                waves=stream_waves)
    if fragment_n or q_in_regs or native_fwd:
        num_splits = 1  # Explicit CuTe experiment bypasses automatic Triton dispatch.

    # Single-query inference has no query reuse for the 128-row CuTe MMA tile.
    # Fixed KV tasks reduce vectors directly and need neither padded query rows nor
    # a CPU-built work table. Explicit split/atomic options retain the CuTe path.
    if (num_splits == "auto" and balanced_chunk is None and atomic_dtype is None
            and q.shape[0] > 0 and q.shape[1] == 1 and q.shape[2] > 0 and k.shape[1] > 0
            and q.shape[2] == k.shape[2] == v.shape[2]
            and q.shape[-1] in (64, 128) and v.shape[-1] in (64, 128)
            and q.dtype in (torch.float16, torch.bfloat16)):
        from nanochat.softplus_decode import softplus_decode
        return softplus_decode(q, k, v, left, alpha, scale)

    if num_splits == "auto" and balanced_chunk is None and atomic_dtype is None:
        from nanochat.softplus_fixed_kv import auto_fixed_kv_chunk, fixed_kv_forward
        chunk = auto_fixed_kv_chunk(q, k, v, -1 if left is None else left)
        if chunk is not None:
            return fixed_kv_forward(q, k, v, -1, alpha, scale, chunk=chunk, bm=128)

    from flash_attn_4.interface import _flash_attn_fwd

    if num_splits == "auto":
        # Prefill uses fixed work per program, rather than a fixed split count per row.
        num_splits = 1
        if balanced_chunk is None:
            balanced_chunk = auto_balanced_chunk(
                q.shape[0], q.shape[2], q.shape[1], k.shape[1],
                causal=True, window_left=left, device=q.device,
                tile_m=_forward_tile_shape(q)[0], tile_n=_forward_tile_n(q),
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
        softplus_fragment_n=fragment_n, softplus_q_in_regs=q_in_regs,
        softplus_stream_workers=native_workers, softplus_stream_tail=stream_tail,
        softplus_stream_atomic=stream_atomic, softplus_stream_tiles=stream_tiles,
        softplus_stream_global=stream_global, softplus_stream_complete=stream_complete,
        sm120_global_lpt=whole_lpt, sm120_owner_pair=owner_pair, sm120_head_lpt=head_lpt, softplus_kv_split_size=kv_split_size, softplus_kv_major=kv_major,
        softplus_warp_overlap=warp_overlap, softplus_poly_estrin=poly_estrin,
        tile_mn=_forward_tile_shape(q),
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
                m_chunk: int, bwd_impl: int, fwd_chunk: int = 0,
                triton_chunk: int = 0, stream_waves: int = 0,
                fragment_n: int = 0, q_in_regs: bool = False,
                early_dv: bool = False, native_bwd: int = 0, native_workers: int = 0,
                stream_tail: bool = False, warp_overlap: bool = False,
                poly_estrin: bool = False, stream_atomic: bool = False, stream_tiles: bool = False, stream_global: bool = False, stream_complete: bool = False, whole_lpt: bool = False, kv_split_size: int = 0, kv_major: bool = False, owner_pair: bool = False, head_lpt: bool = False) -> torch.Tensor:
    if stream_waves > 0:
        from nanochat.softplus_stream_k import stream_k_forward
        return stream_k_forward(q, k, v, window, alpha, scale, waves=stream_waves)
    if triton_chunk > 0:
        from nanochat.softplus_fixed_kv import fixed_kv_forward
        return fixed_kv_forward(q, k, v, window, alpha, scale, chunk=triton_chunk, bm=128)
    from flash_attn_4.interface import _flash_attn_fwd

    left = None if window < 0 else window
    # splits > 1 aggregates with atomic_add into a zeroed fp32 buffer.
    out = None
    if splits > 1 or fwd_chunk > 0:
        out = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=torch.float32, device=q.device)
    o, *_ = _flash_attn_fwd(
        q, k, v, softmax_scale=scale, causal=left is None,
        window_size_left=left, window_size_right=0 if left is not None else None,
        attn_kind="softplus", softplus_alpha=alpha, softplus_num_splits=splits, out=out,
        softplus_balanced_chunk=fwd_chunk if fwd_chunk > 0 else None,
        softplus_fragment_n=fragment_n, softplus_q_in_regs=q_in_regs,
        softplus_stream_workers=native_workers, softplus_stream_tail=stream_tail,
        softplus_stream_atomic=stream_atomic, softplus_stream_tiles=stream_tiles,
        softplus_stream_global=stream_global, softplus_stream_complete=stream_complete,
        sm120_global_lpt=whole_lpt, sm120_owner_pair=owner_pair, sm120_head_lpt=head_lpt, softplus_kv_split_size=kv_split_size, softplus_kv_major=kv_major,
        softplus_warp_overlap=warp_overlap, softplus_poly_estrin=poly_estrin,
        tile_mn=_forward_tile_shape(q),
    )
    return o.to(q.dtype) if out is not None else o


@_op_fwd_fa4.register_fake
def _(q, k, v, window, alpha, scale, splits, m_chunk, bwd_impl, fwd_chunk=0, triton_chunk=0,
      stream_waves=0, fragment_n=0, q_in_regs=False, early_dv=False, native_bwd=0,
      native_workers=0, stream_tail=False, warp_overlap=False, poly_estrin=False, stream_atomic=False, stream_tiles=False, stream_global=False, stream_complete=False, whole_lpt=False, kv_split_size=0, kv_major=False, owner_pair=False, head_lpt=False):
    return torch.empty_like(q)


@torch.library.custom_op("softplus_attn_fa4::bwd", mutates_args=(), device_types="cuda")
def _op_bwd_fa4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                do: torch.Tensor, window: int, alpha: float, scale: float,
                m_chunk: int, early_dv: bool = False, native_bwd: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        softplus_early_dv=early_dv or native_bwd in (7,8),
        softplus_share_pd=native_bwd in (1,2,3,4,6),
        sm120_owner_pair=native_bwd in (5,6), softplus_native_bwd=native_bwd == 2,
        softplus_inline_scale=native_bwd == 3,
        sm120_bwd_tile=(64,128,1,1) if native_bwd == 4 else ((64,64,1,1) if native_bwd in (7,8) else None),
        sm120_bwd_num_threads=256 if native_bwd in (7,8) else 128,
        sm120_bwd_warp_layout=(4,2,4) if native_bwd == 8 else None,
    )
    return dq, dk, dv


@_op_bwd_fa4.register_fake
def _(q, k, v, out, do, window, alpha, scale, m_chunk, early_dv=False, native_bwd=0):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context_fa4(ctx, inputs, output):
    q, k, v, window, alpha, scale, _splits, m_chunk, bwd_impl, _fwd_chunk, _triton_chunk, _stream_waves, _fragment_n, _q_in_regs, early_dv, native_bwd, _workers, _tail, _overlap, _estrin, _atomic, _tiles, _global, _complete, _whole, _kv_split_size, _kv_major, _owner_pair, _head_lpt = inputs
    ctx.save_for_backward(q, k, v, output)
    ctx.window, ctx.alpha, ctx.scale, ctx.m_chunk = window, alpha, scale, m_chunk
    ctx.bwd_impl = bwd_impl
    ctx.early_dv = early_dv
    ctx.native_bwd = native_bwd


def _backward_fa4(ctx, do):
    q, k, v, out = ctx.saved_tensors
    if ctx.bwd_impl == 1:
        # Triton backward under FA4's forward. FA4 accumulates dQ atomically into a 96 MiB
        # fp32 buffer and converts it in a second pass -- 1.22 ms of pure bandwidth per
        # call, both passes already at the machine's roof, so it is a fixed cost that a
        # short windowed backward feels more than a full-causal one. The Triton backward
        # keeps dQ in registers and pays recompute instead, which is the better trade for
        # a window: 3.923 vs 4.280 ms at B8 T2048 H12 W=512, and the other way round
        # without a window.
        import nanochat.softplus_attention  # registers softplus_attn::bwd

        dq, dk, dv = torch.ops.softplus_attn.bwd(
            q, k, v, do, ctx.window, ctx.alpha, ctx.scale
        )
    else:
        dq, dk, dv = torch.ops.softplus_attn_fa4.bwd(
            q, k, v, out, do, ctx.window, ctx.alpha, ctx.scale, ctx.m_chunk, ctx.early_dv, ctx.native_bwd
        )
    return (dq, dk, dv) + (None,) * 26


torch.library.register_autograd(
    "softplus_attn_fa4::fwd", _backward_fa4, setup_context=_setup_context_fa4
)


def softplus_attn_fa4_func(q, k, v, causal=True, window_size=(None, None),
                           alpha=1.0, softmax_scale=None, num_splits="auto", bwd_m_chunk="auto",
                           bwd_impl=0, balanced_chunk=None, stream_waves=0,
                           fragment_n=0, q_in_regs=False, early_dv=None,
                           bwd_schedule="auto", cute_stream_waves=0, stream_tail=False,
                           warp_overlap=False, poly_estrin=False, stream_atomic=False, stream_tiles=False, stream_global=False, stream_complete=False, whole_lpt=False, kv_split_size=0, kv_major=False, owner_pair=False, head_lpt=False):
    """Differentiable softplus attention with automatic fixed-KV forward dispatch.

    `stream_waves`, `fragment_n`, and `q_in_regs` explicitly select experimental
    forward schedules. `early_dv=True` consumes P before dP in the CuTe backward.
    With early_dv=None, SM120 BF16 D128 full-causal square sequences of at least
    2048 tokens use early-dV. False explicitly retains the previous backward.
    `bwd_schedule="auto"` also shares P/dS storage for SM120 BF16 D64 square
    sequences >=2048. "previous", "shared", "inline", and "scaled" override it.
    "scaled" rounds scaled dO to the input dtype and can underflow in FP16.
    `cute_stream_waves=2` selects same-mainloop CuTe Stream-K; `stream_tail=True`
    splits only the final query tiles. `poly_estrin` and `warp_overlap` are opt-in.
    `stream_atomic=True` uses one zeroed FP32 accumulator per split query tile;
    unsplit tiles still write directly. Requires cute_stream_waves > 0 or kv_split_size.
    `kv_split_size=256/512/1024` sets a hard token cap and selects one CTA per segment.
    `kv_major=True` reorders segments for KV reuse across query tiles.
    `stream_tiles=True` instead assigns one CTA per cost-bounded segment,
    avoiding the persistent loop. It cannot be combined with stream_tail.
    `head_lpt=True` orders complete query tiles longest-first within each head.
    `owner_pair=True` pairs complete query tiles without an output workspace.
    `bwd_schedule="paired"/"paired_shared"` separately pairs complete KV owners.
    `bwd_schedule="warp8"` uses eight warps and one Q/dO stage on SM120,
    reducing D128 accumulator pressure without splitting KV owners. Auto selects
    it for BF16 D128 square full-causal sequences >=2048 with unsplit backward
    owners and early-dV enabled. Explicit early_dv=False retains the prior path.
    `bwd_schedule="warp8_dkv"` is an opt-in D128 variant distributing dK/dV
    warps as 2x4 instead of 4x2. Measured gains are small and workload-dependent;
    automatic selection retains the original warp8 layout.
    These options are preserved through autograd and torch.compile.

    `num_splits="auto"` enables fixed-work KV tasks at measured GB10 training shapes.
    `balanced_chunk` explicitly sets KV blocks per task for other experiments. Other
    training shapes retain the original unsplit forward: occupancy alone does not
    predict whether extra accumulator traffic pays. Integer `num_splits` retains the
    older fixed-split-count experiment.
    """
    left, right = window_size
    assert causal, "softplus attention is causal-only here"
    assert right in (0, None), "only a zero right window is supported"
    window = -1 if left is None or left < 0 or left >= k.size(1) else int(left)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    _validate_schedule_options(num_splits, balanced_chunk, stream_waves, fragment_n, q_in_regs)
    stream_tiles = bool(stream_tiles or kv_split_size)
    native_fwd, native_workers = _native_forward_options(
        q, num_splits, balanced_chunk, stream_waves, cute_stream_waves,
        stream_tail, warp_overlap, poly_estrin, stream_atomic, stream_tiles, stream_global, stream_complete, whole_lpt, kv_split_size, kv_major, owner_pair, head_lpt)
    if early_dv is None:
        # Small but repeated gains on global D128 BF16; FP16 and local/D64 did
        # not generalize. Keep this a regime rule, not a per-shape timing table.
        early_dv = (bwd_impl == 0 and window < 0 and q.dtype == torch.bfloat16
                    and q.shape[-1] == v.shape[-1] == 128
                    and q.shape[2] == k.shape[2] == v.shape[2]
                    and q.shape[1] == k.shape[1] and q.shape[1] >= 2048
                    and torch.cuda.get_device_capability(q.device)[0] == 12)
    if early_dv and bwd_impl != 0:
        raise ValueError("early_dv requires the CuTe backward")
    if stream_waves or fragment_n or q_in_regs or native_fwd:
        num_splits = 1
    triton_chunk = 0
    if num_splits == "auto":
        num_splits = 1
        if balanced_chunk is None:
            from nanochat.softplus_fixed_kv import auto_fixed_kv_chunk
            triton_chunk = auto_fixed_kv_chunk(q, k, v, window) or 0
    if balanced_chunk is not None:
        if balanced_chunk <= 0:
            raise ValueError("balanced_chunk must be positive")
        num_splits = 1
    if bwd_schedule in ("paired", "paired_shared"):
        if window>=0 or torch.cuda.get_device_capability(q.device)[0]!=12:
            raise ValueError("paired backward requires SM120 full causal attention")
        if bwd_m_chunk=="auto":bwd_m_chunk=0
        if bwd_m_chunk!=0:raise ValueError("paired backward uses unsplit KV owners")
    if bwd_schedule == "kv_group2":
        if torch.cuda.get_device_capability(q.device)[0]!=12 or (q.shape[-1] != 64 or v.shape[-1] != 64):
            raise ValueError("kv_group2 requires SM120 D64")
        if bwd_m_chunk=="auto":bwd_m_chunk=0
        if bwd_m_chunk!=0:raise ValueError("kv_group2 uses unsplit KV owners")
    if bwd_m_chunk == "auto":
        bwd_m_chunk = auto_bwd_m_chunk(
            q.shape[0], q.shape[2], q.shape[1], k.shape[1],
            causal=window < 0, window_left=None if window < 0 else window,
            window_right=None if window < 0 else 0,
            device=q.device,
        ) or 0
    modes = {"previous": 0, "shared": 1, "scaled": 2, "inline": 3, "kv_group2": 4, "paired": 5, "paired_shared": 6, "warp8": 7, "warp8_dkv": 8}
    if bwd_schedule == "auto":
        # Conservative measured regime; preserve FP16 and hybrid backward behavior.
        native_bwd = int(bwd_impl == 0 and q.dtype == torch.bfloat16
                         and q.shape[-1] == v.shape[-1] == 64
                         and q.shape[2] == k.shape[2] == v.shape[2]
                         and q.shape[1] == k.shape[1] and q.shape[1] >= 2048
                         and torch.cuda.get_device_capability(q.device)[0] == 12)
        if (bwd_impl == 0 and early_dv and bwd_m_chunk == 0 and window < 0
                and q.dtype == torch.bfloat16 and q.shape[-1] == v.shape[-1] == 128
                and q.shape[2] == k.shape[2] == v.shape[2]
                and q.shape[1] == k.shape[1] and q.shape[1] >= 2048
                and torch.cuda.get_device_capability(q.device)[0] == 12):
            native_bwd = 7
    elif bwd_schedule in modes:
        native_bwd = modes[bwd_schedule]
    else:
        raise ValueError("bwd_schedule must be auto or one of " + ", ".join(modes))
    if native_bwd in (7,8) and torch.cuda.get_device_capability(q.device)[0] != 12:
        raise ValueError("warp8 backward requires SM120")
    if native_bwd == 8 and not (q.shape[-1] == k.shape[-1] == v.shape[-1] == 128):
        raise ValueError("warp8_dkv backward requires D128")
    if native_bwd and bwd_impl != 0:
        raise ValueError("native backward schedules require the CuTe backward")
    return torch.ops.softplus_attn_fa4.fwd(q, k, v, window, float(alpha), scale,
                                           int(num_splits), int(bwd_m_chunk), int(bwd_impl),
                                           int(balanced_chunk or 0), triton_chunk, int(stream_waves),
                                           int(fragment_n), bool(q_in_regs), bool(early_dv), native_bwd,
                                           native_workers, bool(stream_tail), bool(warp_overlap), bool(poly_estrin), bool(stream_atomic), bool(stream_tiles), bool(stream_global), bool(stream_complete), bool(whole_lpt), int(kv_split_size), bool(kv_major), bool(owner_pair), bool(head_lpt))


def softplus_owner_schedule_options(q, k, window_size=(None, None)):
    """Opt-in experimental forward policy; returns kwargs for either public API.

    Calibrated candidates on GB10 BF16, square full-causal D64/128 only. Other
    regimes return no overrides. This never changes automatic/default dispatch
    or the caller's backward schedule. Call with the same window_size as attention.
    """
    if (not q.is_cuda or q.ndim != 4 or k.ndim != 4 or q.dtype != torch.bfloat16
            or q.shape != k.shape or q.shape[-1] not in (64,128) or q.shape[1]<2048):
        return {}
    left,right=window_size
    if right not in (None,0) or (left is not None and 0<=left<k.shape[1]):
        return {}
    if torch.cuda.get_device_capability(q.device)[0]!=12:
        return {}
    sms=torch.cuda.get_device_properties(q.device).multi_processor_count
    m=(q.shape[1]+63)//64
    bh=q.shape[0]*q.shape[2]
    if bh*m<2*sms:
        return dict(cute_stream_waves=2,stream_tiles=True,stream_atomic=True)
    if m<2*sms and bh*((m+1)//2)>=2*sms:
        return dict(owner_pair=True)
    return dict(head_lpt=True)
