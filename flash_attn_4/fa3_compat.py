"""FA3-shaped adapter around FA4's CuTe `flash_attn_func`.

nanochat/flash_attention.py speaks the FA3 interface:

    flash_attn_func(q, k, v, causal=True, window_size=(left, right))

with q/k/v in (B, T, H, D) layout and -1 meaning "unlimited" in window_size.
FA4 takes the same (left, right) tuple but spells "unlimited" as None, so this
module translates between the two and nothing else.

Only the training-side forward is adapted. FA4 exposes no
`flash_attn_with_kvcache`, so inference (Engine / KV cache) must keep using the
SDPA path in nanochat/flash_attention.py.
"""

import os

import torch

_LOAD_ERROR = None
try:
    from flash_attn_4.interface import (
        flash_attn_func as _fa4_flash_attn_func,
        _flash_attn_fwd,
        _flash_attn_bwd,
    )
except Exception as e:  # cutlass DSL missing, version mismatch, ...
    _fa4_flash_attn_func = None
    _flash_attn_fwd = _flash_attn_bwd = None
    _LOAD_ERROR = e

# FA4's SM120 path (this GPU) is BF16/FP16 only, and head_dim must be a multiple of 8.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def load_error():
    """The exception that prevented importing FA4, or None."""
    return _LOAD_ERROR


def is_available(device=None):
    """True if FA4 can be used for training attention on this device."""
    if _fa4_flash_attn_func is None or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    # interface.py asserts compute capability 8.x/9.x/10.x/11.x/12.x
    return major in (8, 9, 10, 11, 12)


def can_run(q, window_size=(-1, -1)):
    """Cheap per-call guard mirroring the kernels' own constraints."""
    if not is_available(q.device):
        return False
    if q.dtype not in _SUPPORTED_DTYPES:
        return False
    if q.shape[-1] % 8 != 0:
        return False
    return True


def _to_fa4_window(window_size, seqlen_k):
    """FA3 (-1 = unlimited) -> FA4 (None = unlimited).

    A window that already spans the whole sequence is dropped entirely: nanochat's
    'L' layers pass (sequence_len, 0), which is just causal attention, and letting
    FA4 know that avoids the local-attention code path.
    """
    left, right = window_size
    if left is None or left < 0 or left >= seqlen_k:
        left = None
    if right is None or right < 0:
        right = None
    return (left, right)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None):
    """FA3-compatible signature; q, k, v are (B, T, H, D), returns just the output.

    FA4's autograd Function always returns (out, lse) even with return_lse=False,
    while FA3 returns the output tensor alone. Unwrap so callers see one tensor.
    """
    if _fa4_flash_attn_func is None:
        raise RuntimeError(f"FA4 is not importable: {_LOAD_ERROR}")
    if _HAS_CUSTOM_OP and os.environ.get("FA4_CUSTOM_OP", "1") != "0":
        # opaque torch.library op: safe to call from inside a torch.compile graph
        return flash_attn_func_custom_op(q, k, v, causal=causal, window_size=window_size,
                                         softmax_scale=softmax_scale)
    out = _fa4_flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=_to_fa4_window(window_size, k.size(1)),
    )
    return out[0] if isinstance(out, tuple) else out


# =============================================================================
# torch.compile support: FA4 as an opaque custom op
# =============================================================================
# Calling FA4 directly from a compiled model makes dynamo trace into its Python
# dispatch (JIT cache lookups, tile selection, CuTe argument adaptation) and break
# the graph in dozens of places -- measured 52 graph breaks for a d12 forward+
# backward, which costs more than FA4's kernels save.
#
# torch._dynamo.allow_in_graph does not fix it: dynamo then runs the function on
# FakeTensors, and FA4 reaches a real data_ptr() (interface.py:3750).
#
# So wrap the raw fwd/bwd entry points in torch.library custom ops with explicit
# fake (meta) rules. Dynamo sees one opaque node per attention call, keeps a single
# graph, and only runs the real kernels on real tensors.

_HAS_CUSTOM_OP = False

if _flash_attn_fwd is not None:
    _LIB = "flash_attn_4"

    @torch.library.custom_op(f"{_LIB}::attn_fwd", mutates_args=(), device_types="cuda")
    def _attn_fwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        softmax_scale: float, causal: bool, window_left: int, window_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, _p, _row_max = _flash_attn_fwd(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=None if window_left < 0 else window_left,
            window_size_right=None if window_right < 0 else window_right,
            # Force the LSE: _flash_attn_fwd only materializes it when an input has
            # requires_grad, and inside a custom op autograd is already detached.
            return_lse=True,
        )
        return out, lse

    @_attn_fwd.register_fake
    def _(q, k, v, softmax_scale, causal, window_left, window_right):
        b, t, h, _ = q.shape
        out = q.new_empty((b, t, h, v.shape[-1]))
        lse = q.new_empty((b, h, t), dtype=torch.float32)
        return out, lse

    @torch.library.custom_op(f"{_LIB}::attn_bwd", mutates_args=(), device_types="cuda")
    def _attn_bwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor,
        softmax_scale: float, causal: bool, window_left: int, window_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            softmax_scale, causal, 0.0,  # softcap
            window_size_left=None if window_left < 0 else window_left,
            window_size_right=None if window_right < 0 else window_right,
            deterministic=False,
        )
        return dq, dk, dv

    @_attn_bwd.register_fake
    def _(q, k, v, out, dout, lse, softmax_scale, causal, window_left, window_right):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fwd_setup_context(ctx, inputs, output):
        q, k, v, softmax_scale, causal, window_left, window_right = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_left = window_left
        ctx.window_right = window_right

    def _fwd_backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.flash_attn_4.attn_bwd(
            q, k, v, out, dout, lse,
            ctx.softmax_scale, ctx.causal, ctx.window_left, ctx.window_right,
        )
        return dq, dk, dv, None, None, None, None

    torch.library.register_autograd(
        f"{_LIB}::attn_fwd", _fwd_backward, setup_context=_fwd_setup_context
    )
    _HAS_CUSTOM_OP = True


def has_custom_op():
    """True if the torch.library wrappers registered (i.e. compile-safe FA4)."""
    return _HAS_CUSTOM_OP


def flash_attn_func_custom_op(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None):
    """Same contract as flash_attn_func(), but through the compile-safe custom op."""
    left, right = _to_fa4_window(window_size, k.size(1))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    out, _lse = torch.ops.flash_attn_4.attn_fwd(
        q, k, v, scale, causal,
        -1 if left is None else left,
        -1 if right is None else right,
    )
    return out
