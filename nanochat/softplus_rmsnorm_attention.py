"""
Softplus attention with an output RMSNorm ("Softplus Attention with Normalization"):

    u_i = sum_{j in K_i} softplus(q_i . k_j * scale) v_j,    o_i = u_i / sqrt(mean(u_i^2) + eps)

per head. The RMSNorm gain is not applied here: the attention output goes straight into c_proj,
so CausalSelfAttention folds the gain into it (c_proj(gamma * o) = o @ (W diag(gamma))^T), which
costs a 768x768 elementwise product instead of passes over the (B, T, H, D) activations.

Training uses the FA4 CuTe kernels from the flash-attention repo (flash_attn.cute, installed in
editable mode; SM80/SM120 only): softplus in place of the online softmax, the RMSNorm in the
epilogue, and the RMSNorm backward fused into the backward preprocess kernel. Inference with a
KV cache uses flash_attn.cute.softplus_decode: the same kernel, with the cache split into chunks
for single-token decode when there are few (batch, head) pairs, summed before the RMSNorm.
"""
import torch

RMS_EPS = 1e-6


def _window_left(window_size):
    left = window_size[0]
    return -1 if left is None or left < 0 else int(left)


# torch.library wrappers, as in softplus_attention.py: called directly from a compiled model,
# dynamo would trace into the CuTe launcher and break the graph.
@torch.library.custom_op("softplus_rmsnorm_attn::fwd", mutates_args=(), device_types="cuda")
def _op_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            window_left: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd
    out, rms, _, _ = _flash_attn_fwd(
        q, k, v, causal=True,
        window_size_left=None if window_left < 0 else window_left,
        window_size_right=None if window_left < 0 else 0,
        return_lse=True, attn_fn="softplus", rms_eps=eps,
    )
    return out, rms


@_op_fwd.register_fake
def _(q, k, v, window_left, eps):
    B, T, H, _ = q.shape
    out = q.new_empty(B, T, H, v.shape[-1])
    rms = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    return out, rms


@torch.library.custom_op("softplus_rmsnorm_attn::bwd", mutates_args=(), device_types="cuda")
def _op_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
            dout: torch.Tensor, rms: torch.Tensor,
            window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_bwd
    dq, dk, dv = _flash_attn_bwd(
        q, k, v, out, dout.contiguous(), rms, None, True,
        window_size_left=None if window_left < 0 else window_left,
        window_size_right=None if window_left < 0 else 0,
        attn_fn="softplus",
    )
    return dq, dk, dv


@_op_bwd.register_fake
def _(q, k, v, out, dout, rms, window_left):
    return tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v, window_left, eps = inputs
    out, rms = output
    ctx.save_for_backward(q, k, v, out, rms)
    ctx.window_left = window_left


def _backward(ctx, dout, _drms):
    q, k, v, out, rms = ctx.saved_tensors
    dq, dk, dv = torch.ops.softplus_rmsnorm_attn.bwd(q, k, v, out, dout, rms, ctx.window_left)
    return dq, dk, dv, None, None


torch.library.register_autograd("softplus_rmsnorm_attn::fwd", _backward, setup_context=_setup_context)


def softplus_rmsnorm_attn_func(q, k, v, window_size=(-1, 0), eps=RMS_EPS):
    """Causal training path. q, k, v are (B, T, H, D) with nanochat's (left, 0) window."""
    assert window_size[1] in (0, None), "softplus_rmsnorm attention is causal-only here"
    out, _ = torch.ops.softplus_rmsnorm_attn.fwd(q, k, v, _window_left(window_size), float(eps))
    return out


def softplus_rmsnorm_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                                       window_size=(-1, 0), eps=RMS_EPS):
    """Inference path: writes k, v into the cache at cache_seqlens (per batch entry), then attends
    the first cache_seqlens + Tq keys causally. The caller advances cache_seqlens."""
    from flash_attn.cute.softplus_decode import softplus_attn_with_kvcache
    left = _window_left(window_size)
    # varlen split: deterministic, so greedy samples don't change between runs (the atomic split is
    # faster for single-sequence decode at >= 2K keys but its low bits vary run to run)
    return softplus_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                                      window_size=(None if left < 0 else left, 0), eps=eps,
                                      split_impl="varlen")
