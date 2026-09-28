"""Python side of the hopper_softplus builds: loading, calling, and a float64 reference."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import load_variant  # noqa: E402

RMS_EPS = 1e-6
_ops = {}


def ops(name):
    if name not in _ops:
        _ops[name] = load_variant(name)
    return _ops[name]


LOG2E = 1.4426950408889634


def prescale_q(q):
    """For the pre_* builds: fold softmax_scale * log2e into Q (nanochat would do it in the q norm)."""
    return (q.float() * (LOG2E / math.sqrt(q.shape[-1]))).to(q.dtype)


def fwd(name, q, k, v, causal=True, window_left=-1, num_splits=1, softmax_scale=None):
    """q, k, v: (B, T, H, D) bf16. Returns (out, lse_or_rms).

    pre_* builds expect q = prescale_q(q_raw); the scale they are given is ignored.

    For the stock build the second output is the softmax LSE; for softplus builds it is the
    per-row rms of U (what a backward pass would need), shape (B, H, T), fp32.
    """
    wl = window_left if window_left is not None else -1
    wr = 0 if (causal or wl >= 0) else -1
    out, lse, *_ = ops(name).fwd(
        q, k, v, softmax_scale=softmax_scale, is_causal=causal,
        window_size_left=wl, window_size_right=wr, num_splits=num_splits)
    return out, lse


def bwd(name, dout, q, k, v, out, lse, causal=True, window_left=-1, softmax_scale=None, gain=None):
    """Returns (dq, dk, dv). `lse` is what fwd returned (softmax LSE, or the rms for softplus).
    *_g builds: dout is dZ for Z = out * gain, and `gain` (fp32, (h * d,)) is required."""
    wl = window_left if window_left is not None else -1
    wr = 0 if (causal or wl >= 0) else -1
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    kw = {} if gain is None else {"gain": gain}
    ops(name).bwd(dout, q, k, v, out, lse, dq, dk, dv, softmax_scale=softmax_scale, is_causal=causal,
                  window_size_left=wl, window_size_right=wr, **kw)
    return dq, dk, dv


class SoftplusAttn(torch.autograd.Function):
    """Autograd wrapper for a bwd_* build. pre_* variants: pass q = prescale_q(q_raw) and
    softmax_scale = ln2 (then the kernel's scale*log2e is 1 and dq comes out w.r.t. the prescaled q)."""

    @staticmethod
    def forward(ctx, q, k, v, name, causal, window_left, softmax_scale):
        out, lse = fwd(name, q, k, v, causal=causal, window_left=window_left, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.cfg = (name, causal, window_left, softmax_scale)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        name, causal, wl, scale = ctx.cfg
        dq, dk, dv = bwd(name, dout.contiguous(), q, k, v, out, lse, causal=causal, window_left=wl,
                         softmax_scale=scale)
        return dq, dk, dv, None, None, None, None


def mask(Tq, Tk, causal, window_left, device):
    i = torch.arange(Tq, device=device)[:, None] + (Tk - Tq)
    j = torch.arange(Tk, device=device)[None, :]
    m = torch.ones(Tq, Tk, dtype=torch.bool, device=device)
    if causal or (window_left is not None and window_left >= 0):
        m &= j <= i
    if window_left is not None and window_left >= 0:
        m &= j >= i - window_left
    return m


def rexp(x):
    """relu(x) + exp(-|x|)/2: softplus's asymptotes, C1, convex (kFnRexp)."""
    return torch.relu(x) + 0.5 * torch.exp(-x.abs())


FNS = {"softplus": torch.nn.functional.softplus, "rexp": rexp}


def fn_of(variant):
    return "rexp" if "rexp" in variant else "softplus"


def reference(q, k, v, causal=True, window_left=-1, eps=RMS_EPS, dtype=torch.float64, fn="softplus"):
    """fn-attention (softplus by default) + RMSNorm over the head dim, in `dtype`. Returns (O, U, rms)."""
    B, Tq, H, D = q.shape
    Tk = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    qd, kd, vd = (t.to(dtype).transpose(1, 2) for t in (q, k, v))       # (B, H, T, D)
    s = torch.einsum("bhqd,bhkd->bhqk", qd, kd) * scale
    a = FNS[fn](s)
    a = a.masked_fill(~mask(Tq, Tk, causal, window_left, q.device), 0.0)
    u = torch.einsum("bhqk,bhkd->bhqd", a, vd)
    rms = torch.sqrt(u.pow(2).mean(-1, keepdim=True) + eps)
    o = u / rms
    return o.transpose(1, 2), u.transpose(1, 2), rms.squeeze(-1)
