"""
Softplus-shaped attention with an output RMSNorm, on the FA3 (Hopper) kernels of the flash-attention
repo's `lsq/softplus-fa3` branch:

    u_i = sum_{j in K_i} f(q_i . k_j * scale) v_j,    o_i = u_i / sqrt(mean(u_i^2) + eps)

per head, with f either exact softplus or a cheap stand-in for it (same asymptotes: 0 far left,
x far right; convex, C2), which is not a softplus approximation -- it defines a different attention:

    "rexp_rmsnorm"      f(x) = relu(x) + exp(-|x|) / 2                        (one exp2)
    "softplus_rmsnorm"  f(x) = softplus(x)                                    (the FA3 build of it)

The RMSNorm gain is folded into c_proj by CausalSelfAttention, as for softplus_rmsnorm.

Training uses the FA3 kernels (SM90 only), loaded as prebuilt shared libraries: the sources are
vendored in third_party/fa3_softplus, `bash trains/build_fa3_softplus.sh` builds them (see _LIB_DIR
below for where they are looked up). Inference
with a KV cache, and every non-SM90 device, uses the float32 torch reference below: correct, but
it materializes the score matrix, so it is for sampling/eval and tests, not for training.
"""
import os

import torch

RMS_EPS = 1e-6  # compiled into the kernels (FLASHATTN_SOFTPLUS_EPS); keep the two in sync
LN2, LOG2E = 0.6931471805599453, 1.4426950408889634
# rexp runs the "prescaled" build: q arrives already multiplied by log2e/sqrt(D) (one elementwise op
# that torch.compile fuses into q's norm-and-scale) and the kernel is told softmax_scale = ln2, which
# turns the exp2 argument into a sign-bit OR instead of an FMUL per score (~7% faster forward).
KIND_TO_BUILD = {"rexp_rmsnorm": "fn_rexp_pre",
                 # exact softplus (not a stand-in): the same attention as softplus_rmsnorm_attention.py's
                 # CuTe kernels (SM80/SM120), here on SM90. log1p(2^-|x|) is a cubic in fp32 (hopper/softplus.h
                 # kSpPoly3: max abs error ~1e-4 of softplus, below bf16's resolution of A); ~0.9x FA3's speed
                 "softplus_rmsnorm": "bwd_sp_poly3"}
# NANOCHAT_FA3_DQFUSE=1: backward builds that fold dQ's memset and fp32->bf16 postprocess kernel into
# the backward kernel (atomic arrival counts per dQ row block, converter CTAs at the end of the grid;
# dQaccum lives in a persistent per-device buffer, all-zero between calls -- one stream at a time).
# Same math (dK/dV bit-identical, dQ up to the atomics' summation order) but SLOWER: +8-20% backward,
# the DRAM traffic is the same and the conversion can't be overlapped well (RESULTS.md in the
# flash-attention repo, "a negative result"). Kept for reference; leave it off.
DQF_BUILDS = {"fn_rexp_pre": "fn_rexp_pre_dqf", "fn_rexp_pre_du": "fn_rexp_pre_du_dqf"}
DQFUSE = os.environ.get("NANOCHAT_FA3_DQFUSE", "0") == "1"


def _bwd_build(build):
    return DQF_BUILDS.get(build, build) if DQFUSE else build


# NANOCHAT_FA3_DGAIN (default on): attention + the output gain as one autograd node, Z = O * gamma, whose
# backward hands dZ and the fp32 gamma straight to a *_g build. Unfused, torch.compile's backward
# materializes dO = bf16(fp32(dZ) * gamma) (it drops the parameter's bf16 cast) in a kernel that reads
# dZ and writes dO, only for the attention backward's preprocess to read dO back; the *_g preprocess
# computes the very same bf16 value in registers instead. Bit-identical results (loss, dK, dV, dgamma,
# every weight gradient; dQ as always up to its fp32 atomics' order), one fewer pass over dZ and dO.
# dgamma keeps the expression autograd derived for `y * gamma.to(bf16)`, so inductor emits the same
# reduction. Only for the FP8 c_proj path (the bf16 one folds gamma into the weight instead).
G_BUILDS = {"fn_rexp_pre": "fn_rexp_pre_g", "bwd_sp_poly3": "bwd_sp_poly3_g"}
# NANOCHAT_FA3_DGAIN=2: the *_gr builds, whose preprocess also reduces dgamma = sum_tokens dZ * O (it reads
# both anyway), so the separate reduction kernel over dZ and O goes too. fp32 sums in another order and not
# rounded to bf16 as autograd's are: not bit-identical to 0/1 (more accurate, ~1e-7 relative to float64).
GR_BUILDS = {"fn_rexp_pre": "fn_rexp_pre_gr", "bwd_sp_poly3": "bwd_sp_poly3_gr"}
# Inference with a KV cache: the same forward (same function approximation) with FA3's split-KV
# path, whose splits reduce U with atomics in the kernel (hopper/epilogue_fwd.hpp) -- long cache,
# small batch decode gets the whole GPU without a combine kernel.
SPLIT_BUILDS = {"rexp_rmsnorm": "split_fn_rexp_pre", "softplus_rmsnorm": "split_sp_poly3"}
DGAIN_MODE = os.environ.get("NANOCHAT_FA3_DGAIN", "1")  # "0" off, "1" bit-identical, "2" + fused dgamma
DGAIN = DGAIN_MODE in ("1", "2")


@torch.library.custom_op("fn_rmsnorm_attn::bwd_g", mutates_args=(), device_types="cuda")
def _op_bwd_g(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
              dz: torch.Tensor, rms: torch.Tensor, gamma: torch.Tensor, bwd_build: str,
              softmax_scale: float, window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _ops(bwd_build).bwd(dz.contiguous(), q, k, v, out, rms, dq, dk, dv,
                        softmax_scale=softmax_scale if softmax_scale > 0 else None, is_causal=True,
                        window_size_left=window_left, window_size_right=0, gain=gamma.contiguous())
    return dq, dk, dv


@_op_bwd_g.register_fake
def _(q, k, v, out, dz, rms, gamma, bwd_build, softmax_scale, window_left):
    return tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))


@torch.library.custom_op("fn_rmsnorm_attn::bwd_gr", mutates_args=(), device_types="cuda")
def _op_bwd_gr(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
               dz: torch.Tensor, rms: torch.Tensor, gamma: torch.Tensor, bwd_build: str,
               softmax_scale: float, window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    res = _ops(bwd_build).bwd(dz.contiguous(), q, k, v, out, rms, dq, dk, dv,
                              softmax_scale=softmax_scale if softmax_scale > 0 else None, is_causal=True,
                              window_size_left=window_left, window_size_right=0, gain=gamma.contiguous())
    return dq, dk, dv, res[3].sum(0)  # the preprocess's per-CTA dgamma partials


@_op_bwd_gr.register_fake
def _(q, k, v, out, dz, rms, gamma, bwd_build, softmax_scale, window_left):
    return (*(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v)),
            torch.empty(gamma.shape, device=q.device, dtype=torch.float32))


class _AttnGain(torch.autograd.Function):
    # Forward: exactly the unfused ops (the attention op, then CausalSelfAttention's
    # y.contiguous().view(B, T, -1) * gamma.to(y.dtype)), so inductor fuses and rounds them the same.
    @staticmethod
    def forward(ctx, q, k, v, gamma, fwd_build, bwd_build, softmax_scale, window_left):
        B, T = q.shape[:2]
        y, rms = torch.ops.fn_rmsnorm_attn.fwd(q, k, v, fwd_build, bwd_build, softmax_scale, window_left)
        y2 = y.contiguous().view(B, T, -1)
        ctx.save_for_backward(q, k, v, y, rms, gamma)
        ctx.cfg = (bwd_build, softmax_scale, window_left)
        return y2 * gamma.to(dtype=y2.dtype)

    @staticmethod
    def backward(ctx, dz):
        q, k, v, y, rms, gamma = ctx.saved_tensors
        B, T, H, D = q.shape
        if ctx.cfg[0].endswith("_gr"):  # NANOCHAT_FA3_DGAIN=2: dgamma comes out of the preprocess
            dq, dk, dv, dgamma = torch.ops.fn_rmsnorm_attn.bwd_gr(q, k, v, y, dz.reshape(B, T, H, D), rms, gamma, *ctx.cfg)
            return dq, dk, dv, dgamma.to(gamma.dtype), None, None, None, None
        dq, dk, dv = torch.ops.fn_rmsnorm_attn.bwd_g(q, k, v, y, dz.reshape(B, T, H, D), rms, gamma, *ctx.cfg)
        # what autograd derives for gamma through `y2 * gamma.to(bf16)`: mul, sum to gamma's shape, cast
        dgamma = (dz * y.view(B, T, -1)).sum([0, 1], keepdim=True).view(-1).to(gamma.dtype)
        return dq, dk, dv, dgamma, None, None, None, None


def fn_rmsnorm_attn_gain(q, k, v, kind, gamma, window_size=(-1, 0)):
    """y.view(B, T, -1) * gamma.to(y.dtype) with y = fn_rmsnorm_attn_func(q, k, v, kind, ...): same
    values, same gradients, bit for bit; the backward skips materializing dy (see DGAIN above).
    Training path only (SM90, bf16, D=128, no KV cache); gamma is the fp32 (H * D,) parameter."""
    assert window_size[1] in (0, None) and q.shape[2] == k.shape[2]
    left = _window_left(window_size)
    scale = -1.0
    fwd_build = bwd_build = KIND_TO_BUILD[kind]
    if kind == "rexp_rmsnorm":
        q = q * (LOG2E / q.shape[-1] ** 0.5)
        scale = LN2
    builds = GR_BUILDS if DGAIN_MODE == "2" else G_BUILDS
    return _AttnGain.apply(q.contiguous(), k.contiguous(), v.contiguous(), gamma,
                           fwd_build, builds[bwd_build], scale, left)


def dgain_ok(q, kind):
    """Whether fn_rmsnorm_attn_gain applies: the switch, the device, and a *_g build for the kind."""
    return DGAIN and _fa3_ok(q) and KIND_TO_BUILD.get(kind) in G_BUILDS and not DQFUSE
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The kernels' sources are vendored in third_party/fa3_softplus (hopper/ + hopper_softplus/ + cutlass
# headers); `bash trains/build_fa3_softplus.sh` builds them into its build dir. $NANOCHAT_FA3_FN_DIR
# overrides; a flash-attention checkout next to this repo (lsq/softplus-fa3) is the fallback.
_VENDORED_BUILD = os.path.join(_REPO, "third_party", "fa3_softplus", "hopper_softplus", "build")
_SIBLING_BUILD = os.path.join(os.path.dirname(_REPO), "flash-attention", "hopper_softplus", "build")
_LIB_DIR = os.environ.get("NANOCHAT_FA3_FN_DIR") or (
    _VENDORED_BUILD if os.path.isdir(_VENDORED_BUILD) else _SIBLING_BUILD)
_loaded = {}


def _ops(build):
    if build not in _loaded:
        path = os.path.join(_LIB_DIR, build, f"fa3_{build}.so")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Build it in the flash-attention repo (branch lsq/softplus-fa3): "
                f"python hopper_softplus/build.py {build}, or point NANOCHAT_FA3_FN_DIR at the build dir.")
        torch.ops.load_library(path)
        _loaded[build] = getattr(torch.ops, f"fa3_{build}")
    return _loaded[build]


def _window_left(window_size):
    left = window_size[0]
    return -1 if left is None or left < 0 else int(left)


# ------------------------------------------------------------------------------------------------
# The elementwise functions and a float32 reference (inference, non-SM90, tests)

def rexp(x):
    return torch.relu(x) + 0.5 * torch.exp(-x.abs())


FNS = {"rexp_rmsnorm": rexp, "softplus_rmsnorm": torch.nn.functional.softplus}


def reference_attention(q, k, v, kind, window_left=-1, q_offset=0, eps=RMS_EPS):
    """q: (B, Tq, H, D), k/v: (B, Tk, H, D). Query i sits at absolute position q_offset + i and
    attends keys j <= q_offset + i (and >= q_offset + i - window_left). float32 math."""
    B, Tq, H, D = q.shape
    Tk = k.shape[1]
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / D ** 0.5
    i = torch.arange(Tq, device=q.device)[:, None] + q_offset
    j = torch.arange(Tk, device=q.device)[None, :]
    allowed = j <= i
    if window_left >= 0:
        allowed &= j >= i - window_left
    a = FNS[kind](s).masked_fill(~allowed, 0.0)
    u = torch.einsum("bhqk,bkhd->bqhd", a, v.float())
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps)
    return u.to(q.dtype)


# ------------------------------------------------------------------------------------------------
# torch.library wrappers, as in softplus_rmsnorm_attention.py: called directly from a compiled
# model, dynamo would trace into the extension call and break the graph.

@torch.library.custom_op("fn_rmsnorm_attn::fwd", mutates_args=(), device_types="cuda")
def _op_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fwd_build: str, bwd_build: str,
            softmax_scale: float, window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
    out, rms, *_ = _ops(fwd_build).fwd(
        q, k, v, softmax_scale=softmax_scale if softmax_scale > 0 else None, is_causal=True,
        window_size_left=window_left, window_size_right=0, num_splits=1)
    return out, rms


@_op_fwd.register_fake
def _(q, k, v, fwd_build, bwd_build, softmax_scale, window_left):
    B, T, H, _ = q.shape
    return q.new_empty(B, T, H, v.shape[-1]), torch.empty(B, H, T, device=q.device, dtype=torch.float32)


@torch.library.custom_op("fn_rmsnorm_attn::bwd", mutates_args=(), device_types="cuda")
def _op_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
            dout: torch.Tensor, rms: torch.Tensor, bwd_build: str, softmax_scale: float,
            window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _ops(bwd_build).bwd(dout.contiguous(), q, k, v, out, rms, dq, dk, dv,
                        softmax_scale=softmax_scale if softmax_scale > 0 else None, is_causal=True,
                        window_size_left=window_left, window_size_right=0)
    return dq, dk, dv


@_op_bwd.register_fake
def _(q, k, v, out, dout, rms, bwd_build, softmax_scale, window_left):
    return tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v, fwd_build, bwd_build, softmax_scale, window_left = inputs
    out, rms = output
    ctx.save_for_backward(q, k, v, out, rms)
    ctx.cfg = (bwd_build, softmax_scale, window_left)


def _backward(ctx, dout, _drms):
    q, k, v, out, rms = ctx.saved_tensors
    dq, dk, dv = torch.ops.fn_rmsnorm_attn.bwd(q, k, v, out, dout, rms, *ctx.cfg)
    return dq, dk, dv, None, None, None, None


torch.library.register_autograd("fn_rmsnorm_attn::fwd", _backward, setup_context=_setup_context)


def _fa3_ok(q):
    return (q.is_cuda and torch.cuda.get_device_capability(q.device)[0] == 9
            and q.dtype == torch.bfloat16 and q.shape[-1] == 128)


def fn_rmsnorm_attn_func(q, k, v, kind, window_size=(-1, 0)):
    """Causal training path. q, k, v are (B, T, H, D) with nanochat's (left, 0) window."""
    assert window_size[1] in (0, None), f"{kind} attention is causal-only here"
    assert q.shape[2] == k.shape[2], f"{kind}: the FA3 build is MHA-only (no GQA)"
    left = _window_left(window_size)
    if not _fa3_ok(q):
        return reference_attention(q, k, v, kind, left)
    scale = -1.0  # kernel default, 1/sqrt(D)
    fwd_build = bwd_build = KIND_TO_BUILD[kind]
    if kind == "rexp_rmsnorm":
        q = q * (LOG2E / q.shape[-1] ** 0.5)
        scale = LN2
    out, _ = torch.ops.fn_rmsnorm_attn.fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                                           fwd_build, _bwd_build(bwd_build), scale, left)
    return out


def fn_rmsnorm_attn_with_kvcache(q, k_cache, v_cache, kind, k=None, v=None, cache_seqlens=None,
                                 window_size=(-1, 0)):
    """Inference path: writes k, v into the cache at cache_seqlens[0] (the engine keeps all batch
    entries at one position), then attends the first cache_seqlens + Tq keys causally. The caller
    advances cache_seqlens. SM90: the split-KV FA3 builds (SPLIT_BUILDS); elsewhere the float32
    reference."""
    Tq = q.shape[1]
    pos = int(cache_seqlens[0].item())
    if k is not None and v is not None:
        k_cache[:, pos:pos + Tq] = k
        v_cache[:, pos:pos + Tq] = v
    end = pos + Tq
    kc, vc = k_cache[:, :end], v_cache[:, :end]  # views: FA3 takes any batch stride
    left = _window_left(window_size)
    if _fa3_ok(q) and kind in SPLIT_BUILDS and q.shape[2] == kc.shape[2]:
        # FA3's causal mask is bottom-right aligned for seqlen_q < seqlen_k: query i (absolute
        # position pos + i) attends keys <= pos + i, and the window counts back from there -- the
        # cache semantics exactly.
        # Too few (query block, batch, head) tiles to fill the GPU -- small-batch decode, a short
        # chunk against a long cache: the split-KV build, FA3's num_splits heuristic, the splits'
        # partial U summed with atomics in the kernel (no combine kernel: U is unnormalized until the
        # RMSNorm). Otherwise the training build's forward: 10-15% faster than the split build run
        # with one split (bench_all_h100_prefill.md in the flash-attention repo).
        scale = -1.0
        if kind == "rexp_rmsnorm":
            q = q * (LOG2E / q.shape[-1] ** 0.5)
            scale = LN2
        B, Tq_, H = q.shape[:3]
        tiles = B * H * -(-Tq_ // 128)
        # FA3's heuristic sizes the splits by the whole cache, not the window: a short window (the
        # SWA layers) has too few keys to split (decode, 8 x 32k cache, W=512: 24.8 us auto, 16.3 us unsplit)
        short_window = 0 <= left <= 2048
        if tiles < torch.cuda.get_device_properties(q.device).multi_processor_count and not short_window:
            out, *_ = _ops(SPLIT_BUILDS[kind]).fwd(
                q.contiguous(), kc, vc, softmax_scale=scale if scale > 0 else None, is_causal=True,
                window_size_left=left, window_size_right=0, num_splits=0)
        else:
            out, *_ = _ops(KIND_TO_BUILD[kind]).fwd(
                q.contiguous(), kc, vc, softmax_scale=scale if scale > 0 else None, is_causal=True,
                window_size_left=left, window_size_right=0, num_splits=1)
        return out
    return reference_attention(q, kc, vc, kind, left, q_offset=pos)


# ------------------------------------------------------------------------------------------------
# attention + gain + FP8 c_proj as one op, with the RMSNorm backward fused into c_proj's grad GEMM
#
# Unfused, the backward of  out = c_proj(gamma * y)  writes dz = dout @ W, autograd's gamma multiply
# reads it back and writes dy (and reads y for dgamma), and the attention backward's preprocess reads
# y and dy to write dU. Here one Triton GEMM (nanochat/fused_proj_rmsnorm.py) goes from dout straight
# to dU and dgamma, and the attention backward runs a build that takes dU as is (*_du: its preprocess
# no longer touches O or dO). The forward and the FP8 numerics are Float8Linear's, operation for
# operation; the backward is more accurate than the unfused one (two fewer bf16 roundings).
# Enabled with NANOCHAT_FA3_FUSE_PROJ=1 when c_proj is a Float8Linear (--fp8).

DU_BUILD = {"fn_rexp_pre": "fn_rexp_pre_du"}
FUSE_PROJ = os.environ.get("NANOCHAT_FA3_FUSE_PROJ", "0") == "1"


@torch.library.custom_op("fn_attn_proj::rmsnorm_bwd", mutates_args=(), device_types="cuda")
def _rmsnorm_bwd(go_fp8: torch.Tensor, go_inv: torch.Tensor, w_fp8: torch.Tensor, w_inv: torch.Tensor,
                 y: torch.Tensor, gamma: torch.Tensor, rms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from nanochat.fused_proj_rmsnorm import proj_bwd_rmsnorm
    B, T, H, D = y.shape
    du, dgamma = proj_bwd_rmsnorm(go_fp8, go_inv, w_fp8.t().contiguous(), w_inv, y.view(B * T, H * D), gamma, rms, T)
    return du.view(B, T, H, D), dgamma


@_rmsnorm_bwd.register_fake
def _(go_fp8, go_inv, w_fp8, w_inv, y, gamma, rms):
    return torch.empty_like(y), torch.empty(gamma.shape, device=y.device, dtype=torch.float32)


class _AttnProjFP8(torch.autograd.Function):
    # Only the kernels are opaque ops; the FP8 quantization, the gain and the cuBLAS GEMMs stay plain
    # torch code (as in Float8Linear's _Float8Matmul), so torch.compile fuses them like the unfused
    # path's (as an opaque op they ran as ~10 eager elementwise kernels: 2x slower sublayer).

    @staticmethod
    def forward(ctx, q, k, v, gamma, weight, fwd_build, bwd_build, softmax_scale, window_left):
        from nanochat.fp8 import _to_fp8
        B, T, H, D = q.shape
        y, rms = torch.ops.fn_rmsnorm_attn.fwd(q, k, v, fwd_build, bwd_build, softmax_scale, window_left)
        # exactly the unfused path: c_proj(y * gamma.to(y.dtype)) through Float8Linear's forward
        z = y.reshape(B * T, H * D) * gamma.to(y.dtype)
        in_fp8, in_inv = _to_fp8(z, torch.float8_e4m3fn)
        w_fp8, w_inv = _to_fp8(weight, torch.float8_e4m3fn)
        out = torch._scaled_mm(in_fp8, w_fp8.t(), scale_a=in_inv, scale_b=w_inv, out_dtype=y.dtype, use_fast_accum=True)
        ctx.save_for_backward(q, k, v, gamma, y, rms, in_fp8, in_inv, w_fp8, w_inv)
        ctx.cfg = (bwd_build, softmax_scale, window_left)
        return out.view(B, T, -1)

    @staticmethod
    def backward(ctx, dout):
        from nanochat.fp8 import _to_col_major, _to_fp8
        q, k, v, gamma, y, rms, in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors
        B, T, H, D = q.shape
        go_fp8, go_inv = _to_fp8(dout.reshape(B * T, -1), torch.float8_e5m2)
        # grad input, fused: dout -> dz -> dy -> dU, and dgamma (Float8Linear's GEMM 1 + everything after)
        du, dgamma = torch.ops.fn_attn_proj.rmsnorm_bwd(go_fp8, go_inv, w_fp8, w_inv, y, gamma, rms)
        # grad weight: Float8Linear's GEMM 2, unchanged
        dweight = torch._scaled_mm(go_fp8.t().contiguous(), _to_col_major(in_fp8), scale_a=go_inv, scale_b=in_inv,
                                   out_dtype=dout.dtype, use_fast_accum=False)
        # a *_du build: takes dU where the others take dO, and its preprocess skips O and dO
        dq, dk, dv = torch.ops.fn_rmsnorm_attn.bwd(q, k, v, y, du, rms, *ctx.cfg)
        return dq, dk, dv, dgamma.to(gamma.dtype), dweight, None, None, None, None


def fn_rmsnorm_attn_proj_fp8(q, k, v, kind, gamma, weight, window_size=(-1, 0)):
    """out = c_proj(gamma * fn_rmsnorm_attn(q, k, v)) with c_proj an FP8 (Float8Linear-equivalent)
    projection of `weight` [n_out, H*D]. Returns (B, T, n_out). Training path (SM90, bf16, D=128);
    rexp_rmsnorm only (the kinds with a *_du build)."""
    assert _fa3_ok(q) and q.shape[2] == k.shape[2] and window_size[1] in (0, None)
    assert KIND_TO_BUILD[kind] in DU_BUILD, f"{kind}: no *_du build for the fused c_proj backward"
    left = _window_left(window_size)
    build = KIND_TO_BUILD[kind]
    q = q * (LOG2E / q.shape[-1] ** 0.5)  # rexp's prescaled build
    return _AttnProjFP8.apply(q.contiguous(), k.contiguous(), v.contiguous(), gamma, weight,
                              build, _bwd_build(DU_BUILD[build]), LN2, left)
