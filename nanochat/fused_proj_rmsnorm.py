"""
Backward of  out = c_proj(gamma * y),  y = RMSNorm(U)  per head, fused into the FP8 grad-input GEMM.

For rexp_rmsnorm attention the output RMSNorm's backward needs the per-(token, head)
statistic mean_d(dy * y) and then dU = (dy - y * mean_d(dy * y)) / rms. Unfused, that is three
passes over B*T*C activations after the grad-input GEMM writes dz = dout @ W: autograd's gamma
multiply (read dz, write dy, read y for dgamma), and the attention backward's preprocess (read y,
read dy, write dU). Here the GEMM's epilogue does all of it from the fp32 accumulator:

    dz = dout @ W                  FP8 x FP8 -> fp32, as Float8Linear's backward (below)
    dy = dz * gamma
    c  = mean_d(dy * y)            per token, per head: one program owns a full head (D columns)
    dU = (dy - y * c) / rms        -> bf16, what the attention backward consumes directly
    dgamma = sum_tokens(dz * y)    per-program partial sums, reduced in a second (tiny) step

FP8 numerics are Float8Linear's (nanochat/fp8.py) exactly: dout quantized to e5m2 and W to e4m3
with its tensorwise `_to_fp8`, and "precise" accumulation (use_fast_accum=False) -- the hardware
FP8 accumulator is promoted to fp32 every 128 reduction elements, which reproduces cuBLAS's
precise mode bit for bit (checked in tests/test_fused_proj_rmsnorm.py). The epilogue is fp32 and
rounds once, where the unfused path rounds dz and dy to bf16 on the way.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _proj_bwd_rmsnorm_kernel(
    go_ptr, wt_ptr, sa_ptr, sb_ptr,          # dout fp8 e5m2 [M, N]; W^T fp8 e4m3 [K, N]; inverse scales
    y_ptr, gamma_ptr, rms_ptr,               # attention output y [M, K] bf16; gamma [K]; rms [B, H, T] fp32
    du_ptr, dgamma_part_ptr,                 # dU [M, K] bf16; dgamma partials [cdiv(M, BM), K] fp32
    M, N, K, T, stride_rms_b, stride_rms_h,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    # head fastest: the H programs of one row block run together and share its dout tile in L2
    # (M fastest re-read dout from DRAM once per head: -12% at d16)
    NH = K // D
    pid = tl.program_id(0)
    pid_m = pid // NH
    h = pid - pid_m * NH
    rm = pid_m * BM + tl.arange(0, BM)
    rk = h * D + tl.arange(0, D)
    rn = tl.arange(0, BN)
    m_ok = rm < M
    b = rm // T
    t = rm - b * T
    # issued before the GEMM so the epilogue's reads overlap it (another -11%)
    y = tl.load(y_ptr + rm[:, None] * K + rk[None, :], mask=m_ok[:, None], other=0.0)
    r = tl.load(rms_ptr + b * stride_rms_b + h * stride_rms_h + t, mask=m_ok, other=1.0)

    acc = tl.zeros((BM, D), dtype=tl.float32)
    for n0 in range(0, N, BN):
        n_ok = (n0 + rn) < N
        a = tl.load(go_ptr + rm[:, None] * N + (n0 + rn)[None, :], mask=m_ok[:, None] & n_ok[None, :], other=0.0)
        bt = tl.load(wt_ptr + rk[:, None] * N + (n0 + rn)[None, :], mask=n_ok[None, :], other=0.0)
        # a fresh hardware accumulation per BN=128 chunk, promoted to fp32: cuBLAS's precise mode
        acc += tl.dot(a, tl.trans(bt))
    dz = acc * (tl.load(sa_ptr) * tl.load(sb_ptr))

    g = tl.load(gamma_ptr + rk).to(tl.float32)
    y = y.to(tl.float32)
    dy = dz * g[None, :]
    c = tl.sum(dy * y, axis=1) * (1.0 / D)
    du = (dy - y * c[:, None]) / r[:, None]
    tl.store(du_ptr + rm[:, None] * K + rk[None, :], du.to(du_ptr.dtype.element_ty), mask=m_ok[:, None])
    # rows past M load as 0, so they add nothing; deterministic: no atomics
    tl.store(dgamma_part_ptr + pid_m * K + rk, tl.sum(dz * y, axis=0))


def proj_bwd_rmsnorm(go_fp8, go_inv, wt_fp8, w_inv, y, gamma, rms, T, BM=64, num_warps=4, num_stages=4):
    """go_fp8 [M, N] e5m2, wt_fp8 [K, N] e4m3 (= W^T, row-major), y [M, K] bf16 with K = H*D,
    gamma [K], rms [B, H, T] fp32. Returns (dU [M, K] in y's dtype, dgamma [K] fp32)."""
    M, N = go_fp8.shape
    K = wt_fp8.shape[0]
    B, H, T_ = rms.shape
    D = K // H
    assert T_ == T and B * T == M and D * H == K and D in (64, 128), (go_fp8.shape, wt_fp8.shape, rms.shape)
    assert go_fp8.is_contiguous() and wt_fp8.is_contiguous() and y.is_contiguous() and rms.is_contiguous()
    du = torch.empty_like(y)
    nm = triton.cdiv(M, BM)
    part = torch.empty(nm, K, device=y.device, dtype=torch.float32)
    # BM=64 / 4 warps / 4 stages: best of a sweep on H100 at d12-d20 shapes (M = 32 x 2048)
    _proj_bwd_rmsnorm_kernel[(nm * H,)](
        go_fp8, wt_fp8, go_inv, w_inv, y, gamma, rms, du, part,
        M, N, K, T, rms.stride(0), rms.stride(1),
        BM=BM, BN=128, D=D, num_warps=num_warps, num_stages=num_stages)
    return du, part.sum(0)
