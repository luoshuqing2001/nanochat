"""
Fused FP8 c_proj grad-input GEMM + gamma + per-head RMSNorm backward (nanochat/fused_proj_rmsnorm.py)
against a float64 reference on the same FP8 operands, next to the unfused path it replaces
(cuBLAS precise FP8 GEMM -> bf16 dz -> bf16 dz*gamma -> fp32 RMSNorm backward from bf16 inputs).

    python -m pytest tests/test_fused_proj_rmsnorm.py -q     (needs a CUDA GPU with FP8: SM89+)
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9), reason="needs FP8")


def make(B=4, T=2048, H=8, D=128, N=1024, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K, M = H * D, B * T
    u = torch.randn(B, T, H, D, device="cuda", generator=g)
    rms = (u.pow(2).mean(-1) + 1e-6).sqrt() * (0.5 + torch.rand(B, T, H, 1, device="cuda", generator=g).squeeze(-1))
    y = (u / u.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).bfloat16().reshape(M, K)   # what attention outputs
    rms = rms.permute(0, 2, 1).contiguous()                                               # (B, H, T) like FA3
    gamma = (1 + 0.1 * torch.randn(K, device="cuda", generator=g))
    dout = torch.randn(M, N, device="cuda", generator=g).bfloat16() * 1e-3
    w = torch.randn(N, K, device="cuda", generator=g) * 0.02                               # c_proj weight [out, in]
    return y, rms, gamma, dout, w, T


def reference64(go8, gi, w8, wi, y, gamma, rms, T):
    dz = (go8.double() * gi.double()) @ (w8.double() * wi.double())
    M, K = dz.shape
    B, H, _ = rms.shape
    D = K // H
    dy = (dz * gamma.double()).view(M, H, D)
    yy = y.double().view(M, H, D)
    r = rms.double().permute(0, 2, 1).reshape(M, H, 1)
    du = (dy - yy * (dy * yy).mean(-1, keepdim=True)) / r
    return du.view(M, K), (dz * y.double()).sum(0)


def unfused(go8, gi, w8, wi, y, gamma, rms, T):
    from nanochat.fp8 import _to_col_major
    dz = torch._scaled_mm(go8, _to_col_major(w8), scale_a=gi, scale_b=wi, out_dtype=torch.bfloat16, use_fast_accum=False)
    gb = gamma.to(torch.bfloat16)
    dy = dz * gb                                         # autograd of y * gamma.to(bf16), in bf16
    dgamma = (dz * y).sum(0).float()                     # ... and its gamma gradient, reduced from bf16
    M, K = dz.shape
    B, H, _ = rms.shape
    D = K // H
    dyf, yf = dy.float().view(M, H, D), y.float().view(M, H, D)
    r = rms.permute(0, 2, 1).reshape(M, H, 1)
    du = ((dyf - yf * (dyf * yf).mean(-1, keepdim=True)) / r).to(torch.bfloat16)   # the FA3 preprocess
    return du.view(M, K), dgamma


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("shape", [dict(), dict(B=2, T=1000, H=6, N=768), dict(B=1, T=2048, H=10, N=1280)])
def test_fused_vs_reference(shape):
    from nanochat.fp8 import _to_fp8
    from nanochat.fused_proj_rmsnorm import proj_bwd_rmsnorm
    y, rms, gamma, dout, w, T = make(**shape)
    go8, gi = _to_fp8(dout, torch.float8_e5m2)
    w8, wi = _to_fp8(w, torch.float8_e4m3fn)
    du64, dg64 = reference64(go8, gi, w8, wi, y, gamma, rms, T)
    du_f, dg_f = proj_bwd_rmsnorm(go8, gi, w8.t().contiguous(), wi, y, gamma, rms, T)
    du_u, dg_u = unfused(go8, gi, w8, wi, y, gamma, rms, T)
    e_f, e_u = rel(du_f, du64), rel(du_u, du64)
    g_f, g_u = rel(dg_f, dg64), rel(dg_u, dg64)
    print(f"\n{shape}: dU rel err fused {e_f:.2e} unfused {e_u:.2e} | dgamma fused {g_f:.2e} unfused {g_u:.2e}")
    assert e_f <= e_u * 1.05 and e_f < 5e-3      # never worse than what it replaces
    assert g_f <= g_u * 1.05 and g_f < 1e-3


def test_deterministic():
    from nanochat.fp8 import _to_fp8
    from nanochat.fused_proj_rmsnorm import proj_bwd_rmsnorm
    y, rms, gamma, dout, w, T = make()
    go8, gi = _to_fp8(dout, torch.float8_e5m2)
    w8, wi = _to_fp8(w, torch.float8_e4m3fn)
    wt = w8.t().contiguous()
    a = proj_bwd_rmsnorm(go8, gi, wt, wi, y, gamma, rms, T)
    b = proj_bwd_rmsnorm(go8, gi, wt, wi, y, gamma, rms, T)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
