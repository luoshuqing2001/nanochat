"""*_du builds (backward consumes dU directly) vs the regular builds (backward derives dU from dO)."""
import torch

from sp_attn import bwd, fwd

torch.manual_seed(0)
ok = True
for base, du_build in (("fn_rexp_pre", "fn_rexp_pre_du"),):
    for causal, wl in ((True, -1), (True, 511), (False, -1)):
        B, T, H, D = 2, 1536, 4, 128
        q, k, v, do = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        scale = 0.6931471805599453 if "pre" in base else None
        o, rms = fwd(base, q, k, v, causal=causal, window_left=wl, softmax_scale=scale)
        # the RMSNorm backward the regular preprocess does (fp32 from bf16, one bf16 rounding)
        of, dof = o.float(), do.float()
        du = ((dof - of * (dof * of).mean(-1, keepdim=True)) / rms.permute(0, 2, 1).unsqueeze(-1)).bfloat16()
        g_ref = bwd(base, do, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
        g_du = bwd(du_build, du, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
        e = max(((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(g_du, g_ref))
        ok &= e < 2e-3
        print(f"{du_build:<16} causal={causal!s:<5} W={wl:>4}  max rel diff vs {base}: {e:.2e}")
print("OK" if ok else "FAILED")
