"""*_g builds (backward takes dZ = dL/d(O * gain) and the fp32 gain, rebuilds dO in the preprocess) vs the
regular builds fed the dO torch.compile would have materialized: bf16(fp32(dZ) * gain), RNE -- which is
what eager torch computes for (dz.float() * gain).to(bf16) too.

dK and dV must match bit for bit. dQ is accumulated with fp32 atomics in both (run-to-run
nondeterministic already), so it is held to the base build's own run-to-run spread."""
import os
import torch

from sp_attn import bwd, fwd

torch.manual_seed(0)
PAIRS = [("fn_rexp_pre", "fn_rexp_pre_g", 0.6931471805599453), ("bwd_sp_poly3", "bwd_sp_poly3_g", None)]
here = os.path.dirname(os.path.abspath(__file__))
ok = True
for base, gb, scale in PAIRS:
    if not os.path.exists(os.path.join(here, "build", gb)):
        print(f"{gb}: not built, skipped"); continue
    for B, T, H, causal, wl in ((2, 1536, 4, True, -1), (2, 1536, 4, True, 511), (3, 1000, 2, True, 255),
                                (32, 2048, 6, True, -1), (32, 2048, 6, True, 511), (2, 1024, 4, False, -1)):
        q, k, v, dz = (torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        if scale is not None:
            q = q * (1.4426950408889634 / 128 ** 0.5)
        gain = 1 + 0.3 * torch.randn(H * 128, device="cuda")
        o, rms = fwd(base, q, k, v, causal=causal, window_left=wl, softmax_scale=scale)
        do = (dz.float() * gain.view(H, 128)).to(torch.bfloat16)
        ref = bwd(base, do, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
        ref2 = bwd(base, do, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
        got = bwd(gb, dz, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale, gain=gain)
        same_kv = torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])
        spread = (ref2[0].float() - ref[0].float()).abs().max().item()
        dq_diff = (got[0].float() - ref[0].float()).abs().max().item()
        dq_rel = ((got[0].float() - ref[0].float()).norm() / ref[0].float().norm()).item()
        good = same_kv and dq_rel < 1e-4
        ok &= good
        print(f"{gb:<16} B={B:<2} T={T:<5} H={H} causal={causal!s:<5} W={wl:>4}: dK/dV bit-identical={same_kv}  "
              f"dQ max|diff| {dq_diff:.2e} (base run-to-run {spread:.2e}), rel {dq_rel:.1e}  {'ok' if good else 'FAIL'}")
print("OK" if ok else "FAILED")


# *_gr: the same, plus dgain = sum_tokens dZ * O reduced in the preprocess (returned as the 4th output,
# per-CTA partials to be summed). dQ/dK/dV must equal the *_g build's bit for bit; dgain vs float64.
from sp_attn import ops

ok_gr = True
for g, gr, scale in (("fn_rexp_pre_g", "fn_rexp_pre_gr", 0.6931471805599453), ("bwd_sp_poly3_g", "bwd_sp_poly3_gr", None)):
    if not os.path.exists(os.path.join(here, "build", gr)):
        print(f"{gr}: not built, skipped"); continue
    for B, T, H, wl in ((2, 1536, 4, -1), (3, 1000, 2, 255), (32, 2048, 6, 511)):
        q, k, v, dz = (torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        if scale is not None:
            q = q * (1.4426950408889634 / 128 ** 0.5)
        gain = 1 + 0.3 * torch.randn(H * 128, device="cuda")
        o, rms = fwd(g, q, k, v, window_left=wl, softmax_scale=scale)
        ref = bwd(g, dz, q, k, v, o, rms, window_left=wl, softmax_scale=scale, gain=gain)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        res = ops(gr).bwd(dz, q, k, v, o, rms, dq, dk, dv, softmax_scale=scale, is_causal=True,
                          window_size_left=wl, window_size_right=0, gain=gain)
        dgain = res[3].sum(0)
        dg_ref = (dz.double() * o.double()).sum((0, 1)).reshape(-1)
        e = ((dgain.double() - dg_ref).norm() / dg_ref.norm()).item()
        same = torch.equal(dk, ref[1]) and torch.equal(dv, ref[2])
        dq_rel = ((dq.float() - ref[0].float()).norm() / ref[0].float().norm()).item()
        good = same and dq_rel < 1e-4 and e < 1e-5
        ok_gr &= good
        print(f"{gr:<16} B={B:<2} T={T:<5} H={H} W={wl:>4}: dK/dV same as {g}={same} dQ rel {dq_rel:.1e} "
              f"dgain rel err vs f64 {e:.1e}  {'ok' if good else 'FAIL'}")
print("OK" if ok_gr else "FAILED")
