"""*_dqf builds (dQ memset + fp32->bf16 postprocess folded into the backward kernel, via per-row-block
arrival counters; dQaccum persists all-zero between calls) vs the same builds without it.

dK / dV must match bit for bit (untouched path); dQ up to the order of the fp32 atomic adds, which
neither build fixes. Repeated calls must give the same dQ: if a call left dQaccum or a counter
non-zero, the next one would be off by a whole tile. Shapes grow (buffer reallocation) and shrink."""
import torch

from sp_attn import bwd, fwd

torch.manual_seed(0)
ok = True
PAIRS = [  # base, dqf, softmax_scale (rexp's prescaled builds take q * log2e/sqrt(D) and scale ln2)
    ("fn_rexp_pre", "fn_rexp_pre_dqf", 0.6931471805599453),
    ("fn_rexp_pre_du", "fn_rexp_pre_du_dqf", 0.6931471805599453),  # dout is dU here: same inputs to both
    ("bwd_stock", "bwd_stock_dqf", None),                       # stock softmax: the change is function-agnostic
]
CASES = [  # B, T, H, causal, window_left
    (2, 1536, 4, True, -1), (2, 1536, 4, True, 511), (2, 1536, 4, False, -1),
    (3, 1000, 2, True, -1), (3, 1000, 2, True, 255), (1, 4096, 6, True, 511),
    (32, 2048, 6, True, -1), (32, 2048, 6, True, 511),  # d12 training shapes
    (1, 200, 3, True, -1), (2, 1536, 4, True, 511),     # shrink after growing
]
for base, dqf, scale in PAIRS:
    for B, T, H, causal, wl in CASES:
        q, k, v, do = (torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        if scale is not None:
            q = q * (1.4426950408889634 / 128 ** 0.5)
        o, rms = fwd(base, q, k, v, causal=causal, window_left=wl, softmax_scale=scale)
        dq0, dk0, dv0 = bwd(base, do, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
        dqs = []
        for rep in range(3):
            dq1, dk1, dv1 = bwd(dqf, do, q, k, v, o, rms, causal=causal, window_left=wl, softmax_scale=scale)
            dqs.append(dq1)
            same_kv = torch.equal(dk1, dk0) and torch.equal(dv1, dv0)
            e_q = ((dq1.float() - dq0.float()).norm() / dq0.float().norm()).item()
            finite = bool(torch.isfinite(dq1).all())
            good = same_kv and e_q < 1e-3 and finite
            ok &= good
            if rep == 0 or not good:
                print(f"{dqf:<18} B={B:<2} T={T:<5} H={H} causal={causal!s:<5} W={wl:>4} rep {rep}: "
                      f"dK/dV identical={same_kv}  dQ rel diff {e_q:.1e}  {'ok' if good else 'FAIL'}")
        e_rep = max(((a.float() - dqs[0].float()).norm() / dqs[0].float().norm()).item() for a in dqs[1:])
        ok &= e_rep < 1e-3
        if e_rep >= 1e-3:
            print(f"   repeated calls disagree: {e_rep:.1e}  FAIL")
print("OK" if ok else "FAILED")
