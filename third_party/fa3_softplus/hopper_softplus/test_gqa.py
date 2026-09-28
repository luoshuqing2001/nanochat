"""GQA / MQA with PackGQA (the *_pgqa builds): rexp_rmsnorm forward, non-split and split-KV (the atomic
split epilogue addressing packed (token, query head) rows), against the float64 reference with each KV
head repeated over its group. Also repeated calls with other shapes in between (self-cleaning workspace)."""
import math

import torch

from sp_attn import fwd, prescale_q, reference

torch.manual_seed(0)
LN2 = math.log(2.0)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


ok = True
CASES = [  # B, Tq, Tk, Hq, Hkv, window_left
    (1, 1, 4096, 32, 8, -1), (4, 1, 30000, 32, 8, -1), (2, 1, 8192, 16, 1, -1), (1, 1, 65536, 32, 8, 511),
    (2, 1, 4096, 10, 10, -1), (1, 128, 8192, 32, 8, -1), (1, 1000, 1000, 16, 4, -1), (2, 3, 5000, 8, 2, -1),
]
for B, Tq, Tk, Hq, Hkv, wl in CASES:
    q = torch.randn(B, Tq, Hq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Tk, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Tk, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    g = Hq // Hkv
    o_ref, _, rms_ref = reference(q, k.repeat_interleave(g, 2), v.repeat_interleave(g, 2), causal=True,
                                  window_left=wl, fn="rexp")
    qp = prescale_q(q)
    outs = {"non-split": fwd("fn_rexp_pre_pgqa", qp, k, v, window_left=wl, softmax_scale=LN2)}
    for ns in (1, 3, 8, 0):
        o, rms = fwd("split_fn_rexp_pre_pgqa", qp, k, v, window_left=wl, num_splits=ns, softmax_scale=LN2)
        fwd("split_fn_rexp_pre_pgqa", qp[:, :1].contiguous(), k, v, window_left=wl, num_splits=5, softmax_scale=LN2)
        o2, _ = fwd("split_fn_rexp_pre_pgqa", qp, k, v, window_left=wl, num_splits=ns, softmax_scale=LN2)
        outs[f"split {ns}"] = (o, rms)
        ok &= (o2.float() - o.float()).abs().max().item() < 5e-2
    line = []
    for name, (o, rms) in outs.items():
        e, er = rel(o, o_ref), rel(rms, rms_ref)
        good = torch.isfinite(o).all().item() and e < 5e-3 and er < 1e-3
        ok &= good
        line.append(f"{name} {e:.1e}/{er:.0e}" + ("" if good else " FAIL"))
    print(f"B={B} Tq={Tq:<4} Tk={Tk:<5} H={Hq}/{Hkv} W={wl:>3}: " + "  ".join(line))
print("OK" if ok else "FAILED")
