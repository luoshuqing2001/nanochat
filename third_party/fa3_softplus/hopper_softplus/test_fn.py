"""
Forward and backward of the softplus-shaped replacement (fn_rexp) against float64
autograd through the same function (sp_attn.rexp), with the output RMSNorm.

    python hopper_softplus/test_fn.py
"""
import torch

import math

from sp_attn import SoftplusAttn, fn_of, prescale_q, reference

torch.manual_seed(0)
CASES = [(2, 1024, 4, True, -1), (2, 2048, 3, True, 511), (1, 777, 3, True, -1), (2, 1024, 2, False, -1)]


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    print(f"{'variant':<8} {'B':>2} {'T':>5} {'H':>2} {'causal':>6} {'W':>4} {'qscale':>6} {'O':>9} {'rms':>9} {'dq':>9} {'dk':>9} {'dv':>9}")
    worst = 0.0
    for name in ("fn_rexp", "fn_rexp_pre"):
        for (B, T, H, causal, wl) in CASES:
            for qs in (1.0, 4.0):
                D = 128
                q = (torch.randn(B, T, H, D, device="cuda") * qs).bfloat16()
                k, v, do = (torch.randn(B, T, H, D, device="cuda").bfloat16() for _ in range(3))
                qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
                o_ref, _, rms_ref = reference(qd, kd, vd, causal=causal, window_left=wl, fn=fn_of(name))
                g = torch.autograd.grad(o_ref, (qd, kd, vd), do.double())
                qi, ki, vi = (t.clone().requires_grad_() for t in (q, k, v))
                pre = name.endswith("_pre")
                scale = math.log(2.0) if pre else None
                o = SoftplusAttn.apply(prescale_q(qi) if pre else qi, ki, vi, name, causal, wl, scale)
                o.backward(do)
                from sp_attn import fwd
                _, rms = fwd(name, prescale_q(q) if pre else q, k, v, causal=causal, window_left=wl, softmax_scale=scale)
                e = [rel(o, o_ref.detach()), rel(rms, rms_ref.detach()), rel(qi.grad, g[0]), rel(ki.grad, g[1]), rel(vi.grad, g[2])]
                worst = max(worst, *e)
                print(f"{name:<8} {B:>2} {T:>5} {H:>2} {str(causal):>6} {wl:>4} {qs:>6} " + " ".join(f"{x:>9.2e}" for x in e))
    print("worst:", f"{worst:.2e}", "OK" if worst < 1e-2 else "FAIL")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
