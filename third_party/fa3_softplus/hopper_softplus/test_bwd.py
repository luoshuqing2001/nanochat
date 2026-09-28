"""
Backward of softplus attention + RMSNorm (bwd_* builds) against float64 autograd on the reference.

Error is reported relative to the float64 gradient and next to what the stock FA3 softmax backward
achieves against *its* float64 reference -- the bar a BF16 attention backward normally clears.

    python hopper_softplus/test_bwd.py
"""
import math

import torch

from sp_attn import LOG2E, SoftplusAttn, prescale_q, reference

torch.manual_seed(0)
LN2 = math.log(2.0)

CASES = [  # B, T, H, causal, window_left
    (2, 1024, 4, True, -1),
    (2, 2048, 3, True, 511),
    (1, 777, 3, True, -1),
    (2, 1024, 2, False, -1),
]


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


def softmax_ref(q, k, v, causal, wl):
    from sp_attn import mask
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask(q.shape[1], k.shape[1], causal, wl, q.device), float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)


def main():
    print(f"{'variant':<14} {'B':>2} {'T':>5} {'H':>2} {'causal':>6} {'W':>4} {'dq':>9} {'dk':>9} {'dv':>9}")
    worst = 0.0
    for (B, T, H, causal, wl) in CASES:
        D = 128
        q0 = torch.randn(B, T, H, D, device="cuda")
        k0 = torch.randn(B, T, H, D, device="cuda")
        v0 = torch.randn(B, T, H, D, device="cuda")
        do = torch.randn(B, T, H, D, device="cuda")
        q, k, v, dob = (t.bfloat16() for t in (q0, k0, v0, do))
        # float64 references on the BF16-rounded inputs
        qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
        o_ref = reference(qd, kd, vd, causal=causal, window_left=wl)[0]
        gq, gk, gv = torch.autograd.grad(o_ref, (qd, kd, vd), dob.double())
        for name in ("bwd_sp_poly3", "bwd_pre_poly3", "bwd_pre_naive"):
            qi = q.clone().requires_grad_()
            ki, vi = k.clone().requires_grad_(), v.clone().requires_grad_()
            if name.startswith("bwd_pre"):
                qp = prescale_q(qi)          # differentiable: the autograd chain adds the scale to dq
                o = SoftplusAttn.apply(qp, ki, vi, name, causal, wl, LN2)
            else:
                o = SoftplusAttn.apply(qi, ki, vi, name, causal, wl, None)
            o.backward(dob)
            e = (rel(qi.grad, gq), rel(ki.grad, gk), rel(vi.grad, gv))
            worst = max(worst, *e)
            print(f"{name:<14} {B:>2} {T:>5} {H:>2} {str(causal):>6} {wl:>4} {e[0]:>9.2e} {e[1]:>9.2e} {e[2]:>9.2e}")
        # the bar: stock FA3 softmax backward vs its own float64 reference
        from sp_attn import ops
        qs, ks, vs = (t.clone() for t in (q, k, v))
        wr = 0 if (causal or wl >= 0) else -1
        out, lse, *_ = ops("bwd_stock").fwd(qs, ks, vs, is_causal=causal, window_size_left=wl, window_size_right=wr)
        dq, dk, dv = torch.empty_like(qs), torch.empty_like(ks), torch.empty_like(vs)
        ops("bwd_stock").bwd(dob, qs, ks, vs, out, lse, dq, dk, dv, is_causal=causal,
                             window_size_left=wl, window_size_right=wr)
        qd2, kd2, vd2 = (t.double().requires_grad_() for t in (q, k, v))
        g2 = torch.autograd.grad(softmax_ref(qd2, kd2, vd2, causal, wl), (qd2, kd2, vd2), dob.double())
        print(f"{'(stock FA3)':<14} {B:>2} {T:>5} {H:>2} {str(causal):>6} {wl:>4} "
              f"{rel(dq, g2[0]):>9.2e} {rel(dk, g2[1]):>9.2e} {rel(dv, g2[2]):>9.2e}")
    print("worst softplus:", f"{worst:.2e}", "OK" if worst < 2e-2 else "FAIL")
    return 0 if worst < 2e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
