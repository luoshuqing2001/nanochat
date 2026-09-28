"""
Split-KV softplus (atomic reduction, last-arriving warp normalizes) against the float64 reference
and against the same build with num_splits=1.

Also checks the self-cleaning workspace: repeated calls, and alternating shapes, must give the same
answer as the first call (a counter or workspace slice left dirty would show up as a drift).

    python hopper_softplus/test_split.py [variant]
"""
import sys

import torch

import math

from sp_attn import fn_of, fwd, prescale_q, reference

torch.manual_seed(0)

CASES = [  # B, Tq, Tk, H, causal, window_left
    (4, 1, 4096, 8, True, -1),       # decode
    (2, 1, 30000, 4, True, -1),      # decode, ragged long KV
    (1, 1, 65536, 6, True, 511),     # decode, sliding window
    (2, 256, 8192, 4, True, -1),     # chunked prefill against a long cache
    (1, 2048, 2048, 4, True, -1),    # causal prefill: early m_blocks leave later splits empty
    (1, 777, 777, 3, True, -1),      # ragged rows and columns
    (2, 1024, 1024, 2, False, -1),   # non-causal
]


def rel_l2(a, b):
    return ((a - b).double().norm() / b.double().norm().clamp_min(1e-30)).item()


def main(variant):
    ok = True
    print(f"{'B':>2} {'Tq':>5} {'Tk':>6} {'H':>2} {'causal':>6} {'W':>4} {'splits':>6} "
          f"{'relL2 vs f64':>12} {'vs splits=1':>11} {'rms':>8} {'repeat':>7}")
    for (B, Tq, Tk, H, causal, wl) in CASES:
        D = 128
        q = torch.randn(B, Tq, H, D, device="cuda").bfloat16()
        k = torch.randn(B, Tk, H, D, device="cuda").bfloat16()
        v = torch.randn(B, Tk, H, D, device="cuda").bfloat16()
        qi = prescale_q(q) if ("pre_" in variant or variant.endswith("_pre")) else q
        sc = math.log(2.0) if "rexp_pre" in variant else None  # rexp's prescaled builds: softmax_scale = ln2
        o_ref, _, rms_ref = reference(q, k, v, causal=causal, window_left=wl, fn=fn_of(variant))
        o1, _ = fwd(variant, qi, k, v, causal=causal, window_left=wl, num_splits=1, softmax_scale=sc)
        for ns in (2, 3, 8, 32):
            o, rms = fwd(variant, qi, k, v, causal=causal, window_left=wl, num_splits=ns, softmax_scale=sc)
            # interleave another shape, then repeat: exercises the workspace/counter reset
            fwd(variant, qi[:, :1].contiguous(), k, v, causal=causal, window_left=wl, num_splits=5, softmax_scale=sc)
            o_again, _ = fwd(variant, qi, k, v, causal=causal, window_left=wl, num_splits=ns, softmax_scale=sc)
            e = rel_l2(o, o_ref)
            e1 = rel_l2(o, o1)
            er = rel_l2(rms, rms_ref)
            erep = (o_again.float() - o.float()).abs().max().item()
            good = torch.isfinite(o).all().item() and e < 5e-3 and er < 1e-3 and erep < 5e-2
            ok &= good
            print(f"{B:>2} {Tq:>5} {Tk:>6} {H:>2} {str(causal):>6} {wl:>4} {ns:>6} "
                  f"{e:>12.2e} {e1:>11.2e} {er:>8.1e} {erep:>7.1e}" + ("" if good else "   <-- FAIL"))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "split_pre_poly3"))
