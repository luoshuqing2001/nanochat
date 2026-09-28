"""
Numerical checks for the softplus FA3 builds against a float64 reference.

Two references, as in docs/softplus_attention_gpu_validation.md section 8.2:
  * float64 math: total error of the kernel.
  * float64 math on the *same BF16-rounded A* the kernel feeds the PV GEMM is not available from
    outside, so the second baseline is the exact-softplus build (sp_mufu2): the increment
    another variant adds on top of it isolates the approximation's own contribution.

    python hopper_softplus/test_correctness.py [variant ...]
"""
import sys

import torch

from sp_attn import fwd, prescale_q, reference

torch.manual_seed(0)

SHAPES = [  # B, T, H, causal, window_left
    (2, 1024, 4, True, -1),
    (2, 2048, 6, True, -1),
    (2, 2048, 6, True, 511),
    (1, 777, 3, True, -1),       # ragged length: partial last tile
    (1, 4096, 2, True, 511),
    (2, 1024, 4, False, -1),
]


def rel_l2(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-30)).item()


def check(variant, B, T, H, causal, wl, scale_q=1.0):
    D = 128
    q = (torch.randn(B, T, H, D, device="cuda") * scale_q).bfloat16()
    k = torch.randn(B, T, H, D, device="cuda").bfloat16()
    v = torch.randn(B, T, H, D, device="cuda").bfloat16()
    q_in = prescale_q(q) if variant.startswith(("pre_", "tile_")) else q
    out, rms = fwd(variant, q_in, k, v, causal=causal, window_left=wl)
    o_ref, _, rms_ref = reference(q, k, v, causal=causal, window_left=wl)
    assert torch.isfinite(out).all(), "non-finite output"
    e_o = rel_l2(out.double(), o_ref)
    e_max = (out.double() - o_ref).abs().max().item()
    e_rms = rel_l2(rms.double(), rms_ref)
    # what BF16 rounding of the exact output alone costs: the floor any kernel sits on
    e_floor = rel_l2(o_ref.bfloat16().double(), o_ref)
    return e_o, e_max, e_rms, e_floor


def main(variants):
    print(f"{'variant':<11} {'B':>2} {'T':>5} {'H':>2} {'causal':>6} {'W':>4} {'qscale':>6} "
          f"{'relL2(O)':>9} {'max|dO|':>8} {'relL2(rms)':>10} {'bf16 floor':>10}")
    worst = {}
    for v in variants:
        for (B, T, H, causal, wl) in SHAPES:
            for qs in (1.0, 4.0):   # 4x: larger scores, exercises the y>0 branch and big softplus
                e_o, e_max, e_rms, e_floor = check(v, B, T, H, causal, wl, qs)
                worst[v] = max(worst.get(v, 0.0), e_o)
                print(f"{v:<11} {B:>2} {T:>5} {H:>2} {str(causal):>6} {wl:>4} {qs:>6} "
                      f"{e_o:>9.2e} {e_max:>8.2e} {e_rms:>10.2e} {e_floor:>10.2e}")
    print("\nworst relL2(O):", {k: f"{x:.2e}" for k, x in worst.items()})
    bad = {k: x for k, x in worst.items() if x > 2e-2}
    if bad:
        print("FAIL (> 2e-2):", bad)
        return 1
    print("OK: all variants within 2e-2 relative L2 of the float64 reference")
    return 0


if __name__ == "__main__":
    vs = sys.argv[1:] or ["sp_mufu2", "sp_poly3", "sp_poly4", "sp_softexp", "sp_mix", "sp_naive"]
    raise SystemExit(main(vs))
