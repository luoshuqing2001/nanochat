"""
Softplus attention on this machine: correctness, speed against FA4, and whether the
split-K / atomic aggregation path is worth anything here.

    python trains/bench_softplus.py                     # d20 shapes
    python trains/bench_softplus.py --depth 12 --batch 128

Everything was tuned on a GB10: 48 SMs and ~206 GB/s of achieved bandwidth. A B200 has
roughly 3x the SMs and 39x the bandwidth, and two of the conclusions there depend on
exactly that ratio, so they need re-measuring rather than assuming:

  * split-K pays for balance with traffic (one extra partial output per split). On GB10
    the fixed cost alone -- zeroing the fp32 accumulator and the scale-and-cast pass --
    was 1.74 ms against a 1.83 ms kernel, so it never won at training shapes. Divide the
    traffic terms by 39 and that verdict can flip.
  * issuing query tiles longest-first is free and helped only when programs were scarce.

Needs an idle GPU.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import nanochat.softplus_attention as SA


def tm(fn, iters=20, warmup=8, reps=3):
    def once():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000
    return sorted(once() for _ in range(reps))[reps // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64, help="micro-batch, i.e. DEVICE_BATCH_SIZE")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    dim = ((args.depth * 64 + args.head_dim - 1) // args.head_dim) * args.head_dim
    B, T, H, D = args.batch, args.seq_len, dim // args.head_dim, args.head_dim
    short = -(-T // 4 // 128) * 128          # nanochat's 'S' window
    print(f"{props.name}: {props.multi_processor_count} SMs | d{args.depth} B{B} T{T} H{H} D{D} bf16")

    x = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    bw = 2 * x.numel() * 4 / 2**30 / (tm(lambda: x.mul(2.0), 50, 10) / 1000)
    print(f"achieved bandwidth ~{bw:.0f} GB/s (GB10 measured ~206)\n")

    base = [torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    try:
        import flash_attn_4.fa3_compat as fa4
        have_fa4 = fa4.has_custom_op()
    except Exception:
        have_fa4 = False

    print(f"{'layer':>10} {'impl':>12} {'fwd':>9} {'fwd+bwd':>10}")
    per_layer = {}
    for label, win in ((f"S ({short})", short), ("L (full)", -1)):
        for name in ("softplus", "FA4"):
            if name == "FA4" and not have_fa4:
                continue
            ws = (win, 0) if win >= 0 else (T, 0)
            if name == "softplus":
                f = lambda: SA.softplus_attn_func(*base, window_size=(win, 0))
                def fb():
                    q, k, v = (t.detach().clone().requires_grad_(True) for t in base)
                    SA.softplus_attn_func(q, k, v, window_size=(win, 0)).sum().backward()
            else:
                f = lambda: fa4.flash_attn_func(*base, causal=True, window_size=ws)
                def fb():
                    q, k, v = (t.detach().clone().requires_grad_(True) for t in base)
                    fa4.flash_attn_func(q, k, v, causal=True, window_size=ws).sum().backward()
            fwd, both = tm(f), tm(fb, 10, 4)
            per_layer[(label, name)] = both
            print(f"{label:>10} {name:>12} {fwd:>8.2f}ms {both:>9.2f}ms")

    # what that means for a whole model: SSSL puts 3 of every 4 layers on the short window
    n_short = args.depth - max(1, args.depth // 4)
    n_long = args.depth - n_short
    print(f"\nd{args.depth} SSSL = {n_short} windowed + {n_long} full-context layers, attention per micro-batch:")
    for name in ("softplus", "FA4"):
        k1, k2 = (f"S ({short})", name), ("L (full)", name)
        if k1 in per_layer and k2 in per_layer:
            print(f"  {name:>9}: {n_short * per_layer[k1] + n_long * per_layer[k2]:7.1f} ms")

    # split-K: does the traffic still dominate on this machine?
    print(f"\nsplit-K forward, full context (GB10: never won at training shapes)")
    ref = SA.softplus_attn_func(*base, window_size=(-1, 0))
    programs = (T + 63) // 64 * B * H
    print(f"  {'no split':>12} {tm(lambda: SA.softplus_attn_func(*base, window_size=(-1, 0))):7.2f}ms"
          f"   ({programs:,} query-tile programs, {programs / props.multi_processor_count:.0f}x SMs)")
    for sp in (2, 4, 8):
        try:
            got = SA._fwd_splitk(*base, -1, 1.0, D ** -0.5, sp)
            err = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            t = tm(lambda: SA._fwd_splitk(*base, -1, 1.0, D ** -0.5, sp))
            print(f"  {'splits ' + str(sp):>12} {t:7.2f}ms   (rel err {err:.1e})")
        except Exception as e:
            print(f"  splits {sp}: {type(e).__name__}")

    print(f"\nreverse tile order heuristic: {'on' if SA._reverse_tiles(programs) else 'off'} at this shape")


if __name__ == "__main__":
    main()
