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


def sweep(base, short, T, D):
    """Search BLOCK_M/BLOCK_N/warps/stages for forward and backward, per window regime.

    The defaults in _fwd_config were swept on a GB10: 48 SMs, 228 KB of shared memory per
    SM, ~206 GB/s. A B200 has 148 SMs and far more shared memory and bandwidth, so larger
    tiles and deeper pipelining should win there and the GB10 choice is probably wrong.
    Paste the winners back into _fwd_config and _bwd."""
    import triton
    q, k, v = base
    B, _, H, _ = q.shape
    do = torch.randn_like(q)
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    strides = (q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2))

    combos = [(bm, bn, w, st) for bm in (64, 128, 256) for bn in (64, 128)
              for w in (4, 8, 16) for st in (2, 3, 4)]
    print(f"sweeping {len(combos)} configurations x 2 windows x 2 phases. Each one compiles "
          f"before it runs, so expect minutes, and run with python -u or no pipe: a pipe "
          f"buffers this output until the end.", flush=True)

    for label, win in ((f"S ({short})", short), ("L (full)", -1)):
        for phase in ("forward", "backward"):
            rows = []
            print(f"\n{label} {phase}:", end="", flush=True)
            for n_done, (bm, bn, warps, stages) in enumerate(combos):
                if True:
                    if True:
                        if True:
                            print(f"\r{label} {phase}: {n_done + 1}/{len(combos)} "
                                  f"({bm}x{bn} w{warps} s{stages})    ", end="", flush=True)
                            cfg = dict(WINDOW=win, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn,
                                       num_warps=warps, num_stages=stages)
                            try:
                                if phase == "forward":
                                    o = torch.empty_like(q)
                                    fn = lambda: SA._fwd_kernel[(triton.cdiv(T, bm), B, H)](
                                        q, k, v, o, *strides, o.stride(0), o.stride(1), o.stride(2),
                                        T, D ** -0.5, 1.0, REVERSE=False, **cfg)
                                else:
                                    a = (*strides, do.stride(0), do.stride(1), do.stride(2),
                                         T, D ** -0.5, 1.0)
                                    def fn():
                                        SA._bwd_kv_kernel[(triton.cdiv(T, bn), B, H)](
                                            q, k, v, do, dk, dv, *a, REVERSE=False, **cfg)
                                        SA._bwd_q_kernel[(triton.cdiv(T, bm), B, H)](
                                            q, k, v, do, dq, *a, REVERSE=False, **cfg)
                                rows.append((tm(fn, 10, 4, 1), bm, bn, warps, stages))
                            except Exception:
                                pass
            rows.sort()
            print(f"\r{label} {phase}: top 5 of {len(rows)}/{len(combos)} that compiled" + " " * 20,
                  flush=True)
            for t, bm, bn, w, st in rows[:5]:
                print(f"  {t:7.2f}ms  BLOCK_M={bm:<4} BLOCK_N={bn:<4} num_warps={w:<3} "
                      f"num_stages={st}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64, help="micro-batch, i.e. DEVICE_BATCH_SIZE")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--sweep", action="store_true",
                    help="search launch configurations on this machine instead of benchmarking; "
                         "the shipped defaults were swept on a GB10 and do not transfer")
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

    if args.sweep:
        sweep(base, short, T, D)
        return

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
