"""
FA4 vs SDPA: numerical parity and speed, at the shapes this repo actually trains on.

Both implementations are compared against a float32 SDPA reference, so "closer to
the reference" is the meaningful statement, not "identical to each other" (bf16
kernels differ in accumulation order by design).

    python trains/bench_attention.py
    python trains/bench_attention.py --depth 12 --batch 16 --seq-len 2048

Needs an idle GPU: it reports timings.
"""

import argparse
import os
import sys
import time

# running this as a script path puts trains/ on sys.path, not the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import nanochat.flash_attention as fa


def set_impl(name):
    """Force 'fa4' / 'sdpa' / 'fa3' and re-resolve the module-level switches."""
    fa._override_impl = name
    fa.USE_FA3 = fa._resolve_use_fa3()
    fa.USE_FA4 = fa._resolve_use_fa4()
    return fa.impl_name()


def sdpa_reference_fp32(q, k, v, window):
    """(B, T, H, D) causal attention in fp32, with an explicit sliding-window mask."""
    qs, ks, vs = (t.transpose(1, 2).float() for t in (q, k, v))
    T = qs.size(2)
    row = torch.arange(T, device=q.device).unsqueeze(1)
    col = torch.arange(T, device=q.device).unsqueeze(0)
    mask = col <= row
    if 0 <= window < T:
        mask = mask & ((row - col) <= window)
    y = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask)
    return y.transpose(1, 2)


def run_once(q, k, v, window_size, backward):
    y = fa.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    if backward:
        y.sum().backward()
    return y


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def err(x, ref):
    d = (x.float() - ref).abs()
    return d.max().item(), d.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window-pattern", type=str, default="SSSL")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    n_head = (args.depth * 64 + args.head_dim - 1) // args.head_dim * args.head_dim // args.head_dim
    B, T, H, D = args.batch, args.seq_len, n_head, args.head_dim

    # nanochat's window sizes: 'S' = ceil(seq_len/4) rounded up to 128, 'L' = seq_len
    short = -(-args.seq_len // 4 // 128) * 128
    configs = [("L  (full context)", args.seq_len), (f"S  (window {short})", short)]

    print(f"GPU: {torch.cuda.get_device_name(0)} | HAS_FA3={fa.HAS_FA3} HAS_FA4={fa.HAS_FA4}")
    print(f"shape: B={B} T={T} H={H} D={D} bf16  (d{args.depth}, pattern {args.window_pattern})")
    print()

    torch.manual_seed(0)
    base = [torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3)]

    results = {}
    for label, window in configs:
        window_size = (window, 0)
        ref = sdpa_reference_fp32(*base, window)

        print(f"--- {label} ---")
        for impl in ("sdpa", "fa4"):
            if impl == "fa4" and not fa.HAS_FA4:
                print("  fa4: unavailable")
                continue
            got = set_impl(impl)
            assert got == impl, f"asked for {impl}, resolved to {got}"

            q, k, v = (t.clone().requires_grad_(True) for t in base)
            y = run_once(q, k, v, window_size, backward=True)
            fwd_max, fwd_mean = err(y, ref)
            grads = [q.grad.clone(), k.grad.clone(), v.grad.clone()]

            fwd_ms = timeit(lambda: fa.flash_attn_func(
                base[0], base[1], base[2], causal=True, window_size=window_size),
                iters=args.iters)

            def fwdbwd():
                qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in base)
                run_once(qq, kk, vv, window_size, backward=True)
            fwdbwd_ms = timeit(fwdbwd, iters=max(5, args.iters // 2), warmup=3)

            mem = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            results[(label, impl)] = (fwd_ms, fwdbwd_ms, grads)
            print(f"  {impl:4s} | fwd {fwd_ms:8.2f} ms | fwd+bwd {fwdbwd_ms:8.2f} ms | "
                  f"peak {mem:7.0f} MiB | err vs fp32: max {fwd_max:.4f} mean {fwd_mean:.6f}")

        if (label, "fa4") in results and (label, "sdpa") in results:
            gs, gf = results[(label, "sdpa")][2], results[(label, "fa4")][2]
            gd = max((a.float() - b.float()).abs().max().item() for a, b in zip(gs, gf))
            sp_f = results[(label, "sdpa")][0] / results[(label, "fa4")][0]
            sp_fb = results[(label, "sdpa")][1] / results[(label, "fa4")][1]
            print(f"  -> grad max |fa4 - sdpa|: {gd:.4f}")
            print(f"  -> FA4 speedup: fwd {sp_f:.2f}x | fwd+bwd {sp_fb:.2f}x")
        print()

    # project onto one training step of the real model
    pattern = args.window_pattern.upper()
    layers = [pattern[i % len(pattern)] for i in range(args.depth)]
    layers[-1] = "L"  # gpt.py forces the last layer to full context
    n_long, n_short = layers.count("L"), layers.count("S")
    print(f"d{args.depth} with pattern {pattern}: {n_short} sliding-window layers + {n_long} full-context layers")
    for impl in ("sdpa", "fa4"):
        key_l, key_s = (configs[0][0], impl), (configs[1][0], impl)
        if key_l in results and key_s in results:
            per_step = n_long * results[key_l][1] + n_short * results[key_s][1]
            print(f"  {impl:4s}: attention fwd+bwd ≈ {per_step:7.1f} ms per micro-batch")


if __name__ == "__main__":
    main()
