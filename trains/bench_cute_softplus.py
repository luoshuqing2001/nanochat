"""Forward-only benchmark: CuTe softplus (flash_attn_4) vs Triton softplus vs FA4 softmax.

Attention-only, so the numbers are kernel time, not step time. `--depth`/`--batch`
reproduce a nanochat layer mix: window_pattern SSSL, so 3/4 of layers are windowed.
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from flash_attn_4.softplus_api import (
    softplus_attn_fa4, softplus_attn_fa4_func, auto_num_splits, auto_balanced_chunk,
)
from flash_attn_4.interface import _flash_attn_fwd
from nanochat.softplus_attention import softplus_attn_func


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_shape(B, T, H, D, W, splits_list):
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    sc = 1.0 / math.sqrt(D)
    rows = {}
    rows["fa4 softmax"] = timeit(lambda: _flash_attn_fwd(
        q, k, v, softmax_scale=sc, causal=(W is None), window_size_left=W,
        window_size_right=(0 if W is not None else None)))
    rows["triton softplus"] = timeit(lambda: softplus_attn_func(
        q, k, v, True, (W if W is not None else -1, 0), 1.0, sc))
    for s in splits_list:
        rows[f"cute softplus s={s}"] = timeit(lambda s=s: softplus_attn_fa4(
            q, k, v, True, (W, 0), 1.0, sc, num_splits=s))
    return rows


def bench_inference(splits_list):
    """Where tile splitting actually pays: one sequence, long context."""
    print("\n=== long-context inference, single sequence (fwd only) ===")
    print("  columns are uniform splits; auto uses the balanced scheduler instead")
    print("  shape                        " + "".join(f"  s={s:<6}" for s in splits_list) + "   auto")
    for tag, B, Tq, Tk, H, D in [
        ("prefill H1  T4096",  1, 4096,  4096,  1, 128),
        ("prefill H1  T8192",  1, 8192,  8192,  1, 128),
        ("prefill H1  T16384", 1, 16384, 16384, 1, 128),
        ("prefill H12 T8192",  1, 8192,  8192,  12, 128),
        ("decode  H6  Tk4096",  1, 1, 4096,  6, 128),
        ("decode  H6  Tk16384", 1, 1, 16384, 6, 128),
        ("decode  H6  Tk65536", 1, 1, 65536, 6, 128),
        ("decode  H12 Tk65536", 1, 1, 65536, 12, 128),
    ]:
        q = torch.randn(B, Tq, H, D, device="cuda", dtype=torch.bfloat16)
        k, v = (torch.randn(B, Tk, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        sc = 1.0 / math.sqrt(D)
        ms = [timeit(lambda s=s: softplus_attn_fa4(q, k, v, True, (None, 0), 1.0, sc, num_splits=s))
              for s in splits_list]
        a = auto_balanced_chunk(B, H, Tq, Tk)
        ams = timeit(lambda: softplus_attn_fa4(q, k, v, True, (None, 0), 1.0, sc, num_splits="auto"))
        print(f"  {tag:<28}" + "".join(f"{m:8.3f} " for m in ms)
              + f"  chunk={a} {ms[0]/ams:.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8])
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    W_short = max(128, ((args.seqlen // 4 + 127) // 128) * 128) - 1
    for label, W in [(f"S window={W_short + 1}", W_short), ("L full causal", None)]:
        print(f"\n=== B{args.batch} T{args.seqlen} H{args.heads} D{args.headdim} | {label} ===")
        rows = bench_shape(args.batch, args.seqlen, args.heads, args.headdim, W, args.splits)
        base = rows["triton softplus"]
        for name, ms in rows.items():
            print(f"  {name:<22} {ms:7.3f} ms   {base / ms:5.2f}x vs triton softplus")
    bench_inference(args.splits)
