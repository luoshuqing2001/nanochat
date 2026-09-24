"""Compare fixed KV tasks with the existing CuTe balanced forward on an idle GPU.

Includes allocation/zero/cast costs. CUDA events report device time; wall time also
includes Python dispatch. Run from the repo: python trains/bench_softplus_decode.py
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.testing
from flash_attn_4.softplus_api import softplus_attn_fa4, auto_balanced_chunk
from nanochat.softplus_decode import softplus_decode


def timing(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    device, wall = [], []
    for _ in range(3):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        begin.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1000 / iters)
        device.append(begin.elapsed_time(end) / iters)
    return {"device_ms": statistics.median(device), "wall_ms": statistics.median(wall),
            "graph_ms": triton.testing.do_bench_cudagraph(fn, rep=100)}


def bench_training():
    from flash_attn_4.softplus_api import softplus_attn_fa4_func
    from nanochat.softplus_attention import softplus_attention
    base = [torch.randn(8, 2048, 12, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for _ in range(3)]
    grad = torch.randn_like(base[0])
    rows = []
    for w in (511, 2048):
        def old():
            out = softplus_attn_fa4_func(*base, window_size=(w, 0), bwd_impl=1)
            return torch.autograd.grad(out, base, grad)
        def new():
            out = softplus_attention(*base, window_size=(w, 0))
            return torch.autograd.grad(out, base, grad)
        for name, fn in (("old_hybrid", old), ("hybrid", new)):
            row = dict(phase="fwd_bwd", batch=8, cache=2048, heads=12, window=w,
                       impl=name, **timing(fn, 10))
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def bench_dynamic():
    """One call per fresh cache length, including the old work-table construction."""
    q = torch.randn(1, 1, 6, 128, device="cuda", dtype=torch.bfloat16)
    kc, vc = [torch.randn(1, 4200, 6, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
    rows = []
    for w in (None, 511):
        def old(k, v):
            chunk = auto_balanced_chunk(1, 6, 1, k.shape[1], window_left=w, device=q.device)
            return softplus_attn_fa4(q, k, v, window_size=(w, 0), balanced_chunk=chunk)
        def new(k, v):
            return softplus_attn_fa4(q, k, v, window_size=(w, 0), num_splits="auto")
        # Compile all signatures before measurement, then clear only the old table cache.
        for t in range(4097, 4161):
            for fn in (old, new):
                fn(kc[:, :t], vc[:, :t])
        torch.cuda.synchronize()
        from flash_attn_4.balanced_scheduler import _build_work_table_cached
        for name, fn in (("cute_balanced", old), ("auto", new)):
            times = []
            for _ in range(3):
                _build_work_table_cached.cache_clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for t in range(4097, 4161):
                    fn(kc[:, :t], vc[:, :t])
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000 / 64)
            row = dict(phase="dynamic_decode", window=w, impl=name,
                       wall_ms=statistics.median(times))
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(12)
    torch.set_num_threads(2)
    print(torch.cuda.get_device_name(), flush=True)
    rows = []
    shapes = [(1, 6, 4096, None), (1, 6, 16384, None), (1, 6, 65536, None),
              (1, 12, 65536, None), (1, 6, 65536, 511), (1, 6, 65536, 512),
              (8, 6, 4096, None)]
    if args.quick:
        shapes = shapes[:2]
    for b, h, t, w in shapes:
        d = args.head_dim
        q = torch.randn(b, 1, h, d, device="cuda", dtype=torch.bfloat16)
        # Sliced, overallocated caches match the inference wrapper's layout.
        k, v = [torch.randn(b, t + 17, h, d, device="cuda", dtype=q.dtype)[:, :t]
                for _ in range(2)]
        start = max(0, t - w - 1) if w is not None else 0
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k[:, start:].float()) * d ** -0.5
        ref = torch.einsum("bhqk,bkhd->bqhd", torch.nn.functional.softplus(scores),
                           v[:, start:].float()) / (t - start)
        def old_path():
            chunk = auto_balanced_chunk(b, h, 1, t, window_left=w, device=q.device)
            return softplus_attn_fa4(q, k, v, window_size=(w, 0), balanced_chunk=chunk)
        funcs = {"cute_balanced": old_path,
                 "auto": lambda: softplus_attn_fa4(q, k, v, window_size=(w, 0), num_splits="auto")}
        for chunk in (64, 128, 256, 512):
            funcs[f"fixed_{chunk}"] = lambda chunk=chunk: softplus_decode(q, k, v, w, chunk_size=chunk)
        for name, fn in funcs.items():
            out = fn().float()
            error = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-9))
            assert error < 0.01, (name, error)
            row = dict(batch=b, heads=h, head_dim=d, cache=t, window=w, impl=name, error=error,
                       **timing(fn, 10 if args.quick else 30))
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.dynamic:
        rows.extend(bench_dynamic())
    if args.training:
        rows.extend(bench_training())
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
