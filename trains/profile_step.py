"""
Where does a training step actually spend its time?

Runs the real model (same construction path as base_train: meta build, optional FP8
conversion, torch.compile, MuonAdamW) on synthetic tokens, so the dataloader is out
of the picture, and reports a CUDA-time breakdown by kernel category plus the
achieved FLOPs and MFU.

    python trains/profile_step.py --depth 20 --device-batch-size 64 --fp8
    python trains/profile_step.py --depth 20 --device-batch-size 64 --no-compile

Needs an idle GPU: timings from a shared one are meaningless.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.common import COMPUTE_DTYPE, get_peak_flops
from nanochat.gpt import GPT, GPTConfig

# kernel-name substrings -> bucket. First match wins, so order matters.
BUCKETS = [
    ("attention (FA4/SDPA)", ("flash", "fmha", "attention", "sm100_fmha", "sm120_fmha")),
    ("GEMM fp8", ("scaled_mm", "float8", "fp8")),
    # nvjet_* is cuBLAS's Blackwell GEMM family; without it the bulk of the GEMM time
    # lands in "other" and the model looks far more bandwidth-bound than it is.
    ("GEMM bf16", ("gemm", "cutlass", "ampere", "nvjet", "sm90", "sm100", "sm80",
                   "matmul", "addmm", " mm ")),
    ("loss / softmax", ("softmax", "cross_entropy", "nll", "tanh")),
    ("optimizer", ("muon", "adamw", "foreach", "polar")),
    ("collectives", ("nccl", "all_reduce", "reduce_scatter", "all_gather")),
    ("elementwise / norm", ("elementwise", "vectorized", "norm", "triton_poi", "triton_red", "copy", "cat")),
]


def bucket_of(name):
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--aspect-ratio", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window-pattern", type=str, default="SSSL")
    ap.add_argument("--device-batch-size", type=int, default=32)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--fp8-recipe", type=str, default="tensorwise")
    ap.add_argument("--muon-variant", type=str, default="nanochat")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--rows", type=int, default=15)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    B, T = args.device_batch_size, args.seq_len

    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    cfg = GPTConfig(sequence_len=T, vocab_size=args.vocab_size, n_layer=args.depth,
                    n_head=model_dim // args.head_dim, n_kv_head=model_dim // args.head_dim,
                    n_embd=model_dim, window_pattern=args.window_pattern)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=dev)
    model.init_weights()

    if args.fp8:
        import torch.nn as nn
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        def filt(mod, fqn):
            return (isinstance(mod, nn.Linear) and mod.in_features % 16 == 0
                    and mod.out_features % 16 == 0 and min(mod.in_features, mod.out_features) >= 128)
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(args.fp8_recipe),
                                   module_filter_fn=filt)
        n8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        print(f"FP8: converted {n8} linear layers ({args.fp8_recipe})")

    flops_per_token = model.estimate_flops()   # counts Float8Linear since the nn.Linear fix
    opt = model.setup_optimizer(muon_variant=args.muon_variant)
    run = model if args.no_compile else torch.compile(model, dynamic=False)

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, args.vocab_size, (B, T), device=dev, generator=g)
    y = torch.randint(0, args.vocab_size, (B, T), device=dev, generator=g)

    def step():
        loss = run(x, y)
        loss.backward()
        opt.step()
        model.zero_grad(set_to_none=True)

    print(f"d{args.depth} dim {model_dim} | B={B} T={T} | dtype {COMPUTE_DTYPE} | "
          f"compile={'off' if args.no_compile else 'on'} | {flops_per_token:.3e} FLOPs/token")
    print(f"warmup {args.warmup} steps (compile + JIT)...")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps

    peak = get_peak_flops(torch.cuda.get_device_name(0))
    tok_s = B * T / dt
    achieved = flops_per_token * tok_s
    print(f"\nstep: {dt*1000:.1f} ms | {tok_s:,.0f} tok/s | {achieved/1e12:.0f} TFLOPS"
          + (f" | MFU {100*achieved/peak:.1f}%" if peak != float("inf") else " | MFU n/a (GPU not in the peak table)"))
    print(f"peak memory: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            step()
        torch.cuda.synchronize()

    # Only real CUDA kernels. key_averages() also carries the CPU-side ops that
    # launched them, and an op's self_device_time_total counts its kernels' time, so
    # including both double-counts: the totals came out at roughly twice the measured
    # step time and the split between categories was distorted, since an op and its
    # kernel do not always share a name.
    from torch.autograd import DeviceType
    events = [
        e for e in prof.key_averages()
        if e.self_device_time_total > 0 and e.device_type == DeviceType.CUDA
    ]
    total = sum(e.self_device_time_total for e in events)
    groups = {}
    for e in events:
        groups.setdefault(bucket_of(e.key), 0.0)
        groups[bucket_of(e.key)] += e.self_device_time_total

    print(f"\nCUDA time by category (total {total/1e3:.1f} ms over 2 steps)")
    for label, us in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {label:24s} {us/1e3:8.1f} ms  {100*us/total:5.1f}%")

    print(f"\ntop {args.rows} kernels")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[:args.rows]:
        print(f"  {e.self_device_time_total/1e3:8.1f} ms  {100*e.self_device_time_total/total:5.1f}%  "
              f"{e.key[:88]}")


if __name__ == "__main__":
    main()
