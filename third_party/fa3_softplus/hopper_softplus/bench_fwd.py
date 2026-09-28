"""
Forward-pass timing: softplus+RMSNorm FA3 variants vs stock FA3 softmax (same build, same flags)
and the kernels-hub FA3 nanochat actually trains with.

Timing follows docs/softplus_attention_gpu_validation.md section 10.1: preallocated inputs,
warmup, then several rounds of back-to-back launches captured in a CUDA graph, median of rounds,
and the implementations interleaved round by round so clock/thermal drift hits all of them.
TFLOPS counts only unmasked (query, key) pairs: 4 * D per pair (QK^T and PV).

    python hopper_softplus/bench_fwd.py            # default shape set
    python hopper_softplus/bench_fwd.py --quick
"""
import argparse
import os
import statistics

import torch

from sp_attn import fwd, prescale_q

os.environ.setdefault("HF_HOME", "/home/sky/pretrain_data/lsq_dev/nanochat/.cache/hf")

SHAPES = [  # name, B, T, H, causal, window_left
    ("d12 train (full)", 32, 2048, 6, True, -1),
    ("d12 train (SWA 512)", 32, 2048, 6, True, 511),
    ("d20 train (full)", 32, 2048, 10, True, -1),
    ("d20 train (SWA 512)", 32, 2048, 10, True, 511),
    ("long causal 8k", 4, 8192, 16, True, -1),
    ("long causal 16k", 2, 16384, 16, True, -1),
    ("non-causal 4k", 8, 4096, 16, False, -1),
    ("non-causal 2k", 32, 2048, 8, False, -1),
]


def pairs(T, causal, wl):
    if not causal and wl < 0:
        return T * T
    if wl < 0:
        return T * (T + 1) // 2
    return sum(min(i, wl) + 1 for i in range(T))


def hub_fa3():
    try:
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception as e:  # network or cache miss: skip that column rather than fail
        print(f"(kernels-hub FA3 unavailable: {type(e).__name__}: {e})")
        return None


def graph_timer(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    return g


def bench(fns, rounds=21, target_ms=60.0, graphs_per_fn=3):
    """fns: dict name -> zero-arg callable. Returns name -> median ms per call.

    Each fn is captured into `graphs_per_fn` separate CUDA graphs and all their replays pooled:
    one capture carries a persistent bias of up to ~3% (identical kernels under different captures
    were measured 0.415-0.433 ms), which a single capture per fn turns into a fake difference."""
    for fn in fns.values():   # warmup + JIT
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    # size iterations per graph so each replay is ~target_ms
    graphs = {}
    for name, fn in fns.items():
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(5):
            fn()
        e1.record(); e1.synchronize()
        per = e0.elapsed_time(e1) / 5
        iters = max(3, min(500, int(target_ms / max(per, 1e-3))))
        graphs[name] = ([graph_timer(fn, iters) for _ in range(graphs_per_fn)], iters)
    samples = {n: [] for n in fns}
    names = list(fns)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            gs, iters = graphs[name]
            g = gs[r % len(gs)]
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            samples[name].append(e0.elapsed_time(e1) / iters)
    return {n: statistics.median(v) for n, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sp_mufu2,sp_naive,sp_poly3,sp_poly4,sp_softexp,sp_mix")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shapes", default="", help="comma-separated substrings of shape names to keep")
    args = ap.parse_args()
    variants = args.variants.split(",")
    hub = hub_fa3()
    shapes = SHAPES[:2] if args.quick else SHAPES
    if args.shapes:
        keep = args.shapes.split(",")
        shapes = [s for s in SHAPES if any(k in s[0] for k in keep)]
    dev = torch.cuda.get_device_name()
    print(f"# {dev}, torch {torch.__version__}, bf16, D=128, forward only, ms (TFLOPS)")
    cols = (["hub FA3"] if hub else []) + ["stock FA3"] + variants
    print("| shape | B | T | H | W | " + " | ".join(cols) + " | best softplus / stock |")
    print("|---|---:|---:|---:|---:|" + "---:|" * len(cols) + "---:|")
    for name, B, T, H, causal, wl in shapes:
        D = 128
        q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        q_pre = prescale_q(q)  # pre_* variants: scale folded into Q upstream, not timed
        fns = {}
        if hub:
            fns["hub FA3"] = lambda: hub.flash_attn_func(q, k, v, causal=causal, window_size=(wl, 0 if causal else -1))
        fns["stock FA3"] = lambda: fwd("stock", q, k, v, causal=causal, window_left=wl)
        for var in variants:
            qq = q_pre if (var.startswith(("pre_", "tile_")) or var.endswith("_pre")) else q
            fns[var] = (lambda var=var, qq=qq: fwd(var, qq, k, v, causal=causal, window_left=wl))
        ms = bench(fns)
        flops = 4 * B * H * D * pairs(T, causal, wl)
        cells = [f"{ms[c]:.3f} ({flops / ms[c] / 1e9:.0f})" for c in cols]
        real = [c for c in variants if not c.startswith("diag_")] or variants
        best = min(real, key=lambda c: ms[c])
        print(f"| {name} | {B} | {T} | {H} | {wl} | " + " | ".join(cells)
              + f" | {best} {ms['stock FA3'] / ms[best]:.3f}x |", flush=True)
        del q, k, v, q_pre
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
