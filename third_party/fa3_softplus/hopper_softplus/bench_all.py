"""
rexp_rmsnorm vs FA3 softmax, kernel level, on H100 -- training, prefill and decode.

  train    forward, backward, forward+backward at nanochat's training shapes (B=32 x 2048; heads of
           d12/d16/d20; full causal and the SWA-512 short window) and one long-context shape.
           FA3: bwd_stock. rexp: fn_rexp_pre (q prescaled by log2e/sqrt(D), softmax_scale = ln2).
  prefill  forward only, inference shapes (B=1 long prompts, a batched prompt, SWA, chunked prefill
           against a long cache). Both sides get FA3's num_splits heuristic ("auto") and the best of
           a sweep. FA3: stock (split + combine kernel). rexp: split_fn_rexp_pre (atomic split
           reduction, no combine kernel); 1 split = fn_rexp_pre's non-split path.
  decode   Tq = 1 against a KV cache, same two sides as prefill.

Timing: CUDA graphs, several captures per function, rounds alternating order, median (bench_fwd.bench).
"x" columns are FA3 time / rexp time: > 1 means rexp is faster.

    python hopper_softplus/bench_all.py train|prefill|decode > hopper_softplus/bench_all_h100_<section>.md
"""
import math
import sys

import torch

from bench_fwd import bench
from sp_attn import bwd, fwd, prescale_q

LN2 = math.log(2.0)
D = 128
SWEEP = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)


def rnd(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def train():
    shapes = [  # name, B, T, H, window_left
        ("d12 full", 32, 2048, 6, -1), ("d12 SWA 512", 32, 2048, 6, 512),
        ("d16 full", 32, 2048, 8, -1), ("d16 SWA 512", 32, 2048, 8, 512),
        ("d20 full", 32, 2048, 10, -1), ("d20 SWA 512", 32, 2048, 10, 512),
        ("long causal 8k", 4, 8192, 16, -1),
    ]
    print(f"# train, {torch.cuda.get_device_name()}, bf16, D=128, causal, ms")
    print("| shape | B | T | H | W | fwd FA3 | fwd rexp | x | bwd FA3 | bwd rexp | x | bwd rexp, dU given | x | fwd+bwd FA3 | fwd+bwd rexp | x |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, B, T, H, wl in shapes:
        q, k, v, do = rnd(B, T, H, D), rnd(B, T, H, D), rnd(B, T, H, D), rnd(B, T, H, D)
        qp = prescale_q(q)
        o_s, l_s = fwd("bwd_stock", q, k, v, window_left=wl)
        o_r, l_r = fwd("fn_rexp_pre", qp, k, v, window_left=wl, softmax_scale=LN2)
        fns = {
            ("fwd", "fa3"): lambda: fwd("bwd_stock", q, k, v, window_left=wl),
            ("fwd", "rexp"): lambda: fwd("fn_rexp_pre", qp, k, v, window_left=wl, softmax_scale=LN2),
            ("bwd", "fa3"): lambda: bwd("bwd_stock", do, q, k, v, o_s, l_s, window_left=wl),
            ("bwd", "rexp"): lambda: bwd("fn_rexp_pre", do, qp, k, v, o_r, l_r, window_left=wl, softmax_scale=LN2),
            # the *_du build (dU produced upstream, as with NANOCHAT_FA3_FUSE_PROJ): no RMSNorm preprocess
            ("bwd", "rexp_du"): lambda: bwd("fn_rexp_pre_du", do, qp, k, v, o_r, l_r, window_left=wl, softmax_scale=LN2),
        }
        def both_s():
            o, l = fwd("bwd_stock", q, k, v, window_left=wl)
            bwd("bwd_stock", do, q, k, v, o, l, window_left=wl)
        def both_r():
            o, l = fwd("fn_rexp_pre", qp, k, v, window_left=wl, softmax_scale=LN2)
            bwd("fn_rexp_pre", do, qp, k, v, o, l, window_left=wl, softmax_scale=LN2)
        fns[("all", "fa3")], fns[("all", "rexp")] = both_s, both_r
        ms = bench(fns, rounds=15, target_ms=60.0)
        cells = []
        for p in ("fwd", "bwd", "all"):
            a, b = ms[(p, "fa3")], ms[(p, "rexp")]
            cells += [f"{a:.3f}", f"{b:.3f}", f"**{a / b:.3f}**"]
            if p == "bwd":
                c = ms[("bwd", "rexp_du")]
                cells += [f"{c:.3f}", f"**{a / c:.3f}**"]
        print(f"| {name} | {B} | {T} | {H} | {wl} | " + " | ".join(cells) + " |", flush=True)
        del q, k, v, do, qp, o_s, o_r
        torch.cuda.empty_cache()


def splits_table(title, shapes, sweep=SWEEP, header=True):
    if header:
        print(f"# {title}, {torch.cuda.get_device_name()}, bf16, D=128, causal (bottom-right aligned), us; "
              f"split sweep {sweep}")
        print("| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | rexp auto | rexp best (splits) | rexp 1 split | x auto | x best |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, B, Tq, Tk, H, wl in shapes:
        q, k, v = rnd(B, Tq, H, D), rnd(B, Tk, H, D), rnd(B, Tk, H, D)
        qp = prescale_q(q)
        fns = {("fa3", 0): lambda: fwd("stock", q, k, v, window_left=wl, num_splits=0),
               ("rexp", 0): lambda: fwd("split_fn_rexp_pre", qp, k, v, window_left=wl, num_splits=0, softmax_scale=LN2)}
        for ns in sweep:
            fns[("fa3", ns)] = lambda ns=ns: fwd("stock", q, k, v, window_left=wl, num_splits=ns)
            fns[("rexp", ns)] = lambda ns=ns: fwd("split_fn_rexp_pre", qp, k, v, window_left=wl, num_splits=ns, softmax_scale=LN2)
        fns[("rexp", "nosplit")] = lambda: fwd("fn_rexp_pre", qp, k, v, window_left=wl, softmax_scale=LN2)
        ms = bench(fns, rounds=9, target_ms=30.0)
        fb = min(sweep, key=lambda n: ms[("fa3", n)])
        rb = min(tuple(sweep) + ("nosplit",), key=lambda n: ms[("rexp", n)])
        us = lambda key: ms[key] * 1e3
        rb_label = "1, non-split build" if rb == "nosplit" else rb
        print(f"| {name} | {B} | {Tq} | {Tk} | {H} | {wl} | {us(('fa3', 0)):.1f} | {us(('fa3', fb)):.1f} ({fb}) "
              f"| {us(('rexp', 0)):.1f} | {us(('rexp', rb)):.1f} ({rb_label}) | {us(('rexp', 'nosplit')):.1f} "
              f"| **{ms[('fa3', 0)] / ms[('rexp', 0)]:.2f}** | **{ms[('fa3', fb)] / ms[('rexp', rb)]:.2f}** |", flush=True)
        del q, k, v, qp
        torch.cuda.empty_cache()


PREFILL_SWEEP = (1, 2, 4, 8)  # FA3's split workspace is splits x B x H x Tq x D fp32: 21 GB at 64 x 8x8k


def prefill(rows=None):
    shapes = [  # name, B, Tq, Tk, H, window_left
        ("prompt 2k", 1, 2048, 2048, 10, -1),
        ("prompt 8k", 1, 8192, 8192, 10, -1),
        ("prompt 32k", 1, 32768, 32768, 10, -1),
        ("batched prompts 8x8k", 8, 8192, 8192, 10, -1),
        ("prompt 32k, SWA 512", 1, 32768, 32768, 10, 512),
        ("chunked prefill 512 vs 32k cache", 1, 512, 32768, 10, -1),
        ("chunked prefill 2k vs 32k cache", 4, 2048, 32768, 10, -1),
    ]
    splits_table("prefill (forward only)", shapes[rows] if rows else shapes, sweep=PREFILL_SWEEP, header=not rows)


def decode():
    splits_table("decode (Tq = 1)", [
        ("decode 2k", 1, 1, 2048, 10, -1),
        ("decode 8k", 1, 1, 8192, 10, -1),
        ("decode 32k", 1, 1, 32768, 10, -1),
        ("decode 128k", 1, 1, 131072, 10, -1),
        ("decode batch 8, 8k", 8, 1, 8192, 10, -1),
        ("decode batch 8, 32k", 8, 1, 32768, 10, -1),
        ("decode batch 32, 4k", 32, 1, 4096, 10, -1),
        ("decode batch 64, 2k", 64, 1, 2048, 10, -1),
        ("decode SWA 512, batch 8, 32k", 8, 1, 32768, 10, 512),
    ])


if __name__ == "__main__":
    if sys.argv[1] == "prefill" and len(sys.argv) > 2:  # e.g. "3:" to resume from the 4th shape
        a, _, b = sys.argv[2].partition(":")
        prefill(slice(int(a) if a else None, int(b) if b else None))
    else:
        {"train": train, "prefill": prefill, "decode": decode}[sys.argv[1]]()
