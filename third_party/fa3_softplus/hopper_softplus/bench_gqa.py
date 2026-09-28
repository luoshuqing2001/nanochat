"""
GQA / MQA decode (and multi-token decode, e.g. speculative verification), rexp_rmsnorm vs FA3 softmax,
both with PackGQA (the group's query heads share a CTA and one read of their KV head):
  FA3:  stock_pgqa, split + combine kernel.  rexp: split_fn_rexp_pre_pgqa (atomic split reduction) and
  fn_rexp_pre_pgqa (non-split). "auto" = FA3's num_splits heuristic, best = min over a sweep.
x = FA3 / rexp (> 1: rexp faster). TB/s = KV bytes / time.

    python hopper_softplus/bench_gqa.py > hopper_softplus/bench_gqa_h100.md
"""
import math

import torch

from bench_fwd import bench
from sp_attn import fwd, prescale_q

LN2 = math.log(2.0)
SWEEP = (1, 2, 4, 6, 8, 12, 13, 16, 26, 32)
SHAPES = [  # name, B, Tq, Tk, Hq, Hkv
    ("GQA 32/8, 8k", 1, 1, 8192, 32, 8),
    ("GQA 32/8, 32k", 1, 1, 32768, 32, 8),
    ("GQA 32/8, 128k", 1, 1, 131072, 32, 8),
    ("GQA 32/8, batch 8, 32k", 8, 1, 32768, 32, 8),
    ("GQA 64/8, 32k", 1, 1, 32768, 64, 8),
    ("GQA 64/8, 128k", 1, 1, 131072, 64, 8),
    ("MQA 32/1, 32k", 1, 1, 32768, 32, 1),
    ("MQA 32/1, batch 8, 32k", 8, 1, 32768, 32, 1),
    ("GQA 32/8, 4 tokens, 32k", 1, 4, 32768, 32, 8),
    ("GQA 32/8, 8 tokens, 32k", 1, 8, 32768, 32, 8),
    ("GQA 64/8, 8 tokens, 32k", 1, 8, 32768, 64, 8),
    ("MQA 32/1, 8 tokens, 32k", 1, 8, 32768, 32, 1),
]


def main():
    print(f"# GQA decode, {torch.cuda.get_device_name()}, bf16, D=128, causal, us; split sweep {SWEEP}")
    print("| shape | B | Tq | Tk | Hq/Hkv | FA3 auto | FA3 best (splits) | rexp auto | rexp best (splits) | x auto | x best | TB/s FA3 / rexp |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, B, Tq, Tk, Hq, Hkv in SHAPES:
        q = torch.randn(B, Tq, Hq, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, Tk, Hkv, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        qp = prescale_q(q)
        fns = {("fa3", 0): lambda: fwd("stock_pgqa", q, k, v, num_splits=0),
               ("rexp", 0): lambda: fwd("split_fn_rexp_pre_pgqa", qp, k, v, num_splits=0, softmax_scale=LN2),
               ("rexp", "ns"): lambda: fwd("fn_rexp_pre_pgqa", qp, k, v, softmax_scale=LN2)}
        for ns in SWEEP:
            fns[("fa3", ns)] = lambda ns=ns: fwd("stock_pgqa", q, k, v, num_splits=ns)
            fns[("rexp", ns)] = lambda ns=ns: fwd("split_fn_rexp_pre_pgqa", qp, k, v, num_splits=ns, softmax_scale=LN2)
        ms = bench(fns, rounds=9, target_ms=30.0)
        fb = min(SWEEP, key=lambda n: ms[("fa3", n)])
        rb = min(SWEEP + ("ns",), key=lambda n: ms[("rexp", n)])
        gb = 2 * B * Tk * Hkv * 128 * 2 / 1e9
        us = lambda key: ms[key] * 1e3
        rbl = "non-split build" if rb == "ns" else rb
        print(f"| {name} | {B} | {Tq} | {Tk} | {Hq}/{Hkv} | {us(('fa3', 0)):.1f} | {us(('fa3', fb)):.1f} ({fb}) "
              f"| {us(('rexp', 0)):.1f} | {us(('rexp', rb)):.1f} ({rbl}) | **{ms[('fa3', 0)] / ms[('rexp', 0)]:.2f}** "
              f"| **{ms[('fa3', fb)] / ms[('rexp', rb)]:.2f}** | {gb / ms[('fa3', fb)]:.2f} / {gb / ms[('rexp', rb)]:.2f} |", flush=True)
        del q, k, v, qp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
