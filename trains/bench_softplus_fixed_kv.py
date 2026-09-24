"""Fixed-KV prefill/training: correctness and whole-call timings on an idle GPU."""
import argparse
import json
import os
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.testing
from nanochat.softplus_fixed_kv import fixed_kv_forward, fixed_kv_backward, fixed_kv_backward_grouped
from nanochat.softplus_attention import _bwd as triton_backward
from flash_attn_4.softplus_api import (softplus_attn_fa4, softplus_attn_fa4_func,
                                      auto_bwd_m_chunk, auto_balanced_chunk)


def ref(q, k, v, w, alpha):
    tq, tk = q.shape[1], k.shape[1]
    m = torch.arange(tq, device=q.device)[:, None] + tk - tq
    n = torch.arange(tk, device=q.device)[None, :]
    keep = n <= m
    if w >= 0:
        keep &= n >= m - w
    s = torch.einsum("bmhd,bnhd->bhmn", q, k) * q.shape[-1] ** -.5
    p = torch.nn.functional.softplus(s) * keep
    return torch.einsum("bhmn,bnhd->bmhd", p, v) * keep.sum(-1).float().pow(-alpha)[None, :, None, None]


def error(x, y):
    return float(((x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-8)).detach())


def correctness():
    for tq, tk, w, a in [(257, 257, -1, 1.), (300, 300, 111, .5), (113, 511, 127, 1.)]:
        base = [torch.randn(2, t, 2, 64, device="cuda", dtype=torch.bfloat16) for t in (tq, tk, tk)]
        do = torch.randn_like(base[0])
        rbase = [x.float().requires_grad_() for x in base]
        r = ref(*rbase, w, a)
        rg = torch.autograd.grad(r, rbase, do.float())
        for chunk in (64, 256, 512):
            out = fixed_kv_forward(*base, w, a, chunk=chunk)
            grads = fixed_kv_backward(*base, do, w, a, chunk=chunk)
            es = [error(x, y) for x, y in zip((out, *grads), (r, *rg))]
            print("check", tq, tk, w, a, chunk, es, flush=True)
            assert max(es) < .02, es
            grads = fixed_kv_backward_grouped(*base, do, w, a, chunk=chunk)
            es = [error(x, y) for x, y in zip(grads, rg)]
            assert max(es) < .02, es


def timing(fn):
    fn()
    return statistics.median(triton.testing.do_bench_cudagraph(fn, rep=40) for _ in range(3))


def integrated():
    rows = []
    for b, t, h, w in ((1,4096,1,-1), (1,8192,1,-1), (1,16384,1,-1),
                        (1,2048,6,-1), (1,8192,6,-1), (8,2048,12,-1), (8,2048,12,512)):
        xs = [torch.randn(b,t,h,128,device="cuda",dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        do = torch.randn_like(xs[0])
        bc = auto_balanced_chunk(b,h,t,t,window_left=None if w < 0 else w,device=xs[0].device)
        funcs = {
            "prefill_before": lambda: softplus_attn_fa4(*xs, window_size=(w,0), balanced_chunk=bc),
            "prefill_auto": lambda: softplus_attn_fa4(*xs, window_size=(w,0), num_splits="auto"),
            "train_before": lambda: torch.autograd.grad(softplus_attn_fa4_func(
                *xs, window_size=(w,0), num_splits=1, bwd_impl=int(w >= 0)), xs, do),
            "train_auto": lambda: torch.autograd.grad(softplus_attn_fa4_func(
                *xs, window_size=(w,0), num_splits="auto", bwd_impl=int(w >= 0)), xs, do),
        }
        outputs = {name: fn() for name, fn in funcs.items()}
        assert error(outputs["prefill_auto"], outputs["prefill_before"]) < .02
        assert max(error(x,y) for x,y in zip(outputs["train_auto"],outputs["train_before"])) < .02
        for name, fn in funcs.items():
            row = dict(b=b,t=t,h=h,w=w,impl=name,ms=timing(fn))
            rows.append(row)
            print(json.dumps(row),flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--integrated", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    if args.integrated:
        rows = integrated()
        if args.output:
            with open(args.output, "w") as f:
                json.dump(rows,f,indent=2)
        return
    correctness()
    shapes = [(1, 4096, 1, -1), (1, 8192, 1, -1), (8, 2048, 12, -1),
              (8, 2048, 12, 512), (1, 8192, 12, -1)]
    if args.quick:
        shapes = [shapes[0], shapes[2], shapes[3]]
    if args.long:
        shapes = [(1, t, 1, -1) for t in (4096, 8192, 16384)]
    rows = []
    for b, t, h, w in shapes:
        q, k, v = [torch.randn(b, t, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        do = torch.randn_like(q)
        bc = auto_balanced_chunk(b,h,t,t,window_left=None if w < 0 else w,device=q.device)
        base_f = lambda: softplus_attn_fa4(q, k, v, window_size=(w, 0), balanced_chunk=bc)
        o = base_f()
        if w >= 0:
            base_b = lambda: triton_backward(q, k, v, do, w, 1., 128 ** -.5)
        else:
            mc = auto_bwd_m_chunk(b, h, t, t, device=q.device) or 0
            base_b = lambda: torch.ops.softplus_attn_fa4.bwd(q, k, v, o, do, w, 1., 128 ** -.5, mc)
        refg = base_b()
        row = dict(b=b, t=t, h=h, w=w, impl="baseline", fwd_ms=timing(base_f), bwd_ms=timing(base_b))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.long:
            for chunk in (512, 1024, 2048, 4096):
                for name, f in (
                    ("cute_fixed", lambda: softplus_attn_fa4(q, k, v, window_size=(w, 0), balanced_chunk=chunk // 64)),
                    ("triton_fixed", lambda: fixed_kv_forward(q, k, v, w, chunk=chunk, bm=128)),
                ):
                    e = error(f(), o)
                    assert e < .02
                    row = dict(b=b, t=t, h=h, w=w, impl=name, chunk=chunk, error=e, fwd_ms=timing(f))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            continue
        if args.grouped:
            for chunk in (256, 512, 1024, 2048, 4096):
                f = lambda: softplus_attn_fa4(q, k, v, window_size=(w, 0), balanced_chunk=chunk // 64)
                es = [error(f(), o)]
                row = dict(b=b, t=t, h=h, w=w, impl="cute_fixed", chunk=chunk,
                           errors=es, fwd_ms=timing(f))
                rows.append(row)
                print(json.dumps(row), flush=True)
                for bn in (64, 128):
                    g = lambda: fixed_kv_backward_grouped(q, k, v, do, w, chunk=chunk, bn=bn)
                    try:
                        es = [error(x, y) for x, y in zip(g(), refg)]
                        assert max(es) < .02, es
                        row = dict(b=b, t=t, h=h, w=w, impl="grouped", chunk=chunk, bn=bn,
                                   errors=es, bwd_ms=timing(g))
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    except triton.OutOfResources as e:
                        print("resource limit", bn, str(e), flush=True)
            continue
        for chunk in (64, 256, 1024, 4096):
            for bm in (64, 128):
                f = lambda: fixed_kv_forward(q, k, v, w, chunk=chunk, bm=bm)
                g = lambda: fixed_kv_backward(q, k, v, do, w, chunk=chunk)
                es = [error(f(), o)] + [error(x, y) for x, y in zip(g(), refg)]
                assert max(es) < .02, es
                row = dict(b=b, t=t, h=h, w=w, impl="fixed", chunk=chunk, bm=bm,
                           errors=es, fwd_ms=timing(f), bwd_ms=timing(g))
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
