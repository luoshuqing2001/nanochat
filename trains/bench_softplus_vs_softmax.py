"""Current Softplus vs FA4/SDPA Softmax; attention-only CUDA graph timings.

KV cache append, host dispatch, projections, and optimizer are excluded.
SDPA uses the same helper as nanochat's KV-cache inference (including SWA mask).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.testing
from flash_attn_4.interface import _flash_attn_fwd
from flash_attn_4.fa3_compat import flash_attn_func
from flash_attn_4.softplus_api import softplus_attn_fa4, softplus_attn_fa4_func, auto_bwd_m_chunk
from nanochat.softplus_attention import softplus_attention
from nanochat.flash_attention import _sdpa_attention


def measure(funcs):
    samples = {name: [] for name in funcs}
    for fn in funcs.values():
        out = fn()
        for x in out if isinstance(out, tuple) else (out,):
            assert torch.isfinite(x).all().item()
    torch.cuda.synchronize()
    names = list(funcs)
    for trial in range(3):
        for name in names if trial % 2 == 0 else names[::-1]:
            samples[name].append(triton.testing.do_bench_cudagraph(funcs[name], rep=60))
    return {name: {"ms": statistics.median(xs), "samples_ms": xs}
            for name, xs in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--d12-train", action="store_true", help="Also representative d12 batch sizes, training only")
    ap.add_argument("--small-train-diagnosis", action="store_true")
    ap.add_argument("--tune-small", action="store_true")
    ap.add_argument("--small-regression", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "dtype": "bfloat16", "head_dim": 128,
              "softplus_impl": os.environ.get("NANOCHAT_SOFTPLUS_IMPL", "hybrid"),
              "timing": "median of three CUDA graph replays; rep=60 ms", "rows": []}
    if args.small_regression:
        for b,t,h in ((1,2048,6),(1,4096,1),(1,8192,1),(1,16384,1),(8,2048,12)):
            xs = [torch.randn(b,t,h,128,device="cuda",dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
            do = torch.randn_like(xs[0])
            old_chunk = auto_bwd_m_chunk(b,h,t,t,tile_n=128,device=xs[0].device) or 0
            funcs = {
                "softplus_before": lambda: torch.autograd.grad(softplus_attn_fa4_func(*xs,
                    bwd_m_chunk=old_chunk,num_splits="auto" if h == 1 else 1),xs,do),
                "softplus_after": lambda: torch.autograd.grad(softplus_attention(*xs),xs,do),
                "softmax": lambda: torch.autograd.grad(flash_attn_func(*xs,causal=True),xs,do),
            }
            row = dict(b=b,t=t,h=h,old_chunk=old_chunk,results=measure(funcs))
            result["rows"].append(row)
            print(json.dumps(row),flush=True)
            with open(args.output,"w") as f:
                json.dump(result,f,indent=2)
        return
    if args.tune_small:
        from nanochat.softplus_attention import _bwd, _bwd_fused
        from nanochat.softplus_fixed_kv import fixed_kv_backward_grouped, fixed_kv_forward
        xs = [torch.randn(1,2048,6,128,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
        do = torch.randn_like(xs[0])
        out = softplus_attn_fa4(*xs)
        ref = torch.ops.softplus_attn_fa4.bwd(*xs,out,do,-1,1.,128**-.5,0)
        funcs = {"cute_"+str(c): (lambda c=c: torch.ops.softplus_attn_fa4.bwd(*xs,out,do,-1,1.,128**-.5,c)) for c in (0,2,4,8,12,16,24,32)}
        funcs["triton_recompute"] = lambda: _bwd(*xs,do,-1,1.,128**-.5)
        funcs["triton_fused"] = lambda: _bwd_fused(*xs,do,-1,1.,128**-.5)
        for c in (256,512,1024,2048):
            for bm in (32,64):
                funcs[f"grouped_{c}_{bm}"] = lambda c=c,bm=bm: fixed_kv_backward_grouped(*xs,do,chunk=c,bm=bm)
        for name,fn in funcs.items():
            got = fn()
            err = max(float((x.float()-y.float()).abs().max()/y.float().abs().max()) for x,y in zip(got,ref))
            assert err < .02, (name,err)
            row = dict(name=name,error=err,**measure({name:fn})[name])
            result["rows"].append(row)
            print(json.dumps(row),flush=True)
            with open(args.output,"w") as f:
                json.dump(result,f,indent=2)
        for c in (256,512,1024,2048):
            fn = lambda c=c: fixed_kv_forward(*xs,chunk=c,bm=128)
            row = dict(name=f"forward_{c}",**measure({"f":fn})["f"])
            result["rows"].append(row)
            print(json.dumps(row),flush=True)
        with open(args.output,"w") as f:
            json.dump(result,f,indent=2)
        return
    if args.small_train_diagnosis:
        xs = [torch.randn(1,2048,6,128,device="cuda",dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        do = torch.randn_like(xs[0])
        funcs = {
            "softplus_fwd": lambda: softplus_attn_fa4(*xs, num_splits=1),
            "softmax_fwd": lambda: _flash_attn_fwd(*xs, causal=True)[0],
            "softplus_train_auto": lambda: torch.autograd.grad(softplus_attn_fa4_func(*xs),xs,do),
            "softplus_train_bwd_unsplit": lambda: torch.autograd.grad(softplus_attn_fa4_func(*xs,bwd_m_chunk=0),xs,do),
            "softmax_train": lambda: torch.autograd.grad(flash_attn_func(*xs,causal=True),xs,do),
        }
        result["rows"].append(dict(b=1,t=2048,h=6,results=measure(funcs)))
        with open(args.output,"w") as f:
            json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
        return
    shapes = [(1,4096,1,-1), (1,8192,1,-1), (1,16384,1,-1),
              (1,2048,6,-1), (1,8192,6,-1), (8,2048,12,-1), (8,2048,12,512)]
    if args.d12_train:
        shapes = [(b,2048,6,w) for b in (8,32) for w in (-1,511)]
    for phase in (("train",) if args.d12_train else ("decode", "prefill", "train")):
        cases = ([(1,4096,6,-1), (1,16384,6,-1), (1,65536,6,-1),
                  (1,65536,12,-1), (1,65536,6,512), (8,4096,6,-1)]
                 if phase == "decode" else shapes)
        for b,t,h,w in cases:
            tq = 1 if phase == "decode" else t
            q,k,v = [torch.randn(b,n,h,128,device="cuda",dtype=torch.bfloat16)
                     for n in (tq,t,t)]
            print(f"warming {phase} B={b} Tq={tq} Tk={t} H={h} W={w}", flush=True)
            if phase == "train":
                xs = [x.requires_grad_() for x in (q,k,v)]
                do = torch.randn_like(q)
                funcs = {
                    "softplus": lambda: torch.autograd.grad(softplus_attention(*xs, window_size=(w,0)), xs, do),
                    "softmax_fa4": lambda: torch.autograd.grad(flash_attn_func(*xs, causal=True, window_size=(w,0)), xs, do),
                }
            else:
                funcs = {
                    "softplus": lambda: softplus_attn_fa4(q,k,v,window_size=(w,0),num_splits="auto"),
                    "softmax_fa4": lambda: _flash_attn_fwd(q,k,v,causal=w < 0,
                        window_size_left=None if w < 0 else w,
                        window_size_right=None if w < 0 else 0)[0],
                    "softmax_sdpa": lambda: _sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),(w,0),False),
                }
            row = dict(phase=phase,b=b,tq=tq,tk=t,h=h,window_left=w,results=measure(funcs))
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(args.output,"w") as f:
                json.dump(result,f,indent=2)


if __name__ == "__main__":
    main()
