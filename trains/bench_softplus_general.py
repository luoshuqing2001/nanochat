"""Cross-shape attention evaluation, with matched FA4 and SDPA Softmax baselines.

Run before/after with separate output files on an idle GPU. Each case uses the
same random QKV and gradient across implementations. Attention functions differ
mathematically; parity with Softmax is not an accuracy test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton.testing
from flash_attn_4.interface import _flash_attn_fwd, _flash_attn_bwd
from flash_attn_4.fa3_compat import flash_attn_func
from flash_attn_4.softplus_api import softplus_attn_fa4, softplus_attn_fa4_func
from nanochat.softplus_attention import softplus_attention
from nanochat.flash_attention import _sdpa_attention


class TunedSoftmax(torch.autograd.Function):
    """Control: give Softmax the same SM120 tile/pipeline improvements."""
    @staticmethod
    def forward(ctx,q,k,v,w):
        o,lse,*_=_flash_attn_fwd(q,k,v,causal=w<0,window_size_left=None if w<0 else w,
            window_size_right=None if w<0 else 0,tile_mn=(64,128 if q.shape[-1]<=64 else 64),return_lse=True)
        ctx.save_for_backward(q,k,v,o,lse)
        ctx.w=w
        return o
    @staticmethod
    def backward(ctx,g):
        q,k,v,o,lse=ctx.saved_tensors
        dq,dk,dv,*_=_flash_attn_bwd(q,k,v,o,g.contiguous(),lse,causal=ctx.w<0,
            window_size_left=None if ctx.w<0 else ctx.w,window_size_right=None if ctx.w<0 else 0,
            sm120_bwd_tile=(64,64,2,1))
        return dq,dk,dv,None


def cases(holdout=False):
    dense = []
    for d in (64,128):
        for t in (256,2048,8192):
            for w in ((-1,) if t == 256 else (-1,511)):
                dense.append((1,6,t,d,w,"bf16",False))
        dense += [(8,6,2048,d,w,"bf16",False) for w in (-1,511)]
    dense += [(1,1,t,128,-1,"bf16",False) for t in (4096,16384)]
    dense += [(2,4,1536,128,w,"bf16",True) for w in (-1,255)]
    dense += [(1,12,4096,64,-1,"bf16",True),(1,6,2048,128,-1,"fp16",False)]
    out = []
    for phase in ("prefill","train"):
        for b,h,t,d,w,dtype,packed in dense:
            out.append(dict(phase=phase,b=b,h=h,tq=t,tk=t,d=d,w=w,dtype=dtype,packed=packed))
    for d in (64,128):
        for t,w in ((128,-1),(4096,-1),(16384,-1),(65536,-1),(65536,511)):
            out.append(dict(phase="decode",b=1,h=6,tq=1,tk=t,d=d,w=w,dtype="bf16",packed=False))
        for b,h in ((8,6),(1,1)):
            out.append(dict(phase="decode",b=b,h=h,tq=1,tk=4096,d=d,w=-1,dtype="bf16",packed=False))
    out.append(dict(phase="decode",b=2,h=4,tq=1,tk=2053,d=64,w=111,dtype="fp16",packed=True))
    for tq,tk,d,w in ((113,2053,128,-1),(113,2053,128,511),(512,4096,64,-1)):
        out.append(dict(phase="prefill",b=1,h=6,tq=tq,tk=tk,d=d,w=w,dtype="bf16",packed=True))
    if holdout:
        out=[]
        for phase in ('prefill','train'):
            for b,h,t,d,w in ((1,2,1024,128,-1),(2,8,3072,128,-1),(2,8,3072,128,1023),
                               (4,3,1024,64,127),(1,4,12288,128,-1),(8,12,2048,128,511)):
                out.append(dict(phase=phase,b=b,h=h,tq=t,tk=t,d=d,w=w,dtype='bf16',packed=True))
        for b,h,t,d,w in ((4,8,8193,128,-1),(4,8,8193,128,511),(1,2,32769,64,-1),(2,4,511,128,31)):
            out.append(dict(phase='decode',b=b,h=h,tq=1,tk=t,d=d,w=w,dtype='bf16',packed=True))
    if holdout == 'schedule':
        out=[]
        for phase in ('prefill','train'):
            for b,h,t,d,w in ((1,3,1792,64,-1),(2,5,3584,64,-1),(1,5,6144,128,-1),
                               (2,7,2560,128,-1),(1,3,1792,128,383),(2,5,768,64,95)):
                out.append(dict(phase=phase,b=b,h=h,tq=t,tk=t,d=d,w=w,dtype='bf16',packed=True))
        for b,h,t,d,w in ((2,3,10001,128,-1),(2,3,10001,128,383),(1,5,24577,64,-1),(3,2,769,64,95)):
            out.append(dict(phase='decode',b=b,h=h,tq=1,tk=t,d=d,w=w,dtype='bf16',packed=True))
    for c in out:
        c["id"] = f"{c['phase']}_b{c['b']}h{c['h']}_q{c['tq']}k{c['tk']}_d{c['d']}_w{c['w']}_{c['dtype']}" + ("_strided" if c['packed'] else "")
    return out


def measure(funcs,rep,eager):
    samples = {name:[] for name in funcs}
    for fn in funcs.values():
        got = fn()
        for x in got if isinstance(got,tuple) else (got,):
            assert torch.isfinite(x).all().item()
    torch.cuda.synchronize()
    for trial in range(3):
        names = list(funcs) if trial % 2 == 0 else list(funcs)[::-1]
        for name in names:
            samples[name].append(triton.testing.do_bench_cudagraph(funcs[name],rep=rep))
    result = {name:dict(ms=statistics.median(xs),samples_ms=xs) for name,xs in samples.items()}
    if eager:
        for name,fn in funcs.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            result[name]["eager_ms"] = (time.perf_counter()-start)*100
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--phase",choices=("prefill","decode","train"))
    ap.add_argument("--only",help="Substring of case id")
    ap.add_argument("--rep",type=int,default=30)
    ap.add_argument("--eager",action="store_true")
    ap.add_argument("--schedule-holdout",action="store_true",help="New shapes excluded from schedule tuning")
    ap.add_argument("--native-control",action="store_true",help="Paired previous backward dispatch and CuTe Stream-K")
    ap.add_argument("--schedules",action="store_true",help="Also evaluate explicit schedule candidates")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--holdout",action="store_true",help="Shapes excluded from kernel tuning")
    ap.add_argument("--skip",action="append",default=[],help="Known failing case substring; recorded without a timing")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    root = Path(__file__).resolve().parents[1]
    paths = ("nanochat/softplus_decode.py","nanochat/softplus_fixed_kv.py",
             "nanochat/softplus_attention.py","flash_attn_4/softplus_api.py",
             "flash_attn_4/softplus.py","flash_attn_4/flash_bwd_softplus.py",
             "flash_attn_4/interface.py","nanochat/softplus_math.py",
             "flash_attn_4/flash_fwd.py","flash_attn_4/flash_fwd_softplus.py",
             "flash_attn_4/flash_bwd.py","nanochat/softplus_stream_k.py","flash_attn_4/softplus_stream.py")
    data = dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        environment={k:v for k,v in os.environ.items() if k.startswith(("FA4_SOFTPLUS", "NANOCHAT_SOFTPLUS"))},
        timing=f"median of 3 CUDA graph measurements, rep={args.rep} ms; zeroing/casts included",
        exclusions="KV append, projections, full model, optimizer, cold compilation",
        hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},rows=[])
    if args.resume and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text())
        assert previous["hashes"] == data["hashes"], "Cannot resume across source changes"
        data = previous
    done = {row['id'] for row in data['rows']}
    for c in cases('schedule' if args.schedule_holdout else args.holdout):
        if c['id'] in done or (args.phase and c['phase'] != args.phase) or (args.only and args.only not in c['id']):
            continue
        if any(s in c['id'] for s in args.skip):
            data['rows'].append(dict(c,error="Skipped known illegal memory access in legacy strided hybrid backward"))
            Path(args.output).write_text(json.dumps(data,indent=2))
            continue
        torch.manual_seed(123)
        b,h,tq,tk,d,w = (c[k] for k in ("b","h","tq","tk","d","w"))
        dtype = torch.bfloat16 if c['dtype'] == "bf16" else torch.float16
        xs = []
        for i,t in enumerate((tq,tk,tk)):
            x = (torch.randn(b,t,3,h,d,device="cuda",dtype=dtype)[:,:,i] if c['packed']
                 else torch.randn(b,t,h,d,device="cuda",dtype=dtype))
            xs.append(x.requires_grad_(c['phase'] == "train"))
        q,k,v = xs
        def sdpa():
            return _sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),(w,0),False).transpose(1,2)
        if c['phase'] == "train":
            do = torch.randn_like(q)
            funcs = {
                "softplus":lambda:torch.autograd.grad(softplus_attention(*xs,window_size=(w,0)),xs,do),
                "softmax_fa4":lambda:torch.autograd.grad(flash_attn_func(*xs,causal=True,window_size=(w,0)),xs,do),
                "softmax_sdpa":lambda:torch.autograd.grad(sdpa(),xs,do),
                "softmax_fa4_tuned":lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,w),xs,do),
            }
        else:
            funcs = {
                "softplus":lambda:softplus_attn_fa4(*xs,window_size=(w,0),num_splits="auto"),
                "softmax_fa4":lambda:_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0)[0],
                "softmax_sdpa":sdpa,
                "softmax_fa4_tuned":lambda:_flash_attn_fwd(*xs,causal=w<0,
                    window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                    tile_mn=(64,128 if d<=64 else 64))[0],
            }
        if args.native_control and c['phase'] != 'decode':
            if c['phase'] == 'train':
                from nanochat.softplus_attention import _SOFTPLUS_IMPL, _SOFTPLUS_SPLITS
                hybrid=int(_SOFTPLUS_IMPL == "hybrid" and 0 <= w < tk and d > 64
                           and b*h >= 2*torch.cuda.get_device_properties(q.device).multi_processor_count)
                funcs['softplus_previous'] = lambda:torch.autograd.grad(
                    softplus_attn_fa4_func(*xs,window_size=(w,0),bwd_schedule="previous",
                                          num_splits=_SOFTPLUS_SPLITS,bwd_impl=hybrid),xs,do)
            else:
                funcs['softplus_cute_stream'] = lambda:softplus_attn_fa4(
                    *xs,window_size=(w,0),cute_stream_waves=2)
        if args.schedules and c['phase'] != 'decode':
            if c['phase'] == 'train':
                funcs['softplus_previous'] = lambda:torch.autograd.grad(
                    softplus_attn_fa4_func(*xs,window_size=(w,0),early_dv=False),xs,do)
                funcs['softplus_early_dv'] = lambda:torch.autograd.grad(
                    softplus_attn_fa4_func(*xs,window_size=(w,0),early_dv=True),xs,do)
                funcs['softplus_stream_early'] = lambda:torch.autograd.grad(
                    softplus_attn_fa4_func(*xs,window_size=(w,0),stream_waves=2,early_dv=True),xs,do)
            else:
                funcs['softplus_stream'] = lambda:softplus_attn_fa4(*xs,window_size=(w,0),stream_waves=2)
                funcs['softplus_fragment'] = lambda:softplus_attn_fa4(*xs,window_size=(w,0),fragment_n=32)
        print("warming "+c['id'],flush=True)
        row = dict(c,results=measure(funcs,args.rep,args.eager))
        data['rows'].append(row)
        Path(args.output).write_text(json.dumps(data,indent=2))
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
