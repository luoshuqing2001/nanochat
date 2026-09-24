"""Kernel-family tuning across shapes, not an exact-shape dispatch table."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import triton
from bench_softplus_general import measure
from flash_attn_4.interface import _flash_attn_fwd, _flash_attn_bwd
from flash_attn_4.softplus_api import softplus_attn_fa4
from nanochat.softplus_fixed_kv import fixed_kv_forward
from nanochat.softplus_attention import _bwd_kv_kernel,_bwd_q_kernel


def triton_bwd(q,k,v,g,w,bm,bn,warps,stages):
    b,t,h,d=q.shape
    dq,dk,dv=[torch.empty(x.shape,device=x.device,dtype=x.dtype) for x in (q,k,v)]
    args=(*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],*g.stride()[:3],t,d**-.5,1.)
    kw=dict(H=h,WINDOW=w,HEAD_DIM=d,BLOCK_M=bm,BLOCK_N=bn,num_warps=warps,num_stages=stages)
    _bwd_kv_kernel[(triton.cdiv(t,bn),b,h)](q,k,v,g,dk,dv,*args,REVERSE=False,**kw)
    _bwd_q_kernel[(triton.cdiv(t,bm),b,h)](q,k,v,g,dq,*args,REVERSE=True,**kw)
    return dq,dk,dv


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--phase',choices=('fwd','bwd','cute_bwd'),required=True)
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(71)
    rows=[]
    shapes=[(1,6,t,d,w) for d in (64,128) for t,w in ((256,-1),(2048,-1),(2048,511),(8192,-1))]
    if not args.quick:
        shapes += [(8,6,2048,d,w) for d in (64,128) for w in (-1,511)]
        shapes += [(1,1,4096,128,-1),(1,1,16384,128,-1),(2,4,1536,128,255)]
    for b,h,t,d,w in shapes:
        xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        o=softplus_attn_fa4(*xs,window_size=(w,0))
        funcs={}
        if args.phase=='fwd':
            ref=(o,)
            for bm,bn in ((64,64),(64,128),(128,64),(128,128)):
                funcs[f'cute_{bm}_{bn}']=lambda bm=bm,bn=bn: _flash_attn_fwd(*xs,causal=w<0,
                    window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                    attn_kind='softplus',tile_mn=(bm,bn))[0]
            for bm in (64,128):
                for c in sorted(set((256,512,1024,t))):
                    funcs[f'triton_{bm}_{c}']=lambda bm=bm,c=c: fixed_kv_forward(*xs,window=w,chunk=c,bm=bm)
        else:
            g=torch.randn_like(xs[0])
            funcs['cute']=lambda:torch.ops.softplus_attn_fa4.bwd(*xs,o,g,w,1.,d**-.5,0)
            ref=funcs['cute']()
            if args.phase=='cute_bwd':
                lse=torch.empty(b,h,t,device='cuda',dtype=torch.float32)
                for tile in ((64,64,2,1),(64,64,1,2),(32,64,2,2),(64,32,2,2),(32,128,1,1)):
                    funcs['cute_'+'_'.join(map(str,tile))]=lambda tile=tile: _flash_attn_bwd(*xs,o,g,lse,causal=w<0,
                        window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                        attn_kind='softplus',sm120_bwd_tile=tile)[:3]
            else:
                for bm,bn in ((32,64),(64,32),(64,64),(64,128),(128,64)):
                    for warps,stages in ((4,1),(4,2),(8,1),(8,2)):
                        funcs[f'triton_{bm}_{bn}_{warps}_{stages}']=lambda bm=bm,bn=bn,warps=warps,stages=stages:triton_bwd(*xs,g,w,bm,bn,warps,stages)
        for name,fn in funcs.items():
            print('warming',args.phase,b,h,t,d,w,name,flush=True)
            try:
                got=fn()
                got=got if isinstance(got,tuple) else (got,)
                errors=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-8)) for x,y in zip(got,ref)]
                assert max(errors)<.02,errors
                row=dict(b=b,h=h,t=t,d=d,w=w,name=name,errors=errors,**measure({'f':fn},20,False)['f'])
            except (triton.OutOfResources,AssertionError) as e:
                row=dict(b=b,h=h,t=t,d=d,w=w,name=name,error=str(e))
            rows.append(row)
            Path(args.output).write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
