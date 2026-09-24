"""Work-balanced persistent forward with reduction only at worker boundaries.

Each worker owns a contiguous interval of query-major KV tile iterations. Whole
query tiles write final output directly; split tiles write private FP32 slots.
No output initialization, output atomics, or inter-block waiting is needed.
"""
from bisect import bisect_right
from collections import Counter
from functools import lru_cache
import torch
import triton
import triton.language as tl
from nanochat.softplus_math import softplus_pair
from nanochat.softplus_fixed_kv import _count_scale


@lru_cache(maxsize=128)
def plan_cpu(tq, tk, window, bm, bn, workers):
    """Immutable CPU plan, also usable to verify coverage independently of CUDA."""
    if not (0 < tq <= tk and bm > 0 and bn > 0 and workers > 0):
        raise ValueError('requires 0 < Tq <= Tk and positive tile/worker sizes')
    starts, prefix = [], [0]
    for m in range(triton.cdiv(tq,bm)):
        start=max(0,(m*bm+tk-tq-window)//bn) if window>=0 else 0
        end=triton.cdiv(min(tk,(m+1)*bm+tk-tq),bn)
        starts.append(start)
        prefix.append(prefix[-1]+end-start)
    workers=min(workers,prefix[-1])
    offsets, segments = [0], []
    for worker in range(workers):
        begin,end=worker*prefix[-1]//workers,(worker+1)*prefix[-1]//workers
        while begin<end:
            m=bisect_right(prefix,begin)-1
            stop=min(end,prefix[m+1])
            segments.append((m,starts[m]+begin-prefix[m],starts[m]+stop-prefix[m]))
            begin=stop
        offsets.append(len(segments))
    count=Counter(s[0] for s in segments)
    slots, groups, next_slot = [], [], 0
    for m,lo,hi in segments:
        slot=-1
        if count[m]>1:
            slot=next_slot
            if not groups or groups[-1][0]!=m:
                groups.append((m,slot,count[m]))
            next_slot+=1
        slots.append((m,lo,hi,slot))
    assert len(groups)<=workers-1 and next_slot<=2*(workers-1)
    return tuple(offsets),tuple(slots),tuple(groups),next_slot


@lru_cache(maxsize=128)
def _plan(tq,tk,window,bm,bn,workers,device):
    offsets,segments,groups,nslots=plan_cpu(tq,tk,window,bm,bn,workers)
    return (torch.tensor(offsets,device=device,dtype=torch.int32),
            torch.tensor(segments,device=device,dtype=torch.int32),
            torch.tensor(groups,device=device,dtype=torch.int32),nslots)


@triton.jit
def _forward(Q,K,V,O,P,OFFSETS,SEGMENTS,
             qb,qt,qh,kb,kt,kh,vb,vt,vh,
             TQ:tl.constexpr,TK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
             W:tl.constexpr,ALPHA:tl.constexpr,SCALE:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,NS:tl.constexpr):
    worker,bh=tl.program_id(0),tl.program_id(1)
    b,h=bh//H,bh%H
    begin=tl.load(OFFSETS+worker);end=tl.load(OFFSETS+worker+1)
    d=tl.arange(0,D)
    for seg in range(begin,end):
        mi=tl.load(SEGMENTS+seg*4)
        lo=tl.load(SEGMENTS+seg*4+1);hi=tl.load(SEGMENTS+seg*4+2)
        slot=tl.load(SEGMENTS+seg*4+3)
        m=mi*BM+tl.arange(0,BM)
        q=tl.load(Q+b*qb+m[:,None]*qt+h*qh+d[None,:],m[:,None]<TQ,0)
        acc=tl.zeros((BM,D),tl.float32)
        for ni in range(lo,hi):
            n=ni*BN+tl.arange(0,BN)
            k=tl.load(K+b*kb+n[:,None]*kt+h*kh+d[None,:],n[:,None]<TK,0)
            v=tl.load(V+b*vb+n[:,None]*vt+h*vh+d[None,:],n[:,None]<TK,0)
            s=tl.dot(q,tl.trans(k))*SCALE
            keep=(m[:,None]<TQ)&(n[None,:]<TK)&(n[None,:]<=m[:,None]+TK-TQ)
            if W>=0:
                keep &= n[None,:]>=m[:,None]+TK-TQ-W
            p,_=softplus_pair(s)
            acc=tl.dot(tl.where(keep,p,0.).to(v.dtype),v,acc)
        if slot<0:
            acc *= _count_scale(m,TK,TQ,W,ALPHA)[:,None]
            tl.store(O+((b*TQ+m[:,None])*H+h)*D+d[None,:],acc,m[:,None]<TQ)
        else:
            tl.store(P+((bh*NS+slot)*BM+tl.arange(0,BM)[:,None])*D+d[None,:],acc)


@triton.jit
def _finish(P,O,GROUPS,TQ:tl.constexpr,TK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
            W:tl.constexpr,ALPHA:tl.constexpr,BM:tl.constexpr,NS:tl.constexpr,BLOCK:tl.constexpr):
    group,part,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    mi=tl.load(GROUPS+group*3);first=tl.load(GROUPS+group*3+1);count=tl.load(GROUPS+group*3+2)
    i=part*BLOCK+tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0.,tl.float32)
    for slot in range(first,first+count):
        acc += tl.load(P+(bh*NS+slot)*BM*D+i,i<BM*D,0.)
    m=mi*BM+i//D
    acc *= _count_scale(m,TK,TQ,W,ALPHA)
    tl.store(O+((bh//H*TQ+m)*H+bh%H)*D+i%D,acc,(i<BM*D)&(m<TQ))


def stream_k_forward(q,k,v,window=-1,alpha=1.,scale=None,*,bm=64,bn=64,
                     workers=None,waves=4,num_warps=4,num_stages=2,return_kernel=False):
    b,tq,h,d=q.shape;tk=k.shape[1]
    if (b <= 0 or h <= 0 or not 0<tq<=tk or d not in (64,128) or k.shape!=(b,tk,h,d) or v.shape!=k.shape
            or q.dtype not in (torch.bfloat16,torch.float16)
            or any(not x.is_cuda or x.device!=q.device or x.dtype!=q.dtype or x.stride(-1)!=1 for x in (q,k,v))):
        raise ValueError('stream-K requires CUDA FP16/BF16 equal heads, D64/128, contiguous features, 0<Tq<=Tk')
    if bm not in (32,64,128) or bn not in (32,64,128) or waves<=0:
        raise ValueError('invalid tile sizes or waves')
    if workers is None:
        workers=max(1,triton.cdiv(waves*torch.cuda.get_device_properties(q.device).multi_processor_count,b*h))
    offsets,segments,groups,nslots=_plan(tq,tk,window,bm,bn,workers,str(q.device))
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    partial=torch.empty((b*h,nslots,bm,d),device=q.device,dtype=torch.float32)
    kernel=_forward[(offsets.numel()-1,b*h)](q,k,v,out,partial,offsets,segments,
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],tq,tk,h,d,window,alpha,
        d**-.5 if scale is None else scale,bm,bn,nslots,num_warps=num_warps,num_stages=num_stages)
    if nslots:
        _finish[(groups.shape[0],triton.cdiv(bm*d,256),b*h)](
            partial,out,groups,tq,tk,h,d,window,alpha,bm,nslots,256)
    return (out,kernel) if return_kernel else out
