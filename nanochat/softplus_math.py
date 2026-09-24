"""Shared stable score map and fused backward workspace initialization."""
import os
import triton
import triton.language as tl

_EXACT = tl.constexpr(os.environ.get("FA4_SOFTPLUS_EXACT", "0") == "1")


@triton.jit
def softplus_pair(x):
    y = tl.exp2(-tl.abs(x) * 1.4426950408889634)
    if _EXACT:
        sp = tl.maximum(x, 0.) + tl.log2(1. + y) * .6931471805599453
    else:
        p = .10028652 + y * -.0236890026
        p = -.208668975 + y * p
        p = .324411505 + y * p
        p = -.499187794 + y * p
        p = .999981869 + y * p
        sp = tl.maximum(x, 0.) + y * p
    inv = 1. / (1. + y)
    sig = tl.where(x >= 0., inv, y * inv)
    return sp, sig


@triton.jit
def _prepare(DQ, LSE, DP, NQ:tl.constexpr, NR:tl.constexpr, R:tl.constexpr,
             TQ:tl.constexpr, TK:tl.constexpr, W:tl.constexpr, CAUSAL:tl.constexpr,
             ALPHA:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0,BLOCK)
    tl.store(DQ+i,0.,i<NQ)
    if tl.program_id(0)*BLOCK < NR:
        m = i % R
        count = tl.minimum(m + TK - TQ + 1,TK) if CAUSAL else tl.full((BLOCK,),TK,tl.int32)
        if W >= 0:
            count = tl.minimum(count,W+1)
        n = tl.maximum(count,1).to(tl.float32)
        if ALPHA == 1.:
            c = 1./n
        elif ALPHA == 0.:
            c = tl.full((BLOCK,),1.,tl.float32)
        else:
            c = tl.exp2(tl.log2(n)*-ALPHA)
        tl.store(LSE+i,c,i<NR)
        tl.store(DP+i,c,i<NR)


def prepare_backward(dq_accum,lse,dpsum,tq,tk,causal,window,alpha):
    nq = 0 if dq_accum is None else dq_accum.numel()
    _prepare[(triton.cdiv(max(nq,lse.numel()),1024),)](
        lse if dq_accum is None else dq_accum,lse,dpsum,nq,lse.numel(),lse.shape[-1],
        tq,tk,-1 if window is None else window,causal,alpha,1024)


@triton.jit
def _prepare_scaled_do(DQ, G, GS, NQ:tl.constexpr, NG:tl.constexpr,
                       TQ:tl.constexpr,TK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                       gb:tl.constexpr,gt:tl.constexpr,gh:tl.constexpr,gd:tl.constexpr,
                       W:tl.constexpr,CAUSAL:tl.constexpr,ALPHA:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(DQ+i,0.,i<NQ)
    if tl.program_id(0)*BLOCK<NG:
        d=i%D;h=i//D%H;m=i//(D*H)%TQ;b=i//(D*H*TQ)
        count=tl.minimum(m+TK-TQ+1,TK) if CAUSAL else tl.full((BLOCK,),TK,tl.int32)
        if W>=0:count=tl.minimum(count,W+1)
        n=tl.maximum(count,1).to(tl.float32)
        if ALPHA==1.:c=1./n
        elif ALPHA==0.:c=tl.full((BLOCK,),1.,tl.float32)
        else:c=tl.exp2(tl.log2(n)*-ALPHA)
        g=tl.load(G+b*gb+m*gt+h*gh+d*gd,i<NG,0.).to(tl.float32)
        tl.store(GS+i,g*c,i<NG)


def prepare_scaled_backward(dq_accum,do,tk,causal,window,alpha):
    """Fuse existing dQ zeroing with row scaling of dO; no row-statistic buffers.

    Scaling before input-dtype conversion changes rounding versus scaling P/dS.
    In particular FP16 very small gradients may underflow; this is opt-in.
    """
    import torch
    scaled=torch.empty(do.shape,device=do.device,dtype=do.dtype)
    b,t,h,d=do.shape
    _prepare_scaled_do[(triton.cdiv(max(dq_accum.numel(),do.numel()),1024),)](
        dq_accum,do,scaled,dq_accum.numel(),do.numel(),t,tk,h,d,*do.stride(),
        -1 if window is None else window,causal,alpha,1024)
    return scaled
