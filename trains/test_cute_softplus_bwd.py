"""Gradients of the CuTe softplus backward, against float32 autograd and the Triton kernel.

The CuTe backward runs no preprocess pass and reads no LSE: softplus needs neither. It
gets n^-alpha through the LSE and dPsum buffers instead. See flash_bwd_softplus.py.
"""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn_4.softplus_api import softplus_attn_fa4_func
from nanochat.softplus_attention import softplus_attn_func

def ref(q,k,v,W,alpha,sc):
    qf,kf,vf = (x.transpose(1,2).float() for x in (q,k,v))
    T = qf.shape[-2]; i = torch.arange(T, device=q.device)
    s = (qf @ kf.transpose(-1,-2)) * sc
    keep = i[:,None] >= i[None,:]
    if W is not None: keep &= i[:,None]-i[None,:] <= W
    p = F.softplus(s) * keep
    n = keep.sum(-1).clamp(min=1).float()
    return ((p @ vf) / n[None,None,:,None]**alpha).transpose(1,2)

bad = 0
for B,T,H,D,W,alpha in [(2,512,4,64,None,1.0),(2,1024,6,128,None,1.0),
                        (2,1024,4,128,255,1.0),(2,2048,6,128,511,1.0),
                        (2,1024,6,128,None,0.5)]:
    torch.manual_seed(0)
    base = [torch.randn(B,T,H,D,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    g = torch.randn(B,T,H,D,device="cuda",dtype=torch.bfloat16)
    sc = 1/math.sqrt(D)
    outs = {}
    for name, fn in [("cute", lambda a,b,c: softplus_attn_fa4_func(a,b,c,True,(W,0),alpha,sc)),
                     ("triton", lambda a,b,c: softplus_attn_func(a,b,c,True,(W if W is not None else -1,0),alpha,sc)),
                     ("ref", None)]:
        q,k,v = [t.clone().requires_grad_(True) for t in base]
        if name == "ref":
            o = ref(q,k,v,W,alpha,sc).to(torch.bfloat16)
        else:
            o = fn(q,k,v)
        o.backward(g)
        outs[name] = (o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float())
    tag = f"B{B} T{T} H{H} D{D} W={W} a={alpha}"
    msg = []
    for i, nm in enumerate(["out","dq","dk","dv"]):
        r = outs["ref"][i]; den = max(r.abs().max().item(), 1e-9)
        ec = (outs["cute"][i]-r).abs().max().item()/den
        et = (outs["triton"][i]-r).abs().max().item()/den
        msg.append(f"{nm}: cute={ec:.1e} tri={et:.1e}")
        if ec > max(3*et, 3e-2): bad += 1
    print(f"{tag:<34} " + "  ".join(msg))
print("SUSPECT:", bad)
