import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
torch.set_num_threads(2);torch.manual_seed(812)
xs=[torch.randn(1,128,1,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
g=torch.randn_like(xs[0])
for kind in ('softplus','softmax'):
 o,lse,*_=_flash_attn_fwd(*xs,causal=True,attn_kind=kind,return_lse=True)
 for early in ((False,True) if kind=='softplus' else (False,)):
  ref=None
  for n in (64,32,16):
   out=_flash_attn_bwd(*xs,o,g,lse,causal=True,attn_kind=kind,softplus_early_dv=early,sm120_bwd_tile=(64,n,2,1))[:3]
   if ref is None:ref=out
   print(kind,early,n,[(x.float()-y.float()).abs().max().item() for x,y in zip(out,ref)],flush=True)
   if n==32:print('dv',out[2][0,:8,0,0].tolist(),ref[2][0,:8,0,0].tolist(),flush=True)
