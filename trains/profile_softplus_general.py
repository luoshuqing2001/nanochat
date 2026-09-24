"""A warmed single call for Nsight Compute; profiling starts after compilation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4
from nanochat.softplus_attention import softplus_attention
from nanochat.flash_attention import _sdpa_attention

ap=argparse.ArgumentParser()
ap.add_argument('--phase',choices=('decode','train'),default='decode')
ap.add_argument('--backend',choices=('softplus','sdpa'),default='softplus')
a=ap.parse_args()
torch.set_num_threads(2)
torch.manual_seed(1)
t=65536 if a.phase=='decode' else 8192
q,k,v=[torch.randn(1,n,6,128,device='cuda',dtype=torch.bfloat16).requires_grad_(a.phase=='train')
       for n in (1 if a.phase=='decode' else t,t,t)]
g=torch.randn_like(q)
def fn():
    if a.backend=='softplus':
        if a.phase=='decode':
            return softplus_attn_fa4(q,k,v,num_splits='auto')
        o=softplus_attention(q,k,v)
    else:
        o=_sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),(-1,0),False).transpose(1,2)
    return torch.autograd.grad(o,(q,k,v),g) if a.phase=='train' else o
for _ in range(3):fn()
torch.cuda.synchronize()
torch.cuda.profiler.start()
fn()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
