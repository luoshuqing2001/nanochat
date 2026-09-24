"""Compare direct, atomic, and partial-reduction decode task layouts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from bench_softplus_general import measure
from nanochat.softplus_decode import softplus_decode

ap=argparse.ArgumentParser()
ap.add_argument('--output',required=True)
a=ap.parse_args()
torch.set_num_threads(2)
torch.manual_seed(53)
rows=[]
for d in (64,128):
    for b,h,t in ((1,6,128),(1,6,512),(1,6,4096),(1,6,65536),(8,6,4096)):
        xs=[torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16) for n in (1,t,t)]
        funcs={}
        for reduction in ('atomic','partial'):
            for chunk in (64,256,512):
                funcs[f'{reduction}_{chunk}']=lambda reduction=reduction,chunk=chunk:softplus_decode(*xs,chunk_size=chunk,reduction=reduction)
        r=dict(b=b,h=h,t=t,d=d,results=measure(funcs,30,True))
        rows.append(r)
        Path(a.output).write_text(json.dumps(rows,indent=2))
        print(json.dumps(r),flush=True)
