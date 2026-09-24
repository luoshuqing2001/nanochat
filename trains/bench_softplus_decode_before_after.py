"""Paired decode ablation to distinguish source changes from between-run drift."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nanochat.softplus_decode import softplus_decode
from nanochat.flash_attention import _sdpa_attention
from trains.bench_softplus_general import measure

p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('decode_before_general',a.baseline)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(2)
source=Path(a.baseline).read_text()
result=dict(baseline_source=source,baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=[])
for d in (64,128):
    for t in (128,4096,16384,65536):
        torch.manual_seed(123)
        q,k,v=[torch.randn(1,n,6,d,device='cuda',dtype=torch.bfloat16) for n in (1,t,t)]
        f=dict(before=lambda:m.softplus_decode(q,k,v),after=lambda:softplus_decode(q,k,v),
            sdpa=lambda:_sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),(-1,0),False))
        row=dict(d=d,t=t,results=measure(f,30,False))
        result['rows'].append(row)
        Path(a.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
