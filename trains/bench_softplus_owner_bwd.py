"""Backward-only control: fixed inputs, layouts, tile and unsplit KV ownership."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,_op_bwd_fa4
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    torch.set_num_threads(2);rows=[]
    for t,h in ((4096,1),(32768,1),(32768,6),(65536,6)):
      for d in (64,128):
        torch.manual_seed(483);xs=[torch.randn(1,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)];g=torch.randn_like(xs[0]);o=softplus_attn_fa4(*xs);so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True,tile_mn=(64,128 if d==64 else 64))
        funcs={}
        for name,mode in [('original',1 if d==64 else 0),('paired',6 if d==64 else 5)]:
            funcs[name]=lambda mode=mode:_op_bwd_fa4(*xs,o,g,-1,1.,d**-.5,0,d==128,mode)
        for name,pair in [('softmax_original',False),('softmax_paired',True)]:
            funcs[name]=lambda pair=pair:_flash_attn_bwd(*xs,so,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1),sm120_owner_pair=pair)[:3]
        errors={}
        for old,new in [('original','paired'),('softmax_original','softmax_paired')]:
            aa=funcs[old]();bb=funcs[new]();errors[new]=[float((x.float()-y.float()).abs().max()/x.float().abs().max().clamp_min(1e-20)) for x,y in zip(aa,bb)];assert max(errors[new])<.025
            del aa,bb
        samples={n:[] for n in funcs};names=list(funcs)
        for trial in range(3):
            for n in (names if trial%2==0 else names[::-1]):samples[n].append(timed(funcs[n]))
        row=dict(t=t,h=h,d=d,errors=errors,results={n:dict(ms=statistics.median(v),samples_ms=v) for n,v in samples.items()});rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
