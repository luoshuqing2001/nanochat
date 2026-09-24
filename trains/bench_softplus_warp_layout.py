"""Paired full forward+backward comparison against the previous 8-warp kernel."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
torch.set_num_threads(2);rows=[]
shapes=[(1,2048,6),(8,2048,12),(1,8192,6),(1,32768,6),(1,65536,6)]
if a.reverse:shapes=shapes[::-1]
for b,t,h in shapes:
    torch.manual_seed(176)
    xs=[torch.randn(b,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(xs[0])
    def forward(kind):
        return _flash_attn_fwd(*xs,causal=True,attn_kind=kind,return_lse=kind=='softmax',tile_mn=(64,64),sm120_head_lpt=True)
    def run(kind,layout):
        out,lse,*_=forward(kind)
        if lse is None:lse=torch.empty(b,h,t,device='cuda',dtype=torch.float32)
        return (out,*_flash_attn_bwd(*xs,out,g,lse,causal=True,attn_kind=kind,softplus_early_dv=kind=='softplus',
            sm120_bwd_num_threads=256,sm120_bwd_tile=(64,64,1,1),sm120_bwd_warp_layout=layout))
    fs={}
    for kind in ('softplus','softmax'):
        for layout in ((4,4,4),(4,4,2),(4,2,4)):
            fs[kind+'_'+''.join(map(str,layout))]=lambda kind=kind,layout=layout:run(kind,layout)
    errors={}
    for kind in ('softplus','softmax'):
        ref=fs[kind+'_444']()
        for n,fn in fs.items():
            if n.startswith(kind):
                got=fn();errors[n]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(got,ref)]
                assert max(errors[n])<.005,(n,errors[n])
    names=list(fs)
    if a.reverse:names=names[::-1]
    samples={n:[] for n in fs}
    for trial in range(5):
        for n in (names if trial%2==0 else names[::-1]):samples[n].append(timed(fs[n]))
    row=dict(b=b,t=t,h=h,errors=errors,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()})
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
paths=[Path(__file__),Path('flash_attn_4/interface.py'),Path('flash_attn_4/flash_bwd.py')]
Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),reverse=a.reverse,
    scope='attention forward+backward; same head-local forward; 5 alternating CUDA-graph trials; no whole-model timing',
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
