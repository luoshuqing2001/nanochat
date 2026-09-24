"""Public-autograd forward+backward and matching Softmax controls for warp8."""
import argparse, hashlib, json, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from flash_attn_4.softplus_api import softplus_attn_fa4_func
from trains.bench_softplus_owner_pair_stable import timed

ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
torch.set_num_threads(2)
shapes=[(1,2048,1,128),(1,2048,6,128),(8,2048,12,128),(1,8192,6,128),
        (1,32768,6,128),(1,65536,6,128),(1,8192,6,64)]
if a.reverse:shapes=shapes[::-1]
rows=[]
for b,t,h,d in shapes:
    torch.manual_seed(662)
    xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    g=torch.randn_like(xs[0])
    # Preserve the same forward schedule across before/after. Default backward
    # retains its existing small-grid splitting policy; only its new regime changes.
    def sp_fwd(old=False):
        return softplus_attn_fa4_func(*xs,head_lpt=True,bwd_schedule='previous' if old and d==128 else ('shared' if old else 'auto'))
    def sp_train(old=False):
        out=sp_fwd(old);return (out,*torch.autograd.grad(out,xs,g))
    def sm_fwd(lse=False):
        return _flash_attn_fwd(*xs,causal=True,return_lse=lse,tile_mn=(64,128 if d==64 else 64),sm120_head_lpt=True)
    def sm_train(nt=128,stage=2):
        out,lse,*_=sm_fwd(True)
        return (out,*_flash_attn_bwd(*xs,out,g,lse,causal=True,sm120_bwd_tile=(64,64,stage,1),sm120_bwd_num_threads=nt))
    funcs={'sp_before':lambda:sp_train(True),'sp_after':sp_train,
           'sm_128':sm_train,'sm_256':lambda:sm_train(256,1)}
    before=funcs['sp_before']();after=funcs['sp_after']()
    errors=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(after,before)]
    assert max(errors)<.005,errors
    smold=funcs['sm_128']();smnew=funcs['sm_256']()
    smerrors=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(smnew,smold)]
    assert max(smerrors)<.005,smerrors
    del before,after,smold,smnew
    for phase,ff in [('prefill',{'softplus':lambda:sp_fwd(),'softmax':lambda:sm_fwd()[0]}),('train',funcs)]:
        samples={n:[] for n in ff};names=list(ff)
        if a.reverse:names=names[::-1]
        for trial in range(3):
            for n in (names if trial%2==0 else names[::-1]):samples[n].append(timed(ff[n]))
        row=dict(b=b,t=t,h=h,d=d,phase=phase,errors=errors,softmax_errors=smerrors,
                 results={n:dict(ms=statistics.median(v),samples_ms=v) for n,v in samples.items()})
        rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
paths=[Path(__file__)]+[Path('flash_attn_4')/p for p in ('interface.py','flash_bwd.py','flash_bwd_softplus.py','softplus_api.py','softplus.py')]
Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,reverse=a.reverse,
    timing='3 alternating trials, one live CUDA graph, 20ms warmup and 60ms target per sample',
    scope='Attention operator forward plus backward, not whole-model training. Same head-local LPT forward before/after.',
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
