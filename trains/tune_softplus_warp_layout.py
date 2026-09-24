"""Matched 8-warp MMA layout search, with gradient checks before timing."""
import argparse,itertools,json,os,statistics,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--long',action='store_true');a=ap.parse_args()
os.environ['CUTE_DSL_KEEP']='cubin'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed
torch.set_num_threads(2);rows=[]
for t in ((32768,65536) if a.long else (2048,8192)):
    torch.manual_seed(662);xs=[torch.randn(1,t,6,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(xs[0]);o=_flash_attn_fwd(*xs,causal=True,attn_kind='softplus')[0]
    so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True)
    funcs={};errs={};resources={}
    for kind in ('softplus','softmax'):
        for layout in itertools.product((4,2),repeat=3):
            name=kind+'_'+''.join(map(str,layout))
            funcs[name]=lambda kind=kind,layout=layout:_flash_attn_bwd(*xs,o if kind=='softplus' else so,g,lse,causal=True,attn_kind=kind,
                softplus_early_dv=kind=='softplus',sm120_bwd_num_threads=256,sm120_bwd_tile=(64,64,1,1),sm120_bwd_warp_layout=layout)
        ref=funcs[kind+'_444']()
        for name,fn in list(funcs.items()):
            if not name.startswith(kind):continue
            old=set(_flash_attn_bwd.compile_cache.cache);out=fn();torch.cuda.synchronize()
            errs[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(out,ref)]
            assert max(errs[name])<.005,(name,errs[name])
            if set(_flash_attn_bwd.compile_cache.cache)-old:
                import cuda.bindings.driver as cu
                res=[]
                for key in set(_flash_attn_bwd.compile_cache.cache)-old:
                    compiled=_flash_attn_bwd.compile_cache.cache[key]
                    err,mod=cu.cuModuleLoadData(compiled.__cubin__);assert int(err)==0
                    for sym in compiled.kernel_info:
                        err,k=cu.cuModuleGetFunction(mod,sym.encode());assert int(err)==0
                        vals={}
                        for label,attr in [('regs','NUM_REGS'),('local','LOCAL_SIZE_BYTES')]:
                            err,val=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),k);assert int(err)==0
                            vals[label]=val
                        res.append(vals)
                    cu.cuModuleUnload(mod)
                resources[name]=res
            print('validated',t,name,flush=True)
    samples={n:[] for n in funcs}
    for trial in range(3):
        for n in (list(funcs) if trial%2==0 else list(funcs)[::-1]):samples[n].append(timed(funcs[n]))
    row=dict(t=t,errors=errs,resources=resources,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()})
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
