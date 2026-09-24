"""Separate dV and dQ/dK: count both complete operators, including setup/conversion."""
import argparse,json,os,statistics,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--long',action='store_true');a=ap.parse_args()
os.environ['CUTE_DSL_KEEP']='cubin'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed
torch.set_num_threads(2);rows=[]
for t,d in ([(32768,128),(65536,128)] if a.long else [(2048,128),(8192,128),(8192,64)]):
    torch.manual_seed(194);xs=[torch.randn(1,t,6,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(xs[0]);out=_flash_attn_fwd(*xs,causal=True,attn_kind='softplus')[0]
    so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True)
    def call(component=None,nt=256,st=1):
        return _flash_attn_bwd(*xs,out,g,lse,causal=True,attn_kind='softplus',softplus_early_dv=True,
            sm120_bwd_num_threads=nt,sm120_bwd_tile=(64,64,st,1),softplus_bwd_component=component,
            softplus_share_pd=component is None and d==64)
    def split(qnt,qst,vnt):
        dq,dk,_=call('qk',qnt,qst);_,_,dv=call('v',vnt,1)
        return dq,dk,dv
    funcs={'fused':lambda:call(None,256 if d==128 else 128,1 if d==128 else 2)}
    for qnt,qst,vnt in ((128,1,128),(128,2,128),(256,1,128),(256,1,256)):
        funcs[f'split_q{qnt}s{qst}_v{vnt}']=lambda qnt=qnt,qst=qst,vnt=vnt:split(qnt,qst,vnt)
    funcs['qk128']=lambda:call('qk',128,1)
    funcs['qk256']=lambda:call('qk',256,1)
    funcs['v128']=lambda:call('v',128,1)
    funcs['v256']=lambda:call('v',256,1)
    funcs['softmax']=lambda:_flash_attn_bwd(*xs,so,g,lse,causal=True,sm120_bwd_num_threads=256,sm120_bwd_tile=(64,64,1,1))
    ref=funcs['fused']();errs={};res={}
    for name,fn in funcs.items():
        print('compile',t,d,name,flush=True)
        keys=set(_flash_attn_bwd.compile_cache.cache);got=fn();torch.cuda.synchronize()
        if name.startswith('split'):
            errs[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(got,ref)]
            assert max(errs[name])<.005,(name,errs[name])
        import cuda.bindings.driver as cu
        res[name]=[]
        for key in set(_flash_attn_bwd.compile_cache.cache)-keys:
            compiled=_flash_attn_bwd.compile_cache.cache[key]
            err,mod=cu.cuModuleLoadData(compiled.__cubin__);assert int(err)==0
            for sym in compiled.kernel_info:
                err,k=cu.cuModuleGetFunction(mod,sym.encode());assert int(err)==0
                vals={}
                for label,attr in [('regs','NUM_REGS'),('local','LOCAL_SIZE_BYTES')]:
                    err,val=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),k);assert int(err)==0
                    vals[label]=val
                res[name].append(vals)
            cu.cuModuleUnload(mod)
    samples={n:[] for n in funcs}
    for trial in range(3):
        for n in (list(funcs) if trial%2==0 else list(funcs)[::-1]):samples[n].append(timed(funcs[n]))
    row=dict(t=t,d=d,errors=errs,resources=res,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()})
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
