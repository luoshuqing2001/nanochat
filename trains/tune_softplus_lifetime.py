"""Exact-math lifetime and narrow KV-owner experiments; no default dispatch changes."""
import argparse, json, os, statistics, sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--deferred',action='store_true')
ap.add_argument('--long',action='store_true')
ap.add_argument('--threads',action='store_true')
ap.add_argument('--output',required=True)
a=ap.parse_args()
os.environ['CUTE_DSL_KEEP']='cubin'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cutlass
import cutlass.cute as cute
from flash_attn_4.softplus import _log1p_poly, LOG2_E
from flash_attn_4.flash_bwd_softplus import SoftplusBackwardMixin
from flash_attn_4.interface import _flash_attn_fwd, _flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed

@cute.jit
def deferred_p(self, acc_S_mn, acc_S_pre_mn, tLSErLSE, softmax_scale, softmax_scale_log2):
    z=cute.make_rmem_tensor(cute.size(acc_S_mn,mode=[1]),cutlass.Float32)
    z.fill(0.)
    for r in cutlass.range(cute.size(acc_S_mn,mode=[0]),unroll_full=True):
        x=acc_S_mn[r,None].load()*softmax_scale
        y=cute.math.exp2(cute.math.abs(x,fastmath=True)*(-LOG2_E),fastmath=True)
        acc_S_pre_mn[r,None].store(cute.where(x>=0.,y,-y))
        p=cute.math.max(x,z.load())+y*_log1p_poly(y,z.load(),self.softplus_estrin)
        acc_S_mn[r,None].store(p if cutlass.const_expr(self.softplus_scaled_do) else p*tLSErLSE[r])

@cute.jit
def deferred_grad(self, acc_S_mn, acc_dP_mn, acc_S_pre_mn, tLSErdPsum, r):
    sy=acc_S_pre_mn[r,None].load()
    y=cute.math.abs(sy,fastmath=True)
    inv=cute.math.rcp(1.+y,approx=True,ftz=True)
    positive=cute.recast_tensor(acc_S_pre_mn,cutlass.Int32)[r,None].load()>=0
    sig=cute.where(positive,inv,y*inv)
    dp=acc_dP_mn[r,None].load()*sig
    return dp if cutlass.const_expr(self.softplus_scaled_do) else dp*tLSErdPsum[r]

if a.deferred:
    SoftplusBackwardMixin.bwd_recompute_p=deferred_p
    SoftplusBackwardMixin.bwd_grad_val=deferred_grad

def resources():
    import cuda.bindings.driver as cu
    rows=[]
    for key,fn in _flash_attn_bwd.compile_cache.cache.items():
        err,mod=cu.cuModuleLoadData(fn.__cubin__)
        assert int(err)==0
        try:
            for sym in fn.kernel_info:
                err,k=cu.cuModuleGetFunction(mod,sym.encode());assert int(err)==0
                row={}
                for name,attr in [('regs','NUM_REGS'),('local','LOCAL_SIZE_BYTES')]:
                    err,val=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),k)
                    assert int(err)==0
                    row[name]=val
                rows.append(row)
        finally:cu.cuModuleUnload(mod)
    return rows

torch.set_num_threads(2)
rows=[]
for t,h,d in ([(32768,6,128),(65536,6,128)] if a.long else [(2048,2,128),(8192,6,128),(8192,6,64)]):
    torch.manual_seed(812)
    xs=[torch.randn(1,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(xs[0])
    o=_flash_attn_fwd(*xs,causal=True,attn_kind='softplus')[0]
    so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True)
    funcs={}
    for n,stage in (((64,2),(64,1)) if a.threads else ((64,2),(32,2),(32,1),(16,2),(16,1))):
        for share in (False,True):
          for nt in ((128,256) if a.threads else (128,)):
            name=f'sp_n{n}_q{stage}_share{int(share)}'+('_t256' if nt==256 else '')
            funcs[name]=lambda n=n,stage=stage,share=share,nt=nt:_flash_attn_bwd(*xs,o,g,lse,causal=True,attn_kind='softplus',softplus_early_dv=True,softplus_share_pd=share,sm120_bwd_tile=(64,n,stage,1),sm120_bwd_num_threads=nt)[:3]
    funcs['softmax']=lambda:_flash_attn_bwd(*xs,so,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1))[:3]
    if a.threads:
        for stage in (1,2):
            funcs[f'softmax_t256_q{stage}']=lambda stage=stage:_flash_attn_bwd(*xs,so,g,lse,causal=True,sm120_bwd_tile=(64,64,stage,1),sm120_bwd_num_threads=256)[:3]
    errors={};res={};failures={}
    ref=funcs['sp_n64_q2_share0']()
    for name,fn in list(funcs.items()):
        print('compile',a.deferred,t,d,name,flush=True)
        try:
            before=len(_flash_attn_bwd.compile_cache.cache)
            got=fn();torch.cuda.synchronize()
            assert all(torch.isfinite(x).all() for x in got)
            if not name.startswith('softmax'):
                errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(got,ref)]
                assert max(errors[name])<.025,errors[name]
            res[name]=resources()[before:]
            del got
        except Exception as e:
            failures[name]=repr(e);del funcs[name];print('FAILED',repr(e),flush=True)
    del ref
    samples={name:[] for name in funcs}
    for trial in range(3):
        for name in (list(funcs) if trial%2==0 else list(funcs)[::-1]):
            samples[name].append(timed(funcs[name]))
    row=dict(t=t,h=h,d=d,deferred=a.deferred,errors=errors,resources=res,failures=failures,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()})
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
