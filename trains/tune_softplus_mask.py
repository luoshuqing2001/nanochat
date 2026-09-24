"""8-warp mask and lifetime experiments, preserving production defaults."""
import argparse, json, os, statistics, sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--mode',choices=('baseline','mask','deferred','combined','estrin','prescale'),required=True)
ap.add_argument('--output',required=True)
a=ap.parse_args()
os.environ['CUTE_DSL_KEEP']='cubin'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cutlass
import cutlass.cute as cute
from flash_attn_4.flash_bwd import FlashAttentionBackwardSm80
from flash_attn_4.flash_bwd_softplus import SoftplusBackwardMixin
from flash_attn_4.softplus import _log1p_poly,LOG2_E,softplus_and_sigmoid_
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed

@cute.jit
def skip_interior(self,acc,m_block,thr_mma,n_block,seqlen,window_left,window_right):
    if cutlass.const_expr(self.is_local):
        self.original_column_mask(acc,m_block,thr_mma,n_block,seqlen,window_left,window_right)
    else:
        needed=(n_block+1)*self.n_block_size>seqlen.seqlen_k
        if cutlass.const_expr(self.is_causal):
            needed=needed or (n_block+1)*self.n_block_size>m_block*self.m_block_size+seqlen.seqlen_k-seqlen.seqlen_q+1
        if needed:
            self.original_column_mask(acc,m_block,thr_mma,n_block,seqlen,window_left,window_right)

@cute.jit
def deferred_p(self,acc_S_mn,acc_S_pre_mn,tLSErLSE,softmax_scale,softmax_scale_log2):
    z=cute.make_rmem_tensor(cute.size(acc_S_mn,mode=[1]),cutlass.Float32);z.fill(0.)
    for r in cutlass.range(cute.size(acc_S_mn,mode=[0]),unroll_full=True):
        x=acc_S_mn[r,None].load()*softmax_scale
        y=cute.math.exp2(cute.math.abs(x,fastmath=True)*(-LOG2_E),fastmath=True)
        acc_S_pre_mn[r,None].store(cute.where(x>=0.,y,-y))
        p=cute.math.max(x,z.load())+y*_log1p_poly(y,z.load(),self.softplus_estrin)
        acc_S_mn[r,None].store(p if cutlass.const_expr(self.softplus_scaled_do) else p*tLSErLSE[r])

@cute.jit
def deferred_grad(self,acc_S_mn,acc_dP_mn,acc_S_pre_mn,tLSErdPsum,r):
    sy=acc_S_pre_mn[r,None].load();y=cute.math.abs(sy,fastmath=True)
    inv=cute.math.rcp(1.+y,approx=True,ftz=True)
    positive=cute.recast_tensor(acc_S_pre_mn,cutlass.Int32)[r,None].load()>=0
    sig=cute.where(positive,inv,y*inv)
    dp=acc_dP_mn[r,None].load()*sig
    return dp if cutlass.const_expr(self.softplus_scaled_do) else dp*tLSErdPsum[r]

@cute.jit
def prescale_p(self,acc_S_mn,acc_S_pre_mn,tLSErLSE,softmax_scale,softmax_scale_log2):
    z=cute.make_rmem_tensor(cute.size(acc_S_mn,mode=[1]),cutlass.Float32);z.fill(0.)
    for r in cutlass.range(cute.size(acc_S_mn,mode=[0]),unroll_full=True):
        sp,sig=softplus_and_sigmoid_(acc_S_mn[r,None].load()*softmax_scale,z.load(),self.softplus_estrin)
        if cutlass.const_expr(not self.softplus_scaled_do):
            sp=sp*tLSErLSE[r];sig=sig*tLSErLSE[r]
        acc_S_mn[r,None].store(sp);acc_S_pre_mn[r,None].store(sig)

@cute.jit
def prescale_grad(self,acc_S_mn,acc_dP_mn,acc_S_pre_mn,tLSErdPsum,r):
    return acc_dP_mn[r,None].load()*acc_S_pre_mn[r,None].load()

if a.mode in ('mask','combined'):
    FlashAttentionBackwardSm80.original_column_mask=FlashAttentionBackwardSm80.mask_column_warps
    FlashAttentionBackwardSm80.mask_column_warps=skip_interior
if a.mode in ('deferred','combined'):
    SoftplusBackwardMixin.bwd_recompute_p=deferred_p
    SoftplusBackwardMixin.bwd_grad_val=deferred_grad
if a.mode=='prescale':
    SoftplusBackwardMixin.bwd_recompute_p=prescale_p
    SoftplusBackwardMixin.bwd_grad_val=prescale_grad

torch.set_num_threads(2);rows=[]
for t,h in ((2048,6),(8192,6),(32768,6),(65536,6)):
    torch.manual_seed(662)
    xs=[torch.randn(1,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(xs[0]);o=_flash_attn_fwd(*xs,causal=True,attn_kind='softplus')[0]
    so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True)
    fs={}
    for kind in ('softplus','softmax'):
        fs[kind]=lambda kind=kind:_flash_attn_bwd(*xs,o if kind=='softplus' else so,g,lse,causal=True,
            attn_kind=kind,softplus_early_dv=kind=='softplus',softplus_poly_estrin=a.mode=='estrin' and kind=='softplus',sm120_bwd_tile=(64,64,1,1),sm120_bwd_num_threads=256)
    for fn in fs.values():
        out=fn();assert all(torch.isfinite(x).all() for x in out)
    resources=[]
    if not rows:
        import cuda.bindings.driver as cu
        for fn in _flash_attn_bwd.compile_cache.cache.values():
            err,mod=cu.cuModuleLoadData(fn.__cubin__);assert int(err)==0
            for symbol in fn.kernel_info:
                err,k=cu.cuModuleGetFunction(mod,symbol.encode());assert int(err)==0
                vals={}
                for name,attr in [('regs','NUM_REGS'),('local','LOCAL_SIZE_BYTES')]:
                    err,v=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),k);assert int(err)==0
                    vals[name]=v
                resources.append(vals)
            cu.cuModuleUnload(mod)
    samples={n:[] for n in fs}
    for trial in range(5):
        for n in (list(fs) if trial%2==0 else list(fs)[::-1]):samples[n].append(timed(fs[n]))
    row=dict(mode=a.mode,t=t,h=h,resources=resources,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()})
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
