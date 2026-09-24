"""Pair complete owners with identical Softmax controls and one live graph."""
import argparse,gc,hashlib,json,os,statistics,sys
from pathlib import Path
if '--resources' in sys.argv:os.environ.setdefault('CUTE_DSL_KEEP','cubin')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func,softplus_owner_schedule_options
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_general import TunedSoftmax
from trains.bench_softplus_completion import GlobalSoftmax
from trains.bench_softplus_long import timed,sampled_reference

class PairedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        out,lse,*_=_flash_attn_fwd(q,k,v,causal=True,return_lse=True,tile_mn=(64,128 if q.shape[-1]==64 else 64),sm120_owner_pair=True)
        ctx.save_for_backward(q,k,v,out,lse)
        return out
    @staticmethod
    def backward(ctx,g):
        q,k,v,out,lse=ctx.saved_tensors
        dq,dk,dv,*_=_flash_attn_bwd(q,k,v,out,g.contiguous(),lse,causal=True,sm120_bwd_tile=(64,64,2,1))
        return dq,dk,dv


class HeadSoftmax(GlobalSoftmax):
    @staticmethod
    def forward(ctx,q,k,v):
        out,lse,*_=_flash_attn_fwd(q,k,v,causal=True,return_lse=True,tile_mn=(64,128 if q.shape[-1]==64 else 64),sm120_head_lpt=True)
        ctx.save_for_backward(q,k,v,out,lse);ctx.w=-1
        return out

class PairedBothSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        out,lse,*_=_flash_attn_fwd(q,k,v,causal=True,return_lse=True,tile_mn=(64,128 if q.shape[-1]==64 else 64),sm120_owner_pair=True)
        ctx.save_for_backward(q,k,v,out,lse)
        return out
    @staticmethod
    def backward(ctx,g):
        q,k,v,out,lse=ctx.saved_tensors
        dq,dk,dv,*_=_flash_attn_bwd(q,k,v,out,g.contiguous(),lse,causal=True,sm120_bwd_tile=(64,64,2,1),sm120_owner_pair=True)
        return dq,dk,dv

def resources():
    import cuda.bindings.driver as cu
    rows=[]
    for key,fn in _flash_attn_fwd.compile_cache.cache.items():
        if fn.__cubin__ is None:continue
        err,module=cu.cuModuleLoadData(fn.__cubin__)
        if int(err):raise RuntimeError(err)
        try:
            for symbol,attrs in fn.kernel_info.items():
                err,kernel=cu.cuModuleGetFunction(module,symbol.encode())
                if int(err):raise RuntimeError(err)
                row=dict(key=str(key),launch_attributes={str(k):v for k,v in attrs.items()})
                for field,attr in [('registers','NUM_REGS'),('local_bytes','LOCAL_SIZE_BYTES'),('static_smem','SHARED_SIZE_BYTES')]:
                    err,value=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),kernel)
                    row[field]=int(value) if int(err)==0 else str(err)
                rows.append(row)
        finally:cu.cuModuleUnload(module)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--quick',action='store_true');ap.add_argument('--reverse',action='store_true');ap.add_argument('--resources',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);rows=[]
    shapes=[(1,1,4096,64),(1,1,4096,128),(8,6,2048,128),(1,6,32768,64),(1,6,32768,128),(1,1,65536,128)]
    if not a.quick:
        shapes += [(1,h,t,d) for t in (32768,65536) for h in (1,6) for d in (64,128) if (1,h,t,d) not in shapes]
        shapes += [(1,6,2048,64),(8,6,2048,64),(1,6,8192,64),(1,6,8192,128),(1,3,12288,64),(1,3,12288,128),(1,1,49152,64),(1,1,49152,128)]
    opts={'default':{},'native_unsplit':dict(num_splits=1),'whole_lpt':dict(whole_lpt=True),'owner_pair':dict(owner_pair=True),'head_lpt':dict(head_lpt=True)}
    for b,h,t,d in shapes:
        torch.manual_seed(947);xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)];g=torch.randn_like(xs[0])
        with torch.no_grad():
            idx,ref=sampled_reference(xs);out=softplus_attn_fa4(*xs,owner_pair=True)
            sample_error=float((out[:,idx].float()-ref).abs().max()/ref.abs().max());assert sample_error<.025
            del out,ref
        sms=torch.cuda.get_device_properties().multi_processor_count
        owners=b*h*((t+63)//64);paired_owners=b*h*((t+127)//128)
        bounded=dict(cute_stream_waves=2,stream_tiles=True,stream_atomic=True)
        # Candidate, not default policy: compare before adopting a threshold.
        hybrid=softplus_owner_schedule_options(xs[0],xs[1])
        for phase in ('prefill','train'):
            print('warming',phase,b,h,t,d,flush=True)
            funcs={}
            phase_opts=dict(opts,bounded_atomic=bounded,hybrid_candidate=hybrid)
            if phase=='train':
                bwd=dict(bwd_schedule='paired_shared' if d==64 else 'paired')
                phase_opts.update(bwd_pair_control=dict(bwd_schedule='shared' if d==64 else 'previous',bwd_m_chunk=0),bwd_pair_only=bwd,paired_both=dict(owner_pair=True,**bwd))
            for name,kw in phase_opts.items():
                if phase=='prefill':funcs[name]=lambda kw=kw:softplus_attn_fa4(*xs,**kw)
                else:funcs[name]=lambda kw=kw:torch.autograd.grad(softplus_attn_fa4_func(*xs,**kw),xs,g)
            ref=funcs['native_unsplit']();ref=ref if isinstance(ref,tuple) else (ref,);errors={}
            for name,fn in funcs.items():
                out=fn();out=out if isinstance(out,tuple) else (out,)
                errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(out,ref)]
                assert max(errors[name])<.025,(name,errors[name])
                del out
            del ref
            if phase=='prefill':
                for name,kw in [('softmax_tuned',{}),('softmax_global',dict(sm120_global_lpt=True)),('softmax_pair',dict(sm120_owner_pair=True)),('softmax_head',dict(sm120_head_lpt=True))]:
                    funcs[name]=lambda kw=kw:_flash_attn_fwd(*xs,causal=True,tile_mn=(64,128 if d==64 else 64),**kw)[0]
            else:
                funcs['softmax_tuned']=lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,-1),xs,g)
                funcs['softmax_global']=lambda:torch.autograd.grad(GlobalSoftmax.apply(*xs,-1),xs,g)
                funcs['softmax_pair']=lambda:torch.autograd.grad(PairedSoftmax.apply(*xs),xs,g)
                funcs['softmax_head']=lambda:torch.autograd.grad(HeadSoftmax.apply(*xs),xs,g)
                funcs['softmax_paired_both']=lambda:torch.autograd.grad(PairedBothSoftmax.apply(*xs),xs,g)
            old=funcs['softmax_tuned']();new=funcs['softmax_pair']()
            old=old if isinstance(old,tuple) else (old,);new=new if isinstance(new,tuple) else (new,)
            for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
            if phase=='train':
                both=funcs['softmax_paired_both']()
                for x,y in zip(old,both):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
                del both
            del old,new,x,y
            names=list(funcs);samples={n:[] for n in names}
            if a.reverse:names.reverse()
            for trial in range(3):
                for name in (names if trial%2==0 else names[::-1]):samples[name].append(timed(funcs[name]))
            row=dict(phase=phase,b=b,h=h,t=t,d=d,hybrid_options=hybrid,errors=errors,sampled_reference_error=sample_error,results={n:dict(ms=statistics.median(v),samples_ms=v) for n,v in samples.items()})
            rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        del xs,g,funcs;gc.collect();torch.cuda.empty_cache()
    paths=[Path('flash_attn_4')/p for p in ('softplus_api.py','interface.py','flash_fwd.py','flash_fwd_softplus.py','flash_bwd.py','flash_bwd_softplus.py','global_query_scheduler.py')]+[Path(__file__)]
    meta=dict(gpu=torch.cuda.get_device_name(),sms=torch.cuda.get_device_properties().multi_processor_count,torch=torch.__version__,reverse=a.reverse,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if a.resources:meta['resources']=resources()
    Path(a.output+'.metadata.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
