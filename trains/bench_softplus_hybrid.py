"""Paired hybrid atomic experiments; all initialization and finish kernels included."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func
from flash_attn_4.interface import _flash_attn_fwd
from trains.bench_softplus_general import measure,TunedSoftmax


def plan_stats(b,h,t,d,w,options,sms):
    from flash_attn_4.softplus_stream import stream_plan_cpu,tile_plan_cpu
    result={}
    for name,kw in options.items():
        waves=kw.get('cute_stream_waves',0)
        if not waves:continue
        workers=max(1,(waves*sms+b*h-1)//(b*h))
        args=(t,t,w,64,128 if d==64 else 64,workers)
        offsets,segs,groups,slots=(tile_plan_cpu(*args) if kw.get('stream_tiles') else stream_plan_cpu(*args,kw.get('stream_tail',False)))
        nslots=len(groups) if kw.get('stream_atomic') else slots
        result[name]=dict(ctas=(len(offsets)-1)*b*h,split_query_tiles=len(groups)*b*h,
                          partial_slots=nslots*b*h,workspace_bytes=nslots*b*h*64*d*4)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--tiles',action='store_true')
    ap.add_argument('--profile',choices=('default','atomic','private'))
    ap.add_argument('--phase',choices=('prefill','train','both'),default='both')
    a=ap.parse_args();torch.set_num_threads(2)
    shapes=[(1,6,2048,d,w,False) for d in (64,128) for w in (-1,511)]
    if not a.quick:
        shapes += [(b,h,t,d,w,False) for b,h,t in ((1,6,8192),(8,6,2048),(1,1,4096)) for d in (64,128) for w in (-1,511)]
        shapes += [(2,5,3584,64,-1,True),(1,5,6144,128,-1,True),(1,3,1792,128,383,True),(2,4,769,64,95,True)]
    rows=[]
    for b,h,t,d,w,strided in shapes:
        torch.manual_seed(735)
        xs=[(torch.randn(b,t,3,h,d,device='cuda',dtype=torch.bfloat16)[:,:,i] if strided else torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16)).requires_grad_() for i in range(3)]
        g=torch.randn_like(xs[0])
        opts={'default':{},'private2':dict(cute_stream_waves=2),'tail_private2':dict(cute_stream_waves=2,stream_tail=True)}
        for waves in (1,2,4):
            for tail in (False,True):
                opts[f'atomic{waves}'+('_tail' if tail else '')]=dict(cute_stream_waves=waves,stream_tail=tail,stream_atomic=True)
        if a.tiles:
            opts={'default':{},'private2':dict(cute_stream_waves=2),'atomic2':dict(cute_stream_waves=2,stream_atomic=True)}
            for waves in (2,4,8):
                for atomic in (False,True):
                    opts[f'tiles{waves}_'+('atomic' if atomic else 'private')]=dict(cute_stream_waves=waves,stream_tiles=True,stream_atomic=atomic)
        for phase in ('prefill','train') if a.phase=='both' else (a.phase,):
            funcs={}
            for name,kw in opts.items():
                if phase=='prefill':funcs[name]=lambda kw=kw:softplus_attn_fa4(*xs,window_size=(w,0),num_splits='auto',**kw)
                else:funcs[name]=lambda kw=kw:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),**kw),xs,g)
            if a.profile:
                fn=funcs[dict(default='default',atomic='atomic2',private='private2')[a.profile]]
                for _ in range(3):fn()
                torch.cuda.synchronize();torch.cuda.profiler.start();fn();torch.cuda.synchronize();torch.cuda.profiler.stop();return
            ref=funcs['default']();ref=ref if isinstance(ref,tuple) else (ref,);errors={}
            print('warming',phase,b,h,t,d,w,flush=True)
            for name,fn in funcs.items():
                out=fn();out=out if isinstance(out,tuple) else (out,)
                errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(out,ref)]
                assert max(errors[name])<.025,(name,errors[name])
            if phase=='train':funcs['softmax_tuned']=lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,w),xs,g)
            else:funcs['softmax_tuned']=lambda:_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,tile_mn=(64,128 if d==64 else 64))[0]
            row=dict(phase=phase,b=b,h=h,t=t,d=d,w=w,strided=strided,errors=errors,plans=plan_stats(b,h,t,d,w,opts,torch.cuda.get_device_properties().multi_processor_count),results=measure(funcs,30,True))
            rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    paths=[Path('flash_attn_4')/p for p in ('softplus_stream.py','softplus_api.py','interface.py','flash_fwd_softplus.py')]
    Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),timing='alternating order 3 x 30ms CUDA graph; eager 10 calls; allocation/zero/finish included',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))

if __name__=='__main__':main()
