"""Paired end-to-end attention timing for global and last-completer schedules."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_general import measure,TunedSoftmax
from flash_attn_4.softplus_stream import global_plan_cpu,stream_plan_cpu,capped_plan_cpu


class WideSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,w):
        o,lse,*_=_flash_attn_fwd(q,k,v,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,tile_mn=(64,128 if q.shape[-1]==64 else 64),return_lse=True)
        ctx.save_for_backward(q,k,v,o,lse);ctx.w=w
        return o
    @staticmethod
    def backward(ctx,g):
        q,k,v,o,lse=ctx.saved_tensors;w=ctx.w
        dq,dk,dv,*_=_flash_attn_bwd(q,k,v,o,g.contiguous(),lse,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,sm120_bwd_tile=(64,128,1,1))
        return dq,dk,dv,None


class GlobalSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,w):
        o,lse,*_=_flash_attn_fwd(q,k,v,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,tile_mn=(64,128 if q.shape[-1]==64 else 64),return_lse=True,sm120_global_lpt=w<0)
        ctx.save_for_backward(q,k,v,o,lse);ctx.w=w
        return o
    @staticmethod
    def backward(ctx,g):
        q,k,v,o,lse=ctx.saved_tensors;w=ctx.w
        dq,dk,dv,*_=_flash_attn_bwd(q,k,v,o,g.contiguous(),lse,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,sm120_bwd_tile=(64,64,2,1))
        return dq,dk,dv,None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--quick',action='store_true');ap.add_argument('--controls-only',action='store_true');ap.add_argument('--kv-caps',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);rows=[];sms=torch.cuda.get_device_properties().multi_processor_count
    shapes=[(1,6,2048,d,w,False) for d in (64,128) for w in (-1,511)]
    shapes += [(1,1,4096,d,-1,False) for d in (64,128)]
    if not a.quick:
        shapes += [(b,6,t,d,w,False) for b,t in ((1,8192),(8,2048)) for d in (64,128) for w in (-1,511)]
        shapes += [(2,5,3584,64,-1,True),(1,5,6144,128,-1,True),(1,3,1792,128,383,True),(2,4,769,64,95,True)]
    for b,h,t,d,w,strided in shapes:
        torch.manual_seed(912)
        xs=[(torch.randn(b,t,3,h,d,device='cuda',dtype=torch.bfloat16)[:,:,i] if strided else torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16)).requires_grad_() for i in range(3)]
        g=torch.randn_like(xs[0]);opts={'default':{},'persistent_private':dict(cute_stream_waves=2),'persistent_complete':dict(cute_stream_waves=2,stream_complete=True),'global_private':dict(cute_stream_waves=2,stream_tiles=True,stream_global=True),'global_complete':dict(cute_stream_waves=2,stream_tiles=True,stream_global=True,stream_complete=True)}
        if a.kv_caps:
            opts={'default':{},'native_unsplit':dict(num_splits=1),
                  'whole_lpt':dict(whole_lpt=w<0)}
            for cap in (256,512,1024,2048):
                opts[f'cap{cap}_atomic']=dict(kv_split_size=cap,stream_atomic=True)
                opts[f'cap{cap}_global_atomic']=dict(kv_split_size=cap,stream_atomic=True,stream_global=True)
            opts['cap512_private']=dict(kv_split_size=512)
        plans={}
        for name,kw in opts.items():
            if not kw or 'num_splits' in kw or 'whole_lpt' in kw:continue
            if kw.get('kv_split_size'):plan=capped_plan_cpu(t,t,w,64,128 if d==64 else 64,kw['kv_split_size'])
            elif kw.get('stream_global'):plan=global_plan_cpu(t,t,w,64,128 if d==64 else 64,2*sms,b*h)
            else:plan=stream_plan_cpu(t,t,w,64,128 if d==64 else 64,max(1,(2*sms+b*h-1)//(b*h)))
            plans[name]=dict(ctas=(len(plan[0])-1)*b*h,split_query_tiles=len(plan[2])*b*h,partial_bytes=(len(plan[2]) if kw.get("stream_atomic") else plan[3])*b*h*64*d*4)
        for phase in ('prefill','train'):
            funcs={}
            for name,kw in opts.items():
                if phase=='prefill':funcs[name]=lambda kw=kw:softplus_attn_fa4(*xs,window_size=(w,0),**kw)
                else:funcs[name]=lambda kw=kw:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),**kw),xs,g)
            if phase=='train' and d==64 and not a.kv_caps:
                funcs['kv_group2']=lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),bwd_schedule='kv_group2'),xs,g)
                funcs['global_kv_group2']=lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),bwd_schedule='kv_group2',**opts['global_complete']),xs,g)
            if a.controls_only:
                funcs={k:v for k,v in funcs.items() if k in ('default','global_private','global_complete')}
                if phase=='prefill':funcs['whole_lpt']=lambda:softplus_attn_fa4(*xs,window_size=(w,0),num_splits='auto',whole_lpt=w<0)
                else:funcs['whole_lpt']=lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),whole_lpt=w<0),xs,g)
            print('warming',phase,b,h,t,d,w,flush=True)
            ref=funcs['default']();ref=ref if isinstance(ref,tuple) else (ref,);errors={}
            for name,fn in funcs.items():
                out=fn();out=out if isinstance(out,tuple) else (out,)
                errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(out,ref)]
                assert max(errors[name])<.025,(name,errors[name])
            if phase=='train':
                funcs['softmax_tuned']=lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,w),xs,g)
                if d==64 and not a.controls_only and not a.kv_caps:funcs['softmax_wide']=lambda:torch.autograd.grad(WideSoftmax.apply(*xs,w),xs,g)
            else:funcs['softmax_tuned']=lambda:_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,tile_mn=(64,128 if d==64 else 64))[0]
            if a.controls_only or a.kv_caps:
                if phase=='train':funcs['softmax_global']=lambda:torch.autograd.grad(GlobalSoftmax.apply(*xs,w),xs,g)
                else:funcs['softmax_global']=lambda:_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,tile_mn=(64,128 if d==64 else 64),sm120_global_lpt=w<0)[0]
                base=funcs['softmax_tuned']();new=funcs['softmax_global']()
                base=base if isinstance(base,tuple) else (base,);new=new if isinstance(new,tuple) else (new,)
                for x,y in zip(base,new):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
            row=dict(phase=phase,b=b,h=h,t=t,d=d,w=w,strided=strided,errors=errors,plans=plans,results=measure(funcs,30,False))
            rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    paths=[Path('flash_attn_4')/p for p in ('softplus_stream.py','softplus_api.py','interface.py','flash_fwd.py','flash_fwd_softplus.py','global_query_scheduler.py')]
    Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),sms=sms,torch=torch.__version__,timing='median of 3 alternating CUDA graph trials, 30ms; zero/cast/reduction included',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))

if __name__=='__main__':main()
