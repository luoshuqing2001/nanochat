"""Long-context paired attention benchmark with one live CUDA graph at a time."""
import argparse,gc,hashlib,json,math,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func
from flash_attn_4.interface import _flash_attn_fwd
from trains.bench_softplus_general import TunedSoftmax
from trains.bench_softplus_completion import GlobalSoftmax
from flash_attn_4.softplus_stream import capped_plan_cpu


def timed(fn):
    # A fresh graph is destroyed before the next variant. Include all kernels.
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    graph.replay()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();graph.replay();end.record();end.synchronize()
    n=max(1,min(10,math.ceil(30/max(start.elapsed_time(end),.001))))
    start.record()
    for _ in range(n):graph.replay()
    end.record();end.synchronize()
    ms=start.elapsed_time(end)/n
    del out,graph
    return ms


def sampled_reference(xs):
    q,k,v=xs;t=q.shape[1];d=q.shape[-1]
    idx=torch.tensor(sorted(set([0,1,63,64,255,511,1023,t//4,t//2,t-2,t-1])),device=q.device)
    qq=q[:,idx].float().transpose(1,2);kk=k.float().transpose(1,2);vv=v.float().transpose(1,2)
    scores=(qq@kk.transpose(-1,-2))*d**-.5
    mask=torch.arange(t,device=q.device)[None,:]<=idx[:,None]
    p=torch.nn.functional.softplus(scores).masked_fill(~mask,0)
    ref=(p@vv)/(idx+1)[None,None,:,None]
    return idx,ref.transpose(1,2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--lengths',type=int,nargs='+',default=[32768,65536]);ap.add_argument('--heads',type=int,nargs='+',default=[1,6]);ap.add_argument('--dims',type=int,nargs='+',default=[64,128]);ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);rows=[]
    opts={'default':{},'whole_lpt':dict(whole_lpt=True)}
    for cap in (1024,4096,8192):
        opts[f'cap{cap}']=dict(kv_split_size=cap,stream_atomic=True)
        opts[f'cap{cap}_kv']=dict(kv_split_size=cap,stream_atomic=True,kv_major=True)
    for t in a.lengths:
      for h in a.heads:
       for d in a.dims:
        torch.manual_seed(193);xs=[torch.randn(1,t,h,d,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)];g=torch.randn_like(xs[0])
        print('shape',t,h,d,flush=True)
        with torch.no_grad():
            idx,ref=sampled_reference(xs)
            out=softplus_attn_fa4(*xs)
            sampled_error=float((out[:,idx].float()-ref).abs().max()/ref.abs().max())
            assert sampled_error<.025,sampled_error
            del out,ref
        for phase in ('prefill','train'):
            funcs={}
            for name,kw in opts.items():
                if phase=='prefill':funcs[name]=lambda kw=kw:softplus_attn_fa4(*xs,**kw)
                else:funcs[name]=lambda kw=kw:torch.autograd.grad(softplus_attn_fa4_func(*xs,**kw),xs,g)
            print('validate',phase,t,h,d,flush=True)
            ref=funcs['default']();ref=ref if isinstance(ref,tuple) else (ref,);errors={}
            for name,fn in funcs.items():
                out=fn();out=out if isinstance(out,tuple) else (out,)
                errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(out,ref)]
                assert max(errors[name])<.025,(name,errors[name])
                del out
            del ref
            if phase=='prefill':
                funcs['softmax_tuned']=lambda:_flash_attn_fwd(*xs,causal=True,tile_mn=(64,128 if d==64 else 64))[0]
                funcs['softmax_global']=lambda:_flash_attn_fwd(*xs,causal=True,tile_mn=(64,128 if d==64 else 64),sm120_global_lpt=True)[0]
            else:
                funcs['softmax_tuned']=lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,-1),xs,g)
                funcs['softmax_global']=lambda:torch.autograd.grad(GlobalSoftmax.apply(*xs,-1),xs,g)
            samples={n:[] for n in funcs};names=list(funcs)
            if a.reverse:names.reverse()
            for trial in range(3):
                for name in (names if trial%2==0 else names[::-1]):
                    ms=timed(funcs[name]);samples[name].append(ms)
                print('trial',phase,t,h,d,trial,flush=True)
            plans={}
            for cap in (1024,4096,8192):
                plan=capped_plan_cpu(t,t,-1,64,128 if d==64 else 64,cap)
                plans[str(cap)]=dict(ctas=(len(plan[0])-1)*h,split_tiles=len(plan[2])*h,fp32_partial_bytes=len(plan[2])*h*64*d*4)
            row=dict(phase=phase,b=1,t=t,h=h,d=d,results={n:dict(ms=statistics.median(ss),samples_ms=ss) for n,ss in samples.items()},errors=errors,sampled_reference_error=sampled_error,plans=plans)
            rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        del xs,g,funcs;gc.collect();torch.cuda.empty_cache()
    paths=[Path('flash_attn_4')/p for p in ('softplus_stream.py','softplus_api.py','interface.py','flash_fwd.py','flash_fwd_softplus.py','global_query_scheduler.py')]+[Path(__file__)]
    Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,reverse=a.reverse,timing='3 alternating trials, one live CUDA graph, adaptive 1-10 replays targeting 30ms; zero/reduce/cast included',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))

if __name__=='__main__':main()
