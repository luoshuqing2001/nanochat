"""Whole GPT F/F+B with one CUDA graph alive at a time, alternating capture order.
No optimizer/data loader/KV-cache writes. Detached loss avoids retaining autograd
nodes across capture streams. A common capture stream is used for all variants.
"""
import os
os.environ.setdefault('NANOCHAT_DTYPE','bfloat16')
import argparse,gc,json,statistics,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nanochat.gpt import GPT,GPTConfig
import nanochat.softplus_attention as attention
from flash_attn_4.softplus_api import softplus_attn_fa4_func
from trains.bench_softplus_general import TunedSoftmax
from trains.bench_softplus_completion import GlobalSoftmax


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--quick',action='store_true');ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);rows=[];original=attention.softplus_attention
    shapes=[(1,2048,64,'L'),(1,2048,128,'L')]
    if not a.quick:shapes += [(1,2048,128,'SSSL'),(1,4096,128,'L')]
    try:
        for b,t,d,pattern in shapes:
            torch.manual_seed(912)
            with torch.device('cuda'):
                model=GPT(GPTConfig(sequence_len=t,n_layer=12,n_head=6,n_kv_head=6,n_embd=6*d,vocab_size=32768,window_pattern=pattern,attn_kind='softplus'))
            model.init_weights()
            with torch.no_grad():
                for block in model.transformer.h:
                    block.attn.c_proj.weight.normal_(std=.01);block.mlp.c_proj.weight.normal_(std=.01)
            tokens=torch.randint(0,32768,(b,t),device='cuda');targets=torch.randint_like(tokens,0,32768)
            variants={'default':{},'global_private':dict(cute_stream_waves=2,stream_tiles=True,stream_global=True),'global_complete':dict(cute_stream_waves=2,stream_tiles=True,stream_global=True,stream_complete=True),'whole_lpt':dict(whole_lpt=True),'softmax_tuned':None,'softmax_global':None}
            capture_stream=torch.cuda.Stream()
            for phase in ('prefill','train'):
                funcs={}
                for name,options in variants.items():
                    def fn(name=name,options=options):
                        def backend(q,k,v,window_size=(-1,0),alpha=1.):
                            if name=='softmax_global':return GlobalSoftmax.apply(q,k,v,window_size[0])
                            if name=='softmax_tuned':return TunedSoftmax.apply(q,k,v,window_size[0])
                            kw=options if window_size[0]<0 else {}
                            return softplus_attn_fa4_func(q,k,v,window_size=window_size,alpha=alpha,**kw)
                        attention.softplus_attention=backend
                        if phase=='prefill':
                            with torch.no_grad():return model(tokens)
                        model.zero_grad(set_to_none=True)
                        loss=model(tokens,targets);loss.backward();return loss.detach()
                    funcs[name]=fn
                ref=funcs['default']().detach().clone();errors={}
                for name,fn in funcs.items():
                    out=fn().detach();assert torch.isfinite(out).all()
                    if not name.startswith('softmax'):
                        errors[name]=float((out.float()-ref.float()).abs().max()/ref.float().abs().max().clamp_min(1e-20));assert errors[name]<.02
                del ref,out
                samples={name:[] for name in funcs}
                orders=[]
                # Recapture serially in each trial: graph allocation order and replay
                # order both alternate, no simultaneously retained activation pools.
                for trial in range(4):
                    names=list(funcs)
                    if bool(trial%2)^a.reverse:names.reverse()
                    orders.append(names)
                    for name in names:
                        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                        capture_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(capture_stream):
                            for _ in range(2):funcs[name]()
                        torch.cuda.current_stream().wait_stream(capture_stream)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=capture_stream):output=funcs[name]()
                        for _ in range(5):graph.replay()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):graph.replay()
                        end.record();end.synchronize()
                        samples[name].append(start.elapsed_time(end)/10)
                        del output,graph;model.zero_grad(set_to_none=True);gc.collect()
                result={name:dict(ms=statistics.median(v),samples_ms=v) for name,v in samples.items()}
                row=dict(phase=phase,b=b,t=t,d=d,h=6,layers=12,vocab=32768,pattern=pattern,capture_orders=orders,errors=errors,results=result)
                rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
            del funcs,fn,model;gc.collect();torch.cuda.empty_cache()
    finally:attention.softplus_attention=original
    paths=[Path('flash_attn_4')/p for p in ('softplus_api.py','softplus_stream.py','flash_fwd.py','flash_fwd_softplus.py','interface.py')]
    Path(a.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),timing='4 alternating serial captures, 5 warm replays then 10 measured; CUDA events; one graph alive',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))

if __name__=='__main__':main()
