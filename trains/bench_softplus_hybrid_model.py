"""Whole nanochat GPT forward / forward+backward; no optimizer or data loading.
All variants use the same parameters and token inputs. Attention backend is swapped
only in this process. CUDA graphs include projections, MLP, vocabulary head/loss.
"""
import os
os.environ.setdefault('NANOCHAT_DTYPE','bfloat16')
import argparse,json,sys,statistics,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nanochat.gpt import GPT,GPTConfig
import nanochat.softplus_attention as attention
from flash_attn_4.softplus_api import softplus_attn_fa4_func
from trains.bench_softplus_general import TunedSoftmax


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--quick',action='store_true');ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);rows=[]
    shapes=[(1,2048,128,'L'),(1,2048,128,'SSSL')]
    if not a.quick:shapes += [(1,2048,64,'L'),(1,4096,128,'L')]
    original=attention.softplus_attention
    try:
        for b,t,d,pattern in shapes:
            torch.manual_seed(735)
            config=GPTConfig(sequence_len=t,n_layer=12,n_head=6,n_kv_head=6,n_embd=6*d,vocab_size=32768,window_pattern=pattern,attn_kind='softplus')
            with torch.device('cuda'):model=GPT(config)
            model.init_weights()
            # Nonzero residual projections so attention gradients are exercised.
            with torch.no_grad():
                for block in model.transformer.h:
                    block.attn.c_proj.weight.normal_(std=.01)
                    block.mlp.c_proj.weight.normal_(std=.01)
            tokens=torch.randint(0,config.vocab_size,(b,t),device='cuda');targets=torch.randint_like(tokens,0,config.vocab_size)
            variants={'default':{},'private2':dict(cute_stream_waves=2),'atomic2':dict(cute_stream_waves=2,stream_atomic=True),'tiles2_private':dict(cute_stream_waves=2,stream_tiles=True),'tiles2_atomic':dict(cute_stream_waves=2,stream_tiles=True,stream_atomic=True),'softmax_tuned':None}
            if a.reverse:variants=dict(reversed(list(variants.items())))
            for phase in ('prefill','train'):
                funcs={}
                for name,options in variants.items():
                    def fn(name=name,options=options):
                        def backend(q,k,v,window_size=(-1,0),alpha=1.):
                            if name=='softmax_tuned':return TunedSoftmax.apply(q,k,v,window_size[0])
                            # Local windows showed no benefit from splitting; preserve default.
                            kw=options if window_size[0]<0 else {}
                            return softplus_attn_fa4_func(q,k,v,window_size=window_size,alpha=alpha,**kw)
                        attention.softplus_attention=backend
                        if phase=='prefill':
                            with torch.no_grad():return model(tokens)
                        model.zero_grad(set_to_none=True)
                        loss=model(tokens,targets);loss.backward();return loss
                    funcs[name]=fn
                errors={};ref=funcs['default']().detach().clone()
                for name,fn in funcs.items():
                    out=fn().detach();assert torch.isfinite(out).all()
                    if name!='softmax_tuned':
                        errors[name]=float((out.float()-ref.float()).abs().max()/ref.float().abs().max().clamp_min(1e-20));assert errors[name]<.02
                del ref,out
                # Static CUDA graphs eliminate Python dispatch noise. Capture one per variant,
                # then alternate replay order. Graph memory is released after each phase.
                graphs={};outputs={}
                for name,fn in funcs.items():
                    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):fn()
                    torch.cuda.current_stream().wait_stream(side)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):outputs[name]=fn()
                    graphs[name]=graph
                samples={name:[] for name in funcs}
                for trial in range(5):
                    names=list(funcs) if trial%2==0 else list(funcs)[::-1]
                    for name in names:
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):graphs[name].replay()
                        end.record();end.synchronize()
                        samples[name].append(start.elapsed_time(end)/10)
                result={name:dict(ms=statistics.median(v),samples_ms=v) for name,v in samples.items()}
                row=dict(phase=phase,b=b,t=t,d=d,h=6,layers=12,vocab=32768,pattern=pattern,capture_order=list(variants),errors=errors,results=result)
                rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
                del graphs,outputs,graph;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
    finally:attention.softplus_attention=original

if __name__=='__main__':main()
