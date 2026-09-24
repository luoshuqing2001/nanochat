"""Paired native Softplus ablations; includes preparation and postprocessing."""
import argparse,hashlib,json,sys,os,tempfile
if "--resources-only" in sys.argv:
    os.environ.setdefault("CUTE_DSL_KEEP","cubin")
    os.environ.setdefault("CUTE_DSL_DUMP_DIR",tempfile.mkdtemp(prefix="softplus_native_cubin_"))
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from trains.bench_softplus_general import measure


def resources(cache, before, name, phase, d):
    import cuda.bindings.driver as cuda
    entries=[]
    for key,fn in cache.cache.items():
        if key in before:continue
        err,module=cuda.cuModuleLoadData(fn.__cubin__)
        if int(err):raise RuntimeError(str(err))
        try:
            for symbol in fn.kernel_info:
                err,kernel=cuda.cuModuleGetFunction(module,symbol.encode())
                if int(err):raise RuntimeError(str(err))
                row=dict(variant=name,phase=phase,d=d)
                for field,attr in (("registers","NUM_REGS"),("local_bytes","LOCAL_SIZE_BYTES")):
                    err,value=cuda.cuFuncGetAttribute(getattr(cuda.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),kernel)
                    if int(err):raise RuntimeError(str(err))
                    row[field]=int(value)
                shared=(40960 if d==64 else 49152) if phase=='fwd' else (58368 if d==64 else 99328)
                if phase=='bwd' and name!='previous':shared-=8192
                if phase=='bwd' and name in ('scaled_do','inline'):shared-=1024
                row['dynamic_smem_bytes']=shared
                err,=cuda.cuFuncSetAttribute(kernel,cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared)
                if int(err):raise RuntimeError(str(err))
                err,blocks=cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel,128,shared)
                if int(err):raise RuntimeError(str(err))
                row['resident_blocks_per_sm']=blocks
                entries.append(row)
        finally:cuda.cuModuleUnload(module)
    return entries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--resources-only',action='store_true')
    ap.add_argument('--phase',choices=('bwd','fwd'),default='bwd')
    a=ap.parse_args();torch.set_num_threads(2)
    shapes=[(1,6,2048,d,w) for d in (64,128) for w in (-1,511)]
    if not a.quick:
        shapes += [(b,6,t,d,w) for b,t in ((1,8192),(8,2048)) for d in (64,128) for w in (-1,511)]
        shapes += [(1,5,6144,128,-1),(2,7,2560,128,-1),(1,3,1792,128,383)]
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'flash_attn_4').glob('*.py'))+list((root/'nanochat').glob('softplus*.py'))
    metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,phase=a.phase,
                  timing=None if a.resources_only else "median of 3 CUDA graph trials, 30 ms each; preparation and finish included",
                  resources_only=a.resources_only,
                  hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    Path(a.output+'.metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    resource_rows=[]
    for b,h,t,d,w in shapes:
        torch.manual_seed(731)
        xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        kw=dict(causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,attn_kind='softplus')
        o=_flash_attn_fwd(*xs,**kw)[0];g=torch.randn_like(o);lse=torch.empty(b,h,t,device='cuda')
        funcs={}
        if a.phase=='bwd':
            for name,native,share,early,inline in (('previous',False,False,d==128 and w<0,False),('share_pd',False,True,True,False),('scaled_do',True,True,True,False),('inline',False,True,True,True)):
                funcs[name]=lambda native=native,share=share,early=early,inline=inline:_flash_attn_bwd(*xs,o,g,lse,**kw,
                    softplus_native_bwd=native,softplus_share_pd=share,softplus_early_dv=early,softplus_inline_scale=inline)[:3]
        else:
            workers=max(1,(2*torch.cuda.get_device_properties().multi_processor_count+b*h-1)//(b*h))
            opts={'previous':{},'estrin':{'softplus_poly_estrin':True},
                  'warp_overlap':{'softplus_warp_overlap':True},
                  'cute_stream':{'softplus_stream_workers':workers},
                  'cute_tail':{'softplus_stream_workers':workers,'softplus_stream_tail':True}}
            for name,options in opts.items():
                funcs[name]=lambda options=options:(_flash_attn_fwd(*xs,**kw,tile_mn=(64,128 if d==64 else 64),**options)[0],)
        print('warming',b,h,t,d,w,flush=True)
        if a.resources_only:
            cache=(_flash_attn_bwd if a.phase=='bwd' else _flash_attn_fwd).compile_cache
            # Forward's initial O already warmed the baseline.
            torch.cuda.synchronize()
            cache.cache.clear()
            for name,fn in funcs.items():
                before=set(cache.cache)
                fn();torch.cuda.synchronize()
                resource_rows.extend(dict(x,b=b,h=h,t=t,w=w) for x in resources(cache,before,name,a.phase,d))
            Path(a.output).write_text(json.dumps(resource_rows,indent=2))
            continue
        ref=funcs['previous']();errors={}
        for name,fn in funcs.items():
            errors[name]=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(fn(),ref)]
            assert max(errors[name])<.025,errors
        row=dict(b=b,h=h,t=t,d=d,w=w,errors=errors,results=measure(funcs,30,False))
        rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
