"""Paired schedule ablations; all allocation, reduction, and conversion kernels included."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import os
import tempfile
if "--resources-only" in sys.argv or "--resources" in sys.argv:
    os.environ.setdefault("CUTE_DSL_KEEP", "cubin")
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", tempfile.mkdtemp(prefix="softplus_schedule_cubin_"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from trains.bench_softplus_general import measure
from flash_attn_4.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn_4.softplus_api import softplus_attn_fa4, softplus_attn_fa4_func


def cute_resources():
    import cuda.bindings.driver as cuda
    info=[]
    _,device=cuda.cuDeviceGet(0)
    for label,cache in [('fwd',_flash_attn_fwd.compile_cache),('bwd',_flash_attn_bwd.compile_cache)]:
        for key,fn in cache.cache.items():
            try:
                cubin=fn.__cubin__
                if cubin is None:
                    info.append(dict(kind=label,key=str(key),unavailable="Run --resources-only to retain cubin"))
                    continue
                err,module=cuda.cuModuleLoadData(cubin)
                if int(err):
                    raise RuntimeError(str(err))
                for symbol,attrs in fn.kernel_info.items():
                    err,kernel=cuda.cuModuleGetFunction(module,symbol.encode())
                    if int(err):
                        raise RuntimeError(str(err))
                    entry=dict(kind=label,key=str(key),launch_attributes={str(k):v for k,v in attrs.items()})
                    for field,attr in [('registers','NUM_REGS'),('local_bytes','LOCAL_SIZE_BYTES'),('static_smem','SHARED_SIZE_BYTES')]:
                        err,value=cuda.cuFuncGetAttribute(getattr(cuda.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),kernel)
                        entry[field]=int(value) if int(err)==0 else str(err)
                    # Exact for the tested BF16 M64, D64/128 layouts (all sizes
                    # satisfy the struct's 1024/128-byte alignments). The CUDA
                    # static_smem attribute excludes this dynamic allocation.
                    if label == 'fwd':
                        d,dv,m,n,threads=key[1],key[2],key[26],key[27],key[29]
                        qbytes,kbytes,vbytes=2*m*d,2*n*d,2*n*dv
                        shared=kbytes+(max(qbytes,vbytes) if key[-2] else qbytes+vbytes)
                        entry.update(d=d,m=m,n=n,fragment=key[-3],q_in_regs=key[-2])
                    else:
                        d,dv,m,n,threads=key[2],key[3],key[8],key[9],key[10]
                        sq,so=key[12],key[13]
                        shared=2*(n*d+n*dv+m*d*sq+m*dv*so)+2*4*m*sq+4*m*n
                        entry.update(d=d,m=m,n=n,early_dv=key[-1])
                    entry['dynamic_smem_layout_bytes']=shared
                    entry['threads']=threads
                    err,=cuda.cuFuncSetAttribute(kernel,cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared)
                    if int(err)==0:
                        err,blocks=cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel,threads,shared)
                        if int(err)==0:entry['static_resident_blocks_per_sm']=blocks
                    info.append(entry)
                cuda.cuModuleUnload(module)
            except Exception as e:
                info.append(dict(kind=label,key=str(key),unavailable=str(e)))

    return info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['bwd','fwd','stream','policy'],default='bwd')
    ap.add_argument('--output',required=True)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--resources-only',action='store_true')
    ap.add_argument('--resources',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    paths=['flash_attn_4/softplus_api.py','flash_attn_4/interface.py','flash_attn_4/flash_fwd.py',
           'flash_attn_4/flash_fwd_softplus.py','flash_attn_4/flash_bwd.py',
           'flash_attn_4/flash_bwd_softplus.py','nanochat/softplus_stream_k.py']
    Path(args.output+'.metadata.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,args=vars(args),hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}),indent=2))
    rows=[]
    resources=[]
    shapes=[(1,6,2048,d,w) for d in (64,128) for w in (-1,511)]
    if not args.quick:
        shapes += [(1,6,256,d,-1) for d in (64,128)]
        shapes += [(1,6,8192,d,w) for d in (64,128) for w in (-1,511)]
        shapes += [(8,6,2048,d,w) for d in (64,128) for w in (-1,511)]
        shapes += [(1,1,4096,128,-1),(1,1,16384,128,-1),(2,4,1536,128,255)]
    for b,h,t,d,w in shapes:
        torch.manual_seed(222)
        xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        o=softplus_attn_fa4(*xs,window_size=(w,0))
        g=torch.randn_like(o)
        lse=torch.empty(b,h,t,device='cuda',dtype=torch.float32)
        funcs={}
        if args.phase=='policy':
            xs=[x.requires_grad_() for x in xs]
            funcs['previous']=lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0),early_dv=False),xs,g)
            funcs['auto']=lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0)),xs,g)
        elif args.phase=='bwd':
            for early in (False,True):
                funcs['early_dv' if early else 'previous']=lambda early=early: _flash_attn_bwd(*xs,o,g,lse,
                    causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                    attn_kind='softplus',softplus_early_dv=early)[:3]
        elif args.phase=='fwd':
            for nl,nf,qr in ((128 if d==64 else 64,0,False),(64,32,False),(128,32,False),(128,64,False),(64,32,True),(128,32,True),(128,64,True),(128 if d==64 else 64,0,True)):
                funcs[f'load{nl}_frag{nf}_qreg{int(qr)}']=lambda nl=nl,nf=nf,qr=qr: (_flash_attn_fwd(*xs,
                    causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                    attn_kind='softplus',tile_mn=(64,nl),softplus_fragment_n=nf,softplus_q_in_regs=qr)[0],)
        else:
            from nanochat.softplus_stream_k import stream_k_forward
            funcs['previous']=lambda:(softplus_attn_fa4(*xs,window_size=(w,0)),)
            for bm,bn,warps,waves,stages in ((64,64,4,1,1),(64,64,4,2,1),(64,64,4,2,2),(64,64,4,4,1),(64,64,8,2,1),(64,128,4,2,1),(32,64,4,2,1)):
                funcs[f'm{bm}n{bn}w{warps}waves{waves}stages{stages}']=lambda bm=bm,bn=bn,warps=warps,waves=waves,stages=stages:(
                    stream_k_forward(*xs,window=w,bm=bm,bn=bn,num_warps=warps,waves=waves,num_stages=stages),)
        if args.phase == 'stream' and args.resources_only:
            for stages in (1,2):
                _,kernel=stream_k_forward(*xs,window=w,waves=2,num_stages=stages,return_kernel=True)
                resources.append(dict(b=b,h=h,t=t,d=d,w=w,stages=stages,registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared))
        print('warming',args.phase,b,h,t,d,w,flush=True)
        ref=next(iter(funcs.values()))()
        for name,fn in funcs.items():
            got=fn()
            for x,y in zip(got,ref):
                assert torch.isfinite(x).all()
                assert (x.float()-y.float()).abs().max() < .005*y.float().abs().max().clamp_min(1e-10),(name,b,h,t,d,w)
        row=dict(b=b,h=h,t=t,d=d,w=w,results={} if args.resources_only else measure(funcs,30,False))
        rows.append(row)
        Path(args.output).write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
    Path(args.output+'.resources.json').write_text(json.dumps(resources+cute_resources(),indent=2))

if __name__=='__main__':
    main()
