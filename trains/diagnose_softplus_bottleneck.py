"""Matched schedule, math ablations, resource evidence. Ablations are NOT attention implementations."""
import argparse,collections,hashlib,json,os,re,statistics,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['baseline','log3','no_log','no_sigmoid'],default='baseline');ap.add_argument('--output',required=True);ap.add_argument('--profile',action='store_true');ap.add_argument('--only-long',action='store_true');a=ap.parse_args()
os.environ['CUTE_DSL_KEEP']='cubin'
os.environ['FA4_SOFTPLUS_EXACT']='0';os.environ['FA4_SOFTPLUS_DIRECT']='0';os.environ['FA4_SOFTPLUS_PACKED']='0';os.environ['FA4_SOFTPLUS_LOG_DEGREE']='3' if a.mode=='log3' else '5'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32
import flash_attn_4.softplus as sp
from flash_attn_4.softplus import _stable_softplus

@cute.jit
def no_log(y,zero,estrin:cutlass.Constexpr=False):
    return zero.reshape(y.shape)

@cute.jit
def no_sigmoid(x,zero,estrin:cutlass.Constexpr=False):
    zero=zero.reshape(x.shape)
    value=_stable_softplus(x,zero,estrin)
    # Preserve mask validity. This is intentionally not the Softplus derivative.
    sig=cute.where(x == -Float32.inf,zero,zero+1.0)
    return value,sig

if a.mode=='no_log':sp._log1p_poly=no_log
if a.mode=='no_sigmoid':sp._stable_pair=no_sigmoid
from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
from flash_attn_4.softplus_api import softplus_attn_fa4,_op_bwd_fa4
from trains.bench_softplus_owner_pair_stable import timed


def collect_resources(fn,tag):
    import cuda.bindings.driver as cu
    out=[];cubin=fn.__cubin__
    if cubin is None:return [dict(tag=tag,error='no retained cubin')]
    folder=Path('/tmp/softplus_bottleneck_cubins');folder.mkdir(exist_ok=True)
    path=folder/(a.mode+'_'+tag+'.cubin');path.write_bytes(cubin)
    result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],capture_output=True,text=True,check=True)
    path.with_suffix('.sass').write_text(result.stdout)
    counts=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',result.stdout))
    err,module=cu.cuModuleLoadData(cubin)
    if int(err):raise RuntimeError(err)
    try:
        for symbol,attrs in fn.kernel_info.items():
            err,kernel=cu.cuModuleGetFunction(module,symbol.encode())
            if int(err):raise RuntimeError(err)
            row=dict(tag=tag,symbol=symbol,static_sass_counts=dict(counts),cubin_sha256=hashlib.sha256(cubin).hexdigest())
            for field,attr in [('registers','NUM_REGS'),('local_bytes','LOCAL_SIZE_BYTES'),('static_smem','SHARED_SIZE_BYTES')]:
                err,value=cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+attr),kernel)
                row[field]=int(value) if int(err)==0 else str(err)
            out.append(row)
    finally:cu.cuModuleUnload(module)
    return out


def profile(fn):
    # Event durations are attribution evidence, not the stable timing baseline.
    from torch.profiler import profile as profiler,ProfilerActivity
    fn();torch.cuda.synchronize()
    try:
        with profiler(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
            for _ in range(3):fn()
            torch.cuda.synchronize()
        groups={}
        for ev in p.events():
            if 'CUDA' in str(ev.device_type):
                name=ev.name;v=groups.setdefault(name,dict(count=0,total_us=0.))
                v['count']+=1;v['total_us']+=ev.time_range.elapsed_us()
        return groups
    except Exception as e:return dict(error=str(e))


def main():
    torch.set_num_threads(2);rows=[];resources=[]
    for t,d in (((65536,128),) if a.only_long else ((32768,64),(32768,128),(65536,128))):
        torch.manual_seed(591);xs=[torch.randn(1,t,6,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)];g=torch.randn_like(xs[0]);scale=d**-.5
        fwd=dict(softplus=lambda:softplus_attn_fa4(*xs,head_lpt=True),softmax=lambda:_flash_attn_fwd(*xs,causal=True,tile_mn=(64,128 if d==64 else 64),sm120_head_lpt=True)[0])
        o=fwd['softplus']();so,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True,tile_mn=(64,128 if d==64 else 64),sm120_head_lpt=True)
        bwd={}
        for name,early,mode in [('sp_plain',False,0),('sp_early',True,0),('sp_shared',True,1)]:
            bwd[name]=lambda early=early,mode=mode:_op_bwd_fa4(*xs,o,g,-1,1.,scale,0,early,mode)
        bwd['softmax']=lambda:_flash_attn_bwd(*xs,so,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1))[:3]
        for phase,funcs,cache in [('prefill',fwd,_flash_attn_fwd.compile_cache),('backward',bwd,_flash_attn_bwd.compile_cache)]:
            print('warming',a.mode,phase,t,d,flush=True)
            # Record newly compiled main kernels per candidate, not preprocess helpers.
            for name,fn in funcs.items():
                before=set(cache.cache);out=fn();torch.cuda.synchronize()
                for x in out if isinstance(out,tuple) else (out,):assert torch.isfinite(x).all()
                for key in set(cache.cache)-before:resources.extend(collect_resources(cache.cache[key],f'{phase}_{name}_{t}_{d}'))
                del out
            # Forward kernels were already warmed to obtain backward inputs.
            if phase=='prefill':
                for key,fn in cache.cache.items():
                    tag=f'prefill_cached_{t}_{d}_{len(resources)}';resources.extend(collect_resources(fn,tag))
            samples={n:[] for n in funcs};names=list(funcs)
            for trial in range(3):
                for n in (names if trial%2==0 else names[::-1]):samples[n].append(timed(funcs[n]))
            row=dict(mode=a.mode,phase=phase,t=t,h=6,d=d,results={n:dict(ms=statistics.median(v),samples_ms=v) for n,v in samples.items()})
            if a.profile and phase=='backward':row['profile']={n:profile(fn) for n,fn in funcs.items()}
            rows.append(row);Path(a.output).write_text(json.dumps(dict(rows=rows,resources=resources),indent=2));print(json.dumps(row),flush=True)
    paths=[Path('flash_attn_4')/p for p in ('softplus.py','softmax.py','flash_bwd_softplus.py','flash_bwd.py','flash_fwd.py','interface.py')]+[Path(__file__)]
    Path(a.output+'.metadata.json').write_text(json.dumps(dict(mode=a.mode,mathematically_valid=a.mode in ('baseline','log3'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,timing='same head-local forward schedule, backward unsplit owners; 3 alternating graph trials, 20ms warmup/60ms target; entire operator included',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))

if __name__=='__main__':main()
