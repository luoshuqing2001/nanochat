"""Final complete attention forward+backward check for two-pass backward."""
import collections,hashlib,json,os,re,statistics,subprocess,sys
from pathlib import Path
os.environ['CUTE_DSL_KEEP']='cubin'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn_4.softplus_api import softplus_attn_fa4,_op_bwd_fa4
from flash_attn_4.softplus_split_backward import softplus_split_backward
from flash_attn_4.interface import _flash_attn_bwd
from trains.bench_softplus_owner_pair_stable import timed
torch.set_num_threads(2);rows=[]
for t in (32768,65536):
    torch.manual_seed(773)
    xs=[torch.randn(1,t,6,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)];g=torch.randn_like(xs[0])
    def run(split):
        o=softplus_attn_fa4(*xs,head_lpt=True)
        gg=softplus_split_backward(*xs,o,g) if split else _op_bwd_fa4(*xs,o,g,-1,1.,128**-.5,0,True,7)
        return o,*gg
    old=run(False);new=run(True)
    errs=[float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)) for x,y in zip(new,old)]
    assert max(errs)<.005
    samples={n:[] for n in ('fused','split')}
    for trial in range(3):
        for name in (list(samples) if trial%2==0 else list(samples)[::-1]):samples[name].append(timed(lambda:run(name=='split')))
    rows.append(dict(t=t,errors=errs,results={n:dict(ms=statistics.median(v),samples=v) for n,v in samples.items()}))
    Path('trains/softplus_split_train.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
assembly=[]
for i,fn in enumerate(_flash_attn_bwd.compile_cache.cache.values()):
    p=Path(f'/tmp/softplus_split_{i}.cubin');p.write_bytes(fn.__cubin__)
    sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(p)],capture_output=True,text=True,check=True).stdout
    counts=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',sass))
    assembly.append(dict(symbols=list(fn.kernel_info),static_sass_counts=dict(counts),cubin_sha256=hashlib.sha256(fn.__cubin__).hexdigest()))
Path('trains/softplus_split_sass.json').write_text(json.dumps(assembly,indent=2))
