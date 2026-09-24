"""Direct Softplus approximation: run each mode in a fresh process."""
import argparse,os,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--mode',type=int,choices=(0,1,2,3,4,5,6),required=True)
p.add_argument('--output',required=True)
p.add_argument('--check-only',action='store_true')
p.add_argument('--quick',action='store_true')
a=p.parse_args()
os.environ['FA4_SOFTPLUS_DIRECT']=str({3:0,4:1,5:0,6:0}.get(a.mode,a.mode))
os.environ['FA4_SOFTPLUS_PACKED']='1' if a.mode in (3,4) else '0'
os.environ['FA4_SOFTPLUS_LOG_DEGREE']=str({5:3,6:4}.get(a.mode,5))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func
from trains.bench_softplus_general import measure,TunedSoftmax

torch.set_num_threads(2)
torch.manual_seed(991)

def reference(xs,w=-1):
 q,k,v=[x.float().transpose(1,2) for x in xs]
 s=q@k.transpose(-1,-2)*q.shape[-1]**-.5
 qi=torch.arange(q.shape[2],device=q.device)[:,None]+k.shape[2]-q.shape[2]
 kj=torch.arange(k.shape[2],device=q.device)[None,:]
 mask=kj<=qi
 if w>=0:mask=mask & (kj>=qi-w)
 weights=F.softplus(s).masked_fill(~mask,0)/mask.sum(-1).clamp_min(1)[None,None,:,None]
 return (weights@v).transpose(1,2)

checks=[]
for dtype in (torch.bfloat16,torch.float16):
 # Single-key batches expose pointwise values and derivatives, including tails.
 scores=torch.cat([torch.linspace(-10,10,513),torch.tensor([-80.,-32.,-8.001,-8.,-7.999,-4.001,-4.,-3.999,0.,3.999,4.,4.001,7.999,8.,8.001,32.,80.])]).to('cuda')
 q=torch.ones(len(scores),1,1,64,device='cuda',dtype=dtype)
 k=(scores[:,None,None,None]/8).expand_as(q).to(dtype).contiguous()
 xs=[x.requires_grad_() for x in (q,k,torch.ones_like(q))]
 rb=[x.detach().float().requires_grad_() for x in xs]
 ref=reference(rb);out=softplus_attn_fa4_func(*xs,num_splits=1,bwd_schedule='previous')
 g=torch.ones_like(out);rg=torch.autograd.grad(ref,rb,g.float());gg=torch.autograd.grad(out,xs,g)
 # FP16 tiny P/dS already underflows; absolute allowance is one output subnormal
 # scaled by the reduction extent for gradients.
 for x,y in zip((out,*gg),(ref,*rg)):
  torch.testing.assert_close(x.float(),y,rtol=.012,atol=4e-6 if dtype==torch.float16 else 1e-12)
 checks.append(dict(dtype=str(dtype),pointwise_relative=float(((out.float()-ref).abs()/ref.abs().clamp_min(1e-4)).max().detach())))
 for d,w,scale in ((64,-1,1.),(128,31,1.),(64,-1,8.)):
  xs=[(torch.randn(1,129,3,2,d,device='cuda',dtype=dtype)[:,:,i]*scale).requires_grad_() for i in range(3)]
  rb=[x.detach().float().requires_grad_() for x in xs];ref=reference(rb,w);g=torch.randn_like(xs[0])
  out=softplus_attn_fa4_func(*xs,window_size=(w,0));rg=torch.autograd.grad(ref,rb,g.float());gg=torch.autograd.grad(out,xs,g)
  for x,y in zip((out,*gg),(ref,*rg)):
   err=float(((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)).detach())
   assert err<.025,(dtype,d,w,scale,err)
 # Ensure the environment-selected kernel also survives compiled autograd.
 fn=torch.compile(lambda q,k,v:softplus_attn_fa4_func(q,k,v),fullgraph=True)
 out=fn(*xs);torch.autograd.grad(out,xs,g)

data=dict(mode=a.mode,gpu=torch.cuda.get_device_name(),checks=checks,rows=[],
          source_hash=hashlib.sha256(Path('flash_attn_4/softplus.py').read_bytes()).hexdigest())
Path(a.output).write_text(json.dumps(data,indent=2))
print('checks passed',checks,flush=True)
if a.check_only:sys.exit(0)
shapes=[(1,6,2048,d,w) for d in (64,128) for w in (-1,511)]
if not a.quick:
 shapes += [(b,6,t,d,w) for b,t in ((1,8192),(8,2048)) for d in (64,128) for w in (-1,511)]
 shapes += [(2,5,3584,64,-1),(1,5,6144,128,-1),(1,3,1792,128,383)]
for b,h,t,d,w in shapes:
 torch.manual_seed(731)
 xs=[torch.randn(b,t,h,d,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
 g=torch.randn_like(xs[0])
 funcs={'prefill':lambda:softplus_attn_fa4(*xs,window_size=(w,0),num_splits='auto'),
        'train':lambda:torch.autograd.grad(softplus_attn_fa4_func(*xs,window_size=(w,0)),xs,g),
        'softmax_prefill':lambda:TunedSoftmax.apply(*xs,w),
        'softmax_train':lambda:torch.autograd.grad(TunedSoftmax.apply(*xs,w),xs,g)}
 row=dict(b=b,h=h,t=t,d=d,w=w,results=measure(funcs,30,False))
 data['rows'].append(row);Path(a.output).write_text(json.dumps(data,indent=2));print(json.dumps(row),flush=True)
