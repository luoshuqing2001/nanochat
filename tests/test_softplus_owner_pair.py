"""Sequential complete-owner pairs: masks, gradients, and shared-memory reuse."""
import unittest
import torch
from test_softplus_fixed_kv import reference

class TestPairCoverage(unittest.TestCase):
    def test_ownership(self):
        for m in (1,2,3,31,32,33,512,1024):
            jobs=[(m-1-p,p) if m-1-p!=p else (p,) for p in range((m+1)//2)]
            self.assertEqual(sorted(x for job in jobs for x in job),list(range(m)))
            if m%2==0:self.assertEqual(len({sum(x+1 for x in job) for job in jobs}),1)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestOwnerPair(unittest.TestCase):
    def test_forward_grad(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        from flash_attn_4.interface import _flash_attn_fwd
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d in ((1,1,64),(65,129,64),(129,257,128),(256,256,128)):
                xs=[torch.randn(2,t,3,2,d,device='cuda',dtype=dtype)[:,:,i].requires_grad_() for i,t in enumerate((tq,tk,tk))]
                rb=[x.detach().float().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
                expected=reference(*rb,-1,.5);rg=torch.autograd.grad(expected,rb,g.float())
                fn=lambda q,k,v:softplus_attn_fa4_func(q,k,v,alpha=.5,owner_pair=True)
                torch._dynamo.reset()
                out=torch.compile(fn,fullgraph=True)(*xs);grads=torch.autograd.grad(out,xs,g)
                for x,y in zip((out,*grads),(expected,*rg)):
                    self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)
                kw=dict(causal=True,tile_mn=(64,128 if d==64 else 64),return_lse=True)
                old=_flash_attn_fwd(*xs,**kw);new=_flash_attn_fwd(*xs,sm120_owner_pair=True,**kw)
                for x,y in zip(old[:2],new[:2]):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)

    def test_replay_and_streams(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4
        torch.set_num_threads(2)
        xs=[torch.randn(1,769,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        fn=lambda:softplus_attn_fa4(*xs,owner_pair=True)
        for _ in range(3):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=fn()
        for _ in range(20):graph.replay()
        torch.cuda.synchronize();torch.testing.assert_close(actual,softplus_attn_fa4(*xs,num_splits=1),rtol=.02,atol=1e-4)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()];outs=[]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):outs.append(fn())
        for stream in streams:stream.synchronize()
        for out in outs:torch.testing.assert_close(out,actual,rtol=0,atol=0)

    def test_invalid_combinations(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4
        q=torch.randn(1,129,1,64,device='cuda',dtype=torch.bfloat16)
        for kw in (dict(whole_lpt=True),dict(kv_split_size=1024),dict(window_size=(31,0))):
            with self.assertRaises(ValueError):softplus_attn_fa4(q,q,q,owner_pair=True,**kw)

    def test_backward_pair(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
        torch.set_num_threads(2)
        for dtype,d,tq,tk in ((torch.bfloat16,64,129,257),(torch.float16,128,193,257),(torch.bfloat16,128,513,513)):
            xs=[torch.randn(1,t,2,d,device='cuda',dtype=dtype).requires_grad_() for t in (tq,tk,tk)]
            rb=[x.detach().float().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
            expected=reference(*rb,-1,1.);rg=torch.autograd.grad(expected,rb,g.float())
            for mode in ('paired','paired_shared'):
                torch._dynamo.reset()
                fn=torch.compile(lambda q,k,v:softplus_attn_fa4_func(q,k,v,owner_pair=True,bwd_schedule=mode),fullgraph=True)
                out=fn(*xs);grads=torch.autograd.grad(out,xs,g)
                for x,y in zip((out,*grads),(expected,*rg)):
                    self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)
            out,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True,sm120_owner_pair=True)
            old=_flash_attn_bwd(*xs,out,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1))[:3]
            new=_flash_attn_bwd(*xs,out,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1),sm120_owner_pair=True)[:3]
            for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=.02,atol=1e-3)

    def test_head_local_lpt(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        from flash_attn_4.interface import _flash_attn_fwd
        torch.set_num_threads(2)
        for d in (64,128):
            xs=[torch.randn(2,t,3,2,d,device='cuda',dtype=torch.bfloat16)[:,:,i].requires_grad_() for i,t in enumerate((129,257,257))];g=torch.randn_like(xs[0])
            old=softplus_attn_fa4_func(*xs,num_splits=1);expected=torch.autograd.grad(old,xs,g)
            fn=torch.compile(lambda q,k,v:softplus_attn_fa4_func(q,k,v,head_lpt=True),fullgraph=True)
            out=fn(*xs);grads=torch.autograd.grad(out,xs,g)
            for x,y in zip((out,*grads),(old,*expected)):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
            kw=dict(causal=True,return_lse=True,tile_mn=(64,128 if d==64 else 64))
            old=_flash_attn_fwd(*xs,**kw);new=_flash_attn_fwd(*xs,sm120_head_lpt=True,**kw)
            for x,y in zip(old[:2],new[:2]):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
            with self.assertRaises(ValueError):softplus_attn_fa4_func(*xs,head_lpt=True,owner_pair=True)
