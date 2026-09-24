"""Global scheduling and non-spinning last-completer regression tests."""
import unittest
from collections import Counter
import torch
from test_softplus_fixed_kv import reference


class TestGlobalPlan(unittest.TestCase):
    def test_coverage_and_slots(self):
        from flash_attn_4.softplus_stream import global_plan_cpu
        from nanochat.softplus_stream_k import plan_cpu
        for tq,tk in ((1,1),(113,513),(257,257),(4096,4096)):
            for w in (-1,0,511):
                for bh in (1,6,48):
                    offsets,segs,groups,slots=global_plan_cpu(tq,tk,w,64,64,48,bh)
                    _,whole,_,_=plan_cpu(tq,tk,w,64,64,1)
                    expand=lambda xs:Counter((m,n) for m,lo,hi,_ in xs for n in range(lo,hi))
                    self.assertEqual(expand(segs),expand(whole))
                    self.assertEqual(offsets,tuple(range(len(segs)+1)))
                    self.assertEqual(sorted(s for *_,s in segs if s>=0),list(range(slots)))
                    for m,first,n in groups:
                        self.assertLessEqual(n,4)
                        self.assertEqual(sorted(s for mm,_,_,s in segs if mm==m),list(range(first,first+n)))
                    if w>=0:self.assertEqual(slots,0)

    def test_split_is_exercised(self):
        from flash_attn_4.softplus_stream import global_plan_cpu
        self.assertGreater(global_plan_cpu(4096,4096,-1,64,64,48,1)[3],0)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestCompletion(unittest.TestCase):
    def test_forward(self):
        from flash_attn_4.interface import _flash_attn_fwd
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,alpha in ((129,257,64,-1,1.),(113,513,128,31,.5),(65,129,64,0,0.),(257,257,128,-1,1.)):
                xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,i] for i,t in enumerate((tq,tk,tk))]
                ref=reference(*(x.float() for x in xs),w,alpha)
                for opts in (dict(softplus_stream_workers=3),dict(softplus_stream_workers=17,softplus_stream_tiles=True),dict(softplus_stream_workers=48,softplus_stream_tiles=True,softplus_stream_global=True)):
                    out=_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,attn_kind='softplus',softplus_alpha=alpha,softplus_stream_complete=True,**opts)[0]
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertLess(float((out.float()-ref).abs().max()/ref.abs().max().clamp_min(1e-20)),.025)

    def test_compile_grad(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.set_num_threads(2)
        for global_order in (False,True):
            xs=[torch.randn(1,257,2,64,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
            rb=[x.detach().float().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
            ref=reference(*rb,-1,1.);rg=torch.autograd.grad(ref,rb,g.float())
            def fn(q,k,v):return softplus_attn_fa4_func(q,k,v,cute_stream_waves=2,stream_tiles=True,stream_global=global_order,stream_complete=True,bwd_schedule="auto" if global_order else "kv_group2")
            for run in (fn,torch.compile(fn,fullgraph=True)):
                out=run(*xs);grads=torch.autograd.grad(out,xs,g)
                for actual,expected in zip((out,*grads),(ref,*rg)):
                    self.assertLess(float((actual.float()-expected).abs().max()/expected.abs().max().clamp_min(1e-20)),.025)

    def test_whole_lpt(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        from trains.bench_softplus_completion import GlobalSoftmax
        from trains.bench_softplus_general import TunedSoftmax
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            xs=[torch.randn(2,t,3,2,64,device='cuda',dtype=dtype)[:,:,i].requires_grad_() for i,t in enumerate((129,257,257))]
            g=torch.randn_like(xs[0])
            out=softplus_attn_fa4_func(*xs,num_splits=1);expected=torch.autograd.grad(out,xs,g)
            run=torch.compile(lambda q,k,v:softplus_attn_fa4_func(q,k,v,whole_lpt=True),fullgraph=True)
            got=run(*xs);grads=torch.autograd.grad(got,xs,g)
            for x,y in zip((got,*grads),(out,*expected)):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)
            old=TunedSoftmax.apply(*xs,-1);ref=torch.autograd.grad(old,xs,g)
            new=GlobalSoftmax.apply(*xs,-1);actual=torch.autograd.grad(new,xs,g)
            for x,y in zip((new,*actual),(old,*ref)):torch.testing.assert_close(x,y,rtol=.02,atol=1e-4)

    def test_kv_group2(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w in ((129,257,64,-1),(113,513,64,31)):
                xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,i].requires_grad_() for i,t in enumerate((tq,tk,tk))]
                rb=[x.detach().float().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
                ref=reference(*rb,w,1.);rg=torch.autograd.grad(ref,rb,g.float())
                out=softplus_attn_fa4_func(*xs,window_size=(w,0),bwd_schedule="kv_group2")
                grads=torch.autograd.grad(out,xs,g)
                for index,(actual,expected) in enumerate(zip((out,*grads),(ref,*rg))):
                    self.assertLess(float((actual.float()-expected).abs().max()/expected.abs().max().clamp_min(1e-20)),.025,(dtype,tq,tk,d,w,index))

    def test_replay_and_concurrent_streams(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.set_num_threads(2)
        # Both a many-split persistent plan and the global split plan are exercised.
        for t,h,opts in ((257,2,{}),(4096,1,dict(stream_tiles=True,stream_global=True))):
            xs=[torch.randn(1,t,h,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            fn=lambda:softplus_attn_fa4_func(*xs,cute_stream_waves=2,stream_complete=True,**opts)
            ref=softplus_attn_fa4_func(*xs,num_splits=1)
            fn();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            for _ in range(30):graph.replay()
            torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=.02,atol=1e-4)
            streams=[torch.cuda.Stream(),torch.cuda.Stream()];outs=[]
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):outs.append(fn())
            for stream in streams:torch.cuda.current_stream().wait_stream(stream)
            for out in outs:torch.testing.assert_close(out,ref,rtol=.02,atol=1e-4)

if __name__=='__main__':unittest.main()
