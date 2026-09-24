"""Native backward and polynomial evaluation regressions."""
import unittest
import torch
from test_softplus_fixed_kv import reference


class TestNativePlan(unittest.TestCase):
    def test_hybrid_tile_coverage(self):
        from collections import Counter
        from flash_attn_4.softplus_stream import tile_plan_cpu
        from nanochat.softplus_stream_k import plan_cpu
        for tq,tk in ((1,1),(113,513),(257,257),(2053,4097)):
            for w in (-1,0,31,511):
                for workers in (1,3,17,128):
                    offsets,segs,groups,slots=tile_plan_cpu(tq,tk,w,64,64,workers)
                    _,ref,_,_=plan_cpu(tq,tk,w,64,64,1)
                    expand=lambda xs:Counter((m,n) for m,lo,hi,_ in xs for n in range(lo,hi))
                    self.assertEqual(expand(segs),expand(ref))
                    self.assertEqual(offsets,tuple(range(len(segs)+1)))
                    self.assertEqual(sorted(slot for *_,slot in segs if slot>=0),list(range(slots)))
                    for m,first,count in groups:
                        self.assertEqual(sorted(x[3] for x in segs if x[0]==m),list(range(first,first+count)))

    def test_atomic_plan_slots(self):
        from flash_attn_4.softplus_stream import build_stream_plan
        for tiles in (False,True):
            for tq,tk,w in ((129,257,-1),(2053,4097,511),(1,1,0)):
                table,groups,slots,nw=build_stream_plan(tq,tk,w,64,64,17,False,'cpu',True,tiles)
                segments=table[nw:].tolist()
                self.assertEqual(slots,len(groups))
                mapping={m:slot for m,slot,count in groups.tolist()}
                self.assertTrue(all(count==1 for _,_,count in groups.tolist()))
                for m,lo,hi,slot in segments:
                    self.assertEqual(slot,mapping.get(m,-1))

    def test_tail_coverage(self):
        from collections import Counter
        from flash_attn_4.softplus_stream import stream_plan_cpu
        from nanochat.softplus_stream_k import plan_cpu
        for tq,tk in ((1,1),(113,513),(257,257),(2053,4097)):
            for w in (-1,0,31,511):
                for workers in (1,3,17):
                    for tail in (False,True):
                        offsets,segs,groups,slots=stream_plan_cpu(tq,tk,w,64,64,workers,tail)
                        _,ref,_,_=plan_cpu(tq,tk,w,64,64,1)
                        expand=lambda xs:Counter((m,n) for m,lo,hi,_ in xs for n in range(lo,hi))
                        self.assertEqual(expand(segs),expand(ref))
                        self.assertEqual(offsets[-1],len(segs))
                        for m,first,count in groups:
                            self.assertEqual([x[3] for x in segs if x[0]==m],list(range(first,first+count)))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestNativeBackward(unittest.TestCase):
    def test_forward_candidates(self):
        from flash_attn_4.interface import _flash_attn_fwd
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,a in ((129,257,64,-1,1.),(113,513,128,31,.5),(65,129,64,0,0.),(257,257,128,-1,1.)):
                xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,i] for i,t in enumerate((tq,tk,tk))]
                ref=reference(*(x.float() for x in xs),w,a)
                for opts in (dict(softplus_poly_estrin=True),dict(softplus_warp_overlap=True),dict(softplus_stream_workers=3),
                             dict(softplus_stream_workers=3,softplus_stream_tail=True),dict(softplus_stream_workers=1),
                             dict(softplus_stream_workers=3,softplus_stream_atomic=True),
                             dict(softplus_stream_workers=3,softplus_stream_atomic=True,softplus_stream_tail=True),
                             dict(softplus_stream_workers=17,softplus_stream_atomic=True,softplus_stream_tiles=True),
                             dict(softplus_stream_workers=17,softplus_stream_tiles=True)):
                    with self.subTest(dtype=dtype,tq=tq,d=d,w=w,opts=opts):
                        out=_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,
                            window_size_right=None if w<0 else 0,attn_kind='softplus',softplus_alpha=a,
                            tile_mn=(64,128 if d==64 else 64),**opts)[0]
                        self.assertTrue(torch.isfinite(out).all())
                        self.assertLess(float((out.float()-ref).abs().max()/ref.abs().max().clamp_min(1e-20)),.02)

    def test_compiled_api(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.set_num_threads(2)
        for opts in (dict(bwd_schedule="shared", cute_stream_waves=1, stream_atomic=True, stream_tiles=True),
                     dict(bwd_schedule="shared", cute_stream_waves=1, stream_tiles=True),
                     dict(bwd_schedule="shared", cute_stream_waves=1, stream_atomic=True),
                     dict(bwd_schedule="inline", cute_stream_waves=1, stream_tail=True, stream_atomic=True),
                     dict(bwd_schedule="shared", cute_stream_waves=1),
                     dict(bwd_schedule="inline", cute_stream_waves=1, stream_tail=True),
                     dict(bwd_schedule="scaled", poly_estrin=True),
                     dict(bwd_schedule="shared", warp_overlap=True)):
            xs=[torch.randn(1,129,2,64,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
            rb=[x.detach().float().requires_grad_() for x in xs]
            ref=reference(*rb,-1,1.);g=torch.randn_like(xs[0]);rg=torch.autograd.grad(ref,rb,g.float())
            def fn(q,k,v):return softplus_attn_fa4_func(q,k,v,**opts)
            for run in (fn,torch.compile(fn,fullgraph=True)):
                out=run(*xs);gg=torch.autograd.grad(out,xs,g)
                for x,y in zip((out,*gg),(ref,*rg)):
                    self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)

    def test_atomic_graph_replay(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for tiles in (False,True):
            xs=[torch.randn(1,257,2,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            fn=lambda:softplus_attn_fa4_func(*xs,cute_stream_waves=2,stream_atomic=True,stream_tiles=tiles)
            expected=fn().clone()
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fn();fn()
            torch.cuda.current_stream().wait_stream(side)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            for _ in range(5):graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out,expected,rtol=.02,atol=1e-4)

    def test_default_shared(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        xs=[torch.randn(1,2048,2,64,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        g=torch.randn_like(xs[0])
        expected=softplus_attn_fa4_func(*xs,bwd_schedule="previous")
        rg=torch.autograd.grad(expected,xs,g)
        out=torch.compile(softplus_attn_fa4_func,fullgraph=True)(*xs)
        for x,y in zip((out,*torch.autograd.grad(out,xs,g)),(expected,*rg)):
            self.assertLess(float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)),.005)

    def test_negative_and_small_grad(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for dtype in (torch.bfloat16,torch.float16):
            for score,grad in ((-12.,1.),(0.,1e-4)):
                q=torch.ones(1,129,1,64,device='cuda',dtype=dtype)
                xs=[x.requires_grad_() for x in (q,torch.full_like(q,score/8),torch.ones_like(q))]
                rb=[x.detach().float().requires_grad_() for x in xs]
                ref=reference(*rb,-1,1.);g=torch.full_like(q,grad);rg=torch.autograd.grad(ref,rb,g.float())
                old=softplus_attn_fa4_func(*xs,bwd_schedule="previous")
                old_grads=torch.autograd.grad(old,xs,g)
                for mode in ("shared","inline"):
                    out=softplus_attn_fa4_func(*xs,bwd_schedule=mode)
                    gg=torch.autograd.grad(out,xs,g)
                    # FP16 normalized tiny P/dS already underflows in the previous
                    # implementation. Require no extra error from the storage change.
                    targets=(old,*old_grads) if dtype==torch.float16 else (ref,*rg)
                    for x,y in zip((out,*gg),targets):
                        torch.testing.assert_close(x.float(),y.float(),rtol=.04,atol=6e-8 if dtype==torch.float16 else 1e-10)

    def test_gradients(self):
        from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
        torch.set_num_threads(2)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,a in ((129,257,64,-1,1.),(113,513,128,31,.5),(65,129,64,0,0.),(257,257,128,-1,1.)):
                xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,i] for i,t in enumerate((tq,tk,tk))]
                rb=[x.float().requires_grad_() for x in xs]
                ref=reference(*rb,w,a);g=torch.randn_like(xs[0])
                rg=torch.autograd.grad(ref,rb,g.float())
                kw=dict(causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                        attn_kind='softplus',softplus_alpha=a)
                out=_flash_attn_fwd(*xs,**kw)[0]
                lse=torch.empty(1,2,tq,device='cuda')
                for mode in (0,1,2):
                    native=mode==1
                    with self.subTest(dtype=dtype,tq=tq,tk=tk,d=d,w=w,native=native):
                        got=_flash_attn_bwd(*xs,out,g,lse,**kw,softplus_native_bwd=native,softplus_share_pd=True,softplus_inline_scale=mode==2)[:3]
                        for actual,expected in zip(got,rg):
                            self.assertTrue(torch.isfinite(actual).all())
                            err=(actual.float()-expected).abs().max()/expected.abs().max().clamp_min(1e-20)
                            self.assertLess(float(err),.025)

if __name__=='__main__':unittest.main()
