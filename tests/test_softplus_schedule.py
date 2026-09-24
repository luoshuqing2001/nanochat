"""Schedule coverage, private partial ownership, masks, and gradient regressions."""
import unittest
from collections import Counter
import torch
from nanochat.softplus_stream_k import plan_cpu
from test_softplus_fixed_kv import reference


class TestWorkPlan(unittest.TestCase):
    def test_coverage_and_boundary_storage(self):
        for tq,tk in ((1,1),(65,65),(113,2053),(257,513),(1024,1024)):
            for window in (-1,0,31,111,4096):
                for bm,bn in ((32,64),(64,32),(64,128)):
                    expected=set()
                    for m in range((tq+bm-1)//bm):
                        for ni in range((tk+bn-1)//bn):
                            if ni*bn<min(tk,(m+1)*bm+tk-tq) and (window<0 or (ni+1)*bn>max(0,m*bm+tk-tq-window)):
                                expected.add((m,ni))
                    for workers in (1,2,7,96,10000):
                        offsets,segs,groups,slots=plan_cpu(tq,tk,window,bm,bn,workers)
                        got=Counter((m,n) for m,lo,hi,_ in segs for n in range(lo,hi))
                        self.assertEqual(set(got),expected)
                        self.assertEqual(set(got.values()),{1})
                        work=[sum(s[2]-s[1] for s in segs[a:b]) for a,b in zip(offsets,offsets[1:])]
                        self.assertLessEqual(max(work)-min(work),1)
                        self.assertLessEqual(len(groups),len(work)-1)
                        self.assertLessEqual(slots,2*(len(work)-1))
                        self.assertEqual(sorted(s[3] for s in segs if s[3]>=0),list(range(slots)))
                        for m,first,count in groups:
                            self.assertEqual([s[3] for s in segs if s[0]==m],list(range(first,first+count)))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestSchedulesGPU(unittest.TestCase):
    def check(self,got,ref,tol=.02):
        got,ref=got.detach().float(),ref.detach().float()
        self.assertTrue(torch.isfinite(got).all())
        self.assertLess(float((got-ref).abs().max()),tol*float(ref.abs().max().clamp_min(1e-25)))
        self.assertLess(float((got-ref).square().mean().sqrt()),tol*float(ref.square().mean().sqrt().clamp_min(1e-25)))

    def inputs(self,tq,tk,d,dtype):
        return [torch.randn(2,t,3,3,d,device='cuda',dtype=dtype)[:,:,i].detach().requires_grad_() for i,t in enumerate((tq,tk,tk))]

    def test_fragment_forward(self):
        from flash_attn_4.interface import _flash_attn_fwd
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,a in ((257,257,64,-1,1.),(113,513,128,31,.5),(65,129,64,0,0.),(113,513,128,-1,1.)):
                xs=self.inputs(tq,tk,d,dtype)
                ref=reference(*(x.float() for x in xs),w,a)
                for nl,nf,qr in ((64,32,False),(128,32,True),(128,64,False),(64,0,True)):
                    with self.subTest(dtype=dtype,tq=tq,tk=tk,d=d,w=w,nl=nl,nf=nf,qr=qr):
                        o=_flash_attn_fwd(*xs,causal=w<0,window_size_left=None if w<0 else w,
                            window_size_right=None if w<0 else 0,attn_kind='softplus',softplus_alpha=a,
                            tile_mn=(64,nl),softplus_fragment_n=nf,softplus_q_in_regs=qr)[0]
                        self.check(o,ref)

    def test_early_dv_gradients(self):
        from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,a,chunk in ((257,257,64,-1,1.,None),(113,513,128,31,.5,None),(65,129,64,0,0.,2),(257,257,128,-1,1.,2)):
                xs=self.inputs(tq,tk,d,dtype)
                rb=[x.detach().float().requires_grad_() for x in xs]
                ref=reference(*rb,w,a);g=torch.randn_like(xs[0])
                rg=torch.autograd.grad(ref,rb,g.float())
                kw=dict(causal=w<0,window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,attn_kind='softplus',softplus_alpha=a)
                out=_flash_attn_fwd(*xs,**kw)[0]
                lse=torch.empty(2,3,tq,device='cuda',dtype=torch.float32)
                gg=_flash_attn_bwd(*xs,out,g,lse,**kw,softplus_early_dv=True,softplus_balanced_m_chunk=chunk)[:3]
                for x,y in zip(gg,rg):self.check(x,y)

    def test_compiled_training_schedules(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for options in (dict(stream_waves=2), dict(fragment_n=32, q_in_regs=True), dict()):
            xs=self.inputs(113,257,128,torch.bfloat16)
            rb=[x.detach().float().requires_grad_() for x in xs]
            ref=reference(*rb,31,.5);g=torch.randn_like(xs[0])
            rg=torch.autograd.grad(ref,rb,g.float())
            def fn(q,k,v):
                return softplus_attn_fa4_func(q,k,v,window_size=(31,0),alpha=.5,
                                              early_dv=True,**options)
            compiled=torch.compile(fn,fullgraph=True)
            for run in (fn,compiled):
                out=run(*xs);gg=torch.autograd.grad(out,xs,g)
                for x,y in zip((out,*gg),(ref,*rg)):self.check(x,y)

    def test_automatic_early_dv_compiled(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        xs=[torch.randn(1,2048,2,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        g=torch.randn_like(xs[0])
        expected=softplus_attn_fa4_func(*xs,early_dv=True)
        grads=torch.autograd.grad(expected,xs,g)
        compiled=torch.compile(softplus_attn_fa4_func,fullgraph=True)
        actual=compiled(*xs)
        for x,y in zip((actual,*torch.autograd.grad(actual,xs,g)),(expected,*grads)):
            self.check(x,y,.005)

    def test_negative_early_dv(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for d in (64,128):
            q=torch.ones(1,65,1,d,device='cuda',dtype=torch.bfloat16)
            xs=[x.requires_grad_() for x in (q,torch.full_like(q,-32/d**.5),torch.ones_like(q))]
            rb=[x.detach().float().requires_grad_() for x in xs]
            ref=reference(*rb,-1,1.);rg=torch.autograd.grad(ref.sum(),rb)
            out=softplus_attn_fa4_func(*xs,early_dv=True)
            gg=torch.autograd.grad(out.sum(),xs)
            for x,y in zip((out,*gg),(ref,*rg)):self.check(x,y)

    def test_stream_forward(self):
        from nanochat.softplus_stream_k import stream_k_forward
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,a in ((257,257,64,-1,1.),(113,513,128,31,.5),(65,129,64,0,0.),(113,513,128,-1,1.)):
                xs=self.inputs(tq,tk,d,dtype)
                ref=reference(*(x.float() for x in xs),w,a)
                for workers in (1,7,128):
                    for bm,bn in ((32,64),(64,32),(64,128)):
                        with self.subTest(dtype=dtype,tq=tq,tk=tk,d=d,w=w,workers=workers,bm=bm,bn=bn):
                            o=stream_k_forward(*xs,window=w,alpha=a,workers=workers,bm=bm,bn=bn)
                            self.check(o,ref)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
