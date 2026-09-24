"""General attention optimization regressions, including packed QKV gradients."""
import unittest
from unittest.mock import patch
import torch
from test_softplus_fixed_kv import reference


@unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
class TestGeneralSoftplus(unittest.TestCase):
    def check(self,got,ref,tol=.02):
        got,ref=got.detach().float(),ref.detach().float()
        self.assertTrue(torch.isfinite(got).all())
        diff=got-ref
        self.assertLess(float(diff.abs().max()),tol*float(ref.abs().max().clamp_min(1e-20)))
        self.assertLess(float(diff.square().mean().sqrt()),tol*float(ref.square().mean().sqrt().clamp_min(1e-20)))

    def test_packed_hybrid_and_triton_backward(self):
        import nanochat.softplus_attention as sa
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for dtype in (torch.bfloat16,torch.float16):
            for d,w,a in ((64,31,1.),(128,127,.5),(128,-1,0.),(64,0,1.)):
                torch.manual_seed(81)
                packed=torch.randn(2,257,3,3,d,device='cuda',dtype=dtype)
                xs=[packed[:,:,i].detach().requires_grad_() for i in range(3)]
                g=torch.randn_like(xs[0])
                rb=[x.detach().float().requires_grad_() for x in xs]
                ro=reference(*rb,w,a)
                rg=torch.autograd.grad(ro,rb,g.float())
                for backend in (0,1):
                    with self.subTest(dtype=dtype,d=d,w=w,a=a,backend=backend):
                        o=softplus_attn_fa4_func(*xs,window_size=(w,0),alpha=a,bwd_impl=backend)
                        gg=torch.autograd.grad(o,xs,g)
                        for x,y in zip((o,*gg),(ro,*rg)):
                            self.check(x,y)

    def test_negative_score_gradient(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        # score=-32: 1-exp(-softplus(score)) cancels to zero in FP32,
        # while the stable sigmoid retains the nonzero BF16 gradient.
        q=torch.ones(1,64,1,64,device='cuda',dtype=torch.bfloat16)
        k=torch.full_like(q,-4.)
        v=torch.ones_like(q)
        xs=[x.requires_grad_() for x in (q,k,v)]
        rb=[x.detach().float().requires_grad_() for x in xs]
        ref=reference(*rb,-1,1.)
        rg=torch.autograd.grad(ref.sum(),rb)
        o=softplus_attn_fa4_func(*xs)
        gg=torch.autograd.grad(o.sum(),xs)
        for x,y in zip((o,*gg),(ref,*rg)):
            self.check(x,y)

    def test_decode_reduction_and_boundary_lengths(self):
        from nanochat.softplus_decode import softplus_decode
        for dtype in (torch.bfloat16,torch.float16):
            for t,w in ((1,None),(65,None),(513,None),(2053,111),(4097,1024)):
                xs=[torch.randn(2,n,4,128,device='cuda',dtype=dtype)[...,::2] for n in (1,t,t)]
                q,k,v=xs
                lo=0 if w is None else max(0,t-w-1)
                p=torch.nn.functional.softplus(torch.einsum('bqhd,bkhd->bhqk',q.float(),k[:,lo:].float())*64**-.5)
                ref=torch.einsum('bhqk,bkhd->bqhd',p,v[:,lo:].float())/(t-lo)
                for reduction in ('auto','partial','atomic'):
                    self.check(softplus_decode(*xs,window_left=w,reduction=reduction),ref,.006 if dtype==torch.bfloat16 else .002)

    def test_tuned_softmax_control(self):
        from trains.bench_softplus_general import TunedSoftmax
        from nanochat.flash_attention import _sdpa_attention
        for d in (64,128):
            for w in (-1,31):
                p=torch.randn(2,257,3,3,d,device='cuda',dtype=torch.bfloat16)
                xs=[p[:,:,i].detach().requires_grad_() for i in range(3)]
                g=torch.randn_like(xs[0])
                o=TunedSoftmax.apply(*xs,w)
                gg=torch.autograd.grad(o,xs,g)
                q,k,v=[x.float().transpose(1,2) for x in xs]
                idx=torch.arange(257,device='cuda')
                keep=(idx[:,None]>=idx[None,:])
                if w>=0:
                    keep &= idx[:,None]-idx[None,:]<=w
                ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=keep).transpose(1,2)
                rg=torch.autograd.grad(ref,xs,g.float())
                for x,y in zip((o,*gg),(ref,*rg)):
                    self.check(x,y,.025)

    def test_compiled_strided_window(self):
        import nanochat.softplus_attention as sa
        with patch.object(sa,'_SOFTPLUS_IMPL','hybrid'):
            def fn(q,k,v):
                return sa.softplus_attention(q,k,v,window_size=(127,0))
            compiled=torch.compile(fn,fullgraph=True)
            p=torch.randn(2,257,3,4,128,device='cuda',dtype=torch.bfloat16)
            xs=[p[:,:,i].detach().requires_grad_() for i in range(3)]
            g=torch.randn_like(xs[0])
            a=fn(*xs); ga=torch.autograd.grad(a,xs,g)
            b=compiled(*xs); gb=torch.autograd.grad(b,xs,g)
            for x,y in zip((a,*ga),(b,*gb)):
                self.check(x,y,.005)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
