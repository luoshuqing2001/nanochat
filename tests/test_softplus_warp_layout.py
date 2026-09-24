"""Independent gradient checks for explicit SM120 warp-layout choices."""
import itertools
import unittest
import torch
from test_softplus_warp8 import reference


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestWarpLayout(unittest.TestCase):
    def test_compiled_public_and_split(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        xs=[torch.randn(1,129,2,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        rr=[x.detach().float().requires_grad_() for x in xs]
        ref=reference(*rr,-1,1.);g=torch.randn_like(xs[0]);rg=torch.autograd.grad(ref,rr,g.float())
        for chunk in (0,1):
            def fn(q,k,v):
                return softplus_attn_fa4_func(q,k,v,num_splits=1,bwd_m_chunk=chunk,bwd_schedule='warp8_dkv')
            out=torch.compile(fn,fullgraph=True)(*xs)
            for x,y in zip((out,*torch.autograd.grad(out,xs,g)),(ref,*rg)):
                self.assertLess(float((x.detach().float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)

    def test_reference(self):
        from flash_attn_4.interface import _flash_attn_bwd
        torch.set_num_threads(2)
        torch.manual_seed(944)
        for dtype,w,alpha,h,hk in ((torch.bfloat16,-1,1.,4,2),(torch.float16,31,.5,2,2)):
            xs=[torch.randn(1,t,3,heads,128,device='cuda',dtype=dtype)[:,:,1]
                for t,heads in ((113,h),(257,hk),(257,hk))]
            rr=[x.float().requires_grad_() for x in xs]
            out=reference(*rr,w,alpha)
            g=torch.randn_like(xs[0])
            expected=torch.autograd.grad(out,rr,g.float())
            lse=torch.empty(1,h,113,device='cuda',dtype=torch.float32)
            for layout in itertools.product((4,2),repeat=3):
                with self.subTest(dtype=dtype,layout=layout,window=w):
                    actual=_flash_attn_bwd(*xs,out.detach().to(dtype),g,lse,causal=w<0,
                        window_size_left=None if w<0 else w,window_size_right=None if w<0 else 0,
                        attn_kind='softplus',softplus_alpha=alpha,softplus_early_dv=True,
                        sm120_bwd_num_threads=256,sm120_bwd_tile=(64,64,1,1),sm120_bwd_warp_layout=layout)
                    for x,y in zip(actual,expected):
                        self.assertTrue(torch.isfinite(x).all())
                        self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)


if __name__=='__main__':unittest.main()
