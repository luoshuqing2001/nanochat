"""Independent reference for the experimental dV + dQ/dK split."""
import unittest
import torch
from test_softplus_warp8 import reference


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestSplitBackward(unittest.TestCase):
    def test_reference(self):
        from flash_attn_4.softplus_split_backward import softplus_split_backward
        torch.set_num_threads(2);torch.manual_seed(746)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,d,w,alpha in ((129,257,128,-1,1.),(113,257,128,31,.5),(129,65,64,-1,1.),(65,129,64,0,0.)):
                xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,1] for t in (tq,tk,tk)]
                rr=[x.float().requires_grad_() for x in xs];o=reference(*rr,w,alpha)
                g=torch.randn_like(xs[0]);rg=torch.autograd.grad(o,rr,g.float())
                for nt,st in ((128,2),(256,1)):
                    with self.subTest(dtype=dtype,tq=tq,tk=tk,d=d,window=w,nt=nt):
                        grads=softplus_split_backward(*xs,o.detach().to(dtype),g,window=w,alpha=alpha,qk_threads=nt,qk_stages=st,dv_threads=nt)
                        for x,y in zip(grads,rg):
                            self.assertTrue(torch.isfinite(x).all())
                            self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)

    def test_replay_and_extreme(self):
        from flash_attn_4.softplus_split_backward import softplus_split_backward
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for score in (-80.,-12.,0.,120.):
            q=torch.ones(1,129,1,128,device='cuda',dtype=torch.bfloat16)
            xs=[x.requires_grad_() for x in (q,torch.full_like(q,score/128**.5),torch.ones_like(q))]
            out=softplus_attn_fa4_func(*xs,num_splits=1,bwd_m_chunk=0,bwd_schedule='warp8')
            g=torch.ones_like(out);ref=torch.autograd.grad(out,xs,g)
            fn=lambda:softplus_split_backward(*xs,out,g)
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):fn();fn()
            side.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):actual=fn()
            for _ in range(3):graph.replay()
            torch.cuda.synchronize()
            for x,y in zip(actual,ref):torch.testing.assert_close(x,y,rtol=.01,atol=1e-30)


if __name__=='__main__':unittest.main()
