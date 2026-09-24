"""Eight-warp backward: independent gradients, masks, and autograd integration."""
import unittest
import torch


def reference(q,k,v,window,alpha):
    k=k.repeat_interleave(q.shape[2]//k.shape[2],dim=2)
    v=v.repeat_interleave(q.shape[2]//v.shape[2],dim=2)
    qi=torch.arange(q.shape[1],device=q.device)[:,None]+k.shape[1]-q.shape[1]
    ki=torch.arange(k.shape[1],device=q.device)[None,:]
    mask=ki<=qi
    if window>=0:mask=mask & (ki>=qi-window)
    p=torch.nn.functional.softplus(torch.einsum('bqhd,bkhd->bhqk',q,k)*q.shape[-1]**-.5)*mask
    return torch.einsum('bhqk,bkhd->bqhd',p,v)*mask.sum(-1).clamp_min(1).float().pow(-alpha)[None,:,None,None]


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestWarp8(unittest.TestCase):
    def test_reference_gradients(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.set_num_threads(2);torch.manual_seed(445)
        for dtype in (torch.bfloat16,torch.float16):
            for tq,tk,h,hk,d,w,alpha in ((257,257,2,2,128,-1,1.),(113,257,2,2,128,31,.5),
                    (129,65,2,2,128,-1,1.),(65,129,2,2,64,0,0.),(129,257,4,2,128,-1,1.)):
                with self.subTest(dtype=dtype,shape=(tq,tk,h,hk,d),window=w,alpha=alpha):
                    xs=[torch.randn(1,t,heads,d,device='cuda',dtype=dtype).requires_grad_()
                        for t,heads in ((tq,h),(tk,hk),(tk,hk))]
                    rr=[x.detach().float().requires_grad_() for x in xs]
                    ref=reference(*rr,w,alpha);g=torch.randn_like(xs[0])
                    if h != hk:
                        # Isolate backward: the existing packed-GQA forward has
                        # independent reference mismatches and is unchanged here.
                        from flash_attn_4.interface import _flash_attn_bwd
                        out=ref.detach().to(dtype)
                        lse=torch.empty(1,h,tq,device='cuda',dtype=torch.float32)
                        grads=_flash_attn_bwd(*xs,out,g,lse,causal=True,attn_kind='softplus',
                            softplus_early_dv=True,sm120_bwd_tile=(64,64,1,1),sm120_bwd_num_threads=256)
                    else:
                        out=softplus_attn_fa4_func(*xs,num_splits=1,bwd_m_chunk=0,bwd_schedule='warp8',
                                                 window_size=(None,None) if w<0 else (w,0),alpha=alpha)
                        grads=torch.autograd.grad(out,xs,g)
                    actual=(out,*grads);expected=(ref,*torch.autograd.grad(ref,rr,g.float()))
                    for label,x,y in zip(('out','dq','dk','dv'),actual,expected):
                        self.assertTrue(torch.isfinite(x).all())
                        err=(x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)
                        self.assertLess(float(err),.025,label)

    def test_extreme_scores(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        for score in (-80.,-12.,0.,120.):
            q=torch.ones(1,129,1,128,device='cuda',dtype=torch.bfloat16)
            xs=[x.requires_grad_() for x in (q,torch.full_like(q,score/128**.5),torch.ones_like(q))]
            ref=softplus_attn_fa4_func(*xs,num_splits=1,bwd_m_chunk=0,bwd_schedule='previous',early_dv=True)
            out=softplus_attn_fa4_func(*xs,num_splits=1,bwd_m_chunk=0,bwd_schedule='warp8')
            g=torch.ones_like(out)
            for x,y in zip((out,*torch.autograd.grad(out,xs,g)),(ref,*torch.autograd.grad(ref,xs,g))):
                torch.testing.assert_close(x,y,rtol=.01,atol=1e-30)

    def test_compiled_default(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        xs=[torch.randn(1,2048,6,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
        g=torch.randn_like(xs[0])
        old=softplus_attn_fa4_func(*xs,bwd_schedule='previous')
        expected=(old,*torch.autograd.grad(old,xs,g))
        def fn(q,k,v):return softplus_attn_fa4_func(q,k,v)
        out=torch.compile(fn,fullgraph=True)(*xs)
        for x,y in zip((out,*torch.autograd.grad(out,xs,g)),expected):
            self.assertLess(float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)),.005)

    def test_softmax_control(self):
        from flash_attn_4.interface import _flash_attn_fwd,_flash_attn_bwd
        xs=[torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        out,lse,*_=_flash_attn_fwd(*xs,causal=True,return_lse=True);g=torch.randn_like(out)
        old=_flash_attn_bwd(*xs,out,g,lse,causal=True,sm120_bwd_tile=(64,64,2,1))
        new=_flash_attn_bwd(*xs,out,g,lse,causal=True,sm120_bwd_tile=(64,64,1,1),sm120_bwd_num_threads=256)
        for x,y in zip(new,old):
            self.assertLess(float((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-20)),.005)

    def test_narrow_owner_gradients(self):
        from flash_attn_4.interface import _flash_attn_bwd
        for d,n in ((64,16),(128,32)):
            xs=[torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16) for t in (113,257,257)]
            rr=[x.float().requires_grad_() for x in xs]
            ref=reference(*rr,-1,1.);g=torch.randn_like(xs[0])
            expected=torch.autograd.grad(ref,rr,g.float())
            lse=torch.empty(1,2,113,device='cuda',dtype=torch.float32)
            actual=_flash_attn_bwd(*xs,ref.to(xs[0].dtype),g,lse,causal=True,attn_kind='softplus',
                softplus_early_dv=True,sm120_bwd_tile=(64,n,2,1))
            for x,y in zip(actual,expected):
                self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)


if __name__=='__main__':unittest.main()
