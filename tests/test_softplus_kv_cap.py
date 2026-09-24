"""Hard KV token bounds, direct-write bypass, and atomic replay correctness."""
import unittest
from collections import Counter
import torch
from test_softplus_fixed_kv import reference

class TestCapPlan(unittest.TestCase):
    def test_coverage(self):
        from flash_attn_4.softplus_stream import capped_plan_cpu
        from nanochat.softplus_stream_k import plan_cpu
        for tq,tk in ((1,1),(113,513),(769,769),(2048,2048)):
            for bn in (64,128):
                for w in (-1,0,511):
                    for cap in (256,512,1024):
                        offsets,segs,groups,slots=capped_plan_cpu(tq,tk,w,64,bn,cap)
                        whole=plan_cpu(tq,tk,w,64,bn,1)[1]
                        reordered=capped_plan_cpu(tq,tk,w,64,bn,cap,True)
                        self.assertEqual(sorted(reordered[1]),sorted(segs))
                        self.assertEqual(reordered[2:],(groups,slots))
                        self.assertEqual(reordered[1],tuple(sorted(segs,key=lambda x:(x[1],-x[0]))))
                        expand=lambda xs:Counter((m,n) for m,lo,hi,_ in xs for n in range(lo,hi))
                        self.assertEqual(expand(segs),expand(whole))
                        self.assertEqual(offsets,tuple(range(len(segs)+1)))
                        self.assertTrue(all(0<(hi-lo)*bn<=cap for _,lo,hi,_ in segs))
                        self.assertEqual(sorted(s for *_,s in segs if s>=0),list(range(slots)))
                        for m,lo,hi,_ in whole:
                            parts=[s for mm,_,_,s in segs if mm==m]
                            if (hi-lo)*bn<=cap:self.assertEqual(parts,[-1])
                        for m,first,n in groups:
                            self.assertEqual([s for mm,_,_,s in segs if mm==m],list(range(first,first+n)))
        for cap in (-1,0,65):
            with self.assertRaises(ValueError):capped_plan_cpu(512,512,-1,64,64,cap)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TestCapGPU(unittest.TestCase):
    def test_forward_backward_replay(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4,softplus_attn_fa4_func
        torch.set_num_threads(2)
        for d,dtype,tq,tk,w,cap in ((64,torch.bfloat16,769,769,-1,256),(128,torch.float16,513,769,-1,512),(128,torch.bfloat16,257,769,511,256),(64,torch.bfloat16,129,257,31,512)):
            xs=[torch.randn(1,t,3,2,d,device='cuda',dtype=dtype)[:,:,i].requires_grad_() for i,t in enumerate((tq,tk,tk))]
            rb=[x.detach().float().requires_grad_() for x in xs];g=torch.randn_like(xs[0])
            ref=reference(*rb,w,1.);rg=torch.autograd.grad(ref,rb,g.float())
            for extra in (dict(stream_atomic=True),dict(stream_atomic=True,stream_global=True),dict(stream_atomic=True,kv_major=True),dict(kv_major=True),dict(stream_complete=True,kv_major=True)):
                opts=dict(kv_split_size=cap,window_size=(w,0),**extra)
                fn=lambda q,k,v:softplus_attn_fa4_func(q,k,v,**opts)
                torch._dynamo.reset()
                out=torch.compile(fn,fullgraph=True)(*xs)
                grads=torch.autograd.grad(out,xs,g)
                for x,y in zip((out,*grads),(ref,*rg)):
                    self.assertLess(float((x.float()-y).abs().max()/y.abs().max().clamp_min(1e-20)),.025)
                forward=lambda:softplus_attn_fa4(*xs,**opts)
                for _ in range(3):forward()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=forward()
                for _ in range(10):graph.replay()
                torch.cuda.synchronize()
                self.assertLess(float((actual.float()-ref).abs().max()/ref.abs().max()),.025)
                del graph
