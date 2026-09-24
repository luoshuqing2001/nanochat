"""Fixed KV prefill, training gradients, and compiled dispatch regressions."""
import unittest
from unittest.mock import patch

import torch
from nanochat.softplus_fixed_kv import fixed_kv_attention, fixed_kv_forward, _plan, _backward_plan


def reference(q, k, v, window, alpha):
    tq, tk = q.shape[1], k.shape[1]
    m = torch.arange(tq, device=q.device)[:, None] + tk - tq
    n = torch.arange(tk, device=q.device)[None, :]
    mask = n <= m
    if window >= 0:
        mask &= n >= m - window
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * q.shape[-1] ** -.5
    p = torch.nn.functional.softplus(scores) * mask
    return torch.einsum("bhqk,bkhd->bqhd", p, v) * mask.sum(-1).float().pow(-alpha)[None, :, None, None]


class TestPlans(unittest.TestCase):
    def test_fixed_intervals_cover_every_valid_pair_once(self):
        for tq, tk, window in ((257, 257, -1), (113, 511, 127), (129, 257, 0)):
            for chunk in (64, 256, 512):
                with self.subTest(tq=tq, tk=tk, window=window, chunk=chunk):
                    qpos = torch.arange(tq)[:, None] + tk - tq
                    kpos = torch.arange(tk)[None, :]
                    valid = kpos <= qpos
                    if window >= 0:
                        valid &= kpos >= qpos - window
                    tasks, _ = _plan(tq, tk, window, 128, chunk, "cpu")
                    counts = torch.zeros(tq, tk, dtype=torch.int32)
                    for mi, n, _ in tasks.tolist():
                        self.assertEqual(n % chunk, 0)
                        counts[mi*128:(mi+1)*128, n:n+chunk] += 1
                    self.assertTrue(torch.all(counts[valid] == 1))
                    tasks, _ = _backward_plan(tq, tk, window, 64, chunk, "cpu")
                    counts.zero_()
                    for ni, m, _ in tasks.tolist():
                        self.assertEqual(m % chunk, 0)
                        counts[m:m+chunk, ni*64:(ni+1)*64] += 1
                    self.assertTrue(torch.all(counts[valid] == 1))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFixedKV(unittest.TestCase):
    def assert_error(self, x, ref, tol=.02):
        diff = x.float() - ref.float()
        self.assertTrue(torch.isfinite(diff).all())
        self.assertLess(float(diff.abs().max()), tol * float(ref.abs().max().clamp_min(1e-6)))
        self.assertLess(float(diff.square().mean().sqrt()), tol * float(ref.square().mean().sqrt().clamp_min(1e-6)))

    def test_autograd_and_strided_inputs(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        torch.manual_seed(73)
        for dtype in (torch.bfloat16, torch.float16):
            for tq, tk, w, a, d in ((257, 257, -1, 1., 128), (300, 300, 111, .5, 64),
                                   (113, 511, 127, 0., 64), (129, 257, 0, 1., 64)):
                base = [torch.randn(2, t, 3, 2, d, device="cuda", dtype=dtype)[:, :, i]
                        for i, t in enumerate((tq, tk, tk))]
                do = torch.randn_like(base[0])
                rb = [x.float().detach().requires_grad_() for x in base]
                ro = reference(*rb, w, a)
                rg = torch.autograd.grad(ro, rb, do.float())
                for chunk in (128, 256, 512):
                    for impl in ("triton", "cute"):
                        with self.subTest(dtype=dtype, shape=(tq,tk,w,a,d), chunk=chunk, impl=impl):
                            xs = [x.detach().requires_grad_() for x in base]
                            if impl == "triton":
                                out = fixed_kv_attention(*xs, w, a, kv_chunk=chunk, q_chunk=chunk)
                            else:
                                tile_n = 128 if d == 64 else 64
                                out = softplus_attn_fa4_func(*xs, window_size=(w,0), alpha=a,
                                        balanced_chunk=chunk//tile_n, bwd_m_chunk=chunk//64)
                            grads = torch.autograd.grad(out, xs, do)
                            for x, r in zip((out, *grads), (ro, *rg)):
                                self.assert_error(x.detach(), r.detach())

    def test_compile_and_model_dispatch(self):
        import nanochat.softplus_attention as sa
        for impl, t, h, d in (("fixed",257,2,64), ("fixed_triton",257,2,64),
                               ("hybrid",257,2,64), ("hybrid",4096,1,128),
                               ("hybrid",2048,6,128)):
            base = [torch.randn(1,t,h,d,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
            g = torch.randn_like(base[0])
            with self.subTest(impl=impl), patch.object(sa, "_SOFTPLUS_IMPL", impl), \
                    patch.object(sa, "_SOFTPLUS_KV_CHUNK", 256), patch.object(sa, "_SOFTPLUS_Q_CHUNK", 256):
                def fn(q, k, v):
                    return sa.softplus_attention(q, k, v, window_size=(t, 0))
                compiled = torch.compile(fn, fullgraph=True)
                eager_x = [x.detach().requires_grad_() for x in base]
                e = fn(*eager_x)
                eg = torch.autograd.grad(e, eager_x, g)
                compiled_x = [x.detach().requires_grad_() for x in base]
                c = compiled(*compiled_x)
                cg = torch.autograd.grad(c, compiled_x, g)
                for x, r in zip((c, *cg), (e, *eg)):
                    self.assert_error(x.detach(), r.detach(), .005)

    def test_small_batch_auto_reference(self):
        from flash_attn_4.softplus_api import softplus_attn_fa4_func
        from nanochat.softplus_fixed_kv import auto_fixed_kv_chunk
        torch.manual_seed(107)
        # Packed QKV projections have noncontiguous sequence/head strides.
        base = torch.randn(1,2048,3,6,128,device="cuda",dtype=torch.bfloat16)
        xs = [base[:,:,i].detach().requires_grad_() for i in range(3)]
        props = torch.cuda.get_device_properties(xs[0].device)
        if props.major == 12 and props.multi_processor_count == 48:
            # Smaller CuTe query tiles now supply enough parallelism at H6.
            self.assertIsNone(auto_fixed_kv_chunk(*xs))
        g = torch.randn_like(xs[0])
        out = softplus_attn_fa4_func(*xs)
        grads = torch.autograd.grad(out,xs,g)
        rb = [x.detach().float().requires_grad_() for x in xs]
        ref = reference(*rb,-1,1.)
        rg = torch.autograd.grad(ref,rb,g.float())
        for x,r in zip((out,*grads),(ref,*rg)):
            self.assert_error(x,r)

    def test_prefill_cache_append(self):
        import nanochat.softplus_attention as sa
        q, k, v = [torch.randn(2,113,2,64,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
        for impl in ("fixed", "fixed_triton"):
            with self.subTest(impl=impl), patch.object(sa,"_SOFTPLUS_IMPL",impl), \
                    patch.object(sa,"_SOFTPLUS_KV_CHUNK",256):
                kc, vc = [torch.randn(2,600,2,64,device="cuda",dtype=q.dtype) for _ in range(2)]
                got = sa.softplus_attn_with_kvcache(q,kc,vc,k,v,torch.tensor([398,398]),window_size=(127,0))
                self.assertTrue(torch.equal(kc[:,398:511], k))
                self.assertTrue(torch.equal(vc[:,398:511], v))
                expected = reference(q.float(),kc[:,:511].float(),vc[:,:511].float(),127,1.)
                self.assert_error(got,expected)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
