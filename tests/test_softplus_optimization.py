"""Regressions for fixed-KV decode and full-context hybrid backend selection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

from nanochat.softplus_decode import softplus_decode


def reference(q, k, v, window, alpha, scale):
    start = max(0, k.shape[1] - window - 1) if window is not None and window >= 0 else 0
    k, v = k[:, start:].float(), v[:, start:].float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) * scale
    return torch.einsum("bhqk,bkhd->bqhd", torch.nn.functional.softplus(scores), v) / k.shape[1] ** alpha


def check_decode_reference(dtype, t, window, alpha, d, dv):
    torch.manual_seed(42)
    # Noncontiguous dimensions and overallocated cache exercise all supplied strides.
    q = torch.randn(2, 1, 3, d * 2, device="cuda", dtype=dtype)[..., ::2]
    k = torch.randn(2, t + 19, 3, d * 2, device="cuda", dtype=dtype)[:, :t, :, ::2]
    v = torch.randn(2, t + 19, 3, dv * 2, device="cuda", dtype=dtype)[:, :t, :, ::2]
    scale = .17
    ref = reference(q, k, v, window, alpha, scale)
    for chunk in (None, 64, 256, 512):
        out = softplus_decode(q, k, v, window, alpha, scale, chunk)
        assert out.shape == (2, 1, 3, dv) and out.dtype == dtype
        tol = .005 if dtype == torch.bfloat16 else .001
        assert (out.float() - ref).abs().max() <= tol * ref.abs().max().clamp_min(1e-6)


def check_decode_auto_and_cache_append():
    from nanochat.softplus_attention import softplus_attn_with_kvcache
    from flash_attn_4.softplus_api import softplus_attn_fa4
    torch.manual_seed(5)
    q = torch.randn(2, 1, 6, 128, device="cuda", dtype=torch.bfloat16)
    kc, vc = [torch.randn(2, 1080, 6, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
    k, v = [torch.randn_like(q) for _ in range(2)]
    pos = 1027
    got = softplus_attn_with_kvcache(q, kc, vc, k, v, torch.tensor([pos, pos]), window_size=(511, 0))
    assert torch.equal(kc[:, pos:pos+1], k) and torch.equal(vc[:, pos:pos+1], v)
    ref = reference(q, kc[:, :pos+1], vc[:, :pos+1], 511, 1., 128 ** -.5)
    assert (got.float() - ref).abs().max() < .005 * ref.abs().max()
    # Explicit CuTe splits stay available and agree with the new automatic route.
    old = softplus_attn_fa4(q, kc[:, :pos+1], vc[:, :pos+1], window_size=(511, 0), num_splits=2)
    assert (old.float() - got.float()).abs().max() < .01 * ref.abs().max()


def check_full_context_hybrid_gradients():
    import nanochat.softplus_attention as sa
    assert sa.HAS_FA4_SOFTPLUS
    torch.manual_seed(8)
    base = [torch.randn(1, 257, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = torch.randn_like(base[0])
    q, k, v = [x.clone().requires_grad_() for x in base]
    out = sa.softplus_attention(q, k, v, window_size=(257, 0))
    out.backward(g)
    qr, kr, vr = [x.float().requires_grad_() for x in base]
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr) * 64 ** -.5
    keep = torch.ones(257, 257, device="cuda", dtype=torch.bool).tril()
    p = torch.nn.functional.softplus(scores) * keep
    ref = torch.einsum("bhqk,bkhd->bqhd", p, vr) / torch.arange(1, 258, device="cuda")[None, :, None, None]
    ref.backward(g.float())
    for got, expected in [(out, ref), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)]:
        assert (got.float() - expected).abs().max() < .02 * expected.abs().max()


class TestSoftplusDispatch(unittest.TestCase):
    def test_backward_uses_sm120_tile_width(self):
        from flash_attn_4.softplus_api import auto_bwd_m_chunk
        with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(major=12,multi_processor_count=48)):
            self.assertIsNone(auto_bwd_m_chunk(1,6,2048,2048))
            # Truly low-parallelism shapes still permit balanced tasks.
            self.assertIsNotNone(auto_bwd_m_chunk(1,1,4096,4096))
            # Long queries already expose enough blocks with the pipelined kernel.
            self.assertIsNone(auto_bwd_m_chunk(1,1,16384,16384))

    def test_hybrid_normalizes_window_before_dispatch(self):
        import nanochat.softplus_attention as sa
        q = torch.empty(8, 2048, 12, 128, device="meta")
        with patch.object(sa, "HAS_FA4_SOFTPLUS", True), patch.object(sa, "_SOFTPLUS_IMPL", "hybrid"), \
                patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=48)):
            for window, expected in [(None, 0), (-1, 0), (2048, 0), (4096, 0), (511, 1), (0, 1)]:
                with self.subTest(window=window), patch.object(sa, "_FA4_FUNC") as fn:
                    sa.softplus_attention(q, q, q, window_size=(window, 0))
                    self.assertEqual(fn.call_args.kwargs["bwd_impl"], expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSoftplusGPU(unittest.TestCase):
    def test_decode_reference(self):
        for dtype in (torch.bfloat16, torch.float16):
            for args in [(1, None, 1., 64, 64), (255, None, .5, 128, 64),
                         (64, None, 1., 64, 64), (65, None, 1., 64, 64),
                         (256, 0, 1., 64, 128), (257, 256, 0., 128, 128),
                         (513, None, 1., 128, 128), (1025, 511, .5, 64, 64),
                         (1024, None, 1., 128, 128), (1025, None, 1., 128, 128),
                         (4097, 512, 1., 128, 128), (4097, 10000, 1., 128, 128)]:
                with self.subTest(dtype=dtype, args=args):
                    check_decode_reference(dtype, *args)

    def test_decode_auto_and_cache_append(self):
        check_decode_auto_and_cache_append()

    def test_full_context_hybrid_gradients(self):
        import nanochat.softplus_attention as sa
        with patch.object(sa, "_SOFTPLUS_IMPL", "hybrid"):
            check_full_context_hybrid_gradients()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
