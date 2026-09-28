"""
rexp_rmsnorm / softplus_rmsnorm: FA3 custom op vs the float32 torch reference (forward and gradients),
KV-cache decode vs the training path, and a compiled GPT forward/backward without graph breaks.
Needs an SM90 GPU and the prebuilt kernels (see nanochat/fn_rmsnorm_attention.py).

    python -m pytest tests/test_fn_rmsnorm_attention.py -q
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="needs SM90")

KINDS = ["rexp_rmsnorm", "softplus_rmsnorm"]  # softplus_rmsnorm: its FA3 build on SM90


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("window", [(-1, 0), (255, 0)])
def test_op_matches_reference(kind, window):
    from nanochat.fn_rmsnorm_attention import fn_rmsnorm_attn_func, reference_attention
    torch.manual_seed(0)
    B, T, H, D = 2, 700, 4, 128
    q, k, v, do = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
    out = fn_rmsnorm_attn_func(qs, ks, vs, kind, window_size=window)
    out.backward(do)
    qr, kr, vr = (t.float().requires_grad_() for t in (q, k, v))
    ref = reference_attention(qr, kr, vr, kind, window[0])
    ref.backward(do.float())
    assert rel(out, ref) < 5e-3
    for g, gr in ((qs.grad, qr.grad), (ks.grad, kr.grad), (vs.grad, vr.grad)):
        assert rel(g, gr) < 1e-2


@pytest.mark.parametrize("kind", KINDS)
def test_kvcache_decode_matches_prefill(kind):
    from nanochat.fn_rmsnorm_attention import fn_rmsnorm_attn_func, fn_rmsnorm_attn_with_kvcache
    torch.manual_seed(0)
    B, T, H, D = 2, 96, 4, 128
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    full = fn_rmsnorm_attn_func(q, k, v, kind, window_size=(31, 0))
    kc = torch.zeros(B, 128, H, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    seqlens = torch.zeros(B, dtype=torch.int32, device="cuda")
    n0 = 40  # prefill 40 tokens, then decode one at a time
    outs = [fn_rmsnorm_attn_with_kvcache(q[:, :n0], kc, vc, kind, k=k[:, :n0], v=v[:, :n0],
                                         cache_seqlens=seqlens, window_size=(31, 0))]
    seqlens += n0
    for t in range(n0, T):
        outs.append(fn_rmsnorm_attn_with_kvcache(q[:, t:t + 1], kc, vc, kind, k=k[:, t:t + 1],
                                                 v=v[:, t:t + 1], cache_seqlens=seqlens,
                                                 window_size=(31, 0)))
        seqlens += 1
    assert rel(torch.cat(outs, 1), full) < 5e-3


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_gpt_no_graph_breaks(kind):
    import torch._dynamo
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
                    window_pattern="SL", attn_kind=kind)
    with torch.device("cuda"):
        model = GPT(cfg)
    model.init_weights()
    x = torch.randint(0, 512, (2, 256), device="cuda")
    explain = torch._dynamo.explain(model)(x, x)
    assert explain.graph_break_count == 0, explain.break_reasons
    loss = torch.compile(model)(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("window", [(-1, 0), (511, 0)])
def test_kvcache_kernel_matches_reference(kind, window):
    """The split-KV kernel path of the KV-cache API (long cache, prefill chunk then single-token
    decode, batch 1 and 4) against the float32 reference on the same cache."""
    import nanochat.fn_rmsnorm_attention as fna
    torch.manual_seed(0)
    for B in (1, 4):
        H, D, Tmax = 6, 128, 8192
        kc = torch.zeros(B, Tmax, H, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        seqlens = torch.zeros(B, dtype=torch.int32, device="cuda")
        for Tq in (6000, 1, 1, 37, 1):
            q, k, v = (torch.randn(B, Tq, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            pos = int(seqlens[0])
            out = fna.fn_rmsnorm_attn_with_kvcache(q, kc, vc, kind, k=k, v=v, cache_seqlens=seqlens, window_size=window)
            ref = fna.reference_attention(q, kc[:, :pos + Tq], vc[:, :pos + Tq], kind, window[0], q_offset=pos)
            assert rel(out, ref) < 5e-3, (B, Tq, rel(out, ref))
            seqlens += Tq
