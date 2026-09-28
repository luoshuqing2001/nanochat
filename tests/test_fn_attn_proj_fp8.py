"""
NANOCHAT_FA3_FUSE_PROJ: attention + gain + FP8 c_proj as one op vs the unfused path it replaces
(fn_rmsnorm_attn_func, y * gamma, Float8Linear) and vs a float32 reference, output and all gradients;
and a compiled GPT (--fp8) using it without graph breaks. Needs SM90 and the *_du kernel builds.

    python -m pytest tests/test_fn_attn_proj_fp8.py -q
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="needs SM90")


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run(fused, q, k, v, gamma, w, dout, kind, window):
    import nanochat.fn_rmsnorm_attention as fna
    from nanochat.fp8 import Float8Linear
    qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
    g, ww = gamma.clone().requires_grad_(), w.clone().requires_grad_()
    B, T, H, D = q.shape
    if fused:
        out = fna.fn_rmsnorm_attn_proj_fp8(qs, ks, vs, kind, g, ww, window_size=window)
    else:
        proj = Float8Linear(H * D, w.shape[0], bias=False, device="cuda")
        proj.weight = torch.nn.Parameter(ww)
        y = fna.fn_rmsnorm_attn_func(qs, ks, vs, kind, window_size=window)
        out = proj(y.reshape(B, T, H * D) * g.to(y.dtype))
        ww = proj.weight
    out.backward(dout)
    return out, qs.grad, ks.grad, vs.grad, g.grad, ww.grad


def reference(q, k, v, gamma, w, dout, kind, window):
    from nanochat.fn_rmsnorm_attention import reference_attention
    qs, ks, vs, g, ww = (t.float().requires_grad_() for t in (q, k, v, gamma, w))
    B, T, H, D = q.shape
    out = (reference_attention(qs, ks, vs, kind, window[0]).reshape(B, T, H * D) * g) @ ww.t()
    out.backward(dout.float())
    return out, qs.grad, ks.grad, vs.grad, g.grad, ww.grad


def inputs(B, T, H, D, N, seed=0):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    gamma = 1 + 0.2 * torch.randn(H * D, device="cuda")                    # fp32 master, as in the model
    w = torch.randn(N, H * D, device="cuda") / (H * D) ** 0.5
    dout = torch.randn(B, T, N, device="cuda", dtype=torch.bfloat16)
    return q, k, v, gamma, w, dout


NAMES = ["out", "dq", "dk", "dv", "dgamma", "dW"]


@pytest.mark.parametrize("kind", ["rexp_rmsnorm"])
@pytest.mark.parametrize("window,T", [((-1, 0), 704), ((255, 0), 1024)])
def test_fused_matches_unfused(kind, window, T):
    args = inputs(2, T, 4, 128, 384)
    fused = run(True, *args, kind, window)
    unfused = run(False, *args, kind, window)
    ref = reference(*args, kind, window)
    for name, a, b, r in zip(NAMES, fused, unfused, ref):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        ea, eb = rel(a, r), rel(b, r)
        print(f"{kind} {window} {name}: fused {ea:.2e} unfused {eb:.2e} fused-vs-unfused {rel(a, b):.2e}")
        if name == "out":
            assert torch.equal(a, b)          # same forward, operation for operation
        else:
            assert ea < 1.2 * eb + 1e-3       # no worse than the path it replaces


def test_fused_deterministic():
    args = inputs(2, 512, 4, 128, 384, seed=2)
    a = run(True, *args, "rexp_rmsnorm", (-1, 0))
    b = run(True, *args, "rexp_rmsnorm", (-1, 0))
    for name, x, y in zip(NAMES, a, b):
        if name != "dk" and name != "dv" and name != "dq":   # attention bwd uses atomics for dQ
            assert torch.equal(x, y), name


def test_compiled_gpt_fp8_fused_no_graph_breaks(monkeypatch):
    import torch._dynamo
    import nanochat.fn_rmsnorm_attention as fna
    from nanochat.fp8 import Float8Linear, convert_to_float8_training
    from nanochat.gpt import GPT, GPTConfig
    monkeypatch.setattr(fna, "FUSE_PROJ", True)
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
                    window_pattern="SL", attn_kind="rexp_rmsnorm")
    with torch.device("cuda"):
        model = GPT(cfg)
    model.init_weights()
    # base_train's filter: FP8 only where the dims allow it (not e.g. the 24-wide smear gate)
    convert_to_float8_training(model, module_filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and
                               m.in_features % 16 == 0 and m.out_features % 16 == 0 and
                               min(m.in_features, m.out_features) >= 128)
    assert all(type(b.attn.c_proj) is Float8Linear for b in model.transformer.h)
    calls = []
    real = fna.fn_rmsnorm_attn_proj_fp8
    monkeypatch.setattr(fna, "fn_rmsnorm_attn_proj_fp8", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    x = torch.randint(0, 512, (2, 256), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, x)
    assert len(calls) == cfg.n_layer
    monkeypatch.setattr(fna, "fn_rmsnorm_attn_proj_fp8", real)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        explain = torch._dynamo.explain(model)(x, x)
        assert explain.graph_break_count == 0, explain.break_reasons
        loss = torch.compile(model)(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
