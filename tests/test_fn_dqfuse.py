"""
NANOCHAT_FA3_DQFUSE: the *_dqf backward builds (dQ memset + postprocess folded into the backward
kernel) vs the regular ones -- attention op, the fused attention + FP8 c_proj op, and a compiled GPT.
dK/dV (and everything downstream of them only) must not change; dQ only by fp32 summation order.
Needs SM90 and the *_dqf builds (python hopper_softplus/build.py fn_rexp_pre_dqf fn_rexp_pre_du_dqf).

    python -m pytest tests/test_fn_dqfuse.py -q
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="needs SM90")


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def grads(fn, dqfuse, *inputs, dout):
    import nanochat.fn_rmsnorm_attention as fna
    fna.DQFUSE = dqfuse
    try:
        xs = [t.clone().requires_grad_() for t in inputs]
        fn(*xs).backward(dout)
        return [x.grad for x in xs]
    finally:
        fna.DQFUSE = False


@pytest.mark.parametrize("kind", ["rexp_rmsnorm"])
@pytest.mark.parametrize("window", [(-1, 0), (511, 0)])
def test_op_dqfuse_matches(kind, window):
    from nanochat.fn_rmsnorm_attention import fn_rmsnorm_attn_func
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(4, 2048, 6, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    f = lambda q, k, v: fn_rmsnorm_attn_func(q, k, v, kind, window_size=window)
    ref = grads(f, False, q, k, v, dout=do)
    for rep in range(2):  # the second call runs on the buffers the first left behind
        got = grads(f, True, q, k, v, dout=do)
        assert rel(got[0], ref[0]) < 1e-4
        assert torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])


@pytest.mark.parametrize("kind", ["rexp_rmsnorm"])
def test_fused_proj_dqfuse_matches(kind):
    from nanochat.fn_rmsnorm_attention import fn_rmsnorm_attn_proj_fp8
    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 2048, 6, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    gamma = 1 + 0.1 * torch.randn(768, device="cuda")
    w = torch.randn(768, 768, device="cuda") / 768 ** 0.5
    do = torch.randn(4, 2048, 768, device="cuda", dtype=torch.bfloat16)
    f = lambda q, k, v, g, w: fn_rmsnorm_attn_proj_fp8(q, k, v, kind, g, w, window_size=(511, 0))
    ref = grads(f, False, q, k, v, gamma, w, dout=do)
    got = grads(f, True, q, k, v, gamma, w, dout=do)
    assert rel(got[0], ref[0]) < 1e-4
    for a, b in zip(got[1:], ref[1:]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["rexp_rmsnorm"])
def test_compiled_gpt_dqfuse(kind):
    import torch._dynamo
    import nanochat.fn_rmsnorm_attention as fna
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=512, vocab_size=512, n_layer=4, n_head=2, n_kv_head=2, n_embd=256,
                    window_pattern="SSSL", attn_kind=kind)
    with torch.device("cuda"):
        model = GPT(cfg)
    model.init_weights()
    for b in model.transformer.h:  # c_proj is zero at init: no gradient would reach q/k/v
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.02)
    x = torch.randint(0, 512, (4, 512), device="cuda")
    out = {}
    for dqfuse in (False, True):
        fna.DQFUSE = dqfuse
        torch._dynamo.reset()
        model.zero_grad(set_to_none=True)
        if dqfuse:
            explain = torch._dynamo.explain(model)(x, x)
            assert explain.graph_break_count == 0, explain.break_reasons
        loss = torch.compile(model)(x, x)
        loss.backward()
        out[dqfuse] = (loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    fna.DQFUSE = False
    assert out[True][0] == out[False][0]
    worst = max(rel(out[True][1][n], g) for n, g in out[False][1].items() if g.norm() > 0)
    assert worst < 1e-2, worst  # dQ's summation order, propagated through a few layers
