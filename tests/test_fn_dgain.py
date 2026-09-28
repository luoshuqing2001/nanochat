"""
NANOCHAT_FA3_DGAIN: attention + the output gain as one autograd node, the attention backward taking dZ
and gamma (*_g builds) instead of a materialized dy -- must be bit-identical to the unfused path.

What is bit-reproducible run to run at all: the loss and every gradient that dQ doesn't reach (dQ is
accumulated with fp32 atomics in any FA3 backward). In an L-layer model that is the lm_head, the last
block's MLP, c_proj, gain, c_k, c_v (dK/dV have no atomics), not c_q or anything upstream of it. The
test first checks that on the unfused path (run twice), then holds DGAIN to exactly those tensors,
and everything else to the unfused path's own run-to-run spread.

    python -m pytest tests/test_fn_dgain.py -q
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="needs SM90")


def run(model, x, dgain, mode="1"):
    import nanochat.fn_rmsnorm_attention as fna
    import torch._inductor.config as inductor_config
    # One reduction config per kernel: inductor otherwise benchmarks a persistent and a looped variant
    # of e.g. the dgamma reduction on first compile and keeps the faster -- two summation orders, so two
    # fresh compiles of the *same* program can differ in the last bit. Pinning it makes "bit-identical"
    # testable; it is a property of torch.compile, not of either path.
    inductor_config.triton.persistent_reductions = False
    fna.DGAIN, fna.DGAIN_MODE = dgain, mode
    torch._dynamo.reset()
    model.zero_grad(set_to_none=True)
    loss = torch.compile(model, dynamic=False)(x, x)
    loss.backward()
    fna.DGAIN, fna.DGAIN_MODE = True, "1"
    return loss.detach().clone(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("kind", ["rexp_rmsnorm", "softplus_rmsnorm"])
def test_dgain_bit_identical(kind):
    import os
    import nanochat.fn_rmsnorm_attention as fna
    from nanochat.fp8 import Float8Linear, convert_to_float8_training
    from nanochat.gpt import GPT, GPTConfig
    build = fna.G_BUILDS[fna.KIND_TO_BUILD[kind]]
    if not os.path.exists(os.path.join(fna._LIB_DIR, build)):
        pytest.skip(f"{build} not built")
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=2048, vocab_size=8192, n_layer=2, n_head=6, n_kv_head=6, n_embd=768,
                    window_pattern="SL", attn_kind=kind)
    with torch.device("cuda"):
        model = GPT(cfg)
    model.init_weights()
    for b in model.transformer.h:  # a trained-like state: c_proj is zero and the gain one at init
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.02)
        torch.nn.init.normal_(b.mlp.c_proj.weight, std=0.01)
        b.attn.attn_gamma.data.add_(0.3 * torch.randn_like(b.attn.attn_gamma))
    convert_to_float8_training(model, module_filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and
                               m.in_features % 16 == 0 and m.out_features % 16 == 0 and min(m.in_features, m.out_features) >= 128)
    assert all(type(b.attn.c_proj) is Float8Linear for b in model.transformer.h)
    x = torch.randint(0, 8192, (8, 2048), device="cuda")
    loss0, g0 = run(model, x, False)
    loss1, g1 = run(model, x, False)
    loss2, g2 = run(model, x, True)
    last = f"transformer.h.{cfg.n_layer - 1}."
    exact = [n for n in g0 if n.startswith("lm_head") or (n.startswith(last) and "c_q" not in n)]
    assert any("attn_gamma" in n for n in exact) and any("c_v" in n for n in exact)
    assert torch.equal(loss0, loss1) and all(torch.equal(g0[n], g1[n]) for n in exact), "unfused path not reproducible"
    assert torch.equal(loss2, loss0), (loss2.item(), loss0.item())
    for n in exact:
        assert torch.equal(g2[n], g0[n]), n
    for n in g0:  # the rest carry dQ's atomic-order noise (run to run too): same algorithm, tiny spread
        norm = g0[n].float().norm().clamp_min(1e-30)
        rel = ((g2[n].float() - g0[n].float()).norm() / norm).item()
        spread = ((g1[n].float() - g0[n].float()).norm() / norm).item()
        # scalar gradients summed over many cancelling terms (smear_lambda) amplify the atomics' noise
        assert rel < max(20 * spread, 1e-2), (n, rel, spread)


@pytest.mark.parametrize("kind", ["rexp_rmsnorm", "softplus_rmsnorm"])
def test_dgain_mode2_fused_dgamma(kind):
    """NANOCHAT_FA3_DGAIN=2 (*_gr builds, dgamma reduced in the attention preprocess) vs mode 1: the same
    forward and every gradient but dgamma's bit for bit where they are reproducible at all; dgamma only
    changes its summation (fp32, not rounded to bf16), so it is close to mode 1's."""
    import os
    import nanochat.fn_rmsnorm_attention as fna
    from nanochat.fp8 import convert_to_float8_training
    from nanochat.gpt import GPT, GPTConfig
    build = fna.GR_BUILDS[fna.KIND_TO_BUILD[kind]]
    if not os.path.exists(os.path.join(fna._LIB_DIR, build)):
        pytest.skip(f"{build} not built")
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=2048, vocab_size=8192, n_layer=2, n_head=6, n_kv_head=6, n_embd=768,
                    window_pattern="SL", attn_kind=kind)
    with torch.device("cuda"):
        model = GPT(cfg)
    model.init_weights()
    for b in model.transformer.h:
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.02)
        torch.nn.init.normal_(b.mlp.c_proj.weight, std=0.01)
        b.attn.attn_gamma.data.add_(0.3 * torch.randn_like(b.attn.attn_gamma))
    convert_to_float8_training(model, module_filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and
                               m.in_features % 16 == 0 and m.out_features % 16 == 0 and min(m.in_features, m.out_features) >= 128)
    x = torch.randint(0, 8192, (8, 2048), device="cuda")
    loss1, g1 = run(model, x, True, "1")
    loss2, g2 = run(model, x, True, "2")
    assert torch.equal(loss1, loss2)
    last = f"transformer.h.{cfg.n_layer - 1}."
    for n in g1:
        if "attn_gamma" in n:  # every layer's: only the summation differs
            rel = ((g2[n].float() - g1[n].float()).norm() / g1[n].float().norm()).item()
            assert rel < 1e-2, (n, rel)
        elif n.startswith("lm_head") or (n.startswith(last) and "c_q" not in n):
            assert torch.equal(g2[n], g1[n]), n
