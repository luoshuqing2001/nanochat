"""--attn-gain 0: the *_rmsnorm kinds without the output RMSNorm's learned gain. Same forward as the gain at
its identity init (bit for bit: gamma = 1 multiplies nothing), same gradients up to the FA3 dQ atomics'
run-to-run noise, no attn_gamma parameter, and the optimizer setup still accounts for every parameter."""
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@pytest.mark.parametrize("kind", ["rexp_rmsnorm", "softmax_rmsnorm"])
def test_nogain_matches_identity_gain(kind):
    if kind == "rexp_rmsnorm" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("rexp_rmsnorm is FA3 (SM90)")
    cfg = dict(sequence_len=256, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
               window_pattern="L", attn_kind=kind)
    torch.manual_seed(0)
    ref = GPT(GPTConfig(**cfg, attn_gain=True)).cuda()
    ref.init_weights()
    model = GPT(GPTConfig(**cfg, attn_gain=False)).cuda()
    assert all(b.attn.attn_gamma is None for b in model.transformer.h)
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and all("attn_gamma" in k for k in unexpected) and unexpected
    idx = torch.randint(0, 512, (2, 256), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l_ref, l = ref(idx, targets=idx), model(idx, targets=idx)
    l_ref.backward(); l.backward()
    assert torch.equal(l_ref, l)
    g_ref = dict(ref.named_parameters())
    for n, p in model.named_parameters():
        err = (p.grad.float() - g_ref[n].grad.float()).norm() / g_ref[n].grad.float().norm().clamp_min(1e-30)
        assert err < 1e-2, (n, err.item())
    model.setup_optimizer()  # its parameter-count assert covers the missing gains
