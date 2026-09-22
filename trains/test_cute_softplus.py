"""Correctness of the CuTe softplus forward in flash_attn_4 vs a PyTorch reference."""
import sys, math
sys.path.insert(0, "/home/shuqing/nanochat")
import torch
import torch.nn.functional as F
from flash_attn_4.softplus_api import softplus_attn_fa4


def ref(q, k, v, causal, window_left, alpha, scale):
    # q,k,v: (B, T, H, D) -> (B, H, T, D). Tq may differ from Tk (decode), in which
    # case the causal mask is bottom-right aligned, as FA4 defines it.
    qf, kf, vf = (x.transpose(1, 2).float() for x in (q, k, v))
    Tq, Tk = qf.shape[-2], kf.shape[-2]
    s = (qf @ kf.transpose(-1, -2)) * scale
    i = torch.arange(Tq, device=q.device)[:, None] + (Tk - Tq)
    j = torch.arange(Tk, device=q.device)[None, :]
    keep = i >= j
    if window_left is not None:
        keep &= i - j <= window_left
    p = F.softplus(s) * keep
    n = keep.sum(-1).clamp(min=1).float()
    o = (p @ vf) / n[None, None, :, None] ** alpha
    return o.transpose(1, 2)


def run(B, T, H, D, causal, window_left, alpha, splits=1, seed=0, Tq=None):
    torch.manual_seed(seed)
    Tq = T if Tq is None else Tq
    q = torch.randn(B, Tq, H, D, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    scale = 1.0 / math.sqrt(D)
    out = softplus_attn_fa4(
        q, k, v, causal=True, window_size=(window_left, 0), alpha=alpha,
        softmax_scale=scale, num_splits=splits,
    )
    r = ref(q, k, v, causal, window_left, alpha, scale)
    err = (out.float() - r).abs().max().item()
    den = r.abs().max().item()
    tag = f"B{B} Tq{Tq} Tk{T} H{H} D{D} W={window_left} a={alpha} splits={splits}"
    print(f"{tag:<52} max_abs_err={err:.3e}  rel={err/max(den,1e-9):.3e}  "
          f"{'OK' if err/max(den,1e-9) < 2e-2 else 'FAIL'}")
    return err / max(den, 1e-9)


if __name__ == "__main__":
    bad = 0
    for args in [
        (1, 256, 1, 64, True, None, 1.0),
        (2, 512, 4, 64, True, None, 1.0),
        (1, 512, 2, 128, True, None, 1.0),
        (2, 1024, 6, 128, True, None, 1.0),
        (2, 1024, 6, 128, True, None, 0.5),
        (1, 300, 3, 64, True, None, 1.0),     # ragged seqlen
        (2, 1024, 4, 128, True, 255, 1.0),    # SWA
        (2, 2048, 6, 128, True, 511, 1.0),    # nanochat's S layers
        # tile splitting + atomic-add aggregation
        (2, 1024, 6, 128, True, None, 1.0, 2),
        (2, 1024, 6, 128, True, None, 1.0, 4),
        (2, 1024, 6, 128, True, None, 0.5, 4),
        (1, 256, 1, 64, True, None, 1.0, 4),   # more splits than key blocks
        (2, 2048, 6, 128, True, 511, 1.0, 4),  # SWA + splits
        (2, 2048, 6, 128, True, 511, 1.0, 8),
        # long-context inference: one query token against a KV cache, split
        (1, 4096, 6, 128, True, None, 1.0, 8, 0, 1),
        (1, 16384, 12, 128, True, None, 1.0, 8, 0, 1),
        (1, 4096, 6, 128, True, None, 1.0, 1, 0, 1),   # same, unsplit, as a cross-check
        (1, 8192, 1, 128, True, None, 1.0, 4),         # single-sequence prefill, split
    ]:
        try:
            if run(*args) >= 2e-2:
                bad += 1
        except Exception as e:
            print(f"{args} -> EXC {type(e).__name__}: {str(e)[:300]}")
            bad += 1
    print("FAILURES:", bad)
