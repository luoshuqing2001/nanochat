"""
Training diagnostics: gradient, update and activation statistics.

base_train.py logs loss and throughput; this module adds the internals you need to
tell a healthy run from one that is quietly diverging:

  * gradient L2 norms, globally and per optimizer kind (Muon / AdamW) and per layer
  * update-to-parameter ratios, ||dw|| / ||w||, the standard Muon tuning signal
  * attention log-sum-exp, an upper bound on the max attention logit (flash kernels
    never materialize the logits, so the exact max needs a small recompute, which
    this module does on a subsample)
  * per-layer activation RMS (residual stream, attention output, MLP output) and the
    learnable residual scalars

Everything is opt-in and periodic: see --diagnostics-every in scripts/base_train.py.
All the heavy work happens under no_grad on the uncompiled model, so the compiled
training step is untouched.
"""

import json

import torch

import nanochat.gpt as gpt_module


# -----------------------------------------------------------------------------
# Gradients and updates
# -----------------------------------------------------------------------------
@torch.no_grad()
def grad_norms(model, optimizer):
    """L2 norms of the current .grad values. One CPU sync for the whole report."""
    per_kind_sq, layer_sq = {}, {}

    for group in optimizer.param_groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        if not grads:
            continue
        sq = torch.stack(torch._foreach_norm(grads)).pow(2).sum()
        kind = group["kind"]
        per_kind_sq[kind] = per_kind_sq[kind] + sq if kind in per_kind_sq else sq

    for name, p in model.named_parameters():
        if p.grad is None or ".h." not in name:
            continue
        layer = int(name.split(".h.")[1].split(".")[0])
        sq = p.grad.detach().float().pow(2).sum()
        layer_sq[layer] = layer_sq[layer] + sq if layer in layer_sq else sq

    kinds = sorted(per_kind_sq)
    layers = sorted(layer_sq)
    if not kinds:
        return {}
    packed = torch.stack([per_kind_sq[k] for k in kinds] + [layer_sq[i] for i in layers])
    packed = packed.sqrt().cpu()  # the single sync point
    out = {f"grad_norm/{k}": packed[i].item() for i, k in enumerate(kinds)}
    out["grad_norm/global"] = float(sum(v * v for v in out.values()) ** 0.5)
    per_layer = [packed[len(kinds) + i].item() for i in range(len(layers))]
    if per_layer:
        out["grad_norm/layer_min"] = min(per_layer)
        out["grad_norm/layer_max"] = max(per_layer)
        out["grad_norm/layers"] = per_layer
    return out


class UpdateTracker:
    """Measures ||dw|| / ||w|| across an optimizer step.

    snapshot() clones the parameters (fp32 masters: ~1.1 GB for d12) and ratios()
    frees them again, so this only costs memory on diagnostic steps.
    """

    def __init__(self):
        self._saved = None

    @torch.no_grad()
    def snapshot(self, optimizer):
        self._saved = [
            (group["kind"], [p.detach().clone() for p in group["params"]])
            for group in optimizer.param_groups
        ]

    @torch.no_grad()
    def ratios(self, optimizer):
        if self._saved is None:
            return {}
        delta_sq, weight_sq = {}, {}
        for (kind, before), group in zip(self._saved, optimizer.param_groups):
            after = list(group["params"])
            if not after:
                continue
            deltas = torch._foreach_sub(after, before)
            d = torch.stack(torch._foreach_norm(deltas)).pow(2).sum()
            w = torch.stack(torch._foreach_norm(after)).pow(2).sum()
            delta_sq[kind] = delta_sq[kind] + d if kind in delta_sq else d
            weight_sq[kind] = weight_sq[kind] + w if kind in weight_sq else w
        self._saved = None  # free the clones
        kinds = sorted(delta_sq)
        if not kinds:
            return {}
        packed = torch.stack(
            [delta_sq[k] for k in kinds] + [weight_sq[k] for k in kinds]
        ).sqrt().cpu()
        n = len(kinds)
        return {
            f"update_ratio/{k}": (packed[i].item() / max(packed[n + i].item(), 1e-12))
            for i, k in enumerate(kinds)
        }


# -----------------------------------------------------------------------------
# Activations, attention logits and the residual scalars
# -----------------------------------------------------------------------------
class _AttentionSpy:
    """Stands in for nanochat.gpt.flash_attn during a diagnostic forward.

    Records the softmax log-sum-exp per layer (an upper bound on the max logit:
    max_logit <= lse <= max_logit + log(n_keys)) and, on a subsample of queries,
    the exact max logit.
    """

    def __init__(self, real, max_query_sample=128):
        self._real = real
        self._max_query_sample = max_query_sample
        self.lse_max, self.lse_mean, self.logit_max = [], [], []
        self.flash_attn_with_kvcache = real.flash_attn_with_kvcache

    def flash_attn_func(self, q, k, v, causal=False, window_size=(-1, -1)):
        lse = None
        try:
            import flash_attn_4.fa3_compat as fa4
            if fa4.has_custom_op() and fa4.can_run(q, window_size):
                left, right = fa4._to_fa4_window(window_size, k.size(1))
                out, lse = torch.ops.flash_attn_4.attn_fwd(
                    q, k, v, q.shape[-1] ** -0.5, causal,
                    -1 if left is None else left, -1 if right is None else right,
                )
        except Exception:
            lse = None
        if lse is None:
            out = self._real.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
        else:
            self.lse_max.append(lse.max())
            self.lse_mean.append(lse.mean())

        # exact max logit on a few queries: (n, H, D) @ (B, H, D, T) stays small
        n = min(self._max_query_sample, q.size(1))
        qs = q[:1, -n:].transpose(1, 2).float()          # (1, H, n, D)
        ks = k[:1].transpose(1, 2).float()               # (1, H, T, D)
        logits = torch.matmul(qs, ks.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        t_q, t_k = logits.size(-2), logits.size(-1)
        row = torch.arange(t_k - t_q, t_k, device=q.device).unsqueeze(1)
        col = torch.arange(t_k, device=q.device).unsqueeze(0)
        allowed = col <= row
        window = window_size[0]
        if 0 <= window < t_k:
            allowed = allowed & ((row - col) <= window)
        self.logit_max.append(logits.masked_fill(~allowed, float("-inf")).max())
        return out


@torch.no_grad()
def activation_report(model, x, max_query_sample=128):
    """One uncompiled forward with hooks. `model` must be the raw (uncompiled) GPT."""
    stats, handles = {}, []
    block_out, attn_out, mlp_out = [], [], []

    def rms(t):
        return t.detach().float().pow(2).mean().sqrt()

    for block in model.transformer.h:
        handles.append(block.register_forward_hook(lambda m, i, o: block_out.append(rms(o))))
        handles.append(block.attn.register_forward_hook(lambda m, i, o: attn_out.append(rms(o))))
        handles.append(block.mlp.register_forward_hook(lambda m, i, o: mlp_out.append(rms(o))))

    real = gpt_module.flash_attn
    spy = _AttentionSpy(real, max_query_sample=max_query_sample)
    gpt_module.flash_attn = spy
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        gpt_module.flash_attn = real
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    def pack(name, values):
        if not values:
            return
        v = torch.stack(values).float().cpu()
        stats[f"{name}/min"] = v.min().item()
        stats[f"{name}/max"] = v.max().item()
        stats[f"{name}/per_layer"] = [round(t, 5) for t in v.tolist()]

    pack("act_rms/block", block_out)
    pack("act_rms/attn", attn_out)
    pack("act_rms/mlp", mlp_out)
    pack("attn_lse/max", spy.lse_max)
    pack("attn_lse/mean", spy.lse_mean)
    pack("attn_logit/max", spy.logit_max)

    # learnable residual scalars (this fork's whole point)
    for name in ("resid_lambdas", "x0_lambdas"):
        p = getattr(model, name, None)
        if p is not None:
            v = p.detach().float().cpu()
            stats[f"scalars/{name}"] = [round(t, 5) for t in v.tolist()]
            stats[f"scalars/{name}/min"] = v.min().item()
            stats[f"scalars/{name}/max"] = v.max().item()
    for name in ("smear_lambda", "backout_lambda"):
        p = getattr(model, name, None)
        if p is not None:
            stats[f"scalars/{name}"] = p.detach().float().flatten()[0].item()
    return stats


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def format_line(step, stats):
    """A compact human-readable line; the full record goes out as JSON beside it."""
    def g(key, default=float("nan")):
        return stats.get(key, default)

    parts = [f"diag {step:05d}"]
    if "grad_norm/global" in stats:
        parts.append(
            f"gnorm {g('grad_norm/global'):.3e} (muon {g('grad_norm/muon'):.2e} "
            f"adamw {g('grad_norm/adamw'):.2e}, layers {g('grad_norm/layer_min'):.1e}"
            f"-{g('grad_norm/layer_max'):.1e})"
        )
    if "update_ratio/muon" in stats:
        parts.append(f"upd/par muon {g('update_ratio/muon'):.2e} adamw {g('update_ratio/adamw'):.2e}")
    if "attn_logit/max/max" in stats:
        parts.append(f"attn logit max {g('attn_logit/max/max'):.2f} lse max {g('attn_lse/max/max'):.2f}")
    if "act_rms/block/max" in stats:
        parts.append(f"act rms block {g('act_rms/block/min'):.2f}-{g('act_rms/block/max'):.2f} "
                     f"attn {g('act_rms/attn/max'):.2f} mlp {g('act_rms/mlp/max'):.2f}")
    if "scalars/resid_lambdas/min" in stats:
        parts.append(f"resid_lambda {g('scalars/resid_lambdas/min'):.3f}-{g('scalars/resid_lambdas/max'):.3f} "
                     f"x0_lambda {g('scalars/x0_lambdas/min'):.3f}-{g('scalars/x0_lambdas/max'):.3f}")
    return " | ".join(parts)


def format_json(step, stats):
    """One machine-readable line, picked up by the launcher's log parser."""
    record = {"type": "diag", "step": step}
    record.update(stats)
    return "diag_json " + json.dumps(record)
