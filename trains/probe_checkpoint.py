"""
Offline model-internals probe: run the diagnostics on saved checkpoints.

The in-loop diagnostics (--diagnostics-every) only exist for runs started after they
were added, and they sample while training. This probe works on any checkpoint, so it
can be pointed at a finished run, at an older experiment, or at several steps of the
same run to see how the internals moved.

    # last checkpoint of a run
    python trains/probe_checkpoint.py --model-tag d12_hybridswa_muon_20260920_202806

    # specific steps, and dump the full per-layer records
    python trains/probe_checkpoint.py --model-tag <tag> --steps 500,1000,1500 \
        --out logs/<run_id>/probe.json

What it reports per checkpoint: gradient L2 norms (a real backward on a fixed batch,
per optimizer kind and per layer), attention max logit and log-sum-exp per layer,
per-layer activation RMS (residual stream / attention out / MLP out), and the learnable
residual scalars.

NOTE: this needs the GPU. Do not run it against a live training job -- they will fight
over the device and both get slower.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat import diagnostics
from nanochat.checkpoint_manager import find_last_step, load_model
from nanochat.common import get_base_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer


def probe_step(model, x, y, with_grads=True):
    stats = diagnostics.activation_report(model, x)
    if with_grads:
        model.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer = model.setup_optimizer()  # groups only; never stepped
        stats.update(diagnostics.grad_norms(model, optimizer))
        stats["loss"] = loss.item()
        model.zero_grad(set_to_none=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", type=str, required=True, help="checkpoint dir name under base_checkpoints/")
    ap.add_argument("--steps", type=str, default="", help="comma-separated steps (default: the last one)")
    ap.add_argument("--batch-size", type=int, default=8, help="batch for the probe forward/backward")
    ap.add_argument("--no-grads", action="store_true", help="skip the backward pass (activations only)")
    ap.add_argument("--out", type=str, default="", help="write the full records here as JSON")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")

    ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", args.model_tag)
    assert os.path.isdir(ckpt_dir), f"no such checkpoint dir: {ckpt_dir}"
    steps = ([int(s) for s in args.steps.split(",") if s.strip()]
             or [find_last_step(ckpt_dir)])

    tokenizer = get_tokenizer()
    records = []
    for step in steps:
        model, _tok, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=step)
        seq_len = meta["model_config"]["sequence_len"]
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.batch_size, seq_len, split="val", device=device)
        x, y = next(loader)

        for p in model.parameters():
            p.requires_grad_(not args.no_grads)
        stats = probe_step(model, x, y, with_grads=not args.no_grads)
        stats["step"] = step
        stats["model_tag"] = args.model_tag
        records.append(stats)

        print(f"\n=== {args.model_tag} @ step {step} "
              f"(val_bpb at save: {meta.get('val_bpb')}) ===")
        print(diagnostics.format_line(step, stats))
        for key, label in [("attn_logit/max/per_layer", "attn max logit"),
                           ("attn_lse/max/per_layer", "attn lse max"),
                           ("act_rms/block/per_layer", "block out RMS"),
                           ("act_rms/attn/per_layer", "attn out RMS"),
                           ("act_rms/mlp/per_layer", "mlp out RMS"),
                           ("grad_norm/layers", "grad norm"),
                           ("scalars/resid_lambdas", "resid_lambda"),
                           ("scalars/x0_lambdas", "x0_lambda")]:
            if key in stats:
                vals = " ".join(f"{v:7.3f}" for v in stats[key])
                print(f"  {label:<15}: {vals}")
        del model
        torch.cuda.empty_cache()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
