"""
Whole-model step time on one GPU, softmax vs rexp_rmsnorm (or any --kinds), as base_train builds the
model (--fp8, torch.compile, hybrid SWA "SSSL", head_dim 128). Not training: random tokens, nothing
written. Two measurements per kind:

  train    forward + backward (the optimizer step is identical for every kind, so left out)
  prefill  inference forward over a full batch of prompts (no_grad, no KV cache)

Median of --reps rounds of 10 back-to-back iterations; the kinds alternate every round.

    python -m scripts.bench_attn_kinds [--depths 12,16,20] [--kinds softmax,rexp_rmsnorm] [--batch 32]
"""
import argparse
import statistics

import torch

from nanochat.fp8 import convert_to_float8_training
from nanochat.gpt import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="12,16,20")
ap.add_argument("--kinds", default="softmax,rexp_rmsnorm")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--prefill-batch", type=int, default=8)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--vocab", type=int, default=32768)
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()


def fp8_filter(m, fqn):  # base_train's
    return (isinstance(m, torch.nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0
            and min(m.in_features, m.out_features) >= 128)


def timed(fn, iters=10):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def build(depth, kind):
    n_embd = depth * 64
    cfg = GPTConfig(sequence_len=args.seq, vocab_size=args.vocab, n_layer=depth, n_head=n_embd // 128,
                    n_kv_head=n_embd // 128, n_embd=n_embd, window_pattern="SSSL", attn_kind=kind)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    for b in model.transformer.h:  # trained-like: c_proj is zero at init
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.5 / n_embd ** 0.5)
        torch.nn.init.normal_(b.mlp.c_proj.weight, std=0.5 / (4 * n_embd) ** 0.5)
    convert_to_float8_training(model, module_filter_fn=fp8_filter)
    return model


print(f"batch {args.batch} x {args.seq} (train), {args.prefill_batch} x {args.seq} (prefill), {torch.cuda.get_device_name()}")
kinds = args.kinds.split(",")
for depth in (int(d) for d in args.depths.split(",")):
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    xp = x[:args.prefill_batch]
    steps = {}
    for kind in kinds:
        torch._dynamo.reset()
        model = build(depth, kind)
        mc = torch.compile(model, dynamic=False)

        def train_step(model=model, mc=mc):
            mc(x, x).backward()
            model.zero_grad(set_to_none=True)

        def prefill(mc=mc):
            with torch.no_grad():
                mc(xp)

        for _ in range(3):
            train_step(); prefill()
        torch.cuda.synchronize()
        steps[kind] = (model, train_step, prefill)
    t = {(k, p): [] for k in kinds for p in ("train", "prefill")}
    for r in range(args.reps):
        for kind in (kinds if r % 2 == 0 else kinds[::-1]):
            _, train_step, prefill = steps[kind]
            t[(kind, "train")].append(timed(train_step))
            t[(kind, "prefill")].append(timed(prefill))
    base = kinds[0]
    for p in ("train", "prefill"):
        cells = []
        for kind in kinds:
            ms = statistics.median(t[(kind, p)])
            rel = statistics.median(t[(base, p)]) / ms
            cells.append(f"{kind} {ms:8.2f} ms" + ("" if kind == base else f" ({base}/{kind} = {rel:.3f})"))
        print(f"d{depth} {p:8s}: " + "  ".join(cells), flush=True)
    del steps
    torch.cuda.empty_cache()
