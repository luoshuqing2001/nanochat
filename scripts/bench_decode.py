"""
Model-level inference on one GPU with nanochat's KV cache (engine.KVCache, GPT.forward(kv_cache=...)):
prefill of a prompt, then single-token decode steps; softmax vs rexp_rmsnorm, random weights, bf16.

rexp_rmsnorm's KV-cache path runs the split-KV FA3 build (atomic reduction of the splits' partial U,
no combine kernel); "rexp (reference)" re-times its decode steps with that path disabled -- the
float32 torch reference it used before -- for comparison.

    python -m scripts.bench_decode [--depth 20] [--cases 1x2048,1x8192,1x32768,8x8192]
"""
import argparse
import statistics

import torch

import nanochat.fn_rmsnorm_attention as fna
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=20)
ap.add_argument("--cases", default="1x2048,1x8192,1x32768,8x8192")
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--vocab", type=int, default=32768)
args = ap.parse_args()
cases = [tuple(int(x) for x in c.split("x")) for c in args.cases.split(",")]
max_ctx = max(t for _, t in cases)


def build(kind):
    n_embd = args.depth * 64
    # sequence_len sets the long window: make it cover the longest context, so "L" layers are full
    cfg = GPTConfig(sequence_len=max_ctx, vocab_size=args.vocab, n_layer=args.depth, n_head=n_embd // 128,
                    n_kv_head=n_embd // 128, n_embd=n_embd, window_pattern="SSSL", attn_kind=kind)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    for b in model.transformer.h:
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.5 / n_embd ** 0.5)
        torch.nn.init.normal_(b.mlp.c_proj.weight, std=0.5 / (4 * n_embd) ** 0.5)
    return model.eval()


def event_ms(fn):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


@torch.no_grad()
def run(model, B, T, reference_decode=False):
    cfg = model.config
    kv = KVCache(batch_size=B, num_heads=cfg.n_kv_head, seq_len=T + args.steps + 8, head_dim=cfg.n_embd // cfg.n_head,
                 num_layers=cfg.n_layer, device="cuda", dtype=torch.bfloat16)
    prompt = torch.randint(0, args.vocab, (B, T), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefill_ms = event_ms(lambda: model.forward(prompt, kv_cache=kv))
        tok = torch.randint(0, args.vocab, (B, 1), device="cuda")
        saved = fna.SPLIT_BUILDS
        if reference_decode:
            fna.SPLIT_BUILDS = {}
        try:
            times = [event_ms(lambda: model.forward(tok, kv_cache=kv)) for _ in range(args.steps)]
        finally:
            fna.SPLIT_BUILDS = saved
    return prefill_ms, statistics.median(times[4:])


print(f"d{args.depth}, bf16, {torch.cuda.get_device_name()}; decode = median ms per step over {args.steps} steps")
models = {k: build(k) for k in ("softmax", "rexp_rmsnorm")}
for B, T in cases:
    for m in models.values():  # warm up: kernels, allocator, the first-call paths -- at this shape
        run(m, B, T)
    res = {}
    for k, m in models.items():  # prefill: median of 3 (a single prompt is one call)
        runs = [run(m, B, T) for _ in range(3)]
        res[k] = (statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs))
    ref = run(models["rexp_rmsnorm"], B, T, reference_decode=True)[1]
    (ps, ds), (pr, dr) = res["softmax"], res["rexp_rmsnorm"]
    print(f"B={B:<2} ctx={T:<6} prefill: softmax {ps:8.2f} ms  rexp {pr:8.2f} ms ({ps / pr:.3f}x) | "
          f"decode: softmax {ds:6.3f} ms  rexp {dr:6.3f} ms ({ds / dr:.3f}x)  rexp before (reference) {ref:7.3f} ms "
          f"({ref / dr:.1f}x slower than now)", flush=True)
