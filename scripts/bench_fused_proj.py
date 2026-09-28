"""
NANOCHAT_FA3_FUSE_PROJ or NANOCHAT_FA3_DQFUSE (--toggle) on/off: step time on one GPU. Not training -- random tokens, forward + backward
only (the optimizer is the same either way), no checkpoints, nothing written.

    python -m scripts.bench_fused_proj [--depths 12,16,20] [--kind rexp_rmsnorm] [--batch 32]

1) the attention sublayer alone (attention + gain + FP8 c_proj), compiled, per window size;
2) the whole model as base_train builds it (--fp8, torch.compile, hybrid SWA "SSSL").
Timings: median of --reps, each over 10 back-to-back iterations; on/off alternate to cancel drift.
"""
import argparse
import statistics

import torch

import nanochat.fn_rmsnorm_attention as fna
from nanochat.fp8 import Float8Linear, convert_to_float8_training
from nanochat.gpt import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="12,16,20")
ap.add_argument("--kind", default="rexp_rmsnorm")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--vocab", type=int, default=32768)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--sublayer-only", action="store_true")
ap.add_argument("--toggle", default="fuse_proj", choices=["fuse_proj", "dqfuse", "dgain", "dgain2"],
                help="what is off/on: NANOCHAT_FA3_FUSE_PROJ (attention + gain + FP8 c_proj op) or "
                     "NANOCHAT_FA3_DQFUSE (dQ memset + postprocess folded into the attention backward) or "
                     "NANOCHAT_FA3_DGAIN (attention + gain as one autograd node, bit-identical), or "
                     "dgain2: NANOCHAT_FA3_DGAIN 1 -> 2 (dgamma reduced in the attention preprocess)")
args = ap.parse_args()


FLAG = {"fuse_proj": "FUSE_PROJ", "dqfuse": "DQFUSE", "dgain": "DGAIN", "dgain2": "DGAIN_MODE"}[args.toggle]
VALUE = {False: "1", True: "2"} if args.toggle == "dgain2" else {False: False, True: True}


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


def compare(make_step):
    """make_step(fuse) -> zero-arg callable. Returns (off_ms, on_ms)."""
    # no torch._dynamo.reset() in between: compilation is lazy, so a reset makes the first step
    # retrace later, with whatever FUSE_PROJ holds then. Instead every call sets the flag, and
    # dynamo's guard on it keeps one compiled graph per setting.
    torch._dynamo.reset()  # per comparison, so earlier shapes don't use up the recompile limit
    steps = {}
    for fuse in (False, True):
        inner = make_step(fuse)
        def step(inner=inner, fuse=fuse):
            setattr(fna, FLAG, VALUE[fuse])
            inner()
        steps[fuse] = step
        for _ in range(3):
            step()                 # compile + warm up with this setting
        calls, real_ops = [], fna._ops
        fna._ops = lambda b: calls.append(b) or real_ops(b)  # which builds the step calls (looked up per call)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        fna._ops = real_ops
        names = [e.key for e in prof.key_averages()]
        if args.toggle == "fuse_proj":
            used = any("proj_bwd_rmsnorm" in n for n in names)
        elif args.toggle == "dqfuse":  # the postprocess kernel is what dqfuse removes
            used = not any("PostprocessConvertdQ" in n for n in names)
        elif args.toggle == "dgain":  # a *_g build ran during this step
            used = any(b.endswith("_g") for b in calls)
        else:  # dgain2: a *_gr build ran
            used = any(b.endswith("_gr") for b in calls)
        assert used == fuse, f"{FLAG}={fuse} but the {args.toggle} path used={used}"
    torch.cuda.synchronize()
    t = {False: [], True: []}
    for _ in range(args.reps):
        for fuse in (False, True):
            t[fuse].append(timed(steps[fuse]))
    return statistics.median(t[False]), statistics.median(t[True])


def sublayer(n_embd, window):
    H, D = n_embd // 128, 128
    B, T = args.batch, args.seq
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    gamma = torch.nn.Parameter(1 + 0.1 * torch.randn(n_embd, device="cuda"))
    proj = Float8Linear(n_embd, n_embd, bias=False, device="cuda")
    dout = torch.randn(B, T, n_embd, device="cuda", dtype=torch.bfloat16)

    def make_step(fuse):
        def f(q, k, v):
            if fuse and args.toggle == "fuse_proj":
                return fna.fn_rmsnorm_attn_proj_fp8(q, k, v, args.kind, gamma, proj.weight, window_size=window)
            y = fna.fn_rmsnorm_attn_func(q, k, v, args.kind, window_size=window)
            return proj(y.reshape(B, T, n_embd) * gamma.to(y.dtype))
        fc = torch.compile(f, dynamic=False)
        return lambda: fc(q, k, v).backward(dout)
    return compare(make_step)


def whole_model(depth):
    n_embd = depth * 64
    cfg = GPTConfig(sequence_len=args.seq, vocab_size=args.vocab, n_layer=depth, n_head=n_embd // 128,
                    n_kv_head=n_embd // 128, n_embd=n_embd, window_pattern="SSSL", attn_kind=args.kind)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    for b in model.transformer.h:   # a trained-like state: c_proj is zero at init
        torch.nn.init.normal_(b.attn.c_proj.weight, std=0.5 / n_embd ** 0.5)
        torch.nn.init.normal_(b.mlp.c_proj.weight, std=0.5 / (4 * n_embd) ** 0.5)
    convert_to_float8_training(model, module_filter_fn=fp8_filter)
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")

    def make_step(fuse):
        mc = torch.compile(model, dynamic=False)
        def step():
            mc(x, y).backward()
            model.zero_grad(set_to_none=True)
        return step
    out = compare(make_step)
    del model
    torch.cuda.empty_cache()
    return out


print(f"toggle={args.toggle} kind={args.kind} batch={args.batch} x {args.seq}, {torch.cuda.get_device_name()}")
for depth in (int(d) for d in args.depths.split(",")):
    n_embd = depth * 64
    short = -(-args.seq // 4 // 128) * 128
    for name, window in ((("S", (short, 0)), ("L", (args.seq, 0))) if not args.toggle.startswith("dgain") else ()):
        off, on = sublayer(n_embd, window)
        print(f"d{depth} attention sublayer fwd+bwd, window {name}={window[0]:4d}: "
              f"off {off:7.3f} ms  on {on:7.3f} ms  ({off / on - 1:+.2%})", flush=True)
    if args.sublayer_only:
        continue
    off, on = whole_model(depth)
    print(f"d{depth} whole model fwd+bwd (x{depth} layers): "
          f"off {off:7.2f} ms  on {on:7.2f} ms  ({off / on - 1:+.2%})", flush=True)
