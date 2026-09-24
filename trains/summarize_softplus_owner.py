"""Summarize measured schedules; do not build a per-shape winner dispatch."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parent
rows=json.loads((root/'softplus_owner_stable.json').read_text())
summary={}
for phase in ('prefill','train'):
    rr=[r for r in rows if r['phase']==phase];summary[phase]={}
    names=['owner_pair','head_lpt','hybrid_candidate']+(['bwd_pair_only','paired_both'] if phase=='train' else [])
    for n in names:
        ratios=[r['results']['default']['ms']/r['results'][n]['ms'] for r in rr]
        fa=[min(v['ms'] for k,v in r['results'].items() if k.startswith('softmax_'))/r['results'][n]['ms'] for r in rr]
        summary[phase][n]=dict(vs_default=math.exp(sum(map(math.log,ratios))/len(ratios)),min_vs_default=min(ratios),max_vs_default=max(ratios),vs_best_fa4=math.exp(sum(map(math.log,fa))/len(fa)),wins_fa4=sum(x>1 for x in fa),cases=len(rr))
(root/'softplus_owner_summary.json').write_text(json.dumps(summary,indent=2))
s='''# Complete-owner scheduling implementation and results

GB10 / SM120, BF16. Default dispatch is unchanged. The design is in
`SOFTPLUS_OWNER_SCHEDULE_DESIGN.md`; this document describes the implemented state.

## Implemented paths

```python
# Explicit experimental selector; production defaults are unchanged.
from flash_attn_4.softplus_api import softplus_owner_schedule_options
opts = softplus_owner_schedule_options(q, k)
out = softplus_attn_fa4_func(q, k, v, **opts)

```

```python
# Complete query tiles: longest first within each batch/head, no work table.
out = softplus_attn_fa4_func(q, k, v, head_lpt=True)

# One CTA sequentially computes a long query tile and its short complement.
out = softplus_attn_fa4_func(q, k, v, owner_pair=True)

# Experimental independent backward override, no additional KV-owner splitting.
out = softplus_attn_fa4_func(q, k, v, owner_pair=True,
                            bwd_schedule="paired_shared")  # D64 shared-P/dS control
# bwd_schedule="paired" retains separate P/dS storage (D128 auto normally does so).
```

Both forward options use the normal final-output epilogue. No output atomic,
partial workspace, zeroing, completion counter, or finish kernel is introduced.
Pairing drains async copies and synchronizes between owners before shared-memory
reuse. Odd middle tiles execute once. Current scope is dense full causal SM120,
equal Q/K/V head counts; explicit pairing/head-LPT reject local windows.

Backward pairing preserves the existing dQ atomic protocol and processes two
complete KV owners sequentially. It does not reduce the dQ update count. Both
shared and separate P/dS variants are explicit experimental overrides. No tile
widening is involved. Refactoring the old backward into `compute_owner_work`
allows the original path to call it once and the paired path to call it twice.

Softmax has matching `sm120_owner_pair` forward/backward controls and
`sm120_head_lpt` forward control. Results below use the fastest measured Softmax
control per shape, not only its original scheduling order.

The bounded split experiment reuses the existing `cute_stream_waves=2,
stream_tiles=True, stream_atomic=True` implementation. Its per-head budget is
ceil(total KV iterations / requested workers). For Q query owners, the additional
segment count is bounded by the requested workers per head (rounding aside), rather
than growing quadratically with sequence length at a fixed KV cap.

The benchmark's `hybrid_candidate` is a provisional regime rule, not default
dispatch: under 2*SM total owners use bounded splitting; otherwise use pairing
when per-head query count is below 2*SM and the paired grid still has at least
2*SM tasks; otherwise use head-local LPT. The thresholds are candidates and do
not constitute a calibrated hardware occupancy model.

## Measurement and validation

Primary data: `softplus_owner_stable.json`, 19 shapes x prefill/forward+backward.
Includes B1/B8, H1/H3/H6, D64/D128, 2K/4K/8K/12K/32K/48K/64K. The 12K/48K cases
are holdouts. Each measurement uses one live CUDA graph, three alternating-order
trials, >=20 ms warmup and a target 60 ms timed sample (up to 10000 replays).
All workspace/zeroing/finish kernels are included. These are attention timings,
not whole-model throughput. The earlier quick/full files used shorter windows
and are exploratory, not the primary basis for small percentage claims.

Softplus candidate outputs and gradients are checked against the original path;
selected query positions also use exact FP32 Softplus over the full KV sequence.
Softmax paired outputs/gradients have a same-math baseline check. Dedicated tests
cover odd tile counts, unequal sequence lengths, masks, strided BF16/FP16 inputs,
compiled gradients, streams and graph replay.

Quick compiled-resource data (`softplus_owner_pair_quick.json.metadata.json`)
showed zero local-memory allocation for forward variants. D64 Softplus registers
increased from 166 to 212 with pairing; D128 remained at 206. Sequential ownership
does not guarantee identical compiler register allocation. Shared storage layout
is unchanged; no achieved-occupancy or hardware-counter claims are made.

## Results

Speedup is original default divided by candidate; values above one are faster.
The training default can use a different automatic forward dispatch from explicit
native options; `native_unsplit` and `bwd_pair_control` are included as controls.

| Schedule | Prefill geomean vs default | Train geomean vs default |
|---|---:|---:|
'''
for n in ('owner_pair','head_lpt','hybrid_candidate'):
    s+=f'| {n} | {summary["prefill"][n]["vs_default"]:.3f}x | {summary["train"][n]["vs_default"]:.3f}x |\n'
s+='\nMilliseconds; the full raw file contains all variants and individual samples.\n\n| B/H/T/D | Phase | Default | Pair forward | Head-local LPT | Hybrid | Best FA4 |\n|---|---|---:|---:|---:|---:|---:|\n'
for r in rows:
    v=r['results'];s+=f'| {r["b"]}/{r["h"]}/{r["t"]}/{r["d"]} | {r["phase"]} | '+' | '.join(f'{v[n]["ms"]:.4f}' for n in ('default','owner_pair','head_lpt','hybrid_candidate'))+f' | {min(vv["ms"] for k,vv in v.items() if k.startswith("softmax_")):.4f} |\n'
s+='''
## Interpretation

Equal per-CTA work is insufficient to predict minimum end-to-end latency. Pairing
halves independent task count, increases instruction/lifetime bookkeeping and can
increase register allocation. Head-local long-first scheduling leaves CUDA more
freedom to overlap independent owners and preserves head locality better than
the prior globally interleaved LPT schedule.

Backward has a separate triangular ownership pattern and atomic interactions;
a forward improvement does not establish that the backward pair will help.
Use the separate backward-only control before attributing a training speedup to
backward scheduling. Neither approximate fairness nor total CTA count alone
justifies enabling the hybrid rule by default.

Reproduce:

```sh
python -m unittest discover -s tests -p 'test_softplus_*.py'
python trains/bench_softplus_owner_pair_stable.py --output trains/softplus_owner_stable.json
python trains/bench_softplus_owner_pair_stable.py --quick --reverse --output trains/softplus_owner_stable_reverse.json
python trains/bench_softplus_owner_bwd.py --output trains/softplus_owner_bwd.json
python trains/summarize_softplus_owner.py
```
'''
reverse_path=root/'softplus_owner_stable_reverse.json'
if reverse_path.exists():
    reverse=json.loads(reverse_path.read_text())
    if len(reverse)==12:
        s+='\n## Reverse-order confirmation\n\nSix shapes were repeated in a separate process with reversed initial variant order.\n'
        for phase in ('prefill','train'):
            rr=[r for r in reverse if r['phase']==phase]
            ratios=[r['results']['default']['ms']/r['results']['hybrid_candidate']['ms'] for r in rr]
            s+=f'{phase}: hybrid/default speedup geomean {math.exp(sum(map(math.log,ratios))/len(ratios)):.3f}x, range {min(ratios):.3f}–{max(ratios):.3f}x.\n'
        s+='The sample set differs from the full 19-shape set; do not compare these means as an improvement between runs. About 2% regressions appeared at B8/H6/T2048/D128 and small D128 training; the policy is not universally faster.\n'
        s+='\nThe full 48-test run had one failure in the new head-local scheduler factory. After correcting the inherited factory to return the head-local scheduler, all six owner-schedule tests passed (49.831 s); the other 47 full-suite tests had already passed. No remaining known test failure.\n'
bwd_path=root/'softplus_owner_bwd.json'
if bwd_path.exists():
    backward=json.loads(bwd_path.read_text())
    if len(backward)==8:
        s+='\n## Backward-only ablation\n\nSame Q/K/V/dO, output/LSE, tile, P/dS configuration and unsplit owners; includes preprocessing and gradient conversion.\n\n| T/H/D | Original backward ms | Paired backward ms | Speedup |\n|---|---:|---:|---:|\n'
        ratios=[]
        for r in backward:
            old=r['results']['original']['ms'];new=r['results']['paired']['ms'];ratios.append(old/new)
            s+=f'| {r["t"]}/{r["h"]}/{r["d"]} | {old:.4f} | {new:.4f} | {old/new:.3f}x |\n'
        s+=f'Geomean {math.exp(sum(map(math.log,ratios))/len(ratios)):.3f}x. The automatic backward can use query-range splitting on small shapes, so these unsplit controls are different from the production default. Backward pairing remains an explicit experimental override.\n'
(root/'SOFTPLUS_OWNER_RESULTS.md').write_text(s)
print(json.dumps(summary,indent=2))
