"""Aggregate both execution orders; keep candidate names fixed across shapes."""
import json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
paths=[root/'softplus_long_kv_major.json',root/'softplus_long_kv_major_reverse.json']
allrows=[json.loads(p.read_text()) for p in paths]
assert len(allrows[0])==len(allrows[1])==16
key=lambda r:(r['phase'],r['t'],r['h'],r['d'])
second={key(r):r for r in allrows[1]};rows=[]
for r in allrows[0]:
    other=second[key(r)];out={k:v for k,v in r.items() if k!='results'}
    out['results']={n:dict(ms=statistics.median(v['samples_ms']+other['results'][n]['samples_ms']),samples_ms=v['samples_ms']+other['results'][n]['samples_ms']) for n,v in r['results'].items()}
    rows.append(out)
summary={}
for phase in ('prefill','train'):
    rr=[r for r in rows if r['phase']==phase];summary[phase]={}
    for n in rr[0]['results']:
        ratios=[min(r['results']['softmax_tuned']['ms'],r['results']['softmax_global']['ms'])/r['results'][n]['ms'] for r in rr]
        summary[phase][n]=dict(geomean_vs_fa4=math.exp(sum(map(math.log,ratios))/len(ratios)),wins=sum(x>1 for x in ratios),cases=len(rr))
(root/'softplus_long_summary.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
s='''# Softplus long-context scheduling: 32K / 64K

NVIDIA GB10, BF16, B1, H1/H6, D64/D128, full causal. No default dispatch changes.

## Implementation

```python
out = softplus_attn_fa4_func(q, k, v, kv_split_size=1024,
                            stream_atomic=True, kv_major=True)
```

The new `kv_major` option reorders the same bounded KV tasks by KV start position,
then descending query tile. Each head visits different output owners sharing a KV
interval before moving to the next interval. This aims to reuse K/V and separate
updates to the same atomic output slot. It changes neither Softplus math nor tile
sizes, output workspace, or the number of CTAs. Short tiles retain direct writes.
The original query-major order remains selectable and remains the default.
Private partial and last-completer reductions also accept the new order.

Cache reuse and contention reduction are hypotheses motivated by task order, not
hardware-counter measurements. Reordering has no universal speedup guarantee.

## Method

Two processes with opposite initial variant orders, each with three alternating
trials. Aggregate timings are medians of the six samples. Only one CUDA graph is
alive at a time; every timing includes zeroing, reduction and output conversion.
Adaptive 1–10 replays target at least 30 ms per sample. Each candidate is warmed
before graph capture. Train means attention forward plus backward, not optimizer,
full-model training or data loading. No decode changes were made.

All candidates are checked against original Softplus outputs and gradients at
full sequence length. Eleven selected query positions are additionally checked
against PyTorch FP32 exact Softplus over the entire KV sequence. This avoids
allocating a dense T-by-T reference. Small regression cases cover compiled
gradients, unequal lengths, BF16/FP16, windows, strided inputs and graph replay.

FA4 is Softmax FA4 on this GPU, with the tuned SM120 tiles and a second control
using whole-query LPT scheduling. Comparisons below use the faster of these two
controls for each shape. Softplus and Softmax are different mathematical operators;
we compare latency, not numerical parity between them.

## Measurements

Milliseconds; lower is better. `1024 KV` is the new order. `8192 KV` changes both
segment size and order. `Whole LPT` is the previously implemented unsplit schedule.

| T | H | D | Phase | Default | 1024 original | 1024 KV | 8192 KV | Whole LPT | FA4 best |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
'''
for r in rows:
    v=r['results'];ns=('default','cap1024','cap1024_kv','cap8192_kv','whole_lpt')
    s+=f'| {r["t"]//1024}K | {r["h"]} | {r["d"]} | {r["phase"]} | '+' | '.join(f'{v[n]["ms"]:.3f}' for n in ns)+f' | {min(v["softmax_tuned"]["ms"],v["softmax_global"]["ms"]):.3f} |\n'
s+='\nGeometric mean speedup against the faster FA4 control, eight shapes per phase:\n\n| Schedule | Prefill | Train |\n|---|---:|---:|\n'
for n in ('default','whole_lpt','cap1024','cap1024_kv','cap4096_kv','cap8192_kv'):
    s+=f'| {n} | {summary["prefill"][n]["geomean_vs_fa4"]:.3f}x | {summary["train"][n]["geomean_vs_fa4"]:.3f}x |\n'
s+='''
## Interpretation and limits

At 64K/H6/D128, KV-major changes the 1024-cap prefill from 134.237 ms to
107.037 ms (1.254x), and forward+backward from 406.926 ms to 381.942 ms (1.065x).
However, default unsplit Softplus is still faster than both fixed-cap variants.
At 32K, KV-major does not produce a consistent improvement. The new option remains
experimental and is not automatically enabled. No tested Softplus prefill option
beats the faster FA4 control in these eight shapes. D64 training beats FA4 in all
four D64 cases, including with the preexisting default; D128 training does not.

Both KV-cap regression tests passed (21.282 s), followed by all seven existing
completion/scheduling regression tests (47.946 s). Across the first long-context
run, maximum normalized error versus the old Softplus outputs/gradients was
0.000178; sampled FP32 exact Softplus reference error was at most 0.00435.
Each independent reverse-order run also checks correctness before timing.

Longer sequences create more independent query CTAs even at B1/H1. On SM120 an
M64 tile gives 512 query tiles at 32K and 1024 at 64K, before multiplying by heads.
A fixed 1024 cap also makes total segment count grow approximately quadratically
with T. At 64K, H1, D128, there are 33,280 segment CTAs instead of 1024 whole-query
CTAs. Output atomic updates and Q reloads grow with the number of segments.
Increasing the cap reduces those costs; KV-major ordering cannot remove them.

Backward is unchanged, so a training win cannot be attributed entirely to this
forward scheduling change. Whole-query ordering also benefits Softmax. These
experiments do not establish universal Softplus superiority or a cross-GPU rule.

Reproduce:

```sh
python trains/bench_softplus_long.py --output trains/softplus_long_kv_major.json
python trains/bench_softplus_long.py --reverse --output trains/softplus_long_kv_major_reverse.json
python trains/summarize_softplus_long.py
```
'''
(root/'SOFTPLUS_LONG_RESULTS.md').write_text(s)
print(json.dumps(summary,indent=2))
