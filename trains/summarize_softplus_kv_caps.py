"""Summarize paired cap timings without selecting per-shape winners for dispatch."""
import json,math
from pathlib import Path

p=Path(__file__).with_name('softplus_kv_caps_full.json')
rows=json.loads(p.read_text())
names=('cap256_atomic','cap512_atomic','cap1024_atomic','cap2048_atomic',
       'cap256_global_atomic','cap512_global_atomic','cap1024_global_atomic',
       'cap2048_global_atomic','cap512_private','whole_lpt')
def gm(xs):return math.exp(sum(map(math.log,xs))/len(xs))
summary={}
for phase in ('prefill','train'):
    rr=[r for r in rows if r['phase']==phase]
    summary[phase]={}
    for name in names:
        ratios=[r['results']['default']['ms']/r['results'][name]['ms'] for r in rr]
        summary[phase][name]=dict(geomean=gm(ratios),minimum=min(ratios),maximum=max(ratios),cases=len(rr))
Path(__file__).with_name('softplus_kv_caps_summary.json').write_text(json.dumps(summary,indent=2))
report=Path(__file__).with_name('SOFTPLUS_KV_CAP_RESULTS.md')
s=report.read_text().split('\n## Measurements\n')[0]
s+='\n## Measurements\n\nSpeedup = original default / candidate, higher is better. '
s+=f'{len(rows)} paired cases.\n\n| Schedule | Prefill geomean | Train geomean |\n|---|---:|---:|\n'
for name in names:s+=f'| {name} | {summary["prefill"][name]["geomean"]:.3f}x | {summary["train"][name]["geomean"]:.3f}x |\n'
s+='\nSelected B1/H6/T2048 full-causal cases, milliseconds:\n\n| D | Phase | Default | Cap256 | Cap512 | Cap1024 | Whole LPT | Softmax tuned | Softmax LPT |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    if (r['b'],r['h'],r['t'],r['w'])==(1,6,2048,-1):
        names2=('default','cap256_atomic','cap512_atomic','cap1024_atomic','whole_lpt','softmax_tuned','softmax_global')
        s+=f'| {r["d"]} | {r["phase"]} | '+' | '.join(f'{r["results"][n]["ms"]:.4f}' for n in names2)+' |\n'
s+='''
The fixed cap guarantees a bound on KV iterations, not equal runtime for every CTA.
Short final segments, causal/window masking and atomic contention remain. A sliding
window of 512 visible keys per row can span more than 512 physical KV positions for
an M64 query tile, so a cap of 512 can still split that tile. Small caps increase
CTA startup work, Q reloads, and output traffic; they do not remove QK/PV work or
Softplus elementwise work. These are structural costs, not profiler measurements.
No default dispatch changes are made based on these experiments.
'''
report.write_text(s)
print(json.dumps(summary,indent=2))
