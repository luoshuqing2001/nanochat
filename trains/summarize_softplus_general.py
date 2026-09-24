"""Summarize recorded timings without rerunning GPU work."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def gm(xs):
    return math.exp(sum(map(math.log,xs))/len(xs)) if xs else None

def main():
    before=json.loads((ROOT/'softplus_general_before_gb10.json').read_text())
    old={r['id']:r for r in before['rows'] if 'results' in r}
    records=[]
    for group,file in [('main','softplus_general_after_gb10.json'),('holdout','softplus_general_holdout_gb10.json')]:
        for r in json.loads((ROOT/file).read_text())['rows']:
            vals=r['results'];sp=vals['softplus']['ms']
            best=min((k for k in vals if k!='softplus'),key=lambda k:vals[k]['ms'])
            production='softmax_fa4' if r['phase']=='train' else 'softmax_sdpa'
            records.append(dict(group=group,id=r['id'],phase=r['phase'],b=r['b'],h=r['h'],tq=r['tq'],tk=r['tk'],d=r['d'],w=r['w'],softplus_ms=sp,
                best_softmax=best,best_softmax_ms=vals[best]['ms'],speedup_best=vals[best]['ms']/sp,
                speedup_production=vals[production]['ms']/sp,
                speedup_previous=old[r['id']]['results']['softplus']['ms']/sp if r['id'] in old else None,
                eager_speedup_production=vals[production].get('eager_ms',float('nan'))/vals['softplus'].get('eager_ms',float('nan'))))
    summary=[]
    for group in ('main','holdout','all'):
        for phase in ('prefill','decode','train'):
            rows=[r for r in records if r['phase']==phase and (group=='all' or r['group']==group)]
            summary.append(dict(group=group,phase=phase,n=len(rows),
                gm_previous=gm([r['speedup_previous'] for r in rows if r['speedup_previous'] is not None]),
                gm_production=gm([r['speedup_production'] for r in rows]),
                gm_best=gm([r['speedup_best'] for r in rows]),
                wins_over_3pct=sum(r['speedup_best']>1.03 for r in rows),
                ties_3pct=sum(1/1.03<=r['speedup_best']<=1.03 for r in rows),
                losses_over_3pct=sum(r['speedup_best']<1/1.03 for r in rows),
                gm_eager_production=gm([r['eager_speedup_production'] for r in rows])))
    (ROOT/'softplus_general_summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'softplus_general_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
