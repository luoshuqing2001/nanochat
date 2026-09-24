"""Summarize paired fixed-policy results, never choose the fastest per shape."""
import argparse,json,math
from pathlib import Path

def summarize(path):
    rows=json.loads(Path(path).read_text());out={}
    for phase in ('prefill','train'):
        subset=[r for r in rows if r['phase']==phase]
        if not subset:continue
        stats={}
        for name in subset[0]['results']:
            ratios=[r['results']['default']['ms']/r['results'][name]['ms'] for r in subset]
            fa4=[r['results']['softmax_tuned']['ms']/r['results'][name]['ms'] for r in subset]
            stats[name]=dict(speedup_vs_default=math.exp(sum(map(math.log,ratios))/len(ratios)),min=min(ratios),max=max(ratios),wins=sum(x>1 for x in ratios),speedup_vs_tuned_fa4=math.exp(sum(map(math.log,fa4))/len(fa4)))
        out[phase]=dict(cases=len(subset),variants=stats)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+');ap.add_argument('--output',required=True);a=ap.parse_args()
    out={p:summarize(p) for p in a.inputs};Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
