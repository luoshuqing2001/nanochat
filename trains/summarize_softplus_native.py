"""Summarize graph and eager timings against each case's fastest Softmax baseline."""
import argparse
import json
import math
from pathlib import Path


def geometric(xs):
    return math.exp(sum(map(math.log,xs))/len(xs)) if xs else None


def summarize(path):
    data=json.loads(Path(path).read_text())
    result={}
    for phase in ('prefill','decode','train'):
        rows=[r for r in data['rows'] if r['phase']==phase and 'results' in r]
        result[phase]={}
        for metric in ('ms','eager_ms'):
            ratios=[];previous=[];selected=[]
            for r in rows:
                timings=r['results']
                if metric not in timings['softplus']:continue
                soft=timings['softplus'][metric]
                base=min(v[metric] for k,v in timings.items() if k.startswith('softmax_'))
                ratios.append(base/soft)
                if 'softplus_previous' in timings:
                    value=timings['softplus_previous'][metric]/soft
                    previous.append(value)
                    if r['d']==64 and r['dtype']=='bf16' and r['tq']==r['tk'] and r['tq']>=2048:
                        selected.append(value)
            result[phase][metric]=dict(cases=len(ratios),vs_fastest_softmax=geometric(ratios),
                wins=sum(x>1 for x in ratios),vs_previous=geometric(previous),
                selected_default_cases=len(selected),selected_default_vs_previous=geometric(selected),
                selected_range=[min(selected),max(selected)] if selected else None)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True)
    args=p.parse_args();result={x:summarize(x) for x in args.inputs}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
