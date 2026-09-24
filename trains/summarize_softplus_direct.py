"""Compare separate-process approximation runs on matched shapes."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('candidates',nargs='+');a=p.parse_args()
key=lambda r:tuple(r[k] for k in ('b','h','t','d','w'))
base={key(r):r for r in json.loads(Path(a.baseline).read_text())['rows']}
summary={}
for path in a.candidates:
 data=json.loads(Path(path).read_text());rows=data['rows'];entry={}
 for phase in ('prefill','train'):
  ratios=[base[key(r)]['results'][phase]['ms']/r['results'][phase]['ms'] for r in rows]
  entry[phase]=dict(cases=len(ratios),geomean=math.exp(sum(map(math.log,ratios))/len(ratios)),min=min(ratios),max=max(ratios),
     d64_geomean=math.exp(sum(math.log(base[key(r)]['results'][phase]['ms']/r['results'][phase]['ms']) for r in rows if r['d']==64)/sum(r['d']==64 for r in rows)))
 summary[path]=entry
Path(a.output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
