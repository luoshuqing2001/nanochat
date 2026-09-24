"""Longer timing windows for the owner-pair experiment, including short kernels."""
import gc,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import trains.bench_softplus_owner_pair as bench


def timed(fn):
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    graph.replay()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();graph.replay();end.record();end.synchronize()
    estimate=max(begin.elapsed_time(end),.001)
    for _ in range(max(1,min(2000,math.ceil(20/estimate)))):graph.replay()
    n=max(1,min(10000,math.ceil(60/estimate)))
    begin.record()
    for _ in range(n):graph.replay()
    end.record();end.synchronize()
    ms=begin.elapsed_time(end)/n
    del out,graph
    return ms

if __name__=='__main__':
    bench.timed=timed
    bench.main()
    p=Path(sys.argv[sys.argv.index('--output')+1]+'.metadata.json')
    metadata=json.loads(p.read_text())
    metadata['timing']='3 alternating trials, one live graph; >=20ms warmup and target60ms/event sample; up to10000 replays'
    metadata['hashes'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.write_text(json.dumps(metadata,indent=2))
