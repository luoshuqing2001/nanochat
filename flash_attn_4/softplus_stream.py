"""CuTe Stream-K work assignment; the existing MMA/prefetch loop is preserved."""
from dataclasses import dataclass
from functools import lru_cache
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from quack.cute_dsl_utils import ParamsBase
from nanochat.softplus_stream_k import plan_cpu
from flash_attn_4.tile_scheduler import WorkTileInfo


@lru_cache(maxsize=128)
def stream_plan_cpu(tq,tk,window,bm,bn,workers,tail=False):
    if not tail:
        return plan_cpu(tq,tk,window,bm,bn,workers)
    nm=(tq+bm-1)//bm
    first=max(0,nm-workers)
    if first==0:return plan_cpu(tq,tk,window,bm,bn,workers)
    _,whole,_,_=plan_cpu(tq,tk,window,bm,bn,1)
    offsets,segs,groups,slots=plan_cpu(tq-first*bm,tk,window,bm,bn,workers)
    segs=whole[:first]+tuple((m+first,lo,hi,slot) for m,lo,hi,slot in segs)
    groups=tuple((m+first,slot,n) for m,slot,n in groups)
    offsets=tuple(range(first))+tuple(first+x for x in offsets)
    return offsets,segs,groups,slots


@lru_cache(maxsize=128)
def tile_plan_cpu(tq,tk,window,bm,bn,workers):
    """One CTA per segment; only long tiles split, never join short query tiles."""
    _,whole,_,_=plan_cpu(tq,tk,window,bm,bn,1)
    budget=max(1,(sum(hi-lo for _,lo,hi,_ in whole)+workers-1)//workers)
    segs,groups=[],[]
    slots=0
    for m,lo,hi,_ in whole:
        n=(hi-lo+budget-1)//budget
        if n>1:groups.append((m,slots,n))
        for part in range(n):
            segs.append((m,lo+(hi-lo)*part//n,lo+(hi-lo)*(part+1)//n,slots if n>1 else -1))
            if n>1:slots+=1
    # Long tasks first reduces the final scheduling wave's critical path.
    segs.sort(key=lambda x:x[2]-x[1],reverse=True)
    return tuple(range(len(segs)+1)),tuple(segs),tuple(groups),slots


@lru_cache(maxsize=128)
def global_plan_cpu(tq,tk,window,bm,bn,parallelism,bh):
    """LPT simulation across all heads; split only a predicted critical-path tile.

    Costs are heuristic KV-iteration equivalents: 2 per CTA setup, 2 per split
    epilogue. At most four parts per tile, at least four KV iterations per part.
    Virtual lanes are a planning model, not physical SM affinity.
    """
    import heapq
    if parallelism<=0 or bh<=0:raise ValueError('positive parallelism and batch*heads required')
    _,whole,_,_=plan_cpu(tq,tk,window,bm,bn,1)
    parts=[1]*len(whole)
    def jobs(counts):
        out=[]
        for (m,lo,hi,_),n in zip(whole,counts):
            for j in range(n):
                out.append((m,lo+(hi-lo)*j//n,lo+(hi-lo)*(j+1)//n))
        return sorted(out,key=lambda x:x[2]-x[1]+(2 if counts[x[0]]>1 else 0),reverse=True)
    def simulate(counts):
        lanes=[(0,i) for i in range(min(parallelism,len(jobs(counts))*bh))]
        heapq.heapify(lanes)
        finish,critical=0,0
        for m,lo,hi in jobs(counts):
            cost=hi-lo+2+(2 if counts[m]>1 else 0)
            for _ in range(bh):
                time,lane=heapq.heappop(lanes);time+=cost
                if time>finish:finish,critical=time,m
                heapq.heappush(lanes,(time,lane))
        return finish,critical
    # Try splitting a band of equally long tiles together: splitting just one of
    # several tied tail tasks cannot change the predicted makespan.
    if window<0:
        for _ in range(4):
            before,_=simulate(parts)
            best,best_parts=before,parts
            lengths=sorted({hi-lo for _,lo,hi,_ in whole},reverse=True)
            # Bound cold host planning time; it is cached and outside captured work.
            step=max(1,len(lengths)//16)
            for threshold in lengths[::step]:
                candidate=parts.copy()
                for m,lo,hi,_ in whole:
                    if hi-lo>=threshold and parts[m]<4 and hi-lo>=4*(parts[m]+1):
                        candidate[m]+=1
                if candidate==parts:continue
                after,_=simulate(candidate)
                if after<best:best,best_parts=after,candidate
            if best>=before*.99:break
            parts=best_parts
    segs,groups,slots=[],[],0
    for (m,lo,hi,_),n in zip(whole,parts):
        if n>1:groups.append((m,slots,n))
        for j in range(n):
            segs.append((m,lo+(hi-lo)*j//n,lo+(hi-lo)*(j+1)//n,slots if n>1 else -1))
            if n>1:slots+=1
    segs.sort(key=lambda x:x[2]-x[1]+(2 if x[3]>=0 else 0),reverse=True)
    return tuple(range(len(segs)+1)),tuple(segs),tuple(groups),slots


@lru_cache(maxsize=128)
def capped_plan_cpu(tq,tk,window,bm,bn,kv_split_size,kv_major=False):
    """One CTA per bounded KV interval; whole query tiles bypass reduction."""
    if not isinstance(kv_split_size,int) or kv_split_size <= 0 or kv_split_size % bn:
        raise ValueError("kv_split_size must be a positive multiple of the KV tile size")
    budget=kv_split_size//bn
    _,whole,_,_=plan_cpu(tq,tk,window,bm,bn,1)
    segs,groups=[],[]
    slots=0
    for m,lo,hi,_ in whole:
        n=(hi-lo+budget-1)//budget
        if n>1:groups.append((m,slots,n))
        for begin in range(lo,hi,budget):
            segs.append((m,begin,min(begin+budget,hi),slots if n>1 else -1))
            if n>1:slots+=1
    # Interleave output owners while reusing a KV interval across query tiles.
    if kv_major:segs.sort(key=lambda x:(x[1],-x[0]))
    return tuple(range(len(segs)+1)),tuple(segs),tuple(groups),slots


@lru_cache(maxsize=128)
def build_stream_plan(tq,tk,window,bm,bn,workers,tail,device,atomic=False,tiles=False,global_order=False,bh=1,kv_split_size=0,kv_major=False):
    if kv_split_size:
        offsets,segs,groups,slots=capped_plan_cpu(tq,tk,window,bm,bn,kv_split_size,kv_major)
    elif global_order:
        offsets,segs,groups,slots=global_plan_cpu(tq,tk,window,bm,bn,workers,bh)
    elif tiles:
        offsets,segs,groups,slots=tile_plan_cpu(tq,tk,window,bm,bn,workers)
    else:
        offsets,segs,groups,slots=stream_plan_cpu(tq,tk,window,bm,bn,workers,tail)
    if atomic:
        # Only boundary query tiles need a zeroed accumulator, one slot per tile.
        group_slot = {m:i for i,(m,_,_) in enumerate(groups)}
        segs = tuple((m,lo,hi,group_slot[m] if slot>=0 else -1) for m,lo,hi,slot in segs)
        groups = tuple((m,i,1) for i,(m,_,_) in enumerate(groups))
        slots = len(groups)
    nw=len(offsets)-1
    rows=[(a+nw,b+nw,0,0) for a,b in zip(offsets,offsets[1:])]+list(segs)
    return (torch.tensor(rows,device=device,dtype=torch.int32),
            torch.tensor(groups,device=device,dtype=torch.int32),slots,nw)


class StreamCausalScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head:Int32
        num_batch:Int32
        workers:Int32
        work_table:cute.Tensor

    @staticmethod
    def to_underlying_arguments(args,**kwargs):
        return StreamCausalScheduler.Params(args.num_head,args.num_batch,args.num_splits,args.work_table)

    @staticmethod
    def get_grid_shape(params,**kwargs):
        return (params.workers,params.num_head,params.num_batch)

    @staticmethod
    def create(params,**kwargs):
        return StreamCausalScheduler()

    def initial_work_tile_info(self,**kwargs):
        return WorkTileInfo((Int32(0),Int32(0),Int32(0),Int32(0)),cutlass.Boolean(True))


class GlobalStreamCausalScheduler(StreamCausalScheduler):
    @staticmethod
    def get_grid_shape(params,**kwargs):
        return (params.workers*params.num_head*params.num_batch,1,1)


@lru_cache(maxsize=128)
def completion_metadata(tq,tk,window,bm,bn,workers,tail,tiles,global_order,bh,device,kv_split_size=0,kv_major=False):
    if kv_split_size:plan=capped_plan_cpu(tq,tk,window,bm,bn,kv_split_size,kv_major)
    elif global_order:plan=global_plan_cpu(tq,tk,window,bm,bn,workers,bh)
    elif tiles:plan=tile_plan_cpu(tq,tk,window,bm,bn,workers)
    else:plan=stream_plan_cpu(tq,tk,window,bm,bn,workers,tail)
    groups=plan[2]
    rows=[(-1,0,0)]*((tq+bm-1)//bm)
    for group,(m,first,count) in enumerate(groups):rows[m]=(group,first,count)
    return torch.tensor(rows,device=device,dtype=torch.int32),len(groups)
