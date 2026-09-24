"""Query resource-limited residency for retained diagnostic backward binaries."""
import json
from pathlib import Path
import torch
import cuda.bindings.driver as cu
torch.cuda.init()
torch.cuda.set_device(0)
_context_anchor = torch.empty(1, device='cuda')
root=Path(__file__).resolve().parent
source=json.loads((root/'softplus_bottleneck_baseline.json').read_text())
rows=[]
for r in source['resources']:
    tag=r['tag']
    if not tag.startswith('backward_'):continue
    d=int(tag.rsplit('_',1)[1]);share='sp_shared' in tag
    # BM=BN=64, BF16, Q stages=2, dO stages=1. Exact extents and alignments
    # of _get_shared_storage_cls: K,V,Q,dO; two FP32 row arrays; P and optional dS.
    shared=2*(64*d+64*d+2*64*d+64*d)+2*4*64*2+64*64*2*(1 if share else 2)
    blob=(Path('/tmp/softplus_bottleneck_cubins')/('baseline_'+tag+'.cubin')).read_bytes()
    err,module=cu.cuModuleLoadData(blob)
    if int(err):raise RuntimeError(err)
    try:
        err,kernel=cu.cuModuleGetFunction(module,r['symbol'].encode())
        if int(err):raise RuntimeError(err)
        err,=cu.cuFuncSetAttribute(kernel,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared)
        if int(err):raise RuntimeError(err)
        err,blocks=cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel,128,shared)
        if int(err):raise RuntimeError(err)
        rows.append(dict(tag=tag,threads=128,registers=r['registers'],local_bytes=r['local_bytes'],dynamic_shared_bytes=shared,resource_limited_ctas_per_sm=blocks,static_local_instructions={k:v for k,v in r['static_sass_counts'].items() if k.startswith(('LDL','STL'))}))
    finally:cu.cuModuleUnload(module)
(root/'softplus_bottleneck_occupancy.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
