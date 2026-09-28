"""
dqfuse (*_dqf builds: dQ memset + fp32->bf16 postprocess folded into the backward kernel, via atomic
arrival counters per dQ row block) vs the same builds without it, and vs stock FA3 softmax.
Same table as bench_bwd.py; the last column is the best fwd+bwd against stock.

    python hopper_softplus/bench_bwd_dqf.py > hopper_softplus/bench_bwd_h100_dqf.md
"""
import bench_bwd
from bench_bwd import LN2

bench_bwd.VARIANTS = {
    "stock": ("bwd_stock", None, False), "stock_dqf": ("bwd_stock_dqf", None, False),
    "fn_rexp_pre": ("fn_rexp_pre", LN2, True), "fn_rexp_pre_dqf": ("fn_rexp_pre_dqf", LN2, True),
}

if __name__ == "__main__":
    bench_bwd.main()
