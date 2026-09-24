# Softplus attention backward, built on FA4's SM80/SM120 backward kernel.
#
# The softmax backward needs two row statistics: the LSE, to recompute P from the
# scores, and D = rowsum(dO * O), to form dS = P * (dP - D). Softplus needs neither.
# With O_i = n_i^-alpha * sum_j softplus(s_ij) v_j,
#
#     P_ij  = softplus(s_ij)                        (recomputed directly, no LSE)
#     dS_ij = n_i^-alpha * dP_ij * sigmoid(s_ij)    (element-wise, no row sum)
#
# so the whole `flash_bwd_preprocess` pass that exists to produce D can be skipped,
# and the LSE never has to be stored by the forward or read back here.
#
# The `n_i^-alpha` factor has to reach two places: dV sums over query rows, so it must
# be folded into P before the dV gemm; and dS carries it for dQ/dK. Rather than
# recomputing the row index in the kernel, it rides in through the LSE and dPsum
# buffers -- both are already per-query-row float32 tensors with a gmem -> smem -> rmem
# path, and softplus has no use for either. The caller fills them with n^-alpha.

import math

import cutlass
import cutlass.cute as cute
from quack import layout_utils

from flash_attn_4.balanced_scheduler import BalancedCausalScheduler
from flash_attn_4.flash_bwd import FlashAttentionBackwardSm80
from flash_attn_4.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn_4.softplus import softplus_and_sigmoid_


class SoftplusBackwardMixin:
    """Swaps the two score-map-dependent steps of FA4's backward for softplus ones."""

    def __init__(self, *args, balanced_m_chunk=None, early_dv=False, native=False, share_pd=False, poly_estrin=False, row_scale_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus_estrin = bool(poly_estrin)
        self.softplus_row_params = row_scale_params
        self.softplus_scaled_do = bool(native)
        self.softplus_native = bool(native or row_scale_params is not None)
        self.softplus_share_pd = bool(share_pd or self.softplus_native)
        self.softplus_early_dv = bool(early_dv or self.softplus_share_pd)
        if self.softplus_share_pd:
            assert not self.share_QV_smem, "P/dS aliasing requires separate Q/V storage"
        # Balanced scheduling, mirrored from the forward: a constant number of query
        # tiles per CTA and as many CTAs per KV block as it needs. dK/dV then have to be
        # aggregated with atomic_add, which turns on FA4's GQA accumulator path.
        self.balanced_m_chunk = None if balanced_m_chunk is None else int(balanced_m_chunk)
        if self.balanced_m_chunk is not None:
            self.m_blocks_per_chunk = self.balanced_m_chunk
            self.tile_scheduler_cls = BalancedCausalScheduler
            self.dkv_atomic = True

    @cute.jit
    def fill_softplus_scale(self, target, m_block, thr_mma):
        if cutlass.const_expr(self.softplus_row_params is None):
            target.fill(1.0)
        else:
            tq,tk,window,alpha,causal = self.softplus_row_params
            coords=thr_mma.partition_C(cute.make_identity_tensor((self.m_block_size,self.n_block_size)))
            coords_mn=layout_utils.reshape_acc_to_mn(coords)
            for r in cutlass.range_constexpr(cute.size(target)):
                row=m_block*self.m_block_size+coords_mn[r,0][0]
                count=cutlass.min(row+tk-tq+1,tk) if cutlass.const_expr(causal) else tk
                if cutlass.const_expr(window>=0):count=cutlass.min(count,window+1)
                n=cutlass.Float32(cutlass.max(count,1))
                if cutlass.const_expr(alpha==1.):value=cute.arch.rcp_approx(n)
                elif cutlass.const_expr(alpha==0.):value=cutlass.Float32(1.)
                else:value=cute.math.exp2(cute.math.log2(n,fastmath=True)*(-alpha),fastmath=True)
                target[r]=value

    @cute.jit
    def bwd_recompute_p(
        self, acc_S_mn, acc_S_pre_mn, tLSErLSE, softmax_scale, softmax_scale_log2
    ) -> None:
        # tLSErLSE carries n^-alpha, not the LSE. See the module docstring.
        zero_frag = cute.make_rmem_tensor(cute.size(acc_S_mn, mode=[1]), cutlass.Float32)
        zero_frag.fill(0.0)
        zero = zero_frag.load()
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            sp, sig = softplus_and_sigmoid_(acc_S_mn[r, None].load() * softmax_scale, zero, self.softplus_estrin)
            acc_S_pre_mn[r, None].store(sig)
            # dV sums softplus(s_ij) * dO_i over query rows, so n^-alpha belongs on P.
            acc_S_mn[r, None].store(sp if cutlass.const_expr(self.softplus_scaled_do) else sp * tLSErLSE[r])

    @cute.jit
    def bwd_grad_val(self, acc_S_mn, acc_dP_mn, acc_S_pre_mn, tLSErdPsum, r):
        # acc_S_pre holds sigmoid(s) from bwd_recompute_p; tLSErdPsum holds n^-alpha.
        if cutlass.const_expr(self.softplus_scaled_do):
            return acc_dP_mn[r, None].load() * acc_S_pre_mn[r, None].load()
        return (
            acc_dP_mn[r, None].load() * acc_S_pre_mn[r, None].load() * tLSErdPsum[r]
        )


class FlashAttentionBackwardSm80Softplus(SoftplusBackwardMixin, FlashAttentionBackwardSm80):
    pass


class FlashAttentionBackwardSm120Softplus(SoftplusBackwardMixin, FlashAttentionBackwardSm120):
    pass
