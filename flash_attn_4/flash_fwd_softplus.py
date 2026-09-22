# Softplus attention forward, built on FA4's SM80/SM120 forward kernel.
#
# The mainloop is FA4's, unchanged. Only two things differ from softmax attention:
#
#   1. the score map (flash_attn_4/softplus.py), swapped in through `score_map_cls`;
#   2. the `n_i^-alpha` row scale, applied here in the epilogue.
#
# (2) lives in the epilogue rather than in the score map because it depends on the
# query row index, not on the scores: row i attends to n_i keys, and n_i is known
# from `m_block` plus the MMA's C-partition -- the same way mask.py derives the row
# index for causal masking. Doing it here costs one multiply per output element,
# once, on values already in registers, instead of a separate pass over O.

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import layout_utils

from flash_attn_4 import utils

from flash_attn_4.balanced_scheduler import BalancedCausalScheduler
from flash_attn_4.block_info import BlockInfo
from flash_attn_4.flash_fwd import FlashAttentionForwardBase, FlashAttentionForwardSm80
from flash_attn_4.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn_4.softplus import Softplus


class SoftplusForwardMixin:
    """Turns an FA4 forward kernel into a softplus one.

    `window_size_left` is a compile-time argument here, unlike in the softmax kernel
    where it is a runtime one. The epilogue needs the attended-key count per row, and
    baking the window in lets `n_i` fold into the compiled code; nanochat's SWA window
    is fixed per layer type, so this costs nothing in practice. Pass the same value
    given to the kernel at launch, or None for full causal / non-causal.
    """

    score_map_cls = Softplus

    def __init__(
        self,
        *args,
        softplus_alpha: float = 1.0,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        num_splits: int = 1,
        balanced_chunk: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.softplus_alpha = float(softplus_alpha)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        if window_size_left is not None:
            assert self.is_local or self.is_causal, "window_size_left needs a causal/local kernel"
        # Tile splitting: each query tile's key range is cut into `num_splits` pieces,
        # one program each, aggregated with atomic_add. Softmax cannot do this without a
        # combine pass that rescales by the per-split LSE; softplus can, because the
        # partial sums are already the final quantity -- the n^-alpha row scale is linear,
        # so scaling each partial and adding is the same as adding then scaling.
        #
        # The win is on the critical path, not on total work: under a causal mask the
        # longest program goes from n_block_max tiles to ceil(n_block_max / num_splits),
        # which is what balances the load. The cost is num_splits x more programs, most of
        # them short, plus the atomic traffic. Measure before believing it helps -- on both
        # GB10 and B200 the Triton version of this lost at training shapes and won only
        # when the query-tile program count was under ~3x the SM count.
        self.num_splits = int(num_splits)
        self.is_split_kv = self.num_splits > 1
        # Balanced scheduling: a constant `balanced_chunk` key blocks per CTA, and as many
        # CTAs per query tile as its key range needs. Unlike num_splits this equalizes the
        # CTAs instead of scaling them all down -- the longest CTA is `balanced_chunk`
        # blocks whatever the query tile, and no CTA is empty, so no work is wasted.
        # Mutually exclusive with num_splits; both use the atomic epilogue.
        self.balanced_chunk = None if balanced_chunk is None else int(balanced_chunk)
        if self.balanced_chunk is not None:
            assert self.num_splits == 1, "balanced_chunk and num_splits are alternatives"
            self.is_split_kv = True
            self.n_blocks_per_split = self.balanced_chunk
            self.tile_scheduler_cls = BalancedCausalScheduler

    def _check_type(self, mQ_type, mK_type, mV_type, mO_type, *args):
        # The split path accumulates into an fp32 O, which upstream's check rejects
        # because it insists O match Q/K/V. Check it against Q instead and defer.
        if self.is_split_kv:
            assert mO_type in (Float32, mQ_type), "atomic aggregation needs fp32 or the input dtype"
            mO_type = mQ_type
        return FlashAttentionForwardBase._check_type(
            self, mQ_type, mK_type, mV_type, mO_type, *args
        )

    @cute.jit
    def apply_count_scale(
        self,
        acc_O: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        seqlen,
    ) -> None:
        """Scale each row of acc_O by n_i^-alpha, n_i = number of keys row i attends to."""
        alpha = self.softplus_alpha
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(tiled_mma.get_slice(tidx).partition_C(cO))
        # Causal offset, as in mask.py: with seqlen_q != seqlen_k the diagonal is shifted.
        causal_offset = seqlen.seqlen_k - seqlen.seqlen_q + 1
        for r in cutlass.range(cute.size(acc_O_mn, mode=[0]), unroll_full=True):
            row_idx = tOcO_mn[r, 0][0] + m_block * self.tile_m
            if const_expr(self.is_causal or self.is_local):
                n_keys = row_idx + causal_offset
                if const_expr(self.window_size_left is not None):
                    n_keys = cutlass.min(n_keys, Int32(self.window_size_left + 1))
                n_keys = cutlass.min(n_keys, seqlen.seqlen_k)
            else:
                n_keys = seqlen.seqlen_k
            # Rows past seqlen_q are never written; keep n_keys >= 1 so the scale is finite.
            n_keys = cutlass.max(n_keys, Int32(1))
            n_f32 = Float32(n_keys)
            if const_expr(alpha == 1.0):
                row_scale = cute.arch.rcp_approx(n_f32)
            else:
                row_scale = cute.math.exp2(
                    cute.math.log2(n_f32, fastmath=True) * (-alpha), fastmath=True
                )
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale)

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        self.apply_count_scale(acc_O, tiled_mma, tidx, m_block, seqlen)
        if const_expr(self.is_split_kv):
            # Splits accumulate into a shared fp32 O; no smem staging, no combine kernel.
            # The balanced scheduler emits no empty CTAs by construction, so it skips the
            # guard that uniform splitting needs.
            if const_expr(self.balanced_chunk is not None):
                self.epilogue_atomic(acc_O, mO, seqlen, tiled_mma, tidx, m_block, head_idx, batch_idx)
            elif not self.split_is_empty(seqlen, m_block):
                self.epilogue_atomic(acc_O, mO, seqlen, tiled_mma, tidx, m_block, head_idx, batch_idx)
            return None
        # Not super(): zero-arg super() inside a @cute.jit method resolves back to this
        # override and recurses forever. Name the base implementation directly.
        return FlashAttentionForwardBase.epilogue(
            self,
            acc_O, lse, mO, mLSE, sO, seqlen, gmem_tiled_copy_O, tma_atom_O,
            tiled_mma, tidx, m_block, head_idx, batch_idx,
        )

    @cute.jit
    def split_is_empty(self, seqlen, m_block: Int32) -> cutlass.Boolean:
        """True when this program's slice of the key range is empty.

        Query tile m has ceil(m_keys / tile_n) key blocks, so with `num_splits` programs
        per query tile the last few get nothing whenever the tile is near the top of the
        causal triangle. Upstream never produces an empty range, so `kernel()` has no
        guard for it and happily processes block 0 -- which would double-count into the
        atomic accumulator. There is no early-exit intrinsic in the DSL, so instead of
        re-indenting 270 lines of upstream `kernel()` behind a guard, the epilogue
        recomputes the range and declines to write.

        The mainloop work of an empty program is still wasted: S*(S-1)/2 tiles across a
        causal pass, against N*(N+1)/2 useful ones for N query tiles, so ~4% at N=16,
        S=4 and less as N grows.
        """
        _, head_y, _ = cute.arch.block_idx()
        split_idx = head_y % self.num_splits
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            True,  # is_split_kv
            self.window_size_left,
            self.window_size_right,
            num_splits=self.num_splits,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, self.num_splits
        )
        return n_block_max <= n_block_min

    @cute.jit
    def epilogue_atomic(
        self,
        acc_O: cute.Tensor,
        mO: cute.Tensor,
        seqlen,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ) -> None:
        """Accumulate this split's partial O into global memory with atomic_add.

        Goes straight from the MMA accumulator in registers to gmem, skipping the
        rmem -> smem -> gmem staging the normal epilogue uses for coalescing: with
        several splits landing on the same rows there is nothing to coalesce anyway.
        `mO` must be float32 and zeroed by the caller.
        """
        # mO is float32 normally; with bf16 atomics it is the output dtype itself, which
        # removes the fp32 workspace, its zeroing and the cast pass -- at the price of
        # accumulating in bf16 across the CTAs that share a query tile.
        # Only the MMA's own threads hold accumulator fragments. The kernel launches more
        # than that, and partition_C wraps around for the rest, so without this guard every
        # element is written once per extra thread group -- silently doubling the output.
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        thr_mma = tiled_mma.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        gO_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gO))
        c_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        row_limit = seqlen.seqlen_q - m_block * self.tile_m
        for r in cutlass.range(cute.size(acc_O_mn, mode=[0]), unroll_full=True):
            if c_mn[r, 0][0] < row_limit:
                for c in cutlass.range(cute.size(acc_O_mn, mode=[1]), unroll_full=True):
                    val = acc_O_mn[r, c]
                    if const_expr(mO.element_type is not Float32):
                        val = val.to(mO.element_type)
                    cute.arch.atomic_add(ptr=utils.elem_pointer(gO_mn, (r, c)), val=val)


class FlashAttentionForwardSm80Softplus(SoftplusForwardMixin, FlashAttentionForwardSm80):
    pass


class FlashAttentionForwardSm120Softplus(SoftplusForwardMixin, FlashAttentionForwardSm120):
    pass
