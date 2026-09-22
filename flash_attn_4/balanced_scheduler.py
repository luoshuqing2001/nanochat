# Work-balanced tile scheduler for causal attention.
#
# The problem it solves: under a causal mask query tile m attends to (m+1) key blocks,
# so a scheduler that launches one CTA per query tile hands out programs whose work
# ranges over N:1. FA4's split-KV divides each query tile's *own* range into a fixed
# number of parts, which scales every program down by the same factor and leaves that
# ratio exactly where it was.
#
# This does the opposite assignment. Every CTA is given the same number of key blocks,
# `chunk`, and a query tile is given as many CTAs as its key range needs:
#
#     CTAs for query tile m = ceil(n_blocks(m) / chunk)
#
# so every CTA does identical work, there are no empty CTAs, and the total work is
# exactly the causal triangle with nothing wasted. The CTAs that share a query tile
# aggregate into O with atomic_add, which softplus permits because it never normalizes
# (see flash_fwd_softplus.py); softmax could not use this schedule without a combine
# pass keyed on each part's LSE.
#
# The linear CTA index is mapped to (m_block, chunk_index) through a table built on the
# host, rather than a closed form, so that sliding-window layers -- whose n_blocks(m)
# flattens out once m passes the window -- work the same way. The table has one entry
# per CTA for a single (batch, head), a few dozen entries at training shapes, and is
# shared across the whole grid.

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from dataclasses import dataclass

from quack.cute_dsl_utils import ParamsBase
from flash_attn_4.tile_scheduler import WorkTileInfo, TileSchedulerArguments


def causal_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left):
    """Key blocks each query tile attends to -- the causal triangle, capped by the window."""
    counts = []
    num_m = max(1, (seqlen_q + tile_m - 1) // tile_m)
    for m in range(num_m):
        n_block_max = (seqlen_k + tile_n - 1) // tile_n
        if causal or window_left is not None:
            n_idx_right = (m + 1) * tile_m + seqlen_k - seqlen_q
            n_block_max = min(n_block_max, (n_idx_right + tile_n - 1) // tile_n)
        n_block_min = 0
        if window_left is not None:
            n_idx_left = m * tile_m + seqlen_k - seqlen_q - window_left
            n_block_min = max(n_idx_left // tile_n, 0)
        counts.append(max(0, n_block_max - n_block_min))
    return counts


@functools.lru_cache(maxsize=256)
def _build_work_table_cached(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left,
                             chunk, device):
    import torch

    rows = []
    for m, n_blocks in enumerate(
        causal_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left)
    ):
        for c in range((n_blocks + chunk - 1) // chunk):
            rows.append((m, c))
    if not rows:                      # degenerate shapes still need one CTA
        rows.append((0, 0))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def build_work_table(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left, chunk, device):
    """(m_block, chunk_index) for every CTA of one (batch, head), as an int32 (P, 2).

    Mirrors BlockInfo.get_n_block_min_max so the kernel's own range computation lands on
    the same blocks; `chunk` is handed to BlockInfo as num_n_blocks_per_split.

    Cached: the host-side loop and the upload are small but the kernel they feed can be
    30 microseconds, and during decode this is called once per layer per token. Training
    and prefill hit the cache after the first call; decode misses on every new cache
    length, which is why the table build is kept to a list comprehension and one upload.
    """
    return _build_work_table_cached(
        int(seqlen_q), int(seqlen_k), int(tile_m), int(tile_n), bool(causal),
        None if window_left is None else int(window_left), int(chunk), str(device),
    )


class BalancedCausalScheduler:
    """One CTA per `chunk` key blocks, looked up from the work table."""

    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        work_table: cute.Tensor

        @staticmethod
        def create(args: TileSchedulerArguments, *, loc=None, ip=None):
            assert args.work_table is not None, "BalancedCausalScheduler needs a work table"
            return BalancedCausalScheduler.Params(args.num_head, args.num_batch, args.work_table)

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, scheduling_mode=None,
                                loc=None, ip=None) -> "BalancedCausalScheduler.Params":
        return BalancedCausalScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, ctx=None, *, loc=None, ip=None) -> "BalancedCausalScheduler":
        return BalancedCausalScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        # x indexes the work table, so the grid is sized by total work, not by query tiles.
        return (cute.size(params.work_table.shape[0]), params.num_head, params.num_batch)

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        work_idx, head_idx, batch_idx = self._blk_coord
        m_block = Int32(self.params.work_table[work_idx, 0])
        chunk_idx = Int32(self.params.work_table[work_idx, 1])
        # chunk_idx rides in the split slot; BlockInfo turns it into the key range using
        # num_n_blocks_per_split, which is the constant chunk size.
        return WorkTileInfo((m_block, head_idx, batch_idx, chunk_idx), cutlass.Boolean(True))

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return WorkTileInfo((Int32(0), Int32(0), Int32(0), Int32(0)), cutlass.Boolean(False))


def causal_m_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left,
                          window_right=None):
    """Query tiles each KV block is attended by -- the causal triangle, mirrored.

    Mirrors BlockInfo.get_m_block_min_max, which is what the backward's own range
    computation uses.
    """
    counts = []
    num_n = max(1, (seqlen_k + tile_n - 1) // tile_n)
    for n in range(num_n):
        m_block_max = (seqlen_q + tile_m - 1) // tile_m
        m_block_min = 0
        if causal or (window_left is not None and window_right is not None):
            m_idx = n * tile_n + seqlen_q - seqlen_k
            m_idx_right = m_idx if causal else m_idx - window_right
            m_block_min = max(m_block_min, m_idx_right // tile_m)
        if window_left is not None:
            m_idx_left = (n + 1) * tile_n + seqlen_q - seqlen_k + window_left
            m_block_max = min(m_block_max, (m_idx_left + tile_m - 1) // tile_m)
        counts.append(max(0, m_block_max - m_block_min))
    return counts


@functools.lru_cache(maxsize=256)
def _build_bwd_work_table_cached(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left,
                                 window_right, chunk, device):
    import torch

    rows = []
    for n, m_blocks in enumerate(
        causal_m_block_counts(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left,
                              window_right)
    ):
        for c in range((m_blocks + chunk - 1) // chunk):
            rows.append((n, c))
    if not rows:
        rows.append((0, 0))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def build_bwd_work_table(seqlen_q, seqlen_k, tile_m, tile_n, causal, window_left,
                         window_right, chunk, device):
    """(n_block, chunk_index) per CTA for the backward, one (batch, head)'s worth.

    The backward parallelizes over KV blocks rather than query tiles, so the causal
    triangle is mirrored: KV block n is attended by the query tiles at or after it, and
    one CTA per KV block means the first does N query tiles and the last does one. Same
    fix as the forward, transposed -- a constant `chunk` query tiles per CTA, and as many
    CTAs per KV block as it needs. dK and dV then have to be aggregated with atomic_add,
    which is the path FA4 already keeps for GQA.
    """
    return _build_bwd_work_table_cached(
        int(seqlen_q), int(seqlen_k), int(tile_m), int(tile_n), bool(causal),
        None if window_left is None else int(window_left),
        None if window_right is None else int(window_right),
        int(chunk), str(device),
    )
