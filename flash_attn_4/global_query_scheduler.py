"""Whole-query global LPT control, usable by both Softmax and Softplus on SM120."""
from dataclasses import dataclass
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from quack.cute_dsl_utils import ParamsBase
from flash_attn_4.tile_scheduler import WorkTileInfo


class GlobalQueryScheduler:
    @dataclass
    class Params(ParamsBase):
        blocks: Int32
        heads: Int32
        batches: Int32

    @staticmethod
    def to_underlying_arguments(args,**kwargs):
        return GlobalQueryScheduler.Params(args.num_block,args.num_head,args.num_batch)

    @staticmethod
    def get_grid_shape(params,**kwargs):
        return (params.blocks*params.heads*params.batches,1,1)

    def __init__(self,params):self.params=params

    @staticmethod
    def create(params,**kwargs):return GlobalQueryScheduler(params)

    def initial_work_tile_info(self,**kwargs):
        index,_,_=cute.arch.block_idx()
        bh=index % (self.params.heads*self.params.batches)
        m=self.params.blocks-1-index // (self.params.heads*self.params.batches)
        b=bh // self.params.heads
        h=bh % self.params.heads
        return WorkTileInfo((m,h,b,Int32(0)),cutlass.Boolean(True))


class PairedQueryScheduler(GlobalQueryScheduler):
    """Head-local grid; the kernel sequentially owns complementary query tiles."""
    @staticmethod
    def get_grid_shape(params,**kwargs):
        return ((params.blocks+1)//2,params.heads,params.batches)


class HeadLocalQueryScheduler(GlobalQueryScheduler):
    """Reverse query order within each head without a global work table."""
    @staticmethod
    def create(params,**kwargs):
        return HeadLocalQueryScheduler(params)
    @staticmethod
    def get_grid_shape(params,**kwargs):
        return (params.blocks,params.heads,params.batches)

    def initial_work_tile_info(self,**kwargs):
        m,h,b=cute.arch.block_idx()
        return WorkTileInfo((self.params.blocks-1-m,h,b,Int32(0)),cutlass.Boolean(True))
