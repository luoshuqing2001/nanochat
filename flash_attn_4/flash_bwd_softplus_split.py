"""Experimental separate dV and dQ/dK kernels, using the same KV-owner scheduler."""
import cutlass
import cutlass.cute as cute
from quack import layout_utils
from flash_attn_4 import ampere_helpers as sm80_utils
from flash_attn_4.flash_bwd_softplus import FlashAttentionBackwardSm120Softplus
from flash_attn_4.softplus import softplus_, softplus_and_sigmoid_


class SoftplusBackwardQK(FlashAttentionBackwardSm120Softplus):
    softplus_skip_dv = True

    @cute.jit
    def bwd_recompute_p(self,acc_S_mn,acc_S_pre_mn,tLSErLSE,softmax_scale,softmax_scale_log2):
        # Only the derivative is required; no Softplus/log polynomial and no P.
        zero=cute.make_rmem_tensor(cute.size(acc_S_mn,mode=[1]),cutlass.Float32)
        zero.fill(0.)
        for r in cutlass.range(cute.size(acc_S_mn,mode=[0]),unroll_full=True):
            _,sig=softplus_and_sigmoid_(acc_S_mn[r,None].load()*softmax_scale,zero.load(),self.softplus_estrin)
            acc_S_pre_mn[r,None].store(sig)


class SoftplusBackwardV(FlashAttentionBackwardSm120Softplus):
    @cute.jit
    def load_V(self,gmem_thr_copy,tVgV,tVsV,block,seqlen,headdim):
        # V itself is not needed for dV; its shared buffer is epilogue scratch.
        pass

    @cute.jit
    def compute_one_m_block(self,m_block,smem_pipe_read_q,smem_pipe_read_do,
                            smem_pipe_write_q,smem_pipe_write_do,mma_params,
                            smem_copy_params,gmem_copy_params,load_Q_LSE,
                            load_dO_dPsum,m_block_max,softmax_scale,
                            softmax_scale_log2,aux_data=None,mask_fn=None):
        assert self.num_stages_Q == self.num_stages_dO == 1
        assert not self.Mma_dKV_is_RS
        def load_Q_next():
            if m_block+1<m_block_max:load_Q_LSE(m_block+1,0)
            cute.arch.cp_async_commit_group()
        acc=cute.make_rmem_tensor(mma_params.thr_mma_sdp.partition_shape_C((self.m_block_size,self.n_block_size)),cutlass.Float32)
        acc.fill(0.)
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        sm80_utils.gemm(mma_params.thr_mma_sdp,acc,mma_params.tSrQ,mma_params.tSrK,
            smem_copy_params.tSsQ[None,None,None,0],smem_copy_params.tSsK,
            smem_copy_params.smem_thr_copy_QdO,smem_copy_params.smem_thr_copy_KV)
        if cutlass.const_expr(mask_fn is not None):mask_fn(acc,m_block=m_block)
        scale=cute.make_fragment_like(smem_copy_params.tSsLSEMma[None,0])
        cute.autovec_copy(smem_copy_params.tSsLSEMma[None,0],scale)
        mn=layout_utils.reshape_acc_to_mn(acc)
        z=cute.make_rmem_tensor(cute.size(mn,mode=[1]),cutlass.Float32);z.fill(0.)
        for r in cutlass.range(cute.size(mn,mode=[0]),unroll_full=True):
            mn[r,None].store(softplus_(mn[r,None].load()*softmax_scale,z.load(),self.softplus_estrin)*scale[r])
        rp=cute.make_fragment_like(acc,self.dtype);rp.store(acc.load().to(self.dtype))
        cute.copy(smem_copy_params.r2s_thr_copy_PdS,
            smem_copy_params.r2s_thr_copy_PdS.retile(rp),smem_copy_params.tPsP)
        cute.arch.barrier()
        sm80_utils.gemm(mma_params.thr_mma_dkv,mma_params.acc_dV,mma_params.tdVrP,mma_params.tdVrdO,
            smem_copy_params.tdVsPt,smem_copy_params.tdVsdOt[None,None,None,0],
            smem_copy_params.smem_thr_copy_PdSt,smem_copy_params.smem_thr_copy_QdOt,hook_fn=load_Q_next)
        cute.arch.barrier()
        if m_block+1<m_block_max:load_dO_dPsum(m_block+1,0)
        cute.arch.cp_async_commit_group()
