/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, bool Clear_dQaccum, bool Varlen>
class FlashAttnBwdPreprocess {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(std::is_same_v<Element, cutlass::half_t> && ArchTag::kMinComputeCapability >= 75 ||
                  std::is_same_v<Element, cutlass::bfloat16_t> && ArchTag::kMinComputeCapability >= 80 ||
                  std::is_same_v<Element, cutlass::float_e4m3_t> && ArchTag::kMinComputeCapability >= 89);

    static constexpr uint32_t MaxThreadsPerBlock = 256;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int SharedStorageSize = 0;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    // We want kBlockKGmem to be a power of 2 so that when we do the summing,
    // it's just between threads in the same warp
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert((kBlockM * kHeadDim / kGmemElemsPerLoadAccum) % MaxThreadsPerBlock == 0, "MaxThreadsPerBlock must divide kBlockM * kHeadDim / kGmemElemsPerLoadAccum");
    using GmemLayoutAtomAccum = Layout<Shape<Int<MaxThreadsPerBlock>>>;
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomAccum{},
                        Layout<Shape<Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapedPsum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q, head, batch)
    using StridedPsum = cute::Stride<_1, int64_t, int64_t>;
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    // Device side arguments
    struct Arguments {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float *ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        int num_batch;  // We need this to know the size of dq_semaphore in case of varlen
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
#ifdef FLASHATTN_SOFTPLUS
        Element* ptr_dU = nullptr;
        StrideO const stride_dU = {};
#endif
#ifdef FLASHATTN_SOFTPLUS_DOUT_GAIN
        float const* ptr_gain = nullptr;  // (h * dv,): dO = bf16(dout * gain), as the caller's autograd would
#endif
#ifdef FLASHATTN_SOFTPLUS_DGAIN_REDUCE
        float* ptr_dgain_part = nullptr;  // (b * num_m_blocks, h * dv): per-CTA sums of dout * O (dgain)
#endif
    };

    // Kernel entry point API
    struct Params {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float* ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        int num_batch;
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
#ifdef FLASHATTN_SOFTPLUS
        Element* ptr_dU = nullptr;
        StrideO const stride_dU = {};
#endif
#ifdef FLASHATTN_SOFTPLUS_DOUT_GAIN
        float const* ptr_gain = nullptr;  // (h * dv,): dO = bf16(dout * gain), as the caller's autograd would
#endif
#ifdef FLASHATTN_SOFTPLUS_DGAIN_REDUCE
        float* ptr_dgain_part = nullptr;  // (b * num_m_blocks, h * dv): per-CTA sums of dout * O (dgain)
#endif
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        return {
            args.ptr_O,
            args.shape_O,
            args.stride_O,
            args.ptr_dO,
            args.stride_dO,
            args.ptr_dPsum,
            args.shape_dPsum,
            args.stride_dPsum,
            args.ptr_LSE,
            args.stride_LSE,
            args.ptr_LSE_log2,
            args.stride_LSE_log2,
            args.ptr_dQaccum,
            args.shape_dQaccum,
            args.stride_dQaccum,
            args.num_batch,
            args.dq_semaphore,
            args.cu_seqlens,
            args.seqused
#ifdef FLASHATTN_SOFTPLUS
            , args.ptr_dU, args.stride_dU
#endif
#ifdef FLASHATTN_SOFTPLUS_DOUT_GAIN
            , args.ptr_gain
#endif
#ifdef FLASHATTN_SOFTPLUS_DGAIN_REDUCE
            , args.ptr_dgain_part
#endif
        };
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, [[maybe_unused]] char* smem_buf) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});

        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;

        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info(bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused);
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const seqlen_o = seqlen_info.seqlen;
        if (is_varlen && m_block * kBlockM >= seqlen_o) { return; }

#ifndef FLASHATTN_SOFTPLUS_DOUT_IS_DU
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gO = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)
        Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdO = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)

        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(seqlen_info.offset), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));
        static_assert(kBlockM <= MaxThreadsPerBlock);
        float lse = thread_idx < seqlen_o - m_block * kBlockM && thread_idx < kBlockM ? gLSE(thread_idx) : INFINITY;

        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
        Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }

        // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
        Tensor tOrO = make_fragment_like(tOgO);
        Tensor tOrdO = make_fragment_like(tOgdO);
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgO, tOrO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgdO, tOrdO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
        // if (threadIdx.x == 222) { printf("bidx = %d, bidy = %d, bidz = %d, seqlen_o = %d, m_block = %d, seqlen_o - m_block * kBlockM = %d, tOgO addr = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, seqlen_o, m_block, seqlen_o - m_block * kBlockM, &tOgO(0));}

        // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
        Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
        Tensor tOrO_l = make_tensor(tOrO.data(), l);
        Tensor o_fp32 = make_tensor_like<float>(tOrO_l);
        flash::convert_type_out(tOrO_l, o_fp32);
        Tensor tOrdO_l = make_tensor(tOrdO.data(), l);
        Tensor do_fp32 = make_tensor_like<float>(tOrdO_l);
        flash::convert_type_out(tOrdO_l, do_fp32);
#ifdef FLASHATTN_SOFTPLUS_DOUT_GAIN
        // The caller passes dZ, the gradient of Z = O * gain (gain folded into the next projection), not
        // dO: rebuild dO exactly as its autograd would have materialized it -- torch.compile's generated
        // kernel computes fp32(dZ) * gain (gain in fp32: it drops the bf16 cast of the parameter) and
        // stores it as bf16, round to nearest even -- so every bit downstream is unchanged, and dO never
        // round-trips through memory.
        {
            float const* gain_h = params.ptr_gain + bidh * get<1>(params.shape_O);
            static constexpr int kV = CUTE_STATIC_V(size<0>(tOrO));  // (mi, ni) is tOrO(ni % kV, mi, ni / kV)
#ifdef FLASHATTN_SOFTPLUS_DGAIN_REDUCE
            // dgain = sum over tokens of dZ * O, fused here (this kernel reads both anyway): the caller
            // skips its own reduction pass over dZ and O. Per thread over its rows, then the threads that
            // share columns: the two rows of a warp (lanes l, l + 16) by shuffle, the warps through smem.
            // One partial per CTA -- deterministic, no atomics; the caller sums b * num_m_blocks rows.
            static_assert(kGmemThreadsPerRow == 16 && kV == 8 && decltype(size<1>(do_fp32))::value == 8,
                          "dgain reduction: head dim 128, bf16 only");
            float dg[8];
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                dg[ni] = 0.f;
                #pragma unroll
                for (int mi = 0; mi < size<0>(do_fp32); ++mi) { dg[ni] = fmaf(do_fp32(mi, ni), o_fp32(mi, ni), dg[ni]); }
                dg[ni] += __shfl_xor_sync(0xffffffff, dg[ni], 16);
            }
            __shared__ float dg_smem[MaxThreadsPerBlock / 32][kBlockKGmem];
            int const warp = thread_idx / 32, lane = thread_idx % 32;
            if (lane < 16) {
                #pragma unroll
                for (int ni = 0; ni < 8; ++ni) { dg_smem[warp][lane * 8 + ni] = dg[ni]; }
            }
            __syncthreads();
            if (thread_idx < kBlockKGmem) {
                float sum = 0.f;
                #pragma unroll
                for (int w = 0; w < int(MaxThreadsPerBlock / 32); ++w) { sum += dg_smem[w][thread_idx]; }
                int const hdim = get<1>(params.shape_O);
                int const nheads = get<2>(params.shape_O);
                params.ptr_dgain_part[(int64_t(bidb) * gridDim.x + m_block) * nheads * hdim + bidh * hdim + thread_idx] = sum;
            }
#endif
            #pragma unroll
            for (int mi = 0; mi < size<0>(do_fp32); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(do_fp32); ++ni) {
                    int const col = get<1>(tOcO(ni % kV, mi, ni / kV));
                    float const g = col < get<1>(params.shape_O) ? gain_h[col] : 0.f;
                    do_fp32(mi, ni) = __bfloat162float(__float2bfloat16_rn(__fmul_rn(do_fp32(mi, ni), g)));
                }
            }
        }
#endif
        // Sum across the last dimension
        Tensor dP_sum = make_tensor<float>(make_shape(size<0>(o_fp32)));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
            float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
            #pragma unroll
            for (int ni = 1; ni < size<1>(o_fp32); ni++) {
                dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
            }
            flash::SumOp<float> sum_op;
            dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
        }

#ifdef FLASHATTN_SOFTPLUS
        // RMSNorm backward, fused here since dP_sum = sum_d dO*O is exactly the row statistic it needs:
        //   dU = (dO - O * dP_sum / D) / rms
        // The main kernel's "P" is sp2 = softplus/ln2 (see softplus.h); it multiplies dV by ln2 once at
        // the end rather than carrying the factor through every score.
        {
            Tensor mdU = make_tensor(make_gmem_ptr(params.ptr_dU), params.shape_O, params.stride_dU)(_, _, bidh, bidb);
            Tensor gdU = local_tile(mdU, TileShape_MK{}, make_coord(m_block, _0{}));
            Tensor tOgdU = gmem_thr_copy_O.partition_D(gdU);
            Tensor tOrdU = make_fragment_like(tOgdU);
            Tensor tOrdU_l = make_tensor(tOrdU.data(), l);
            float const inv_d = 1.f / float(get<1>(params.shape_O));
            #pragma unroll
            for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
                int const row = get<0>(tOcO(_0{}, mi, _0{}));
                bool const in = row < seqlen_o - m_block * kBlockM;
                float const rms = in ? gLSE(row) : 1.f;
                float const inv_rms = 1.f / rms;
                float const c = dP_sum(mi) * inv_d;
                #pragma unroll
                for (int ni = 0; ni < size<1>(o_fp32); ++ni) {
                    tOrdU_l(mi, ni) = static_cast<Element>((do_fp32(mi, ni) - o_fp32(mi, ni) * c) * inv_rms);
                }
            }
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrdU, tOgdU, tOcO, tOpO, seqlen_o - m_block * kBlockM
            );
        }
#endif
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdPsum = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(dP_sum); ++mi) {
                int const row = get<0>(tOcO(_0{}, mi, _0{}));
                gdPsum(row) = row < seqlen_o - m_block * kBlockM ? dP_sum(mi) : 0;
            }
        }

        int const seqlen_rounded = cute::round_up(seqlen_o, kBlockM);
        Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (thread_idx < seqlen_rounded - m_block * kBlockM && thread_idx < kBlockM) {
            gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
        }
#endif  // FLASHATTN_SOFTPLUS_DOUT_IS_DU: the caller already produced dU (fused into the c_proj
        // backward GEMM, nanochat/fused_proj_rmsnorm.py), and softplus-type attention reads neither
        // LSE nor dP_sum, so this kernel neither loads O/dO nor writes anything but the dQ zeros.

        if constexpr (Clear_dQaccum) {
            Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
            Tensor gdQaccum = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(m_block));
            GmemTiledCopyAccum gmem_tiled_copy_dQaccum;
            auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
            Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
            Tensor zero = make_fragment_like(tdQgdQaccum);
            clear(zero);
            cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, zero, tdQgdQaccum);
        }

        if (params.dq_semaphore != nullptr && thread_idx == 0) {
            int const num_batch = params.num_batch;
            int const num_head = get<2>(params.shape_O);
            params.dq_semaphore[bidh + bidb * num_head + m_block * num_head * num_batch] = 0;
        }

    }

};

} // namespace flash
