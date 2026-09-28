/******************************************************************************
 * Softplus attention with a fused output RMSNorm, as a drop-in for flash::Softmax.
 *
 *   U_i = sum_j softplus(q_i . k_j * scale) v_j,    O_i = U_i / sqrt(mean_d(U_i^2) + eps)
 *
 * Same interface as flash::Softmax so the FA3 mainloop (TMA, WGMMA, warp specialization,
 * intra-warpgroup overlap, pingpong) is untouched:
 *   max_get_scale -> all ones, rescale_o -> no-op: there is no running max to correct for.
 *   online_softmax -> elementwise softplus.
 *   finalize      -> no-op; the mainloop calls normalize_o(tOrO) once the last PV GEMM landed.
 *
 * Everything runs in the base-2 domain the mainloop already hands us (y = s * scale * log2e):
 *   softplus(x) = ln2 * sp2(y),  sp2(y) = max(y, 0) + log2(1 + 2^-|y|).
 * The ln2 is a global factor on U, and RMSNorm is scale invariant up to eps, so it is dropped
 * and eps is rescaled instead: O = U'/sqrt(mean(U'^2) + eps/ln2^2) with U' = U/ln2, exactly.
 * row_sum ends up holding the true rms (of U), for the backward pass.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

// Which elementwise sp2 to use. Compile-time, so there is no branch in the inner loop.
enum SoftplusImpl : int {
    kSpMufu2    = 0,  // max(y,0) + lg2(1 + ex2(-|y|))           2 MUFU
    kSpPoly3    = 1,  // max(y,0) + z*P3(z), z = ex2(-|y|)        1 MUFU + 5 FMA-pipe
    kSpPoly4    = 2,  // same, degree-4 log polynomial            1 MUFU + 6 FMA-pipe
    kSpSoftExp  = 3,  // software 2^x (degree 3) + P3              0 MUFU
    kSpMix      = 4,  // kSpSoftExp on every 4th column, kSpPoly3 elsewhere (FA4-style split)
    kSpNaive    = 5,  // y > 32 ? y : lg2(1 + ex2(y))              2 MUFU, fewest instructions
    // kSpPoly3 as ptxas actually schedules it is ~8.5 FP32-pipe ops per score: FMUL (scale),
    // FADD |y| + FADD -x (MUFU takes no |.|/- modifiers), FMNMX, 4 FFMA. The FP32 pipe, not
    // MUFU, is then the limit. kSpPoly3x works on the raw score s and gets the same function:
    //   max(s,0) as a signed-int max of the bits    -> IMNMX, INT pipe
    //   -|s| * c in one FMUL with operand modifiers -> 1 FP32 op, feeds MUFU.EX2 directly
    //   sp2(s*c)/c = max(s,0) + z * P3(z)/c         -> 1/c folded into runtime coefficients
    // The global 1/c joins the ln2 that RMSNorm already absorbs. 5 FP32 + 1 INT + 1 MUFU.
    kSpPoly3x   = 6,
    // Diagnostics, not softplus: bound the cost of the kernel structure itself.
    kSpIdentity = 7,  // A = s
    kSpExpOnly  = 8,  // A = 2^(s*c): softmax's elementwise work minus the max
    // "Pre": the caller has folded scale*log2e into Q (nanochat can do it for free in the q RMSNorm
    // that precedes attention), so the GEMM already produces y and the per-score FMUL is gone.
    kSpNaivePre = 9,   // lg2(1 + ex2(y)); no overflow guard: needs y < 126, true under QK-norm
                       // (|y| <= sqrt(D) * log2e ~ 16.3 at D=128). 1 FADD + 2 MUFU.
    kSpPoly3Pre = 10,  // relu_int + sign LOP3 + ex2 + 4 FFMA
    kSpPoly2Pre = 11,  // same with the degree-2 fit (rel err 2.8e-3, ~BF16 half-ulp): 3 FFMA
    kSpMixPre   = 12,  // columns alternate NaivePre / Poly3Pre: 1.5 MUFU + ~2.5 FP32 per score
    // Softplus-shaped replacement (not softplus): cheaper than any accurate softplus. With x = s*scale:
    //   kFnRexp: A = relu(x) + e^-|x| / 2. softplus's asymptotes, C1 (both one-sided slopes 1/2 at 0),
    //            convex, exponential tail; A(0) = 0.5 vs ln2. Forward: FMUL(-|s|*c) + MUFU + FFMA
    //            + int relu. Backward: dA/dx = x>=0 ? 1 - z/2 : z/2 with the same z = e^-|x|.
    kFnRexp     = 20,
};

#ifndef FLASHATTN_SOFTPLUS_IMPL
#define FLASHATTN_SOFTPLUS_IMPL 1
#endif
// Backward recomputes A with a base-2 variant (its dS formula relies on P = sp2 = softplus/ln2).
#ifndef FLASHATTN_SOFTPLUS_IMPL_BWD
#define FLASHATTN_SOFTPLUS_IMPL_BWD 1
#endif
#ifndef FLASHATTN_SOFTPLUS_EPS
#define FLASHATTN_SOFTPLUS_EPS 1e-6f
#endif

__device__ __forceinline__ float ex2_approx(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float lg2_approx(float x) {
    float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

// log2(1+z)/z on z in [0,1], relative-error fits (max rel err 4.0e-4 and 5.7e-5). Folding the
// log2e into the fit keeps the whole path in base 2. Both are well below BF16's 2^-9, which is
// the precision A has when it enters the PV GEMM.
__device__ __forceinline__ float log2_ratio_p3(float z) {
    float p = -0.10724405944347382f;
    p = fmaf(z, p,  0.36644792556762695f);
    p = fmaf(z, p, -0.7017236948013306f);
    return fmaf(z, p, 1.442124366760254f);
}
__device__ __forceinline__ float log2_ratio_p2(float z) {
    return fmaf(z, fmaf(z, 0.20442496240139008f, -0.6401774883270264f), 1.4385943412780762f);
}
__device__ __forceinline__ float log2_ratio_p4(float z) {
    float p = 0.059646125882864f;
    p = fmaf(z, p, -0.2271004617214203f);
    p = fmaf(z, p,  0.4418795704841614f);
    p = fmaf(z, p, -0.7169807553291321f);
    return fmaf(z, p, 1.442612648010254f);
}

// 2^u for u in [-126, 0] without MUFU. Floor via the 1.5*2^23 magic constant in round-down
// mode (no F2I), 2^n assembled in the exponent field, 2^r by a degree-3 fit (rel err 7.5e-5).
__device__ __forceinline__ float exp2_soft(float u) {
    u = fmaxf(u, -126.f);                             // also maps -inf to a normal value
    float const magic = 12582912.f;                   // 1.5 * 2^23
    float const f = __fadd_rd(u, magic);              // floor(u) + magic, exactly
    float const n = f - magic;                        // floor(u) as a float
    float const r = u - n;                            // [0, 1)
    float e = 0.07802290469408035f;
    e = fmaf(r, e, 0.22606560587882996f);
    e = fmaf(r, e, 0.6958361864089966f);
    e = fmaf(r, e, 0.9999247193336487f);
    // the low mantissa bits of f are n (two's complement); << 23 moves them to the exponent
    int const bits = (__float_as_int(f) << 23) + 0x3f800000;
    return __int_as_float(bits) * e;
}

// float max(x, 0) on the INT pipe: negative floats are negative as signed ints, and -inf/-0 go to +0.
__device__ __forceinline__ float relu_int(float x) { return __int_as_float(max(__float_as_int(x), 0)); }

template <int Impl>
__device__ __forceinline__ float sp2(float y) {
    if constexpr (Impl == kSpNaive) {
        return y > 32.f ? y : lg2_approx(1.f + ex2_approx(y));
    } else if constexpr (Impl == kSpExpOnly) {  // diagnostic only: wrong function, cheapest cost
        return ex2_approx(y);
    } else if constexpr (Impl == kSpNaivePre) {  // unguarded: bounded scores (QK-norm)
        return lg2_approx(1.f + ex2_approx(y));
    } else {
        // Both on the INT pipe (IMNMX, LOP3): as FP32 ops, -|y| alone cost two FADDs, since
        // MUFU.EX2 takes no operand modifiers.
        float const pos = relu_int(y);
        float const u = __int_as_float(__float_as_int(y) | int(0x80000000));  // -|y|; masked -inf stays -inf -> z = 0
        if constexpr (Impl == kSpMufu2) {
            return pos + lg2_approx(1.f + ex2_approx(u));
        } else if constexpr (Impl == kSpPoly3 || Impl == kSpPoly3Pre) {
            float const z = ex2_approx(u);
            return fmaf(z, log2_ratio_p3(z), pos);
        } else if constexpr (Impl == kSpPoly2Pre) {
            float const z = ex2_approx(u);
            return fmaf(z, log2_ratio_p2(z), pos);
        } else if constexpr (Impl == kSpPoly4) {
            float const z = ex2_approx(u);
            return fmaf(z, log2_ratio_p4(z), pos);
        } else {  // kSpSoftExp
            float const z = exp2_soft(u);
            return fmaf(z, log2_ratio_p3(z), pos);
        }
    }
}

// -|s| * c, the exp2 argument of rexp. With FLASHATTN_SOFTPLUS_PRESCALED the caller has folded
// scale*log2e into Q (and passes softmax_scale = ln2, so c = 1): then it is a sign-bit OR on the
// INT pipe instead of an FMUL.
template <int Impl>
__device__ __forceinline__ float neg_abs_arg(float s, float c) {
#ifdef FLASHATTN_SOFTPLUS_PRESCALED
    return __int_as_float(__float_as_int(s) | int(0x80000000));
#else
    return -fabsf(s) * c;  // one FMUL with operand modifiers
#endif
}

// kFnRexp on a raw score s. c = scale*log2e; k0 as set up in Softplus below.
// Returns g with A = u_factor * g (u_factor = scale).
template <int Impl>
__device__ __forceinline__ float fn_value(float s, float c, float k0) {
    static_assert(Impl == kFnRexp);
    float const z = ex2_approx(neg_abs_arg<Impl>(s, c));        // e^-|x|
    return fmaf(k0, z, relu_int(s));
}

// Same, plus dA/dx for the backward (x = s*scale; FA3 applies the scale to dQ/dK itself).
// The backward needs dS = dP * dA/dx with dA/dx = x >= 0 ? 1 - z/2 : z/2. Twice that is
// 1 + sign(x) * (1 - z), so 2 dS = dP + dP * t with t = (1 - z) carrying x's sign: one FADD for t
// (the sign is a LOP3 on the INT pipe, 1 - z >= 0), one FFMA for dS -- one FP32 op per score fewer
// than forming dA/dx first, i.e. as many as FA3's softmax backward. The factor 2 is taken out of
// dK and dQ with their softmax_scale (kFnRexpDSFactor). Masked (-inf): z = 0, t = -1, 2 dS = 0.
template <int Impl>
__device__ __forceinline__ float fn_value_dstwice(float s, float c, float k0, float& t) {
    static_assert(Impl == kFnRexp);
    float const z = ex2_approx(neg_abs_arg<Impl>(s, c));
    t = __int_as_float(__float_as_int(1.f - z) | (__float_as_int(s) & int(0x80000000)));
    return fmaf(k0, z, relu_int(s));
}
constexpr float kFnRexpDSFactor = 0.5f;  // the backward's dS is 2x the true one (fn_value_dstwice)

template <int kNRows, int Impl = FLASHATTN_SOFTPLUS_IMPL>
struct Softplus {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_sum;  // rms of U per row after normalize_o (the epilogue stores it where LSE goes)
    float const scale_log2;
    // What the accumulated U is off from the true U by: ln2 for the base-2 variants, ln2 * c
    // for kSpPoly3x which also leaves out the score scale. RMSNorm cancels it; eps absorbs it.
    float const u_factor;
    float k0, k1, k2, k3;  // P3 / c, kSpPoly3x only

    CUTLASS_DEVICE Softplus(float const softmax_scale_log2_)
        : scale_log2(softmax_scale_log2_),
          u_factor(Impl == kSpPoly3x ? 0.6931471805599453f * softmax_scale_log2_
                   : Impl == kFnRexp ? 0.6931471805599453f * softmax_scale_log2_   // = softmax scale
                   : 0.6931471805599453f) {
        if constexpr (Impl == kFnRexp) {
            // U accumulates g = relu(s) + k * 2^(-|s| c) = A / scale, k = 1 / (2 scale)
            k0 = 0.5f / (0.6931471805599453f * softmax_scale_log2_);
        }
        if constexpr (Impl == kSpPoly3x) {
            float const inv_c = 1.f / softmax_scale_log2_;
            k0 = 1.442124366760254f * inv_c;  k1 = -0.7017236948013306f * inv_c;
            k2 = 0.36644792556762695f * inv_c; k3 = -0.10724405944347382f * inv_c;
        }
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        TensorT scores_scale;
        cute::fill(scores_scale, 1.f);
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                if constexpr (Impl == kSpIdentity) { continue; }
                if constexpr (Impl == kFnRexp) {
                    scores(mi, ni) = fn_value<Impl>(scores(mi, ni), scale_log2, k0);
                    continue;
                }
                if constexpr (Impl >= kSpNaivePre) {
                    float const y = scores(mi, ni);
                    float const u = __int_as_float(__float_as_int(y) | int(0x80000000));
                    bool const naive = Impl == kSpNaivePre || (Impl == kSpMixPre && (ni % 2 == 0));
                    if (naive) {
                        scores(mi, ni) = lg2_approx(1.f + ex2_approx(y));
                    } else if constexpr (Impl == kSpPoly2Pre) {
                        float const z = ex2_approx(u);
                        scores(mi, ni) = fmaf(z, log2_ratio_p2(z), relu_int(y));
                    } else {
                        float const z = ex2_approx(u);
                        scores(mi, ni) = fmaf(z, log2_ratio_p3(z), relu_int(y));
                    }
                    continue;
                }
                if constexpr (Impl == kSpExpOnly) { scores(mi, ni) = ex2_approx(scores(mi, ni) * scale_log2); continue; }
                if constexpr (Impl == kSpPoly3x) {
                    float const s = scores(mi, ni);
                    float const z = ex2_approx(-fabsf(s) * scale_log2);   // FMUL -|s|, c; MUFU
                    float p = fmaf(z, k3, k2);
                    p = fmaf(z, p, k1);
                    p = fmaf(z, p, k0);
                    scores(mi, ni) = fmaf(z, p, relu_int(s));
                    continue;
                }
                float const y = scores(mi, ni) * scale_log2;
                if constexpr (Impl == kSpMix) {
                    // Static split: a quarter of the columns on the FMA pipe, the rest on MUFU.
                    scores(mi, ni) = (ni % 4 == 3) ? sp2<kSpSoftExp>(y) : sp2<kSpPoly3>(y);
                } else {
                    scores(mi, ni) = sp2<Impl>(y);
                }
            }
        }
    };

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        TensorT scores_scale;
        cute::fill(scores_scale, 1.f);
        return scores_scale;
    };

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) { };

    // RMSNorm epilogue, in registers. With the WGMMA accumulator layout one row of O lives in the
    // 4 threads of a quad, so the sum of squares is a thread-local reduction plus 2 shuffles --
    // the same reduction softmax does for its row sum, done once per tile instead of per K block.
    template<typename Tensor1>
    __forceinline__ __device__ void normalize_o(Tensor1 &acc_o) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        static constexpr int kCols = 4 * CUTE_STATIC_V(size<1>(acc_o_rowcol));  // = head dim V
        float const eps_scaled = FLASHATTN_SOFTPLUS_EPS / (u_factor * u_factor);
        SumOp<float> sum_op;
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float ss = 0.f;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { ss = fmaf(acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni), ss); }
            ss = Allreduce<4>::run(ss, sum_op);
            float const ms = ss * (1.f / kCols) + eps_scaled;
            float const inv_rms = rsqrtf(ms);
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= inv_rms; }
            row_sum(mi) = u_factor * sqrtf(ms);  // rms of U = u_factor * rms of what we accumulated
        }
    };

};

}  // namespace flash
