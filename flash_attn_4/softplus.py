# Softplus attention, in the shape of flash_attn_4/softmax.py's Softmax.
#
# Replaces the softmax score map with an element-wise softplus:
#
#     O_i = n_i^-alpha * sum_j softplus(s_ij) v_j
#
# There is no normalization by a row sum, so nothing in the inner loop depends on
# the other key tiles: no running max, no running sum, and no rescaling of the
# accumulator between tiles. `online_softmax` becomes a pure element-wise map and
# `rescale_O`/`finalize` become no-ops -- the `n_i^-alpha` row scale is a function
# of the row index alone and is applied once in the epilogue (see
# flash_fwd_softplus.py). That is the whole reason this is cheaper than softmax,
# and it is also what makes the split-K path in flash_fwd_softplus.py a plain sum.
#
# Numerics: softplus is evaluated as max(s, 0) + log1p(exp(-|s|)), which never
# exponentiates a positive number, so no row max is needed for stability. The log/exp
# go through log2/exp2 to hit the hardware instructions. This matches
# nanochat/softplus_attention.py's Triton kernel term for term.
#
# The max() must be a real max, not the branchless (s + |s|) / 2: mask.py fills masked
# entries with -inf, and (-inf + inf) / 2 is NaN. With a real max, -inf maps to
# max(-inf, 0) + log(1 + exp2(-inf)) = 0, which is exactly what a masked entry should
# contribute to an unnormalized sum -- so softplus needs no separate masked-entry path.

import math
import os
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)

# log1p(y) for y in [0, 1], as y * poly(y): a degree-5 minimax fit whose maximum absolute
# error is 1.3e-5. bf16's resolution near ln2 is 2.7e-3, 200x coarser, so this is not an
# approximation of the architecture -- its error is small relative to the PV matmul input precision. It exists because the exact form needs a second
# transcendental, and that one log2 costs 4-7% of the forward (see VENDOR.md).
_LOG1P_COEFFS = (
    0.999981869, -0.499187794, 0.324411505, -0.208668975, 0.10028652, -0.0236890026,
)
# FA4_SOFTPLUS_EXACT=1 goes back to log2(1 + exp2(...)) for a reference run.
_USE_POLY = os.environ.get("FA4_SOFTPLUS_EXACT", "0") != "1"
# Experimental direct approximation; set before importing/compiling kernels.
# 1: [0,4] fast path; 2: two intervals covering [0,8]. Exact mode takes priority.
_DIRECT_MODE = int(os.environ.get("FA4_SOFTPLUS_DIRECT", "0"))
if _DIRECT_MODE not in (0, 1, 2):
    raise ValueError("FA4_SOFTPLUS_DIRECT must be 0, 1, or 2")
_DIRECT_COEFFS = ((0.6931471805599453, -2.0, 1.9778681705169956, 0.24362300168752393, -2.4102316457256467, 2.3880393630655834, -1.0622525769358822, 0.18795643474929064), (0.018149927917809738, -0.07194483984836623, 0.14100027895116557, -0.17880879647487108, 0.15799763647440032, -0.09536949444820758, 0.03528865941453619, -0.005977965613571163))

_LOG_DEGREE = int(os.environ.get("FA4_SOFTPLUS_LOG_DEGREE", "5"))
if _LOG_DEGREE not in (3, 4, 5):
    raise ValueError("FA4_SOFTPLUS_LOG_DEGREE must be 3, 4, or 5")
_SHORT_LOG_COEFFS = {
    3: (1.0, -0.48776925260429926, 0.2547399592083375, -0.07382352604409295),
    4: (1.0, -0.49724314306630324, 0.3064121897397085, -0.15682170546507299, 0.040799839351612986),
}
_PACKED_POLY = os.environ.get("FA4_SOFTPLUS_PACKED", "0") == "1"
_DIRECT_DERIV = tuple(i * c * .25 for i, c in enumerate(_DIRECT_COEFFS[0]) if i)


@cute.jit
def _packed_horner(x, coeffs: cutlass.Constexpr):
    assert cute.size(x.shape) % 2 == 0, "packed polynomial requires even fragment size"
    buf = cute.make_rmem_tensor(x.shape, Float32)
    buf.store(x)
    flat = cute.make_tensor(buf.iterator, cute.make_layout(cute.size(buf)))
    result = cute.make_rmem_tensor(cute.size(buf), Float32)
    for j in cutlass.range_constexpr(0, cute.size(buf), 2):
        pair = (flat[j], flat[j + 1])
        value = (Float32(coeffs[-1]), Float32(coeffs[-1]))
        for i in cutlass.range_constexpr(len(coeffs) - 2, -1, -1):
            value = cute.arch.fma_packed_f32x2(value, pair, (Float32(coeffs[i]), Float32(coeffs[i])))
        result[j] = value[0]
        result[j + 1] = value[1]
    return result.load().reshape(x.shape)


@cute.jit
def _log1p_poly(y: cute.TensorSSA, zero: cute.TensorSSA, estrin: cutlass.Constexpr = False):
    if cutlass.const_expr(_LOG_DEGREE != 5):
        coeffs = _SHORT_LOG_COEFFS[_LOG_DEGREE]
        if cutlass.const_expr(_PACKED_POLY):
            return _packed_horner(y, coeffs)
        value = zero.reshape(y.shape) + coeffs[-1]
        for i in cutlass.range_constexpr(len(coeffs) - 2, -1, -1):
            value = value * y + coeffs[i]
        return value
    if cutlass.const_expr(_PACKED_POLY):
        return _packed_horner(y, _LOG1P_COEFFS)
    c0, c1, c2, c3, c4, c5 = _LOG1P_COEFFS
    if cutlass.const_expr(estrin):
        y2 = y * y
        return (c0 + c1 * y) + y2 * ((c2 + c3 * y) + y2 * (c4 + c5 * y))
    return c0 + y * (c1 + y * (c2 + y * (c3 + y * (c4 + y * c5))))


@cute.jit
def _stable_softplus(x: cute.TensorSSA, zero: cute.TensorSSA, estrin: cutlass.Constexpr = False) -> cute.TensorSSA:
    """max(x, 0) + log(1 + exp(-|x|)), element-wise, in log2 space.

    `zero` is a same-shaped vector of zeros, hoisted out by the caller: cute.math.max
    needs both operands to be TensorSSA, and cute.full() recurses to death on this DSL
    version, so the caller materializes it from a zeroed rmem fragment instead.
    """
    abs_x = cute.math.abs(x, fastmath=True)
    # y = exp(-|x|) in (0, 1]; at a masked -inf entry y is 0 and the whole thing is 0.
    y = cute.math.exp2(abs_x * (-LOG2_E), fastmath=True)
    if cutlass.const_expr(_USE_POLY):
        return cute.math.max(x, zero) + y * _log1p_poly(y, zero, estrin)
    return cute.math.max(x, zero) + cute.math.log2(1.0 + y, fastmath=True) * LN2


@cute.jit
def _stable_pair(x: cute.TensorSSA, zero: cute.TensorSSA, estrin: cutlass.Constexpr = False):
    """Evaluate softplus and sigmoid, preserving gradients for very negative scores."""
    y = cute.math.exp2(cute.math.abs(x, fastmath=True) * (-LOG2_E), fastmath=True)
    inv = cute.math.rcp(1.0 + y, approx=True, ftz=True)
    sig = cute.where(x >= 0.0, inv, y * inv)
    if cutlass.const_expr(_USE_POLY):
        sp = cute.math.max(x, zero) + y * _log1p_poly(y, zero, estrin)
    else:
        sp = cute.math.max(x, zero) + cute.math.log2(1.0 + y, fastmath=True) * LN2
    return sp, sig


@cute.jit
def _direct_eval(x, zero, gradient: cutlass.Constexpr):
    zero = zero.reshape(x.shape)
    a = cute.math.abs(x, fastmath=True)
    masked = x == -Float32.inf
    a = cute.where(masked, zero, a)
    limit = 4.0 if cutlass.const_expr(_DIRECT_MODE == 1) else 8.0
    inside = a.reduce(cute.ReductionOp.MAX, init_val=0.0, reduction_profile=0) <= limit
    sp = zero
    sig = zero
    if cute.arch.vote_all_sync(inside):
        if cutlass.const_expr(_DIRECT_MODE == 2):
            high = a > 4.0
            t = cute.where(high, a * 0.25 - 1.0, a * 0.25)
        else:
            t = a * 0.25
        if cutlass.const_expr(_PACKED_POLY and _DIRECT_MODE == 1):
            value = _packed_horner(t, _DIRECT_COEFFS[0])
            deriv = _packed_horner(t, _DIRECT_DERIV) * 4.0 if cutlass.const_expr(gradient) else zero
        else:
            value = zero
            deriv = zero
            for i in cutlass.range_constexpr(7, -1, -1):
                if cutlass.const_expr(gradient):
                    deriv = deriv * t + value
                if cutlass.const_expr(_DIRECT_MODE == 2):
                    coeff = cute.where(high, zero + _DIRECT_COEFFS[1][i], zero + _DIRECT_COEFFS[0][i])
                else:
                    coeff = zero + _DIRECT_COEFFS[0][i]
                value = value * t + coeff
        sp = cute.where(masked, zero, cute.math.max(x, zero) + value)
        if cutlass.const_expr(gradient):
            deriv = deriv * 0.25
            sig = cute.where(masked, zero, cute.where(x >= 0.0, 1.0 + deriv, -deriv))
        else:
            sig = zero
    else:
        if cutlass.const_expr(gradient):
            sp, sig = _stable_pair(x, zero)
        else:
            sp = _stable_softplus(x, zero)
            sig = zero
    return sp, sig


@cute.jit
def softplus_(x: cute.TensorSSA, zero: cute.TensorSSA, estrin: cutlass.Constexpr = False):
    if cutlass.const_expr(_DIRECT_MODE > 0 and _USE_POLY):
        sp, _ = _direct_eval(x, zero, False)
        return sp
    return _stable_softplus(x, zero, estrin)


@cute.jit
def softplus_and_sigmoid_(x: cute.TensorSSA, zero: cute.TensorSSA, estrin: cutlass.Constexpr = False):
    if cutlass.const_expr(_DIRECT_MODE > 0 and _USE_POLY):
        return _direct_eval(x, zero, True)
    return _stable_pair(x, zero, estrin)


@dataclass
class Softplus(ParamsBase):
    """Drop-in replacement for `Softmax` at the three call sites in flash_fwd.py.

    Keeps `Softmax`'s interface (create/reset/online_softmax/rescale_O/finalize) so a
    forward kernel can swap score maps by setting `score_map_cls`, with no other change
    to the mainloop. `row_sum` is carried only because the epilogue takes it as the LSE
    argument; softplus has no LSE, so it stays zero.
    """

    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None
    estrin: cutlass.Constexpr[bool] = False

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
        estrin: bool = False,
    ):
        # Without a score_mod, FA4 folds log2(e) into scale_log2 and passes softmax_scale
        # as None (see utils.compute_softmax_scale_log2). Softplus is not scale-invariant
        # in log2 space, so recover the true scale; it is one scalar multiply, hoisted out
        # of the row loop.
        if softmax_scale is None:
            softmax_scale = scale_log2 * LN2
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softplus(scale_log2, num_rows, row_sum, arch, softmax_scale, estrin)

    def reset(self) -> None:
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> None:
        """Map scores to softplus in place. Returns no row scale: there is nothing to rescale."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        softmax_scale = self.softmax_scale
        zero_frag = cute.make_rmem_tensor(cute.size(acc_S_mn, mode=[1]), Float32)
        zero_frag.fill(0.0)
        zero = zero_frag.load()
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            acc_S_mn[r, None].store(softplus_(acc_S_mn[r, None].load() * softmax_scale, zero, self.estrin))
        return None

    def finalize(self, final_scale: Float32 = 1.0, sink_val=None) -> None:
        """No row sum to divide by. The n^-alpha row scale is applied in the epilogue."""
        return None

    def rescale_O(self, acc_O: cute.Tensor, row_scale) -> None:
        """No-op: without normalization the accumulator is never rescaled mid-loop."""
        return None


class SoftplusEstrin(Softplus):
    @staticmethod
    def create(scale_log2, num_rows, arch=80, softmax_scale=None):
        return Softplus.create(scale_log2, num_rows, arch, softmax_scale, estrin=True)
