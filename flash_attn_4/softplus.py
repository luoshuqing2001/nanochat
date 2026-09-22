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
# approximation of the architecture -- it is the same function to every bit that survives
# the cast into the PV matmul. It exists because the exact form needs a second
# transcendental, and that one log2 costs 4-7% of the forward (see VENDOR.md).
_LOG1P_COEFFS = (
    0.999981869, -0.499187794, 0.324411505, -0.208668975, 0.10028652, -0.0236890026,
)
# FA4_SOFTPLUS_EXACT=1 goes back to log2(1 + exp2(...)) for a reference run.
_USE_POLY = os.environ.get("FA4_SOFTPLUS_EXACT", "0") != "1"


@cute.jit
def softplus_(x: cute.TensorSSA, zero: cute.TensorSSA) -> cute.TensorSSA:
    """max(x, 0) + log(1 + exp(-|x|)), element-wise, in log2 space.

    `zero` is a same-shaped vector of zeros, hoisted out by the caller: cute.math.max
    needs both operands to be TensorSSA, and cute.full() recurses to death on this DSL
    version, so the caller materializes it from a zeroed rmem fragment instead.
    """
    abs_x = cute.math.abs(x, fastmath=True)
    # y = exp(-|x|) in (0, 1]; at a masked -inf entry y is 0 and the whole thing is 0.
    y = cute.math.exp2(abs_x * (-LOG2_E), fastmath=True)
    if cutlass.const_expr(_USE_POLY):
        c0, c1, c2, c3, c4, c5 = _LOG1P_COEFFS
        poly = c4 + y * c5
        poly = c3 + y * poly
        poly = c2 + y * poly
        poly = c1 + y * poly
        poly = c0 + y * poly
        return cute.math.max(x, zero) + y * poly
    return cute.math.max(x, zero) + cute.math.log2(1.0 + y, fastmath=True) * LN2


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

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        # Without a score_mod, FA4 folds log2(e) into scale_log2 and passes softmax_scale
        # as None (see utils.compute_softmax_scale_log2). Softplus is not scale-invariant
        # in log2 space, so recover the true scale; it is one scalar multiply, hoisted out
        # of the row loop.
        if softmax_scale is None:
            softmax_scale = scale_log2 * LN2
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softplus(scale_log2, num_rows, row_sum, arch, softmax_scale)

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
            acc_S_mn[r, None].store(softplus_(acc_S_mn[r, None].load() * softmax_scale, zero))
        return None

    def finalize(self, final_scale: Float32 = 1.0, sink_val=None) -> None:
        """No row sum to divide by. The n^-alpha row scale is applied in the epilogue."""
        return None

    def rescale_O(self, acc_O: cute.Tensor, row_scale) -> None:
        """No-op: without normalization the accumulator is never rescaled mid-loop."""
        return None
