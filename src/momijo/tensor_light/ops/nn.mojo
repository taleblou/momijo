# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / NEURAL-NETWORK OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Neural-Network Operations
# Module:        momijo.tensor_light.ops.nn
# Source:        src/momijo/tensor_light/ops/nn.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines FP32 neural-network operations including activations,
#   normalization, and attention.
#
#   Public declarations include `relu`, `gelu_tanh`, `softmax_last_axis`,
#   `logsumexp_vector`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `relu`, `gelu_tanh`,
#     `softmax_last_axis`, `logsumexp_vector`, `layernorm_last_axis` and
#     related symbols.
#   - Compute stable softmax on the declared last axis.
#   - Provide layer normalization with checked output properties.
#   - Implement the declared 2-D scaled dot-product attention.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
#
# ------------------------------------------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------------------------------------------
#
# Authors:
#   Morteza Taleblou
#   Mitra Daneshmand
#
# Organization:  https://github.com/momijo
# Repository:    https://github.com/taleblou/momijo
#
# Maintainer Website:
#   https://taleblou.ir/
#
# Maintainer LinkedIn:
#   https://www.linkedin.com/in/morteza-taleblou
#
# ------------------------------------------------------------------------------
# LICENSING AND ATTRIBUTION
# ------------------------------------------------------------------------------
#
# License:       MIT
# License File:  LICENSE
#
# Copyright and Licensing:
#   This source file is distributed under the MIT License.
#   The SPDX declarations above identify its applicable license and
#   stated copyright holders.
#
#   Redistributions containing this software or substantial portions
#   of it must retain the applicable copyright and permission notices
#   as required by the MIT License.
#
# Attribution:
#   Acknowledgment of the Momijo project in downstream documentation
#   is appreciated but is not an additional licensing requirement.
#
# ==============================================================================

# FP32-only CPU application operators. All owning tensor payloads remain
# Float32 and all static ranks remain in the existing range 0..8.
# Canonical QA: BEN-031/066/067/068/085/086, TEST-046/047/106/115,
# EX-041/042/054/055. These are implementations, NOT benchmark PASS evidence.
# Strict numerical promises: finite inputs for normalization/attention,
# stable max-subtracted softmax, FP32 ordered dot/accumulation, no TF32.
from std.collections import List
from std.sys.info import size_of
from std.math import exp as math_exp, expm1 as math_expm1, log as math_log, log1p as math_log1p, sqrt as math_sqrt, tanh as math_tanh, abs as math_abs
from std.utils.numerics import isfinite as fp32_isfinite, isinf as fp32_isinf
from ..core.tensor import Tensor
from ..factory.creation import _complete_overwrite_output
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..storage.host_storage import HostStorage
from ..iterator.nd_iterator import NDIndexIterator
from ..internal.status import TensorResult, TensorStatus


@inline(.always)
def _relu_scalar_fp32(value: Float32) -> Float32:
    # Preserve NaNs; define ReLU(-0) = +0, as for maximum(+0, x).
    if value != value:
        return value
    if value > Float32(0.0):
        return value
    return Float32(0.0)


@inline(.always)
def _gelu_tanh_scalar_fp32(value: Float32) -> Float32:
    # The tanh GELU approximation is a separately named operation; it is
    # not advertised as a bitwise-exact reference to the erf formulation.
    if value != value:
        return value
    if Bool(fp32_isinf(value)):
        if value > Float32(0.0):
            return value
        return Float32(0.0)
    var cube = value * value * value
    var argument = Float32(0.7978845608028654) * (
        value + Float32(0.044715) * cube
    )
    return Float32(0.5) * value * (Float32(1.0) + math_tanh(argument))


def relu[rank: Int](input: Tensor[rank, HostStorage]) -> TensorResult[Tensor[rank, HostStorage]]:
    var checked = input._checked_current_layout()
    if not checked.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked.status)
    var layout = checked.payload.value()
    var allocated = _complete_overwrite_output[rank](layout.shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var output = allocated.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var src = input.impl.storage.borrow_span()
    if layout.is_c_contiguous():
        for i in range(layout.shape.numel()):
            dst[i] = _relu_scalar_fp32(src[layout.offset + i])
    else:
        var cursor = NDIndexIterator[rank](layout.shape)
        while not cursor.done():
            dst[cursor.produced] = _relu_scalar_fp32(
                src[layout.physical_offset_unchecked(cursor.coordinates)]
            )
            cursor.advance()
    return (output^)._finish_complete_overwrite()


def gelu_tanh[rank: Int](input: Tensor[rank, HostStorage]) -> TensorResult[Tensor[rank, HostStorage]]:
    var checked = input._checked_current_layout()
    if not checked.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked.status)
    var layout = checked.payload.value()
    var allocated = _complete_overwrite_output[rank](layout.shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var output = allocated.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var src = input.impl.storage.borrow_span()
    if layout.is_c_contiguous():
        for i in range(layout.shape.numel()):
            dst[i] = _gelu_tanh_scalar_fp32(src[layout.offset + i])
    else:
        var cursor = NDIndexIterator[rank](layout.shape)
        while not cursor.done():
            dst[cursor.produced] = _gelu_tanh_scalar_fp32(
                src[layout.physical_offset_unchecked(cursor.coordinates)]
            )
            cursor.advance()
    return (output^)._finish_complete_overwrite()


def softmax_last_axis[rank: Int](
    input: Tensor[rank, HostStorage]
) -> TensorResult[Tensor[rank, HostStorage]] where rank >= 1:
    # A row is a contiguous logical trailing axis but its INPUT may have
    # positive/negative/zero strides. A validated physical span authorizes
    # unchecked indexed loads; output is always new unique C-order storage.
    var verified = input._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](verified.status)
    var layout = verified.payload.value()
    var width = layout.shape.dims.get(rank - 1)
    if width == 0:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var created = _complete_overwrite_output[rank](layout.shape)
    if not created.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](created.status)
    var output = created.payload.take()
    var source = input.impl.storage.borrow_span()
    var destination = output.impl.storage.borrow_mut_span()
    var rows = layout.shape.numel() // width
    var last_step = layout.strides.get(rank - 1)
    for row in range(rows):
        var base = layout.offset
        var remaining = row
        comptime for j in range(rank - 1):
            var axis = rank - 2 - j
            var extent = layout.shape.dims.get(axis)
            var coordinate = remaining % extent
            remaining //= extent
            base += coordinate * layout.strides.get(axis)
        var biggest = source[base]
        for column in range(1, width):
            var value = source[base + column * last_step]
            if value != value:
                biggest = value
                break
            if value > biggest:
                biggest = value
        var out_base = row * width
        if biggest != biggest:
            # Propagate the NaN to the entire logical row without inventing
            # a valid probability for an invalid score vector.
            for column in range(width):
                destination[out_base + column] = biggest
        elif Bool(fp32_isinf(biggest)):
            if biggest < Float32(0.0):
                # All -inf has zero mass: reject instead of dividing 0 by 0.
                return TensorResult[Tensor[rank, HostStorage]](TensorStatus.UNSUPPORTED)
            var positive_infinities = 0
            for column in range(width):
                if source[base + column * last_step] == biggest:
                    positive_infinities += 1
            var probability = Float32(1.0) / Float32(positive_infinities)
            for column in range(width):
                var value = source[base + column * last_step]
                if value == biggest:
                    destination[out_base + column] = probability
                else:
                    destination[out_base + column] = Float32(0.0)
        else:
            # Accumulate in Float64 for ALL widths, including two-element
            # rows: 1 + exp(-17) rounds back to exactly 1 in Float32.
            # Converting the denominator or its reciprocal back to Float32
            # before division would still round the largest probability
            # incorrectly. Tensor data and the output are always Float32;
            # only the scalar denominator and final division are widened.
            # The exponential itself stays Float32 for the CPU hot path.
            var denominator = Float64(0.0)
            for column in range(width):
                var centered = source[base + column * last_step] - biggest
                var weight = math_exp(centered)
                destination[out_base + column] = weight
                denominator += Float64(weight)
            for column in range(width):
                destination[out_base + column] = Float32(
                    Float64(destination[out_base + column]) / denominator
                )
    return (output^)._finish_complete_overwrite()


def logsumexp_vector(input: Tensor[1, HostStorage]) -> TensorResult[Float32]:
    var verified = input._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[Float32](verified.status)
    var layout = verified.payload.value()
    var count = layout.shape.numel()
    if count == 0:
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    var source = input.impl.storage.borrow_span()
    var stride = layout.strides.get(0)
    var biggest = source[layout.offset]
    # Catch NaN in the FIRST slot too, before it can leave maxima == 0
    # and accidentally reach math_log(0) in the compensated path.
    if biggest != biggest:
        return TensorResult[Float32](biggest)
    if count == 1:
        return TensorResult[Float32](biggest)
    for i in range(1, count):
        var value = source[layout.offset + i * stride]
        if value != value:
            return TensorResult[Float32](value)
        if value > biggest:
            biggest = value
    if Bool(fp32_isinf(biggest)):
        return TensorResult[Float32](biggest)
    # Keep the exactly maximal terms SEPARATE from lower-weight terms.
    # Computing log(1 + tail) loses the tail below the Float64 unit roundoff,
    # even if the correct Float32 logsumexp is nonzero (e.g. [0,-38]).
    # math_log1p(tail / maxima) retains these sub-ULP positive masses.
    # All intermediate scalars may be Float64; the Tensor is still Float32.
    var maxima = 0
    var tail = Float64(0.0)
    var compensation = Float64(0.0)
    for i in range(count):
        var value = source[layout.offset + i * stride]
        if value == biggest:
            maxima += 1
        else:
            # Promote BEFORE subtracting to avoid Float32 overflow on
            # opposite-sign finite extrema. Each term is in [0, 1).
            var term = math_exp(Float64(value) - Float64(biggest))
            var next_tail = tail + term
            if math_abs(tail) >= math_abs(term):
                compensation += (tail - next_tail) + term
            else:
                compensation += (term - next_tail) + tail
            tail = next_tail
    var wide_maxima = Float64(maxima)
    var offset = (
        math_log(wide_maxima)
        + math_log1p((tail + compensation) / wide_maxima)
    )
    return TensorResult[Float32](Float32(Float64(biggest) + offset))


def layernorm_last_axis[rank: Int](
    input: Tensor[rank, HostStorage], epsilon: Float32 = Float32(1.0e-5)
) -> TensorResult[Tensor[rank, HostStorage]] where rank >= 1:
    # Population variance with Welford accumulation; use wide intermediates
    # for detected cancellation/overflow, preserving rank and Float32 storage.
    if not Bool(fp32_isfinite(epsilon)) or epsilon <= Float32(0.0):
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
    var checked = input._checked_current_layout()
    if not checked.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked.status)
    var layout = checked.payload.value()
    var width = layout.shape.dims.get(rank - 1)
    if width == 0:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var allocated = _complete_overwrite_output[rank](layout.shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var output = allocated.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var src = input.impl.storage.borrow_span()
    var stride = layout.strides.get(rank - 1)
    var rows = layout.shape.numel() // width
    for row in range(rows):
        var base = layout.offset
        var remaining = row
        comptime for j in range(rank - 1):
            var axis = rank - 2 - j
            var extent = layout.shape.dims.get(axis)
            var coordinate = remaining % extent
            remaining //= extent
            base += coordinate * layout.strides.get(axis)
        var mean = Float32(0.0)
        var m2 = Float32(0.0)
        var use_wide_fallback = False
        for i in range(width):
            var value = src[base + i * stride]
            if not Bool(fp32_isfinite(value)):
                return TensorResult[Tensor[rank, HostStorage]](TensorStatus.UNSUPPORTED)
            var delta = value - mean
            if not Bool(fp32_isfinite(delta)):
                use_wide_fallback = True
                break
            mean += delta / Float32(i + 1)
            var delta2 = value - mean
            var new_m2 = m2 + delta * delta2
            if (not Bool(fp32_isfinite(mean))
                or not Bool(fp32_isfinite(delta2))
                or not Bool(fp32_isfinite(new_m2))):
                use_wide_fallback = True
                break
            m2 = new_m2
        if not use_wide_fallback:
            var variance = m2 / Float32(width)
            # Welford's variance is nonnegative in exact arithmetic. A
            # negative value means the fast accumulator has lost validity;
            # do not silently clip this numerical defect to zero.
            if variance < Float32(0.0):
                use_wide_fallback = True
            else:
                var scale = math_sqrt(variance + epsilon)
                # Finite Float32 Welford arithmetic can still give grossly
                # incorrect centering without ever overflowing. For example,
                # [100000000, 100000008] rounds the mean back to 100000000
                # and incorrectly normalizes to [0, sqrt(2)] instead of
                # approximately [-1, +1]. When the Float32 representation
                # spacing at the offset is relevant to the data spread,
                # recompute the row using Float64 scalar intermediates.
                # Output storage and Tensor element dtype remain Float32.
                var rounding_risk = (
                    math_abs(mean) * Float32(0.00000011920928955078125)
                )
                # The former 1e-3 threshold still missed about 1e-3
                # absolute normalization error for near-constant rows such
                # as 25.25 plus a few binary32 ULPs. Reduce the *relative
                # centering-error budget*, leaving unit-scale activations on
                # the original FP32 fast path but routing cancellation-risk
                # rows to the already existing Float64 scalar fallback.
                if (not Bool(fp32_isfinite(scale)) or scale <= Float32(0.0)
                    or rounding_risk > scale * Float32(0.0001)):
                    use_wide_fallback = True
                else:
                    var inverse_std = Float32(1.0) / scale
                    if not Bool(fp32_isfinite(inverse_std)):
                        use_wide_fallback = True
                    else:
                        for i in range(width):
                            var normalized = (src[base + i * stride] - mean) * inverse_std
                            if not Bool(fp32_isfinite(normalized)):
                                use_wide_fallback = True
                                break
                            dst[row * width + i] = normalized
        if use_wide_fallback:
            # Only exceptional finite FP32 rows use wider INTERMEDIATES.
            # This prevents variance overflow or spurious zero normalized
            # output. The owning Tensor and its output elements remain FP32.
            # Ordinary rows keep the original fast Float32 Welford path.
            var wide_mean = Float64(0.0)
            var wide_m2 = Float64(0.0)
            for i in range(width):
                var value = src[base + i * stride]
                if not Bool(fp32_isfinite(value)):
                    return TensorResult[Tensor[rank, HostStorage]](TensorStatus.UNSUPPORTED)
                var wide_value = Float64(value)
                var wide_delta = wide_value - wide_mean
                wide_mean += wide_delta / Float64(i + 1)
                wide_m2 += wide_delta * (wide_value - wide_mean)
            var wide_variance = wide_m2 / Float64(width)
            if wide_variance < Float64(0.0):
                wide_variance = Float64(0.0)
            var wide_inverse_std = Float64(1.0) / math_sqrt(
                wide_variance + Float64(epsilon)
            )
            for i in range(width):
                dst[row * width + i] = Float32(
                    (Float64(src[base + i * stride]) - wide_mean)
                    * wide_inverse_std
                )
    return (output^)._finish_complete_overwrite()


def checked_attention_wide_scratch_nbytes(value_depth: Int) -> TensorResult[Int]:
    # The compensated attention replay keeps TWO full-width Float64 lists:
    # the numerator and its Neumaier correction. The input and output
    # tensors remain Float32; this calculation only bounds scratch bytes.
    # An FP32-valid output shape need not have an Int-addressable pair of
    # Float64 scratch buffers. Check before List(length=...) allocation.
    if value_depth < 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    var bytes_per_value = size_of[Float64]()
    if value_depth > Int.MAX // (2 * bytes_per_value):
        return TensorResult[Int](TensorStatus.OVERFLOW)
    return TensorResult[Int](value_depth * (2 * bytes_per_value))


def scaled_dot_product_attention_2d(
    q: Tensor[2, HostStorage], k: Tensor[2, HostStorage],
    v: Tensor[2, HostStorage], causal: Bool = False,
) -> TensorResult[Tensor[2, HostStorage]]:
    # IO-aware scalar reference: stable ONLINE softmax keeps O(Q*Dv) output
    # and O(1) auxiliary memory on the usual fast path. Exceptionally large
    # finite FP32 dot products may overflow before softmax; for those rows,
    # replay with Float64 temporaries and O(Dv) scratch, without changing the
    # output Tensor (still Float32, rank 2). No full QxK score matrix.
    # CPU-only; NOT a GPU/flash-attention benchmark PASS.
    var qchecked = q._checked_current_layout()
    if not qchecked.is_ok():
        return TensorResult[Tensor[2, HostStorage]](qchecked.status)
    var kchecked = k._checked_current_layout()
    if not kchecked.is_ok():
        return TensorResult[Tensor[2, HostStorage]](kchecked.status)
    var vchecked = v._checked_current_layout()
    if not vchecked.is_ok():
        return TensorResult[Tensor[2, HostStorage]](vchecked.status)
    var qlayout = qchecked.payload.value()
    var klayout = kchecked.payload.value()
    var vlayout = vchecked.payload.value()
    var queries = qlayout.shape.dims.get(0)
    var depth = qlayout.shape.dims.get(1)
    var keys = klayout.shape.dims.get(0)
    var vdepth = vlayout.shape.dims.get(1)
    if (depth == 0 or depth != klayout.shape.dims.get(1)
        or keys != vlayout.shape.dims.get(0)
        or (causal and queries != keys)):
        return TensorResult[Tensor[2, HostStorage]](TensorStatus.SIZE_MISMATCH)
    if keys == 0 and queries != 0:
        return TensorResult[Tensor[2, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var dims = StaticRankArray[2](0)
    dims.set(0, queries)
    dims.set(1, vdepth)
    var out_shape = TensorShape[2].checked(dims)
    if not out_shape.is_ok():
        return TensorResult[Tensor[2, HostStorage]](out_shape.status)
    # Initialize to zero: the online recurrence explicitly uses old output.
    var created = Tensor[2, HostStorage].zeros_checked(out_shape.payload.value())
    if not created.is_ok():
        return TensorResult[Tensor[2, HostStorage]](created.status)
    var output = created.payload.take()
    if queries == 0 or vdepth == 0:
        return TensorResult[Tensor[2, HostStorage]](output^)
    var dest = output.impl.storage.borrow_mut_span()
    var qdata = q.impl.storage.borrow_span()
    var kdata = k.impl.storage.borrow_span()
    var vdata = v.impl.storage.borrow_span()
    var inv_scale = Float32(1.0) / math_sqrt(Float32(depth))
    for qi in range(queries):
        var row_max = Float32(0.0)
        var row_denom = Float32(0.0)
        var row_initialized = False
        # The index corresponding to row_max lets an apparent FP32 score
        # tie be checked against the actual earlier winning key. This is
        # prepare-free and costs extra dot work only on tied logits.
        var row_max_key_index = 0
        var query_base = qlayout.offset + qi * qlayout.strides.get(0)
        var limit = keys
        if causal:
            limit = qi + 1
        # Extremely long sequences are assigned the numerically robust
        # Float64-accumulator reference path before any partial numerator
        # can lose small contributions. Output tensor storage is still f32.
        var needs_wide_replay = limit > 1048576
        for kj in range(limit):
            if needs_wide_replay:
                break
            var key_base = klayout.offset + kj * klayout.strides.get(0)
            var score = Float32(0.0)
            # FP32 finite products can lose differences in the SUM without
            # overflow: e.g. [1e15,1e15] dot [1e15,-1e15+one_ulp]
            # rounds to zero in FP32, although the Float64 result is huge.
            # Such logits can silently yield uniform attention instead of
            # selecting the higher-scored key. A 16384-magnitude FP32 term
            # already has a ~0.002 ULP: keep this conservative cutoff to
            # protect scores used by an exponential. Checking each term and
            # partial sum avoids invisible cancellation during accumulation.
            # Small, ordinary dot products still use the original FP32 path.
            var imprecise_dot = False
            for d in range(depth):
                var qvalue = qdata[query_base + d * qlayout.strides.get(1)]
                var kvalue = kdata[key_base + d * klayout.strides.get(1)]
                var term = qvalue * kvalue
                # Product underflow is observable after opposite-sign value
                # cancellation: a 1e-46 logit can yield a 1e-9 FP32 result
                # when V contains 1e38 and -1e38. Check the rare zero term.
                if (term == Float32(0.0) and qvalue != Float32(0.0)
                    and kvalue != Float32(0.0)):
                    imprecise_dot = True
                    break
                if term > Float32(16384.0) or term < Float32(-16384.0):
                    imprecise_dot = True
                    break
                var next_score = score + term
                # A sub-ULP but NONZERO contribution can be lost even when
                # both terms and their running sum stay below 16384. If a
                # later term cancels the large partial sum, this discarded
                # contribution can determine the entire softmax result.
                # Recompute only this exceptional row with compensated f64.
                if term != Float32(0.0) and next_score == score:
                    imprecise_dot = True
                    break
                score = next_score
                if score > Float32(16384.0) or score < Float32(-16384.0):
                    imprecise_dot = True
                    break
            if imprecise_dot:
                needs_wide_replay = True
                break
            score *= inv_scale
            if not Bool(fp32_isfinite(score)):
                # Finite FP32 q/k products can overflow, or opposite-sign
                # products can yield Inf-Inf=NaN. Recompute the ENTIRE row
                # with wide arithmetic; its per-value finite checks reject
                # genuinely invalid NaN/Inf inputs.
                needs_wide_replay = True
                break
            # Equal FP32 logits need not be mathematically equal: distinct
            # sums can round to the SAME Float32 value even if no individual
            # term was lost and every partial sum changed. This previously
            # escaped the finite-dot/overflow and exp()-equals-one guards,
            # turning cancellation-sensitive V=[+1e8,-1e8] into zero.
            #
            # Only when a tie occurs do we recompute q dot (current_key -
            # winning_key) in compensated Float64. Subtract the two Float32
            # keys BEFORE multiplying or accumulating, so a shared large
            # logit offset cannot swallow the difference. For equal actual
            # scores, keep the fast path; for a hidden difference, replay the
            # entire row with the established accurate online recurrence.
            if row_initialized and score == row_max:
                var winner_base = (
                    klayout.offset
                    + row_max_key_index * klayout.strides.get(0)
                )
                var tie_delta = Float64(0.0)
                var tie_compensation = Float64(0.0)
                for d in range(depth):
                    var qv = qdata[
                        query_base + d * qlayout.strides.get(1)
                    ]
                    var current_k = kdata[
                        key_base + d * klayout.strides.get(1)
                    ]
                    var previous_k = kdata[
                        winner_base + d * klayout.strides.get(1)
                    ]
                    var tie_term = Float64(qv) * (
                        Float64(current_k) - Float64(previous_k)
                    )
                    var tie_next = tie_delta + tie_term
                    if math_abs(tie_delta) >= math_abs(tie_term):
                        tie_compensation += (
                            (tie_delta - tie_next) + tie_term
                        )
                    else:
                        tie_compensation += (
                            (tie_term - tie_next) + tie_delta
                        )
                    tie_delta = tie_next
                if tie_delta + tie_compensation != Float64(0.0):
                    needs_wide_replay = True
                    break
            elif row_initialized:
                # A distinct Float32 score below/above the current maximum
                # does NOT prove that the *actual* dot products have the
                # same order. Different sequences of correctly rounded
                # partial sums can reverse two very close results, without
                # any tie, overflow, zero product, or swallowed term.
                # Example: 8-term scores reported as 75.35796356 and
                # 75.35795593, whose exact f32-input dot sums are instead
                # 75.35796738 and 75.35796928 (opposite ordering).
                # Only near-neighbors go to the existing wider replay; a
                # margin of two f32 machine epsilons accounts for successive
                # accumulation and scale rounding. This inexpensive check
                # adds no scratch or second dot to ordinary separated logits.
                var score_magnitude = max(
                    Float32(1.0),
                    max(math_abs(score), math_abs(row_max)),
                )
                if (math_abs(score - row_max)
                    <= score_magnitude * Float32(0.0000002384185791015625)):
                    needs_wide_replay = True
                    break
            var weight = Float32(1.0)
            var old_scale = Float32(0.0)
            var previous_denominator = Float64(row_denom)
            if row_initialized:
                if score > row_max:
                    old_scale = math_exp(row_max - score)
                    # exp() can underflow to zero in FP32 although an old
                    # large-magnitude V still contributes to the final row.
                    # The wide replay evaluates the softmax factor in f64.
                    if (old_scale == Float32(0.0)
                        or old_scale == Float32(1.0)):
                        # A rounded-to-one exp() destroys a tiny but real
                        # logit difference when large V values cancel.
                        needs_wide_replay = True
                        break
                    row_denom = row_denom * old_scale + Float32(1.0)
                    row_max = score
                    row_max_key_index = kj
                else:
                    weight = math_exp(score - row_max)
                    # Likewise, a key with an FP32-zero weight can carry a
                    # nonzero output contribution after f64 normalization.
                    if (weight == Float32(0.0)
                        or (weight == Float32(1.0) and score != row_max)):
                        needs_wide_replay = True
                        break
                    old_scale = Float32(1.0)
                    row_denom += weight
            else:
                row_initialized = True
                row_max = score
                row_denom = Float32(1.0)
            # Detect a potentially material accumulation loss. A positive
            # weight may disappear when added to a much larger Float32
            # denominator; enough such terms bias the attention output.
            # Replaying with a wide accumulator is gated by a conservative
            # row-length upper bound, preserving the ordinary fast path.
            var ideal_step = (
                previous_denominator * Float64(old_scale)
                + Float64(weight)
            )
            var one_step_error = math_abs(ideal_step - Float64(row_denom))
            if (one_step_error * Float64(limit)
                > Float64(row_denom) * Float64(1.0e-5)):
                needs_wide_replay = True
                break
            var value_base = vlayout.offset + kj * vlayout.strides.get(0)
            for d in range(vdepth):
                var value = vdata[value_base + d * vlayout.strides.get(1)]
                if not Bool(fp32_isfinite(value)):
                    return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
                var index = qi * vdepth + d
                var previous_value = dest[index]
                var scaled_old = previous_value * old_scale
                var weighted_value = weight * value
                var next_value = scaled_old + weighted_value
                # Even when scores and the denominator are well behaved,
                # a nonzero V contribution may be rounded away in the FP32
                # NUMERATOR. A later cancelling contribution can make the
                # discarded term the entire correct final result. Detect
                # both a lost new contribution and a lost scaled history,
                # then replay this row with f64 scalar accumulators. The
                # output Tensor still holds Float32 and retains its shape.
                # Opposite-sign contributions can nearly cancel without
                # making either term disappear outright. The two rounded
                # products and their FP32 sum can then have a large
                # *relative* error even though every value stays finite.
                # For example, scores [0,1e-6] and V=[1e8,-1e8]
                # yield about -48 on this fast recurrence instead of -50.
                # Use the existing accurate replay only when the remaining
                # numerator is below 1% of the larger contribution. This
                # is a cheap, per-element cancellation guard, not an
                # unconditional Float64 path for normal workloads.
                var opposite_signs = (
                    (scaled_old > Float32(0.0)
                     and weighted_value < Float32(0.0))
                    or (scaled_old < Float32(0.0)
                        and weighted_value > Float32(0.0))
                )
                var cancellation_risk = False
                if opposite_signs:
                    var larger_magnitude = max(
                        math_abs(scaled_old), math_abs(weighted_value)
                    )
                    cancellation_risk = (
                        math_abs(next_value)
                        <= larger_magnitude * Float32(0.01)
                    )
                if (cancellation_risk
                    or (weighted_value != Float32(0.0)
                        and next_value == scaled_old)
                    or (scaled_old != Float32(0.0)
                        and next_value == weighted_value)
                    or (old_scale != Float32(0.0)
                        and previous_value != Float32(0.0)
                        and scaled_old == Float32(0.0))
                    or (weight != Float32(0.0)
                        and value != Float32(0.0)
                        and weighted_value == Float32(0.0))
                    or not Bool(fp32_isfinite(next_value))):
                    needs_wide_replay = True
                    break
                dest[index] = next_value
        if row_initialized and not needs_wide_replay:
            var reciprocal = Float32(1.0) / row_denom
            for d in range(vdepth):
                var normalized = dest[qi * vdepth + d] * reciprocal
                if not Bool(fp32_isfinite(normalized)):
                    needs_wide_replay = True
                    break
                dest[qi * vdepth + d] = normalized
        if needs_wide_replay:
            # A weighted numerator may overflow even when all scores and
            # values are finite and the final weighted mean is representable.
            # One Float64 accumulator per output column prevents overflow,
            # while output elements stay Float32. The scratch is allocated
            # ONLY for exceptional rows, never on the normal hot path.
            var scratch_size = checked_attention_wide_scratch_nbytes(vdepth)
            if not scratch_size.is_ok():
                return TensorResult[Tensor[2, HostStorage]](scratch_size.status)
            var wide_values = List[Float64](length=vdepth, fill=Float64(0.0))
            # Neumaier compensation for each numerator. Float64 alone can
            # still swallow a small value between opposite-sign 1e20 terms.
            # Allocate this scratch ONLY on the exceptional wide path.
            var wide_corrections = List[Float64](length=vdepth, fill=Float64(0.0))
            var wide_max = Float64(0.0)
            var wide_denom = Float64(0.0)
            var wide_initialized = False
            var wide_inv_scale = Float64(1.0) / math_sqrt(Float64(depth))
            # Softmax is invariant to subtracting the SAME anchor logit
            # from every key. Computing q dot (k_j - k_0) in Float64
            # BEFORE summation is essential when logits share a huge
            # offset: even binary64 summing 1e20 + 2 would erase +2.
            # Subtracting in binary64 also preserves FP32 key differences.
            var reference_key_base = klayout.offset
            for kj in range(limit):
                var key_base = klayout.offset + kj * klayout.strides.get(0)
                var wide_score = Float64(0.0)
                var wide_score_correction = Float64(0.0)
                for d in range(depth):
                    var qv = qdata[query_base + d * qlayout.strides.get(1)]
                    var kv = kdata[key_base + d * klayout.strides.get(1)]
                    var reference_kv = kdata[
                        reference_key_base + d * klayout.strides.get(1)
                    ]
                    if (not Bool(fp32_isfinite(qv))
                        or not Bool(fp32_isfinite(kv))
                        or not Bool(fp32_isfinite(reference_kv))):
                        return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
                    var term = Float64(qv) * (
                        Float64(kv) - Float64(reference_kv)
                    )
                    var next_score = wide_score + term
                    if math_abs(wide_score) >= math_abs(term):
                        wide_score_correction += (wide_score - next_score) + term
                    else:
                        wide_score_correction += (term - next_score) + wide_score
                    wide_score = next_score
                # Compensation is added after the cancellation sequence,
                # rather than being immediately swallowed by a large partial
                # sum. The reference-key subtraction remains in Float64.
                wide_score = (wide_score + wide_score_correction) * wide_inv_scale
                # Normal-sized inputs cannot overflow Float64 here; reject
                # pathological evidence rather than emitting an invalid tensor.
                if not Bool(fp32_isfinite(wide_score)):
                    return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
                var weight = Float64(1.0)
                var old_scale = Float64(0.0)
                var weight_remainder = Float64(0.0)
                var old_scale_remainder = Float64(0.0)
                if wide_initialized:
                    if wide_score > wide_max:
                        var delta = wide_max - wide_score
                        old_scale = math_exp(delta)
                        # exp(delta) may round to 1.0 in *Float64* for
                        # deltas such as -1e-20. Preserve expm1(delta) in
                        # a low component for cancellation-sensitive V.
                        old_scale_remainder = (
                            math_expm1(delta) - (old_scale - Float64(1.0))
                        )
                        wide_denom = wide_denom * old_scale + Float64(1.0)
                        wide_max = wide_score
                    else:
                        var delta = wide_score - wide_max
                        weight = math_exp(delta)
                        weight_remainder = (
                            math_expm1(delta) - (weight - Float64(1.0))
                        )
                        old_scale = Float64(1.0)
                        wide_denom += weight
                else:
                    wide_initialized = True
                    wide_max = wide_score
                    wide_denom = Float64(1.0)
                var value_base = vlayout.offset + kj * vlayout.strides.get(0)
                for d in range(vdepth):
                    var value = vdata[value_base + d * vlayout.strides.get(1)]
                    if not Bool(fp32_isfinite(value)):
                        return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
                    var scaled_history = wide_values[d] * old_scale
                    var weighted_value = weight * Float64(value)
                    var next_sum = scaled_history + weighted_value
                    var correction = (
                        wide_corrections[d] * (old_scale + old_scale_remainder)
                        + wide_values[d] * old_scale_remainder
                        + weight_remainder * Float64(value)
                    )
                    if math_abs(scaled_history) >= math_abs(weighted_value):
                        correction += (scaled_history - next_sum) + weighted_value
                    else:
                        correction += (weighted_value - next_sum) + scaled_history
                    wide_values[d] = next_sum
                    wide_corrections[d] = correction
            if not wide_initialized or wide_denom <= Float64(0.0):
                return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
            for d in range(vdepth):
                var numerator = wide_values[d] + wide_corrections[d]
                var normalized = Float32(numerator / wide_denom)
                if not Bool(fp32_isfinite(normalized)):
                    return TensorResult[Tensor[2, HostStorage]](TensorStatus.UNSUPPORTED)
                dest[qi * vdepth + d] = normalized
    return TensorResult[Tensor[2, HostStorage]](output^)
