# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / COMPARISONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Comparisons
# Module:        momijo.tensor_light.ops.comparison
# Source:        src/momijo/tensor_light/ops/comparison.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines elementwise comparisons and the corresponding mask or Boolean
#   result conventions.
#
#   Public declarations include `compare`, `eq`, `ne`, `lt`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for Boolean mask semantics and tensor shape
#     compatibility.
#   - Provide source-declared interfaces such as `compare`, `eq`, `ne`, `lt`,
#     `le` and related symbols.
#   - Produce comparisons according to the declared mask and dtype policy.
#   - Classify NaN, infinity, and finite values through dedicated operators.
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

# FP32 relational predicates return a compact auxiliary byte-mask object.
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..iterator.nd_iterator import NDIndexIterator
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from .mask import TensorMask
from std.utils.numerics import isnan as fp32_isnan, isinf as fp32_isinf, isfinite as fp32_isfinite

# AOT-constant predicate. Runtime opcode dispatch happens ONCE at the
# checked public boundary, never for each output element. Comparisons retain
# IEEE-754 NaN behavior and produce one UInt8 mask byte per logical element.
@inline(.always)
def _compare_fp32_fixed[relation: Int](lhs: Float32, rhs: Float32) -> Bool:
    comptime assert relation >= 0 and relation <= 5, "invalid predicate relation"
    comptime if relation == 0:
        return lhs == rhs
    elif relation == 1:
        return lhs != rhs
    elif relation == 2:
        return lhs < rhs
    elif relation == 3:
        return lhs <= rhs
    elif relation == 4:
        return lhs > rhs
    else:
        return lhs >= rhs


def _compare_fixed[relation: Int, rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a._compare_host[relation](b)
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[TensorMask[rank]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[TensorMask[rank]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[TensorMask[rank]](checked_b.status)
    var dims = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var left_dim = checked_a.payload.value().shape.dims.get(i)
        var right_dim = checked_b.payload.value().shape.dims.get(i)
        if left_dim == right_dim:
            dims.set(i, left_dim)
        elif left_dim == 1:
            dims.set(i, right_dim)
        elif right_dim == 1:
            dims.set(i, left_dim)
        else:
            return TensorResult[TensorMask[rank]](TensorStatus.SIZE_MISMATCH)
    var inferred = TensorShape[rank].checked(dims)
    if not inferred.is_ok():
        return TensorResult[TensorMask[rank]](inferred.status)
    var constructed = TensorMask[rank].zeros_checked(inferred.payload.value())
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var mask = constructed.payload.take()
    var a_steps = checked_a.payload.value().strides.copy()
    var b_steps = checked_b.payload.value().strides.copy()
    comptime for axis in range(rank):
        if checked_a.payload.value().shape.dims.get(axis) == 1:
            a_steps.set(axis, 0)
        if checked_b.payload.value().shape.dims.get(axis) == 1:
            b_steps.set(axis, 0)
    var cursor = NDIndexIterator[rank](inferred.payload.value())
    var linear = 0
    while not cursor.done():
        var a_at = checked_a.payload.value().offset
        var b_at = checked_b.payload.value().offset
        comptime for axis in range(rank):
            var coordinate = cursor.coordinates.get(axis)
            a_at += coordinate * a_steps.get(axis)
            b_at += coordinate * b_steps.get(axis)
        # The layouts have already been validated. Do not rebuild their
        # metadata proof for every generic host-visible element access.
        var x = a.impl.storage.read_host(a_at)
        var y = b.impl.storage.read_host(b_at)
        if not x.is_ok():
            return TensorResult[TensorMask[rank]](x.status)
        if not y.is_ok():
            return TensorResult[TensorMask[rank]](y.status)
        var left = x.payload.value()
        var right = y.payload.value()
        var condition = _compare_fp32_fixed[relation](left, right)
        # zeros_checked allocated exactly one byte per logical position;
        # linear is bounded by the validated result shape. Avoid redundant
        # TensorStatus construction and bounds checks in the element loop.
        if condition:
            mask.bytes[linear] = 1
        else:
            mask.bytes[linear] = 0
        linear += 1
        cursor.advance()
    return TensorResult[TensorMask[rank]](mask)

# Public runtime relation API chooses among six linked AOT variants once.
def compare[rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[rank, SA], b: Tensor[rank, SB], relation: Int
) -> TensorResult[TensorMask[rank]]:
    if relation == 0:
        return _compare_fixed[0](a, b)
    if relation == 1:
        return _compare_fixed[1](a, b)
    if relation == 2:
        return _compare_fixed[2](a, b)
    if relation == 3:
        return _compare_fixed[3](a, b)
    if relation == 4:
        return _compare_fixed[4](a, b)
    if relation == 5:
        return _compare_fixed[5](a, b)
    return TensorResult[TensorMask[rank]](TensorStatus.UNSUPPORTED)


def eq[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[0](a, b)

def ne[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[1](a, b)

def lt[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[2](a, b)

def le[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[3](a, b)

def gt[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[4](a, b)

def ge[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[TensorMask[rank]]:
    return _compare_fixed[5](a, b)


# Unary FP32 predicates use a separate mask representation, never an FP32-as-bool Tensor.
# std.utils.numerics supplies IEEE-754 checks instead of algebraic approximations.
def _fp32_predicate[kind: Int, rank: Int, S: TensorStorage](a: Tensor[rank, S]) -> TensorResult[TensorMask[rank]]:
    comptime if identical(S, HostStorage):
        return a._predicate_host[kind]()
    comptime assert kind >= 0 and kind <= 2, "invalid FP32 predicate"
    if not a.impl.storage.is_host_accessible():
        return TensorResult[TensorMask[rank]](TensorStatus.BACKEND_UNAVAILABLE)
    var validated = a._checked_current_layout()
    if not validated.is_ok():
        return TensorResult[TensorMask[rank]](validated.status)
    var constructed = TensorMask[rank].zeros_checked(validated.payload.value().shape)
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var mask = constructed.payload.take()
    var safe_layout = validated.payload.value()
    var cursor = NDIndexIterator[rank](safe_layout.shape)
    var linear = 0
    while not cursor.done():
        var read = a.impl.storage.read_host(safe_layout.physical_offset_unchecked(cursor.coordinates))
        if not read.is_ok():
            return TensorResult[TensorMask[rank]](read.status)
        var x = read.payload.value()
        var predicate = False
        comptime if kind == 0:
            predicate = Bool(fp32_isnan(x))
        elif kind == 1:
            predicate = Bool(fp32_isinf(x))
        else:
            predicate = Bool(fp32_isfinite(x))
        # The mask came from zeros_checked and linear advances exactly once
        # per valid NDIndexIterator coordinate: no repeated boundary call.
        if predicate:
            mask.bytes[linear] = 1
        else:
            mask.bytes[linear] = 0
        linear += 1
        cursor.advance()
    return TensorResult[TensorMask[rank]](mask)


def isnan[rank: Int, S: TensorStorage](a: Tensor[rank, S]) -> TensorResult[TensorMask[rank]]:
    return _fp32_predicate[0](a)


def isinf[rank: Int, S: TensorStorage](a: Tensor[rank, S]) -> TensorResult[TensorMask[rank]]:
    return _fp32_predicate[1](a)


def isfinite[rank: Int, S: TensorStorage](a: Tensor[rank, S]) -> TensorResult[TensorMask[rank]]:
    return _fp32_predicate[2](a)


# Mixed-rank right-aligned comparisons use the same compile-time rank and
# broadcast convention as arithmetic.binary_nd. Predicates do not create
# an alternate Boolean Tensor dtype.
def _compare_nd_fixed[relation: Int, ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[TensorMask[out_rank]]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[TensorMask[out_rank]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime assert ra >= 0 and ra <= 8 and rb >= 0 and rb <= 8, "rank must be 0..8"
    comptime assert out_rank == max(ra, rb) and out_rank <= 8, "invalid broadcast output rank"
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a._compare_nd_host[relation, rb, out_rank](b)
    comptime if ra == rb and ra == out_rank:
        return _compare_fixed[relation](a, b)
    var valid_a = a._checked_current_layout()
    if not valid_a.is_ok():
        return TensorResult[TensorMask[out_rank]](valid_a.status)
    var valid_b = b._checked_current_layout()
    if not valid_b.is_ok():
        return TensorResult[TensorMask[out_rank]](valid_b.status)
    var dims = StaticRankArray[out_rank](1)
    comptime for i in range(out_rank):
        var da = 1
        var db = 1
        comptime if i >= out_rank - ra:
            da = valid_a.payload.value().shape.dims.get(i - (out_rank - ra))
        comptime if i >= out_rank - rb:
            db = valid_b.payload.value().shape.dims.get(i - (out_rank - rb))
        if da == db:
            dims.set(i, da)
        elif da == 1:
            dims.set(i, db)
        elif db == 1:
            dims.set(i, da)
        else:
            return TensorResult[TensorMask[out_rank]](TensorStatus.SIZE_MISMATCH)
    var target = TensorShape[out_rank].checked(dims)
    if not target.is_ok():
        return TensorResult[TensorMask[out_rank]](target.status)
    var a_steps = StaticRankArray[out_rank](0)
    var b_steps = StaticRankArray[out_rank](0)
    comptime for i in range(ra):
        comptime axis = out_rank - ra + i
        if valid_a.payload.value().shape.dims.get(i) != 1:
            a_steps.set(axis, valid_a.payload.value().strides.get(i))
    comptime for i in range(rb):
        comptime axis = out_rank - rb + i
        if valid_b.payload.value().shape.dims.get(i) != 1:
            b_steps.set(axis, valid_b.payload.value().strides.get(i))
    var constructed = TensorMask[out_rank].zeros_checked(target.payload.value())
    if not constructed.is_ok():
        return TensorResult[TensorMask[out_rank]](constructed.status)
    var mask = constructed.payload.take()
    # Mixed ranks are right-aligned at compile time. HostStorage inputs use
    # the same one-time layout proof as equal-rank comparison, including
    # broadcasting with zero strides and any legal negative input stride.
    var cursor = NDIndexIterator[out_rank](target.payload.value())
    var linear = 0
    while not cursor.done():
        var a_at = valid_a.payload.value().offset
        var b_at = valid_b.payload.value().offset
        comptime for axis in range(out_rank):
            var coordinate = cursor.coordinates.get(axis)
            a_at += coordinate * a_steps.get(axis)
            b_at += coordinate * b_steps.get(axis)
        var av = a.impl.storage.read_host(a_at)
        if not av.is_ok():
            return TensorResult[TensorMask[out_rank]](av.status)
        var bv = b.impl.storage.read_host(b_at)
        if not bv.is_ok():
            return TensorResult[TensorMask[out_rank]](bv.status)
        var x = av.payload.value()
        var y = bv.payload.value()
        # Newly allocated mask and verified output extent: directly write one
        # validated predicate byte, without checking its shape per element.
        var predicate = _compare_fp32_fixed[relation](x, y)
        if predicate:
            mask.bytes[linear] = 1
        else:
            mask.bytes[linear] = 0
        linear += 1
        cursor.advance()
    return TensorResult[TensorMask[out_rank]](mask)


# Mixed-rank entry point selects a fixed relation outside the N-D loop.
def compare_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB], relation: Int
) -> TensorResult[TensorMask[out_rank]]:
    if relation == 0:
        return _compare_nd_fixed[0](a, b)
    if relation == 1:
        return _compare_nd_fixed[1](a, b)
    if relation == 2:
        return _compare_nd_fixed[2](a, b)
    if relation == 3:
        return _compare_nd_fixed[3](a, b)
    if relation == 4:
        return _compare_nd_fixed[4](a, b)
    if relation == 5:
        return _compare_nd_fixed[5](a, b)
    return TensorResult[TensorMask[out_rank]](TensorStatus.UNSUPPORTED)
