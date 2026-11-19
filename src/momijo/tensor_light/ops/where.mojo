# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / CONDITIONAL SELECTION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Conditional Selection
# Module:        momijo.tensor_light.ops.where
# Source:        src/momijo/tensor_light/ops/where.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines conditional selection and mask-guided tensor computation.
#
#   Public declarations include `where`, `where_nd`, `masked_fill`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for Boolean mask semantics and tensor shape
#     compatibility.
#   - Provide source-declared interfaces such as `where`, `where_nd`,
#     `masked_fill`.
#   - Select tensor elements using compatible conditions and broadcasted
#     shapes.
#   - Apply the declared N-D conditional selection interface.
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

# Select Float32 payloads with rank-preserving operand broadcasting.
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage, MutableTensorStorage
from ..storage.host_storage import HostStorage
from ..factory.creation import _complete_overwrite_output
from ..iterator.nd_iterator import NDIndexIterator
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorStatus, TensorResult
from .mask import TensorMask

def where[rank: Int, SA: TensorStorage, SB: TensorStorage](mask: TensorMask[rank], a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a.where(mask, b)
    var valid = mask.metadata_status()
    if valid != TensorStatus.OK:
        return TensorResult[Tensor[rank, HostStorage]](valid)
    # This operation is a CPU materialization. Both operands must satisfy
    # host-accessibility even if a particular mask happens to select one only.
    # Device selection/migration belongs to a separate explicit AOT lowering.
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_b.status)
    # The mask defines the result shape. Float32 operands may have singleton
    # axes, including the NumPy-compatible 1 -> 0 broadcast on empty axes.
    # Resolve all logical-to-physical strides ONCE before allocating output.
    var a_steps = checked_a.payload.value().strides.copy()
    var b_steps = checked_b.payload.value().strides.copy()
    comptime for axis in range(rank):
        var output_dim = mask.shape.dims.get(axis)
        var a_dim = checked_a.payload.value().shape.dims.get(axis)
        var b_dim = checked_b.payload.value().shape.dims.get(axis)
        if a_dim != output_dim and a_dim != 1:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        if b_dim != output_dim and b_dim != 1:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        if a_dim == 1:
            a_steps.set(axis, 0)
        if b_dim == 1:
            b_steps.set(axis, 0)
    var created = _complete_overwrite_output[rank](mask.shape)
    if not created.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](created.status)
    var out = created.payload.take()
    var la = checked_a.payload.value()
    var lb = checked_b.payload.value()
    var destination = out.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](mask.shape)
    var linear = 0
    while not cursor.done():
        var a_at = la.offset
        var b_at = lb.offset
        comptime for axis in range(rank):
            var coordinate = cursor.coordinates.get(axis)
            a_at += coordinate * a_steps.get(axis)
            b_at += coordinate * b_steps.get(axis)
        # Storage still checks its own physical read, but the shape/layout
        # proof does not need to be reconstructed for every selected value.
        var predicate = mask.bytes[linear]
        if predicate != 0 and predicate != 1:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_LAYOUT)
        if predicate == 1:
            var left = a.impl.storage.read_host(a_at)
            if not left.is_ok():
                return TensorResult[Tensor[rank, HostStorage]](left.status)
            destination[linear] = left.payload.value()
        else:
            var right = b.impl.storage.read_host(b_at)
            if not right.is_ok():
                return TensorResult[Tensor[rank, HostStorage]](right.status)
            destination[linear] = right.payload.value()
        linear += 1
        cursor.advance()
    return out._finish_complete_overwrite()


# Mixed-rank variant. The mask provides a static-rank destination, and both
# FP32 operands broadcast from the right as in binary_nd/compare_nd.
# The operation explicitly materializes host output and never migrates a GPU
# Tensor implicitly. No predicate Tensor dtype is introduced.
def where_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    mask: TensorMask[out_rank], a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    comptime assert ra >= 0 and ra <= out_rank and rb >= 0 and rb <= out_rank and out_rank <= 8, "invalid static broadcast ranks"
    comptime if ra == out_rank and rb == out_rank:
        return where(mask, a, b)
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a.where_nd[rb, out_rank](mask, b)
    var status = mask.metadata_status()
    if status != TensorStatus.OK:
        return TensorResult[Tensor[out_rank, HostStorage]](status)
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var a_valid = a._checked_current_layout()
    if not a_valid.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](a_valid.status)
    var b_valid = b._checked_current_layout()
    if not b_valid.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](b_valid.status)
    var a_steps = StaticRankArray[out_rank](0)
    var b_steps = StaticRankArray[out_rank](0)
    comptime for i in range(out_rank):
        var da = 1
        var db = 1
        comptime if i >= out_rank - ra:
            da = a_valid.payload.value().shape.dims.get(i - (out_rank - ra))
            if da != 1:
                a_steps.set(i, a_valid.payload.value().strides.get(i - (out_rank - ra)))
        comptime if i >= out_rank - rb:
            db = b_valid.payload.value().shape.dims.get(i - (out_rank - rb))
            if db != 1:
                b_steps.set(i, b_valid.payload.value().strides.get(i - (out_rank - rb)))
        var output_dim = mask.shape.dims.get(i)
        if (da != 1 and da != output_dim) or (db != 1 and db != output_dim):
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var allocated = _complete_overwrite_output[out_rank](mask.shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    var dst = out.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[out_rank](mask.shape)
    while not cursor.done():
        var a_at = a_valid.payload.value().offset
        var b_at = b_valid.payload.value().offset
        comptime for i in range(out_rank):
            var coordinate = cursor.coordinates.get(i)
            a_at += coordinate * a_steps.get(i)
            b_at += coordinate * b_steps.get(i)
        var predicate = mask.bytes[cursor.produced]
        if predicate != 0 and predicate != 1:
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.INVALID_LAYOUT)
        if predicate == 1:
            var left = a.impl.storage.read_host(a_at)
            if not left.is_ok():
                return TensorResult[Tensor[out_rank, HostStorage]](left.status)
            dst[cursor.produced] = left.payload.value()
        else:
            var right = b.impl.storage.read_host(b_at)
            if not right.is_ok():
                return TensorResult[Tensor[out_rank, HostStorage]](right.status)
            dst[cursor.produced] = right.payload.value()
        cursor.advance()
    return out._finish_complete_overwrite()

def masked_fill[rank: Int, S: MutableTensorStorage](mut dst: Tensor[rank, S], mask: TensorMask[rank], value: Float32) -> TensorStatus:
    comptime if identical(S, HostStorage):
        return dst.masked_fill(mask, value)
    var valid = mask.validated()
    if valid != TensorStatus.OK:
        return valid
    if not dst.impl.storage.is_host_accessible():
        return TensorStatus.BACKEND_UNAVAILABLE
    var checked_dst = dst._checked_current_layout()
    if not checked_dst.is_ok():
        return checked_dst.status
    if not mask.shape.same_extents(checked_dst.payload.value().shape):
        return TensorStatus.SIZE_MISMATCH
    if not checked_dst.payload.value().is_proven_unique():
        return TensorStatus.OVERLAPPING
    var safe = checked_dst.payload.value()
    var cursor = NDIndexIterator[rank](mask.shape)
    var linear = 0
    while not cursor.done():
        if mask.bytes[linear] != 0:
            var at = safe.physical_offset_unchecked(cursor.coordinates)
            var status = dst.impl.storage.write_host(at, value)
            if status != TensorStatus.OK:
                return status
        cursor.advance()
        linear += 1
    return TensorStatus.OK
