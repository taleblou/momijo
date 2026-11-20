# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / CPU BACKEND
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.cpu
# Source:        src/momijo/tensor_light/backend/cpu.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines CPU backend in the backend availability layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `cpu_fill`, `cpu_add_owned_host_fp32`,
#   `cpu_sub_owned_host_fp32`, `cpu_mul_owned_host_fp32`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_fill`,
#     `cpu_add_owned_host_fp32`, `cpu_sub_owned_host_fp32`,
#     `cpu_mul_owned_host_fp32`, `cpu_div_owned_host_fp32` and related
#     symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Reject unsupported targets rather than silently falling back.
#   - Keep backend selection consistent with actual runtime capability.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid device initialization during pure capability queries.
#   - Preserve direct CPU paths when validated and applicable.
#
# Maintainability:
#   - Separate availability checks from native launch authorization.
#   - Make missing GPU/hybrid implementations explicit.
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

# CPU-only mapping of the shared ExecutionPlan, no separate graph runtime.

from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..core.tensor import Tensor
from ..storage.storage import MutableTensorStorage
from ..internal.status import TensorStatus
from ..storage.host_storage import HostStorage
from ..iterator.nd_iterator import NDIndexIterator
from ..kernels.cpu.elementwise import cpu_binary_contiguous_checked
from ..dtype.fp32 import fp32_maximum, fp32_minimum
from std.sys.info import simd_width_of

def cpu_fill[rank: Int, S: MutableTensorStorage](mut tensor: Tensor[rank, S], value: Float32, plan: ExecutionPlan) -> TensorStatus:
    var plan_proof = plan.checked_native_cpu_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    if checked_plan.placement != materialize[ExecutionPlacement.CPU_ONLY]():
        return TensorStatus.DEVICE_MISMATCH
    if checked_plan.total_items != tensor.numel():
        return TensorStatus.SIZE_MISMATCH
    if not tensor.impl.storage.is_host_accessible():
        return TensorStatus.BACKEND_UNAVAILABLE
    return tensor.fill(value)


# CPU_ONLY mapping for three distinct *owning* HostStorage tensors.
# Independent HostAllocation owners rule out source/output aliasing under the
# ownership contract, unlike arbitrary caller-provided Span values. The
# canonical ExecutionPlan is reused; no second runtime or JIT is introduced.
# Supports scalar, noncontiguous and negative-stride layouts (rank 0..8).


# One validated CPU elementwise lowering for independent owning allocations.
# Numeric operation: 0=add, 1=subtract, 2=multiply, 3=divide, 4=maximum, 5=minimum.
# The operation is a comptime parameter, so AOT emits one chosen SIMD/stride path.
def _cpu_binary_owned_host_fp32[rank: Int, operation: Int](
    left: Tensor[rank, HostStorage],
    right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage],
    plan: ExecutionPlan,
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 5, "Unsupported FP32 binary operation"
    var plan_proof = plan.checked_native_cpu_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    if checked_plan.placement != materialize[ExecutionPlacement.CPU_ONLY]() or checked_plan.output_device != 0:
        return TensorStatus.DEVICE_MISMATCH
    var la_result = left._checked_current_layout()
    if not la_result.is_ok():
        return la_result.status
    var lb_result = right._checked_current_layout()
    if not lb_result.is_ok():
        return lb_result.status
    var out_result = result._checked_current_layout()
    if not out_result.is_ok():
        return out_result.status
    var la = la_result.payload.value()
    var lb = lb_result.payload.value()
    var lo = out_result.payload.value()
    if checked_plan.total_items != la.shape.numel():
        return TensorStatus.SIZE_MISMATCH
    if not la.shape.same_extents(lb.shape) or not la.shape.same_extents(lo.shape):
        return TensorStatus.SIZE_MISMATCH
    if not lo.is_proven_unique():
        return TensorStatus.OVERLAPPING
    if la.shape.numel() == 0:
        return TensorStatus.OK
    # The owner contract and complete layout proofs allow unaliased SIMD.
    if la.is_c_contiguous() and lb.is_c_contiguous() and lo.is_c_contiguous():
        if la.is_exhaustive() and lb.is_exhaustive() and lo.is_exhaustive():
            var n = la.shape.numel()
            var source_a = left.impl.storage.borrow_span()
            var source_b = right.impl.storage.borrow_span()
            var destination = result.impl.storage.borrow_mut_span()
            var aa = source_a[la.span_lo:la.span_lo + n]
            var bb = source_b[lb.span_lo:lb.span_lo + n]
            var dd = destination[lo.span_lo:lo.span_lo + n]
            return cpu_binary_contiguous_checked[
                operation, simd_width_of[DType.float32]()
            ](dd, aa, bb)
    # Signed/gapped layouts use one initial physical-span proof and then
    # unchecked coordinates. Each logical output element is written once.
    var src_a = left.impl.storage.borrow_span()
    var src_b = right.impl.storage.borrow_span()
    var dst = result.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](la.shape)
    comptime if operation == 0:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = src_a[at_a] + src_b[at_b]
            cursor.advance()
    elif operation == 1:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = src_a[at_a] - src_b[at_b]
            cursor.advance()
    elif operation == 2:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = src_a[at_a] * src_b[at_b]
            cursor.advance()
    elif operation == 3:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = src_a[at_a] / src_b[at_b]
            cursor.advance()
    elif operation == 4:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = fp32_maximum(src_a[at_a], src_b[at_b])
            cursor.advance()
    else:
        while not cursor.done():
            var at_a = la.physical_offset_unchecked(cursor.coordinates)
            var at_b = lb.physical_offset_unchecked(cursor.coordinates)
            var at_dst = lo.physical_offset_unchecked(cursor.coordinates)
            dst[at_dst] = fp32_minimum(src_a[at_a], src_b[at_b])
            cursor.advance()
    return TensorStatus.OK


def cpu_add_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 0](left, right, result, plan)


def cpu_sub_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 1](left, right, result, plan)


def cpu_mul_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 2](left, right, result, plan)


def cpu_div_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 3](left, right, result, plan)


def cpu_maximum_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 4](left, right, result, plan)


def cpu_minimum_owned_host_fp32[rank: Int](
    left: Tensor[rank, HostStorage], right: Tensor[rank, HostStorage],
    mut result: Tensor[rank, HostStorage], plan: ExecutionPlan,
) -> TensorStatus:
    return _cpu_binary_owned_host_fp32[rank, 5](left, right, result, plan)
