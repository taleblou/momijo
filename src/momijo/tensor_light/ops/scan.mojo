# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / PREFIX SCANS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Prefix Scans
# Module:        momijo.tensor_light.ops.scan
# Source:        src/momijo/tensor_light/ops/scan.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines inclusive and exclusive tensor prefix-scan operations.
#
#   Public declarations include `cumsum`, `cumprod`, `cummin`, `cummax`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for inclusive/exclusive prefix semantics
#     and axis handling.
#   - Provide source-declared interfaces such as `cumsum`, `cumprod`,
#     `cummin`, `cummax`, `exclusive_cumsum` and related symbols.
#   - Traverse scan axes with inclusive/exclusive prefix semantics.
#   - Implement exclusive cumulative sum/product variants.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Preserve inclusive versus exclusive semantics and scan-axis order.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#   - Avoid repeated shape/stride decoding inside scan loops.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
#   - Keep scan test cases for ranks, empty axes, and IEEE behavior.
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

# Host FP32 prefix scans, including arbitrary legal signed strides.
# rank-0 has no scan axis. Empty shapes yield an empty output.
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..factory.creation import _complete_overwrite_output
from ..iterator.nd_iterator import NDIndexIterator
from ..shape.axis import normalize_axis
from ..internal.status import TensorStatus, TensorResult
from ..dtype.fp32 import fp32_minimum, fp32_maximum


def _scan[operation: Int, rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if S == HostStorage:
        # The owning public Tensor type is the single source of HostStorage
        # scan semantics. Generic host-accessible storage uses fallback below.
        return input._scan_host[operation](axis)
    var trusted_input = input._checked_current_layout()
    if not trusted_input.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](trusted_input.status)
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 prefix scan opcode"
    var axis_result = normalize_axis[rank](axis)
    if not axis_result.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](axis_result.status)
    var a = axis_result.payload.value()
    var allocated = _complete_overwrite_output[rank](trusted_input.payload.value().shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    # The destination is a distinct, freshly allocated C-order owner. Once
    # the input layout and output allocation have been checked, reuse the
    # affine input address and output's row-major axis pitch. No unchecked
    # read of input.shape() and no repeated Tensor.get()/set() validation.
    var safe_layout = trusted_input.payload.value()
    var out_data = out.impl.storage.borrow_mut_span()
    var out_pitch = out.impl.layout.strides.get(a)
    var cursor = NDIndexIterator[rank](safe_layout.shape)
    while not cursor.done():
        var position = safe_layout.physical_offset_unchecked(cursor.coordinates)
        var next = Float32(0)
        comptime if operation >= 4:
            comptime if operation == 5:
                next = Float32(1)
            if cursor.coordinates.get(a) != 0:
                var current = input.impl.storage.read_host(
                    position - safe_layout.strides.get(a)
                )
                if not current.is_ok():
                    return TensorResult[Tensor[rank, HostStorage]](current.status)
                var previous = out_data[cursor.produced - out_pitch]
                comptime if operation == 4:
                    next = previous + current.payload.value()
                else:
                    next = previous * current.payload.value()
        else:
            var current = input.impl.storage.read_host(position)
            if not current.is_ok():
                return TensorResult[Tensor[rank, HostStorage]](current.status)
            next = current.payload.value()
            if cursor.coordinates.get(a) != 0:
                var previous = out_data[cursor.produced - out_pitch]
                comptime if operation == 0:
                    next = previous + next
                elif operation == 1:
                    next = previous * next
                elif operation == 2:
                    next = fp32_minimum(previous, next)
                else:
                    next = fp32_maximum(previous, next)
        out_data[cursor.produced] = next
        cursor.advance()
    return (out^)._finish_complete_overwrite()


def cumsum[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[0](input, axis)


def cumprod[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[1](input, axis)


def cummin[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[2](input, axis)


def cummax[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[3](input, axis)


def exclusive_cumsum[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[4](input, axis)


def exclusive_cumprod[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _scan[5](input, axis)
