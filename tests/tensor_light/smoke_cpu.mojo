# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TEST / SMOKE CPU
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Smoke and Regression Tests
# Module:        tests.tensor_light.smoke_cpu
# Source:        tests/tensor_light/smoke_cpu.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines targeted tests for CPU in the FP32 Light implementation.
#
#   The checks have a runnable entry point; this header does not assert that
#   execution or verification has occurred.
#
# Responsibilities:
#   - Exercise smoke CPU with source-defined fixtures and checks.
#   - Assert expected behavior for shape, dtype, storage, and API
#     preconditions.
#   - Report failures through the existing test contract.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Check assertions against stated fixtures and explicit expected results.
#   - Exercise invalid, boundary, and regression inputs when applicable.
#
# Performance Objectives:
#   - Keep correctness oracles separate from timed kernel paths.
#   - Use small deterministic fixtures except in explicit performance tests.
#
# Maintainability:
#   - Keep regression cases aligned with the API being checked.
#   - Separate DECLARED / NOT_RUN cases from executed passing results.
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

from std.sys.terminate import exit
# Minimal smoke exercising ranks 0..8 on HostStorage. This test was not
# compiled in the source-review environment: run with Mojo 1.1 and MAX 26.6.
from momijo.tensor_light.shape.shape import TensorShape
from momijo.tensor_light.shape.shape_storage import StaticRankArray
from momijo.tensor_light.factory.creation import ones
from momijo.tensor_light.ops.reduction import sum_all
from momijo.tensor_light.internal.status import TensorStatus


def check_rank[rank: Int]() -> Bool:
    comptime assert rank >= 0 and rank <= 8
    var dims = StaticRankArray[rank](1)
    # Rank zero is a scalar; ranks 1..8 exercise a nontrivial first axis.
    comptime if rank > 0:
        dims.set(0, 2)
    var made_shape = TensorShape[rank].checked(dims)
    if not made_shape.is_ok():
        return False
    var made_tensor = ones[rank](made_shape.payload.value())
    if not made_tensor.is_ok():
        return False
    var value = made_tensor.payload.take()
    var total = sum_all(value)
    if not total.is_ok():
        return False
    comptime if rank == 0:
        return total.payload.value() == Float32(1.0)
    else:
        return total.payload.value() == Float32(2.0)


def check_empty_rank[rank: Int]() -> Bool:
    # A scalar has exactly one item; only positive ranks can have empty axes.
    comptime if rank > 0:
        var dims = StaticRankArray[rank](1)
        dims.set(0, 0)
        var made_shape = TensorShape[rank].checked(dims)
        if not made_shape.is_ok():
            return False
        var made_tensor = ones[rank](made_shape.payload.value())
        if not made_tensor.is_ok():
            return False
        var value = made_tensor.payload.take()
        var total = sum_all(value)
        if not total.is_ok():
            return False
        return total.payload.value() == Float32(0.0)
    return True


def check_rank2_write() -> Bool:
    var dims = StaticRankArray[2](0)
    dims.set(0, 2)
    dims.set(1, 3)
    var made_shape = TensorShape[2].checked(dims)
    if not made_shape.is_ok():
        return False
    var made_tensor = ones[2](made_shape.payload.value())
    if not made_tensor.is_ok():
        return False
    var value = made_tensor.payload.take()
    var indices = StaticRankArray[2](0)
    indices.set(0, 1)
    indices.set(1, 2)
    if value.set(indices, Float32(9.0)) != TensorStatus.OK:
        return False
    var cell = value.get(indices)
    if not cell.is_ok():
        return False
    if cell.payload.value() != Float32(9.0):
        return False
    var sum_result = sum_all(value)
    if not sum_result.is_ok():
        return False
    return sum_result.payload.value() == Float32(14.0)


def check_strict_fp32_reduction_order() -> Bool:
    # The ordered Float32 result is 1; lane-wise tree reassociation may
    # yield another result. This checks the default math policy, not speed.
    var dims = StaticRankArray[1](4)
    var shape = TensorShape[1].checked(dims)
    if not shape.is_ok():
        return False
    var vector_made = ones[1](shape.payload.value())
    var ones_made = ones[1](shape.payload.value())
    if not vector_made.is_ok() or not ones_made.is_ok():
        return False
    var vector = vector_made.payload.take()
    var rhs = ones_made.payload.take()
    var index = StaticRankArray[1](0)
    # A cancellation-sensitive FP32 sum: 2**24 + 1 - 2**24 + 1.
    for i in range(4):
        index.set(0, i)
        var current = Float32(1.0)
        if i == 0:
            current = Float32(16777216.0)
        elif i == 2:
            current = Float32(-16777216.0)
        if vector.set(index, current) != TensorStatus.OK:
            return False
    var total = vector.sum()
    var average = vector.mean()
    var dot = vector.dot(rhs)
    if not total.is_ok() or not average.is_ok() or not dot.is_ok():
        return False
    if (total.payload.value() != Float32(1.0)
        or average.payload.value() != Float32(0.25)
        or dot.payload.value() != Float32(1.0)):
        return False

    var matrix_dims = StaticRankArray[2](0)
    matrix_dims.set(0, 1)
    matrix_dims.set(1, 4)
    var matrix_shape = TensorShape[2].checked(matrix_dims)
    if not matrix_shape.is_ok():
        return False
    var matrix_made = ones[2](matrix_shape.payload.value())
    if not matrix_made.is_ok():
        return False
    var matrix = matrix_made.payload.take()
    var mindex = StaticRankArray[2](0)
    for i in range(4):
        index.set(0, i)
        mindex.set(1, i)
        var cell = vector.get(index)
        if not cell.is_ok():
            return False
        if matrix.set(mindex, cell.payload.value()) != TensorStatus.OK:
            return False
    var multiplied = matrix.gemv(rhs)
    if not multiplied.is_ok():
        return False
    var output = multiplied.payload.take()
    index.set(0, 0)
    var answer = output.get(index)
    if not answer.is_ok() or answer.payload.value() != Float32(1.0):
        return False
    # Both rank-reducing and keepdims SUM/MEAN must preserve K-axis order.
    var reduced_sum = matrix.sum_axis(1)
    var reduced_mean = matrix.mean_axis(1)
    var kept_sum = matrix.sum_axis_keepdims(1)
    if (not reduced_sum.is_ok() or not reduced_mean.is_ok()
        or not kept_sum.is_ok()):
        return False
    var sum_tensor = reduced_sum.payload.take()
    var mean_tensor = reduced_mean.payload.take()
    var kept_tensor = kept_sum.payload.take()
    var sum_value = sum_tensor.get(index)
    var mean_value = mean_tensor.get(index)
    mindex.set(1, 0)
    var kept_value = kept_tensor.get(mindex)
    if (not sum_value.is_ok() or not mean_value.is_ok()
        or not kept_value.is_ok()):
        return False
    return (sum_value.payload.value() == Float32(1.0)
        and mean_value.payload.value() == Float32(0.25)
        and kept_value.payload.value() == Float32(1.0))


def main():
    comptime for rank in range(9):
        if not check_rank[rank]():
            print("FAIL: rank", rank)
            exit(1)
            return
        if not check_empty_rank[rank]():
            print("FAIL: empty rank", rank)
            exit(1)
            return
    if not check_rank2_write():
        print("FAIL: rank-2 indexing / reduction")
        exit(1)
        return
    if not check_strict_fp32_reduction_order():
        print("FAIL: strict Float32 reduction ordering")
        exit(1)
        return
    print("PASS: CPU FP32 rank 0..8 and rank-2 set/get/sum")
