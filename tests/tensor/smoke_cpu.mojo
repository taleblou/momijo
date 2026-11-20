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
# Subcomponent:  Smoke and Regression Tests
# Module:        tests.tensor.smoke_cpu
# Source:        tests/tensor/smoke_cpu.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines targeted tests for CPU in the full implementation.
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

# Shared CPU correctness categories: scalar/N-D, empty shapes, writes and
# ordered reductions. Generic additionally validates ranks above eight.
from std.sys.terminate import exit
from std.collections import List
from momijo.tensor.validation import TensorStatusCode
from momijo.tensor.shape.shape import RuntimeTensorShape
from momijo.tensor.factory.creation import ones_f32
from momijo.tensor.ops.reduction import sum_all_f32, mean_all_f32


def check_rank(rank: Int) -> Bool:
    var dims = List[Int]()
    for axis in range(rank):
        if axis == 0:
            dims.append(2)
        else:
            dims.append(1)
    var made_shape = RuntimeTensorShape.try_from_dims(dims)
    if not made_shape.is_ok():
        return False
    var made = ones_f32(made_shape.take_value())
    if not made.is_ok():
        return False
    var tensor = made.take_value()
    var total = sum_all_f32(tensor)
    if not total.is_ok():
        return False
    var expected = Float32(2)
    if rank == 0:
        expected = Float32(1)
    return total.take_value() == expected


def check_empty_rank(rank: Int) -> Bool:
    if rank == 0:
        return True
    var dims = List[Int]()
    for axis in range(rank):
        if axis == 0:
            dims.append(0)
        else:
            dims.append(1)
    var made_shape = RuntimeTensorShape.try_from_dims(dims)
    if not made_shape.is_ok():
        return False
    var made = ones_f32(made_shape.take_value())
    if not made.is_ok():
        return False
    var tensor = made.take_value()
    var total = sum_all_f32(tensor)
    if not total.is_ok():
        return False
    return total.take_value() == Float32(0)


def check_rank2_write() -> Bool:
    var dims = List[Int]()
    dims.append(2)
    dims.append(3)
    var shape = RuntimeTensorShape.try_from_dims(dims)
    if not shape.is_ok():
        return False
    var made = ones_f32(shape.take_value())
    if not made.is_ok():
        return False
    var a = made.take_value()
    var index = List[Int]()
    index.append(1)
    index.append(2)
    if a.set(index, Float32(9)) != TensorStatusCode.OK:
        return False
    var cell = a.get(index)
    if not cell.is_ok() or cell.take_value() != Float32(9):
        return False
    var total = sum_all_f32(a)
    if not total.is_ok():
        return False
    return total.take_value() == Float32(14)


def check_strict_fp32_order() -> Bool:
    var dims = List[Int]()
    dims.append(4)
    var shape = RuntimeTensorShape.try_from_dims(dims)
    if not shape.is_ok():
        return False
    var made = ones_f32(shape.take_value())
    if not made.is_ok():
        return False
    var vector = made.take_value()
    var at = List[Int]()
    at.append(0)
    for i in range(4):
        at[0] = i
        var current = Float32(1)
        if i == 0:
            current = Float32(16777216)
        elif i == 2:
            current = Float32(-16777216)
        if vector.set(at, current) != TensorStatusCode.OK:
            return False
    var total = sum_all_f32(vector)
    var mean = mean_all_f32(vector)
    if not total.is_ok() or not mean.is_ok():
        return False
    return total.take_value() == Float32(1) and mean.take_value() == Float32(0.25)


def main():
    for rank in range(13):
        if not check_rank(rank):
            print("FAIL: CPU rank", rank)
            exit(1)
        if not check_empty_rank(rank):
            print("FAIL: empty CPU rank", rank)
            exit(1)
    if not check_rank2_write():
        print("FAIL: CPU rank2 get/set/reduction")
        exit(1)
    if not check_strict_fp32_order():
        print("FAIL: strict ordered Float32 reduction")
        exit(1)
    print("PASS: CPU rank0..12, empty shapes, writes, strict reductions")
