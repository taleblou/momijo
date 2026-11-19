# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR PARALLEL / PARALLEL REDUCTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Parallel Work
# Module:        momijo.tensor_light.parallel.parallel_reduce
# Source:        src/momijo/tensor_light/parallel/parallel_reduce.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines parallel reductions in the CPU parallel work layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `pairwise_sum_fp32`, `reduce_partition_count`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Partition CPU work and validate independent ranges.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `pairwise_sum_fp32`,
#     `reduce_partition_count`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Partition work into valid, disjoint ranges.
#   - Preserve deterministic reductions when required by the API.
#
# Performance Objectives:
#   - Limit scheduling overhead for small workloads.
#   - Align partitions with profitable memory and vector boundaries.
#
# Maintainability:
#   - Keep work distribution separate from kernel semantics.
#   - Test irregular tails and empty partitions.
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

# Pairwise combine over explicit FP32 partial results, no hidden atomics.
from std.collections import Span
from ..internal.status import TensorResult, TensorStatus

def pairwise_sum_fp32[origin: Origin](values: Span[False, Float32, origin]) -> Float32:
    # Actual balanced pairwise FP32 reduction, not a left-to-right scalar
    # fold. Binary-carry partials cover contiguous, power-of-two-sized runs
    # with a deterministic tree and no heap allocation or nested function.
    # 64 levels suffice for any nonnegative signed 64-bit Int length.
    var partial = Array[Float32, 64](fill=Float32(0.0))
    var occupied = Array[Bool, 64](fill=False)
    for i in range(len(values)):
        var accumulator = values[i]
        var level = 0
        while level < 63 and occupied[level]:
            # The earlier run is on the left, preserving index order.
            accumulator = partial[level] + accumulator
            occupied[level] = False
            level += 1
        partial[level] = accumulator
        occupied[level] = True
    # Merge remaining runs from oldest/largest to newest/smallest. Seed
    # with the FP32 SUM identity +0.0, as the serial/CPU-DAG reductions do:
    # returning the raw first partial would return -0.0 for one (or several)
    # negative zeros, in conflict with the library-defined zero-start sum.
    var have_value = False
    var result = Float32(0.0)
    for j in range(64):
        var level = 63 - j
        if occupied[level]:
            if have_value:
                result += partial[level]
            else:
                result = Float32(0.0) + partial[level]
                have_value = True
    return result

def reduce_partition_count(items: Int, grain: Int) -> TensorResult[Int]:
    if items < 0 or grain <= 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    var q = items // grain
    if items % grain != 0:
        q += 1
    return TensorResult[Int](q)
