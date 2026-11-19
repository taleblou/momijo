# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / INDEXING OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.indexing
# Source:        src/momijo/tensor_light/kernels/gpu/indexing.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines indexing operations for the GPU tensor kernel layer.
#
#   Public declarations include `gpu_gather_fp32_unchecked`,
#   `gpu_scatter_unique_fp32_unchecked`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `gpu_gather_fp32_unchecked`,
#     `gpu_scatter_unique_fp32_unchecked`.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Respect the validated layout, bounds, and dtype contract.
#   - Protect scalar remainder and noncontiguous cases.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Use vectorized/parallel paths only when preconditions hold.
#   - Avoid repeated dispatch decisions in per-element loops.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Separate launch/ABI declarations from verified execution.
#   - Benchmark against stable, correct reference paths.
#   - Keep device-specific capability statements conservative.
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

# Indexed reads are race free. Scatter with potentially duplicate destinations
# is deliberately not exposed as an unchecked kernel.
from max.gpu import global_idx
from layout import TileTensor

def gpu_gather_fp32_unchecked[LO: AnyType, LI: AnyType, IL: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], inp: TileTensor[DType.float32, LI, MutAnyOrigin], positions: TileTensor[DType.int32, IL, MutAnyOrigin], count_abi: Int32, source_count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var source_count = Int(source_count_abi)
    var idx = global_idx.x
    if idx < count:
        var src = Int(positions[idx])
        # All device-side indices MUST be checked by the boundary planner.
        # A malformed source index is outside this unchecked kernel contract.
        out[idx] = inp[src]

# Race-free scatter fast path for a PROVEN one-to-one index mapping. This
# deliberately does not implement duplicate-index/last-writer semantics.
# The checked boundary must certify: (1) every position is in [0, dest_count),
# (2) positions are unique, (3) sources and destination are disjoint and
# (4) all arguments are live for the full asynchronous launch.
def gpu_scatter_unique_fp32_unchecked[LO: AnyType, LI: AnyType, IL: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    positions: TileTensor[DType.int32, IL, MutAnyOrigin],
    count_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[Int(positions[idx])] = inp[idx]
