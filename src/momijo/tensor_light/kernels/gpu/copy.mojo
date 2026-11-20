# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / COPYING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.copy
# Source:        src/momijo/tensor_light/kernels/gpu/copy.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor copying for the GPU tensor kernel layer.
#
#   Public declarations include `gpu_copy_fp32`, `gpu_fill_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `gpu_copy_fp32`,
#     `gpu_fill_fp32`.
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

# FP32 GPU copy and fill. Overlap must be proven absent before launch.
from max.gpu import global_idx
from layout import TileTensor, TensorLayout as MaxTensorLayout

# The source and destination may have different static MAX layouts.
# Preconditions: each provides a checked flat contiguous range of
# count Float32 elements and the two physical ranges cannot overlap.
def gpu_copy_fp32[LO: MaxTensorLayout, LI: MaxTensorLayout](output: TileTensor[DType.float32, LO, MutAnyOrigin], inp: TileTensor[DType.float32, LI, MutAnyOrigin], count_abi: Int32) where (
    TileTensor[DType.float32, LO, MutAnyOrigin].flat_rank == 1
    and TileTensor[DType.float32, LI, MutAnyOrigin].flat_rank == 1
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        output[idx] = inp[idx]

def gpu_fill_fp32[L: MaxTensorLayout](output: TileTensor[DType.float32, L, MutAnyOrigin], value: Float32, count_abi: Int32) where (
    TileTensor[DType.float32, L, MutAnyOrigin].flat_rank == 1
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        output[idx] = value
