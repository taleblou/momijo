# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TEST / SMOKE CPU GEMM DEGENERATE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Smoke and Regression Tests
# Module:        tests.tensor_light.smoke_cpu_gemm_degenerate
# Source:        tests/tensor_light/smoke_cpu_gemm_degenerate.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines targeted tests for CPU GEMM degenerate in the FP32 Light
#   implementation.
#
#   The checks have a runnable entry point; this header does not assert that
#   execution or verification has occurred.
#
# Responsibilities:
#   - Exercise smoke CPU GEMM degenerate with source-defined fixtures and
#     checks.
#   - Assert expected behavior for matrix dimensional contracts, reductions,
#     and edge sizes.
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

# Fast C-order FP32 rank-2 shape degeneracies, with strict-mode output
# unchanged. The opt-in K-lane-reassociating GEMV is tested separately.
from std.sys.terminate import exit
from std.sys.info import simd_width_of
from momijo.tensor_light.internal.status import TensorStatus
from momijo.tensor_light.dtype.math_mode import Fp32MathMode
from momijo.tensor_light.dtype.fp32 import fp32_is_negative_zero
from momijo.tensor_light.kernels.cpu.matrix import cpu_mm_autoselect_fp32
from momijo.tensor_light.storage.host_storage import HostStorage


def run_case(m: Int, k: Int, n: Int, mode: Fp32MathMode) -> Bool:
    var a_make = HostStorage.full(m * k, Float32(2.0))
    var b_make = HostStorage.full(k * n, Float32(3.0))
    var c_make = HostStorage.full(m * n, Float32(-77.0))
    if not a_make.is_ok() or not b_make.is_ok() or not c_make.is_ok():
        return False
    var a = a_make.payload.take()
    var b = b_make.payload.take()
    var c = c_make.payload.take()
    for row in range(m):
        for depth in range(k):
            if a.write_host(row * k + depth, Float32(row + 1)) != TensorStatus.OK:
                return False
    for depth in range(k):
        for column in range(n):
            if b.write_host(depth * n + column, Float32(column + 1)) != TensorStatus.OK:
                return False
    var status = cpu_mm_autoselect_fp32[simd_width_of[DType.float32]()](
        c.borrow_mut_span(), a.borrow_span(), b.borrow_span(),
        m, k, n, mode,
    )
    if status != TensorStatus.OK:
        return False
    for row in range(m):
        for column in range(n):
            var cell = c.read_host(row * n + column)
            var expected = Float32(k * (row + 1) * (column + 1))
            if not cell.is_ok() or cell.payload.value() != expected:
                return False
    return True


def strict_k1_signed_zero() -> Bool:
    # A[-0] * B[+1] is -0, but the strict K-reduction performs 0 + (-0)
    # and must produce +0. A bare outer-product path would differ here.
    var a_make = HostStorage.full(1, Float32(-0.0))
    var b_make = HostStorage.full(1, Float32(1.0))
    var c_make = HostStorage.full(1, Float32(-77.0))
    if not a_make.is_ok() or not b_make.is_ok() or not c_make.is_ok():
        return False
    var a = a_make.payload.take()
    var b = b_make.payload.take()
    var c = c_make.payload.take()
    var status = cpu_mm_autoselect_fp32[simd_width_of[DType.float32]()](
        c.borrow_mut_span(), a.borrow_span(), b.borrow_span(),
        1, 1, 1, Fp32MathMode.STRICT_FP32,
    )
    var cell = c.read_host(0)
    return (status == TensorStatus.OK and cell.is_ok()
        and cell.payload.value() == Float32(0.0)
        and not fp32_is_negative_zero(cell.payload.value()))


def main():
    if not strict_k1_signed_zero():
        print("FAIL: strict K=1 GEMM signed-zero accumulation")
        exit(1)
    if not run_case(17, 1, 19, Fp32MathMode.STRICT_FP32):
        print("FAIL: rank-2 K=1 outer-product lowering")
        exit(1)
    if not run_case(1, 129, 17, Fp32MathMode.STRICT_FP32):
        print("FAIL: rank-2 M=1 vecmat lowering")
        exit(1)
    if not run_case(17, 65, 1, Fp32MathMode.STRICT_FP32):
        print("FAIL: rank-2 N=1 strict ordered lowering")
        exit(1)
    if not run_case(17, 65, 1, Fp32MathMode.FASTEST):
        print("FAIL: rank-2 N=1 opt-in GEMV")
        exit(1)
    print("PASS: FP32 degenerate GEMM paths and explicit reassociation gate")
