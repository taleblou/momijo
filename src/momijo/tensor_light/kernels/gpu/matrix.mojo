# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / MATRIX OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.matrix
# Source:        src/momijo/tensor_light/kernels/gpu/matrix.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines matrix operations for the GPU tensor kernel layer.
#
#   Public declarations include `gpu_mm_strict_fp32`, `gpu_mm_simt_col4_fp32`,
#   `gpu_gemv_row_fp32`, `gpu_gemm_tf32_mma`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `gpu_mm_strict_fp32`,
#     `gpu_mm_simt_col4_fp32`, `gpu_gemv_row_fp32`, `gpu_gemm_tf32_mma`,
#     `gpu_mm_shared16_strict_fp32` and related symbols.
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

# Portable strict-FP32 SIMT reference GEMM. This does not use TF32 tensor cores;
# that requires a separately AOT-gated target-native matrix-engine lowering.

from max.gpu import global_idx
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from layout import Layout, LayoutTensor
from layout import TileTensor
from linalg.matmul import matmul as _gpu_linalg_matmul
from max.gpu.host import DeviceContext, DeviceEvent
from ...internal.status import TensorResult, TensorStatus
from ...device.capabilities import GpuCapabilities
from ...dtype.math_mode import Fp32MathMode
from ...dispatch.target_traits import NvidiaInstructionFamily, checked_nvidia_fp32_feature

def gpu_mm_strict_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m_abi: Int32, k_abi: Int32, n_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var m = Int(m_abi)
    var k = Int(k_abi)
    var n = Int(n_abi)
    # PRECONDITION: each independent layout is a flat contiguous row-major
    # view with capacities m*n, m*k and k*n respectively. This is NOT a
    # generic rank-2 strided TileTensor indexing kernel.
    var index = global_idx.x
    if index < m * n:
        var row = index // n
        var col = index % n
        var value = Float32(0.0)
        var a_row = row * k
        for p in range(k):
            value += a[a_row + p] * b[p * n + col]
        out[index] = value


# AOT candidate for wide row-major output: each lane computes four adjacent
# columns of ONE row, reusing each A scalar across four independent accumulators.
# This is true strict-FP32 SIMT arithmetic, not TF32/MMA and not a claim of
# measured superiority. Offsets and products are proven by the host launcher.
# It is safe for an incomplete last four-column tile and k == 0.
def gpu_mm_simt_col4_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m_abi: Int32, k_abi: Int32, n_abi: Int32, tiles_per_row_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var m = Int(m_abi)
    var k = Int(k_abi)
    var n = Int(n_abi)
    var tiles_per_row = Int(tiles_per_row_abi)
    var tile_id = global_idx.x
    if tile_id < m * tiles_per_row:
        var row = tile_id // tiles_per_row
        var col = (tile_id % tiles_per_row) * 4
        var acc0 = Float32(0.0)
        var acc1 = Float32(0.0)
        var acc2 = Float32(0.0)
        var acc3 = Float32(0.0)
        var a_row = row * k
        if col + 3 < n:
            # Full tile: no per-column bounds branches in the hot K loop.
            for p in range(k):
                var av = a[a_row + p]
                var bp = p * n + col
                acc0 += av * b[bp]
                acc1 += av * b[bp + 1]
                acc2 += av * b[bp + 2]
                acc3 += av * b[bp + 3]
        else:
            # The final partial tile never dereferences beyond its row.
            for p in range(k):
                var av = a[a_row + p]
                var bp = p * n + col
                acc0 += av * b[bp]
                if col + 1 < n:
                    acc1 += av * b[bp + 1]
                if col + 2 < n:
                    acc2 += av * b[bp + 2]
        var cp = row * n + col
        out[cp] = acc0
        if col + 1 < n:
            out[cp + 1] = acc1
        if col + 2 < n:
            out[cp + 2] = acc2
        if col + 3 < n:
            out[cp + 3] = acc3


# Baseline native SIMT GEMV: one independent FP32 output per row.
# The offline profile may select a tuned bandwidth/warp variant instead.
# All three TileTensor layouts are independently specialized, flat C-order.
# HOST PROOF: 0 <= rows*cols <= MAX/4, 0 <= row < rows, and disjoint
# destination storage; a and x may alias one another (both read only).
def gpu_gemv_row_fp32[LY: AnyType, LA: AnyType, LX: AnyType](
    out: TileTensor[DType.float32, LY, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    x: TileTensor[DType.float32, LX, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    var row = global_idx.x
    if row < rows:
        var base = row * cols
        var total = Float32(0.0)
        for col in range(cols):
            total += a[base + col] * x[col]
        out[row] = total


# Rank-2 Float32 Tensor-Core-capable AOT lowering candidate. The upstream
# linalg matmul GPU implementation provides the actual MMA/TF32 instruction
# selection (when target/shape support it), rather than emulating TF32 with
# ordinary FP32 scalar multiplication. This symbol is intentionally NOT
# inserted into the v1 DeviceExternalFunction GEMM ABI: that ABI binds only
# strict FP32 SIMT variant 14. A release must offline-compile this candidate,
# inspect its device instructions and bind a separate reviewed TF32 ABI before
# selecting it from Tensor/ExecutionPlan. Source availability alone is NOT a
# trusted executable or a guarantee that all shapes use Tensor Cores.
#
# All three TileTensors have independently proven, zero-copy, rank-2 layouts.
# The caller validates A[M,K], B[K,N], C[M,N] and disjoint output storage.
# TF32 is COMPUTE precision only; all input/output payloads remain Float32.


def gpu_gemm_tf32_mma[LC: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    ctx: DeviceContext, capabilities: GpuCapabilities,
    math_mode: Fp32MathMode,
    development_allow_runtime_compilation: Bool = False,
) -> TensorResult[DeviceEvent]:
    # The feature gate is mandatory even for an expert kernel candidate.
    # It is a SOURCE-level ISA policy: the native loader still authenticates
    # the device and selected precompiled image before enabling this adapter.
    var gate = checked_nvidia_fp32_feature(
        capabilities, NvidiaInstructionFamily.TF32_MMA, math_mode.code,
    )
    if not gate.is_ok():
        return TensorResult[DeviceEvent](gate.status)
    if not gate.payload.value():
        return TensorResult[DeviceEvent](TensorStatus.MATH_MODE_UNSUPPORTED)
    # This is an expert asynchronous HOST adapter, not an owning Tensor API.
    # Its caller MUST retain C/A/B and the DeviceContext until the returned
    # event has completed. In this v1 source release NO public Tensor method
    # supplies an authenticated TF32 artifact binding. This candidate is
    # never selected from an untrusted dispatch key.
    # The native adapter must gate NVIDIA SM80+/SM89 capability and explicit
    # TF32_ALLOWED/FASTEST math mode before requesting this specialization.
    # MAX linalg entry is source-backed; do not treat it as an authenticated
    # native TF32 artifact or enable implicit compilation in a release.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    try:
        _gpu_linalg_matmul[use_tf32=True, target="gpu"](out, a, b, ctx)
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    except:
        # A failed enqueue might already have submitted GPU work. Do not
        # report an ordinary failure and let owners be reclaimed unless the
        # stream is first known to be idle. Failed quiescence poisons the
        # caller's context/owners and requires explicit quarantine.
        try:
            ctx.stream().synchronize()
        except:
            return TensorResult[DeviceEvent](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[DeviceEvent](TensorStatus.NATIVE_LAUNCH_FAILED)


# AOT-only cooperative shared-memory SGEMM for the strict Float32 path.
# This is an independently compiled native kernel candidate. The release
# loader MUST bind its exact 256-thread, grid=(ceil(N/16),ceil(M/16)) ABI;
# do not enqueue it via the optional runtime-compilation development API.
# Every CTA uses 16x16 Float32 tiles. All 256 threads execute BOTH barriers
# for every K block, including CTA edge lanes; zero padding handles M/N/K
# tails and K=0 writes fully initialized zeros. No TF32/MMA is used.
def gpu_mm_shared16_strict_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m_abi: Int32, k_abi: Int32, n_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var m = Int(m_abi)
    var k = Int(k_abi)
    var n = Int(n_abi)
    # HOST PROOF: flat C-order physical capacity, disjoint output, valid
    # element-count arithmetic, and 256 threads per one-dimensional CTA.
    comptime tile_size = 16
    comptime tile_layout = Layout.row_major(tile_size, tile_size)
    var atile = LayoutTensor[
        DType.float32, tile_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var btile = LayoutTensor[
        DType.float32, tile_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var lane = Int(thread_idx.x)
    var local_row = lane // tile_size
    var local_col = lane % tile_size
    var row = Int(block_idx.y) * tile_size + local_row
    var col = Int(block_idx.x) * tile_size + local_col
    var accumulator = Float32(0.0)
    for base in range(0, k, tile_size):
        var av = Float32(0.0)
        var bv = Float32(0.0)
        if row < m and base + local_col < k:
            av = a[row * k + base + local_col]
        if base + local_row < k and col < n:
            bv = b[(base + local_row) * n + col]
        atile[local_row, local_col] = av
        btile[local_row, local_col] = bv
        barrier()
        # Full depth tiles stay compile-time unrolled. In the last partial
        # K tile, skip padded terms entirely: this avoids useless FP32 math
        # and preserves the logical reduction sequence (including signed zero).
        if base + tile_size <= k:
            comptime for inner in range(tile_size):
                accumulator += atile[local_row, inner] * btile[inner, local_col]
        else:
            for inner in range(k - base):
                accumulator += atile[local_row, inner] * btile[inner, local_col]
        barrier()
    if row < m and col < n:
        out[row * n + col] = accumulator


# AOT-only strict-FP32 register-tiled SGEMM for wide C-order matrices.
# Every thread owns FOUR adjacent output columns in the same row. A 256-
# thread CTA computes one 16x64 output tile, with A[16,16] and B[16,64]
# cooperatively cached in shared memory. No Tensor Core or TF32 conversion.
# All lanes participate in both barriers, including partial edge tiles.
# Native release ABI must independently verify SM89, flat FP32 layouts,
# source/destination extents and launch grid=(ceil(N/64),ceil(M/16)), CTA=256.
# This is an unbound SOURCE candidate, not an executable AOT release.
def gpu_mm_shared16_col4_strict_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m_abi: Int32, k_abi: Int32, n_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var m = Int(m_abi)
    var k = Int(k_abi)
    var n = Int(n_abi)
    comptime tile_k = 16
    comptime tile_m = 16
    comptime tile_n = 64
    var shared_a = LayoutTensor[
        DType.float32, Layout.row_major(tile_m, tile_k), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var shared_b = LayoutTensor[
        DType.float32, Layout.row_major(tile_k, tile_n), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var lane = Int(thread_idx.x)
    var local_row = lane // 16
    var local_group = lane % 16
    var row = Int(block_idx.y) * tile_m + local_row
    var col = Int(block_idx.x) * tile_n + local_group * 4
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    var acc2 = Float32(0.0)
    var acc3 = Float32(0.0)
    for base in range(0, k, tile_k):
        var left = Float32(0.0)
        if row < m and base + local_group < k:
            left = a[row * k + base + local_group]
        shared_a[local_row, local_group] = left
        # Each thread fills four consecutive shared-memory B columns.
        comptime for group_lane in range(4):
            var global_col = col + group_lane
            var value = Float32(0.0)
            if base + local_row < k and global_col < n:
                value = b[(base + local_row) * n + global_col]
            shared_b[local_row, local_group * 4 + group_lane] = value
        barrier()
        if base + tile_k <= k:
            comptime for depth in range(tile_k):
                var av = shared_a[local_row, depth]
                acc0 += av * shared_b[depth, local_group * 4]
                acc1 += av * shared_b[depth, local_group * 4 + 1]
                acc2 += av * shared_b[depth, local_group * 4 + 2]
                acc3 += av * shared_b[depth, local_group * 4 + 3]
        else:
            # Avoid padding arithmetic on the incomplete reduction tile.
            for depth in range(k - base):
                var av = shared_a[local_row, depth]
                acc0 += av * shared_b[depth, local_group * 4]
                acc1 += av * shared_b[depth, local_group * 4 + 1]
                acc2 += av * shared_b[depth, local_group * 4 + 2]
                acc3 += av * shared_b[depth, local_group * 4 + 3]
        barrier()
    if row < m:
        if col < n:
            out[row * n + col] = acc0
        if col + 1 < n:
            out[row * n + col + 1] = acc1
        if col + 2 < n:
            out[row * n + col + 2] = acc2
        if col + 3 < n:
            out[row * n + col + 3] = acc3


# AOT-only cooperative GEMV: one full NVIDIA warp owns one row, with
# coalesced accesses and an FP32 lane-group sum. The tree reassociates
# addition, so ONLY explicit FAST_TREE policy may select this variant.
# No branch makes some lanes skip a warp collective (including final row).
def gpu_gemv_warp32_fast_fp32[LY: AnyType, LA: AnyType, LX: AnyType](
    out: TileTensor[DType.float32, LY, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    x: TileTensor[DType.float32, LX, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    # HOST PROOF: flat C-order, launch thread count divisible by 32, and
    # full warp participation for all rows (including inactive last warp).
    var index = global_idx.x
    var row = index // 32
    var lane = index % 32
    var partial = Float32(0.0)
    if row < rows:
        for col in range(lane, cols, 32):
            partial += a[row * cols + col] * x[col]
    var result = warp.sum(partial)
    if row < rows and lane == 0:
        out[row] = result
