# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / MATRIX OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Matrix Operations
# Module:        momijo.tensor_light.ops.matrix
# Source:        src/momijo/tensor_light/ops/matrix.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines matrix multiplication, dot products, and supported vector/batch
#   variants.
#
#   Public declarations include `mm`, `dot`, `gemv`, `vecmat`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for matrix dimensional contracts,
#     reductions, and edge sizes.
#   - Provide source-declared interfaces such as `mm`, `dot`, `gemv`,
#     `vecmat`, `outer` and related symbols.
#   - Validate matrix dimensions and inner-product compatibility.
#   - Handle declared batched matrix multiplication and batch shapes.
#   - Cover N-D and mixed-layout matrix entry points when supported.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Handle degenerate matrix dimensions and accumulation behavior
#     explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#   - Keep tiled/contiguous optimizations behind verified layout
#     preconditions.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
#   - Test tails, transposes, zero-K, and nonuniform operands.
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

# Correct rank-2 reference mapping. Not the tuned AVX2/TF32 GEMM fast path.
from ..core.tensor import Tensor
from ..storage.host_storage import HostStorage
from ..kernels.cpu.matrix import cpu_mm_contiguous
from std.sys.info import simd_width_of
from ..storage.storage import TensorStorage
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..iterator.nd_iterator import NDIndexIterator
from ..factory.creation import zeros, _complete_overwrite_output
from ..internal.status import TensorStatus, TensorResult

def mm[SA: TensorStorage, SB: TensorStorage](a: Tensor[2, SA], b: Tensor[2, SB]) -> TensorResult[Tensor[2, HostStorage]]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[2, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if SA == HostStorage and SB == HostStorage:
        return a.mm(rebind[Tensor[2, HostStorage]](b))
    var a_valid = a._checked_current_layout()
    if not a_valid.is_ok():
        return TensorResult[Tensor[2, HostStorage]](a_valid.status)
    var b_valid = b._checked_current_layout()
    if not b_valid.is_ok():
        return TensorResult[Tensor[2, HostStorage]](b_valid.status)
    var m = a_valid.payload.value().shape.dims.get(0)
    var k = a_valid.payload.value().shape.dims.get(1)
    if k != b_valid.payload.value().shape.dims.get(0):
        return TensorResult[Tensor[2, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var n = b_valid.payload.value().shape.dims.get(1)
    var dims = StaticRankArray[2](0)
    dims.set(0, m)
    dims.set(1, n)
    var shape = TensorShape[2].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[2, HostStorage]](shape.status)
    var built = _complete_overwrite_output[2](shape.payload.value())
    if not built.is_ok():
        return TensorResult[Tensor[2, HostStorage]](built.status)
    var out = built.payload.take()
    # Slow but general host-readable storage: once affine layouts have been
    # checked, calculate each address directly. Repeated Tensor.get/set
    # shape/result construction in every K iteration was unnecessary.
    var left_layout = a_valid.payload.value()
    var right_layout = b_valid.payload.value()
    var output_span = out.impl.storage.borrow_mut_span()
    var left_row_stride = left_layout.strides.get(0)
    var left_depth_stride = left_layout.strides.get(1)
    var right_depth_stride = right_layout.strides.get(0)
    var right_col_stride = right_layout.strides.get(1)
    for i in range(m):
        var left_row = left_layout.offset + i * left_row_stride
        for j in range(n):
            var right_col = right_layout.offset + j * right_col_stride
            var acc = Float32(0.0)
            for p in range(k):
                var lhs = a.impl.storage.read_host(left_row + p * left_depth_stride)
                if not lhs.is_ok():
                    return TensorResult[Tensor[2, HostStorage]](lhs.status)
                var rhs = b.impl.storage.read_host(right_col + p * right_depth_stride)
                if not rhs.is_ok():
                    return TensorResult[Tensor[2, HostStorage]](rhs.status)
                acc += lhs.payload.value() * rhs.payload.value()
            output_span[i * n + j] = acc
    return (out^)._finish_complete_overwrite()


# A rank-1 dot produces a Float32 scalar. It never creates an integer Tensor.
def dot[SA: TensorStorage, SB: TensorStorage](a: Tensor[1, SA], b: Tensor[1, SB]) -> TensorResult[Float32]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    # For the canonical owning CPU tensor, forward to the public Tensor
    # method so there is only one checked SIMD/strided implementation.
    comptime if SA == HostStorage and SB == HostStorage:
        return a.dot(rebind[Tensor[1, HostStorage]](b))
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Float32](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Float32](checked_b.status)
    if checked_a.payload.value().shape.numel() != checked_b.payload.value().shape.numel():
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    var la = checked_a.payload.value()
    var lb = checked_b.payload.value()
    var a_step = la.strides.get(0)
    var b_step = lb.strides.get(0)
    var result = Float32(0.0)
    for i in range(la.shape.numel()):
        var left = a.impl.storage.read_host(la.offset + i * a_step)
        if not left.is_ok():
            return TensorResult[Float32](left.status)
        var right = b.impl.storage.read_host(lb.offset + i * b_step)
        if not right.is_ok():
            return TensorResult[Float32](right.status)
        result += left.payload.value() * right.payload.value()
    return TensorResult[Float32](result)

# Matrix-vector multiplication, with no unnecessary B packing for the GEMV case.
def gemv[SA: TensorStorage, SB: TensorStorage](a: Tensor[2, SA], x: Tensor[1, SB]) -> TensorResult[Tensor[1, HostStorage]]:
    if not a.impl.storage.is_host_accessible() or not x.impl.storage.is_host_accessible():
        return TensorResult[Tensor[1, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if SA == HostStorage and SB == HostStorage:
        return a.gemv(rebind[Tensor[1, HostStorage]](x))
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[1, HostStorage]](checked_a.status)
    var checked_x = x._checked_current_layout()
    if not checked_x.is_ok():
        return TensorResult[Tensor[1, HostStorage]](checked_x.status)
    var m = checked_a.payload.value().shape.dims.get(0)
    var k = checked_a.payload.value().shape.dims.get(1)
    if checked_x.payload.value().shape.numel() != k:
        return TensorResult[Tensor[1, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var out_dims = StaticRankArray[1](0)
    out_dims.set(0, m)
    var out_shape = TensorShape[1].checked(out_dims)
    if not out_shape.is_ok():
        return TensorResult[Tensor[1, HostStorage]](out_shape.status)
    var created = _complete_overwrite_output[1](out_shape.payload.value())
    if not created.is_ok():
        return TensorResult[Tensor[1, HostStorage]](created.status)
    var result = created.payload.take()
    var matrix_layout = checked_a.payload.value()
    var vector_layout = checked_x.payload.value()
    var matrix_row_step = matrix_layout.strides.get(0)
    var matrix_column_step = matrix_layout.strides.get(1)
    var vector_step = vector_layout.strides.get(0)
    var dst = result.impl.storage.borrow_mut_span()
    for i in range(m):
        var acc = Float32(0.0)
        var row_offset = matrix_layout.offset + i * matrix_row_step
        for j in range(k):
            var av = a.impl.storage.read_host(row_offset + j * matrix_column_step)
            if not av.is_ok():
                return TensorResult[Tensor[1, HostStorage]](av.status)
            var xv = x.impl.storage.read_host(vector_layout.offset + j * vector_step)
            if not xv.is_ok():
                return TensorResult[Tensor[1, HostStorage]](xv.status)
            acc += av.payload.value() * xv.payload.value()
        dst[i] = acc
    return (result^)._finish_complete_overwrite()


# Vector @ matrix, complementary to GEMV (matrix @ vector). The rank-1
# result has N elements, and a zero-length shared K yields N zeros.
def vecmat[SA: TensorStorage, SB: TensorStorage](
    x: Tensor[1, SA], b: Tensor[2, SB]
) -> TensorResult[Tensor[1, HostStorage]]:
    if not x.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[1, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if SA == HostStorage and SB == HostStorage:
        return x.vecmat(rebind[Tensor[2, HostStorage]](b))
    var checked_x = x._checked_current_layout()
    if not checked_x.is_ok():
        return TensorResult[Tensor[1, HostStorage]](checked_x.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[1, HostStorage]](checked_b.status)
    var k = checked_b.payload.value().shape.dims.get(0)
    var n = checked_b.payload.value().shape.dims.get(1)
    if checked_x.payload.value().shape.numel() != k:
        return TensorResult[Tensor[1, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var output_dims = StaticRankArray[1](0)
    output_dims.set(0, n)
    var output_shape = TensorShape[1].checked(output_dims)
    if not output_shape.is_ok():
        return TensorResult[Tensor[1, HostStorage]](output_shape.status)
    var allocated = _complete_overwrite_output[1](output_shape.payload.value())
    if not allocated.is_ok():
        return TensorResult[Tensor[1, HostStorage]](allocated.status)
    var output = allocated.payload.take()
    # Non-HostStorage host-readable fallbacks do not reconstruct the same
    # shape and physical offsets at every scalar product. Both layouts were
    # fully verified before allocation; only storage.read_host remains checked.
    var xl = checked_x.payload.value()
    var bl = checked_b.payload.value()
    var dst = output.impl.storage.borrow_mut_span()
    var xs = xl.strides.get(0)
    var rows = bl.strides.get(0)
    var cols = bl.strides.get(1)
    for column in range(n):
        var accumulator = Float32(0.0)
        var column_base = bl.offset + column * cols
        for inner in range(k):
            var xv = x.impl.storage.read_host(xl.offset + inner * xs)
            if not xv.is_ok():
                return TensorResult[Tensor[1, HostStorage]](xv.status)
            var bv = b.impl.storage.read_host(column_base + inner * rows)
            if not bv.is_ok():
                return TensorResult[Tensor[1, HostStorage]](bv.status)
            accumulator += xv.payload.value() * bv.payload.value()
        dst[column] = accumulator
    return (output^)._finish_complete_overwrite()


# Outer product has shape (len(a), len(b)), including empty vectors.
def outer[SA: TensorStorage, SB: TensorStorage](a: Tensor[1, SA], b: Tensor[1, SB]) -> TensorResult[Tensor[2, HostStorage]]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[2, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if SA == HostStorage and SB == HostStorage:
        return a.outer(rebind[Tensor[1, HostStorage]](b))
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[2, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[2, HostStorage]](checked_b.status)
    # Derive the output only from the already owner-validated snapshots.
    # Do not reopen the mutable Tensor facade and consume a separate cached
    # count after the shape/provenance verification boundary.
    var left_count = checked_a.payload.value().shape.numel()
    var right_count = checked_b.payload.value().shape.numel()
    var dims = StaticRankArray[2](0)
    dims.set(0, left_count)
    dims.set(1, right_count)
    var out_shape = TensorShape[2].checked(dims)
    if not out_shape.is_ok():
        return TensorResult[Tensor[2, HostStorage]](out_shape.status)
    var created = _complete_overwrite_output[2](out_shape.payload.value())
    if not created.is_ok():
        return TensorResult[Tensor[2, HostStorage]](created.status)
    var result = created.payload.take()
    var al = checked_a.payload.value()
    var bl = checked_b.payload.value()
    var dst = result.impl.storage.borrow_mut_span()
    var a_count = al.shape.numel()
    var b_count = bl.shape.numel()
    var a_step = al.strides.get(0)
    var b_step = bl.strides.get(0)
    # HostStorage's optimized path is owned by Tensor.outer(); alternate
    # host-readable storage uses checked read_host without alias guesses.
    for i in range(a_count):
        var av = a.impl.storage.read_host(al.offset + i * a_step)
        if not av.is_ok():
            return TensorResult[Tensor[2, HostStorage]](av.status)
        var dst_row = i * b_count
        for j in range(b_count):
            var bv = b.impl.storage.read_host(bl.offset + j * b_step)
            if not bv.is_ok():
                return TensorResult[Tensor[2, HostStorage]](bv.status)
            dst[dst_row + j] = av.payload.value() * bv.payload.value()
    return (result^)._finish_complete_overwrite()


# Strict rank-3 batched matrix multiplication. One output per batch; no
# implicit batch broadcasting or hidden CPU fallback from device-only storage.
def bmm[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[3, SA], b: Tensor[3, SB]
) -> TensorResult[Tensor[3, HostStorage]]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[3, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if SA == HostStorage and SB == HostStorage:
        return a.bmm(rebind[Tensor[3, HostStorage]](b))
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[3, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[3, HostStorage]](checked_b.status)
    var batches = checked_a.payload.value().shape.dims.get(0)
    var m = checked_a.payload.value().shape.dims.get(1)
    var k = checked_a.payload.value().shape.dims.get(2)
    if checked_b.payload.value().shape.dims.get(0) != batches:
        return TensorResult[Tensor[3, HostStorage]](TensorStatus.SIZE_MISMATCH)
    if checked_b.payload.value().shape.dims.get(1) != k:
        return TensorResult[Tensor[3, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var n = checked_b.payload.value().shape.dims.get(2)
    var dims = StaticRankArray[3](0)
    dims.set(0, batches)
    dims.set(1, m)
    dims.set(2, n)
    var shape = TensorShape[3].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[3, HostStorage]](shape.status)
    var made = _complete_overwrite_output[3](shape.payload.value())
    if not made.is_ok():
        return TensorResult[Tensor[3, HostStorage]](made.status)
    var result = made.payload.take()
    if batches == 0:
        # Avoid constructing matrix-element products for an empty batch.
        # Its other extents can be huge even though zero bytes are accessed.
        return (result^)._finish_complete_overwrite()
    # Generic host-readable storage needs checked scalar loads, but the
    # affine shapes/strides and disjoint output were already validated once.
    var al = checked_a.payload.value()
    var bl = checked_b.payload.value()
    var dst = result.impl.storage.borrow_mut_span()
    var a_batch_stride = al.strides.get(0)
    var a_row_stride = al.strides.get(1)
    var a_k_stride = al.strides.get(2)
    var b_batch_stride = bl.strides.get(0)
    var b_k_stride = bl.strides.get(1)
    var b_col_stride = bl.strides.get(2)
    for batch in range(batches):
        var a_batch = al.offset + batch * a_batch_stride
        var b_batch = bl.offset + batch * b_batch_stride
        for row in range(m):
            var a_row = a_batch + row * a_row_stride
            for col in range(n):
                var b_col = b_batch + col * b_col_stride
                var accumulator = Float32(0.0)
                for inner in range(k):
                    var av = a.impl.storage.read_host(a_row + inner * a_k_stride)
                    if not av.is_ok():
                        return TensorResult[Tensor[3, HostStorage]](av.status)
                    var bv = b.impl.storage.read_host(b_col + inner * b_k_stride)
                    if not bv.is_ok():
                        return TensorResult[Tensor[3, HostStorage]](bv.status)
                    accumulator += av.payload.value() * bv.payload.value()
                dst[(batch * m + row) * n + col] = accumulator
    return (result^)._finish_complete_overwrite()


# Rank-specialized matmul overloads preserve the canonical Tensor API.
def matmul[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[1, SA], b: Tensor[1, SB]
) -> TensorResult[Float32]:
    return dot(a, b)


def matmul[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[2, SA], b: Tensor[1, SB]
) -> TensorResult[Tensor[1, HostStorage]]:
    return gemv(a, b)


def matmul[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[1, SA], b: Tensor[2, SB]
) -> TensorResult[Tensor[1, HostStorage]]:
    return vecmat(a, b)


def matmul[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[2, SA], b: Tensor[2, SB]
) -> TensorResult[Tensor[2, HostStorage]]:
    return mm(a, b)


def matmul[SA: TensorStorage, SB: TensorStorage](
    a: Tensor[3, SA], b: Tensor[3, SB]
) -> TensorResult[Tensor[3, HostStorage]]:
    # Unlike strict bmm, general matmul broadcasts singleton batch axes.
    return matmul_nd[3](a, b)


# Rank-static batched matrix multiplication for ranks 2..8. The final two
# axes are (M,K) and (K,N); leading axes are right-aligned, singleton-broadcast
# batch axes. Both operands have the same rank in this overload. Distinct-rank
# batching should use a separate compile-time output-rank specialization.
# Host-only reference lowering; target-optimized AOT GEMM is selected elsewhere.
def matmul_nd[rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[rank, SA], b: Tensor[rank, SB]
) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime assert rank >= 2 and rank <= 8, "batched matmul rank must be 2..8"
    comptime if SA == HostStorage and SB == HostStorage:
        return a.matmul_nd(rebind[Tensor[rank, HostStorage]](b))
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_b.status)
    comptime if rank == 2:
        # The existing SIMD GEMM already handles the non-batched case.
        return rebind_var[TensorResult[Tensor[rank, HostStorage]]](
            mm(rebind[Tensor[2, SA]](a), rebind[Tensor[2, SB]](b))
        )
    var m = checked_a.payload.value().shape.dims.get(rank - 2)
    var k = checked_a.payload.value().shape.dims.get(rank - 1)
    if k != checked_b.payload.value().shape.dims.get(rank - 2):
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var n = checked_b.payload.value().shape.dims.get(rank - 1)
    var dims = StaticRankArray[rank](0)
    comptime for axis in range(rank - 2):
        var da = checked_a.payload.value().shape.dims.get(axis)
        var db = checked_b.payload.value().shape.dims.get(axis)
        if da == db:
            dims.set(axis, da)
        elif da == 1:
            dims.set(axis, db)
        elif db == 1:
            dims.set(axis, da)
        else:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    dims.set(rank - 2, m)
    dims.set(rank - 1, n)
    var shape = TensorShape[rank].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shape.status)
    # K=0 produces the additive identity in EVERY nonempty output element.
    # The complete-overwrite allocation cannot be returned uninitialized.
    if k == 0:
        return zeros[rank](shape.payload.value())
    var made = _complete_overwrite_output[rank](shape.payload.value())
    if not made.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](made.status)
    var out = made.payload.take()
    if out.numel() == 0:
        return (out^)._finish_complete_overwrite()
    # AOT host-owner lowering: both affine mappings were checked once above.
    # Do not allocate TensorResult or repeat storage bounds checks per FMA.
    # NDIndexIterator walks the new C-order output in strictly linear order.
    # Other host-accessible Storage implementations retain checked scalar
    # accesses because they do not promise the HostStorage span capability.
    var output_data = out.impl.storage.borrow_mut_span()
    var source_a = checked_a.payload.value()
    var source_b = checked_b.payload.value()
    var step_a = source_a.strides.get(rank - 1)
    var step_b = source_b.strides.get(rank - 2)
    var cursor = NDIndexIterator[rank](shape.payload.value())
    while not cursor.done():
        var ai = cursor.coordinates.copy()
        var bi = cursor.coordinates.copy()
        comptime for axis in range(rank - 2):
            if source_a.shape.dims.get(axis) == 1:
                ai.set(axis, 0)
            if source_b.shape.dims.get(axis) == 1:
                bi.set(axis, 0)
        ai.set(rank - 1, 0)
        bi.set(rank - 2, 0)
        var base_a = source_a.physical_offset_unchecked(ai)
        var base_b = source_b.physical_offset_unchecked(bi)
        var acc = Float32(0.0)
        for inner in range(k):
            var av = a.impl.storage.read_host(base_a + inner * step_a)
            if not av.is_ok():
                return TensorResult[Tensor[rank, HostStorage]](av.status)
            var bv = b.impl.storage.read_host(base_b + inner * step_b)
            if not bv.is_ok():
                return TensorResult[Tensor[rank, HostStorage]](bv.status)
            acc += av.payload.value() * bv.payload.value()
        # All logical output elements are written once in strict C-order.
        output_data[cursor.produced] = acc
        cursor.advance()
    return (out^)._finish_complete_overwrite()


# Explicit mixed-rank batched matmul, including (2D, ND) and (ND, 2D).
# Rank is specialized offline; output batch axes are right-aligned.
# This is a correct host reference mapping, not a tuned GEMM lowering.
def matmul_mixed[rank_a: Int, rank_b: Int, out_rank: Int,
                 SA: TensorStorage, SB: TensorStorage](
    a: Tensor[rank_a, SA], b: Tensor[rank_b, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    comptime assert rank_a >= 2 and rank_a <= 8, "left matmul rank must be 2..8"
    comptime assert rank_b >= 2 and rank_b <= 8, "right matmul rank must be 2..8"
    comptime assert out_rank >= rank_a and out_rank >= rank_b and out_rank <= 8, "output rank cannot omit input batch axes"
    comptime assert out_rank == max(rank_a, rank_b), "mixed matmul output rank must equal the maximum input rank"
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    # One owning HostStorage implementation in Tensor, not a second
    # free-function GEMM path with independently changing semantics.
    comptime if SA == HostStorage and SB == HostStorage:
        return a.matmul_mixed[rank_b, out_rank](rebind[Tensor[rank_b, HostStorage]](b))
    var left_check = a._checked_current_layout()
    if not left_check.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](left_check.status)
    var right_check = b._checked_current_layout()
    if not right_check.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](right_check.status)
    var left_layout = left_check.payload.value()
    var right_layout = right_check.payload.value()
    var m = left_layout.shape.dims.get(rank_a - 2)
    var k = left_layout.shape.dims.get(rank_a - 1)
    if k != right_layout.shape.dims.get(rank_b - 2):
        return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var n = right_layout.shape.dims.get(rank_b - 1)
    var dims = StaticRankArray[out_rank](1)
    comptime for axis in range(out_rank - 2):
        var da = 1
        var db = 1
        comptime if axis >= out_rank - rank_a:
            da = left_layout.shape.dims.get(axis - (out_rank - rank_a))
        comptime if axis >= out_rank - rank_b:
            db = right_layout.shape.dims.get(axis - (out_rank - rank_b))
        if da == db:
            dims.set(axis, da)
        elif da == 1:
            dims.set(axis, db)
        elif db == 1:
            dims.set(axis, da)
        else:
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    dims.set(out_rank - 2, m)
    dims.set(out_rank - 1, n)
    var created_shape = TensorShape[out_rank].checked(dims)
    if not created_shape.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](created_shape.status)
    if k == 0:
        # Every logical output element is the additive identity.
        return zeros[out_rank](created_shape.payload.value())
    var created_output = _complete_overwrite_output[out_rank](created_shape.payload.value())
    if not created_output.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](created_output.status)
    var output = created_output.payload.take()
    if output.numel() == 0:
        return (output^)._finish_complete_overwrite()
    # Generic host-accessible storages keep the explicitly checked fallback.
    var output_data = output.impl.storage.borrow_mut_span()
    var a_step = left_layout.strides.get(rank_a - 1)
    var b_step = right_layout.strides.get(rank_b - 2)
    var cursor = NDIndexIterator[out_rank](created_shape.payload.value())
    while not cursor.done():
        var ai = StaticRankArray[rank_a](0)
        var bi = StaticRankArray[rank_b](0)
        comptime for axis in range(rank_a - 2):
            var value = cursor.coordinates.get(out_rank - rank_a + axis)
            if left_layout.shape.dims.get(axis) == 1:
                value = 0
            ai.set(axis, value)
        comptime for axis in range(rank_b - 2):
            var value = cursor.coordinates.get(out_rank - rank_b + axis)
            if right_layout.shape.dims.get(axis) == 1:
                value = 0
            bi.set(axis, value)
        ai.set(rank_a - 2, cursor.coordinates.get(out_rank - 2))
        bi.set(rank_b - 1, cursor.coordinates.get(out_rank - 1))
        ai.set(rank_a - 1, 0)
        bi.set(rank_b - 2, 0)
        var a_base = left_layout.physical_offset_unchecked(ai)
        var b_base = right_layout.physical_offset_unchecked(bi)
        var accumulator = Float32(0.0)
        for inner in range(k):
            var av = a.impl.storage.read_host(a_base + inner * a_step)
            if not av.is_ok():
                return TensorResult[Tensor[out_rank, HostStorage]](av.status)
            var bv = b.impl.storage.read_host(b_base + inner * b_step)
            if not bv.is_ok():
                return TensorResult[Tensor[out_rank, HostStorage]](bv.status)
            accumulator += av.payload.value() * bv.payload.value()
        output_data[cursor.produced] = accumulator
        cursor.advance()
    return (output^)._finish_complete_overwrite()
