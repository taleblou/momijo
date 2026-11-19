# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / ARITHMETIC
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Arithmetic
# Module:        momijo.tensor_light.ops.arithmetic
# Source:        src/momijo/tensor_light/ops/arithmetic.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines elementwise tensor arithmetic, scalar operations, and
#   operation-specific broadcast handling.
#
#   Public declarations include `BinaryOpKind`, `binary`, `fma`, `fma_nd`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `BinaryOpKind`, `binary`,
#     `fma`, `fma_nd`, `add` and related symbols.
#   - Align operands and validate elementwise/broadcast result shapes.
#   - Expose fused multiply-add with the source-defined rounding policy.
#   - Support the declared N-D broadcast arithmetic entry points.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
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

# Baseline correct FP32 broadcast arithmetic for equally ranked CPU tensors.
# AOT vector kernels may replace the scalar fallback without changing semantics.
from ..core.tensor import Tensor
from ..storage.host_storage import HostStorage
from ..storage.storage import TensorStorage
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..factory.creation import _complete_overwrite_output
from ..internal.status import TensorStatus, TensorResult
from ..iterator.nd_iterator import NDIndexIterator
from ..dtype.fp32 import fp32_minimum, fp32_maximum
from ..dtype.math_mode import Fp32MathMode
from ..kernels.cpu.elementwise import cpu_add_contiguous, cpu_sub_contiguous, cpu_mul_contiguous, cpu_div_contiguous, cpu_scale_contiguous, cpu_shift_contiguous
from std.sys.info import simd_width_of

struct BinaryOpKind(Copyable):
    var code: Int
    comptime ADD = Self(0)
    comptime SUB = Self(1)
    comptime MUL = Self(2)
    comptime DIV = Self(3)
    comptime MAXIMUM = Self(4)
    comptime MINIMUM = Self(5)
    def __init__(out self, code: Int):
        self.code = code
    def __init__(out self, *, copy: Self):
        self.code = copy.code

# Operation identity is compile-time. The six specializations share one
# FP32 policy for NaN and signed zero, and contain no per-element op switch.
@inline(.always)
def _apply_fp32_binary_fixed[operation: Int](x: Float32, y: Float32) -> Float32:
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 binary opcode"
    comptime if operation == 0:
        return x + y
    elif operation == 1:
        return x - y
    elif operation == 2:
        return x * y
    elif operation == 3:
        return x / y
    elif operation == 4:
        return fp32_maximum(x, y)
    else:
        return fp32_minimum(x, y)


def _binary_generic_fixed[operation: Int, rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[rank, SA], b: Tensor[rank, SB]
) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime assert operation >= 0 and operation <= 5, "invalid fixed binary opcode"
    var a_valid = a._checked_current_layout()
    if not a_valid.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](a_valid.status)
    var b_valid = b._checked_current_layout()
    if not b_valid.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](b_valid.status)
    var out_dims = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var da = a_valid.payload.value().shape.dims.get(i)
        var db = b_valid.payload.value().shape.dims.get(i)
        if da == db:
            out_dims.set(i, da)
        elif da == 1:
            out_dims.set(i, db)
        elif db == 1:
            out_dims.set(i, da)
        else:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var inferred = TensorShape[rank].checked(out_dims)
    if not inferred.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](inferred.status)
    var allocated = _complete_overwrite_output[rank](inferred.payload.value())
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    var dst = out.impl.storage.borrow_mut_span()
    # Broadcast strides are determined once at the validated boundary.
    # Non-HostStorage capabilities retain their checked physical read hooks,
    # but do not reconstruct source coordinates for each output element.
    var a_steps = a_valid.payload.value().strides.copy()
    var b_steps = b_valid.payload.value().strides.copy()
    comptime for axis in range(rank):
        if a_valid.payload.value().shape.dims.get(axis) == 1:
            a_steps.set(axis, 0)
        if b_valid.payload.value().shape.dims.get(axis) == 1:
            b_steps.set(axis, 0)
    var cursor = NDIndexIterator[rank](inferred.payload.value())
    while not cursor.done():
        var a_at = a_valid.payload.value().offset
        var b_at = b_valid.payload.value().offset
        comptime for axis in range(rank):
            var coordinate = cursor.coordinates.get(axis)
            a_at += coordinate * a_steps.get(axis)
            b_at += coordinate * b_steps.get(axis)
        var left = a.impl.storage.read_host(a_at)
        if not left.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](left.status)
        var right = b.impl.storage.read_host(b_at)
        if not right.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](right.status)
        var value = _apply_fp32_binary_fixed[operation](left.payload.value(), right.payload.value())
        dst[cursor.produced] = value
        cursor.advance()
    return out._finish_complete_overwrite()


def binary[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB], kind: BinaryOpKind) -> TensorResult[Tensor[rank, HostStorage]]:
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    if kind.code < 0 or kind.code > 5:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.UNSUPPORTED)
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        if kind.code == 0:
            return a._binary_host[0](b)
        if kind.code == 1:
            return a._binary_host[1](b)
        if kind.code == 2:
            return a._binary_host[2](b)
        if kind.code == 3:
            return a._binary_host[3](b)
        if kind.code == 4:
            return a._binary_host[4](b)
        return a._binary_host[5](b)
    # Non-HostStorage operands dispatch ONCE, not for every FP32 element.
    if kind.code == 0:
        return _binary_generic_fixed[0](a, b)
    if kind.code == 1:
        return _binary_generic_fixed[1](a, b)
    if kind.code == 2:
        return _binary_generic_fixed[2](a, b)
    if kind.code == 3:
        return _binary_generic_fixed[3](a, b)
    if kind.code == 4:
        return _binary_generic_fixed[4](a, b)
    return _binary_generic_fixed[5](a, b)



# The public functional operation uses the primary Tensor FMA implementation.
# It is one FP32 operation, not a multiply allocating a temporary plus add.
def fma[rank: Int](
    a: Tensor[rank, HostStorage], b: Tensor[rank, HostStorage],
    c: Tensor[rank, HostStorage], mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
) -> TensorResult[Tensor[rank, HostStorage]]:
    return a.fma(b, c, mode)


def fma_nd[ra: Int, rb: Int, rc: Int, out_rank: Int](
    a: Tensor[ra, HostStorage], b: Tensor[rb, HostStorage],
    c: Tensor[rc, HostStorage], mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return a.fma_nd[rb, rc, out_rank](b, c, mode)

def add[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    return binary(a, b, BinaryOpKind.ADD)

def sub[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    return binary(a, b, BinaryOpKind.SUB)

def mul[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    return binary(a, b, BinaryOpKind.MUL)

def div[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    return binary(a, b, BinaryOpKind.DIV)


def maximum[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    # The operation is known at compile time: avoid a runtime BinaryOpKind
    # dispatch and call the validated, one-allocation kernel directly.
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a._binary_host[4](b)
    else:
        return _binary_generic_fixed[4](a, b)


def minimum[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB]) -> TensorResult[Tensor[rank, HostStorage]]:
    # The operation is known at compile time: avoid a runtime BinaryOpKind
    # dispatch and call the validated, one-allocation kernel directly.
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a._binary_host[5](b)
    else:
        return _binary_generic_fixed[5](a, b)


# The public functional API is a thin delegation to Tensor's owning methods.
# There is no core -> ops dependency and only one host scalar implementation.
def add_scalar[rank: Int](a: Tensor[rank, HostStorage], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return a.add_scalar(value)


def sub_scalar[rank: Int](a: Tensor[rank, HostStorage], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return a.sub_scalar(value)


def mul_scalar[rank: Int](a: Tensor[rank, HostStorage], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return a.mul_scalar(value)


def div_scalar[rank: Int](a: Tensor[rank, HostStorage], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return a.div_scalar(value)


# Right-aligned NumPy-style broadcasting with compile-time output rank.
# Equal-rank inputs delegate to the owning SIMD Tensor facade.
# Different-rank inputs dispatch once into an AOT-specialized scalar loop.
# No dynamic opcode branching in the loop and no checked output write per item.
def _binary_nd_fixed[operation: Int, ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    comptime assert operation >= 0 and operation <= 5, "invalid fixed binary opcode"
    comptime assert out_rank == max(ra, rb) and out_rank <= 8, "invalid broadcast output rank"
    comptime if identical(SA, HostStorage) and identical(SB, HostStorage):
        return a._binary_nd_host[operation, rb, out_rank](b)
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](checked_b.status)
    var dims = StaticRankArray[out_rank](1)
    comptime for i in range(out_rank):
        var da = 1
        var db = 1
        comptime if i >= out_rank - ra:
            da = checked_a.payload.value().shape.dims.get(i - (out_rank - ra))
        comptime if i >= out_rank - rb:
            db = checked_b.payload.value().shape.dims.get(i - (out_rank - rb))
        if da == db:
            dims.set(i, da)
        elif da == 1:
            dims.set(i, db)
        elif db == 1:
            dims.set(i, da)
        else:
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var shape = TensorShape[out_rank].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](shape.status)
    # Right-align and resolve zero-stride broadcast dimensions once. This is
    # the shared checked-boundary lowering for both host-specialized and
    # generic host-visible storage; rank zero naturally has no active axes.
    var a_steps = StaticRankArray[out_rank](0)
    var b_steps = StaticRankArray[out_rank](0)
    comptime for i in range(ra):
        comptime target_axis = out_rank - ra + i
        if checked_a.payload.value().shape.dims.get(i) != 1:
            a_steps.set(target_axis, checked_a.payload.value().strides.get(i))
    comptime for i in range(rb):
        comptime target_axis = out_rank - rb + i
        if checked_b.payload.value().shape.dims.get(i) != 1:
            b_steps.set(target_axis, checked_b.payload.value().strides.get(i))
    # A disjoint output is written once per logical coordinate.
    var allocated = _complete_overwrite_output[out_rank](shape.payload.value())
    if not allocated.is_ok():
        return TensorResult[Tensor[out_rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    var dst = out.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[out_rank](shape.payload.value())
    while not cursor.done():
        var a_at = checked_a.payload.value().offset
        var b_at = checked_b.payload.value().offset
        comptime for axis in range(out_rank):
            var coordinate = cursor.coordinates.get(axis)
            a_at += coordinate * a_steps.get(axis)
            b_at += coordinate * b_steps.get(axis)
        var x = a.impl.storage.read_host(a_at)
        if not x.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](x.status)
        var y = b.impl.storage.read_host(b_at)
        if not y.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](y.status)
        var result = _apply_fp32_binary_fixed[operation](x.payload.value(), y.payload.value())
        dst[cursor.produced] = result
        cursor.advance()
    return out._finish_complete_overwrite()


def binary_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB], kind: BinaryOpKind
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    comptime assert ra >= 0 and ra <= 8 and rb >= 0 and rb <= 8, "rank must be 0..8"
    comptime assert out_rank == max(ra, rb) and out_rank <= 8, "invalid broadcast output rank"
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    if kind.code < 0 or kind.code > 5:
        return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.UNSUPPORTED)
    comptime if ra == rb and ra == out_rank:
        return binary(a, b, kind)
    if kind.code == 0:
        return _binary_nd_fixed[0](a, b)
    if kind.code == 1:
        return _binary_nd_fixed[1](a, b)
    if kind.code == 2:
        return _binary_nd_fixed[2](a, b)
    if kind.code == 3:
        return _binary_nd_fixed[3](a, b)
    if kind.code == 4:
        return _binary_nd_fixed[4](a, b)
    return _binary_nd_fixed[5](a, b)


def add_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return binary_nd[ra, rb, out_rank](a, b, BinaryOpKind.ADD)


def sub_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return binary_nd[ra, rb, out_rank](a, b, BinaryOpKind.SUB)


def mul_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return binary_nd[ra, rb, out_rank](a, b, BinaryOpKind.MUL)


def div_nd[ra: Int, rb: Int, out_rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[ra, SA], b: Tensor[rb, SB]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return binary_nd[ra, rb, out_rank](a, b, BinaryOpKind.DIV)
