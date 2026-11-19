# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / UNARY OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Unary Operations
# Module:        momijo.tensor_light.ops.unary
# Source:        src/momijo/tensor_light/ops/unary.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines elementwise unary mathematical operations and scalar
#   specialization paths.
#
#   Public declarations include `UnaryKind`, `unary`, `neg`, `abs`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `UnaryKind`, `unary`, `neg`,
#     `abs`, `sqrt` and related symbols.
#   - Apply unary math to tensor elements with the stated dtype rules.
#   - Expose sqrt, exp, log, and other declared transcendental functions.
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

# Primitive Float32 unary functions; checked host baseline.
from std.math import sqrt as math_sqrt, exp as math_exp, log as math_log, sin as math_sin, cos as math_cos, tanh as math_tanh
from std.math import abs as math_abs
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..factory.creation import _complete_overwrite_output
from ..iterator.nd_iterator import NDIndexIterator
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct UnaryKind(Copyable):
    var code: Int
    comptime NEG = Self(0)
    comptime ABS = Self(1)
    comptime SQRT = Self(2)
    comptime EXP = Self(3)
    comptime LOG = Self(4)
    comptime SIN = Self(5)
    comptime COS = Self(6)
    comptime TANH = Self(7)
    comptime RECIPROCAL = Self(8)

# AOT-fixed unary opcode is shared across all host-accessible storage paths.
# Compile-time FP32 unary opcode. Eliminate nine-way runtime dispatch
# from generic host-accessible storage loops.
@inline(.always)
def _unary_scalar_fp32_fixed[operation: Int](x: Float32) -> Float32:
    comptime assert operation >= 0 and operation <= 8, "invalid FP32 unary opcode"
    comptime if operation == 0:
        return -x
    elif operation == 1:
        return math_abs(x)
    elif operation == 2:
        return math_sqrt(x)
    elif operation == 3:
        return math_exp(x)
    elif operation == 4:
        return math_log(x)
    elif operation == 5:
        return math_sin(x)
    elif operation == 6:
        return math_cos(x)
    elif operation == 7:
        return math_tanh(x)
    else:
        return Float32(1.0) / x


def _unary_generic_fixed[operation: Int, rank: Int, S: TensorStorage](
    input: Tensor[rank, S]
) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime assert operation >= 0 and operation <= 8, "invalid fixed unary opcode"
    var verified = input._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](verified.status)
    var allocated = _complete_overwrite_output[rank](verified.payload.value().shape)
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    # HostStorage always returned above. Other host-readable storage uses
    # checked trait-based loads/stores without a redundant host fast path.
    var src_layout = verified.payload.value()
    var destination = out.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](src_layout.shape)
    while not cursor.done():
        var offset = src_layout.physical_offset_unchecked(cursor.coordinates)
        var source = input.impl.storage.read_host(offset)
        if not source.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](source.status)
        destination[cursor.produced] = _unary_scalar_fp32_fixed[operation](source.payload.value())
        cursor.advance()
    return out._finish_complete_overwrite()


def unary[rank: Int, S: TensorStorage](input: Tensor[rank, S], kind: UnaryKind) -> TensorResult[Tensor[rank, HostStorage]]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    if kind.code < 0 or kind.code > 8:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.UNSUPPORTED)
    comptime if identical(S, HostStorage):
        if kind.code == 0:
            return input.unary_host[0]()
        if kind.code == 1:
            return input.unary_host[1]()
        if kind.code == 2:
            return input.unary_host[2]()
        if kind.code == 3:
            return input.unary_host[3]()
        if kind.code == 4:
            return input.unary_host[4]()
        if kind.code == 5:
            return input.unary_host[5]()
        if kind.code == 6:
            return input.unary_host[6]()
        if kind.code == 7:
            return input.unary_host[7]()
        return input.unary_host[8]()
    # Choose the host-accessible AOT scalar kernel once per invocation.
    if kind.code == 0:
        return _unary_generic_fixed[0](input)
    if kind.code == 1:
        return _unary_generic_fixed[1](input)
    if kind.code == 2:
        return _unary_generic_fixed[2](input)
    if kind.code == 3:
        return _unary_generic_fixed[3](input)
    if kind.code == 4:
        return _unary_generic_fixed[4](input)
    if kind.code == 5:
        return _unary_generic_fixed[5](input)
    if kind.code == 6:
        return _unary_generic_fixed[6](input)
    if kind.code == 7:
        return _unary_generic_fixed[7](input)
    return _unary_generic_fixed[8](input)

# Named public entry point; the computation is shared with unary().
def neg[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.NEG)

# Named public entry point; the computation is shared with unary().
def abs[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.ABS)

# Named public entry point; the computation is shared with unary().
def sqrt[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.SQRT)

# Named public entry point; the computation is shared with unary().
def exp[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.EXP)

# Named public entry point; the computation is shared with unary().
def log[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.LOG)

# Named public entry point; the computation is shared with unary().
def sin[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.SIN)

# Named public entry point; the computation is shared with unary().
def cos[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.COS)

# Named public entry point; the computation is shared with unary().
def tanh[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.TANH)

# Named public entry point; the computation is shared with unary().
def reciprocal[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    return unary(input, UnaryKind.RECIPROCAL)
