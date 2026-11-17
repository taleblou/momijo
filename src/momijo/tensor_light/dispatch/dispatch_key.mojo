# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / DISPATCH KEYS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.dispatch_key
# Source:        src/momijo/tensor_light/dispatch/dispatch_key.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines dispatch keys in the dispatch and specialization layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `DispatchKey`, `make_dispatch_key`,
#   `with_operation_kind`, `with_shape_bucket`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `DispatchKey`,
#     `make_dispatch_key`, `with_operation_kind`, `with_shape_bucket`,
#     `dispatch_key_matches` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate dispatch keys and trusted artifact identities before selection.
#   - Treat tuning metadata as hints, not proof of executable code.
#
# Performance Objectives:
#   - Prefer precomputed finite strategies over runtime trial-and-error.
#   - Avoid scheduling or kernel compilation in pure selection utilities.
#
# Maintainability:
#   - Keep release schemas and validation checks versioned together.
#   - Test negative cases for unbound or inconsistent specialization data.
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

# Fixed-FP32 static/AOT dispatch identity, with no runtime dtype dimension.
# The key contains metadata only; its target architecture number cannot
# authenticate a loaded native artifact or establish a hardware capability.
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct DispatchKey(Copyable):
    var operation: Int
    var rank: Int
    var shape_bucket: Int
    var layout_class: Int
    var target_arch: Int
    var math_mode: Int
    var alignment_class: Int
    var reuse_class: Int

    @staticmethod
    def checked(
        operation: Int, rank: Int, shape_bucket: Int, layout_class: Int,
        target_arch: Int, math_mode: Int, alignment_class: Int,
        reuse_class: Int,
    ) -> TensorResult[Self]:
        if rank < 0 or rank > 8:
            return TensorResult[Self](TensorStatus.INVALID_RANK)
        if operation < 0 or shape_bucket < 0 or shape_bucket > 4:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if layout_class < 0 or target_arch < 0 or reuse_class < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # STRICT=0, FMA=1, TF32_ALLOWED=2, FASTEST=3.
        if math_mode < 0 or math_mode > 3:
            return TensorResult[Self](TensorStatus.MATH_MODE_UNSUPPORTED)
        # Zero means alignment unknown. Nonzero byte alignment must be a
        # power of two and never smaller than Float32's natural alignment.
        if alignment_class != 0:
            if alignment_class < 4:
                return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
            if alignment_class & (alignment_class - 1) != 0:
                return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
        return TensorResult[Self](
            Self(operation, rank, shape_bucket, layout_class, target_arch,
                 math_mode, alignment_class, reuse_class)
        )

    def checked_snapshot(self) -> TensorResult[Self]:
        # Do not trust a @fieldwise_init instance supplied by another module.
        return Self.checked(
            self.operation, self.rank, self.shape_bucket, self.layout_class,
            self.target_arch, self.math_mode, self.alignment_class,
            self.reuse_class,
        )


# AOT dispatch helpers preserve every dimension of the validated identity.
# They create a new value rather than mutating a key reused by another plan.
def make_dispatch_key(
    operation: Int, rank: Int, shape_bucket: Int, layout_class: Int,
    target_arch: Int, math_mode: Int, alignment_class: Int,
    reuse_class: Int,
) -> TensorResult[DispatchKey]:
    return DispatchKey.checked(
        operation, rank, shape_bucket, layout_class, target_arch,
        math_mode, alignment_class, reuse_class,
    )


def with_operation_kind(key: DispatchKey, operation: Int) -> TensorResult[DispatchKey]:
    var checked = key.checked_snapshot()
    if not checked.is_ok():
        return checked
    var previous = checked.payload.value()
    return DispatchKey.checked(
        operation, previous.rank, previous.shape_bucket,
        previous.layout_class, previous.target_arch, previous.math_mode,
        previous.alignment_class, previous.reuse_class,
    )


def with_shape_bucket(key: DispatchKey, bucket: Int) -> TensorResult[DispatchKey]:
    var checked = key.checked_snapshot()
    if not checked.is_ok():
        return checked
    var previous = checked.payload.value()
    return DispatchKey.checked(
        previous.operation, previous.rank, bucket,
        previous.layout_class, previous.target_arch, previous.math_mode,
        previous.alignment_class, previous.reuse_class,
    )


def dispatch_key_matches(a: DispatchKey, b: DispatchKey) -> Bool:
    var left = a.checked_snapshot()
    if not left.is_ok():
        return False
    var right = b.checked_snapshot()
    if not right.is_ok():
        return False
    var x = left.payload.value()
    var y = right.payload.value()
    return (
        x.operation == y.operation and x.rank == y.rank
        and x.shape_bucket == y.shape_bucket
        and x.layout_class == y.layout_class
        and x.target_arch == y.target_arch
        and x.math_mode == y.math_mode
        and x.alignment_class == y.alignment_class
        and x.reuse_class == y.reuse_class
    )


def make_dataflow_dispatch_key(
    primitive: Int, opcode: Int, rank: Int, shape_bucket: Int,
    layout_class: Int, target_arch: Int, math_mode: Int,
    alignment_class: Int, reuse_class: Int,
) -> TensorResult[DispatchKey]:
    # Stable source IR -> AOT selector identity. The same encoding is checked
    # against EVERY compute node by validate_finite_aot_dataflow_key().
    # Do not pass a bare family opcode (e.g. binary add=0) as the global
    # DispatchKey.operation: copy=0 and binary add=0 are different kernels.
    # Pure structural bounds; the graph's checked snapshot still validates
    # actual operand arity, rank-specific shapes and memory lifetimes.
    if primitive < 0 or primitive > 8 or opcode < 0:
        return TensorResult[DispatchKey](TensorStatus.INVALID_LAYOUT)
    if ((primitive == 0 or primitive == 3 or primitive == 6
         or primitive == 7 or primitive == 8) and opcode != 0):
        return TensorResult[DispatchKey](TensorStatus.UNSUPPORTED)
    if ((primitive == 1 and opcode > 8)
        or (primitive == 2 and opcode > 5)
        # Argmin/argmax produce integer locations, not FP32 tensors. The
        # bounded FP32 dataflow ABI explicitly rejects these opcodes, so the
        # key constructor must reject them at the SAME source contract.
        or (primitive == 4 and opcode > 4)
        or (primitive == 5 and opcode > 3)):
        return TensorResult[DispatchKey](TensorStatus.UNSUPPORTED)
    # The key builder must enforce every semantic restriction expressible
    # from its parameters, not postpone obviously impossible choices until
    # native binding. A scan has an axis; matrix is dense rank-2; flat
    # concatenation is a 1-D ABI; and FMA is forbidden in strict FP32 mode.
    if primitive == 5 and rank == 0:
        return TensorResult[DispatchKey](TensorStatus.INVALID_RANK)
    if primitive == 6 and rank != 2:
        return TensorResult[DispatchKey](TensorStatus.INVALID_RANK)
    if primitive == 7 and rank != 1:
        return TensorResult[DispatchKey](TensorStatus.INVALID_RANK)
    if primitive == 3 and math_mode == 0:
        return TensorResult[DispatchKey](TensorStatus.MATH_MODE_UNSUPPORTED)
    return DispatchKey.checked(
        primitive * 16 + opcode, rank, shape_bucket, layout_class,
        target_arch, math_mode, alignment_class, reuse_class,
    )
