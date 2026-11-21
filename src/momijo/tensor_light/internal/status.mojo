# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INTERNAL / STATUS AND RESULT TYPES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Internal Validation Utilities
# Module:        momijo.tensor_light.internal.status
# Source:        src/momijo/tensor_light/internal/status.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines status and result types in the internal validation utilities layer
#   of the FP32 Light tensor implementation.
#
#   Public declarations include `TensorStatus`, `TensorResult`, `ok_result`,
#   `error_result`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorStatus`,
#     `TensorResult`, `ok_result`, `error_result`, `is_ok` and related
#     symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Detect integer overflow and invalid metadata before dereference.
#   - Return explicit statuses for unsupported or invalid states.
#
# Performance Objectives:
#   - Use simple inlinable checked primitives where appropriate.
#   - Keep validation work proportional to required metadata.
#
# Maintainability:
#   - Reuse one error vocabulary across dependent modules.
#   - Test boundary arithmetic and invalid-state propagation.
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

# Central checked-result convention. Supports move-only ownership payloads.
# Checked public APIs do not abort; a caller explicitly extracts a present value.
from std.collections import Optional

@fieldwise_init
struct TensorStatus(ImplicitlyCopyable, Equatable):
    var code: Int

    comptime OK = Self(0)
    comptime INVALID_RANK = Self(1)
    comptime INVALID_SHAPE = Self(2)
    comptime OVERFLOW = Self(3)
    comptime OUT_OF_BOUNDS = Self(4)
    comptime INVALID_LAYOUT = Self(5)
    comptime NOT_REPRESENTABLE = Self(6)
    comptime SIZE_MISMATCH = Self(7)
    comptime READ_ONLY = Self(8)
    comptime OVERLAPPING = Self(9)
    comptime INVALID_AXIS = Self(10)
    comptime DEVICE_MISMATCH = Self(11)
    comptime BACKEND_UNAVAILABLE = Self(12)
    comptime UNSUPPORTED = Self(13)
    comptime ALLOCATION_FAILED = Self(14)
    comptime INTERNAL_ERROR = Self(15)
    # Stable additive extension of the specification's error vocabulary.
    # Existing numeric codes 0..15 remain unchanged for source/ABI users.
    comptime INVALID_STRIDE = Self(16)
    comptime INVALID_OFFSET = Self(17)
    comptime INVALID_SLICE = Self(18)
    comptime SHAPE_MISMATCH = Self(19)
    comptime NOT_CONTIGUOUS = Self(20)
    comptime NOT_EXHAUSTIVE = Self(21)
    comptime ALIAS_UNKNOWN = Self(22)
    comptime INVALID_ALIGNMENT = Self(23)
    comptime TRANSFER_UNAVAILABLE = Self(24)
    comptime MATH_MODE_UNSUPPORTED = Self(25)
    comptime HYBRID_PARTITION_INVALID = Self(26)
    # Distinguish an owned, capacity-valid buffer from fully initialized data.
    comptime UNINITIALIZED = Self(27)
    # Recoverable native kernel error with stream quiescence established.
    comptime NATIVE_LAUNCH_FAILED = Self(28)
    # An event wait AND the fallback stream synchronization both failed.
    # Device output owners must not be destroyed/recycled while this holds.
    comptime GPU_QUIESCENCE_UNPROVEN = Self(29)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

# Spec-compatible alias for the same status vocabulary; no duplicate enum.
comptime TensorStatusCode = TensorStatus

struct TensorResult[T: Movable & Deinitable](Movable):
    var status: TensorStatus
    var payload: Optional[Self.T]

    def __init__(out self, var value: Self.T):
        self.status = TensorStatus.OK
        self.payload = Optional[Self.T](value^)

    def __init__(out self, error: TensorStatus):
        if error == TensorStatus.OK:
            self.status = TensorStatus.INTERNAL_ERROR
        else:
            self.status = error
        self.payload = Optional[Self.T](None)

    def is_ok(self) -> Bool:
        return self.status == TensorStatus.OK and Bool(self.payload)

    def has_value(self) -> Bool:
        # Never advertise a payload from a failed status, even if an internal
        # builder constructed a structurally inconsistent result object.
        return self.is_ok()

    def take_checked(mut self) -> Optional[Self.T]:
        if not self.is_ok():
            return None
        return self.payload.take()


    def value_optional(self) -> Optional[Self.T] where conforms_to(Self.T, Copyable):
        if not self.is_ok():
            return None
        return Optional[Self.T](self.payload.value().copy())


def ok_result[T: Movable & Deinitable](var value: T) -> TensorResult[T]:
    return TensorResult[T](value^)


def error_result[T: Movable & Deinitable](error: TensorStatus) -> TensorResult[T]:
    return TensorResult[T](error)


def is_ok[T: Movable & Deinitable](result: TensorResult[T]) -> Bool:
    return result.is_ok()


def status_code[T: Movable & Deinitable](result: TensorResult[T]) -> TensorStatus:
    return result.status


def take_value[T: Movable & Deinitable](mut result: TensorResult[T]) -> Optional[T]:
    # Caller must check Optional; errors never force an unwrap or abort.
    return result.take_checked()


# Single-launch-only failure classifier. Never reuse it for an entire
# chunked or CPU/GPU-split operation: a SIZE_MISMATCH, UNSUPPORTED or
# NOT_REPRESENTABLE in a later chunk can follow completed writes in an
# earlier chunk. Every owning multi-stage caller must invalidate its entire
# output on ANY non-OK return (see Tensor.unsafe_hybrid_*_into).
# A synchronous native adapter has two failure phases. A validated, purely
# structural rejection leaves a previously initialized output untouched.
# NATIVE_LAUNCH_FAILED and TRANSFER_UNAVAILABLE may mean a partial native
# overwrite even when stream synchronization later proved quiescence.
# Fail CLOSED on future/unrecognized status codes: never preserve a readable
# destination for an error whose enqueue phase has not been classified.
def device_write_failure_may_have_enqueued(status: TensorStatus) -> Bool:
    if status == TensorStatus.OK:
        return False
    if (status == TensorStatus.INVALID_RANK
        or status == TensorStatus.INVALID_SHAPE
        or status == TensorStatus.OVERFLOW
        or status == TensorStatus.OUT_OF_BOUNDS
        or status == TensorStatus.INVALID_LAYOUT
        or status == TensorStatus.NOT_REPRESENTABLE
        or status == TensorStatus.SIZE_MISMATCH
        or status == TensorStatus.READ_ONLY
        or status == TensorStatus.OVERLAPPING
        or status == TensorStatus.INVALID_AXIS
        or status == TensorStatus.DEVICE_MISMATCH
        or status == TensorStatus.UNSUPPORTED
        or status == TensorStatus.INVALID_STRIDE
        or status == TensorStatus.INVALID_OFFSET
        or status == TensorStatus.INVALID_SLICE
        or status == TensorStatus.SHAPE_MISMATCH
        or status == TensorStatus.NOT_CONTIGUOUS
        or status == TensorStatus.NOT_EXHAUSTIVE
        or status == TensorStatus.ALIAS_UNKNOWN
        or status == TensorStatus.INVALID_ALIGNMENT
        or status == TensorStatus.MATH_MODE_UNSUPPORTED
        or status == TensorStatus.HYBRID_PARTITION_INVALID
        or status == TensorStatus.UNINITIALIZED):
        return False
    # Includes NATIVE_LAUNCH_FAILED, TRANSFER_UNAVAILABLE,
    # GPU_QUIESCENCE_UNPROVEN, BACKEND_UNAVAILABLE and INTERNAL_ERROR.
    return True
