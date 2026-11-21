# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / LAYOUT FLAGS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_flags
# Source:        src/momijo/tensor_light/core/tensor_flags.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor layout flags and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorFlags`, `compute_tensor_flags`,
#   `TensorFlag`, `has_flag`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorFlags`,
#     `compute_tensor_flags`, `TensorFlag`, `has_flag`, `with_flag`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve owning-tensor invariants and initialized-storage contracts.
#   - Check shape/layout compatibility before changing tensor metadata.
#
# Performance Objectives:
#   - Prefer metadata-only transformations when ownership permits.
#   - Keep hot-path operations specialized without weakening checks.
#
# Maintainability:
#   - Keep the primary tensor ownership model unambiguous.
#   - Document public constructors, moves, and borrow boundaries.
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

# Derived immutable flag snapshot. Layout and storage remain authoritative.
from ..layout.layout import TensorLayout
from ..shape.shape import TensorShape
from ..layout.contiguity import internal_overlap_kind, InternalOverlapKind, is_exhaustive_mapping

@fieldwise_init
struct TensorFlags(Copyable):
    var empty: Bool
    var scalar: Bool
    var c_contiguous: Bool
    var f_contiguous: Bool
    var proven_unique: Bool
    var exhaustive: Bool

def compute_tensor_flags[rank: Int](layout: TensorLayout[rank]) -> TensorFlags:
    # Flags are computed from the mapping rather than its caller-writable
    # overlap/span caches. A malformed cached shape cannot grant writeability
    # or advertise a contiguous fast path to the planner.
    var checked = TensorShape[rank].checked(layout.shape.dims)
    if not checked.is_ok():
        return TensorFlags(False, rank == 0, False, False, False, False)
    var canonical = checked.payload.value()
    if canonical.numel() != layout.shape.numel():
        return TensorFlags(False, rank == 0, False, False, False, False)
    var unique = internal_overlap_kind[rank](canonical, layout.strides) == InternalOverlapKind.NONE
    return TensorFlags(
        canonical.is_empty(), rank == 0,
        layout.is_c_contiguous(), layout.is_f_contiguous(),
        unique, is_exhaustive_mapping[rank](canonical, layout.strides),
    )

@fieldwise_init
struct TensorFlag(Copyable, Equatable):
    # Flags are DERIVED cache bits; they never grant writeability/ownership.
    var code: Int
    comptime EMPTY = Self(0)
    comptime SCALAR = Self(1)
    comptime C_CONTIGUOUS = Self(2)
    comptime F_CONTIGUOUS = Self(3)
    comptime PROVEN_UNIQUE = Self(4)
    comptime EXHAUSTIVE = Self(5)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


def has_flag(flags: TensorFlags, flag: TensorFlag) -> Bool:
    if flag == materialize[TensorFlag.EMPTY]():
        return flags.empty
    if flag == materialize[TensorFlag.SCALAR]():
        return flags.scalar
    if flag == materialize[TensorFlag.C_CONTIGUOUS]():
        return flags.c_contiguous
    if flag == materialize[TensorFlag.F_CONTIGUOUS]():
        return flags.f_contiguous
    if flag == materialize[TensorFlag.PROVEN_UNIQUE]():
        return flags.proven_unique
    if flag == materialize[TensorFlag.EXHAUSTIVE]():
        return flags.exhaustive
    return False


def with_flag(flags: TensorFlags, flag: TensorFlag, value: Bool) -> TensorFlags:
    # Pure snapshot transform. TensorLayout ALWAYS remains authoritative.
    var result = TensorFlags(
        flags.empty, flags.scalar, flags.c_contiguous,
        flags.f_contiguous, flags.proven_unique, flags.exhaustive
    )
    if flag == materialize[TensorFlag.EMPTY]():
        result.empty = value
    elif flag == materialize[TensorFlag.SCALAR]():
        result.scalar = value
    elif flag == materialize[TensorFlag.C_CONTIGUOUS]():
        result.c_contiguous = value
    elif flag == materialize[TensorFlag.F_CONTIGUOUS]():
        result.f_contiguous = value
    elif flag == materialize[TensorFlag.PROVEN_UNIQUE]():
        result.proven_unique = value
    elif flag == materialize[TensorFlag.EXHAUSTIVE]():
        result.exhaustive = value
    return result.copy()
