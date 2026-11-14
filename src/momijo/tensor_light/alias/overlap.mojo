# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ALIAS / MEMORY OVERLAP ANALYSIS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Aliasing and Write Safety
# Module:        momijo.tensor_light.alias.overlap
# Source:        src/momijo/tensor_light/alias/overlap.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory overlap analysis in the aliasing and write safety layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `AliasKind`, `ranges_disjoint`,
#   `ranges_overlap`, `host_output_disjoint_from_input`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Classify alias relations and enforce safe write preconditions.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `AliasKind`,
#     `ranges_disjoint`, `ranges_overlap`, `host_output_disjoint_from_input`,
#     `host_mutable_fp32_spans_disjoint` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Distinguish proven disjointness from unknown overlap.
#   - Reject or safely handle writes through overlapping mappings.
#   - Never promote unknown provenance to proven non-aliasing.
#
# Performance Objectives:
#   - Use constant-time provenance checks before expensive overlap analysis.
#   - Retain fast copy paths only when non-overlap is proven.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Keep classification logic centralized for all consumers.
#   - Regression-test zero strides, negative strides, and shared owners.
#   - Cover recycled blocks, ownership failures, and overlapping buffers.
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

# Conservative alias relation for allocation-provenance-aware access ranges.
# A TensorLayout alone has allocation-relative offsets, not a base identity:
# comparing two independent layouts MUST NOT produce a noalias proof.

from .alias_info import AliasInfo, definitely_disjoint, may_alias
from std.collections import Span
from std.sys.info import size_of

@fieldwise_init
struct AliasKind(Copyable):
    var code: Int
    comptime DISJOINT = Self(0)
    comptime SAME = Self(1)
    comptime UNKNOWN = Self(2)


def ranges_disjoint(a: AliasInfo, b: AliasInfo) -> Bool:
    # These public metadata values are unsealed: even an `empty` claim may
    # be forged. No owner-backed disjointness proof exists at this boundary.
    return definitely_disjoint(a, b)


def ranges_overlap(a: AliasInfo, b: AliasInfo) -> Bool:
    # Means MAY overlap; UNKNOWN external owners must not authorize memcpy,
    # unsafe_as_noalias or unsynchronized concurrent writes.
    return may_alias(a, b)


# Concrete host address proof for borrowed FP32 spans. The output is mutable,
# the input is read-only, and neither integer allocation IDs nor relative
# layout offsets are used as evidence. This predicate NEVER authorizes
# unsafe_as_noalias: it only determines whether in-place writes can modify an
# unread input value. Conservative failure takes the snapshot path.


def host_output_disjoint_from_input[
    dst_origin: Origin[mut=True], src_origin: Origin,
](
    output: Span[True, Float32, dst_origin],
    source: Span[False, Float32, src_origin],
) -> Bool:
    var output_count = len(output)
    var input_count = len(source)
    # Check malformed negative lengths BEFORE the empty-span fast path.
    # A fabricated negative source length must not prove disjointness merely
    # because the destination advertises zero elements.
    if output_count < 0 or input_count < 0:
        return False
    if output_count == 0 or input_count == 0:
        return True
    if output_count > Int.MAX // 4 or input_count > Int.MAX // 4:
        return False
    # Host-only spans share a virtual address space. Their unsafe_ptr()
    # values are examined but never dereferenced by this predicate.
    var out_address = Int(output.unsafe_ptr())
    var in_address = Int(source.unsafe_ptr())
    if out_address <= 0 or in_address <= 0:
        return False
    var out_bytes = output_count * 4
    var in_bytes = input_count * 4
    if out_address > Int.MAX - out_bytes or in_address > Int.MAX - in_bytes:
        return False
    # Avoid a signed overflow in computing either end address.
    if out_address < in_address:
        return in_address - out_address >= out_bytes
    if in_address < out_address:
        return out_address - in_address >= in_bytes
    return False



# Both sides are mutable Float32 work buffers. This symmetric host-address
# check protects reusable GEMM scratch from aliasing the destination. It
# grants no unsafe noalias compiler metadata and performs NO dereferences.
def host_mutable_fp32_spans_disjoint[
    left_origin: Origin[mut=True], right_origin: Origin[mut=True],
](
    left: Span[True, Float32, left_origin],
    right: Span[True, Float32, right_origin],
) -> Bool:
    var left_count = len(left)
    var right_count = len(right)
    if left_count < 0 or right_count < 0:
        return False
    if left_count == 0 or right_count == 0:
        return True
    if left_count > Int.MAX // 4 or right_count > Int.MAX // 4:
        return False
    var first = Int(left.unsafe_ptr())
    var second = Int(right.unsafe_ptr())
    if first <= 0 or second <= 0:
        return False
    var left_bytes = left_count * 4
    var right_bytes = right_count * 4
    if first > Int.MAX - left_bytes or second > Int.MAX - right_bytes:
        return False
    if first < second:
        return second - first >= left_bytes
    if second < first:
        return first - second >= right_bytes
    return False

def host_output_exactly_matches_input[
    dst_origin: Origin[mut=True], src_origin: Origin,
](
    output: Span[True, Float32, dst_origin],
    source: Span[False, Float32, src_origin],
) -> Bool:
    # Exact forward element-for-element alias is safe for FP32 binary kernels
    # that LOAD both operands before writing each SIMD group. Partial/shifted
    # alias remains unsafe and is handled by the source-snapshot fallback.
    if len(output) != len(source):
        return False
    if len(output) == 0:
        return True
    if len(output) < 0 or len(output) > Int.MAX // 4:
        return False
    var dst_address = Int(output.unsafe_ptr())
    var src_address = Int(source.unsafe_ptr())
    var byte_count = len(output) * 4
    if dst_address <= 0 or src_address <= 0:
        return False
    if dst_address > Int.MAX - byte_count or src_address > Int.MAX - byte_count:
        return False
    return dst_address == src_address

# A concrete HOST Span pair is different from two caller-controlled integer
# provenance IDs. The address-range proof below is used only for copy/move
# policy, never as permission to synthesize a Mojo noalias pointer.
def can_use_memcpy[dst_origin: Origin[mut=True], src_origin: Origin](
    output: Span[True, Float32, dst_origin],
    source: Span[False, Float32, src_origin],
) -> Bool:
    if len(output) != len(source):
        return False
    return host_output_disjoint_from_input(output, source)


def requires_memmove[dst_origin: Origin[mut=True], src_origin: Origin](
    output: Span[True, Float32, dst_origin],
    source: Span[False, Float32, src_origin],
) -> Bool:
    # A caller must independently reject unequal logical copy sizes. When
    # any overlap is possible, memmove semantics preserve unread elements.
    if len(output) != len(source):
        # Mismatched counts are not a legal copy contract. Conservatively
        # indicate that ordinary memcpy is unsafe; the checked copy entry
        # must report SIZE_MISMATCH rather than attempting any movement.
        return True
    return not host_output_disjoint_from_input(output, source)


# Mixed element types must be compared in BYTES. In particular, an untrusted
# Int index buffer may share physical bytes with a Float32 scatter target even
# though the public Tensor dtype is fixed to Float32. This check does NOT
# authenticate a foreign allocation or authorize compiler noalias metadata.
def host_output_disjoint_from_indices[
    dst_origin: Origin[mut=True], index_origin: Origin,
](
    output: Span[True, Float32, dst_origin],
    indices: Span[False, Int, index_origin],
) -> Bool:
    var output_count = len(output)
    var index_count = len(indices)
    if output_count < 0 or index_count < 0:
        return False
    if output_count == 0 or index_count == 0:
        return True
    comptime index_size = size_of[Int]()
    if index_size <= 0:
        return False
    if output_count > Int.MAX // 4 or index_count > Int.MAX // index_size:
        return False
    var out_address = Int(output.unsafe_ptr())
    var index_address = Int(indices.unsafe_ptr())
    if out_address <= 0 or index_address <= 0:
        return False
    var out_bytes = output_count * 4
    var index_bytes = index_count * index_size
    if out_address > Int.MAX - out_bytes or index_address > Int.MAX - index_bytes:
        return False
    if out_address < index_address:
        return index_address - out_address >= out_bytes
    if index_address < out_address:
        return out_address - index_address >= index_bytes
    return False
