# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ACCESS / ELEMENT REFERENCES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Access and References
# Module:        momijo.tensor_light.access.element_ref
# Source:        src/momijo/tensor_light/access/element_ref.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines element references in the access and references layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `TensorElementRef`, `TensorElementMutRef`,
#   `replace`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Provide checked tensor element access without exceeding storage bounds.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorElementRef`,
#     `TensorElementMutRef`, `replace`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate index bounds, backing capacity, and access mutability.
#   - Never expose an element reference beyond its owning storage lifetime.
#
# Performance Objectives:
#   - Keep checked metadata setup outside element-wise hot loops.
#   - Avoid copying tensor storage when a verified accessor is sufficient.
#
# Maintainability:
#   - Document borrowing, aliasing, and raw-pointer preconditions.
#   - Test strided, empty, and offset access paths.
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

# Checked lifetime-bound element access; no stand-alone owning element proxy.
from std.collections import Span
from ..internal.status import TensorResult, TensorStatus

struct TensorElementRef[origin: Origin](Copyable):
    var span: Span[False, Float32, Self.origin]
    var at: Int

    def __init__(out self, span: Span[False, Float32, Self.origin], at: Int):
        self.span = span
        self.at = at

    def __init__(out self, *, copy: Self):
        self.span = copy.span
        self.at = copy.at

    def get(self) -> TensorResult[Float32]:
        if self.at < 0 or self.at >= len(self.span):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.span[self.at])

struct TensorElementMutRef[origin: Origin[mut=True]](Movable):
    var span: Span[True, Float32, Self.origin]
    var at: Int

    def __init__(out self, span: Span[True, Float32, Self.origin], at: Int):
        self.span = span
        self.at = at

    def get(self) -> TensorResult[Float32]:
        if self.at < 0 or self.at >= len(self.span):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.span[self.at])

    def set(mut self, value: Float32) -> TensorStatus:
        if self.at < 0 or self.at >= len(self.span):
            return TensorStatus.OUT_OF_BOUNDS
        self.span[self.at] = value
        return TensorStatus.OK


# A checked read-modify-write operation is useful for scalar Tensor indexing.
# It does not expose the underlying Span beyond the borrowed element lifetime.
def replace[origin: Origin[mut=True]](
    mut element: TensorElementMutRef[origin], value: Float32,
) -> TensorResult[Float32]:
    var old = element.get()
    if not old.is_ok():
        return TensorResult[Float32](old.status)
    var status = element.set(value)
    if status != TensorStatus.OK:
        return TensorResult[Float32](status)
    return TensorResult[Float32](old.payload.value())
