# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
# Source: src/momijo/tensor/storage.mojo
# Module: momijo.tensor.storage

from std.collections import List
from std.sys import size_of

from .shape.shape import RuntimeTensorShape
from .validation import TensorResult, TensorStatusCode


struct TensorF32(Movable):
    """An owning CPU Float32 tensor with contiguous row-major elements."""

    var _shape: RuntimeTensorShape
    var _elements: List[Float32]

    def __init__(out self, value: Float32):
        """Constructs a scalar with the supplied value."""
        self._shape = RuntimeTensorShape()
        self._elements = List[Float32](length=1, fill=value)

    @staticmethod
    def _filled(
        var shape: RuntimeTensorShape, value: Float32
    ) -> TensorResult[Self]:
        var count = shape.numel()
        if count > Int.MAX // size_of[Float32]():
            return TensorResult[Self](TensorStatusCode.OVERFLOW)

        var tensor = Self(value)
        tensor._shape = shape^
        if count != 1:
            tensor._elements = List[Float32](length=count, fill=value)
        return TensorResult[Self](tensor^)

    def numel(self) -> Int:
        return len(self._elements)

    def rank(self) -> Int:
        return self._shape.rank()

    def reshape_in_place(mut self, dims: List[Int]) -> TensorStatusCode:
        """Replaces the shape while retaining the owned row-major elements.

        The target must have the same element count. A failed reshape leaves
        the tensor's shape and elements unchanged.
        """
        var requested = RuntimeTensorShape.try_from_dims(dims)
        if not requested.is_ok():
            return requested.status()
        var shape = requested.take_value()
        if shape.numel() != self.numel():
            return TensorStatusCode.SIZE_MISMATCH
        self._shape = shape^
        return TensorStatusCode.OK

    def _elements_in_logical_order(
        self,
    ) -> Span[Float32, origin_of(self._elements)]:
        return Span(self._elements)

    def get(self, index: List[Int]) -> TensorResult[Float32]:
        """Reads one element at the supplied logical coordinates."""
        var offset = self._shape.physical_offset(index)
        if not offset.is_ok():
            return TensorResult[Float32](offset.status())
        return TensorResult[Float32](self._elements[offset.take_value()])

    def set(mut self, index: List[Int], value: Float32) -> TensorStatusCode:
        """Writes one element. Invalid coordinates leave storage unchanged."""
        var offset = self._shape.physical_offset(index)
        if not offset.is_ok():
            return offset.status()
        self._elements[offset.take_value()] = value
        return TensorStatusCode.OK
