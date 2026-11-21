# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
# Source: src/momijo/tensor/shape/shape.mojo
# Module: momijo.tensor.shape.shape

from std.collections import List

from ..validation import TensorResult, TensorStatusCode


struct RuntimeTensorShape(Movable):
    """Runtime dimensions for a dense row-major tensor."""

    var _dims: List[Int]
    var _numel: Int

    def __init__(out self):
        """Constructs a scalar shape."""
        self._dims = List[Int]()
        self._numel = 1

    @staticmethod
    def try_from_dims(dims: List[Int]) -> TensorResult[Self]:
        """Copies validated dimensions. Any zero extent makes the shape empty.
        """
        var is_empty = False
        for extent in dims:
            if extent < 0:
                return TensorResult[Self](TensorStatusCode.INVALID_SHAPE)
            if extent == 0:
                is_empty = True

        var count = 0
        if not is_empty:
            count = 1
            for extent in dims:
                if count > Int.MAX // extent:
                    return TensorResult[Self](TensorStatusCode.OVERFLOW)
                count *= extent

        var shape = Self()
        shape._dims = dims.copy()
        shape._numel = count
        return TensorResult[Self](shape^)

    def rank(self) -> Int:
        return len(self._dims)

    def numel(self) -> Int:
        return self._numel

    def physical_offset(self, index: List[Int]) -> TensorResult[Int]:
        """Maps zero-based coordinates to a contiguous row-major offset."""
        if len(index) != self.rank():
            return TensorResult[Int](TensorStatusCode.INVALID_RANK)
        if self._numel == 0:
            return TensorResult[Int](TensorStatusCode.OUT_OF_BOUNDS)

        var offset = 0
        for axis in range(self.rank()):
            var extent = self._dims[axis]
            var coordinate = index[axis]
            if coordinate < 0 or coordinate >= extent:
                return TensorResult[Int](TensorStatusCode.OUT_OF_BOUNDS)
            # The prefix product bounds this intermediate offset, and the
            # validated full product bounds every prefix product.
            offset = offset * extent + coordinate
        return TensorResult[Int](offset)
