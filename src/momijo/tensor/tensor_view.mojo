# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: src/momijo/tensor/tensor_view.mojo

from momijo.tensor.tensor_base import Tensor
from momijo.tensor.strides import index_to_offset 
 

# Adjust these imports to match your repo structure if needed.
from momijo.tensor.tensor import Tensor
from momijo.tensor.layout import Layout
from momijo.tensor.shape import Shape

struct TensorView[T: Copyable & Movable]:
    var base: Tensor[T]
    var view_layout: Layout
    var view_offset_elems: Int

    fn __init__(out self, base: Tensor[T], view_layout: Layout, view_offset_elems: Int):
        self.base = base
        self.view_layout = view_layout
        self.view_offset_elems = view_offset_elems

    fn shape(self) -> Shape:
        return self.view_layout.shape

    fn numel(self) -> Int:
        # Compute product of shape dims to avoid relying on Layout.numel().
        var n = 1
        var dims = self.view_layout.shape.dims
        var i = 0
        while i < len(dims):
            n *= dims[i]
            i += 1
        return n

    fn dtype_itemsize(self) -> Int:
        return self.base.dtype.itemsize

    # Return type intentionally unspecified to follow your Tensor.data_ptr() signature.
    # If you want a concrete type, annotate both sides consistently.
    fn data_ptr(self):
        return self.base.data_ptr() + (self.view_offset_elems * self.base.dtype.itemsize)

    fn data_ptr_at(self, extra_elem_offset: Int):
        return self.base.data_ptr()
             + ((self.view_offset_elems + extra_elem_offset) * self.base.dtype.itemsize)

    fn __repr__(self) -> String:
        var s = String("TensorView(")
        s += "offset_elems=" + String(self.view_offset_elems)
        s += ", numel=" + String(self.numel())
        s += ")"
        return s



struct TensorView[T: Copyable & Movable](Copyable, Movable):
    var _base: Tensor[T]
    var _shape: List[Int]
    var _strides: List[Int]
    var _offset: Int

    fn __init__(out self, base: Tensor[T], shape: List[Int], strides: List[Int], offset: Int):
        self._base = base
        self._shape = shape
        self._strides = strides
        self._offset = offset

    fn __copyinit__(out self, other: Self):
        self._base = other._base
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset

    fn shape(self) -> List[Int]:
        return self._shape

    fn ndim(self) -> Int:
        return len(self._shape)

    fn size(self) -> Int:
        var prod = 1
        var i = 0
        while i < len(self._shape):
            prod *= self._shape[i]
            i += 1
        return prod

    fn get(self, idx: List[Int]) -> T:
        var off = index_to_offset(idx, self._shape, self._strides, self._offset)
        return self._base._data[off]

    fn set(mut self, idx: List[Int], v: T) -> None:
        var off = index_to_offset(idx, self._shape, self._strides, self._offset)
        self._base._data[off] = v
