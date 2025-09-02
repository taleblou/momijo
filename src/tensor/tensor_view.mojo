# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: momijo/tensor/tensor_view.mojo

# Utilities kept minimal and dependency-safe
fn __module_name__() -> String:
    return String("momijo/tensor/tensor_view.mojo")

fn __self_test__() -> Bool:
    return True

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
