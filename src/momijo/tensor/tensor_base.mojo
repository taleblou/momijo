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
# File: src/momijo/tensor/tensor_base.mojo

from momijo.tensor.device import Device
from momijo.tensor.dtype import DType
from momijo.tensor.layout import Layout
from momijo.tensor.shape import Shape
from momijo.tensor.storage import Storage

fn __module_name__() -> String:
    return String("momijo/tensor/tensor_base.mojo")
fn __self_test__() -> Bool:
    # extend with real checks as needed
    return True

# --- Lightweight helpers (no external deps) ---
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# --- Explicit imports (no relative/re-) ---

# Base tensor header/type-agnostic container
struct Tensor:
    var storage: Storage
    var layout: Layout
    var dtype: DType
    var storage_offset_elems: Int
fn __init__(out self,
                storage: Storage,
                layout: Layout,
                dtype: DType,
                storage_offset_elems: Int = 0) -> None:
        self.storage = storage
        self.layout = layout
        self.dtype = dtype
        self.storage_offset_elems = storage_offset_elems

    # ---------- Introspection ----------
fn shape(self) -> Shape:
        return self.layout.shape
fn strides(self) -> List[Int]:
        return self.layout.strides
fn device(self) -> Device:
        return self.storage.device
fn ndim(self) -> Int:
        return self.layout.shape.rank()
fn size(self, dim: Int) -> Int:
        return self.layout.shape.dim(dim)
fn numel(self) -> Int:
        return self.layout.shape.numel()
fn nbytes(self) -> Int:
        var n = self.numel()
        if n < 0:
            return -1
        return n * self.dtype.itemsize
fn is_contiguous(self) -> Bool:
        return self.layout.is_contiguous()

    # ---------- Storage/Pointer ----------
    # NOTE: Assumes Storage::data_ptr() returns Pointer[UInt8] (byte-level).
    # Offset is in *elements*, so convert to bytes via dtype.itemsize.
fn data_ptr(self) -> Pointer[UInt8]:
        return self.storage.data_ptr() + (self.storage_offset_elems * self.dtype.itemsize)

    # ---------- Lightweight mutators that return a new header ----------
fn with_storage_offset(self, offset_elems: Int) -> Tensor:
        return Tensor(self.storage, self.layout, self.dtype, offset_elems)
fn with_layout(self, layout: Layout) -> Tensor:
        return Tensor(self.storage, layout, self.dtype, self.storage_offset_elems)
fn with_dtype(self, dtype: DType) -> Tensor:
        return Tensor(self.storage, self.layout, dtype, self.storage_offset_elems)
fn __copyinit__(out self, other: Self) -> None:
        self.storage = other.storage
        self.layout = other.layout
        self.dtype = other.dtype
        self.storage_offset_elems = other.storage_offset_elems
fn __moveinit__(out self, deinit other: Self) -> None:
        self.storage = other.storage
        self.layout = other.layout
        self.dtype = other.dtype
        self.storage_offset_elems = other.storage_offset_elems