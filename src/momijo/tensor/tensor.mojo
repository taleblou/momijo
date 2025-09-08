# Project:      Momijo
# Module:       src.momijo.tensor.tensor
# File:         tensor.mojo
# Path:         src/momijo/tensor/tensor.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: Tensor, _FlatIter
#   - Key functions: _require, _shape_product, compute_strides, __init__, __copyinit__, from_list, ndim, size ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


from gpu.host import dim
from momijo.core.ndarray import offset
from momijo.core.shape import Slice
from momijo.core.traits import one, zero
from momijo.core.types import Axis
from momijo.dataframe.helpers import t
from momijo.nn.parameter import data
from momijo.tensor.shape import dim
from momijo.utils.result import f
from pathlib import Path
from pathlib.path import Path
from utils.index import Index

fn _require(cond: Bool, msg: String) -> None:
    if not cond:
        print(String("[REQUIRE FAIL] ") + msg)
fn _shape_product(shape: List[Int]) -> Int:
    var s: Int = 1
    var i = 0
    while i < len(shape):
        var d = shape[i]
        _require(d >= 0, String("Negative dimension: ") + String(d))
        if d == 0:
            s *= 1
        else:
            s *= d
        i += 1
    return s
fn compute_strides(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var st = List[Int]()
    var i = 0
    while i < n:
        st.append(0)
        i += 1
    var acc: Int = 1
    i = 0
    while i < n:
        var j = n - 1 - i
        st[j] = acc
        var d = shape[j]
        if d == 0:
            acc *= 1
        else:
            acc *= d
        i += 1
    return st

struct Tensor[T: Copyable & Movable](Copyable, Movable):
    var shape: List[Int]
    var strides: List[Int]
    var data: List[T]
    var offset: Int
fn __init__(out self, shape: List[Int], fill: T) -> None:
        var size: Int = _shape_product(shape)
        self.shape = shape
        self.strides = compute_strides(shape)
        self.data = List[T]()
        var i = 0
        while i < size:
            self.data.append(fill)
            i += 1
        self.offset = 0
fn __copyinit__(out self, other: Self) -> None:
        var shp = List[Int]()
        var i = 0
        while i < len(other.shape):
            shp.append(other.shape[i])
            i += 1
        var strd = List[Int]()
        i = 0
        while i < len(other.strides):
            strd.append(other.strides[i])
            i += 1
        var buf = List[T]()
        i = 0
        while i < len(other.data):
            buf.append(other.data[i])
            i += 1
        self.shape = shp
        self.strides = strd
        self.data = buf
        self.offset = other.offset

    @staticmethod
fn from_list(shape: List[Int], values: List[T]) -> Tensor[T]:
        var need = _shape_product(shape)
        _require(need == len(values), String("Values length mismatch: need=") + String(need) + String(", got=") + String(len(values)))
        _require(len(values) > 0, String("from_list requires non-empty values"))
        var t = Tensor[T](shape=shape, fill=values[0])
        t.data = List[T]()
        var i = 0
        while i < len(values):
            t.data.append(values[i])
            i += 1
        t.offset = 0
        t.strides = compute_strides(shape)
        return t
fn ndim(self) -> Int:
        return len(self.shape)
fn size(self) -> Int:
        return _shape_product(self.shape)
fn is_contiguous(self) -> Bool:
        return self.strides == compute_strides(self.shape)
fn index(self, idx: List[Int]) -> Int:
        _require(len(idx) == len(self.shape), String("Index rank mismatch"))
        var flat: Int = self.offset
        var i = 0
        while i < len(idx):
            var ii = idx[i]
            var d = self.shape[i]
            if d == 0:
                d = 1
            _require(ii >= 0 and ii < d, String("Index out of bounds at axis ") + String(i))
            flat += ii * self.strides[i]
            i += 1
        return flat
fn get(self, idx: List[Int]) -> T:
        return self.data[self.index(idx)]
fn set(mut self, idx: List[Int], value: T) -> None:
        var pos = self.index(idx)
        self.data[pos] = value
fn reshape(self, new_shape: List[Int]) -> Tensor[T]:
        var old_sz = self.size()
        var new_sz = _shape_product(new_shape)
        _require(old_sz == new_sz, String("Reshape size mismatch"))
        var init_val = self.data[self.offset]
        var out = Tensor[T](shape=new_shape, fill=init_val)
        out.data = self.data
        out.offset = self.offset
        out.strides = compute_strides(new_shape)
        return out
fn slice(self, starts: List[Int], stops: List[Int], steps: List[Int]) -> Tensor[T]:
        _require(len(starts) == len(stops) and len(starts) == len(self.shape), String("Slice rank mismatch"))
        var new_shape = List[Int]()
        var new_offset = self.offset
        var new_strides = self.strides
        var i = 0
        while i < len(self.shape):
            var s = starts[i]
            var e = stops[i]
            var step = steps[i]
            _require(step == 1, String("Only step=1 supported"))
            var dim = self.shape[i]
            if dim == 0:
                dim = 1
            _require(s >= 0 and e <= dim and s <= e, String("Invalid slice bounds at axis ") + String(i))
            new_shape.append(e - s)
            new_offset += s * self.strides[i]
            i += 1
        var init_val = self.data[self.offset]
        var view = Tensor[T](shape=new_shape, fill=init_val)
        view.data = self.data
        view.offset = new_offset
        view.strides = new_strides
        return view
fn transpose(self, perm: List[Int]) -> Tensor[T]:
        _require(len(perm) == self.ndim(), String("Permutation rank mismatch"))
        var n = self.ndim()
        var seen = List[Int]()
        var i = 0
        while i < n:
            seen.append(0)
            i += 1
        i = 0
        while i < n:
            var p = perm[i]
            _require(p >= 0 and p < n, String("Invalid axis in permutation"))
            seen[p] = seen[p] + 1
            i += 1
        i = 0
        while i < n:
            _require(seen[i] == 1, String("Permutation must contain each axis exactly once"))
            i += 1
        var new_shape = List[Int]()
        var new_strides = List[Int]()
        i = 0
        while i < n:
            new_shape.append(self.shape[perm[i]])
            new_strides.append(self.strides[perm[i]])
            i += 1
        var init_val = self.data[self.offset]
        var view = Tensor[T](shape=new_shape, fill=init_val)
        view.data = self.data
        view.offset = self.offset
        view.strides = new_strides
        return view
fn squeeze(self) -> Tensor[T]:
        var keep_shape = List[Int]()
        var keep_strides = List[Int]()
        var i = 0
        while i < self.ndim():
            var d = self.shape[i]
            if d != 1:
                keep_shape.append(d)
                keep_strides.append(self.strides[i])
            i += 1
        if len(keep_shape) == 0:
            keep_shape.append(1)
            keep_strides.append(1)
        var init_val = self.data[self.offset]
        var view = Tensor[T](shape=keep_shape, fill=init_val)
        view.data = self.data
        view.offset = self.offset
        view.strides = keep_strides
        return view
fn unsqueeze(self, axis: Int) -> Tensor[T]:
        var n = self.ndim()
        _require(axis >= -n-1 and axis <= n, String("Axis out of range"))
        var ax = axis
        if ax < 0:
            ax = n + 1 + ax
        var new_shape = List[Int]()
        var i = 0
        while i < ax:
            new_shape.append(self.shape[i])
            i += 1
        new_shape.append(1)
        i = ax
        while i < n:
            new_shape.append(self.shape[i])
            i += 1
        var init_val = self.data[self.offset]
        var view = Tensor[T](shape=new_shape, fill=init_val)
        view.data = self.data
        view.offset = self.offset
        view.strides = compute_strides(new_shape)
        return view
fn to_contiguous(self) -> Tensor[T]:
        if self.is_contiguous():
            return self
        var init_val = self.data[self.offset]
        var out = Tensor[T](shape=self.shape, fill=init_val)
        var it = _FlatIter(self.shape, self.strides, self.offset)
        var lin: Int = 0
        while it.has_next():
            var pos = it.next()
            out.data[lin] = self.data[pos]
            lin += 1
        return out
fn map(self, f: fn(T) -> T) -> Tensor[T]:
        var init_val = self.data[self.offset]
        var out = Tensor[T](shape=self.shape, fill=init_val)
        var it = _FlatIter(self.shape, self.strides, self.offset)
        var lin: Int = 0
        while it.has_next():
            var pos = it.next()
            out.data[lin] = f(self.data[pos])
            lin += 1
        return out
fn apply_inplace(mut self, f: fn(T) -> T):
        var it = _FlatIter(self.shape, self.strides, self.offset)
        while it.has_next():
            var pos = it.next()
            self.data[pos] = f(self.data[pos])
fn zip_map(self, other: Tensor[T], f: fn(T, T) -> T) -> Tensor[T]:
        _require(self.ndim() == other.ndim(), String("Rank mismatch in zip_map"))
        _require(self.shape == other.shape, String("Shape mismatch in zip_map"))
        var init_val = self.data[self.offset]
        var out = Tensor[T](shape=self.shape, fill=init_val)
        var it_a = _FlatIter(self.shape, self.strides, self.offset)
        var it_b = _FlatIter(other.shape, other.strides, other.offset)
        var lin: Int = 0
        while it_a.has_next() and it_b.has_next():
            var pa = it_a.next()
            var pb = it_b.next()
            out.data[lin] = f(self.data[pa], other.data[pb])
            lin += 1
        return out
fn to_list(self) -> List[T]:
        var out = List[T]()
        out.reserve(self.size())
        var it = _FlatIter(self.shape, self.strides, self.offset)
        while it.has_next():
            var pos = it.next()
            out.append(self.data[pos])
        return out
fn deep_copy(self) -> Tensor[T]:
        var init_val = self.data[self.offset]
        var dup = Tensor[T](shape=self.shape, fill=init_val)
        dup.strides = self.strides
        dup.offset = 0
        dup.data = List[T]()
        var i = 0
        while i < len(self.data):
            dup.data.append(self.data[i])
            i += 1
        return dup

    @staticmethod
fn full(shape: List[Int], value: T) -> Tensor[T]:
        return Tensor[T](shape=shape, fill=value)

    @staticmethod
fn zeros(shape: List[Int], zero: T) -> Tensor[T]:
        return Tensor[T](shape=shape, fill=zero)

    @staticmethod
fn ones(shape: List[Int], one: T) -> Tensor[T]:
        return Tensor[T](shape=shape, fill=one)

struct _FlatIter:
    var shape: List[Int]
    var strides: List[Int]
    var offset: Int
    var idx: List[Int]
    var first: Bool
    var total: Int
    var produced: Int
    var pos: Int
fn __init__(out self, shape: List[Int], strides: List[Int], offset: Int) -> None:
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.idx = List[Int]()
        var i = 0
        while i < len(shape):
            self.idx.append(0)
            i += 1
        self.first = True
        self.total = _shape_product(shape)
        self.produced = 0
        self.pos = offset
fn has_next(self) -> Bool:
        return self.produced < self.total
fn next(mut self) -> Int:
        if self.first:
            self.first = False
            self.produced += 1
            return self.pos
        var n = len(self.shape)
        var ax = n - 1
        while ax >= 0:
            var dim = self.shape[ax]
            if dim == 0:
                dim = 1
            self.idx[ax] = self.idx[ax] + 1
            self.pos = self.pos + self.strides[ax]
            if self.idx[ax] < dim:
                break
            self.pos = self.pos - self.idx[ax] * self.strides[ax]
            self.idx[ax] = 0
            ax = ax - 1
        self.produced += 1
        return self.pos
fn __copyinit__(out self, other: Self) -> None:
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.idx = other.idx
        self.first = other.first
        self.total = other.total
        self.produced = other.produced
        self.pos = other.pos
fn __moveinit__(out self, deinit other: Self) -> None:
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.idx = other.idx
        self.first = other.first
        self.total = other.total
        self.produced = other.produced
        self.pos = other.pos