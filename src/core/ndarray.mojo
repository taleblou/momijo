# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/ndarray.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

# A minimal NDArray owning its storage. Views are provided via NDView.
# This version focuses on Float64 for math kernels to keep things simple.

from momijo.core.error import MomijoError
from momijo.core.result import Result
from builtin.dtype import DType
from momijo.core.shape import Shape, Strides, Slice

# -------------------------
# Helpers
# -------------------------
# Shape moved to momijo.core.shape
    fn __init__(out self, dims: List[Int] = List[Int]()):
        self.dims = dims

    fn rank(self) -> Int: return len(self.dims)

    fn count(self) -> Int:
        var n = 1
        var i = 0
        if len(self.dims) == 0: return 0
        while i < len(self.dims):
            n = n * self.dims[i]
            i += 1
        return n

    fn to_string(self) -> String:
        var s = "["
        var i = 0
        while i < len(self.dims):
            if i > 0: s = s + ", "
            s = s + String(self.dims[i])
            i += 1
        s = s + "]"
        return s
# Strides moved to momijo.core.shape
    fn __init__(out self, steps: List[Int] = List[Int]()):
        self.steps = steps

    fn to_string(self) -> String:
        var s = "["
        var i = 0
        while i < len(self.steps):
            if i > 0: s = s + ", "
            s = s + String(self.steps[i])
            i += 1
        s = s + "]"
        return s


@staticmethod
fn compute_row_major_strides(shape: Shape) -> Strides:
    var r = len(shape.dims)
    var steps = List[Int]()
    if r == 0:
        return Strides(steps=steps)
    steps.reserve(r)
    # last axis stride = 1
    var i = 0
    while i < r:
        steps.append(0)
        i += 1
    var stride = 1
    var k = r - 1
    while k >= 0:
        steps[k] = stride
        stride = stride * shape.dims[k]
        k -= 1
    return Strides(steps=steps)


@staticmethod
fn product(xs: List[Int]) -> Int:
    var n = 1
    var i = 0
    if len(xs) == 0: return 0
    while i < len(xs):
        n = n * xs[i]
        i += 1
    return n


@staticmethod
fn check_shape_nonnegative(shape: Shape) -> Bool:
    var i = 0
    while i < len(shape.dims):
        if shape.dims[i] < 0: return False
        i += 1
    return True


@staticmethod
fn same_shape(a: Shape, b: Shape) -> Bool:
    if len(a.dims) != len(b.dims): return False
    var i = 0
    while i < len(a.dims):
        if a.dims[i] != b.dims[i]: return False
        i += 1
    return True


# -------------------------
# Slicing
# -------------------------
# Slice moved to momijo.core.shape

    fn __init__(out self, start: Int = 0, stop: Int = -1, step: Int = 1):
        self.start = start
        self.stop = stop
        self.step = (step <= 0) ? 1 : step


# Normalize a single slice for a given axis length; returns (start, stop, step, new_len)
@staticmethod
fn normalize_slice(s: Slice, axis_len: Int) -> (Int, Int, Int, Int):
    var start = s.start
    var stop = s.stop
    var step = (s.step <= 0) ? 1 : s.step
    if start < 0: start = 0
    if stop < 0 or stop > axis_len: stop = axis_len
    if start > axis_len: start = axis_len
    if stop < start: stop = start
    var new_len = (stop - start + step - 1) // step
    return (start, stop, step, new_len)


# -------------------------
# NDArray
# -------------------------

struct NDArray[T](Copyable, Movable, EqualityComparable):
    var _data: List[T]
    var _shape: Shape
    var _strides: Strides
    var _offset: Int
    var _dtype: DType

    fn __init__(
        out self,
        data: List[T],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        dtype: DType = DType.bool()    # default placeholder; not enforced
    ):
        self._data = data
        self._shape = shape
        self._strides = strides
        self._offset = offset
        self._dtype = dtype

    # ----------- basic info -----------
    fn ndim(self) -> Int: return self._shape.rank()
    fn shape(self) -> Shape: return self._shape
    fn strides(self) -> Strides: return self._strides
    fn offset(self) -> Int: return self._offset
    fn dtype(self) -> DType: return self._dtype

    fn size(self) -> Int:
        return self._shape.count()

    fn is_contiguous(self) -> Bool:
        return same_shape(self._strides_to_shape(), compute_row_major_strides(self._shape))

    fn _strides_to_shape(self) -> Strides:
        # For comparison purposes only; returns strides as-is
        return Strides(steps=self._strides.steps)

    fn to_string(self) -> String:
        return "NDArray{shape=" + self._shape.to_string()
            + ", strides=" + self._strides.to_string()
            + ", offset=" + String(self._offset)
            + ", dtype=" + self._dtype.to_string()
            + "}"

    # ----------- factories -----------
    @staticmethod
    fn zeros(shape: Shape, value: T) -> NDArray[T]:
        # 'value' is used to seed element type; will be set to zero via "value - value"
        var n = shape.count()
        var data = List[T]()
        data.reserve(n)
        var zero = value - value
        var i = 0
        while i < n:
            data.append(zero)
            i += 1
        return NDArray[T](data=data, shape=shape, strides=compute_row_major_strides(shape), offset=0)

    @staticmethod
    fn full(shape: Shape, value: T) -> NDArray[T]:
        var n = shape.count()
        var data = List[T]()
        data.reserve(n)
        var i = 0
        while i < n:
            data.append(value)
            i += 1
        return NDArray[T](data=data, shape=shape, strides=compute_row_major_strides(shape), offset=0)

    @staticmethod
    fn try_from_list(shape: Shape, data: List[T]) -> Result[NDArray[T]]:
        if not check_shape_nonnegative(shape):
            return Result[NDArray[T]].fail(MomijoError.invalid_argument("negative dimension in shape", "momijo.core.ndarray"), NDArray[T].full(Shape(dims=[1]), data[0]))
        var need = shape.count()
        if need != len(data):
            return Result[NDArray[T]].fail(MomijoError.invalid_argument("data length does not match shape", "momijo.core.ndarray"), NDArray[T].full(Shape(dims=[1]), data[0]))
        return Result[NDArray[T]].ok(NDArray[T](data=data, shape=shape, strides=compute_row_major_strides(shape), offset=0))

    # ----------- indexing -----------
    fn _linear_offset(self, index: List[Int]) -> Int:
        var i = 0
        var off = self._offset
        while i < len(index):
            off = off + index[i] * self._strides.steps[i]
            i += 1
        return off

    fn _check_bounds(self, index: List[Int]) -> Bool:
        if len(index) != len(self._shape.dims): return False
        var i = 0
        while i < len(index):
            var v = index[i]
            if v < 0: return False
            if v >= self._shape.dims[i]: return False
            i += 1
        return True

    fn get(self, index: List[Int]) -> T:
        # Caller responsible for bounds correctness for speed.
        var off = self._linear_offset(index)
        return self._data[off]

    fn set(self, index: List[Int], value: T) -> None:
        var off = self._linear_offset(index)
        self._data[off] = value

    # ----------- reshape/transpose -----------
    fn reshape(self, new_shape: Shape) -> NDArray[T]:
        # Requires contiguous layout.
        var n_old = self.size()
        var n_new = new_shape.count()
        if n_old != n_new:
            # keep original if mismatch
            return self
        var contig = NDArray[T](data=self._data, shape=new_shape, strides=compute_row_major_strides(new_shape), offset=self._offset, dtype=self._dtype)
        return contig

    fn transpose(self, perm: List[Int]) -> NDArray[T]:
        # Permute axes; returns a view (strides+shape permuted).
        var r = self.ndim()
        if len(perm) != r:
            return self
        var new_shape = List[Int]()
        var new_strides = List[Int]()
        var i = 0
        while i < r:
            var ax = perm[i]
            new_shape.append(self._shape.dims[ax])
            new_strides.append(self._strides.steps[ax])
            i += 1
        return NDArray[T](data=self._data, shape=Shape(dims=new_shape), strides=Strides(steps=new_strides), offset=self._offset, dtype=self._dtype)

    # ----------- slicing (positive step only) -----------
    fn slice(self, specs: List[Slice]) -> NDArray[T]:
        var r = self.ndim()
        if len(specs) != r:
            return self
        var new_dims = List[Int]()
        var new_strides = List[Int]()
        var new_offset = self._offset
        var i = 0
        while i < r:
            var (start, stop, step, new_len) = normalize_slice(specs[i], self._shape.dims[i])
            new_offset = new_offset + start * self._strides.steps[i]
            new_dims.append(new_len)
            new_strides.append(self._strides.steps[i] * step)
            i += 1
        return NDArray[T](data=self._data, shape=Shape(dims=new_dims), strides=Strides(steps=new_strides), offset=new_offset, dtype=self._dtype)

    # ----------- iteration -----------
    fn iter_flat(self) -> List[T]:
        # Returns a flat copy (contiguous order) for simplicity.
        var out = List[T]()
        out.reserve(self.size())
        var idx = List[Int]()
        var r = self.ndim()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1

        if r == 0:
            return out

        # Lexicographic iteration
        while True:
            out.append(self.get(idx))
            # increment idx
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return out

    # ----------- element-wise ops -----------
    fn fill(self, value: T) -> None:
        var n = self.size()
        var flat = self.iter_flat()
        var i = 0
        # write back in logical order
        var idx = List[Int]()
        var r = self.ndim()
        idx.reserve(r)
        var j = 0
        while j < r:
            idx.append(0)
            j += 1
        if r == 0: return
        var k = 0
        while True:
            self.set(idx, value)
            k += 1
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0 or k >= n:
                break

    fn map_inplace(self, f: fn(T) -> T) -> None:
        var r = self.ndim()
        if r == 0: return
        var idx = List[Int]()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        while True:
            var v = self.get(idx)
            self.set(idx, f(v))
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break

    fn map[U](self, f: fn(T) -> U) -> NDArray[U]:
        var out = NDArray[U].full(self._shape, f(self.get(_zero_index(self.ndim()))))
        # iterate and set
        var r = self.ndim()
        if r == 0: return out
        var idx = List[Int]()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        while True:
            out.set(idx, f(self.get(idx)))
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return out

    fn zip_with[U, R](self, other: NDArray[U], f: fn(T, U) -> R) -> NDArray[R]:
        if not same_shape(self._shape, other._shape):
            return NDArray[R].full(self._shape, f(self.get(_zero_index(self.ndim())), f(self.get(_zero_index(self.ndim())), self.get(_zero_index(self.ndim())))))
        var out = NDArray[R].full(self._shape, f(self.get(_zero_index(self.ndim())), other.get(_zero_index(self.ndim()))))
        var r = self.ndim()
        var idx = List[Int]()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        while True:
            var a = self.get(idx)
            var b = other.get(idx)
            out.set(idx, f(a, b))
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return out

    # ----------- reductions -----------
    fn sum(self, init: T) -> T:
        var acc = init
        var r = self.ndim()
        if r == 0: return acc
        var idx = List[Int]()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        while True:
            acc = acc + self.get(idx)
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return acc

    fn dot(self, other: NDArray[T], init: T) -> T:
        # Flatten both; shapes must match.
        if not same_shape(self._shape, other._shape):
            return init
        var acc = init
        var r = self.ndim()
        if r == 0: return acc
        var idx = List[Int]()
        idx.reserve(r)
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        while True:
            acc = acc + (self.get(idx) * other.get(idx))
            var d = r - 1
            while d >= 0:
                idx[d] = idx[d] + 1
                if idx[d] < self._shape.dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return acc

    # ----------- copy -----------
    fn copy(self) -> NDArray[T]:
        var n = self.size()
        var flat = self.iter_flat()
        # build contiguous copy
        var data = List[T]()
        data.reserve(n)
        var i = 0
        while i < n:
            data.append(flat[i])
            i += 1
        return NDArray[T](data=data, shape=self._shape, strides=compute_row_major_strides(self._shape), offset=0, dtype=self._dtype)


# helpers

@staticmethod
fn _zero_index(r: Int) -> List[Int]:
    var idx = List[Int]()
    var i = 0
    while i < r:
        idx.append(0)
        i += 1
    return idx
