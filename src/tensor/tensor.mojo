# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Minimal Tensor (ndarray-style) with row-major strides and broadcasting basics

from math import max as max_fn, min as min_fn

struct Tensor[T: Copyable & Movable](Copyable, Movable):
    var shape: List[Int]
    var strides: List[Int]
    var data: List[T]
    var offset: Int

    fn __init__(out self, shape: List[Int], fill: T = T()):
        # Validate shape
        var size: Int = 1
        for d in shape:
            assert(d >= 0, "Negative dimension in shape")
            size *= (1 if d == 0 else d)
        self.shape = shape
        self.strides = compute_strides(shape)
        self.data = List[T](repeating: fill, count: size)
        self.offset = 0

    fn from_list(out self, shape: List[Int], values: List[T]):
        var size: Int = 1
        for d in shape:
            size *= (1 if d == 0 else d)
        assert(size == len(values), "Values length does not match shape product")
        self.shape = shape
        self.strides = compute_strides(shape)
        self.data = values
        self.offset = 0

    fn ndim(self) -> Int:
        return len(self.shape)

    fn size(self) -> Int:
        var s: Int = 1
        for d in self.shape:
            s *= (1 if d == 0 else d)
        return s

    fn is_contiguous(self) -> Bool:
        return self.strides == compute_strides(self.shape)

    fn index(self, idx: List[Int]) -> Int:
        assert(len(idx) == len(self.shape), "Index rank mismatch")
        var flat: Int = self.offset
        for i in range(0, len(idx)):
            let ii = idx[i]
            let d = self.shape[i]
            assert(ii >= 0 and ii < d, "Index out of bounds")
            flat += ii * self.strides[i]
        return flat

    fn get(self, idx: List[Int]) -> T:
        return self.data[self.index(idx)]

    fn set(mut self, idx: List[Int], value: T):
        let pos = self.index(idx)
        self.data[pos] = value

    fn reshape(self, new_shape: List[Int]) -> Tensor[T]:
        # Does not copy; only checks total size
        var old: Int = self.size()
        var newp: Int = 1
        for d in new_shape:
            newp *= (1 if d == 0 else d)
        assert(old == newp, "Total size must remain constant in reshape")
        var out = Tensor[T](shape=new_shape)
        out.data = self.data
        out.offset = self.offset
        if self.is_contiguous():
            out.strides = compute_strides(new_shape)
        else:
            # Not strictly correct; for simplicity we keep same strides. Advanced reshape needs view analysis.
            out.strides = compute_strides(new_shape)
        return out

    fn slice(self, starts: List[Int], stops: List[Int], steps: List[Int]) -> Tensor[T]:
        # Basic slicing view (step=1 only enforced for simplicity)
        assert(len(starts) == len(stops) and len(starts) == len(self.shape), "Slice rank mismatch")
        var new_shape = List[Int]()
        var new_offset = self.offset
        var new_strides = self.strides
        for i in range(0, len(self.shape)):
            let s = starts[i]
            let e = stops[i]
            assert(steps[i] == 1, "Only step=1 supported in minimal slice")
            assert(s >= 0 and e <= self.shape[i] and s <= e, "Invalid slice bounds")
            new_shape.append(e - s)
            new_offset += s * self.strides[i]
        var view = Tensor[T](shape=new_shape)
        view.data = self.data
        view.offset = new_offset
        view.strides = new_strides
        return view

    fn map(mut self, f: fn(T) -> T) -> Tensor[T]:
        var out = Tensor[T](shape=self.shape)
        for i in range(0, self.size()):
            out.data[i] = f(self.data[self.offset + i])
        return out

fn compute_strides(shape: List[Int]) -> List[Int]:
    # Row-major
    var n = len(shape)
    var st = List[Int](repeating: 0, count: n)
    var acc: Int = 1
    for i in range(0, n):
        let j = n - 1 - i
        st[j] = acc
        let d = shape[j]
        acc *= (1 if d == 0 else d)
    return st
