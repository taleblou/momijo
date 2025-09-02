# MIT License
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor
# File: momijo/tensor/iterator.mojo

fn __module_name__() -> String:
    return String("momijo/tensor/iterator.mojo")

fn __self_test__() -> Bool:
    # Lightweight smoke test – extend with real checks later.
    return True

# ---- small helpers (no external deps) ---------------------------------------

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

# ---- minimal tensor-shape/view shims (to avoid external deps) ---------------

struct Shape(Copyable, Movable):
    var dims: List[Int]

    fn __init__(out self, dims: List[Int]):
        self.dims = dims

    fn ndim(self) -> Int:
        return len(self.dims)

struct LayoutView(Copyable, Movable):
    var offset: Int
    var strides: List[Int]

    fn __init__(out self, offset: Int, strides: List[Int]):
        self.offset = offset
        self.strides = strides

# ---- ND iterator over an N-D shape ------------------------------------------

struct NDIter(Copyable, Movable):
    var shape: Shape
    var idx: List[Int]
    var started: Bool

    fn __init__(out self, shape: Shape):
        self.shape = shape
        self.idx = List[Int]()
        var k = 0
        var r = shape.ndim()
        while k < r:
            self.idx.append(0)
            k += 1
        self.started = False

    # Returns (has_next, current_index)
    fn next(mut self) -> (Bool, List[Int]):
        var r = self.shape.ndim()
        if r == 0:
            return (False, self.idx)

        if not self.started:
            self.started = True
            return (True, self.idx)

        # advance in row-major order
        var ax = r - 1
        while ax >= 0:
            self.idx[ax] = self.idx[ax] + 1
            if self.idx[ax] < self.shape.dims[ax]:
                return (True, self.idx)
            self.idx[ax] = 0
            ax = ax - 1

        # overflowed
        return (False, self.idx)

# free-function wrapper (some examples import this symbol)
fn NDIter_next(mut x: NDIter) -> (Bool, List[Int]):
    return x.next()

# ---- linear index utility (row-major) ---------------------------------------

fn linear_index(view: LayoutView, idx: List[Int]) -> Int:
    var s = view.offset
    var i = 0
    var n = len(idx)
    while i < n:
        s = s + idx[i] * view.strides[i]
        i += 1
    return s
