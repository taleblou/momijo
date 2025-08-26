# MIT License
# Project: momijo.arrow_core
# File: momijo/arrow_core/array_base.mojo

from momijo.arrow_core.bitmap import Bitmap
from momijo.arrow_core.dtype_arrow import DataType

# A minimal base header for Arrow-like arrays:
# - length
# - dtype
# - validity bitmap (all-valid allowed)
#
# NOTE:
# We store Bitmap directly to avoid ownership/copy semantics of a wrapper
# type. Bitmap in this codebase is copyable/movable, so the struct stays simple.

struct ArrayBase(Copyable, Movable, Sized):
    var length: Int
    var dtype: DataType
    var validity: Bitmap

    fn __init__(out self, length: Int, dtype: DataType, validity: Bitmap):
        self.length = length
        self.dtype = dtype
        self.validity = validity

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    fn len(self) -> Int:
        return self.length


# -------- small module helpers (kept from your scaffold) --------

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

fn __module_name__() -> String:
    return String("momijo/arrow_core/array_base.mojo")

fn __self_test__() -> Bool:
    # cheap smoke test; extend later
    return True
