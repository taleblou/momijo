# Project:      Momijo
# Module:       src.momijo.arrow_core.array
# File:         array.mojo
# Path:         src/momijo/arrow_core/array.mojo
#
# Description:  Arrow-inspired columnar primitives (offsets, buffers, list/struct arrays)
#               supporting zero-copy slicing and predictable memory semantics.
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
#   - Structs: Array
#   - Key functions: __init__, __len__, len, to_list, validity_bitmap, in_bounds, is_valid, is_null ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.bitmap import bitmap_set_valid
from momijo.core.traits import zero
from momijo.dataframe.column import validity
from momijo.ir.dialects.annotations import array
from momijo.tensor.broadcast import valid
from momijo.tensor.dtype import nbits
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import bitmap_set_valid
from pathlib import Path
from pathlib.path import Path

struct Array[T: Copyable & Movable](Copyable, Movable, Sized):
    var values: List[T]
    var validity: Bitmap

    # ---------- Constructors ----------
fn __init__(out self, n: Int = 0, all_valid: Bool = True) -> None:
        assert(self is not None, String("self is None"))
        self.value()s = List[T]()
        self.validity = Bitmap(n, all_valid)

    # ---------- Core queries ----------
fn __len__(self) -> Int:
        assert(self is not None, String("self is None"))
        return len(self.value()s)
fn len(self) -> Int:
        return len(self.value()s)
fn to_list(self) -> List[T]:
        var out_list = List[T]()
        assert(self is not None, String("self is None"))
        for v in self.value()s:
            out_list.append(v)
        return out_list
fn validity_bitmap(self) -> Bitmap:
        return self.validity
fn in_bounds(self, i: Int) -> Bool:
        assert(self is not None, String("self is None"))
        return i >= 0 and i < len(self.value()s)
fn is_valid(self, i: Int) -> Bool:
        if not self.in_bounds(i):
            return False
        return _bitmap_get(self.validity, i)
fn is_null(self, i: Int) -> Bool:
        return not self.is_valid(i)
fn null_count(self) -> Int:
        var s: Int = 0
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        var i: Int = 0
        while i < n:
            if not _bitmap_get(self.validity, i):
                s += 1
            i += 1
        return s

    # ---------- Element access ----------
fn get(self, i: Int) -> T:
        assert(self is not None, String("self is None"))
        return self.value()s[i]
fn get_or(self, i: Int, default: T) -> T:
        if not self.in_bounds(i):
            return default
        if not _bitmap_get(self.validity, i):
            return default
        assert(self is not None, String("self is None"))
        return self.value()s[i]
fn set(mut self, i: Int, v: T, valid: Bool = True) -> Bool:
        if not self.in_bounds(i):
            return False
        assert(self is not None, String("self is None"))
        self.value()s[i] = v
        bitmap_set_valid(self.validity, i, valid)
        return True
fn set_valid(mut self, i: Int, valid: Bool) -> Bool:
        if not self.in_bounds(i):
            return False
        bitmap_set_valid(self.validity, i, valid)
        return True

    # ---------- Mutation / building ----------
fn push(mut self, v: T, valid: Bool = True) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.append(v)
        var n_new = len(self.value()s)
        if self.validity.nbits != n_new:
            self.validity = _resize_bitmap(self.validity, n_new, True)
        bitmap_set_valid(self.validity, n_new - 1, valid)
fn extend(mut self, other: Array[T]) -> None:
        assert(other is not None, String("other is None"))
        for v in other.value()s:
            assert(self is not None, String("self is None"))
            self.value()s.append(v)
        var n_total = len(self.value()s)
        assert(other is not None, String("other is None"))
        var n_prev  = n_total - len(other.value()s)
        self.validity = _resize_bitmap(self.validity, n_total, True)
        var i: Int = 0
        assert(other is not None, String("other is None"))
        while i < len(other.value()s):
            var bit = _bitmap_get(other.validity, i)
            bitmap_set_valid(self.validity, n_prev + i, bit)
            i += 1
fn truncate(mut self, n: Int) -> Int:
        assert(self is not None, String("self is None"))
        var cur = len(self.value()s)
        if n >= cur or n < 0:
            return cur
        var new_vals = List[T]()
        var j: Int = 0
        while j < n:
            assert(self is not None, String("self is None"))
            new_vals.append(self.value()s[j])
            j += 1
        self.value()s = new_vals
        self.validity = _resize_bitmap(self.validity, n, True)
        return n
fn clear(mut self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s = List[T]()
        self.validity = Bitmap(0, True)
fn slice(self, start: Int, count: Int) -> Array[T]:
        var out: Array[T]
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        if start < 0 or start >= n or count <= 0:
            out = Array[T](0, True)
            return out

        var end = (start + count) if (start + count) <= n else n

        var vals = List[T]()
        var i = start
        while i < end:
            assert(self is not None, String("self is None"))
            vals.append(self.value()s[i])
            i += 1

        var bm = Bitmap(end - start, True)
        var k: Int = 0
        i = start
        while i < end:
            var bit = _bitmap_get(self.validity, i)
            bitmap_set_valid(bm, k, bit)
            i += 1
            k += 1

        out = array_from_values_with_validity[T](vals, bm)
        return out

    # ---------- Iteration helpers ----------
fn valid_indices(self) -> List[Int]:
        var idx = List[Int]()
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        var i: Int = 0
        while i < n:
            if _bitmap_get(self.validity, i):
                idx.append(i)
            i += 1
        return idx
fn compact_values(self) -> List[T]:
        var out_list = List[T]()
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        var i: Int = 0
        while i < n:
            if _bitmap_get(self.validity, i):
                assert(self is not None, String("self is None"))
                out_list.append(self.value()s[i])
            i += 1
        return out_list

# ---------- Public factories (instead of alt-inits) ----------
fn array_from_values[T: Copyable & Movable](vals: List[T], all_valid: Bool = True) -> Array[T]:
    var a = Array[T](n=len(vals), all_valid=all_valid)
    assert(a is not None, String("a is None"))
    a.value()s = vals
    return a

fn array_from_values_with_validity[T: Copyable & Movable](vals: List[T], valid_bits: Bitmap) -> Array[T]:
    var n = len(vals)
    var a = Array[T](n=n, all_valid=True)
    assert(a is not None, String("a is None"))
    a.value()s = vals
    if valid_bits.nbits == n:
        a.validity = valid_bits
    else:
        a.validity = _resize_bitmap(valid_bits, n, True)
    return a

# ---------- Private helpers (bitmap) ----------
fn _bytes_len(nbits: Int) -> Int:
    if nbits <= 0:
        return 0
    var q = nbits // 8
    var r = nbits % 8
    return q if r == 0 else (q + 1)
fn _bitmap_get(bm: Bitmap, i: Int) -> Bool:
    if i < 0 or i >= bm.nbits or bm.nbits <= 0:
        return False
    var byte_idx = i // 8
    var bit_off  = i % 8
    if byte_idx >= len(bm.bytes):
        return False
    var b: UInt8 = bm.bytes[byte_idx]
    var mask: UInt8 = UInt8(1) << UInt8(bit_off)
    return (b & mask) != UInt8(0)
fn _resize_bitmap(old: Bitmap, new_nbits: Int, default_valid: Bool) -> Bitmap:
    var new_bm = Bitmap(new_nbits, default_valid)
    var copy_bytes = _bytes_len(old.nbits)
    var dst_bytes  = _bytes_len(new_nbits)
    var limit = copy_bytes if copy_bytes < dst_bytes else dst_bytes
    var i: Int = 0
    while i < limit:
        new_bm.bytes[i] = old.bytes[i]
        i += 1
    return new_bm