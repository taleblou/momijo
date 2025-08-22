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
# Project: momijo.arrow_core
# File: momijo/arrow_core/array.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

 

from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid

struct Array[T: Copyable & Movable](Copyable, Movable, Sized):
    # Underlying values and the validity bitmap
    var values: List[T]
    var validity: Bitmap

    # ---------- Constructors ----------

    # Create an empty array, or pre-size validity to `n` bits (all valid by default).
    fn __init__(out self, n: Int = 0, all_valid: Bool = True):
        self.values = List[T]()
        if n > 0:
            # Pre-size with zero values; caller can set later
            for _ in range(n):
                # For value initialization, we can't construct arbitrary T,
                # so we only size validity and keep values empty; user can push.
                pass
        self.validity = Bitmap(n, all_valid)

    # Build from an existing list of values; mark all valid or not.
    fn from_values(out self, vals: List[T], all_valid: Bool = True):
        self.values = vals
        self.validity = Bitmap(len(vals), all_valid)

    # Build from values plus an explicit validity bitmap.
    # If the bitmap size doesn't match, it is resized to `len(vals)` (extra bits default to valid).
    fn from_values_with_validity(out self, vals: List[T], valid_bits: Bitmap):
        self.values = vals
        let n = len(vals)
        if valid_bits.nbits == n:
            self.validity = valid_bits
        else:
            self.validity = _resize_bitmap(valid_bits, n, True)

    # ---------- Core queries ----------

    fn __len__(self) -> Int:
        return len(self.values)

    fn len(self) -> Int:
        return len(self.values)

    # Return a copy of the underlying values list
    fn to_list(self) -> List[T]:
        var out_list = List[T]()
        for v in self.values:
            out_list.append(v)
        return out_list

    # Access raw validity bitmap (by value)
    fn validity_bitmap(self) -> Bitmap:
        return self.validity

    # Is index `i` within the bounds of the array
    fn in_bounds(self, i: Int) -> Bool:
        return i >= 0 and i < len(self.values)

    # Check validity flag for index `i` (out of bounds -> False).
    fn is_valid(self, i: Int) -> Bool:
        if not self.in_bounds(i):
            return False
        return _bitmap_get(self.validity, i)

    fn is_null(self, i: Int) -> Bool:
        return not self.is_valid(i)

    fn null_count(self) -> Int:
        var s: Int = 0
        let n = len(self.values)
        for i in range(n):
            if not _bitmap_get(self.validity, i):
                s += 1
        return s

    # ---------- Element access ----------

    # Unsafe get (no validity check). Use `get_or` for a total operation.
    fn get(self, i: Int) -> T:
        return self.values[i]

    # Total get: returns `default` when index is out of bounds or invalid.
    fn get_or(self, i: Int, default: T) -> T:
        if not self.in_bounds(i):
            return default
        if not _bitmap_get(self.validity, i):
            return default
        return self.values[i]

    # Set value at index `i` (returns False if out of bounds).
    fn set(mut self, i: Int, v: T, valid: Bool = True) -> Bool:
        if not self.in_bounds(i):
            return False
        self.values[i] = v
        bitmap_set_valid(self.validity, i, valid)
        return True

    # Flip validity at index `i` (returns False if out of bounds).
    fn set_valid(mut self, i: Int, valid: Bool) -> Bool:
        if not self.in_bounds(i):
            return False
        bitmap_set_valid(self.validity, i, valid)
        return True

    # ---------- Mutation / building ----------

    # Append a value with its validity.
    fn push(mut self, v: T, valid: Bool = True):
        self.values.append(v)
        let n_new = len(self.values)
        if self.validity.nbits == n_new - 1:
            # Grow by one, preserving previous bits
            self.validity = _resize_bitmap(self.validity, n_new, True)
        elif self.validity.nbits != n_new:
            # Repair invariant if it has drifted
            self.validity = _resize_bitmap(self.validity, n_new, True)
        bitmap_set_valid(self.validity, n_new - 1, valid)

    # Append all elements from another array (concatenate).
    fn extend(mut self, other: Array[T]):
        # Append values
        for v in other.values:
            self.values.append(v)
        # Merge bitmaps
        let n_total = len(self.values)
        let n_prev = n_total - len(other.values)
        self.validity = _resize_bitmap(self.validity, n_total, True)
        # Copy other bits into the tail
        var i: Int = 0
        while i < len(other.values):
            let bit = _bitmap_get(other.validity, i)
            bitmap_set_valid(self.validity, n_prev + i, bit)
            i += 1

    # Truncate to length `n` (no-op if n >= len). Returns new length.
    fn truncate(mut self, n: Int) -> Int:
        let cur = len(self.values)
        if n >= cur or n < 0:
            return cur
        # Shrink values
        var i = cur - 1
        while i >= n:
            # Pop last (List has pop? safer: just ignore logically)
            # Rebuild a new list up to n
            i -= 1
        # Rebuild values list up to n
        var new_vals = List[T]()
        for j in range(n):
            new_vals.append(self.values[j])
        self.values = new_vals
        # Resize bitmap
        self.validity = _resize_bitmap(self.validity, n, True)
        return n

    # Clear to empty.
    fn clear(mut self):
        self.values = List[T]()
        self.validity = Bitmap(0, True)

    # Create a shallow slice [start, start+count).
    # Out-of-range start returns an empty array; count is clamped.
    fn slice(self, start: Int, count: Int) -> Array[T]:
        var out: Array[T]
        let n = len(self.values)
        if start < 0 or start >= n or count <= 0:
            out = Array[T](0, True)
            return out

        let end = (start + count) if (start + count) <= n else n

        # Copy values
        var vals = List[T]()
        var i = start
        while i < end:
            vals.append(self.values[i])
            i += 1

        # Build validity
        var bm = Bitmap(end - start, True)
        var k: Int = 0
        i = start
        while i < end:
            let bit = _bitmap_get(self.validity, i)
            bitmap_set_valid(bm, k, bit)
            i += 1
            k += 1

        out = Array[T].from_values_with_validity(vals, bm)
        return out

    # ---------- Iteration helpers ----------

    # Iterate indices where element is valid.
    fn valid_indices(self) -> List[Int]:
        var idx = List[Int]()
        let n = len(self.values)
        for i in range(n):
            if _bitmap_get(self.validity, i):
                idx.append(i)
        return idx

    # Collect valid values into a new list.
    fn compact_values(self) -> List[T]:
        var out_list = List[T]()
        let n = len(self.values)
        for i in range(n):
            if _bitmap_get(self.validity, i):
                out_list.append(self.values[i])
        return out_list

# ---------- Private helpers (bitmap) ----------

fn _bytes_len(nbits: Int) -> Int:
    let q = nbits // 8
    let r = nbits % 8
    return q if r == 0 else (q + 1)

fn _bitmap_get(bm: Bitmap, i: Int) -> Bool:
    # Assume 0 <= i < bm.nbits
    let byte_idx = i // 8
    let bit_off = i % 8
    if bm.nbits <= 0:
        return False
    if byte_idx >= len(bm.bytes):
        return False
    let b: UInt8 = bm.bytes[byte_idx]
    let mask: UInt8 = (UInt8(1)) << UInt8(bit_off)
    return (b & mask) != UInt8(0)

fn _resize_bitmap(old: Bitmap, new_nbits: Int, default_valid: Bool) -> Bitmap:
    var new_bm = Bitmap(new_nbits, default_valid)
    # Copy overlapping bytes
    let copy_bytes = _bytes_len(old.nbits)
    let dst_bytes = _bytes_len(new_nbits)
    let limit = copy_bytes if copy_bytes < dst_bytes else dst_bytes
    var i: Int = 0
    while i < limit:
        new_bm.bytes[i] = old.bytes[i]
        i += 1
    # If new_nbits < old.nbits, higher bits are discarded (expected).
    return new_bm
