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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/validity_mask.mojo

from momijo.arrow_core.bitmap import Bitmap

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] > best: best = xs[i]; idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] < best: best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# compute bytes needed to store nbits, using integer bit ops
fn _bytes_for_bits(nbits: Int) -> Int:
    # (nbits + 7) >> 3  == ceil(nbits/8) with integer math
    return (nbits + 7) >> 3

# ---------- module meta ----------
fn __module_name__() -> String:
    return String("momijo/arrow_core/validity_mask.mojo")
fn __self_test__() -> Bool:
    var m = ValidityMask.from_count(8, True)
    if not m.is_valid(0): return False
    if m.byte_len() != _bytes_for_bits(8): return False
    return True

# ---------- ValidityMask ----------
# Note: we keep it Movable (not ExplicitlyCopyable) to avoid copy requirements.
struct ValidityMask(Movable):
    var bm: Bitmap
    var nbits: Int
    var all_valid: Bool

    # Primary ctor (backward compatible)
fn __init__(out self, bm: Bitmap) -> None:
        self.bm = bm
        self.nbits = 0
        self.all_valid = True

    # Overloaded ctor (initialize all fields in one shot)
fn __init__(out self, bm: Bitmap, nbits: Int, all_valid: Bool) -> None:
        self.bm = bm
        self.nbits = nbits
        self.all_valid = all_valid

    # Convenience factory returning a fresh, fully-initialized instance.
    @staticmethod
fn from_count(nbits: Int, all_valid: Bool) -> Self:
        # Construct and return a temporary; this moves, no copy needed.
        return Self(Bitmap(nbits, all_valid), nbits, all_valid)
fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= self.nbits:
            return False
        # Without per-bit access on Bitmap, approximate via all_valid.
        return self.all_valid
fn byte_len(self) -> Int:
        return _bytes_for_bits(self.nbits)