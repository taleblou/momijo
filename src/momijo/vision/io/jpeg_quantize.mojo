# MIT License
# Copyright (c) 2025 Morteza
# Project: momijo
# SPDX-License-Identifier: MIT
# File: src/momijo/vision/io/jpeg_quantize.mojo
# Description: Quantize and zigzag utilities for JPEG encoder (no globals, var-only).

from collections.list import List

# -------- helpers --------
@always_inline
fn _imax(a: Int, b: Int) -> Int:
    return a if a >= b else b

# Symmetric rounding to nearest for integer division by a positive q.
# q >= 1 is enforced. Works the same for positive/negative 'val'.
@always_inline
fn _qdiv(val: Int, q8: UInt8) -> Int:
    var q = _imax(Int(q8), 1)
    if val >= 0:
        # +q//2 then floor-div
        return (val + (q // 2)) // q
    else:
        # symmetric rounding for negatives: -((|val| + q//2) // q)
        return -(((-val) + (q // 2)) // q)

# -------- API: quantize one 8x8 block (natural order in/out) --------
# Quantize one 8x8 DCT block using an 8-bit quantization table.
# dct  : pointer to 64 Int coefficients (input, natural order)
# quant: pointer to 64 UInt8 quantizers   (input, natural order)
# dst  : pointer to 64 Int quantized coeffs (output, natural order)
@always_inline
fn quantize_block(dct: UnsafePointer[Int],
                  quant: UnsafePointer[UInt8],
                  dst: UnsafePointer[Int]):
    var i = 0
    while i < 64:
        dst[i] = _qdiv(dct[i], quant[i])
        i += 1

# -------- Zigzag mapping --------
# IMPORTANT:
# The table below maps: zigzag index 'z' -> natural index 'n'.
# That is, n = ZIGZAG[z]. This is the form used by most JPEG code paths.
# If you need the inverse mapping (natural -> zigzag), use inverse builder below.

@always_inline
fn _zigzag_table_noalloc() -> List[Int]:
    # Local constant list (rebuilt per call, no globals per project rules).
    # Z[z] = natural-index
    var table:List[Int]=([
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ])
    return table.copy()

# Copy natural -> zigzag:
# src : 64 Int (natural order)
# dst : 64 Int (zigzag order)
@always_inline
fn zigzag_copy(src: UnsafePointer[Int], dst: UnsafePointer[Int]):
    # Using Z[z] = natural-index; we want dst[z] = src[natural]
    var Z = _zigzag_table_noalloc()
    var z = 0
    while z < 64:
        var n = Z[z]
        dst[z] = src[n]
        z += 1

# Copy zigzag -> natural (inverse):
# src : 64 Int (zigzag order)
# dst : 64 Int (natural order)
@always_inline
fn inv_zigzag_copy(src: UnsafePointer[Int], dst: UnsafePointer[Int]):
    # Build inverse mapping on the fly (no globals).
    var Z = _zigzag_table_noalloc()     # Z[z] = n
    # Compute inv so that inv[n] = z
    var inv = List[Int]()
    var i = 0
    while i < 64:
        inv.append(0)
        i += 1
    var z = 0
    while z < 64:
        inv[Z[z]] = z
        z += 1
    # Apply: dst[n] = src[inv[n]]
    var n = 0
    while n < 64:
        dst[n] = src[inv[n]]
        n += 1
