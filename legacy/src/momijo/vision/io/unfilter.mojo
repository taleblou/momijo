# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/unfilter.mojo
# Description: PNG scanline unfilter types 0..4 (None, Sub, Up, Average, Paeth).

from collections.list import List

@always_inline
fn _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = (p - a); 
    if pa < 0: pa = -pa
    var pb = (p - b); 
    if pb < 0: pb = -pb
    var pc = (p - c); 
    if pc < 0: pc = -pc
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c



@always_inline
fn _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b
 


# Unfilter in-place-like into new buffer.
# width: pixels, bpp: bytes per pixel (3 for RGB, 4 for RGBA)
# New full signature
fn png_unfilter(width: Int, height: Int, channels: Int, data: List[UInt8], bit_depth: Int) -> (Bool, List[UInt8]):
    if width <= 0 or height <= 0: return (False, List[UInt8]())
    if channels <= 0: return (False, List[UInt8]())
    if not (bit_depth == 1 or bit_depth == 2 or bit_depth == 4 or bit_depth == 8 or bit_depth == 16):
        return (False, List[UInt8]())

    var bits_per_pixel = channels * bit_depth
    var bpp = _ceil_div(bits_per_pixel, 8)
    var row_payload = _ceil_div(width * bits_per_pixel, 8)
    var expected = height * (1 + row_payload)
    if len(data) < expected: return (False, List[UInt8]())

    var out = List[UInt8](); out.reserve(height * row_payload)
    var prev = List[UInt8](); prev.reserve(row_payload)
    var pos = 0; var y = 0
    while y < height:
        var ftype = Int(data[pos]); pos += 1
        if ftype < 0 or ftype > 4: return (False, List[UInt8]())
        var raw = List[UInt8](); raw.reserve(row_payload)
        var x = 0
        while x < row_payload:
            var cur = Int(data[pos + x])
            var left = 0; var up = 0; var ul = 0
            if x >= bpp: left = Int(raw[x - bpp])
            if len(prev) > 0:
                up = Int(prev[x])
                if x >= bpp: ul = Int(prev[x - bpp])
            var val = 0
            if ftype == 0:      val = cur
            elif ftype == 1:    val = (cur + left) & 255
            elif ftype == 2:    val = (cur + up) & 255
            elif ftype == 3:    val = (cur + ((left + up) >> 1)) & 255
            else:               val = (cur + _paeth(left, up, ul)) & 255
            raw.append(UInt8(val))
            x += 1
        pos += row_payload
        var i = 0; 
        while i < len(raw): out.append(raw[i]); i += 1
        prev = raw.copy()
        y += 1

    return (True, out.copy())

# Backwards compatible overload (assumes 8-bit samples)
fn png_unfilter(width: Int, height: Int, channels: Int, data: List[UInt8]) -> (Bool, List[UInt8]):
    return png_unfilter(width, height, channels, data, 8)