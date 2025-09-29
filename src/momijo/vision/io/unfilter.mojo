# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/unfilter.mojo
# Description: PNG scanline filter undo (types 0..4) in-place on packed rows.

# Integer abs helper (no extra imports)
fn iabs(x: Int) -> Int:
    if x < 0:
        return -x
    else:
        return x

# Paeth predictor: a=left, b=up, c=up-left
fn paeth_predictor(a: UInt8, b: UInt8, c: UInt8) -> UInt8:
    var ai = Int(a)
    var bi = Int(b)
    var ci = Int(c)
    var p  = ai + bi - ci
    var pa = iabs(p - ai)
    var pb = iabs(p - bi)
    var pc = iabs(p - ci)

    if pa <= pb and pa <= pc:
        return a
    else:
        if pb <= pc:
            return b
        else:
            return c

# Undo PNG per-row filters in-place.
# data layout: for each row: [filter_type (1 byte)] + [row_bytes payload],
# where row_bytes = width * bpp (bytes per pixel in the packed row).
#
# Notes:
# - Operates in-place on 'data'.
# - 'bpp' must be >= 1 (caller ensures correctness based on color type/bit depth).
fn unfilter_scanlines(data: UnsafePointer[UInt8], width: Int, height: Int, bpp: Int):
    # Basic guards
    if width <= 0 or height <= 0 or bpp <= 0:
        return

    var row_bytes = width * bpp
    var row = 0
    while row < height:
        var offset = row * (row_bytes + 1)
        var ftype = data[offset]
        var cur   = data + offset + 1

        # prev points to previous row payload (without the filter byte)
        var has_prev = row > 0
        var prev: UnsafePointer[UInt8]
        if has_prev:
            prev = data + ((row - 1) * (row_bytes + 1) + 1)
        else:
            prev = cur  # not used when has_prev == false

        # Filter types: 0(None), 1(Sub), 2(Up), 3(Average), 4(Paeth)
        if ftype == UInt8(0):
            # None (no operation)
            var _noop = 0
            _noop = _noop  # keep block non-empty

        else:
            if ftype == UInt8(1):
                # Sub: cur[i] += left(i)
                var i = 0
                while i < row_bytes:
                    var left = UInt8(0)
                    if i >= bpp:
                        left = cur[i - bpp]
                    cur[i] = UInt8((Int(cur[i]) + Int(left)) & 0xFF)
                    i = i + 1

            else:
                if ftype == UInt8(2):
                    # Up: cur[i] += prev[i]   (first row: nothing)
                    if has_prev:
                        var i2 = 0
                        while i2 < row_bytes:
                            cur[i2] = UInt8((Int(cur[i2]) + Int(prev[i2])) & 0xFF)
                            i2 = i2 + 1
                    else:
                        var _noop2 = 0
                        _noop2 = _noop2

                else:
                    if ftype == UInt8(3):
                        # Average: cur[i] += floor((left + up)/2)
                        var i3 = 0
                        while i3 < row_bytes:
                            var left3 = UInt8(0)
                            if i3 >= bpp:
                                left3 = cur[i3 - bpp]
                            var up3 = UInt8(0)
                            if has_prev:
                                up3 = prev[i3]
                            var avg = (Int(left3) + Int(up3)) // 2
                            cur[i3] = UInt8((Int(cur[i3]) + avg) & 0xFF)
                            i3 = i3 + 1

                    else:
                        if ftype == UInt8(4):
                            # Paeth: cur[i] += Paeth(left, up, up-left)
                            var i4 = 0
                            while i4 < row_bytes:
                                var a = UInt8(0)
                                if i4 >= bpp:
                                    a = cur[i4 - bpp]
                                var b = UInt8(0)
                                if has_prev:
                                    b = prev[i4]
                                var c = UInt8(0)
                                if has_prev and i4 >= bpp:
                                    c = prev[i4 - bpp]
                                var pred = paeth_predictor(a, b, c)
                                cur[i4] = UInt8((Int(cur[i4]) + Int(pred)) & 0xFF)
                                i4 = i4 + 1
                        else:
                            # Unknown filter: leave row as-is
                            var _noop3 = 0
                            _noop3 = _noop3

        row = row + 1
