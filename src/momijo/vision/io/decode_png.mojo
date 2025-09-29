# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/decode_png.mojo

from momijo.vision.io.png_decoder import read_be_u32, parse_ihdr, is_png_magic
from momijo.vision.io.inflate import inflate
from momijo.vision.io.unfilter import unfilter_scanlines
from momijo.vision.tensor import Tensor
from momijo.vision.image import Image, ImageMeta, ColorSpace
from momijo.vision.transforms.array import full

# ---- helper: build a 4-char string from bytes ----
fn _fourcc(a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> String: 
    var s = bytes_to_string4(a, b, c, d)
    return s

# ---- helper: minimal fallback image (1x1 RGB black) ----
fn _fallback_image() -> Image:
    var t = full((1, 1, 3), UInt8(0)).tensor()   # packed HWC u8
    var m = ImageMeta().with_colorspace(ColorSpace.SRGB())
    return Image(t, m)

fn decode_png(ptr: UnsafePointer[UInt8], length: Int) -> (Bool, Image):
    if length < 8:
        return (False, _fallback_image())
    if not is_png_magic(ptr):
        return (False, _fallback_image())

    var end_ptr = ptr + length
    var cursor = ptr + 8

    var width: Int = 0
    var height: Int = 0
    var color_type: UInt8 = UInt8(0)
    var idat_total: Int = 0

    # First pass
    while (cursor + 12) <= end_ptr:
        var len_u32 = read_be_u32(cursor)
        var data_len = Int(len_u32)

        var t0 = (cursor + 4)[0]
        var t1 = (cursor + 4)[1]
        var t2 = (cursor + 4)[2]
        var t3 = (cursor + 4)[3]
        var typ = _fourcc(t0, t1, t2, t3)

        var data_ptr = cursor + 8

        if typ == "IHDR":
            var hdr = parse_ihdr(data_ptr)
            width = hdr.width
            height = hdr.height
            color_type = hdr.color_type
        elif typ == "IDAT":
            idat_total = idat_total + data_len
        elif typ == "IEND":
            break

        var advance = 12 + data_len
        cursor = cursor + advance

    if width <= 0 or height <= 0 or idat_total <= 0:
        return (False, _fallback_image())

    # Second pass
    var idat_data = UnsafePointer[UInt8].alloc(idat_total)
    var write_off = 0
    cursor = ptr + 8

    while (cursor + 12) <= end_ptr:
        var len_u32_2 = read_be_u32(cursor)
        var data_len_2 = Int(len_u32_2)

        var u0 = (cursor + 4)[0]
        var u1 = (cursor + 4)[1]
        var u2 = (cursor + 4)[2]
        var u3 = (cursor + 4)[3]
        var typ2 = _fourcc(u0, u1, u2, u3)

        var data_ptr2 = cursor + 8

        if typ2 == "IDAT":
            var i = 0
            while i < data_len_2 and (write_off + i) < idat_total:
                idat_data[write_off + i] = data_ptr2[i]
                i = i + 1
            write_off = write_off + data_len_2
        elif typ2 == "IEND":
            break

        var advance2 = 12 + data_len_2
        cursor = cursor + advance2

    # Inflate
    var channels: Int = 1
    if color_type == UInt8(2):
        channels = 3

    var row_bytes = width * channels
    var scan_bytes = height * (row_bytes + 1)
    var out_buf = UnsafePointer[UInt8].alloc(scan_bytes)

    var produced = inflate(idat_data, idat_total, out_buf, scan_bytes)
    if produced <= 0:
        return (False, _fallback_image())

    # Unfilter
    unfilter_scanlines(out_buf, width, height, channels)

    # Pack HWC tensor (skip 1-byte filter per row)
    var img = full((height, width, channels), UInt8(0))
    var t = img.tensor()
    var dst = t.data()

    var y = 0
    var idx = 0
    while y < height:
        var row_src = out_buf + y * (row_bytes + 1) + 1
        var x = 0
        while x < row_bytes:
            dst[idx] = row_src[x]
            idx = idx + 1
            x = x + 1
        y = y + 1

    var m = ImageMeta().with_colorspace(ColorSpace.SRGB())
    return (True, Image(t, m))




fn bytes_to_string4(a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> String:
    var buf = UnsafePointer[UInt8].alloc(4)
    buf[0] = a
    buf[1] = b
    buf[2] = c
    buf[3] = d
    var s = String(buf, 4)
    UnsafePointer[UInt8].free(buf)
    return s

