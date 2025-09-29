# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/jpeg_color_ycbcr.mojo
# Description: Convert planar Y/Cb/Cr (8-bit) to interleaved RGB (8-bit), no subsampling.

fn _clamp_u8(x: Int) -> UInt8:
    if x < 0:
        return UInt8(0)
    if x > 255:
        return UInt8(255)
    return UInt8(x)

# Fixed-point BT.601 (scaled by 65536)
# R = Y + 1.402   * (Cr - 128)   ->  1.402  * 65536 ≈  91881
# G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128) -> 22554, 46802
# B = Y + 1.772   * (Cb - 128)   ->  1.772  * 65536 ≈ 116130
fn _ycbcr_to_rgb(y: UInt8, cb: UInt8, cr: UInt8) -> (UInt8, UInt8, UInt8):
    var yf  = Int(y)
    var cbf = Int(cb) - 128
    var crf = Int(cr) - 128

    var r = yf + (91881  * crf) // 65536
    var g = yf - (22554  * cbf + 46802 * crf) // 65536
    var b = yf + (116130 * cbf) // 65536

    return (_clamp_u8(r), _clamp_u8(g), _clamp_u8(b))

# Merge Y, Cb, Cr planar buffers into interleaved RGB buffer (HWC order).
fn merge_ycbcr_to_rgb(
    y_ptr: UnsafePointer[UInt8],
    cb_ptr: UnsafePointer[UInt8],
    cr_ptr: UnsafePointer[UInt8],
    out_ptr: UnsafePointer[UInt8],
    width: Int,
    height: Int
):
    var npix = width * height
    var i = 0
    while i < npix:
        var (r, g, b) = _ycbcr_to_rgb(y_ptr[i], cb_ptr[i], cr_ptr[i])
        var base = i * 3
        out_ptr[base + 0] = r
        out_ptr[base + 1] = g
        out_ptr[base + 2] = b
        i = i + 1
