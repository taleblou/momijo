# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/color_ycbcr.mojo
# Description: RGB <-> YCbCr helpers (8-bit, 4:4:4)

# Clamp Int to [0,255] and return UInt8
fn _clamp_u8(x: Int) -> UInt8:
    if x < 0:
        return UInt8(0)
    if x > 255:
        return UInt8(255)
    return UInt8(x)

# Convert one RGB pixel (8-bit) to YCbCr (BT.601, full range, integer approx)
fn _rgb_to_ycbcr_pixel(r: UInt8, g: UInt8, b: UInt8) -> (UInt8, UInt8, UInt8):
    # Using common integer-friendly coefficients
    # Y  ≈  0.299 R + 0.587 G + 0.114 B
    # Cb ≈ -0.168736 R - 0.331264 G + 0.5 B + 128
    # Cr ≈  0.5 R - 0.418688 G - 0.081312 B + 128
    var ri = Int(r)
    var gi = Int(g)
    var bi = Int(b)

    var y  = (77 * ri + 150 * gi + 29 * bi + 128) >> 8         # 0..255
    var cb = ((-43 * ri - 85 * gi + 128 * bi) + 128*256 + 128) >> 8
    var cr = ((128 * ri - 107 * gi - 21 * bi) + 128*256 + 128) >> 8

    return (_clamp_u8(y), _clamp_u8(cb), _clamp_u8(cr))

# Convert interleaved RGB buffer (HWC, 3 channels) to planar Y, Cb, Cr (all 8-bit).
# rgb points to width*height*3 bytes; y_out/cb_out/cr_out point to width*height each.
# ptr -> packed RGB (HWC), out: Y/Cb/Cr planes
fn rgb_to_ycbcr(src: UnsafePointer[UInt8], w: Int, h: Int,
                y: UnsafePointer[UInt8],
                cb: UnsafePointer[UInt8],
                cr: UnsafePointer[UInt8]):
    var npix = w * h
    var si = 0
    var i = 0
    while i < npix:
        var r = Int(src[si + 0])
        var g = Int(src[si + 1])
        var b = Int(src[si + 2])
        # BT.601 full range (integer approx; clamp to [0,255])
        var Y  =  (  77*r + 150*g +  29*b + 128) >> 8         # ≈ 0.299,0.587,0.114
        var Cb = (((-43*r -  85*g + 128*b) + 128) >> 8) + 128 # ≈ -0.1687,-0.3313,0.5
        var Cr = (((128*r - 107*g -  21*b) + 128) >> 8) + 128 # ≈ 0.5,-0.4187,-0.0813
        if Y  < 0: Y = 0; 
        if Y  > 255: Y  = 255
        if Cb < 0: Cb = 0; 
        if Cb > 255: Cb = 255
        if Cr < 0: Cr = 0; 
        if Cr > 255: Cr = 255
        y[i]  = UInt8(Y)
        cb[i] = UInt8(Cb)
        cr[i] = UInt8(Cr)
        si += 3
        i  += 1


# Inverse: one YCbCr pixel to RGB (BT.601, full range, fixed-point)
# Scales are 65536-based to avoid floats.
fn _ycbcr_to_rgb_pixel(y: UInt8, cb: UInt8, cr: UInt8) -> (UInt8, UInt8, UInt8):
    var yf  = Int(y)
    var cbf = Int(cb) - 128
    var crf = Int(cr) - 128

    # Fixed-point coefficients (approx):
    # R = Y + 1.402   * Cr
    # G = Y - 0.34414 * Cb - 0.71414 * Cr
    # B = Y + 1.772   * Cb
    var r = yf + (91881  * crf) / 65536
    var g = yf - (22554  * cbf + 46802 * crf) / 65536
    var b = yf + (116130 * cbf) / 65536

    return (_clamp_u8(r), _clamp_u8(g), _clamp_u8(b))

# Merge planar Y, Cb, Cr to interleaved RGB (HWC).
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
        var r, g, b = _ycbcr_to_rgb_pixel(y_ptr[i], cb_ptr[i], cr_ptr[i])
        var base = i * 3
        out_ptr[base + 0] = r
        out_ptr[base + 1] = g
        out_ptr[base + 2] = b
        i = i + 1
