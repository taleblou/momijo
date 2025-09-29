# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/png.mojo
# Description: PNG read/write facade (pure Mojo). Decode via decode_png, encode via encode_png_tensor.

from momijo.vision.tensor import Tensor
from momijo.vision.transforms.array import full
from momijo.vision.io.decode_png import decode_png
from momijo.vision.io.encode_png import encode_png_tensor
from momijo.vision.io.file_io import _read_file_bytes, _write_file_bytes  # <-- use file_io.mojo

# -------------------------------------------------------------------------
# Capability flag for registry
# -------------------------------------------------------------------------
fn has_png_codec() -> Bool:
    return True

# -------------------------------------------------------------------------
# Fallback dummy tensor constructor
# -------------------------------------------------------------------------
fn _make_dummy_u8_hwc(h: Int, w: Int, c: Int) -> Tensor:
    return full((h, w, c), UInt8(127)).tensor()

# -------------------------------------------------------------------------
# PNG decode: returns (ok=True, tensor) if PNG is valid
# -------------------------------------------------------------------------
fn read_png(path: String) -> (Bool, Tensor):
    # _read_file_bytes returns (ok, bytes)
    var (ok_read, bytes) = _read_file_bytes(path)
    if not ok_read:
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    var n = len(bytes)
    if n == 0:
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    var buf = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        buf[i] = bytes[i]
        i += 1

    # decode_png expects a mutable pointer parameter
    var (ok_dec, img) = decode_png(buf, n)
    # free temp buffer before returning
    UnsafePointer[UInt8].free(buf)

    if not ok_dec:
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    # if decode_png returns Image, use .tensor(); if it returns Tensor already, return img
    return (True, img.tensor())



# -------------------------------------------------------------------------
# Convenience wrapper with synthetic fallback
# -------------------------------------------------------------------------
fn read_png_with_fallback(path: String,
                          fallback_h: Int = 64,
                          fallback_w: Int = 64,
                          fallback_c: Int = 3) -> (Bool, Tensor):
    var (ok, t) = read_png(path)
    if ok:
        return (True, t)

    var h = if fallback_h > 0: fallback_h else: 64
    var w = if fallback_w > 0: fallback_w else: 64
    var c = if fallback_c == 1 or fallback_c == 3: fallback_c else: 3
    var dummy = _make_dummy_u8_hwc(h, w, c)
    return (False, dummy)

# -------------------------------------------------------------------------
# PNG encode (pure Mojo, GRAY/RGB 8-bit, filter 0, stored DEFLATE in zlib)
# -------------------------------------------------------------------------
fn write_png(path: String, t: Tensor, compress_level: Int = 0) -> Bool:
    var (ok_enc, bytes) = encode_png_tensor(t)
    if not ok_enc:
        return False
    return _write_file_bytes(path, bytes)
