# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/jpeg.mojo
# Description: JPEG read/write using pure Mojo encoder/decoder (file_io backend)

from momijo.vision.tensor import Tensor
from momijo.vision.image import Image, ImageMeta, ColorSpace
from momijo.vision.transforms.array import full
from momijo.vision.io.decode_jpeg import decode_jpeg
from momijo.vision.io.encode_jpeg import encode_jpeg
from momijo.vision.io.file_io import _read_file_bytes, _write_file_raw

# ---------------------------- Private helpers ----------------------------

fn _clamp_quality(q: Int) -> Int:
    var v = q
    if v < 1: v = 1
    if v > 100: v = 100
    return v

fn _to_lower(s: String) -> String:
    var out = String()
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch >= 'A' and ch <= 'Z':
            ch = Char(Int(ch) + (Int('a') - Int('A')))
        out = out + String(ch)
        i += 1
    return out

fn _ends_with_lower(s: String, suffix: String) -> Bool:
    var a = _to_lower(s)
    var b = _to_lower(suffix)
    var n = a.__len__()
    var m = b.__len__()
    if m > n: return False
    var i = 0
    while i < m:
        if a[n - m + i] != b[i]:
            return False
        i += 1
    return True

fn _make_dummy_u8_hwc(h: Int, w: Int, c: Int) -> Tensor:
    return full((h, w, c), UInt8(127)).tensor()

# ---------------------------- Public API: Capabilities --------------------

fn has_jpeg_codec() -> Bool:
    return True

# ---------------------------- Public API: Read ----------------------------

fn read_jpeg(path: String) -> (Bool, Tensor):
    # _read_file_bytes returns (ok, bytes)
    var (ok_read, bytes) = _read_file_bytes(path)
    if not ok_read:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    var n = len(bytes)
    if n == 0:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    var ptr = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        ptr[i] = bytes[i]
        i += 1

    var (ok_dec, img) = decode_jpeg(ptr, n)
    # free buffer after use
    UnsafePointer[UInt8].free(ptr)

    if not ok_dec:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    # if decode_jpeg returns Image → use .tensor(); if it already returns Tensor, just return img
    return (True, img.tensor())


fn read_jpeg_with_fallback(path: String,
                           fallback_h: Int = 64,
                           fallback_w: Int = 64,
                           fallback_c: Int = 3) -> (Bool, Tensor):
    var (ok, t) = read_jpeg(path)
    if ok:
        return (True, t)

    var h = if fallback_h > 0: fallback_h else: 64
    var w = if fallback_w > 0: fallback_w else: 64
    var c = if fallback_c == 1 or fallback_c == 3: fallback_c else: 3

    var dummy = _make_dummy_u8_hwc(h, w, c)
    return (False, dummy)

# ---------------------------- Public API: Write ---------------------------

fn write_jpeg(path: String, t: Tensor, quality: Int = 90) -> Bool:
    var q = _clamp_quality(quality)
    var width = t.width()
    var height = t.height()
    var channels = t.channels()

    # Only grayscale for now
    if channels != 1:
        return False

    var in_ptr = t.data()                  # UnsafePointer[UInt8] (ورودی انکدر)
    var max_size = 1024 * 1024
    var out_buf = UnsafePointer[UInt8].alloc(max_size)

    var (ok_enc, used) = encode_jpeg(in_ptr, width, height, out_buf, max_size)
    if not ok_enc:
        UnsafePointer[UInt8].free(out_buf)
        return False

    # Persist encoded bytes
    var ok = _write_file_raw(path, out_buf, used)

    # Free temp buffer before returning
    UnsafePointer[UInt8].free(out_buf)
    return ok


# ---------------------------- Image wrappers ----------------------------

fn read_jpeg_image(path: String) -> Image:
    var (ok, t) = read_jpeg(path)
    assert(ok, "read_jpeg failed: " + path)

    var c = t.shape(2)
    var meta = ImageMeta()
    if c == 1:
        meta = meta.with_colorspace(ColorSpace.GRAY())
    else:
        meta = meta.with_colorspace(ColorSpace.SRGB())

    return Image(t, meta)

fn write_jpeg_image(path: String, img: Image, quality: Int = 90) -> Bool:
    var t = img.tensor()
    return write_jpeg(path, t, quality)
