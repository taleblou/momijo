# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/registry.mojo
# Description: Unified image reader registry. Routes by file extension to PNG/JPEG/PPM backends.

from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType
from momijo.vision.io.jpeg import read_jpeg, read_jpeg_with_fallback
from momijo.vision.io.png  import read_png,  read_png_with_fallback

# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

# Internal: make a small dummy packed HWC u8 tensor (simple gradient).
# Produces a valid Tensor layout using packed_hwc_strides and UInt8 pixels.
fn _make_dummy_u8_hwc(h: Int = 32, w: Int = 32, c: Int = 3) -> Tensor:
    var hh = h
    var ww = w
    var cc = c
    if hh <= 0: hh = 32
    if ww <= 0: ww = 32
    if cc != 1 and cc != 3: cc = 3

    var (s0, s1, s2) = packed_hwc_strides(hh, ww, cc)
    var n = hh * ww * cc
    var buf = UnsafePointer[UInt8].alloc(n)

    var y = 0
    while y < hh:
        var x = 0
        while x < ww:
            var ch = 0
            while ch < cc:
                var v = (y * 7 + x * 11 + ch * 29) % 256
                buf[y * s0 + x * s1 + ch * s2] = UInt8(v)
                ch += 1
            x += 1
        y += 1

    # Prefer factory that takes ownership of the raw buffer if available.
    # Fallback: Tensor.from_u8_hwc (project API used elsewhere).
    return Tensor.from_u8_hwc(buf, hh, ww, cc)

fn _to_lower(s: String) -> String:
    var out = String("")
    var i = 0
    var n = s.__len__()
    while i < n:
        var ch = s[i]
        # ASCII tolower
        if ch >= 'A' and ch <= 'Z':
            out = out + String(Char(ord(ch) + 32))
        else:
            out = out + String(ch)
        i += 1
    return out.copy()

@always_inline
fn _ends_with_lower(s: String, suffix: String) -> Bool:
    var a = _to_lower(s)
    var b = _to_lower(suffix)
    var n = a.__len__()
    var m = b.__len__()
    if m > n:
        return False
    var i = 0
    while i < m:
        if a[n - m + i] != b[i]:
            return False
        i += 1
    return True

@always_inline
fn _ends_with_any_lower(s: String, suf1: String, suf2: String, suf3: String) -> Bool:
    if _ends_with_lower(s, suf1):
        return True
    if _ends_with_lower(s, suf2):
        return True
    if _ends_with_lower(s, suf3):
        return True
    return False

# -------------------------------------------------------------------
# Optional: PPM file-level reader placeholder
# Replace with a real reader that maps `path -> bytes -> decode_ppm_u8_hwc`.
# -------------------------------------------------------------------
fn read_ppm(path: String) -> (Bool, Tensor):
    # No filesystem-backed PPM reader here; return a dummy to signal "unsupported".
    var dummy = _make_dummy_u8_hwc(32, 32, 3)
    return (False, dummy)

# -------------------------------------------------------------------
# Public: Unified read entry points
# -------------------------------------------------------------------

# Route by file extension (case-insensitive).
# Returns (ok, Tensor). If format is unknown, tries JPEG then PNG as a best effort.
fn read_image_auto(path: String) -> (Bool, Tensor):
    # Fast path by extension
    if _ends_with_any_lower(path, ".jpg", ".jpeg", ".jpe"):
        var rj = read_jpeg(path)
        if rj[0]:
            return (True, rj[1].copy())
        # If extension said JPEG but decode failed, try PNG second
        var rp = read_png(path)
        if rp[0]:
            return (True, rp[1].copy())
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    if _ends_with_lower(path, ".png"):
        var rp2 = read_png(path)
        if rp2[0]:
            return (True, rp2[1].copy())
        # Try JPEG as fallback
        var rj2 = read_jpeg(path)
        if rj2[0]:
            return (True, rj2[1].copy())
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    if _ends_with_lower(path, ".ppm"):
        var rppm = read_ppm(path)
        if rppm[0]:
            return (True, rppm[1].copy())
        # Try common formats
        var rj3 = read_jpeg(path)
        if rj3[0]:
            return (True, rj3[1].copy())
        var rp3 = read_png(path)
        if rp3[0]:
            return (True, rp3[1].copy())
        return (False, _make_dummy_u8_hwc(32, 32, 3))

    # Unknown extension: try JPEG then PNG
    var rj4 = read_jpeg(path)
    if rj4[0]:
        return (True, rj4[1].copy())
    var rp4 = read_png(path)
    if rp4[0]:
        return (True, rp4[1].copy())

    return (False, _make_dummy_u8_hwc(32, 32, 3))

# Same as above, but caller can provide fallback H/W/C when decoding fails.
fn read_image_auto_with_fallback(path: String,
                                 fallback_h: Int = 64,
                                 fallback_w: Int = 64,
                                 fallback_c: Int = 3) -> (Bool, Tensor):
    if _ends_with_any_lower(path, ".jpg", ".jpeg", ".jpe"):
        var rj = read_jpeg_with_fallback(path, fallback_h, fallback_w, fallback_c)
        if rj[0]:
            return (True, rj[1].copy())
        # If jpeg-with-fallback still failed (shouldn't), try png-with-fallback
        var rp = read_png_with_fallback(path, fallback_h, fallback_w, fallback_c)
        return (rp[0], rp[1].copy())

    if _ends_with_lower(path, ".png"):
        var rp2 = read_png_with_fallback(path, fallback_h, fallback_w, fallback_c)
        if rp2[0]:
            return (True, rp2[1].copy())
        var rj2 = read_jpeg_with_fallback(path, fallback_h, fallback_w, fallback_c)
        return (rj2[0], rj2[1].copy())

    if _ends_with_lower(path, ".ppm"):
        var rppm = read_ppm(path)
        if rppm[0]:
            return (True, rppm[1].copy())
        # Construct explicit dummy with requested shape
        var h = fallback_h
        var w = fallback_w
        var c = fallback_c
        if h <= 0: h = 64
        if w <= 0: w = 64
        if not (c == 1 or c == 3): c = 3
        return (False, _make_dummy_u8_hwc(h, w, c))

    # Unknown extension: try both with fallbacks
    var rj4 = read_jpeg_with_fallback(path, fallback_h, fallback_w, fallback_c)
    if rj4[0]:
        return (True, rj4[1].copy())
    var rp4 = read_png_with_fallback(path, fallback_h, fallback_w, fallback_c)
    return (rp4[0], rp4[1].copy())
